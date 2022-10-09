from copy import deepcopy

import os, math, time, random
import argparse, logging
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as T
import torchvision.datasets as datasets

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from contextlib import contextmanager

from utils import get_logger, colorstr, setup_seed, unwrap_model, AverageMeter


@contextmanager
def torch_distributed_zero_first(rank):
    """ Decorator to make all processes in distributed training wait for each local_master to do something. """
    if rank not in [-1, 0]:
        dist.barrier(device_ids=[rank])
    yield
    if rank == 0:
        dist.barrier(device_ids=[0])

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

@torch.no_grad()
def valid(model, dataloader, l0_module, args):
    Acc = AverageMeter()
    model.eval()
    l0_module.eval()
    zs = l0_module(True)
    for inputs, targets in dataloader:
        batch_size = inputs.size(0)
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        logits = model(inputs, **zs)

        acc = (logits.argmax(dim=-1)==targets).sum()/batch_size
        dist.barrier()
        reduced_acc = reduce_mean(acc, args.world_size)
        Acc.update(reduced_acc.item(), batch_size)

    sparsity = unwrap_model(l0_module).calculate_model_size(zs)['pruned_sparsity']
    if args.local_rank in [-1, 0]:
        args.logger.info(f"Valid: Acc={Acc.item():.2%}, Sparsity={sparsity:.2%}")
    return acc

def train(epoch, iters, model, dataloader, criterion, optimizer, scheduler, l0_module, l0_optimizer, args, teacher=None):

    Loss = AverageMeter()
    Acc = AverageMeter()

    dataiter = iter(dataloader)
    model.train()
    l0_module.train()
    for it in range(iters):
        try:
            inputs, targets = next(dataiter)
        except:
            dataiter = iter(dataloader)
            inputs, targets = next(dataiter)

        batch_size = inputs.size(0)
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        zs = l0_module(True)
        logits = model(inputs, **zs)
        if teacher is not None:
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            distil_loss = F.kl_div(
                F.log_softmax(logits / args.distill_temp, dim=-1),
                F.softmax(teacher_logits / args.distill_temp, dim=-1),
                reduction="batchmean"
            ) * (args.distill_temp ** 2)
        else:
            distil_loss = F.cross_entropy(logits, targets)

        lagran_loss, expected_sparsity, target_sparsity = unwrap_model(l0_module).lagrangian_regularization(args.global_step)
        loss = distil_loss + lagran_loss


        l0_optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        l0_optimizer.step()
        unwrap_model(l0_module).constrain_parameters()
        optimizer.step()
        scheduler.step()
        
        Loss.update(loss.item(), batch_size)
        Acc.update((logits.argmax(dim=-1)==targets).sum().item()/batch_size, batch_size)

        args.global_step += 1
        if args.local_rank in [-1, 0] and (it % 128 == 0 or it == iters-1):
            lr = scheduler.get_last_lr()[0]
            args.logger.info(f"Epoch {epoch+1:>2d} Iter {it+1:>4d}: loss={Loss.item():6.4f}, acc={Acc.item():6.2%}, LR={lr:.5f}, Sparsity: {round(expected_sparsity, 3)}->{round(target_sparsity, 3)}")

    return Loss.item(), Acc.item()


def main(args):
    # set up logger
    level = logging.DEBUG if "dev" in args.out else logging.INFO

    # level = logging.INFO
    args.local_rank = int(os.environ["LOCAL_RANK"])
    logger = get_logger(
        args = args,
        name = f"CoFi{args.local_rank}",
        level = level if args.local_rank in [-1, 0] else logging.WARN,
        fmt = "%(asctime)s [%(levelname)s] %(message)s" if args.local_rank in [-1, 0] else "%(asctime)s [%(levelname)s @ %(name)s] %(message)s", 
        rank = args.local_rank
    )
    args.logger = logger
    logger.debug(f"Get logger named {colorstr('CoFi')}!")
    logger.debug(f"Distributed available? {dist.is_available()}")

    #setup random seed
    if args.seed and isinstance(args.seed, int):
        setup_seed(args.seed)
        logger.info(f"Setup random seed {colorstr('green', args.seed)}!")
    else:
        logger.info(f"Can not Setup random seed with seed is {colorstr('green', args.seed)}!")

    # init dist params
    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        dist.init_process_group(backend='nccl')
        args.world_size = dist.get_world_size()
        args.n_gpu = torch.cuda.device_count()
        args.local_rank = dist.get_rank()
        # torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        assert dist.is_initialized(), f"Distributed initialization failed!"

    # set device
    args.device = device
    logger.debug(f"Current device: {device}")

    # make dataset
    with torch_distributed_zero_first(args.local_rank):
        IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
        IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)
        train_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        valid_transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])
        train_set = datasets.ImageFolder(os.path.join(args.datafolder, "train"), transform=train_transform)
        valid_set = datasets.ImageFolder(os.path.join(args.datafolder, "val"), transform=valid_transform)

        logger.info(f"Dataset: {colorstr('green', len(train_set))} samples for train, {colorstr('green', len(valid_set))} sampels for valid!")

    # prepare dataloader
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    valid_sampler = SequentialSampler if args.local_rank == -1 else DistributedSampler # SequentialDistributedSampler
    dataloader_config = {
        "batch_size": args.batch_size//args.world_size,
        "num_workers": args.num_workers,
        "drop_last": True,
        "pin_memory": True
    }
    train_loader = DataLoader(train_set, sampler = train_sampler(train_set), **dataloader_config)
    valid_loader = DataLoader(valid_set, sampler = valid_sampler(valid_set), **dataloader_config)

    logger.info(f"Dataloader Initialized. Batch size: {colorstr('green', args.batch_size)}, Num workers: {colorstr('green', args.num_workers)}.")

    with torch_distributed_zero_first(args.local_rank):
        # build model
        # model = models.resnet18(weights=None, num_classes=1000)
        from vit import vit_base_patch16_224
        from l0module import L0Module

        model = vit_base_patch16_224(args.num_classes, raw=False)
        l0_module = L0Module(model.config, lagrangian_warmup=args.sparsity_warmup, target_sparsity=args.sparsity)
        logger.info(f"Model: {colorstr('ViT')}. Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

        # load from pre-trained, before DistributedDataParallel constructor
        # pretrained is str and it exists and is file.
        if isinstance(args.pretrained, str) and os.path.exists(args.pretrained) and os.path.isfile(args.pretrained):
            state_dict = torch.load(args.pretrained, map_location="cpu")

            # rename pre-trained keys
            state_dict = {k[len("module."):] if k.startswith("module.") else k: v.clone() for k, v in state_dict.items()}

            msg = model.load_state_dict(state_dict, strict=False)
            logger.warning(f"Missing keys {msg.missing_keys} in state dict.")
            logger.info(f"Pretrained weights @: {colorstr(str(args.pretrained))} loaded!")
        if args.teacher:
            teacher = deepcopy(model)
        else:
            teacher = None


    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(args.device)
    l0_module.to(args.device)
    if args.teacher:
        teacher.to(args.device)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(args.device)
    # make optimizer, scheduler
    no_decay = ["bias", "layernorm"]
    freeze_keywords = ["embedding"]
    param_group = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
            "weight_decay": args.weight_decay,
        }, {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
            "weight_decay": 0.0,
        }
    ]
    optimizer = optim.AdamW(param_group, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    warm_up = 0
    lr_min = 0.01
    T_max = args.total_step
    lr_lambda = lambda i: i / warm_up if i<warm_up else lr_min + (1-lr_min)*(1.0+math.cos((i-warm_up)/(T_max-warm_up)*math.pi))/2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    l0_params = [
        {
            "params": [p for n, p in l0_module.named_parameters() if "lambda" not in n],
            "weight_decay": 0.0,
            "lr": 0.01
        }, {
            "params": [p for n, p in l0_module.named_parameters() if "lambda" in n],
            "weight_decay": 0.0,
            "lr": - 0.01
        }
    ]

    l0_optimizer = optim.AdamW(l0_params)

    logger.info(f"Optimizer {colorstr('Adam')} and Scheduler {colorstr('Cosine')} selected!")

    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        l0_module = DDP(l0_module, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)



    model.zero_grad()
    l0_module.zero_grad()
    best_acc = 0.0
    args.global_step = 0
    #train loop
    for epoch in range(args.total_step//args.valid_step):
        train_loader.sampler.set_epoch(epoch)
        loss, train_acc = train(epoch, args.valid_step, model, train_loader, criterion, optimizer, scheduler, l0_module, l0_optimizer, args, teacher)

        acc = valid(model, valid_loader, l0_module, args)
        if args.local_rank in [-1, 0]:
            if acc > best_acc:
                best_acc = acc
                torch.save(unwrap_model(model).state_dict(), os.path.join(args.out, "best_model.pth"))
                torch.save(unwrap_model(l0_module).state_dict(), os.path.join(args.out, "best_l0module.pth"))
            torch.save(unwrap_model(model).state_dict(), os.path.join(args.out, "last_model.pth"))
            torch.save(unwrap_model(l0_module).state_dict(), os.path.join(args.out, "last_l0module.pth"))

            # print(acc, best_acc)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--suffix', type=str, required=True, help='exp suffix')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument('--datafolder', type=str, required=True, help='data folder')
    parser.add_argument('--num_classes', type=int, default=1000, help='num classes')
    
    parser.add_argument('--pretrained', default=None, help='directory to pretrained model')
    parser.add_argument('--out', type=str, required=True, help='output folder')
    parser.add_argument('--learning_rate', type=float, default=0.00002)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=96*4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--total_step', type=int, default=2048*15)
    parser.add_argument('--valid_step', type=int, default=2048)

    parser.add_argument("--teacher", action="store_true", default=False)
    parser.add_argument("--distill_temp", type=float, default=2.0)


    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--sparsity_warmup", type=int, default=2048*5)

    args = parser.parse_args()

    args.out = os.path.join(args.out, f"cofi/{args.suffix}")
    os.makedirs(args.out, exist_ok=True)

    main(args)

"""
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 l0ddp.py --suffix dev --datafolder ../../data/imagenet --out ../outputs --pretrained ./cache/pretrained.pth --learning_rate 0.00002 --sparsity_warmup 256 --total_step 1024 --valid_step 512
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 l0ddp.py --suffix dev --datafolder ../../data/imagenet --out ../outputs --pretrained ./cache/pretrained.pth --learning_rate 0.00002 --sparsity_warmup 256 --total_step 1024 --valid_step 128
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 l0ddp.py --suffix spar05_w5t15 --datafolder ../../data/imagenet --out ../outputs --pretrained ./cache/pretrained.pth --learning_rate 0.00002

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 l0ddp.py --suffix spar60_w5t15_teacher --datafolder ../../data/imagenet --out ../outputs --pretrained ./cache/pretrained.pth --learning_rate 0.00002 --sparsity 0.60 --teacher

"""