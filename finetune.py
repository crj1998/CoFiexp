
import os, random, math
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as T


from vit import vit_base_patch16_224
from optimizer import SAM


def linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with tqdm(dataloader, total=len(dataloader), ncols=80, desc="Eval") as t:
        for inputs, targets in t:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)
            t.set_postfix_str(f"Accuracy={correct/total:.2%}")
    return correct/total

def main(args):

    device = torch.device("cuda")
    IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        # T.RandAugment(),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    test_transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])


    train_set = eval(f"CIFAR{args.num_classes}")(args.datapath, download=False, train=True, transform=train_transform)
    valid_set = eval(f"CIFAR{args.num_classes}")(args.datapath, download=False, train=False, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    num_steps_per_epoch = max(len(train_loader), 1)

    model = vit_base_patch16_224()
    model.load_state_dict(torch.load(args.weights))
    model.reset_classifier(args.num_classes)
    model.to(device)

    no_decay = ["bias", "layernorm"]
    freeze_keywords = []
    model_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
            "weight_decay": args.weight_decay,
        }, {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
            "weight_decay": 0.0,
        }
    ]

    # optimizer = optim.AdamW(model_params, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    optimizer = optim.SGD(model_params, lr=args.learning_rate, momentum=0.9)

    # define an optimizer for the "sharpness-aware" update
    # optimizer = SAM(model_params, base_optimizer=optim.AdamW, rho=0.1, adaptive=True, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    # optimizer = SAM(model_params, base_optimizer=optim.SGD, rho=0.1, adaptive=True, lr=args.learning_rate, momentum=0.9)

    lr_scheduler = cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.lr_warmup*num_steps_per_epoch, num_training_steps=args.epochs*num_steps_per_epoch
    )

    # training
    global_step = 0
    evaluate(model, valid_loader, device)
    for epoch in range(1, int(args.epochs)+1):
        model.train()
        with tqdm(train_loader, total=len(train_loader), ncols=80, desc=f"Train({epoch})") as t:
            for step, (inputs, targets) in enumerate(t):
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs)

                loss = F.cross_entropy(logits, targets)
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # first forward-backward pass
                # loss = F.cross_entropy(model(inputs), targets)  # use this loss for any training statistics
                # loss.backward()
                # optimizer.first_step()
                # optimizer.zero_grad()
                
                # # second forward-backward pass
                # F.cross_entropy(model(inputs), targets).backward()  # make sure to do a full forward pass
                # optimizer.second_step()
                # optimizer.zero_grad()

                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()[0]
                
                global_step += 1
                t.set_postfix({"loss": round(loss.detach().item(), 3), "lr": round(lr, 3)})
    
        evaluate(model, valid_loader, device)

        # torch.save(model.state_dict(), "cache/cifar10_finetuned.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Vision Transformer fine-tune")
    parser.add_argument("--datapath", type=str, default="path/to/dataset")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr_warmup", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.00002)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)



    args = parser.parse_args()

    # seed all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True

    args.gpus = min(args.gpus, torch.cuda.device_count())

    main(args)

"""
CUDA_VISIBLE_DEVICES=7 python finetune.py --weights cache/pretrained.pth --datapath ../../data --num_classes 100

setting       accuracy
AdamW          92.34
+sam(0.05)     92.50
+sam(0.10)     92.68
+sam(0.15)     92.66
+asam(0.10)    92.45

SGD
"""