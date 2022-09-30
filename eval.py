import random
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, ImageFolder, ImageNet

import torchvision.transforms as T

def set_seed(seed: int):
    assert isinstance(seed, int)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def valid(epoch, model, dataloader, device):
    model.eval()
    top1, top5, total = 0, 0, 0
    with tqdm(dataloader, total=len(dataloader), desc=f"Valid({epoch})", ncols=100) as t:
        for inputs, targets in t:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            topk = torch.topk(logits, dim=-1, k=5, largest=True, sorted=True).indices
            correct = topk.eq(targets.view(-1, 1).expand_as(topk))
            top1 += correct[:, 0].sum().item()
            top5 += correct[:, :5].sum().item()
            total += targets.size(0)
            t.set_postfix({"Top1": f"{top1/total:.2%}", "Top5": f"{top5/total:.2%}"})
    return top1/total, top5/total

def main(args):
    # print(vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)
    set_seed(args.seed)

    transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

    dataset = ImageFolder(args.datapath, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    from vit import vit_base_patch16_224
    model = vit_base_patch16_224(args.num_classes)
    model.load_state_dict(torch.load(args.weights))
    model.eval().to(device)

    valid(-1, model, dataloader, device)



if __name__ == "__main__":
    import argparse

    assert torch.cuda.is_available(), "Only work in plaform with CUDA enable."

    parser = argparse.ArgumentParser("CoFi for ViT")
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["cifar10", "imagenet"])
    parser.add_argument("--datapath", type=str, default="path/to/imagenet")
    parser.add_argument("--weights", type=str, default="path/to/state_dict.pth")
    parser.add_argument("--name", choices=["tiny", "small", "base"], type=str, default="tiny", help="Autoformer size")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    main(args)

"""
CUDA_VISIBLE_DEVICES=0 python eval.py --weights state_dict.pth --datapath ../../data/imagenet/val
"""