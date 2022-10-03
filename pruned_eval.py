
import time

from tqdm import tqdm
# import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from vit import vit_base_patch16_224
from l0module import L0Module

device = torch.device("cuda")
# device = torch.device("cpu")




IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

test_transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

dataset = CIFAR10("../../data", download=False, train=False, transform=test_transform)

dataloader = DataLoader(
    dataset,
    batch_size=100,
    drop_last=False,
    num_workers=4,
    pin_memory=False,
)

exp_name = "spar70"


model = vit_base_patch16_224(10)
l0module = L0Module(model.config)

model.load_state_dict(torch.load(f"cache/{exp_name}/model.pth"))
l0module.load_state_dict(torch.load(f"cache/{exp_name}/l0_module.pth"))

model.eval()
l0module.eval()

model = model.to(device)
l0module = l0module.to(device)

zs = l0module.forward()
results = l0module.calculate_model_size(zs)
sparsity = results["pruned_sparsity"]
print(results)




latency = total = correct = 0
with torch.no_grad():
    # print(zs)
    # zs["intermediate_z"] = torch.ones_like(zs["intermediate_z"])
    for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Eval", ncols=80):
        t = time.time()
        logits = model(inputs.to(device))
        correct += (logits.argmax(dim=-1).cpu() == targets).sum().item()
        latency += time.time() - t
        total += targets.size(0)

print(f"Accuracy: {correct/total:.2%}")
print(sparsity)
