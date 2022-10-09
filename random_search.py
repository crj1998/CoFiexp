import time, os, random

from tqdm import tqdm
# import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms as T
import torchvision.datasets as datasets

class args:
    datafolder  = "../../data/imagenet"
    batch_size = 128
    num_workers = 4
    world_size = 1

    exp_name = "cofi/spar60_w5t15"
    device = "cuda:0"

@torch.no_grad()
def eval(model, dataloader, device, zs=None):
    latency = total = correct = 0
    if zs is not None:
        for k in zs:
            zs[k] = zs[k].to(device)
    with tqdm(dataloader, total=len(dataloader), desc="Eval", ncols=80) as t:
        for inputs, targets in t:
            inputs = inputs.to(device)
            ts = time.time()
            if zs is not None:
                logits = model(inputs, **zs)
            else:
                logits = model(inputs)
            correct += (logits.argmax(dim=-1).cpu() == targets).sum().item()
            latency += time.time() - ts
            total += targets.size(0)
            t.set_postfix_str(f"Accuracy={correct/total:4.2%}")
    return correct/total, latency/len(dataloader.dataset)




from vit import vit_base_patch16_224
from l0module import L0Module
# seed = 1
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

device = torch.device(args.device)
sparsity = 0.0

model = vit_base_patch16_224(1000)
l0module = L0Module(model.config)

model.load_state_dict(torch.load(f"../outputs/{args.exp_name}/last_model.pth"))
l0module.load_state_dict(torch.load(f"../outputs/{args.exp_name}/last_l0module.pth"))

model = model.to(device)
l0module = l0module.to(device)

model.eval()
l0module.eval()

zs = l0module()
pruned_results = l0module.calculate_model_size(zs)
print(pruned_results)

zs_ = l0module.l0_mask()
zs["intermediate_z"] = zs_["intermediate_z"]
# # zs = l0module()
# # zs = None
results = l0module.calculate_model_size(zs)
sparsity = results["pruned_sparsity"]
print(sparsity)


# for k,v in zs.items():
#     print(k, v.shape)

IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)
transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
dataset = datasets.ImageFolder(os.path.join(args.datafolder, "val"), transform=transform)
dataloader_config = {
    "batch_size": args.batch_size//args.world_size,
    "num_workers": args.num_workers,
    "drop_last": True,
    "pin_memory": True
}
dataloader = DataLoader(dataset, **dataloader_config)



from copy import deepcopy
def random_pruning(zs):
    zs = deepcopy(zs)
    hidden_z = zs["hidden_z"]
    # disabled = round(hidden_z.size(0)*0.3)
    # disabled = 768 - pruned_results["remain_hidden"]
    # zs["hidden_z"][torch.sort(hidden_z).indices[:disabled]] = 0.0
    for layer in range(12):
        # heads_z = zs["heads_z"][layer].squeeze()
        # shape = (1, 12, 1, 1)
        # heads_z[torch.sort(heads_z).indices[:round(heads_z.size(0)*0.2)]] = 0.0
        # zs["heads_z"][layer] = heads_z.reshape(shape)

        intermediate_z = zs["intermediate_z"][layer].squeeze()
        shape = (1, 1, 3072)
        disabled = round(intermediate_z.size(0)*0.8)
        # disabled = 3072 - pruned_results["remain_intermediate"][layer]
        intermediate_z[torch.sort(intermediate_z).indices[:disabled]] = 0.0
        zs["intermediate_z"][layer] = intermediate_z.reshape(shape)
    return zs


zs = random_pruning(zs)

accuracy, latency = eval(model, dataloader, device, zs)
results = l0module.calculate_model_size(zs)
sparsity = results["pruned_sparsity"]
print(results)
print(f"Sparsity: {sparsity:.2%}. Accuracy: {accuracy:.2%}. Latency: {latency*10e3:.1f} ms")

# for k,v in zs.items():
#     print(k, v.shape)
# results = l0module.calculate_model_size(zs)
# sparsity = results["pruned_sparsity"]
# print(sparsity)