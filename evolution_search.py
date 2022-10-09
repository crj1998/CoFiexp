import time, os, random
from functools import partial
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


    

IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)
transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])
dataset = datasets.ImageFolder(os.path.join(args.datafolder, "train20"), transform=transform)
dataloader_config = {
    "batch_size": args.batch_size//args.world_size,
    "num_workers": args.num_workers,
    "drop_last": True,
    "pin_memory": True
}
dataloader = DataLoader(dataset, **dataloader_config)

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
zs_ = l0module.l0_mask()
zs["intermediate_z"] = zs_["intermediate_z"]
results = l0module.calculate_model_size(zs)
sparsity = results["pruned_sparsity"]
print(sparsity)


@torch.no_grad()
def eval(model, dataloader, device, zs=None):
    latency = total = correct = 0
    if zs is not None:
        for k in zs:
            zs[k] = zs[k].to(device)
    i = 0
    with tqdm(dataloader, total=len(dataloader), desc="Eval", ncols=80, disable=True) as t:
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
            i += 1
            # if i> 16:
            #     break
    return correct/total, latency/len(dataloader.dataset)

from copy import deepcopy
cnt = 0
def eval_func(config):
    zsn = deepcopy(zs)
    for layer in range(12):
        intermediate_z = zsn["intermediate_z"][layer].squeeze()
        shape = (1, 1, 3072)
        disabled = 3072 - config[f"layer_{layer}"]
        intermediate_z[torch.sort(intermediate_z).indices[:disabled]] = 0.0
        zsn["intermediate_z"][layer] = intermediate_z.reshape(shape)
    acc, _ = eval(model, dataloader, device, zsn)
    # acc = random.random()
    dims = [config[f"layer_{layer}"] for layer in range(12)]
    global cnt
    cnt += 1
    print(cnt, sum(dims), dims, round(acc, 4))
    return acc


from searcher import EvolutionSearcher

search_space = {
    # f"layer_{layer}": [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16] for layer in range(12)
    f"layer_{layer}": [-256, -128, -64, -32, -8, -2, -1] for layer in range(12)
}
searcher = EvolutionSearcher(eval_func, search_space, evolution_iters=8, optimize='max')
searcher.run([{f"layer_{i}": pruned_results["remain_intermediate"][i] for i in range(12)}])

# for k,v in zs.items():
#     print(k, v.shape)
# results = l0module.calculate_model_size(zs)
# sparsity = results["pruned_sparsity"]
# print(sparsity)