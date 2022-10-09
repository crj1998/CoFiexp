
import time

from tqdm import tqdm
# import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms as T




class Zeros(nn.Identity):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x)

class Residual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, input_tensor):
        return input_tensor


@torch.no_grad()
def prune_vit(model, state_dict, zs):
    # model = deepcopy(model)
    # state_dict = model.state_dict()
    config = model.config

    hidden_z = zs['hidden_z'].detach()
    heads_z = zs['heads_z'].detach().squeeze()
    intermediate_z = zs['intermediate_z'].detach().squeeze()
    mha_z = zs['mha_z'].detach().squeeze()
    ffn_z = zs['ffn_z'].detach().squeeze() 

    hidden_size = (hidden_z>0.0).sum().item()
    num_heads = [(i>0.0).sum().item() for i in heads_z]
    intermediates = [(i>0.0).sum().item() for i in intermediate_z]
    mha_layer = [i.item()>0.0 for i in mha_z]
    ffn_layer = [i.item()>0.0 for i in ffn_z]

    # prepare model and init
    
    model.embedding.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
    key = 'embedding.cls_token'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0] * hidden_z[hidden_z>0.0]

    num_patches = model.embedding.num_patches
    model.embedding.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
    key = 'embedding.position_embedding'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0] * hidden_z[hidden_z>0.0]

    model.embedding.projection = nn.Conv2d(config.num_channels, hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
    key = 'embedding.projection.weight'
    value = state_dict[key]
    state_dict[key] = value[hidden_z > 0.0]*hidden_z[hidden_z > 0.0].reshape(-1, 1, 1, 1)
    key = 'embedding.projection.bias'
    value = state_dict[key]
    state_dict[key] = value[hidden_z > 0.0]*hidden_z[hidden_z > 0.0]

    for layer, module in enumerate(model.encoder):
        num_attention_heads = num_heads[layer]
        intermediate_size = intermediates[layer]
        mha = mha_layer[layer]
        ffn = ffn_layer[layer]

        if mha:
            module.layernorm_before = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
            key = f'encoder.{layer}.layernorm_before.weight'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]
            key = f'encoder.{layer}.layernorm_before.bias'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]

            module.mha.num_attention_heads = num_attention_heads
            attention_head_size = module.mha.attention_head_size

            head_dim = heads_z[layer].repeat_interleave(attention_head_size)
            
            all_head_size = num_attention_heads * attention_head_size
            module.mha.all_head_size = all_head_size
            module.mha.query = nn.Linear(hidden_size, all_head_size, bias=config.qkv_bias)
            module.mha.key = nn.Linear(hidden_size, all_head_size, bias=config.qkv_bias)
            module.mha.value = nn.Linear(hidden_size, all_head_size, bias=config.qkv_bias)

            for k1 in ['query', 'key', 'value']:
                key = f'encoder.{layer}.mha.{k1}.weight'
                value = state_dict[key]
                state_dict[key] = value[head_dim > 0.0][:, hidden_z > 0.0]
                
                key = f'encoder.{layer}.mha.{k1}.bias'
                value = state_dict[key]
                state_dict[key] = value[head_dim > 0.0]

            module.mha.proj = nn.Linear(all_head_size, hidden_size)
            key = f'encoder.{layer}.mha.proj.weight'
            value = state_dict[key]
            state_dict[key] = mha * value[hidden_z > 0.0][: , head_dim > 0.0] * head_dim[head_dim > 0.0] * hidden_z[hidden_z > 0.0].reshape(-1, 1)
            key = f'encoder.{layer}.mha.proj.bias'
            value = state_dict[key]
            state_dict[key] = mha * value[hidden_z > 0.0] * hidden_z[hidden_z > 0.0]

        else:
            module.layernorm_before = nn.Identity()
            module.mha = Zeros()
            del state_dict[f'encoder.{layer}.layernorm_before.weight']
            del state_dict[f'encoder.{layer}.layernorm_before.bias']
            for k1 in ['query', 'key', 'value', 'proj']:
                for k2 in ['weight', 'bias']:
                    del state_dict[f'encoder.{layer}.mha.{k1}.{k2}']
            
        if ffn:
            inter_dim = intermediate_z[layer]

            module.layernorm_after = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
            key = f'encoder.{layer}.layernorm_after.weight'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]
            key = f'encoder.{layer}.layernorm_after.bias'
            value = state_dict[key]
            state_dict[key] = value[..., hidden_z > 0.0]

            module.ffn.dense1 = nn.Linear(hidden_size, intermediate_size)
            key = f'encoder.{layer}.ffn.dense1.weight'
            value = state_dict[key]
            state_dict[key] = value[inter_dim>0.0][: , hidden_z > 0.0]
            key = f'encoder.{layer}.ffn.dense1.bias'
            value = state_dict[key]
            state_dict[key] = value[inter_dim>0.0]
    
            module.ffn.dense2 = nn.Linear(intermediate_size, hidden_size)
            key = f'encoder.{layer}.ffn.dense2.weight'
            value = state_dict[key]
            state_dict[key] = ffn * inter_dim[inter_dim>0.0] * value[hidden_z > 0.0][:, inter_dim > 0.0] * hidden_z[hidden_z > 0.0].reshape(-1, 1)
            key = f'encoder.{layer}.ffn.dense2.bias'
            value = state_dict[key]
            state_dict[key] = ffn * value[hidden_z > 0.0] * hidden_z[hidden_z > 0.0]

        else:
            module.layernorm_after = nn.Identity()
            module.ffn = Zeros()
            del state_dict[f'encoder.{layer}.layernorm_after.weight']
            del state_dict[f'encoder.{layer}.layernorm_after.bias']
            for k1 in ['dense1', 'dense2']:
                for k2 in ['weight', 'bias']:
                    del state_dict[f'encoder.{layer}.ffn.{k1}.{k2}']

    model.layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
    key = 'layernorm.weight'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0]
    key = 'layernorm.bias'
    value = state_dict[key]
    state_dict[key] = value[..., hidden_z > 0.0]

    model.classifier = nn.Linear(hidden_size, config.num_labels)
    key = "classifier.weight"
    value = state_dict[key]
    state_dict[key] = value[:, hidden_z > 0.0] * hidden_z[hidden_z > 0.0]
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def eval(model, dataloader, device, zs=None):
    latency = total = correct = 0
    for inputs, targets in tqdm(dataloader, total=len(dataloader), desc="Eval", ncols=80):
        inputs = inputs.to(device)
        t = time.time()
        if zs is not None:
            logits = model(inputs, **zs)
        else:
            logits = model(inputs)
        correct += (logits.argmax(dim=-1).cpu() == targets).sum().item()
        latency += time.time() - t
        total += targets.size(0)
    return correct/total, latency/len(dataloader.dataset)

if __name__ == '__main__':

    from vit import vit_base_patch16_224
    from l0module import L0Module
    # seed = 1
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    device = torch.device("cuda:7")
    # device = torch.device("cpu")

    # exp_name = "spar70"

    model = vit_base_patch16_224(1000)
    l0module = L0Module(model.config)

    # model.load_state_dict(torch.load(f"cache/{exp_name}/model.pth"))
    # l0module.load_state_dict(torch.load(f"cache/{exp_name}/l0_module.pth"))
    exp_name = "cofi/spar65_w5t15"
    model.load_state_dict(torch.load(f"../outputs/{exp_name}/last_model.pth"))
    l0module.load_state_dict(torch.load(f"../outputs/{exp_name}/last_l0module.pth"))

    model = model.to(device)
    l0module = l0module.to(device)

    model.eval()
    l0module.eval()

    # zs = l0module.forward()
    zs = l0module.l0_mask()
    # zs = None
    results = l0module.calculate_model_size(zs)
    sparsity = results["pruned_sparsity"]


    IMAGENET_DEFAULT_MEAN = (0.5, 0.5, 0.5)
    IMAGENET_DEFAULT_STD = (0.5, 0.5, 0.5)

    # test_transform = T.Compose([
    #     T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    #     T.ToTensor(),
    #     T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # ])

    # dataset = CIFAR10("../../data", download=False, train=False, transform=test_transform)
    # dataloader = DataLoader(dataset, batch_size=100, num_workers=2)

    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    dataset = ImageFolder("../../data/imagenet/val", transform=transform)
    dataloader_config = {
        "batch_size": 128,
        "num_workers": 4,
        "drop_last": True,
        "pin_memory": True
    }
    dataloader = DataLoader(dataset, **dataloader_config)

    accuracy, latency = eval(model, dataloader, device, zs)
    print(f"Sparsity: {sparsity:.2%}. Accuracy: {accuracy:.2%}. Latency: {latency*10e3:.1f} ms")

    # pruned_model = vit_base_patch16_224(1000, True)
    # pruned_model = prune_vit(pruned_model, model.state_dict(), zs)
    # pruned_model.eval()
    # pruned_model = pruned_model.to(device)
    # accuracy, latency = eval(pruned_model, dataloader, device)
    # print(f"Sparsity: {sparsity:.2%}. Accuracy: {accuracy:.2%}. Latency: {latency*10e3:.1f} ms")

    # from fvcore import nn as fvnn
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # params_base = fvnn.parameter_count(model)['']
    # params_pruned = fvnn.parameter_count(pruned_model)['']
    # sparsity = 1 - (params_pruned / params_base)
    # print(sparsity, f"{params_pruned/1e6:.1f}")
