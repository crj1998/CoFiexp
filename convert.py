from collections import OrderedDict

from transformers import ViTForImageClassification


# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True)
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# print(model)
state_dict = model.state_dict()

new_state_dict = OrderedDict()
name_mapping = {
    "vit.embeddings.cls_token": "embedding.cls_token",
    "vit.embeddings.position_embeddings": "embedding.position_embedding",
    "vit.embeddings.patch_embeddings.projection.weight": "embedding.projection.weight",
    "vit.embeddings.patch_embeddings.projection.bias": "embedding.projection.bias",
    "vit.encoder.layer.{layer}.attention.attention.query.weight": "encoder.{layer}.mha.query.weight",
    "vit.encoder.layer.{layer}.attention.attention.query.bias": "encoder.{layer}.mha.query.bias",
    "vit.encoder.layer.{layer}.attention.attention.key.weight": "encoder.{layer}.mha.key.weight",
    "vit.encoder.layer.{layer}.attention.attention.key.bias": "encoder.{layer}.mha.key.bias",
    "vit.encoder.layer.{layer}.attention.attention.value.weight": "encoder.{layer}.mha.value.weight",
    "vit.encoder.layer.{layer}.attention.attention.value.bias": "encoder.{layer}.mha.value.bias",
    "vit.encoder.layer.{layer}.attention.output.dense.weight": "encoder.{layer}.mha.proj.weight",
    "vit.encoder.layer.{layer}.attention.output.dense.bias": "encoder.{layer}.mha.proj.bias",
    "vit.encoder.layer.{layer}.intermediate.dense.weight": "encoder.{layer}.ffn.dense1.weight",
    "vit.encoder.layer.{layer}.intermediate.dense.bias": "encoder.{layer}.ffn.dense1.bias",
    "vit.encoder.layer.{layer}.output.dense.weight": "encoder.{layer}.ffn.dense2.weight",
    "vit.encoder.layer.{layer}.output.dense.bias": "encoder.{layer}.ffn.dense2.bias",
    "vit.encoder.layer.{layer}.layernorm_before.weight": "encoder.{layer}.layernorm_before.weight",
    "vit.encoder.layer.{layer}.layernorm_before.bias": "encoder.{layer}.layernorm_before.bias",
    "vit.encoder.layer.{layer}.layernorm_after.weight": "encoder.{layer}.layernorm_after.weight",
    "vit.encoder.layer.{layer}.layernorm_after.bias": "encoder.{layer}.layernorm_after.bias",
    "vit.layernorm.weight": "layernorm.weight",
    "vit.layernorm.bias": "layernorm.bias",
    "classifier.weight": "classifier.weight",
    "classifier.bias": "classifier.bias",
}
for k, v in state_dict.items():
    if k.startswith("vit.encoder.layer"):
        layer = k.split(".")[3]
        nk = name_mapping[k.replace(layer, "{layer}")].format(layer=layer)
    else:
        nk = name_mapping[k]
    new_state_dict[nk] = v
    # print(k, tuple(v.shape))

for k, v in new_state_dict.items():

    print(k, tuple(v.shape))

import torch
torch.save(new_state_dict, "state_dict.pth")