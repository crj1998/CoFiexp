import math

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, inputs, hidden_z=None):
        if hidden_z is not None:
            remaining_index = torch.where(~hidden_z.eq(0))[0]
            compressed_input = torch.index_select(inputs, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            output = inputs.clone()
            output[:, :, remaining_index] = normed_input
        else:
            output = F.layer_norm(inputs, self.normalized_shape, self.weight, self.bias, self.eps)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, hidden_size, image_size=224, patch_size=16, num_channels=3, hidden_dropout_prob=0.0) -> None:
        super().__init__()

        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, inputs: torch.Tensor, hidden_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_channels, height, width = inputs.shape
        assert (num_channels, height, width) == (self.num_channels, *self.image_size), "Shape mismatch"

        embeddings = self.projection(inputs).flatten(start_dim=2).transpose(1, 2)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embedding

        if hidden_z is not None:
            embeddings = embeddings.mul(hidden_z)

        embeddings = self.dropout(embeddings)

        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, qkv_bias=True, attention_dropout_rate=0.0, hidden_dropout_rate=0.0) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, all_head_size, bias=qkv_bias)
        self.key   = nn.Linear(hidden_size, all_head_size, bias=qkv_bias)
        self.value = nn.Linear(hidden_size, all_head_size, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(attention_dropout_rate)

        self.proj = nn.Linear(all_head_size, hidden_size)
        self.proj_dropout = nn.Dropout(hidden_dropout_rate)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        shape = (*x.size()[:-1], self.num_attention_heads, self.attention_head_size)
        x = x.view(shape).permute(0, 2, 1, 3)
        return x

    def forward(
        self, inputs: torch.Tensor,
        heads_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None
    ):
        q = self.transpose_for_scores(self.query(inputs))
        k = self.transpose_for_scores(self.key(inputs))
        v = self.transpose_for_scores(self.value(inputs))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        if heads_z is not None:
            context_layer = context_layer.mul(heads_z)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        shape = (*context_layer.size()[:-2], self.all_head_size)
        context_layer = context_layer.view(shape)

        outputs = self.proj(context_layer)
        if hidden_z is not None:
            outputs = outputs.mul(hidden_z)
        outputs = self.proj_dropout(outputs)

        return outputs

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_rate=0.0) -> None:
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_rate)

    def forward(self, 
        hidden_states: torch.Tensor,
        intermediate_z: Optional[torch.Tensor] = None,
        hidden_z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        if intermediate_z is not None:
            hidden_states = hidden_states.mul(intermediate_z)

        hidden_states = self.dense2(hidden_states)
        if hidden_z is not None:
            hidden_states = hidden_states.mul(hidden_z)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class TransformerLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size, qkv_bias=True, attention_dropout_rate=0.0, hidden_dropout_rate=0.0, layer_norm_eps=1e-12) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_attention_heads, qkv_bias, attention_dropout_rate, hidden_dropout_rate)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size, hidden_dropout_rate)
        self.layernorm_before = LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layernorm_after  = LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(
        self, inputs: torch.Tensor,
        hidden_z: Optional[torch.Tensor] = None,
        heads_z: Optional[torch.Tensor] = None,
        mha_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        ffn_z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # in ViT, layernorm is applied before self-attention
        mha_outputs = self.mha(
            self.layernorm_before(inputs, hidden_z=hidden_z),
            heads_z = heads_z,
            hidden_z = hidden_z,
        )
        if mha_z is not None:
            mha_outputs = mha_outputs.mul(mha_z)
        # first residual connection
        hidden_states = inputs + mha_outputs

        # in ViT, layernorm is also applied after self-attention
        ffn_outputs = self.ffn(
            self.layernorm_after(hidden_states, hidden_z=hidden_z),
            intermediate_z = intermediate_z,
            hidden_z = hidden_z,
        )
        if ffn_z is not None:
            ffn_outputs = ffn_outputs.mul(ffn_z)
        outputs = hidden_states + ffn_outputs

        return outputs

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = inputs[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = PatchEmbedding(config.hidden_size, config.image_size, config.patch_size, config.num_channels, config.hidden_dropout_prob)
        self.encoder = nn.ModuleList([
            TransformerLayer(
                config.hidden_size, config.num_attention_heads, config.intermediate_size, 
                config.qkv_bias, config.attention_probs_dropout_prob, config.hidden_dropout_prob,
                config.layer_norm_eps
            ) for _ in range(config.num_hidden_layers)
        ])

        self.layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.pooler = Pooler(config.hidden_size)
        self.pooler = nn.Identity()

        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def reset_classifier(self, num_classes):
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(
        self, inputs: torch.Tensor, 
        hidden_z: Optional[torch.Tensor] = None,
        heads_z: Optional[torch.Tensor] = None,
        mha_z: Optional[torch.Tensor] = None,
        intermediate_z: Optional[torch.Tensor] = None,
        ffn_z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        hidden_states = self.embedding(inputs, hidden_z=hidden_z)

        for layer, module in enumerate(self.encoder):
            hidden_states = module(
                hidden_states,
                hidden_z = hidden_z,
                heads_z = heads_z[layer] if heads_z is not None else None,
                mha_z = mha_z[layer] if mha_z is not None else None,
                intermediate_z=intermediate_z[layer] if intermediate_z is not None else None,
                ffn_z=ffn_z[layer] if ffn_z is not None else None,                
            )

        encoder_output = self.layernorm(hidden_states, hidden_z=hidden_z)
        pooled_output = self.pooler(encoder_output) if self.pooler is not None else None

        if hidden_z is not None:
            pooled_output = pooled_output.mul(hidden_z)

        logits = self.classifier(pooled_output[:, 0, :])

        return logits


def vit_base_patch16_224(num_classes=1000):
    class ViTConfig:
        model_type = "vit"
        num_labels = num_classes

        hidden_size = 768
        num_hidden_layers = 12
        num_attention_heads = 12
        intermediate_size = 4 * 768
        hidden_act = "gelu"
        hidden_dropout_prob = 0.0
        attention_probs_dropout_prob = 0.0
        initializer_range = 0.02
        layer_norm_eps = 1e-12
        image_size = 224
        patch_size = 16
        num_channels = 3
        qkv_bias = True
        # encoder_stride = 16


    config = ViTConfig()
    model = VisionTransformer(config)

    return model

if __name__ == '__main__':
    model = vit_base_patch16_224(1000)
    model.eval()
    dummy_input = torch.rand(1, 3, 224, 224)
    state_dict = torch.load("state_dict.pth")
    model.load_state_dict(state_dict)
    with torch.no_grad():
        logits = model(dummy_input)
        print(logits.shape)
    
    from l0module import L0Module
    l0_module = L0Module(model.config, lagrangian_warmup=200, target_sparsity=0.5)
    l0_module.train()
    zs = l0_module()
    l0_module.eval()
    zs = l0_module()
    result = l0_module.calculate_model_size(zs)

    with torch.no_grad():
        logits = model(dummy_input, **zs)
        print(logits.shape)