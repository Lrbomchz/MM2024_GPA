from functools import partial

import torch
import torch.nn as nn

import vision_transformer
'''
import timm

class Attention(vision_transformer.Attention):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
    def forward(self, x, get_weight=False):
        B, N, C = x.shape
        #print(B,N,C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if get_weight:
            return attn.data.detach()

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(vision_transformer.Block):

    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.attn = Attention(**kwargs)

    def forward(self, x, get_weight=False):
        if get_weight:
            return self.attn(self.norm1(x), get_weight)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
'''

class VisionTransformer(vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, block_and_head = (-1, -1), depth=12, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm
        self.nblock, self.nhead = block_and_head
        
    def forward_features(self, x):
        if self.nblock == -1 or self.nhead == -1:
            return 0
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            if idx != self.nblock:
                x = blk(x)
            if idx == self.nblock:
                x = blk(x, get_weight=True)
                atten = torch.sum(x,axis=2)
                if self.nhead != -1:
                    atten_ref = atten[:, self.nhead, 1:]
                else:
                    atten_ref = torch.sum(atten[:, :, 1:], axis=1).unsqueeze(1)
                    
                atten_min = torch.min(atten_ref, axis=-1).values.unsqueeze(dim=-1)
                atten_max = torch.max(atten_ref, axis=-1).values.unsqueeze(dim=-1)
                #print(atten_ref.shape, atten_min.shape, atten_max.shape)
                atten_ref = (atten_ref - atten_min)/(atten_max - atten_min)
                return atten_ref
    
    def forward(self, x):
        x = x.detach()
        x = self.forward_features(x)
        return x

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model