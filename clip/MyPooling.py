import torch
import torch.nn.functional as F
from torch import nn


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        
    def change(self):
        new_positional_embedding = torch.cat((self.positional_embedding[:1], self.positional_embedding), dim=0)
        self.positional_embedding = nn.Parameter(new_positional_embedding)
        self.half()

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        mean_x = x.mean(dim=0, keepdim=True)
        x = torch.cat([mean_x, mean_x, x], dim=0)  # (HW+2)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+2)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0], x[1]
    
    
class MyPooling(nn.Module):
    
    def __init__(self, clip_attn_pool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.attnpool = AttentionPool2d(7, 2048, 32, 1024)
        self.load_from(clip_attn_pool)
        
    def load_from(self, clip_attn_pool):
        self.attnpool.load_state_dict(clip_attn_pool.state_dict())
        self.attnpool.change()
        
    def forward(self, clip_featmaps):
        cls_featvecs, bg_featvecs = self.attnpool(clip_featmaps)
        return cls_featvecs, bg_featvecs
        
        