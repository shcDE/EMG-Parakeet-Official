import torch
import torch.nn as nn
from timm.layers import DropPath
from rotary_embedding_torch import RotaryEmbedding


class BaseFeedForward(nn.Module):
    """Gaddy-style base FFN: Linear → ReLU → Dropout → Linear → Dropout."""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class RotarySelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class RotaryTransformerBlock(nn.Module):
    """Standard Pre-Norm Residual, Base ReLU FFN."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotarySelfAttentionBlock(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BaseFeedForward(dim, int(dim * mlp_ratio), dropout=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RoPETransformer(nn.Module):
    """
    Ablation 3: RoPE + Base ReLU FFN + Standard Residual (no mHC, no GEGLU)
    """
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, drop_rate=0.0):
        super().__init__()

        self.blocks = nn.ModuleList([
            RotaryTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=0.1,
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x