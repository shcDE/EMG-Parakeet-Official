import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


# ==================== Gaddy-style Attention ====================

class LearnedRelativePositionalEmbedding(nn.Module):
    def __init__(self, max_relative_pos, num_heads, embedding_dim, unmasked=False,
                 heads_share_embeddings=False, add_to_values=False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values

        num_embeddings = 2 * max_relative_pos - 1 if unmasked else max_relative_pos
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2

        initial_stddev = embedding_dim ** (-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)

    def forward(self, query, saved_state=None):
        if saved_state is not None and "prev_key" in saved_state:
            length = saved_state["prev_key"].shape[-2] + 1
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used_embeddings = self.get_embeddings_for_query(length)
        values_embeddings = used_embeddings[..., 1] if self.add_to_values else None
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        positional_logits = self.relative_to_absolute_indexing(positional_logits, decoder_step)
        return positional_logits, values_embeddings

    def get_embeddings_for_query(self, length):
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded = nn.functional.pad(self.embeddings, (0, 0, 0, 0, pad_length, pad_length))
            return padded.narrow(-3, start_pos, 2 * length - 1)
        else:
            with torch.no_grad():
                padded = nn.functional.pad(self.embeddings, (0, 0, 0, 0, pad_length, 0))
            return padded.narrow(-3, start_pos, length)

    def calculate_positional_logits(self, query, relative_embeddings):
        if self.heads_share_embeddings:
            positional_logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            positional_logits = positional_logits.contiguous().view(
                positional_logits.shape[0], -1, positional_logits.shape[-1]
            )
        length = query.size(0)
        if length > self.max_relative_pos:
            pad_length = length - self.max_relative_pos
            positional_logits[:, :, :pad_length] -= 1e8
            if self.unmasked:
                positional_logits[:, :, -pad_length:] -= 1e8
        return positional_logits

    def relative_to_absolute_indexing(self, x, decoder_step):
        length, bsz_heads, _ = x.shape
        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)
        if self.unmasked:
            x = nn.functional.pad(x, (0, 1))
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = nn.functional.pad(x, (0, length - 1))
            x = x.view(bsz_heads, length + 1, 2 * length - 1)
            return x[:, :length, length - 1:]
        else:
            x = nn.functional.pad(x, (1, 0))
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length + 1, length)
            return x[:, 1:, :]


class MultiHeadAttention(nn.Module):
    """Gaddy-style MHA with Learned Relative Positional Embedding (batch_first=True)."""
    def __init__(self, d_model, n_head, dropout=0.1,
                 relative_positional=True, relative_positional_distance=100):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        d_qkv = d_model // n_head
        assert d_qkv * n_head == d_model
        self.d_qkv = d_qkv

        self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
        self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
        nn.init.xavier_normal_(self.w_q)
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)

        self.dropout = nn.Dropout(dropout)

        if relative_positional:
            self.relative_positional = LearnedRelativePositionalEmbedding(
                relative_positional_distance, n_head, d_qkv, True
            )
        else:
            self.relative_positional = None

    def forward(self, x):
        """
        Args:
            x: (batch_size, length, d_model)  — batch_first format
        Returns:
            (batch_size, length, d_model)
        """
        # Convert to (length, batch_size, d_model) for einsum compatibility
        x_t = x.transpose(0, 1)  # (L, B, D)

        q = torch.einsum('tbf,hfa->bhta', x_t, self.w_q)
        k = torch.einsum('tbf,hfa->bhta', x_t, self.w_k)
        v = torch.einsum('tbf,hfa->bhta', x_t, self.w_v)

        logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)

        if self.relative_positional is not None:
            q_pos = q.permute(2, 0, 1, 3)  # (L, B, H, D)
            l, b, h, d = q_pos.size()
            position_logits, _ = self.relative_positional(q_pos.reshape(l, b * h, d))
            logits = logits + position_logits.view(b, h, l, l)

        probs = F.softmax(logits, dim=-1)
        probs = self.dropout(probs)

        o = torch.einsum('bhqk,bhka->bhqa', probs, v)
        out = torch.einsum('bhta,haf->tbf', o, self.w_o)

        # Back to (B, L, D)
        return out.transpose(0, 1)


# ==================== Transformer ====================

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


class BaseTransformerBlock(nn.Module):
    """Standard Pre-Norm Residual, Base ReLU FFN, Learned Relative Positional Attention."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 relative_positional_distance=100):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            d_model=dim, n_head=num_heads, dropout=drop,
            relative_positional=True,
            relative_positional_distance=relative_positional_distance,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = BaseFeedForward(dim, int(dim * mlp_ratio), dropout=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BaseTransformer(nn.Module):
    """
    Ablation 4: Learned Relative Positional Attention + Base ReLU FFN + Standard Residual
      ≈ Gaddy (2020) baseline architecture (Pre-Norm variant)
    """
    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, drop_rate=0.0,
                 relative_positional_distance=100):
        super().__init__()

        self.blocks = nn.ModuleList([
            BaseTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=0.1,
                relative_positional_distance=relative_positional_distance,
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x