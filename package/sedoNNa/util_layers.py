import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)
        self.cross_o = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.nhead, D // self.nhead).transpose(1, 2)  # (B, H, T, D/H)

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).reshape(B, T, H * Dh)

    def forward(self, tgt, memory):
        # --- self attention (no mask)
        q = self._split_heads(self.q_proj(tgt))
        k = self._split_heads(self.k_proj(tgt))
        v = self._split_heads(self.v_proj(tgt))
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        tgt2 = self._merge_heads(attn_out)
        tgt = self.norm1(tgt + self.dropout(self.o_proj(tgt2)))

        # --- cross attention (no mask)
        q = self._split_heads(self.cross_q(tgt))
        k = self._split_heads(self.cross_k(memory))
        v = self._split_heads(self.cross_v(memory))
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        tgt2 = self._merge_heads(attn_out)
        tgt = self.norm2(tgt + self.dropout(self.cross_o(tgt2)))

        # --- feedforward
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt
    

class TransformerDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(layers[0].d_model)

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.norm(tgt)