import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalMLPPositionalEmbedding(nn.Module):
    def __init__(self, dim=128):
        """
        Sinusoidal positional encoding + MLP
        """
        super().__init__()
        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        # x: [B, seq_len]
        sine = torch.sin(x[:, :, None] * self.div_term[None, None, :].to(x.device))
        cosine = torch.cos(x[:, :, None] * self.div_term[None, None, :].to(x.device))
        encoding = torch.cat([sine, cosine], dim=-1)
        encoding = F.relu(self.fc1(encoding))
        encoding = self.fc2(encoding)
        return encoding


class FluxTransformerDecoder(nn.Module):
    def __init__(self, n_physical_param=10, n_wavelength=602,
                 d_model=128, nhead=8, num_layers=4, learnedPE=False):
        super().__init__()
        self.d_model = d_model
        self.seq_len = n_wavelength
        self.learnedPE = learnedPE
        # Memory embedding
        self.encoder_proj = SinusoidalMLPPositionalEmbedding(d_model)
        self.physical_param_embd = nn.Parameter(torch.randn(1, n_physical_param, d_model))
        self.memory_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # Query embedding
        if learnedPE:
            self.query = nn.Parameter(torch.randn(1, n_wavelength, d_model))
        else:
            self.query_embd = SinusoidalMLPPositionalEmbedding(d_model)
            grid = torch.linspace(0, 1, steps=n_wavelength)[None, :]
            self.register_buffer("grid", grid)
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True  # pre-layernorm
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # post-norm after final layer
        )
        # Output projection
        self.output_proj = nn.Linear(d_model, 1)
        
    
    def forward(self, physical_param):
        """
        physical_param: [B, n_physical_param]
        Returns: [B, n_wavelength]
        """
        B = physical_param.shape[0]
        # Memory: embed + positional + scale
        memory = self.encoder_proj(physical_param) + self.physical_param_embd  # [B, n_physical_param, d_model]
        memory = memory * (self.d_model ** 0.5)  # scale
        memory = self.memory_ff(memory)
        # Query / target
        if self.learnedPE:
            tgt = self.query.repeat(B, 1, 1)
        else:
            tgt = self.query_embd(self.grid).repeat(B, 1, 1)
        # Transformer decoder
        decoded = self.decoder(tgt, memory)  # [B, seq_len, d_model]
        # Project to scalar
        out = self.output_proj(decoded)  # [B, seq_len, 1]
        return out.squeeze(-1)           # [B, seq_len]