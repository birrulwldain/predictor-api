# app/model.py

import torch
import torch.nn as nn
import math
import json
import numpy as np

# Konfigurasi arsitektur model
MODEL_CONFIG = {
    "input_dim": 1, "d_model": 32, "nhead": 4, "num_encoder_layers": 2,
    "dim_feedforward": 64, "dropout": 0.2, "seq_length": 4096,
    "attn_factor": 5, "num_classes": 18
}

# --- Definisi Kelas-kelas Model (Lengkap) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, factor=5):
        super(ProbSparseSelfAttention, self).__init__()
        self.d_model, self.nhead, self.d_k, self.factor = d_model, nhead, d_model // nhead, factor
        self.q_linear, self.k_linear, self.v_linear, self.out_linear = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, L, _ = x.shape; H, D = self.nhead, self.d_k
        Q, K, V = self.q_linear(x).view(B, L, H, D).transpose(1, 2), self.k_linear(x).view(B, L, H, D).transpose(1, 2), self.v_linear(x).view(B, L, H, D).transpose(1, 2)
        U = min(L, int(self.factor * math.log(L)) if L > 1 else L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        if mask is not None: scores.masked_fill_(mask == 0, -float('inf'))
        top_k, _ = torch.topk(scores, U, dim=-1)
        scores.masked_fill_(scores < top_k[..., -1, None], -float('inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_linear(context)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, attn_factor):
        super(EncoderLayer, self).__init__()
        self.self_attention = ProbSparseSelfAttention(d_model, nhead, dropout, attn_factor)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
    def forward(self, x, mask=None):
        x = self.norm1(x + self.dropout1(self.self_attention(x, mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x

class InformerModel(nn.Module):
    def __init__(self, **kwargs):
        super(InformerModel, self).__init__()
        self.d_model = kwargs["d_model"]
        self.embedding = nn.Linear(kwargs["input_dim"], self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, kwargs["seq_length"])
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=self.d_model, **{k:v for k,v in kwargs.items() if k not in ['input_dim', 'd_model', 'seq_length', 'num_classes']}) for _ in range(kwargs["num_encoder_layers"])])
        self.decoder = nn.Linear(self.d_model, kwargs["num_classes"])
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers: x = layer(x)
        return self.decoder(x)

def load_assets():
    """Memuat semua aset yang dibutuhkan dan mengembalikannya."""
    print("Memuat model dan aset-aset penting...")
    
    model = InformerModel(**MODEL_CONFIG)
    model.load_state_dict(torch.load("assets/informer_multilabel_model.pth", map_location='cpu'))
    model.eval()

    with open("assets/element-map-18a.json", 'r') as f:
        element_map = json.load(f)
    
    with open("assets/wavelengths_grid.json", 'r') as f:
        target_wavelengths = np.array(json.load(f), dtype=np.float32)
        
    print("Aset berhasil dimuat.")
    return model, element_map, target_wavelengths