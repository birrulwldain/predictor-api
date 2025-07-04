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
        
        # --- PERBAIKAN DI SINI ---
        # Secara eksplisit memberikan hanya argumen yang dibutuhkan oleh EncoderLayer.
        # Ini akan menyelesaikan TypeError yang Anda lihat.
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=self.d_model,
                nhead=kwargs["nhead"],
                dim_feedforward=kwargs["dim_feedforward"],
                dropout=kwargs["dropout"],
                attn_factor=kwargs["attn_factor"]
            ) for _ in range(kwargs["num_encoder_layers"])
        ])
        # --- AKHIR PERBAIKAN ---
        
        self.decoder = nn.Linear(self.d_model, kwargs["num_classes"])

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers: 
            x = layer(x)
        return self.decoder(x)
import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve

def als_baseline_correction(y, lam, p, niter=10):
    """
    Applies Asymmetric Least Squares (ALS) baseline correction.

    Parameters
    ----------
    y : array_like
        The input signal (spectrum data).
    lam : float
        Lambda parameter for the ALS algorithm. Controls smoothness.
        Larger lambda means smoother baseline.
    p : float
        Asymmetry parameter for the ALS algorithm. Controls how much
        the baseline is allowed to follow the signal.
        0 < p < 1. Smaller p means baseline is more likely to be below the signal.
    niter : int, optional
        Number of iterations for the ALS algorithm. Default is 10.

    Returns
    -------
    array_like
        The estimated baseline.
    """
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())  # Precompute D.T * D
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0)
        Z = W + D
        baseline = spsolve(Z, w * y)
        w = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline

def load_assets():
    """Memuat semua aset yang dibutuhkan dan mengembalikannya."""
    print("Memuat model dan aset-aset penting...")
    
    model = InformerModel(**MODEL_CONFIG)
    state_dict = torch.load("assets/informer_multilabel_model.pth", map_location='cpu')
    # Remove '_orig_mod.' prefix if it exists (common with torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    with open("assets/element-map-18a.json", 'r') as f:
        element_map = json.load(f)
    
    with open("assets/wavelengths_grid.json", 'r') as f:
        target_wavelengths = np.array(json.load(f), dtype=np.float32)
        
    print("Aset berhasil dimuat.")
    return model, element_map, target_wavelengths