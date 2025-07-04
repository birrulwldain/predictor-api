# src/worker.py

# Impor pustaka standar dan yang dibutuhkan
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import math
from scipy.signal import find_peaks
from io import StringIO

# =============================================================================
# BAGIAN 1: DEFINISI KELAS MODEL
# Definisi ini wajib ada agar kita bisa membuat struktur model sebelum memuat bobot.
# =============================================================================

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
        # Perbaikan untuk memastikan device konsisten
        return x + self.pe[:, :x.size(1), :].to(x.device)

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, factor=5):
        super(ProbSparseSelfAttention, self).__init__()
        self.d_model, self.nhead, self.d_k, self.factor = d_model, nhead, d_model // nhead, factor
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, L, _ = x.shape; H, D = self.nhead, self.d_k
        Q = self.q_linear(x).view(B, L, H, D).transpose(1, 2)
        K = self.k_linear(x).view(B, L, H, D).transpose(1, 2)
        V = self.v_linear(x).view(B, L, H, D).transpose(1, 2)
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
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class InformerModel(nn.Module):
    def __init__(self, **kwargs):
        super(InformerModel, self).__init__()
        self.d_model = kwargs["d_model"]
        self.embedding = nn.Linear(kwargs["input_dim"], self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model, kwargs["seq_length"])
        self.encoder_layers = nn.ModuleList([EncoderLayer(**kwargs) for _ in range(kwargs["num_encoder_layers"])])
        self.decoder = nn.Linear(self.d_model, kwargs["num_classes"])
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers: x = layer(x)
        return self.decoder(x)

# =============================================================================
# BAGIAN 2: INISIALISASI SAAT STARTUP WORKER
# Model dan aset dimuat sekali saja saat worker pertama kali aktif untuk efisiensi.
# =============================================================================

MODEL_CONFIG = {
    "input_dim": 1, "d_model": 32, "nhead": 4, "num_encoder_layers": 2,
    "dim_feedforward": 64, "dropout": 0.2, "seq_length": 4096,
    "attn_factor": 5, "num_classes": 18
}
TARGET_MAX_INTENSITY = 0.8

# Aset-aset ini secara otomatis disediakan oleh Cloudflare berdasarkan file wrangler.toml
model = InformerModel(**MODEL_CONFIG)
model.load_state_dict(torch.load(MODEL_ASSET, map_location='cpu'))
model.eval()

element_map = json.loads(ELEMENT_MAP_ASSET)
class_names = list(element_map.keys())
target_wavelengths = np.array(json.loads(WAVELENGTH_GRID_ASSET), dtype=np.float32)

print("Worker initialized successfully with model and assets.")

# =============================================================================
# BAGIAN 3: FUNGSI-FUNGSI LOGIKA
# =============================================================================

def prepare_asc_data(asc_content_string: str) -> np.ndarray:
    """Memproses string konten file .asc menjadi spektrum yang siap diprediksi."""
    df = pd.read_csv(StringIO(asc_content_string), sep=r'\s+', names=['wavelength', 'intensity'], comment='#')
    original_wavelengths, original_spectrum = df['wavelength'].values, df['intensity'].values
    
    # Koreksi Baseline
    anchor_points = np.percentile(original_spectrum, 10)
    baseline_points = original_spectrum < anchor_points
    if np.sum(baseline_points) > 5:
        coeffs = np.polyfit(original_wavelengths[baseline_points], original_spectrum[baseline_points], deg=5)
        baseline = np.polyval(coeffs, original_wavelengths)
        spectrum_corrected = original_spectrum - baseline
        spectrum_corrected[spectrum_corrected < 0] = 0
    else:
        spectrum_corrected = original_spectrum - np.min(original_spectrum)

    # Normalisasi
    max_val = np.max(spectrum_corrected)
    processed_spectrum = (spectrum_corrected / max_val) * TARGET_MAX_INTENSITY if max_val > 0 else spectrum_corrected
    
    # Resampling
    resampled_spectrum = np.interp(target_wavelengths, original_wavelengths, processed_spectrum)
    return resampled_spectrum.astype(np.float32)

# =============================================================================
# BAGIAN 4: HANDLER UTAMA WORKER
# Objek ini akan menangani setiap permintaan (request) yang masuk ke URL Worker Anda.
# =============================================================================

class Handler:
    async def fetch(self, request, env):
        # Header untuk mengizinkan Cross-Origin Resource Sharing (CORS)
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        }

        # Browser akan mengirim request OPTIONS terlebih dahulu (preflight)
        if request.method == 'OPTIONS':
            return self.response(b'', status=204, headers=headers)
            
        if request.method != 'POST':
            return self.response(json.dumps({"error": "Method Not Allowed"}), status=405, headers=headers)

        try:
            # 1. Dapatkan data .asc dari body request
            asc_content = await request.text()
            
            # 2. Proses dan siapkan data
            spectrum_data = prepare_asc_data(asc_content)
            
            # 3. Deteksi Puncak
            peak_indices, _ = find_peaks(spectrum_data, height=0.03, distance=8)
            
            # 4. Lakukan Prediksi
            input_tensor = torch.from_numpy(spectrum_data[np.newaxis, :, np.newaxis])
            with torch.no_grad():
                output_logits = model(input_tensor)
                full_probabilities = torch.sigmoid(output_logits).squeeze(0).cpu().numpy()
            full_predictions = (full_probabilities > 0.7).astype(int)

            # 5. Format Anotasi untuk dikirim ke Plotly di frontend
            annotations = []
            for i, peak_idx in enumerate(peak_indices):
                prediction_at_peak = full_predictions[peak_idx]
                detected_indices = np.where(prediction_at_peak == 1)[0]
                if detected_indices.size > 0:
                    label_lines = [f"{class_names[j]} ({full_probabilities[peak_idx, j]*100:.1f}%)" for j in detected_indices if class_names[j] != 'background']
                    if label_lines:
                        annotations.append(dict(
                            x=float(target_wavelengths[peak_idx]), 
                            y=float(spectrum_data[peak_idx]), 
                            text="<br>".join(label_lines), 
                            showarrow=True, arrowhead=2, ax=0, ay=-(40 + (i % 5) * 20),
                            font=dict(color="#ffffff"), bgcolor="#003366", opacity=0.8, borderpad=4
                        ))
            
            # 6. Kemas semua hasil ke dalam JSON untuk dikirim kembali ke frontend
            results = {
                "wavelengths": target_wavelengths.tolist(),
                "spectrum_data": spectrum_data.tolist(),
                "peak_wavelengths": target_wavelengths[peak_indices].tolist(),
                "peak_intensities": spectrum_data[peak_indices].tolist(),
                "annotations": annotations,
            }
            
            return self.response(json.dumps(results), headers={**headers, 'Content-Type': 'application/json'})

        except Exception as e:
            # Mengirim pesan error yang jelas jika terjadi masalah
            error_message = {"error": f"An unexpected error occurred: {str(e)}"}
            return self.response(json.dumps(error_message), status=500, headers=headers)
            
    def response(self, body, status=200, headers={}):
        """Fungsi helper untuk membuat objek Response."""
        # Impor Response di sini untuk kompatibilitas lingkungan worker
        from pyodide.http import Response
        return Response(body, status=status, headers=headers)

# Objek 'handler' ini yang akan diekspor dan dipanggil oleh Cloudflare
handler = Handler()