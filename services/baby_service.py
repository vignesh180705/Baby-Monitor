import torch
import torch.nn as nn
import numpy as np
import librosa
import argparse
import os
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GFCCTransformerModel(nn.Module):
    def __init__(self, input_dim=13, cnn_channels=64, transformer_dim=128, 
                 num_heads=4, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(cnn_channels)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cnn_channels, nhead=num_heads,
            dim_feedforward=transformer_dim, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attention = nn.Linear(cnn_channels, 1)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        attn = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(attn * x, dim=1)
        return self.classifier(context)

model = GFCCTransformerModel(input_dim=13, num_classes=2).to(DEVICE)
model.load_state_dict(torch.load("services/gfcc_transformer.pth", map_location=DEVICE))
model.eval()

def extract_gfcc_features(file_path):
    sig, sr = librosa.load(file_path, sr=None, mono=True)
    sig = sig.astype(float)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)

    window = SlidingWindow(0.025, 0.01, "hamming")
    features = gfcc(
        sig, fs=sr, num_ceps=13, window=window,
        nfilts=24, nfft=512, normalize="mvn"
    )
    return features.astype(np.float32)

def preprocess_audio(file_path, max_len=1000):
    features = extract_gfcc_features(file_path)
    
    if features.shape[0] > max_len:
        features = features[:max_len]
    else:
        pad = max_len - features.shape[0]
        features = np.pad(features, ((0, pad), (0, 0)))
        
    return features

def predict_audio(file_path, model):
    features = preprocess_audio(file_path)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()

    return pred, prob.cpu().numpy()[0]

def classify_audio_file(audio_path, model_path="gfcc_transformer.pth"):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # model = GFCCTransformerModel(input_dim=13, num_classes=2).to(DEVICE)
    # model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    # model.eval()

    prediction, probabilities = predict_audio(audio_path, model)
    
    class_map = {0: "The baby is not crying.", 1: "The baby is crying."}
    
    return {
        "predicted_label": class_map[prediction],
        "confidence": {
            "Non_Cry": float(probabilities[0]),
            "Cry": float(probabilities[1])
        }
    }
# parser = argparse.ArgumentParser(description="Classify baby cry audio as 'Cry' or 'Non Cry'.")
# parser.add_argument("audio_path", type=str, help="Path to the audio file you want to analyze")
# args = parser.parse_args()
# test_audio_path = args.audio_path

def get_cry_result(test_audio_path):
    try:
        result = classify_audio_file(
            test_audio_path,
            model_path="services/gfcc_transformer.pth"
        )
        return result

    except FileNotFoundError as e:
        return {
            "predicted_label": "Error",
            "confidence": {"Non_Cry": 0, "Cry": 0}
        }