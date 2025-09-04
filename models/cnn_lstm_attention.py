# models/cnn_lstm_attention.py
import os
import torch
import torch.nn as nn
import torchvision.models as models

# Make CuBLAS deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

# CNN Encoder (ResNet50 -> per-patch features of size 2048)
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True):
        super(ResNet50Encoder, self).__init__()
        res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        self.features = nn.Sequential(*list(res.children())[:-2])  # (B,2048,H/32,W/32)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Force output to 7x7 grid

        # Freeze all params
        for param in self.features.parameters():
            param.requires_grad = False

        # Optionally unfreeze last block (layer4)
        if fine_tune:
            for param in self.features[-1].parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.features(x)       # (B,2048,H/32,W/32)
        x = self.avgpool(x)        # (B,2048,7,7)
        B, C, H, W = x.size()
        x = x.view(B, C, -1)       # (B,2048,49)
        x = x.permute(0, 2, 1)     # (B,49,2048)
        return x

# Attention Module
class Attention(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(feature_size + hidden_size, hidden_size)
        self.attn_weights = nn.Linear(hidden_size, 1)

    def forward(self, features, hidden_state):
        hidden_state_exp = hidden_state.unsqueeze(1).repeat(1, features.size(1), 1)
        combined = torch.cat((features, hidden_state_exp), dim=2)   # (B, num_patches, feat+hidden)
        attn_hidden = torch.tanh(self.attention(combined))          # (B, num_patches, hidden)
        attention_logits = self.attn_weights(attn_hidden).squeeze(2)  # (B, num_patches)
        attention_weights = torch.softmax(attention_logits, dim=1)
        context = (features * attention_weights.unsqueeze(2)).sum(dim=1)  # (B, feature_size)
        return context, attention_weights

# Decoder with Attention (LSTM)
class DecoderWithAttention(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, embedding_dim, vocab_size):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(feature_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(0.0 if num_layers == 1 else 0.5),
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions=None, max_seq_len=None, teacher_forcing_ratio=0.9):
        B = features.size(0)
        device = features.device
        max_seq_len = max_seq_len if max_seq_len is not None else (captions.size(1) if captions is not None else 20)

        h = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, B, self.hidden_size, device=device)
        outputs = torch.zeros(B, max_seq_len, self.fc.out_features, device=device)

        word_input = torch.full((B,), 2, dtype=torch.long, device=device)  # <sos>=2

        for t in range(1, max_seq_len):
            embeddings = self.embedding(word_input)               # (B, emb)
            context, _ = self.attention(features, h[-1])          # (B, feat)
            lstm_input = torch.cat([embeddings, context], dim=1).unsqueeze(1)  # (B,1,emb+feat)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            logits = self.fc(out.squeeze(1))                      # (B,V)
            outputs[:, t, :] = logits

            top1 = logits.argmax(1)
            if captions is not None and torch.rand(1).item() < teacher_forcing_ratio:
                word_input = captions[:, t]
            else:
                word_input = top1

        return outputs

# Wrapper Model
class ImgCap(nn.Module):
    def __init__(self, lstm_hidden_size, feature_size=2048, num_layers=1, vocab_size=None, embedding_dim=512, fine_tune=True):
        super(ImgCap, self).__init__()
        self.cnn = ResNet50Encoder(fine_tune=fine_tune)
        self.decoder = DecoderWithAttention(
            feature_size=feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size
        )

    def forward(self, images, captions):
        features = self.cnn(images)
        logits = self.decoder(features, captions)
        return logits

    @torch.no_grad()
    def generate(self, images, idx2tok, max_len=20, device=None, start_idx=2, end_idx=3):
        device = device or next(self.parameters()).device
        images = images.to(device)
        features = self.cnn(images)
        B = features.size(0)

        h = torch.zeros(self.decoder.lstm.num_layers, B, self.decoder.lstm.hidden_size, device=device)
        c = torch.zeros(self.decoder.lstm.num_layers, B, self.decoder.lstm.hidden_size, device=device)

        word_input = torch.full((B,), start_idx, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        finished = [False] * B

        for t in range(max_len):
            embeddings = self.decoder.embedding(word_input)
            context, _ = self.decoder.attention(features, h[-1])
            lstm_input = torch.cat([embeddings, context], dim=1).unsqueeze(1)
            out, (h, c) = self.decoder.lstm(lstm_input, (h, c))
            logits = self.decoder.fc(out.squeeze(1))
            top1 = logits.argmax(1)

            for i in range(B):
                if not finished[i]:
                    token_idx = top1[i].item()
                    if token_idx == end_idx:
                        finished[i] = True
                    else:
                        sequences[i].append(token_idx)

            word_input = top1
            if all(finished):
                break

        outputs = []
        for seq in sequences:
            tokens = [idx2tok(idx) for idx in seq]
            outputs.append(" ".join(tokens))
        return outputs

# Factory Function (fixed)
def CNN_LSTM_Attention(vocab_size, feature_size=2048, hidden_size=512, num_layers=1,
                       embedding_dim=512, fine_tune=True):
    return ImgCap(
        lstm_hidden_size=hidden_size,
        feature_size=feature_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        fine_tune=fine_tune
    )

