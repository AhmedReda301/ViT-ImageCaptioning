# models/cnn_lstm.py
import os
import torch
import torch.nn as nn
import torchvision.models as models

# Make CuBLAS deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)


# CNN Encoder
class ResNet50Vec(nn.Module):
    """
    ResNet50 backbone that outputs a global image feature vector of size out_dim
    """
    def __init__(self, pretrained=True, out_dim=1024, fine_tune=True, dropout=0.5):
        super(ResNet50Vec, self).__init__()
        res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Replace classification head
        in_features = res.fc.in_features
        res.fc = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Freeze all except fc or optionally unfreeze some layers
        for name, param in res.named_parameters():
            param.requires_grad = ("fc" in name)

        if fine_tune:
            for param in list(res.layer4.parameters()):
                param.requires_grad = True

        self.model = res
        self.out_dim = out_dim

    def forward(self, x):
        return self.model(x)  # (B, out_dim)


# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim, vocab_size, dropout=0.5):
        super(LSTMDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(input_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features, captions):
        """
        image_features: (B, input_size)
        captions: (B, T)
        """
        batch_size = image_features.size(0)
        device = image_features.device

        proj_img = self.projection(image_features).unsqueeze(1)  # (B,1,emb)
        emb_text = self.embedding(captions[:, :-1])             # (B,T-1,emb)

        lstm_input = torch.cat([proj_img, emb_text], dim=1)     # (B,T,emb)
        lstm_out, _ = self.lstm(lstm_input)
        logits = self.fc(lstm_out)                               # (B,T,V)

        return logits

    @torch.no_grad()
    def generate_stepwise(self, image_features, idx2tok, max_len=20, start_idx=2, end_idx=3):
        device = image_features.device
        B = image_features.size(0)
        proj_img = self.projection(image_features)  # (B,emb)

        h = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=device)

        word_input = torch.full((B,), start_idx, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        finished = [False] * B

        for t in range(max_len):
            lstm_in = proj_img.unsqueeze(1) if t == 0 else self.embedding(word_input).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_in, (h, c))
            logits = self.fc(out.squeeze(1))
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

        texts = []
        for seq in sequences:
            tokens = [idx2tok(i) for i in seq]
            texts.append(" ".join(tokens))
        return texts


# Wrapper
class ImgCap(nn.Module):
    def __init__(self, cnn_feature_size, lstm_hidden_size, num_layers, vocab_size, embedding_dim, fine_tune=True, dropout=0.5):
        super(ImgCap, self).__init__()
        self.cnn = ResNet50Vec(out_dim=cnn_feature_size, fine_tune=fine_tune, dropout=dropout)
        self.decoder = LSTMDecoder(
            input_size=cnn_feature_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )

    def forward(self, images, captions):
        features = self.cnn(images)
        logits = self.decoder(features, captions)
        return logits

    @torch.no_grad()
    def generate(self, images, idx2tok, max_len=20, device=None, start_idx=2, end_idx=3):
        device = device or next(self.parameters()).device
        images = images.to(device)
        feats = self.cnn(images)
        return self.decoder.generate_stepwise(feats, idx2tok, max_len=max_len, start_idx=start_idx, end_idx=end_idx)


# Factory function
def CNN_LSTM(vocab_size, cnn_feature_size=1024, lstm_hidden_size=1024, num_layers=1,
             embedding_dim=512, fine_tune=True, dropout=0.5):
    return ImgCap(
        cnn_feature_size=cnn_feature_size,
        lstm_hidden_size=lstm_hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        fine_tune=fine_tune,
        dropout=dropout
    )

