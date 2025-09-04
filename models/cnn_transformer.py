# models/cnn_transformer.py
import os
import math
import torch
import torch.nn as nn
import torchvision.models as models

# Make CuBLAS deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

# Positional Encoding (sine/cosine)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x + self.pe[:, : x.size(1)]
        return x

# CNN Encoder (ResNet50)
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True):
        super(ResNet50Encoder, self).__init__()
        res = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        self.features = nn.Sequential(*list(res.children())[:-2])  # (B,2048,H/32,W/32)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # (B,2048,7,7)

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

# Decoder with Transformer
class DecoderWithTransformer(nn.Module):
    def __init__(self, feature_size, embedding_dim, num_layers, vocab_size, nhead=8, ff_dim=2048):
        super(DecoderWithTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)

        # Project CNN features (2048 -> embedding_dim)
        self.feature_proj = nn.Linear(feature_size, embedding_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, features, captions):
        """
        features: (B, num_patches, feature_size)
        captions: (B, T)
        """
        device = features.device
        B, T = captions.size()

        # Embed captions + add positional encoding
        tgt_embeddings = self.embedding(captions)  # (B, T, D)
        tgt_embeddings = self.pos_encoder(tgt_embeddings)

        # Project CNN features to embedding dim
        memory = self.feature_proj(features)  # (B, num_patches, D)

        # Generate subsequent mask (no cheating future tokens)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        out = self.transformer_decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=tgt_mask
        )  # (B, T, D)

        logits = self.fc(out)  # (B, T, V)
        return logits

# Wrapper Model
class ImgCap(nn.Module):
    def __init__(self, feature_size=2048, num_layers=2, vocab_size=None, embedding_dim=512, fine_tune=True):
        super(ImgCap, self).__init__()
        self.cnn = ResNet50Encoder(fine_tune=fine_tune)
        self.decoder = DecoderWithTransformer(
            feature_size=feature_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            vocab_size=vocab_size
        )

    def forward(self, images, captions):
        features = self.cnn(images)  # (B, 49, 2048)
        logits = self.decoder(features, captions)
        return logits

    @torch.no_grad()
    def generate(self, images, idx2tok, max_len=20, device=None, start_idx=2, end_idx=3):
        """
        Greedy decoding for caption generation
        """
        device = device or next(self.parameters()).device
        images = images.to(device)
        features = self.cnn(images)             # (B, 49, 2048)
        memory = self.decoder.feature_proj(features)  # (B, 49, D)

        B = features.size(0)
        word_input = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]

        for t in range(max_len):
            tgt_embeddings = self.decoder.embedding(word_input)   # (B, t, D)
            tgt_embeddings = self.decoder.pos_encoder(tgt_embeddings)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embeddings.size(1)).to(device)

            out = self.decoder.transformer_decoder(
                tgt=tgt_embeddings,
                memory=memory,
                tgt_mask=tgt_mask
            )  # (B, t, D)

            logits = self.decoder.fc(out[:, -1, :])  # (B, V)
            top1 = logits.argmax(1)                  # (B,)

            for i in range(B):
                token_idx = top1[i].item()
                if token_idx == end_idx:
                    continue
                else:
                    sequences[i].append(token_idx)

            word_input = torch.cat([word_input, top1.unsqueeze(1)], dim=1)

        outputs = []
        for seq in sequences:
            tokens = [idx2tok(idx) for idx in seq]
            outputs.append(" ".join(tokens))
        return outputs

# Factory Function
def CNN_Transformer(vocab_size, feature_size=2048, embedding_dim=512, num_layers=2, fine_tune=True):
    return ImgCap(
        feature_size=feature_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        fine_tune=fine_tune
    )
