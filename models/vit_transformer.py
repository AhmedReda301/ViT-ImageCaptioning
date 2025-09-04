# models/vit_transformer.py
import math
import torch
import torch.nn as nn
import torchvision.models as models

# Positional Encoding
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


# ViT Encoder
class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True, fine_tune=True):
        super(ViTEncoder, self).__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        vit = models.vit_b_16(weights=weights)

        self.encoder = vit.encoder
        self.patch_embed = vit.conv_proj
        self.cls_token = vit.class_token
        self.pos_embed = vit.encoder.pos_embedding
        self.hidden_dim = vit.hidden_dim  # usually 768

        for p in self.parameters():
            p.requires_grad = False

        # Optionally unfreeze 
        if fine_tune:
            for p in self.encoder.layers[-1].parameters():
                p.requires_grad = True

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.patch_embed(x)                  # (B, D, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)         # (B, N, D)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)           # (B, 1+N, D)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.encoder(x)                      # (B, 1+N, D)
        memory = x[:, 1:, :]                     # (B, N, D) drop CLS
        return memory


# Transformer Decoder
class DecoderWithTransformer(nn.Module):
    def __init__(self, feature_size, embedding_dim, num_layers, vocab_size, nhead=8, ff_dim=2048):
        super(DecoderWithTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)

        # Project ViT features (768 -> embedding_dim)
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

        # Project ViT features to embedding dim
        memory = self.feature_proj(features)  # (B, num_patches, D)

        # Generate subsequent mask
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
    def __init__(self, feature_size=768, num_layers=2, vocab_size=None, embedding_dim=512, fine_tune=True):
        super(ImgCap, self).__init__()
        self.vit = ViTEncoder(fine_tune=fine_tune)
        self.decoder = DecoderWithTransformer(
            feature_size=feature_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            vocab_size=vocab_size
        )

    def forward(self, images, captions):
        features = self.vit(images)  # (B, N, 768)
        logits = self.decoder(features, captions)
        return logits

    @torch.no_grad()
    def generate(self, images, idx2tok, max_len=20, device=None, start_idx=2, end_idx=3):
        device = device or next(self.parameters()).device
        images = images.to(device)
        features = self.vit(images)                   # (B, N, 768)
        memory = self.decoder.feature_proj(features)  # (B, N, D)

        B = features.size(0)
        word_input = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]

        for t in range(max_len):
            tgt_embeddings = self.decoder.embedding(word_input)
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
def VIT_Transformer(vocab_size, feature_size=768, embedding_dim=512, num_layers=2, fine_tune=True):
    return ImgCap(
        feature_size=feature_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        fine_tune=fine_tune
    )
