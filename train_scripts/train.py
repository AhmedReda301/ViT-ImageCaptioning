# train_scripts/train.py
import os
from math import exp
from tqdm import tqdm  # For progress bars
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  
import pickle  

# Make CuBLAS deterministic for reproducibility (helps with consistent GPU results)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# Evaluation utilities for computing metrics like BLEU, ROUGE, etc.
from eval_utils.eval import evaluate_all
from eval_utils.save_metrices import save_training_plots, save_history_json

# Project configuration and hyperparameters
from data_utils.config import (
    DEVICE, EPOCHS, LR, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA,
    set_seed, SEED, VOCAB_PATH
)
# Data loading utilities
from data_utils.load_data import get_dataloaders
# Model imports
from models.cnn_lstm import CNN_LSTM
from models.cnn_lstm_attention import CNN_LSTM_Attention
from models.cnn_transformer import CNN_Transformer
from models.vit_transformer import VIT_Transformer

def get_model_by_name(model_name):
    """
    Map string name to model class
    """
    name = model_name.lower()
    if name == "cnn_lstm":
        return CNN_LSTM
    elif name == "cnn_lstm_attention":
        return CNN_LSTM_Attention
    elif name == "cnn_transformer":
        return CNN_Transformer
    elif name == "vit_transformer":
        return VIT_Transformer
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _build_idx2token(vocab):
    """
    Construct a function that maps token index -> token string.
    Also return padding index for loss masking.
    """
    if hasattr(vocab, "lookup_token"):  # Vocabulary object (like torchtext Vocab)
        def idx2tok(i): return vocab.lookup_token(i)
        pad_idx = vocab["<pad>"] if "<pad>" in vocab else None
    elif isinstance(vocab, dict):  # Plain dict {token: idx}
        inv = {v: k for k, v in vocab.items()}
        def idx2tok(i): return inv.get(i, "<unk>")
        pad_idx = vocab.get("<pad>", None)
    elif isinstance(vocab, list):  # List of tokens
        def idx2tok(i): return vocab[i] if 0 <= i < len(vocab) else "<unk>"
        pad_idx = vocab.index("<pad>") if "<pad>" in vocab else None
    else:
        raise TypeError("Unsupported vocab type")
    return idx2tok, pad_idx


def _load_or_build_vocab(resume):
    """
    Load saved vocabulary if resuming, otherwise return None to build new vocab.
    """
    resume_vocab = None
    if resume:
        if os.path.exists(VOCAB_PATH):
            with open(VOCAB_PATH, "rb") as f:
                resume_vocab = pickle.load(f)
            print(f"[INFO] Vocabulary loaded from {VOCAB_PATH}")
        else:
            print("[WARNING] Resume=True but no saved vocab found. A new vocab will be built.")
    return resume_vocab


def train(model_name, fine_tune=True, resume=True, resume_vocab=True):
    """
    Main training function
    """
    # Set random seed for reproducibility
    set_seed(SEED)

    # Load or build vocab
    resume_vocab = _load_or_build_vocab(resume_vocab)

    # Load dataloaders, vocab, and grouped captions
    train_loader, val_loader, test_loader, dataset_vocab, grouped_captions = get_dataloaders(
        resume_vocab=resume_vocab
    )
    vocab_size = len(dataset_vocab) if not hasattr(dataset_vocab, "vocab_size") else dataset_vocab.vocab_size

    # Instantiate model and send to device (GPU/CPU)
    model_constructor = get_model_by_name(model_name)
    model = model_constructor(vocab_size=vocab_size, fine_tune=fine_tune).to(DEVICE)

    # Build token index <-> string mappings
    idx2tok, pad_idx = _build_idx2token(dataset_vocab)

    # Define loss (CrossEntropy) ignoring padding tokens
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # Define LR scheduler
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # History dictionary to track losses, perplexities, and evaluation metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_ppl': [], 'val_ppl': [], 'val_metrics': []
    }

    best_val_loss = float("inf")
    model_dir = os.path.join("results", model_name)
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pkl")

    # Resume checkpoint if exists
    if resume and os.path.exists(best_model_path):
        print(f"Resuming weights from checkpoint: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint.get('history', history)
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        print("Weights, optimizer, scheduler loaded. Training will start from epoch 1.")
    else:
        print("Starting training from scratch. Random weights will be used.")

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss_sum = 0.0
        print(f"\nEpoch [{epoch}/{EPOCHS}]")

        # Training phase
        for batch in tqdm(train_loader, desc="Training", leave=False):
            images = batch['image'].to(DEVICE)
            captions = batch['caption'].to(DEVICE)

            # Forward pass
            logits = model(images, captions)
            seq_len = captions[:, 1:].size(1)
            logits = logits[:, :seq_len, :]

            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))

            # Backprop and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / max(1, len(train_loader))

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_refs = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch['image'].to(DEVICE)
                captions = batch['caption'].to(DEVICE)
                image_names = batch['image_name']

                # Forward pass for loss
                logits = model(images, captions)
                seq_len = captions[:, 1:].size(1)
                logits = logits[:, :seq_len, :]

                loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
                val_loss_sum += loss.item()

                # Generate captions for evaluation
                generated_texts = model.generate(images, idx2tok=idx2tok)  # list of strings

                # Append predictions
                all_preds.extend(generated_texts)

                # Append references properly (each element should be list of strings)
                for img_name in image_names:
                    all_refs.append(grouped_captions[img_name])  # list of strings per image

        # Compute average validation loss
        val_loss = val_loss_sum / max(1, len(val_loader))

        # Evaluate metrics
        val_metrics = evaluate_all(all_refs, all_preds)


        # Compute perplexity
        train_ppl = exp(train_loss) if train_loss < 20 else float('inf')
        val_ppl = exp(val_loss) if val_loss < 20 else float('inf')

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)
        history['val_metrics'].append(val_metrics)

        # Print epoch summary
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items() if isinstance(v, (int, float))])
        print(f"Train Loss: {train_loss:.4f} (PPL {train_ppl:.2f}) | "
              f"Val Loss: {val_loss:.4f} (PPL {val_ppl:.2f}) | {metrics_str}")

        # Step the LR scheduler
        scheduler.step()
        print(f"Current LR: {scheduler.get_last_lr()}")

        # Save plots and history json
        save_training_plots(history, model_name)
        save_history_json(history, model_name)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history
            }, best_model_path)
            print(f"Saved new best model as {best_model_path} with val_loss={val_loss:.4f}")
