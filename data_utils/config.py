# data_utils/config.py
"""
Configuration file for Image Captioning project.

Defines device, data paths, training parameters,
and reproducibility settings.
"""

import os
from pathlib import Path
import random
import numpy as np
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Raw data directories & files
PROJECT_DIR = Path(os.getenv("IMG_CAP_PROJECT_DIR", r"D:\Projects\DL\Image-Captioning"))
FLICKR_IMAGES_DIR = PROJECT_DIR / "Data" / "flickr30k" / "flickr30k_images" / "flickr30k_images"
FLICKR_RESULTS_FILE = PROJECT_DIR / "Data" / "flickr30k" / "results.csv"

# Sampled data directories & files
SAMPLED_IMAGES_DIR = PROJECT_DIR / "Data" / "sampled" / "sampled_images"
SAMPLED_RESULTS_FILE = PROJECT_DIR / "Data" / "sampled" / "sampled_results.csv"

# Preprocessed data file (saved .pt tensors)
PREPROCESSED_PATH = PROJECT_DIR / "Data" / "preprocessed" / "flickr_preprocessed.pt"

# Training parameters
BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 20
SEED = 42

# Scheduler parameters
SCHEDULER_STEP_SIZE = 5   
SCHEDULER_GAMMA = 0.8  

# Model paths
MODEL_DIR = PROJECT_DIR / "results"
CNN_LSTM_ATTENTION_BEST = MODEL_DIR / "cnn_lstm_attention_best.pth"
VOCAB_PATH = PROJECT_DIR / "vocab" / "vocab.pkl"

def set_seed(seed: int = SEED):
    """Fix random seeds for reproducibility across Python, NumPy and PyTorch."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour for CUDA (may slow down)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # older PyTorch versions
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


