# data_utils/load_data.py
from torch.utils.data import DataLoader, random_split
import torch
import torchvision.transforms as transforms
import numpy as np
import spacy

from .config import (
    SAMPLED_IMAGES_DIR,
    SAMPLED_RESULTS_FILE,
    BATCH_SIZE,
    SEED,
    set_seed
)

from .Flickr30 import Flickr30, collate_fn 

# Load the NLP tokenizer 
nlp = spacy.load("en_core_web_sm") 

# Fix randomness for reproducibility
set_seed(SEED)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
    ], p=0.4),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def worker_init_fn(worker_id):
    """
    Initializes the random seed for each data loader worker for reproducibility.

    Args:
        worker_id (int): Worker process ID.
    """
    np.random.seed(SEED + worker_id)

def get_dataloaders(val_split: float = 0.15, test_split: float = 0.15, 
                    num_workers: int = 4, resume_vocab: dict = None):
    """
    resume_vocab: optional dict loaded from checkpoint to use exact same vocab
    """
    full_dataset = Flickr30(
        SAMPLED_IMAGES_DIR,
        SAMPLED_RESULTS_FILE,
        transform=transform,
        resume_vocab=resume_vocab,  # pass saved vocab when resuming
        nlp=nlp
    )

    dataset_vocab = full_dataset.vocab

    test_size = int(len(full_dataset) * test_split)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size - test_size

    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=num_workers, 
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader, test_loader, dataset_vocab, full_dataset.grouped_captions


