# data_utils/Flickr30.py
import cv2
import os
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter, defaultdict
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from .config import VOCAB_PATH

def tokenize_caption(caption, nlp):
    """ Tokenize + lowercase + remove punctuation """
    if not isinstance(caption, str):
        caption = ""
    return [t.text.lower() for t in nlp(caption) if not t.is_punct]

def build_vocab(captions, nlp, min_freq=3):
    """
    Builds a vocabulary from a list of captions.

    Args:
        captions (list of str): List of caption strings.
        nlp: Tokenizer object (e.g., spaCy model) for tokenizing captions.
        min_freq (int, optional): Minimum frequency for a token to be included in the vocabulary. Defaults to 3.

    Returns:
        vocab (list): List of vocabulary tokens including special tokens.
        stoi (dict): Mapping from token (str) to index (int).
        itos (dict): Mapping from index (int) to token (str).
    """
    tokenized = [tokenize_caption(c, nlp) for c in captions]
    all_tokens = [tok for cap in tokenized for tok in cap]

    counter = Counter(all_tokens)
    special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]
    vocab = special_tokens + [tok for tok, count in counter.items() if count >= min_freq]

    stoi = {word: idx for idx, word in enumerate(vocab)}
    itos = {idx: word for idx, word in enumerate(vocab)}

    return vocab, stoi, itos


def save_vocab(vocab_data, vocab_path):
    """
    Saves the vocabulary data to a file using pickle.

    Args:
        vocab_data (dict): Dictionary containing 'vocab', 'stoi', and 'itos'.
        vocab_path (str): Path to save the vocabulary file.
    """
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"[INFO] Vocabulary saved at {vocab_path}")


def encode_caption(tokens, stoi):
    """
    Encodes a list of tokens into a tensor of indices using the vocabulary mapping.

    Args:
        tokens (list of str): List of caption tokens.
        stoi (dict): Mapping from token (str) to index (int).

    Returns:
        torch.Tensor: Tensor of token indices including <sos> and <eos> tokens.
    """
    indices = [stoi["<sos>"]]
    indices.extend(stoi.get(t, stoi["<unk>"]) for t in tokens)
    indices.append(stoi["<eos>"])
    return torch.tensor(indices, dtype=torch.long)


def decode_caption(indices, itos):
    """
    Decodes a tensor of indices back into a caption string.

    Args:
        indices (torch.Tensor or list): Sequence of token indices.
        itos (dict): Mapping from index (int) to token (str).

    Returns:
        str: Decoded caption string (without special tokens).
    """
    tokens = [itos.get(idx.item(), "<unk>") for idx in indices]
    words = [t for t in tokens if t not in ("<sos>", "<eos>", "<pad>")]
    return " ".join(words)


class Flickr30(Dataset):
    def __init__(self, images_folder_path, labels_path,
                 vocab=None, vocab_path=VOCAB_PATH,
                 transform=None, nlp=None, resume_vocab=None):

        self.images_folder_path = images_folder_path
        self.labels = pd.read_csv(labels_path, delimiter='|')
        self.labels.columns = self.labels.columns.str.strip()
        self.labels = self.labels.dropna(subset=['comment'])
        self.transform = transform

        self.nlp = nlp 

        # Vocabulary handling 
        if resume_vocab is not None:
            self.vocab = resume_vocab['vocab']
            self.stoi = resume_vocab['stoi']
            self.itos = resume_vocab['itos']
        elif vocab is None:
            self.vocab, self.stoi, self.itos = build_vocab(self.labels['comment'].tolist(), self.nlp)
            save_vocab({'vocab': self.vocab, 'stoi': self.stoi, 'itos': self.itos}, vocab_path)
        else:
            self.vocab = vocab
            self.stoi = {word: idx for idx, word in enumerate(vocab)}
            self.itos = {idx: word for idx, word in enumerate(vocab)}

        # Pre-tokenize once
        self.tokenized_captions = [tokenize_caption(c, self.nlp) for c in self.labels['comment'].tolist()]

        # Group captions per image
        self.grouped_captions = defaultdict(list)
        for _, row in self.labels.iterrows():
            self.grouped_captions[row['image_name']].append(row['comment'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.labels.iloc[idx]['image_name']
        caption_tokens = self.tokenized_captions[idx]

        # Load + convert image
        image_path = os.path.join(self.images_folder_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        tensor_caption = encode_caption(caption_tokens, self.stoi)

        return image, tensor_caption, image_name


def collate_fn(batch):
    """ Custom collate function for DataLoader """
    images, captions, image_names = zip(*batch)
    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)  # <pad> index = 0

    return {"image": images, "caption": captions, "image_name": image_names}

