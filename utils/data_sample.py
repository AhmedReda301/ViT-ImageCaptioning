# utils/data_sample.py
import os
import sys

# add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import pandas as pd
from data_utils.config import (
    FLICKR_IMAGES_DIR,
    FLICKR_RESULTS_FILE,
    SAMPLED_IMAGES_DIR,
    SAMPLED_RESULTS_FILE,
    SEED
)

def sample_data(
    frac: float = 0.05,   
    random_state: int = SEED
):
    """
    Create a smaller sampled dataset from Flickr30K.
    
    Args:
        frac (float): Fraction of dataset to sample.
        random_state (int): Random seed for reproducibility.
    """

    # 1. Read the full CSV
    df = pd.read_csv(FLICKR_RESULTS_FILE, delimiter="|")
    df.columns = df.columns.str.strip()  # remove extra spaces

    # Drop NaN captions
    df = df.dropna(subset=["comment"])

    # 2. Sample the data
    df_sample = df.sample(frac=frac, random_state=random_state)

    # Make sure output directories exist
    os.makedirs(SAMPLED_IMAGES_DIR, exist_ok=True)

    # 3. Copy the sampled images
    copied_images = set()
    for _, row in df_sample.iterrows():
        image_name = row["image_name"]
        src_path = os.path.join(FLICKR_IMAGES_DIR, image_name)
        dst_path = os.path.join(SAMPLED_IMAGES_DIR, image_name)

        if not os.path.exists(src_path):
            print(f"Warning: Image {src_path} not found, skipping.")
            continue

        if image_name not in copied_images:  # avoid copying duplicates
            shutil.copy(src_path, dst_path)
            copied_images.add(image_name)

    # 4. Save the new CSV
    df_sample.to_csv(SAMPLED_RESULTS_FILE, index=False, sep="|")

    print(f"Sampled dataset created:")
    print(f"- CSV saved at: {SAMPLED_RESULTS_FILE}")
    print(f"- {len(copied_images)} images copied to {SAMPLED_IMAGES_DIR}")

if __name__ == "__main__":
    # Example: take 2% sample
    sample_data(frac=0.02)
