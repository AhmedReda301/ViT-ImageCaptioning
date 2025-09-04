# data_utils/verify_data.py
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from config import FLICKR_IMAGES_DIR, FLICKR_RESULTS_FILE


def check_images_path():
    """Check if the Flickr images directory exists and list sample files."""
    if FLICKR_IMAGES_DIR.exists():
        images = [f for f in os.listdir(FLICKR_IMAGES_DIR) if f.lower().endswith(".jpg")]
        print(f"Path exists: {FLICKR_IMAGES_DIR}")
        print(f"Found {len(images)} images")
        print("First 5 images:", images[:5])
        return images
    else:
        raise FileNotFoundError(f"Path does NOT exist: {FLICKR_IMAGES_DIR}")


def load_results_file():
    """Load results.csv safely and return DataFrame."""
    if not FLICKR_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {FLICKR_RESULTS_FILE}")

    # Some CSVs may have extra spaces around column names -> strip them
    df = pd.read_csv(FLICKR_RESULTS_FILE, delimiter="|")
    df.columns = df.columns.str.strip()
    print(f"Results file loaded: {FLICKR_RESULTS_FILE}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def display_sample_image(df, index=0):
    """Display an image with its caption from the DataFrame."""
    if index >= len(df):
        raise IndexError(f"Index {index} out of range (DataFrame length {len(df)})")

    row = df.iloc[index]
    img_name = row["image_name"]
    caption = row[df.columns[-1]]  # last column = comment/caption

    img_path = FLICKR_IMAGES_DIR / img_name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.title(caption, fontsize=10)
    plt.axis("off")
    plt.show()
    print(f"Displayed: {img_name} | Caption: {caption}")


if __name__ == "__main__":
    images = check_images_path()
    df = load_results_file()
    display_sample_image(df, index=6)  
