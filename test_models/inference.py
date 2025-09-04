# inference.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import matplotlib.pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from data_utils.config import DEVICE, FLICKR_IMAGES_DIR
from data_utils.load_data import get_dataloaders
from train_scripts.train import get_model_by_name, _build_idx2token, _load_or_build_vocab


def load_best_model(model_name, fine_tune=True):
    """
    Load the best saved checkpoint for inference
    """
    # Load vocab
    resume_vocab = _load_or_build_vocab(resume=True)

    # Load test loader
    _, _, test_loader, dataset_vocab, grouped_captions = get_dataloaders(
        resume_vocab=resume_vocab
    )

    vocab_size = len(dataset_vocab) if not hasattr(dataset_vocab, "vocab_size") else dataset_vocab.vocab_size

    # Build model
    model_constructor = get_model_by_name(model_name)
    model = model_constructor(vocab_size=vocab_size, fine_tune=fine_tune).to(DEVICE)

    # Load best checkpoint
    model_dir = os.path.join("results", model_name)
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pkl")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No checkpoint found at {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded best model from {best_model_path}")

    # Build idx->token converter
    idx2tok, _ = _build_idx2token(dataset_vocab)

    return model, test_loader, idx2tok, grouped_captions

def generate_and_compare(model_name="vit_transformer", num_samples=5, images_dir=FLICKR_IMAGES_DIR, save_dir="captioned_images"):
    """
    Generate captions on test set and compare with ground truth, showing ORIGINAL images + texts
    Saves captioned images to `save_dir`.
    """
    # Create folder if not exists
    os.makedirs(save_dir, exist_ok=True)

    model, test_loader, idx2tok, grouped_captions = load_best_model(model_name)

    # Take only first batch from test set
    batch = next(iter(test_loader))
    images = batch["image"].to(DEVICE)
    image_names = batch["image_name"]

    # Generate predictions
    with torch.no_grad():
        generated_captions = model.generate(images, idx2tok=idx2tok)

    # Show results and save images
    for i in range(min(num_samples, len(image_names))):
        img_name = image_names[i]
        gen_caption = generated_captions[i]
        gt_captions = grouped_captions[img_name]

        # Load original image from disk
        img_path = os.path.join(images_dir, img_name)
        orig_img = Image.open(img_path).convert("RGB")

        # Plot original image with space for captions below
        fig, ax = plt.subplots(figsize=(7, 9)) 
        ax.imshow(orig_img)
        ax.axis("off")

        # Starting y position for text (below image)
        y_text = orig_img.size[1] + 10  # 10 pixels below image

        # Add generated caption
        plt.text(
            10, y_text,
            f"Generated: {gen_caption}",
            fontsize=10, color="blue", wrap=True
        )
        y_text += 25  # Space after generated caption

        # Add ground truth captions
        for j, ref in enumerate(gt_captions):
            plt.text(
                10, y_text + j * 18,  # Increase space between each GT caption
                f"Ground Truth {j+1}: {ref}",
                fontsize=9, color="green", wrap=True
            )

        # Save figure
        save_path = os.path.join(save_dir, f"captioned_{img_name}")
        plt.savefig(save_path, bbox_inches="tight")  # Save image with captions
        plt.close(fig)  # Close figure to free memory

        print(f"[INFO] Saved captioned image to {save_path}")



if __name__ == "__main__":
    # Change images_dir to your test images path
    generate_and_compare(model_name="vit_transformer", 
                         num_samples=5, images_dir=FLICKR_IMAGES_DIR)
