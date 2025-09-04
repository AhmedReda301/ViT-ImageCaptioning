# caption_single.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from data_utils.config import DEVICE
from data_utils.load_data import get_dataloaders
from train_scripts.train import get_model_by_name, _build_idx2token, _load_or_build_vocab


def load_best_model(model_name, fine_tune=True):
    """
    Load best trained checkpoint
    """
    resume_vocab = _load_or_build_vocab(resume=True)

    # we only need vocab, no test loader here
    _, _, _, dataset_vocab, _ = get_dataloaders(resume_vocab=resume_vocab)

    vocab_size = len(dataset_vocab) if not hasattr(dataset_vocab, "vocab_size") else dataset_vocab.vocab_size

    # build model
    model_constructor = get_model_by_name(model_name)
    model = model_constructor(vocab_size=vocab_size, fine_tune=fine_tune).to(DEVICE)

    # load checkpoint
    model_dir = os.path.join("results", model_name)
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pkl")

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No checkpoint found at {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # build idx2tok
    idx2tok, _ = _build_idx2token(dataset_vocab)

    print(f"[INFO] Loaded best model from {best_model_path}")
    return model, idx2tok


def preprocess_image(img_path, image_size=224):
    """
    Preprocess image into tensor
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    return img, img_tensor.to(DEVICE)


def generate_caption_for_image(model_name, img_path, save_path=None):
    """
    Generate caption for one custom image
    """
    # load model
    model, idx2tok = load_best_model(model_name)

    # preprocess image
    orig_img, img_tensor = preprocess_image(img_path)

    # generate caption
    with torch.no_grad():
        generated_caption = model.generate(img_tensor, idx2tok=idx2tok)[0]

    # plot image with caption
    plt.figure(figsize=(7, 7))
    plt.imshow(orig_img)
    plt.axis("off")
    plt.title(f"Generated: {generated_caption}", color="blue", fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[INFO] Saved captioned image to {save_path}")
    else:
        plt.show()

    return generated_caption


if __name__ == "__main__":
    # example usage
    custom_image = r"D:\Lionel-Messi-Argentina-Netherlands-World-Cup-Qatar-2022.webp"
    caption = generate_caption_for_image("vit_transformer", custom_image, save_path=r"D:\Projects\DL\Image-Captioning\captioned_test.jpg")
    print("[RESULT]", caption)

