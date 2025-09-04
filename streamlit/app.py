# streamlit/app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st
import matplotlib.pyplot as plt
from data_utils.config import DEVICE
from data_utils.load_data import get_dataloaders
from train_scripts.train import get_model_by_name, _build_idx2token, _load_or_build_vocab


# Model Utilities
@st.cache_resource
def load_best_model(model_name="vit_transformer", fine_tune=True):
    """Load best trained checkpoint"""
    resume_vocab = _load_or_build_vocab(resume=True)
    _, _, _, dataset_vocab, _ = get_dataloaders(resume_vocab=resume_vocab)

    vocab_size = len(dataset_vocab) if not hasattr(dataset_vocab, "vocab_size") else dataset_vocab.vocab_size

    model_constructor = get_model_by_name(model_name)
    model = model_constructor(vocab_size=vocab_size, fine_tune=fine_tune).to(DEVICE)

    model_dir = os.path.join("results", model_name)
    best_model_path = os.path.join(model_dir, f"{model_name}_best.pkl")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    idx2tok, _ = _build_idx2token(dataset_vocab)
    return model, idx2tok


def preprocess_image(uploaded_file, image_size=224):
    """Preprocess image into tensor"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor.to(DEVICE)


def generate_caption(model, idx2tok, img_tensor):
    """Generate caption for one image"""
    with torch.no_grad():
        caption = model.generate(img_tensor, idx2tok=idx2tok)[0]
    return caption


# Streamlit App
def main():
    st.set_page_config(page_title="Image Captioning App", layout="centered")

    st.title("Image Captioning App")
    st.write("Upload an image and let the trained model generate a caption.")

    # Sidebar
    st.sidebar.header("Model Options")
    model, idx2tok = load_best_model("vit_transformer")

    with st.sidebar.expander("Model Details"):
        num_params = sum(p.numel() for p in model.parameters())
        st.write(f"Parameters: {num_params:,}")
        st.write("Checkpoint: results/vit_transformer/vit_transformer_best.pkl")

    # File uploader
    uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Preprocess
            orig_img, img_tensor = preprocess_image(uploaded_file)

            # Generate caption
            caption = generate_caption(model, idx2tok, img_tensor)

            # Display image
            st.image(orig_img, width=400)

            # Styled caption
            st.markdown(
                f"""
                <p style='text-align: center; font-size:21px;  color:#ADD8E6;'>
                    Generated Caption: <b>{caption}</b>
                </p>
                """,
                unsafe_allow_html=True
            )

            # Save image with caption overlay
            output_path = f"captioned_{uploaded_file.name}"
            plt.figure(figsize=(7, 7))
            plt.imshow(orig_img)
            plt.axis("off")
            plt.title(f"Generated Caption: {caption}", color="blue", fontsize=12)
            plt.savefig(output_path, format="jpeg", bbox_inches="tight", dpi=300)
            plt.close()

            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Captioned Image",
                    data=f,
                    file_name=f"captioned_{uploaded_file.name}",
                    mime="image/jpeg"
                )


if __name__ == "__main__":
    main()
