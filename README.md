<span style="color:#FFA500;"> ViT-ImageCaptioning</span>

ImgCap is an image captioning model designed to automatically generate descriptive captions for images.
It includes four model variants:

ViT + Transformer

CNN + Transformer

CNN + LSTM

CNN + LSTM with Attention mechanism

Note: Only the ViT + Transformer model has been trained so far. Training is computationally intensive and may take up to 3 days on a high-end GPU.



<span style="color:#32CD32;"> Key Highlights:</span>

Modeling: ViT or CNN encoders combined with Transformer or LSTM decoders.

Attention Mechanism: Optional in CNN + LSTM variant.

Evaluation Metrics: BLEU-1/2/3/4, CIDEr.

Training Optimizations: Mixed precision (Float16) and torch.compile for efficient GPU usage.


<span style="color:#32CD32;"> Dataset:</span>

Source: Flickr30k Dataset - Kaggle

Description: 31,784 images, each with 5 captions

Purpose: Diverse image captioning tasks

Download via Kaggle API:

kaggle datasets download -d hsankesara/flickr-image-dataset -p /path/to/data/Flickr30
