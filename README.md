# <span style="color:#FFA500;"> ViT-ImageCaptioning</span>

A <span style="color:#00BFFF;">deep learning</span> project for automatically generating **descriptive captions** for images.  
It includes full <span style="color:#32CD32;">data preprocessing</span>, <span style="color:#32CD32;">model training</span>, and <span style="color:#32CD32;">evaluation</span> using **four model variants**:

  1. **ViT + Transformer**  
  2. **CNN + Transformer**  
  3. **CNN + LSTM**  
  4. **CNN + LSTM with Attention mechanism**  

> **Note:** Currently, only the **ViT + Transformer** model has been trained. The other variants have not been trained yet. The ViT model is still undergoing training and requires more time to reach optimal accuracy. Note that training this model is computationally intensive and can take long time.


## <span style="color:#1E90FF;"> Usage</span>

1. **Clone the repository:**
  ```bash
  git clone https://github.com/AhmedReda301/ViT-ImageCaptioning.git
  ```
2. **Install the required dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
  ```bash
  python -q -m spacy download en_core_web_sm
  ```
3. **Download the model checkpoint (manual step):**  
   - ImgCap (ViT + Transformer): [Download checkpoint](https://www.kaggle.com/models/ahmedredaahmedali/vittransformer)

4. **Run the main script:**
  ```bash
  python main.py
  ```

### <span style="color:#1E90FF;"> Sample Output:</span>

![Captioned Image](streamlit/imgs/img3.png)

## <span style="color:#1E90FF;"> Dataset:</span>

The **Flickr30k** dataset consists of 31,784 images, each accompanied by five captions.  
It provides a wide variety of scenes and objects, making it ideal for diverse image captioning tasks.

- **To download the dataset:**  
  - Enable Kaggle’s public API by following the instructions [here](https://www.kaggle.com/docs/api).  

### **Option 1: Full Dataset (100%)**
Run the following command to download the full dataset **or** click [Download Dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset):
```bash
kaggle datasets download -d hsankesara/flickr-image-dataset -p /path/to/data/Flickr30
```

### **Option2 Sample Dataset (~2% of total):**  
Run the following command to download the sample dataset **or** click [Download Sample Dataset](https://www.kaggle.com/datasets/ahmedredaahmedali/flickr30k-dataset-2-sample):
```bash
kaggle datasets download -d ahmedredaahmedali/flickr30k-dataset-2-sample -p /path/to/data/Flickr30_sample
```
I have created a small sample of approximately 2% of the full dataset.  
This allows you to **test your pipeline** and verify that everything works correctly before training your model on the full dataset.  


## <span style="color:#32CD32;"> Model Architecture:</span>

The model used is **ImgCap (ViT + Transformer)** for image captioning. It consists of two main components: a **Vision Transformer (ViT) encoder** and a **Transformer-based decoder**.
total parametets: 

---

### **ViT Encoder**

- **Backbone:** Pretrained **ViT-B/16** from `torchvision.models`
- **Input:** Image tensor `(B, 3, H, W)`  
- **Patch Embedding:** Images divided into patches and projected to embedding dimension (768)  
- **Class Token:** Added to the patch embeddings for global representation  
- **Positional Encoding:** Added to patches to retain positional information  
- **Transformer Encoder Layers:** Processes the sequence of patches + class token  
- **Output:** Patch feature sequence `(B, num_patches, 768)`  
- **Fine-tuning Strategy:**  
  - All ViT parameters frozen by default  
  - Optionally, last encoder layers can be unfrozen for fine-tuning

---

### **Transformer Decoder**

- **Embedding Layer:** Converts token indices to embeddings  
- **Positional Encoding:** Added to token embeddings to retain order  
- **Feature Projection:** Projects ViT features (768) to decoder embedding dimension (default 512)  
- **Transformer Decoder Layers:** Multi-layer Transformer decoder attends to ViT features  
- **Output Layer:** Linear layer projecting decoder output to vocabulary size  
- **Sequence Generation:**  
  - Uses teacher forcing during training  
  - For inference, generates tokens step by step using predicted tokens

---

### **ImgCap (ViT + Transformer) Wrapper**

- Combines **ViT Encoder** and **Transformer Decoder**  
- Forward pass:
  1. Images → ViT encoder → Patch features  
  2. Patch features + Captions → Transformer decoder → Predicted token logits  
- Caption Generation: Autoregressive generation using `<sos>` and `<eos>` tokens  
- **Key Hyperparameters:**  
  - Feature size: 768  
  - Embedding dimension: 512  
  - Number of decoder layers: 2  
  - Max sequence length: 20  

---

### **Key Notes**

- The model leverages **pretrained ViT features** to capture rich visual representations.  
- Decoder is fully trainable to adapt to the captioning task.  
- Positional encodings are used in both encoder and decoder to maintain sequence order.  
- Optimized for **Float16 precision** and `torch.compile` for efficient GPU training.







## <span style="color:#1E90FF;"> Model Evaluation:</span>

| Model                    | Epochs | BLEU-1  | BLEU-2   | BLEU-4 | METEOR | ROUGE-L | CIDEr   |
|--------------------------|--------|---------|----------|--------|--------|---------|---------|
| ImgCap ViT + Transformer | 18     | 0.6858  |0.5282    | 0.2982 | 0.4979 | 0.5128  | 0.6596  |

**Note:** The models are still undertrained. With further training and more epochs, the BLEU and CIDEr scores are expected to improve.








