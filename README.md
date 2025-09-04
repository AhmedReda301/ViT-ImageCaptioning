# ViT-ImageCaptioning
ImgCap is an image captioning model designed to automatically generate descriptive captions for images. It consists of four model variants:

1- ViT + Transformer
2- CNN + Transformer
3- CNN + LSTM
4- CNN + LSTM with Attention mechanism

Currently, only the ViT + Transformer model has been trained. The other variants have not been trained yet. The ViT model is still undergoing training and requires more time to reach optimal accuracy. Note that training this model is computationally intensive and can take up to three days.

usage

1- Clone the repository:
git clone https://github.com/AhmedReda301/ViT-ImageCaptioning.git

2- Install the required dependencies:
  1-pip3 install -r requirements.txt
  2-python3 -q -m spacy download en_core_web_sm

3- Download the model checkpoint (manual step):


4- Run the main script:
python3 main.py

Sample Output
![Captioned Image]()
![Captioned Image](streamlit/imgs/img3.png)



