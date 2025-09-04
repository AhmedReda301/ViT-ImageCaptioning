# train_scripts/main.py
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_scripts.train import train  

if __name__ == "__main__":
    print("Starting training...")

    model_name = "cnn_lstm_attention"   # "cnn_lstm" or "cnn_lstm_attention"
    resume = False                       # True = resume from last checkpoint, False = start fresh
    resume_vocab = False                  # True = resume vocabulary, False = build new vocabulary

    history = train(model_name=model_name, resume=resume, resume_vocab=resume_vocab)

    print("Training completed.")

