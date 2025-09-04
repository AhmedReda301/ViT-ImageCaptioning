# eval_utils/save_metrices.py
import matplotlib.pyplot as plt
import os
import json

def save_training_plots(history, model_name):
    """
    Save updated training curves (loss, perplexity, metrics) for the latest epoch only.
    Overwrites previous plots so we always have the latest state.
    """
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(history.get("train_loss", []), label='Train Loss')
    plt.plot(history.get("val_loss", []), label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss.png"))  # overwrite every time
    plt.close()

    # Perplexity (optional)
    if "train_ppl" in history and "val_ppl" in history:
        plt.figure()
        plt.plot(history.get("train_ppl", []), label='Train PPL')
        plt.plot(history.get("val_ppl", []), label='Val PPL')
        plt.title('Perplexity per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(output_dir, "perplexity.png"))  # overwrite
        plt.close()

    # Metrics (BLEU, METEOR, ROUGE-L, CIDEr)
    if "val_metrics" in history and len(history["val_metrics"]) > 0:
        metrics = history["val_metrics"]
        keys = metrics[0].keys()
        for key in keys:
            plt.figure()
            values = [m.get(key, 0.0) for m in metrics]
            plt.plot(values, marker='o', label=key)
            plt.title(f'{key} per Epoch')
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(output_dir, f"{key}.png"))  # overwrite
            plt.close()

def save_history_json(history, model_name):
    """
    Save or update history.json after each epoch.
    Always contains all epochs up to current.
    """
    output_dir = os.path.join("results", model_name)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "history.json")
    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)
