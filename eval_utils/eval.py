# eval_utils/eval.py
import logging
import torch
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from rouge import Rouge

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helpers
def normalize_text(s: str) -> str:
    """Lowercase + remove punctuation for fairer eval."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def _to_token_lists(references, predictions):
    """
    Convert lists of detokenized strings into token lists suitable for BLEU/METEOR.
    references: list[list[str]]
    predictions: list[str]
    """
    ref_tokens_list = [[normalize_text(r).split() for r in refs] for refs in references]
    pred_tokens_list = [normalize_text(p).split() for p in predictions]
    return ref_tokens_list, pred_tokens_list


# Metrics
def compute_bleu(references, predictions):
    """
    Compute BLEU-1, BLEU-2, and BLEU-4 scores for image captioning predictions.

    Args:
        references (list[list[str]]): List of reference captions for each image.
        predictions (list[str]): List of predicted captions.

    Returns:
        dict: Dictionary with average BLEU-1, BLEU-2, and BLEU-4 scores.
    """
    smooth_fn = SmoothingFunction().method4
    ref_tokens_list, pred_tokens_list = _to_token_lists(references, predictions)

    bleu1_scores, bleu2_scores, bleu4_scores = [], [], []
    for refs_tok, pred_tok in zip(ref_tokens_list, pred_tokens_list):
        bleu1 = sentence_bleu(refs_tok, pred_tok, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
        bleu2 = sentence_bleu(refs_tok, pred_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
        bleu4 = sentence_bleu(refs_tok, pred_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu4_scores.append(bleu4)

    return {
        "BLEU-1": sum(bleu1_scores) / max(1, len(bleu1_scores)),
        "BLEU-2": sum(bleu2_scores) / max(1, len(bleu2_scores)),
        "BLEU-4": sum(bleu4_scores) / max(1, len(bleu4_scores)),
    }


def compute_meteor(references, predictions):
    """
    Compute METEOR scores for image captioning predictions.

    Args:
        references (list[list[str]]): List of reference captions for each image.
        predictions (list[str]): List of predicted captions.

    Returns:
        dict: Dictionary with average METEOR scores.
    """
    ref_tokens_list, pred_tokens_list = _to_token_lists(references, predictions)
    scores = [
        meteor_score(refs_tok, pred_tok)
        for refs_tok, pred_tok in zip(ref_tokens_list, pred_tokens_list)
    ]
    return {"METEOR": sum(scores) / max(1, len(scores))}


def compute_rouge(references, predictions):
    """
    Compute the average ROUGE-L score for image captioning predictions.

    Args:
        references (list[list[str]]): List of reference captions for each image.
        predictions (list[str]): List of predicted captions.

    Returns:
        dict: Dictionary with average ROUGE-L score.
    """
    rouge = Rouge()
    scores = []
    for refs, pred in zip(references, predictions):
        try:
            best = max(
                (rouge.get_scores(normalize_text(pred), normalize_text(r))[0]['rouge-l']['f'] for r in refs),
                default=0.0
            )
        except Exception as e:
            logger.warning(f"ROUGE error: {e}")
            best = 0.0
        scores.append(best)
    return {"ROUGE-L": sum(scores) / max(1, len(scores))}


def compute_cider(references, predictions):
    gts = {i: refs for i, refs in enumerate(references)}   # refs is already list[str]
    res = {i: [pred] for i, pred in enumerate(predictions)}

    cider_scorer = Cider()
    try:
        score, _ = cider_scorer.compute_score(gts, res)
    except Exception as e:
        logger.warning(f"CIDEr error: {e}")
        score = 0.0
    return {"CIDEr": score}

def build_eval_lists(dataloader, model, dataset, device):
    """
    Compute the average CIDEr score for image captioning predictions.

    Args:
        references (list[list[str]]): List of reference captions for each image.
        predictions (list[str]): List of predicted captions.

    Returns:
        dict: Dictionary with average CIDEr score.
    """
    model.eval()
    references, predictions = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            image_names = batch["image_name"]

            # Ground truth: grouped_captions (list of strings per image)
            for name in image_names:
                refs = dataset.grouped_captions[name]
                references.append(refs)

            # Model predictions (assumes you have caption_image or generate function)
            outputs = []
            for img in images:
                pred = model.generate(img.unsqueeze(0))  # or caption_image(...)
                outputs.append(pred)
            predictions.extend(outputs)

    assert len(references) == len(predictions), f"Refs={len(references)}, Preds={len(predictions)}"
    return references, predictions


def evaluate_all(references, predictions):
    """
    Evaluate image captioning predictions using BLEU, METEOR, ROUGE-L, and CIDEr metrics.

    Args:
        references (list[list[str]]): List of reference captions for each image.
        predictions (list[str]): List of predicted captions.

    Returns:
        dict: Dictionary with all metric scores.
    """
    results = {}
    results.update(compute_bleu(references, predictions))
    results.update(compute_meteor(references, predictions))
    results.update(compute_rouge(references, predictions))
    results.update(compute_cider(references, predictions))
    return results
