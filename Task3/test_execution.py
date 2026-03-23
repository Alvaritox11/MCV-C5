# test_execution.py

import os
import json
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
from tqdm import tqdm
import evaluate

from models import BaselineModel
from dataset import get_dataloaders

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

device = "cuda" if torch.cuda.is_available() else "cpu"


def safe_decode(tokenizer, token_ids):
    """
    Decode token ids into text.
    """
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids).strip()

    cleaned = []
    for tid in token_ids:
        tid = int(tid)
        if tid == getattr(tokenizer, "eos_id", None):
            break
        if tid in {
            getattr(tokenizer, "pad_id", -999999),
            getattr(tokenizer, "sos_id", -999999),
        }:
            continue
        cleaned.append(tid)

    if hasattr(tokenizer, "idx2token"):
        toks = [tokenizer.idx2token[i] for i in cleaned]
        if getattr(tokenizer, "text_level", "word") == "char":
            return "".join(toks).strip()
        return " ".join(toks).strip()

    return " ".join(map(str, cleaned)).strip()


def load_checkpoint_cfg(checkpoint):
    if "cfg" in checkpoint:
        return checkpoint["cfg"]
    if "config" in checkpoint:
        return checkpoint["config"]

    raise ValueError(
        "The checkpoint does not contain 'cfg' or 'config'. "
        "Your training code appears to save 'config', so make sure the checkpoint is valid."
    )


def load_model_with_resize(model, state_dict):
    model_state = model.state_dict()

    for name, param in state_dict.items():
        if name in model_state and param.shape != model_state[name].shape:
            print(f"⚠️ Auto-resizing {name} from {param.shape} to {model_state[name].shape}")
            new_param = model_state[name].clone()

            slices = tuple(
                slice(0, min(d1, d2)) for d1, d2 in zip(param.shape, new_param.shape)
            )
            new_param[slices] = param[slices]
            state_dict[name] = new_param

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")


@torch.no_grad()
def evaluate_model(model, crit, dataloader, tokenizer, cfg):
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []
    predictions_to_save = []

    for img_ids, imgs, captions, all_raw_captions in tqdm(dataloader, desc="Evaluating"):
        imgs, captions = imgs.to(device), captions.to(device)

        # ---- LOSS: same logic as train.py ----
        if cfg.get("teacher_forcing", False):
            loss_output = model(imgs, captions)
        else:
            loss_output = model(imgs)

        loss = crit(loss_output, captions[:, 1:])
        total_loss += loss.item()

        # ---- METRICS: always autoregressive generation ----
        gen_output = model(imgs)
        pred_ids = torch.argmax(gen_output, dim=1)

        refs_for_batch = []
        for b in range(imgs.size(0)):
            refs_for_batch.append(
                [all_raw_captions[j][b] for j in range(len(all_raw_captions))]
            )

        batch_preds = []
        for i in range(pred_ids.shape[0]):
            pred_str = safe_decode(tokenizer, pred_ids[i].tolist())

            batch_preds.append(pred_str)
            all_preds.append(pred_str)
            all_refs.append(refs_for_batch[i])

            current_id = img_ids[i].item() if torch.is_tensor(img_ids[i]) else img_ids[i]
            predictions_to_save.append({
                "image_id": int(current_id),
                "prediction": pred_str,
                "references": refs_for_batch[i]
            })

    avg_loss = total_loss / len(dataloader)

    bleu1_score = bleu.compute(
        predictions=all_preds,
        references=all_refs,
        max_order=1
    )["bleu"]

    bleu2_score = bleu.compute(
        predictions=all_preds,
        references=all_refs,
        max_order=2
    )["bleu"]

    rouge_score = rouge.compute(
        predictions=all_preds,
        references=all_refs
    )["rougeL"]

    meteor_score = meteor.compute(
        predictions=all_preds,
        references=all_refs
    )["meteor"]

    metrics = {
        "Valid Loss": avg_loss,
        "BLEU-1": bleu1_score * 100,
        "BLEU-2": bleu2_score * 100,
        "ROUGE-L": rouge_score * 100,
        "METEOR": meteor_score * 100,
    }

    return metrics, predictions_to_save


def save_metrics_txt(metrics, path, checkpoint_path):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(f"Valid Loss : {metrics['Valid Loss']:.6f}\n")
        f.write(f"BLEU-1     : {metrics['BLEU-1']:.4f}\n")
        f.write(f"BLEU-2     : {metrics['BLEU-2']:.4f}\n")
        f.write(f"ROUGE-L    : {metrics['ROUGE-L']:.4f}\n")
        f.write(f"METEOR     : {metrics['METEOR']:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/ghome/group05/gerard/MCV-C5/Task3/outputs/sweep_base_resnet50_gru_word_20260321_190631/checkpoints/best_model.pth",
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from checkpoint cfg",
    )
    parser.add_argument(
        "--save_predictions",
        type=str,
        default=None,
        help="Optional path to save generated captions as JSON",
    )
    parser.add_argument(
        "--save_metrics",
        type=str,
        default="/ghome/group05/gerard/MCV-C5/Task3/outputs/eval_metrics.txt",
        help="Path to save evaluation metrics as TXT",
    )
    args = parser.parse_args()

    print(f"Using device: {device}")
    print(f"Loading checkpoint from: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    cfg = deepcopy(load_checkpoint_cfg(checkpoint))
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    print("Building dataloaders and tokenizer...")
    train_loader, valid_loader, test_loader, tokenizer = get_dataloaders(cfg)

    # In your dataset.py, test_loader is built from cfg["val_ann_path"],
    # so it is effectively the real validation partition.
    eval_loader = test_loader
    print("Evaluation split: test_loader (mapped to val_ann_path in dataset.py)")

    print("Initializing Model...")
    model = BaselineModel(cfg, tokenizer).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    state_dict = checkpoint["model_state_dict"]
    load_model_with_resize(model, state_dict)

    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print("\nRunning evaluation...")
    metrics, predictions = evaluate_model(model, crit, eval_loader, tokenizer, cfg)

    print("\nGlobal Metrics")
    print("--------------")
    print(f"Valid Loss : {metrics['Valid Loss']:.6f}")
    print(f"BLEU-1     : {metrics['BLEU-1']:.4f}")
    print(f"BLEU-2     : {metrics['BLEU-2']:.4f}")
    print(f"ROUGE-L    : {metrics['ROUGE-L']:.4f}")
    print(f"METEOR     : {metrics['METEOR']:.4f}")

    print("\nSome sample predictions:")
    for sample in predictions[:5]:
        print("-" * 80)
        print(f"Image ID   : {sample['image_id']}")
        print(f"Prediction : {sample['prediction']}")
        print(f"References : {sample['references']}")

    if args.save_predictions is not None:
        pred_dir = os.path.dirname(args.save_predictions)
        if pred_dir:
            os.makedirs(pred_dir, exist_ok=True)

        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)

        print(f"\nSaved predictions to: {args.save_predictions}")

    save_metrics_txt(metrics, args.save_metrics, args.checkpoint)
    print(f"Saved metrics to: {args.save_metrics}")


if __name__ == "__main__":
    main()
