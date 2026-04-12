import os
import json
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import cv2

from Task4.Qwen_code.dataset import get_qwen_dataloaders
from Task4.Qwen_code.models import BaselineModel, HFTransformerModel, ViTLlamaModel

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(model, optimizer, dataloader, cfg):
    model.train()
    total_loss = 0
    accumulation_steps = cfg.get("gradient_accumulation_steps", 4)
    optimizer.zero_grad()
    loop = tqdm(dataloader, desc="Training", leave=False)

    for i, (img_ids, pixel_values, labels, _) in enumerate(loop):
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
        labels = labels.to(device)
        attention_mask = (labels != -100).long()

        outputs = model(pixel_values=pixel_values, labels=labels, decoder_attention_mask=attention_mask)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if (i + 1) % accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        loop.set_postfix(loss=loss.item() * accumulation_steps)

    # Handle leftover batches if dataset size isn't divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


def get_token_list_from_ids(tokenizer, ids):
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx == tokenizer.eos_id:
            break
        if idx not in [tokenizer.sos_id, tokenizer.pad_id]:
            if hasattr(tokenizer, "idx2token"):
                tokens.append(tokenizer.idx2token[idx])
            else:
                # subword tokenizer fallback
                tok = tokenizer.tokenizer.convert_ids_to_tokens([idx])[0]
                tokens.append(tok)
    return tokens


def save_attention_plot(unnorm_img, attn_maps_img, pred_ids_img, tokenizer, plot_path, max_words=6):
    """
    unnorm_img: (H, W, 3) numpy image in [0,1]
    attn_maps_img: (seq_len, num_regions)
    pred_ids_img: (seq_len,)
    """
    token_list = get_token_list_from_ids(tokenizer, pred_ids_img.tolist())

    if len(token_list) == 0:
        token_list = ["<no prediction>"]

    num_words = min(max_words, len(token_list), attn_maps_img.shape[0])

    fig, axes = plt.subplots(1, num_words, figsize=(4 * num_words, 4))
    if num_words == 1:
        axes = [axes]

    for t in range(num_words):
        attn = attn_maps_img[t].detach().cpu().numpy()  # (num_regions,)
        side = int(np.sqrt(attn.shape[0]))

        if side * side != attn.shape[0]:
            # fallback: skip overlay if shape is not square
            axes[t].imshow(unnorm_img)
            axes[t].axis('off')
            axes[t].set_title(token_list[t], fontsize=10)
            continue

        attn = attn.reshape(side, side)
        attn = cv2.resize(attn, (unnorm_img.shape[1], unnorm_img.shape[0]))
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        axes[t].imshow(unnorm_img)
        axes[t].imshow(attn, cmap='jet', alpha=0.45)
        axes[t].axis('off')
        axes[t].set_title(token_list[t], fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close(fig)


def evaluate_and_log(model, dataloader, epoch, output_dir, tokenizer, cfg, run_predictions):
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []

    plot_saved = False

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for img_ids, pixel_values, labels, all_raw_captions in loop:
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            labels = labels.to(device)
            
            # Add mask here as well
            attention_mask = (labels != -100).long()

            loss_output = model(
                pixel_values=pixel_values, 
                labels=labels, 
                decoder_attention_mask=attention_mask
            )
            loss = loss_output.loss
            total_loss += loss.item()

            # ---- METRICS / GENERATION: Hugging Face generate ----
            generated_ids = model.generate(
                pixel_values, 
                max_length=cfg.get("max_length", tokenizer.model_max_length)
            )
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            refs_for_batch = []
            for b in range(pixel_values.size(0)):
                refs_for_batch.append([all_raw_captions[j][b] for j in range(len(all_raw_captions))])

            for i, pred_str in enumerate(batch_preds):
                current_id = str(img_ids[i].item() if torch.is_tensor(img_ids[i]) else img_ids[i])

                if current_id not in run_predictions:
                    run_predictions[current_id] = {
                        "references": refs_for_batch[i],
                        "predictions": {}
                    }
                run_predictions[current_id]["predictions"][f"epoch_{epoch}"] = pred_str

                all_preds.append(pred_str)
                all_refs.append(refs_for_batch[i])

            # ---- QUALITATIVE PLOTS ----
            if not plot_saved:
                plot_saved = True
                
                mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
                std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)

                for img_idx in range(min(2, pixel_values.size(0))):
                    unnorm_img = pixel_values[img_idx] * std + mean
                    unnorm_img = torch.clamp(unnorm_img, 0, 1).cpu().permute(1, 2, 0).numpy()

                    gt_text = "\n".join([f"- {txt}" for txt in refs_for_batch[img_idx][:3]])
                    pred_text = batch_preds[img_idx]

                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(unnorm_img)
                    ax.axis('off')
                    ax.set_title(f"PRED: {pred_text}\n\nGT:\n{gt_text}", fontsize=10, loc='left')

                    plot_path = os.path.join(output_dir, "plots", f"epoch_{epoch}_img_{img_idx}.png")
                    plt.savefig(plot_path, bbox_inches='tight')
                    wandb.log({f"Qualitative/Epoch_{epoch}_Img_{img_idx}": wandb.Image(plot_path)})
                    plt.close(fig)

    avg_loss = total_loss / len(dataloader)

    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(run_predictions, f, indent=4)

    bleu1_score = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)['bleu']
    bleu2_score = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)['bleu']
    rouge_score = rouge.compute(predictions=all_preds, references=all_refs)['rougeL']
    meteor_score = meteor.compute(predictions=all_preds, references=all_refs)['meteor']

    metrics = {
        "Valid Loss": avg_loss,
        "BLEU-1": bleu1_score * 100,
        "BLEU-2": bleu2_score * 100,
        "ROUGE-L": rouge_score * 100,
        "METEOR": meteor_score * 100
    }
    return metrics


def main():
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config.json file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    output_dir = os.path.join("outputs", cfg["run_name"])
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    wandb.init(
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
        name=cfg["run_name"],
        config=cfg
    )
    print("Initializing Transformer Model...")

    model_type = cfg.get("model_type", "hf_transformer")

    if model_type == "vit_qwen_05b":
        from Task4.Qwen_code.models import ViTQwen05BModel
        model = ViTQwen05BModel(cfg).to(device)
        print(f"Device: {device}")
        print(f"Model device: {next(model.parameters()).device}")
        tokenizer = model.tokenizer

    elif model_type == "vit_qwen_15b":
        from Task4.Qwen_code.models import ViTQwen15BModel
        model = ViTQwen15BModel(cfg).to(device)
        tokenizer = model.tokenizer

    else:
        model = HFTransformerModel(cfg).to(device)


    print("Initializing DataLoaders...")
    train_loader, valid_loader, test_loader = get_qwen_dataloaders(cfg, tokenizer)

    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-4))

    best_valid_loss = float('inf')

    early_stopping_enabled = cfg.get("early_stopping", False)
    patience = cfg.get("early_stopping_patience", 3)
    min_delta = cfg.get("early_stopping_min_delta", 0.0)
    epochs_without_improvement = 0

    run_predictions = {}

    print("Starting Training Loop...")
    for epoch in range(1, cfg.get("epochs", 10) + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, cfg)
        metrics = evaluate_and_log(model, valid_loader, epoch, output_dir, tokenizer, cfg, run_predictions)
        valid_loss = metrics["Valid Loss"]

        print(f"Epoch {epoch}/{cfg.get('epochs', 10)} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"Metrics: BLEU-1: {metrics['BLEU-1']:.1f}% | ROUGE-L: {metrics['ROUGE-L']:.1f}%")

        wandb.log({"epoch": epoch, "train_loss": train_loss, **metrics})

        if valid_loss < best_valid_loss - min_delta:
            print(f"Validation loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}. Saving checkpoint!")
            best_valid_loss = valid_loss
            epochs_without_improvement = 0

            ckpt_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, ckpt_path)
            wandb.save(ckpt_path)
        else:
            epochs_without_improvement += 1
            print(
                f"No significant validation improvement for {epochs_without_improvement} epoch(s). "
                f"Best valid loss: {best_valid_loss:.4f}"
            )

            if early_stopping_enabled and epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    wandb.finish()
    print("Training Complete!") 

if __name__ == "__main__":
    main()