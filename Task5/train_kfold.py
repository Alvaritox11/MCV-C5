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

from dataset import get_kfold_dataloaders
from models import BaselineModel

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_epoch(model, optimizer, crit, dataloader, cfg, scaler):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    loop = tqdm(dataloader, desc="Training", leave=False)

    for img_ids, imgs, captions, _ in loop:
        imgs, captions = imgs.to(device), captions.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            output = model(
                imgs,
                captions=captions,
                return_loss_logits=True
            )

            logits_tf = output["logits_tf"]
            loss = crit(logits_tf, captions[:, 1:])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        total_loss += loss.detach()
        loop.set_postfix(loss=float(loss.detach()))

    return (total_loss / len(dataloader)).item()


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


def evaluate_and_log(model, crit, dataloader, epoch, fold_idx, output_dir, tokenizer, cfg, run_predictions):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    all_preds = []
    all_refs = []

    plot_saved = False

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            loop = tqdm(dataloader, desc="Evaluating", leave=False)
            for img_ids, imgs, captions, all_raw_captions in loop:
                imgs, captions = imgs.to(device), captions.to(device)

                # ---- SINGLE FORWARD ----
                output = model(
                    imgs,
                    captions=captions,
                    return_attention=True,
                    return_loss_logits=True
                )

                logits_tf = output["logits_tf"]      # for loss
                gen_output = output["logits_gen"]    # for generation
                attn_maps = output["attn"]

                # ---- LOSS ----
                loss = crit(logits_tf, captions[:, 1:])
                total_loss += loss.detach()

                # ---- PREDICTIONS ----
                pred_ids = torch.argmax(gen_output, dim=1)

                refs_for_batch = []
                for b in range(imgs.size(0)):
                    refs_for_batch.append([all_raw_captions[j][b] for j in range(len(all_raw_captions))])

                batch_preds = []
                for i in range(pred_ids.shape[0]):
                    pred_str = tokenizer.decode(pred_ids[i].tolist())

                    current_id = str(img_ids[i].item() if torch.is_tensor(img_ids[i]) else img_ids[i])

                    if current_id not in run_predictions:
                        run_predictions[current_id] = {
                            "references": refs_for_batch[i],
                            "predictions": {}
                        }
                    run_predictions[current_id]["predictions"][f"fold_{fold_idx}_epoch_{epoch}"] = pred_str

                    batch_preds.append(pred_str)
                    all_preds.append(pred_str)
                    all_refs.append(refs_for_batch[i])

                # ---- QUALITATIVE PLOTS ----
                if not plot_saved:
                    plot_saved = True
                    

                    for img_idx in range(min(2, imgs.size(0))):
                        unnorm_img = imgs[img_idx] * std + mean
                        unnorm_img = torch.clamp(unnorm_img, 0, 1).cpu().permute(1, 2, 0).numpy()

                        gt_text = "\n".join([f"- {txt}" for txt in refs_for_batch[img_idx][:3]])
                        pred_text = batch_preds[img_idx]

                        # Original qualitative image-caption plot
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(unnorm_img)
                        ax.axis('off')
                        ax.set_title(f"PRED: {pred_text}\n\nGT:\n{gt_text}", fontsize=10, loc='left')

                        plot_path = os.path.join(output_dir, "plots", f"fold_{fold_idx}_epoch_{epoch}_img_{img_idx}.png")
                        plt.savefig(plot_path, bbox_inches='tight')
                        wandb.log({f"Fold_{fold_idx}/Qualitative/Epoch_{epoch}_Img_{img_idx}": wandb.Image(plot_path)})
                        plt.close(fig)

                        # Attention plot only if attention is active and maps are available
                        if cfg.get("use_attention", False) and attn_maps is not None:
                            attn_plot_path = os.path.join(
                                output_dir,
                                "plots",
                                f"fold_{fold_idx}_epoch_{epoch}_img_{img_idx}_attention.png"
                            )

                            save_attention_plot(
                                unnorm_img=unnorm_img,
                                attn_maps_img=attn_maps[img_idx],
                                pred_ids_img=pred_ids[img_idx],
                                tokenizer=tokenizer,
                                plot_path=attn_plot_path,
                                max_words=6
                            )

                            wandb.log({
                                f"Fold_{fold_idx}/Attention/Epoch_{epoch}_Img_{img_idx}": wandb.Image(attn_plot_path)
                            })

    avg_loss = (total_loss / len(dataloader)).item()

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


def train_single_fold(fold_idx, train_loader, valid_loader, tokenizer, cfg, output_dir, fold_predictions):
    """Train a single fold and return the best validation metrics."""
    print(f"\n{'='*60}")
    print(f"Starting Fold {fold_idx + 1}/{cfg.get('k_folds', 5)}")
    print(f"{'='*60}\n")

    # Initialize model for this fold
    model = BaselineModel(cfg, tokenizer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-4))
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    scaler = torch.cuda.amp.GradScaler()

    best_valid_loss = float('inf')
    best_metrics = None

    early_stopping_enabled = cfg.get("early_stopping", False)
    patience = cfg.get("early_stopping_patience", 3)
    min_delta = cfg.get("early_stopping_min_delta", 0.0)
    epochs_without_improvement = 0

    for epoch in range(1, cfg.get("epochs", 10) + 1):
        train_loss = train_one_epoch(model, optimizer, crit, train_loader, cfg, scaler)
        metrics = evaluate_and_log(model, crit, valid_loader, epoch, fold_idx, output_dir, tokenizer, cfg, fold_predictions)
        valid_loss = metrics["Valid Loss"]

        print(f"Fold {fold_idx + 1} | Epoch {epoch}/{cfg.get('epochs', 10)} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"Metrics: BLEU-1: {metrics['BLEU-1']:.1f}% | ROUGE-L: {metrics['ROUGE-L']:.1f}%")

        # Log with fold-specific prefix
        wandb.log({
            "fold": fold_idx,
            "epoch": epoch,
            f"Fold_{fold_idx}/train_loss": train_loss,
            f"Fold_{fold_idx}/valid_loss": valid_loss,
            f"Fold_{fold_idx}/BLEU-1": metrics['BLEU-1'],
            f"Fold_{fold_idx}/BLEU-2": metrics['BLEU-2'],
            f"Fold_{fold_idx}/ROUGE-L": metrics['ROUGE-L'],
            f"Fold_{fold_idx}/METEOR": metrics['METEOR']
        })

        if valid_loss < best_valid_loss - min_delta:
            print(f"Validation loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}. Saving checkpoint!")
            best_valid_loss = valid_loss
            best_metrics = metrics.copy()
            epochs_without_improvement = 0

            ckpt_path = os.path.join(output_dir, "checkpoints", f"best_model_fold_{fold_idx}.pth")
            torch.save({
                'fold': fold_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'metrics': metrics
            }, ckpt_path)
            wandb.save(ckpt_path)
        else:
            epochs_without_improvement += 1
            print(
                f"No significant validation improvement for {epochs_without_improvement} epoch(s). "
                f"Best valid loss: {best_valid_loss:.4f}"
            )

            if early_stopping_enabled and epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch} epochs for fold {fold_idx + 1}.")
                break

    return best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config.json file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    k_folds = cfg.get("k_folds", 5)
    
    output_dir = os.path.join("outputs", cfg["run_name"])
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    wandb.init(
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
        name=cfg["run_name"],
        config=cfg
    )

    # Store metrics from all folds
    all_fold_metrics = []
    all_fold_predictions = {}

    print(f"Starting {k_folds}-Fold Cross-Validation...")
    
    for fold_idx in range(k_folds):
        # Get dataloaders for this specific fold
        print(f"\nPreparing data for Fold {fold_idx + 1}/{k_folds}...")
        train_loader, valid_loader, tokenizer = get_kfold_dataloaders(cfg, fold_idx)
        
        fold_predictions = {}
        
        # Train this fold
        best_metrics = train_single_fold(
            fold_idx=fold_idx,
            train_loader=train_loader,
            valid_loader=valid_loader,
            tokenizer=tokenizer,
            cfg=cfg,
            output_dir=output_dir,
            fold_predictions=fold_predictions
        )
        
        all_fold_metrics.append(best_metrics)
        all_fold_predictions[f"fold_{fold_idx}"] = fold_predictions
        
        # Log fold summary
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1} Summary:")
        print(f"  Valid Loss: {best_metrics['Valid Loss']:.4f}")
        print(f"  BLEU-1: {best_metrics['BLEU-1']:.2f}%")
        print(f"  BLEU-2: {best_metrics['BLEU-2']:.2f}%")
        print(f"  ROUGE-L: {best_metrics['ROUGE-L']:.2f}%")
        print(f"  METEOR: {best_metrics['METEOR']:.2f}%")
        print(f"{'='*60}\n")

    # Save all predictions
    with open(os.path.join(output_dir, "kfold_predictions.json"), "w") as f:
        json.dump(all_fold_predictions, f, indent=4)

    # Calculate and log average metrics across all folds
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation Results (k={k_folds})")
    print(f"{'='*60}\n")
    
    avg_metrics = {}
    std_metrics = {}
    
    for metric_name in all_fold_metrics[0].keys():
        values = [fold_metrics[metric_name] for fold_metrics in all_fold_metrics]
        avg_metrics[metric_name] = np.mean(values)
        std_metrics[metric_name] = np.std(values)
        
        print(f"{metric_name}: {avg_metrics[metric_name]:.4f} ± {std_metrics[metric_name]:.4f}")
        
        wandb.log({
            f"KFold_Average/{metric_name}": avg_metrics[metric_name],
            f"KFold_Std/{metric_name}": std_metrics[metric_name]
        })
    
    # Save summary
    summary = {
        "k_folds": k_folds,
        "individual_folds": all_fold_metrics,
        "average_metrics": avg_metrics,
        "std_metrics": std_metrics
    }
    
    with open(os.path.join(output_dir, "kfold_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n{'='*60}")
    print("K-Fold Cross-Validation Complete!")
    print(f"{'='*60}\n")
    
    wandb.finish()


if __name__ == "__main__":
    main()