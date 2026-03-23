# train.py

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
import datetime

# import vocab
from dataset import get_dataloaders
from models import BaselineModel

bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, optimizer, crit, dataloader, cfg):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=False)

    for img_ids, imgs, captions, _ in loop:
        imgs, captions = imgs.to(device), captions.to(device)

        optimizer.zero_grad()

        if cfg.get("teacher_forcing", False):
            preds = model(imgs, captions)
        else:
            preds = model(imgs)

        # Shift captions by 1 to match the seq_len-1 output of the model
        loss = crit(preds, captions[:, 1:])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def build_optimizer(model, cfg):
    opt_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("learning_rate", 1e-4)
    weight_decay = cfg.get("weight_decay", 0.0)

    # Only optimize trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.get('optimizer')}")


def build_scheduler(optimizer, cfg):
    if not cfg.get("lr_decay", False):
        return None

    return optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,   # decay every 5 epochs
        gamma=0.5      # halve the LR
    )


def evaluate_and_log(model, crit, dataloader, epoch, output_dir, tokenizer, cfg, run_predictions):
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []

    plot_saved = False

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for img_ids, imgs, captions, all_raw_captions in loop:
            imgs, captions = imgs.to(device), captions.to(device)

            # ---- LOSS: match training mode ----
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
                refs_for_batch.append([all_raw_captions[j][b] for j in range(len(all_raw_captions))])

            batch_preds = []
            for i in range(pred_ids.shape[0]):
                # pred_str = ""
                # for char_id in pred_ids[i]:
                #     char = vocab.IDX2CHAR[char_id.item()]
                #     if char == '<EOS>': break
                #     if char not in ['<SOS>', '<PAD>']: pred_str += char

                pred_str = tokenizer.decode(pred_ids[i].tolist())

                # for dictionary prediction storage
                current_id = str(img_ids[i].item() if torch.is_tensor(img_ids[i]) else img_ids[i])
                
                if current_id not in run_predictions:
                    run_predictions[current_id] = {
                        "references": refs_for_batch[i],
                        "predictions": {}
                    }
                run_predictions[current_id]["predictions"][f"epoch_{epoch}"] = pred_str

                batch_preds.append(pred_str)
                all_preds.append(pred_str)
                all_refs.append(refs_for_batch[i])

            if not plot_saved:
                plot_saved = True
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

                for img_idx in range(min(2, imgs.size(0))):
                    unnorm_img = imgs[img_idx] * std + mean
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
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, required=True, help='Path to the config.json file')
    #args = parser.parse_args()

    with open("/ghome/group05/gerard/MCV-C5/Task3/configs/sweep_base.json", 'r') as f:
        cfg = json.load(f)

    

    wandb.init(
        project=cfg["wandb_project"],
        entity=cfg["wandb_entity"],
        name=cfg.get("run_name", None),
        config=cfg
    )


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = cfg.get("run_name", "experiment")
    timed_run_name = f"{base_name}_{timestamp}"

    # IMPORTANT: from here on, use sweep-overridden config
    cfg = dict(wandb.config)

    output_dir = os.path.join("outputs", timed_run_name)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    print("Initializing DataLoaders...")
    train_loader, valid_loader, test_loader, tokenizer = get_dataloaders(cfg)

    print("Initializing Baseline Model...")
    model = BaselineModel(cfg, tokenizer).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    crit = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    best_valid_loss = float('inf')

    early_stopping_enabled = cfg.get("early_stopping", False)
    patience = cfg.get("early_stopping_patience", 3)
    min_delta = cfg.get("early_stopping_min_delta", 0.0)
    epochs_without_improvement = 0

    run_predictions = {}

    print("Starting Training Loop...")
    for epoch in range(1, cfg.get("epochs", 10) + 1):
        train_loss = train_one_epoch(model, optimizer, crit, train_loader, cfg)
        metrics = evaluate_and_log(model, crit, valid_loader, epoch, output_dir, tokenizer, cfg, run_predictions)
        if scheduler is not None:
            scheduler.step()
        valid_loss = metrics["Valid Loss"]

        print(f"Epoch {epoch}/{cfg.get('epochs', 10)} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"Metrics: BLEU-1: {metrics['BLEU-1']:.1f}% | METEOR: {metrics['METEOR']:.1f}%")

        wandb.log({"epoch": epoch, "train_loss": train_loss, **metrics})

        if valid_loss < best_valid_loss - min_delta:
            print(f"🌟 Validation loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}. Saving checkpoint!")
            best_valid_loss = valid_loss
            epochs_without_improvement = 0

            ckpt_path = os.path.join(output_dir, "checkpoints", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
                'config': cfg
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