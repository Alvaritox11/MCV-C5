import os
import json
import argparse
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import KittiMotsDataset, detection_collate_fn
from src.metrics import CocoEvaluator
from src.models.sam_finetune_wrapper import SAMFineTuneWrapper
from src.visualize import create_wandb_image
from torch.utils.data import Subset
import random
from src.augmentations import transformations

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

@torch.inference_mode()
def evaluate_finetuned_sam(wrapper, dataloader, dataset, config):
    evaluator = CocoEvaluator(dataset)
    log_interval = config.get("log_image_interval", 50)

    for step, (images, targets) in enumerate(tqdm(dataloader, desc="Validation")):
        image = images[0]
        target = targets[0]

        if len(target["boxes"]) == 0:
            continue

        prompt_kwargs = {"input_boxes": [[target["boxes"].tolist()]]}
        pred_masks, pred_scores = wrapper.predict(image, prompt_kwargs)
        pred_labels = target["labels"].tolist()
        pred_boxes = target["boxes"].tolist()

        if step % log_interval == 0:
            wandb_img = create_wandb_image(
                image=image,
                masks=pred_masks,
                boxes=pred_boxes,
                title=f"Val step {step} | finetuned SAM bbox prompts"
            )
            wandb.log({"qualitative_results/val_predictions": wandb_img})

        prediction = {
            "image_id": target["image_id"].item(),
            "masks": pred_masks,
            "scores": pred_scores,
            "labels": pred_labels
        }
        evaluator.update([prediction])

    return evaluator.summarize()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=config["wandb_project"],
        entity=config.get("wandb_entity"),
        name=config["wandb_run_name"],
        config=config
    )

    train_transform = transformations(config.get("augmentation", "none"))

    train_dataset = KittiMotsDataset(
        root_dir=config["data_dir"],
        split="train",
        return_masks=True,
        transform=train_transform
    )

    val_dataset = KittiMotsDataset(
        root_dir=config["data_dir"],
        split="val",
        return_masks=True,
        transform=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=detection_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=detection_collate_fn
    )

    wrapper = SAMFineTuneWrapper(device=device, model_name=config["sam_model_name"])
    wrapper.load_checkpoint(config["checkpoint_path"])
    
    if config["method"] == "evaluate":
        stats, map_car, map_ped = evaluate_finetuned_sam(wrapper, val_loader, val_dataset, config)

        if stats is not None:
            map_5095 = float(stats[0])
            map_50 = float(stats[1])

            wandb.log({
                "val/mAP_0.50_0.95": map_5095,
                "val/mAP_0.50": map_50,
                "val/mAP_Car": map_car,
                "val/mAP_Pedestrian": map_ped
            })

            print("\n--- Evaluation Results ---")
            print(f"mAP (0.50:0.95): {map_5095:.4f}")
            print(f"mAP (0.50):      {map_50:.4f}")
            print(f"mAP Car:         {map_car:.4f}")
            print(f"mAP Pedestrian:  {map_ped:.4f}")

        wandb.finish()
        return

    optimizer = torch.optim.AdamW(
        wrapper.get_trainable_parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"]
    )
    best_map = -1.0
    ckpt_path = config["checkpoint_path"]

    if config["method"] == "train":
        for epoch in range(config["epochs"]):
            wrapper.model.train()
            epoch_loss = 0.0
            epoch_bce = 0.0
            epoch_dice = 0.0
            valid_steps = 0

            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
            for step, (images, targets) in enumerate(progress):
                image = images[0]
                target = targets[0]

                if len(target["boxes"]) == 0 or len(target["masks"]) == 0:
                    continue

                out = wrapper.forward_train(
                    image=image,
                    boxes=target["boxes"],
                    gt_masks=target["masks"]
                )
                if out is None:
                    continue

                loss = out["loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_bce += out["bce_loss"].item()
                epoch_dice += out["dice_loss"].item()
                valid_steps += 1

                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bce": f"{out['bce_loss'].item():.4f}",
                    "dice": f"{out['dice_loss'].item():.4f}"
                })

            mean_loss = epoch_loss / max(valid_steps, 1)
            mean_bce = epoch_bce / max(valid_steps, 1)
            mean_dice = epoch_dice / max(valid_steps, 1)

            wandb.log({
                "train/epoch": epoch + 1,
                "train/loss": mean_loss,
                "train/bce": mean_bce,
                "train/dice": mean_dice,
                "train/lr": optimizer.param_groups[0]["lr"]
            })

            stats, map_car, map_ped = evaluate_finetuned_sam(wrapper, val_loader, val_dataset, config)

            if stats is not None:
                map_5095 = float(stats[0])
                map_50 = float(stats[1])

                wandb.log({
                    "val/epoch": epoch + 1,
                    "val/mAP_0.50_0.95": map_5095,
                    "val/mAP_0.50": map_50,
                    "val/mAP_Car": map_car,
                    "val/mAP_Pedestrian": map_ped
                })

                print(f"\nEpoch {epoch+1}")
                print(f"train loss: {mean_loss:.4f}")
                print(f"val mAP(0.50:0.95): {map_5095:.4f}")

                if map_5095 > best_map:
                    best_map = map_5095
                    wrapper.save_checkpoint(ckpt_path)
                    print(f"Saved best checkpoint to {ckpt_path}")

            scheduler.step()
    wandb.finish()

if __name__ == "__main__":
    main()