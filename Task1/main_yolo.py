import os
import json
import argparse
from pathlib import Path
import time

import torch
import wandb
#from wandb.integration.ultralytics import add_wandb_callback

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import KittiMotsDataset, detection_collate_fn
from src.metrics import CocoEvaluator
from src.models.yolo_wrapper import YoloWrapper

def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)

@torch.inference_mode()
def evaluate_coco(model_wrapper, dataloader, dataset, confidence_threshold: float):
    if hasattr(model_wrapper, "model") and hasattr(model_wrapper.model, "eval"):
        model_wrapper.model.eval()

    evaluator = CocoEvaluator(dataset)
    total_images = 0
    inference_time = 0.0

    for images, targets in tqdm(dataloader, desc="Evaluating (CocoEvaluator)"):
        start_inf = time.time()

        predictions = model_wrapper.predict(images, confidence_threshold=confidence_threshold)

        inference_time += time.time() - start_inf
        total_images += len(images)

        for pred, target in zip(predictions, targets):
            pred["image_id"] = target["image_id"].item()

        evaluator.update(predictions)

    fps = total_images / inference_time if inference_time > 0 else 0
    stats, map_car, map_ped = evaluator.summarize()

    return stats, map_car, map_ped, fps

def transformations(aug_name: str):
    if aug_name == "none":
        return {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "erasing": 0.0,
            "auto_augment": None,
        }

    elif aug_name == "simple":
        return {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 0.0,
            "erasing": 0.0,
            "auto_augment": None,
        }

    elif aug_name == "weather":
        return {
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.6,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "erasing": 0.2,
            "auto_augment": "randaugment",  # adds stronger color/weather-like effects
        }

    else:
        raise ValueError(f"Unknown augmentation: {aug_name}")

def main():
    parser = argparse.ArgumentParser(description="C5 YOLO Pipeline (Ultralytics train + optional CocoEvaluator eval)")
    parser.add_argument("--config", type=str, default="configs/yolo_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    mode = cfg.get("mode", "train").lower()  # train / evaluate / train_then_evaluate
    
    # --- W&B (optional) ---
    use_wandb = bool(cfg.get("use_wandb", True))

    # Only initialize W&B manually if NOT training
    if use_wandb:
        wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity", None),
            name=cfg.get("run_name", "run_name_undefined"),
            config=cfg,
        )

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- YOLO Pipeline | Mode: {mode.upper()} | Device: {device} ---")

    # --- Wrapper ---
    model_path = cfg.get("yolo_model_path", "yolo26x.pt")
    wrapper = YoloWrapper(model_path=model_path, device=str(device))

    # --- Dataset for CocoEvaluator eval ---
    # Only needed if you run mode=evaluate or train_then_evaluate
    def build_eval_loaders():
        data_dir = cfg["data_dir"]  # original KITTI-MOTS root ("/home/mcv/datasets/C5/KITTI-MOTS/")
        split = cfg.get("eval_split", "val")
        batch_size = int(cfg.get("eval_batch_size", 16))
        eval_dataset = KittiMotsDataset(root_dir=data_dir, split=split, transforms=None)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=detection_collate_fn,
        )
        return eval_dataset, eval_loader

    # --- TRAIN with Ultralytics ---
    if mode in ("train", "train_then_evaluate"):
        # Ultralytics expects YOLO-exported dataset yaml 
        yolo_data_yaml = cfg["yolo_data_yaml"]


        submit_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
        print(f"SLURM_SUBMIT_DIR: {submit_dir}")
        project = str(Path(submit_dir) / cfg.get("yolo_project", "runs"))
        name = cfg.get("run_name", "run_name_undefined")

        
        aug_args  = transformations(cfg.get("yolo_augmentation", "none"))
        # IMPORTANT: workers should not exceed Slurm cores. Set in config accordingly.
        results = wrapper.model.train(
            data=yolo_data_yaml,
            epochs=int(cfg.get("epochs", 50)),
            imgsz=int(cfg.get("imgsz", 640)),
            batch=int(cfg.get("batch_size", 16)),
            device=cfg.get("yolo_device", 0),  # 0 or "0" or "cpu"
            workers=int(cfg.get("workers", 8)),
            lr0=float(cfg.get("learning_rate", 1e-2)),
            weight_decay=float(cfg.get("weight_decay", 5e-4)),
            project=project,
            name=name,
            pretrained=True,
            save = True,
            **aug_args
        )

        # Get the actual save directory from Ultralytics
        save_dir = getattr(results, "save_dir", None)
        if save_dir is None:
            # fallback to config-based path
            save_dir = Path(project) / name
        else:
            save_dir = Path(save_dir)

        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"Reloading best checkpoint: {best_pt}")
            wrapper = YoloWrapper(model_path=str(best_pt), device=str(device))
        else:
            print(f"WARNING: best.pt not found at {best_pt}")

    # --- EVALUATE with CocoEvaluator ---
    if mode in ("evaluate", "train_then_evaluate"):
        eval_dataset, eval_loader = build_eval_loaders()

        stats = evaluate_coco(
            wrapper,
            eval_loader,
            eval_dataset,
            confidence_threshold=float(cfg.get("confidence_threshold", 0.25)),
        )
        mAP = float(stats[0]) if stats is not None else 0.0
        print(f"Evaluation Complete. mAP (0.5:0.95): {mAP:.4f}")

        if use_wandb:
            wandb.log({"val/mAP_0.5_0.95": mAP})

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()