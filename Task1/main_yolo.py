import json
import argparse
from pathlib import Path

import torch
import wandb
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
    # Your wrapper.predict returns standardized dicts -> CocoEvaluator expects those
    if hasattr(model_wrapper, "model") and hasattr(model_wrapper.model, "eval"):
        model_wrapper.model.eval()

    evaluator = CocoEvaluator(dataset)

    for images, targets in tqdm(dataloader, desc="Evaluating (CocoEvaluator)"):
        predictions = model_wrapper.predict(images, confidence_threshold=confidence_threshold)

        for pred, target in zip(predictions, targets):
            pred["image_id"] = target["image_id"].item()

        evaluator.update(predictions)

    stats = evaluator.summarize()
    return stats


def main():
    parser = argparse.ArgumentParser(description="C5 YOLO Pipeline (Ultralytics train + optional CocoEvaluator eval)")
    parser.add_argument("--config", type=str, default="configs/yolo_config.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- W&B (optional) ---
    use_wandb = bool(cfg.get("use_wandb", True))
    if use_wandb:
        wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity", None),
            name=cfg.get("wandb_run_name", "yolo_run"),
            config=cfg,
        )

    mode = cfg.get("mode", "train").lower()  # train / evaluate / train_then_evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- YOLO Pipeline | Mode: {mode.upper()} | Device: {device} ---")

    # --- Wrapper (unchanged) ---
    model_path = cfg.get("yolo_model_path", "yolo26x.pt")
    wrapper = YoloWrapper(model_path=model_path, device=str(device))

    # --- Dataset for YOUR CocoEvaluator eval (uses original KITTI-MOTS structure) ---
    # Only needed if you run mode=evaluate or train_then_evaluate
    def build_eval_loaders():
        data_dir = cfg["data_dir"]  # original KITTI-MOTS root (with training/image_02 and instances_txt)
        split = cfg.get("eval_split", "val")
        batch_size = int(cfg.get("eval_batch_size", 8))
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
        # Ultralytics expects YOLO-exported dataset yaml (the one you created in ~/KITTI-MOTS_YOLO/data.yaml)
        yolo_data_yaml = cfg["yolo_data_yaml"]

        # Where to store runs
        project = cfg.get("yolo_project", "runs/detect")
        name = cfg.get("yolo_run_name", cfg.get("wandb_run_name", "yolo_train"))

        # IMPORTANT: workers should not exceed Slurm cores. Set in config accordingly.
        results = wrapper.model.train(
            data=yolo_data_yaml,
            epochs=int(cfg.get("epochs", 50)),
            imgsz=int(cfg.get("imgsz", 640)),
            batch=int(cfg.get("batch_size", 16)),
            device=cfg.get("yolo_device", 0),  # 0 or "0" or "cpu"
            workers=int(cfg.get("workers", 8)),
            lr0=float(cfg.get("learning_rate", 1e-3)),
            weight_decay=float(cfg.get("weight_decay", 5e-4)),
            project=project,
            name=name,
            pretrained=True,
        )

        # Reload best weights (so evaluate uses best.pt, not last)
        best_pt = Path(project) / name / "weights" / "best.pt"
        if best_pt.exists():
            print(f"Reloading best checkpoint: {best_pt}")
            wrapper = YoloWrapper(model_path=str(best_pt), device=str(device))
        else:
            print(f"WARNING: best.pt not found at {best_pt} (will evaluate current in-memory model if requested).")

        # You can log Ultralytics results object if desired:
        if use_wandb and results is not None:
            wandb.log({"train/done": 1})

    # --- EVALUATE with YOUR CocoEvaluator ---
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