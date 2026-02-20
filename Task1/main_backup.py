import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import KittiMotsDataset_HF, KittiMotsDataset_FRNN
from src.metrics import CocoEvaluator

from src.models.detr_wrapper import DetrWrapper
from src.models.frnn_wrapper import FasterRCNNWrapper
from src.models.yolo_wrapper import YoloWrapper
# from src.models.rcnn_wrapper import FasterRCNNWrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='inference',
                        choices=['inference', 'eval', 'train'],
                        help='Execution mode: inference only, evaluation (COCO metrics), or training')
    parser.add_argument('--model', type=str, default='detr',
                        choices=['yolo', 'detr', 'fasterrcnn'],
                        help='Model framework to use')
    parser.add_argument('--data_root', type=str, default='/home/mcv/datasets/C5/KITTI-MOTS/',
                        help='Path to KITTI-MOTS dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)

    # ---- wandb ----
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='FRNN', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='Team5-C5', help='wandb entity/team (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name (optional)')
    parser.add_argument('--wandb_tags', type=str, default='', help='Comma-separated wandb tags (optional)')
    parser.add_argument('--wandb_notes', type=str, default='', help='wandb notes (optional)')

    return parser.parse_args()


def get_model(args):
    if args.model == 'detr':
        dataset = KittiMotsDataset_HF(root_dir=args.data_root, split='val')
        return DetrWrapper(device=args.device), dataset
    elif args.model == 'yolo':
        return YoloWrapper(args)
    elif args.model == 'fasterrcnn':
        # return FasterRCNNWrapper(args)
        dataset = KittiMotsDataset_FRNN(root_dir=args.data_root, split='val')
        return FasterRCNNWrapper(device=args.device), dataset
    else:
        raise ValueError(f"Unknown model: {args.model}")


def collate_fn(batch):
    return tuple(zip(*batch))


def _maybe_init_wandb(args):
    """Initialize wandb if enabled; returns wandb module or None."""
    if not args.use_wandb:
        return None

    try:
        import wandb
    except ImportError as e:
        raise ImportError("wandb is not installed. `pip install wandb` or run without --use_wandb") from e

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags if tags else None,
        notes=args.wandb_notes if args.wandb_notes else None,
        config=vars(args),
    )
    return wandb


def _log_coco_metrics_to_wandb(wandb, evaluator):
    """
    Try to log COCO metrics if available.
    This is intentionally defensive because different CocoEvaluator implementations store results differently.
    """
    if wandb is None or evaluator is None:
        return

    metrics = {}

    # Common patterns you might have in your CocoEvaluator implementation:
    # - evaluator.coco_eval.stats (pycocotools COCOeval)
    # - evaluator.stats dict
    # - evaluator.results_summary dict
    if hasattr(evaluator, "coco_eval") and evaluator.coco_eval is not None:
        # coco_eval may be dict per iouType, e.g. {'bbox': COCOeval}
        ce = evaluator.coco_eval
        if isinstance(ce, dict) and "bbox" in ce and hasattr(ce["bbox"], "stats"):
            stats = ce["bbox"].stats
            # COCOeval stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
            keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
            metrics.update({f"coco/bbox_{k}": float(v) for k, v in zip(keys, stats)})
        elif hasattr(ce, "stats"):
            stats = ce.stats
            keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
            metrics.update({f"coco/bbox_{k}": float(v) for k, v in zip(keys, stats)})

    if hasattr(evaluator, "stats") and isinstance(evaluator.stats, dict):
        # If your evaluator already provides a dict of metrics
        metrics.update({f"coco/{k}": float(v) for k, v in evaluator.stats.items()})

    if metrics:
        wandb.log(metrics)


def main():
    args = get_args()
    print(f"--- Starting C5 Project Task 1 ---")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")

    wandb = _maybe_init_wandb(args)

    # Model initialization
    model, dataset = get_model(args)
    # Dataset initialization
    print("Loading Dataset...")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    print(f"Dataset Loaded: {len(dataset)} images.")

    if wandb is not None:
        wandb.config.update(
            {"num_images": len(dataset), "split": "val"},
            allow_val_change=True
        )

    # Evaluator initialization
    evaluator = None
    if args.mode == 'eval':
        evaluator = CocoEvaluator(dataset)

    # Inference Loop
    print("Starting Inference Loop...")

    for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
        batch_preds = model.predict(images)

        for i, pred in enumerate(batch_preds):
            target_id = targets[i]['image_id'].item()
            pred['image_id'] = target_id

            if evaluator:
                evaluator.results.append(pred)

        # Optional: log progress every N batches
        if wandb is not None:
            wandb.log({"progress/batch_idx": batch_idx})

    # Final Evaluation
    if args.mode == 'eval' and evaluator:
        print("\n--- Running COCO Evaluation ---")
        evaluator.summarize()
        print("-------------------------------")

        # Try to log metrics if available in evaluator
        _log_coco_metrics_to_wandb(wandb, evaluator)

        # Also store the number of predictions as a sanity check
        if wandb is not None:
            wandb.log({"debug/num_predictions": len(evaluator.results)})

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
