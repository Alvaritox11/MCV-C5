import os
import json
import time
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from domainshift.dataset_ds import OxfordPetDataset, detection_collate_fn
from domainshift.eval_ds import CocoEvaluatorOxfordPet 
from src.models.sam_finetune_wrapper import SAMFineTuneWrapper


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


@torch.inference_mode()
def evaluate_finetuned_sam(wrapper, dataloader, dataset):
    print("Dataset size:", len(dataset))
    evaluator = CocoEvaluatorOxfordPet(dataset)

    total_inference_time = 0.0
    total_images = 0

    for images, targets in tqdm(dataloader, desc="Validation"):
        image = images[0]
        target = targets[0]

        if len(target["boxes"]) == 0:
            continue

        prompt_kwargs = {
            "input_boxes": [target["boxes"].tolist()]
        }

        if wrapper.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        pred_masks, pred_scores = wrapper.predict(image, prompt_kwargs)

        if wrapper.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_inference_time += elapsed
        total_images += 1

        prediction = {
            "image_id": int(target["image_id"].item()),
            "masks": pred_masks,
            "scores": pred_scores,
            "labels": target["labels"].tolist()
        }
        evaluator.update([prediction])

    stats, ap_by_class = evaluator.summarize()
    avg_time = total_inference_time / max(total_images, 1)
    return stats, ap_by_class, avg_time


def run_single_evaluation(config, checkpoint_path, label_mode):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordPetDataset(
        root_dir=config["data_dir"],
        split=config.get("eval_split", "test"),
        transform=None,
        label_mode=label_mode
    )

    print("Dataset root:", config["data_dir"])
    print("Label mode:", label_mode)
    print("Valid samples:", len(dataset))
    print("Categories:", dataset.categories[:5], "..." if len(dataset.categories) > 5 else "")


    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=detection_collate_fn
    )

    wrapper = SAMFineTuneWrapper(
        device=device,
        model_name=config["sam_model_name"]
    )

    if checkpoint_path is not None:
        wrapper.load_checkpoint(checkpoint_path)
        print(f"Loaded fine-tuned checkpoint: {checkpoint_path}")
    else:
        print(f"Using base pretrained model: {config['sam_model_name']}")

    stats, ap_by_class, avg_time = evaluate_finetuned_sam(wrapper, dataloader, dataset)

    if stats is None:
        return None

    return {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "ap_by_class": ap_by_class,
        "inference_time_per_image_sec": avg_time,
        "inference_time_per_image_ms": avg_time * 1000.0
    }


def print_results(title, results):
    print(f"\n=== {title} ===")
    print(f"mAP50:      {results['mAP_50']:.4f}")
    print(f"mAP50:95:   {results['mAP_50_95']:.4f}")
    print(f"time/img:   {results['inference_time_per_image_ms']:.2f} ms")
    print("AP by class:")
    for cls_name, ap in results["ap_by_class"].items():
        print(f"  {cls_name}: {ap:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--label_mode", type=str, choices=["species", "breed"], required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    print("\nEvaluating BASE model...")
    base_results = run_single_evaluation(
        config=config,
        checkpoint_path=None,
        label_mode=args.label_mode
    )

    if base_results is not None:
        print_results(f"base pretrained ({args.label_mode})", base_results)

    if args.checkpoint is not None:
        print("\nEvaluating FINE-TUNED model...")
        ft_results = run_single_evaluation(
            config=config,
            checkpoint_path=args.checkpoint,
            label_mode=args.label_mode
        )

        if ft_results is not None:
            print_results(f"fine-tuned ({args.label_mode})", ft_results)

if __name__ == "__main__":
    main()