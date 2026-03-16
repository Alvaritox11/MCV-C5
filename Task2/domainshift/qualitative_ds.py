import os
import json
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
from tqdm import tqdm

from domainshift.dataset_ds import OxfordPetDataset, detection_collate_fn
from src.models.sam_finetune_wrapper import SAMFineTuneWrapper


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0
    return float(inter / union)


def compute_mask_area_ratio(mask: np.ndarray) -> float:
    return float(mask.sum()) / float(mask.shape[0] * mask.shape[1])


def build_overlay(base_mask: np.ndarray, ft_mask: np.ndarray):
    """
    Yellow -> pretrained only
    Red    -> fine-tuned only
    Orange -> overlap
    """
    h, w = base_mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)

    base_only = np.logical_and(base_mask == 1, ft_mask == 0)
    ft_only = np.logical_and(base_mask == 0, ft_mask == 1)
    both = np.logical_and(base_mask == 1, ft_mask == 1)

    # Yellow: pretrained only
    overlay[base_only] = [1.0, 0.85, 0.0, 0.35]

    # Red: fine-tuned only
    overlay[ft_only] = [1.0, 0.2, 0.2, 0.35]

    # Orange: overlap
    overlay[both] = [1.0, 0.45, 0.0, 0.50]

    return overlay


def save_visualization(
    image,
    gt_mask: np.ndarray,
    base_mask: np.ndarray,
    ft_mask: np.ndarray,
    save_path: str,
    title: str,
    subtitle_lines=None,
):
    if subtitle_lines is None:
        subtitle_lines = []

    overlay = build_overlay(base_mask, ft_mask)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.imshow(overlay)

    # GT contour in lime
    ax.contour(gt_mask.astype(np.float32), levels=[0.5], colors=["lime"], linewidths=2)

    legend_elements = [
        Patch(facecolor=(1.0, 0.85, 0.0, 0.35), edgecolor="none", label="Pretrained only"),
        Patch(facecolor=(1.0, 0.2, 0.2, 0.35), edgecolor="none", label="Fine-tuned only"),
        Patch(facecolor=(1.0, 0.45, 0.0, 0.50), edgecolor="none", label="Overlap"),
        Patch(facecolor="none", edgecolor="lime", linewidth=2, label="GT contour"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95)

    full_title = title
    if subtitle_lines:
        full_title += "\n" + "\n".join(subtitle_lines)

    ax.set_title(full_title, fontsize=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def collect_predictions(base_wrapper, ft_wrapper, dataloader, dataset):
    records = []

    cat_id_to_name = {cat["id"]: cat["name"] for cat in dataset.categories}

    for idx, (images, targets) in enumerate(tqdm(dataloader, desc="Collecting qualitative predictions")):
        image = images[0]
        target = targets[0]

        if len(target["boxes"]) == 0:
            continue

        prompt_kwargs = {
            "input_boxes": [target["boxes"].tolist()]
        }

        base_masks, base_scores = base_wrapper.predict(image, prompt_kwargs)
        if len(base_masks) == 0:
            continue
        base_mask = np.asarray(base_masks[0], dtype=np.uint8)

        ft_masks, ft_scores = ft_wrapper.predict(image, prompt_kwargs)
        if len(ft_masks) == 0:
            continue
        ft_mask = np.asarray(ft_masks[0], dtype=np.uint8)

        gt_mask = target["masks"][0].cpu().numpy().astype(np.uint8)

        base_iou = mask_iou(base_mask, gt_mask)
        ft_iou = mask_iou(ft_mask, gt_mask)

        base_area = compute_mask_area_ratio(base_mask)
        ft_area = compute_mask_area_ratio(ft_mask)
        gt_area = compute_mask_area_ratio(gt_mask)

        sample_meta = dataset.samples[idx]
        image_stem = sample_meta["image_stem"]
        class_id = sample_meta["class_id"]
        class_name = cat_id_to_name.get(class_id, str(class_id))

        records.append({
            "idx": idx,
            "image_stem": image_stem,
            "class_id": class_id,
            "class_name": class_name,
            "base_iou": base_iou,
            "ft_iou": ft_iou,
            "base_score": float(base_scores[0]),
            "ft_score": float(ft_scores[0]),
            "base_area": base_area,
            "ft_area": ft_area,
            "gt_area": gt_area,
            "base_mask": base_mask,
            "ft_mask": ft_mask,
            "gt_mask": gt_mask,
            "image": image,
        })

    return records


def save_iou_csv(records, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_stem",
            "class_name",
            "base_iou",
            "ft_iou",
            "iou_gap_base_minus_ft",
            "base_score",
            "ft_score",
            "base_area_ratio",
            "ft_area_ratio",
            "gt_area_ratio",
        ])

        for r in records:
            writer.writerow([
                r["image_stem"],
                r["class_name"],
                f"{r['base_iou']:.6f}",
                f"{r['ft_iou']:.6f}",
                f"{(r['base_iou'] - r['ft_iou']):.6f}",
                f"{r['base_score']:.6f}",
                f"{r['ft_score']:.6f}",
                f"{r['base_area']:.6f}",
                f"{r['ft_area']:.6f}",
                f"{r['gt_area']:.6f}",
            ])


def save_iou_histogram(records, save_path):
    base_ious = [r["base_iou"] for r in records]
    ft_ious = [r["ft_iou"] for r in records]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(base_ious, bins=25, alpha=0.6, label="Pretrained SAM")
    ax.hist(ft_ious, bins=25, alpha=0.6, label="Fine-tuned SAM")

    ax.set_title("IoU distribution on Oxford-IIIT Pet", fontsize=14)
    ax.set_xlabel("IoU")
    ax.set_ylabel("Number of images")
    ax.legend()
    ax.grid(True, alpha=0.25)

    mean_base = np.mean(base_ious)
    mean_ft = np.mean(ft_ious)

    ax.axvline(mean_base, linestyle="--", linewidth=2, alpha=0.9)
    ax.axvline(mean_ft, linestyle="--", linewidth=2, alpha=0.9)

    ax.text(mean_base + 0.01, ax.get_ylim()[1] * 0.9, f"Base mean: {mean_base:.3f}")
    ax.text(mean_ft + 0.01, ax.get_ylim()[1] * 0.8, f"FT mean: {mean_ft:.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_distinct(records, condition_fn, score_fn, k=3):
    candidates = [r for r in records if condition_fn(r)]
    candidates = sorted(candidates, key=score_fn, reverse=True)

    selected = []
    used = set()

    for r in candidates:
        if r["image_stem"] in used:
            continue
        selected.append(r)
        used.add(r["image_stem"])
        if len(selected) == k:
            break

    return selected


def choose_examples(records):
    # Good: both models strong
    good_examples = select_distinct(
        records,
        condition_fn=lambda r: r["base_iou"] >= 0.85 and r["ft_iou"] >= 0.85,
        score_fn=lambda r: 0.5 * (r["base_iou"] + r["ft_iou"]),
        k=3
    )

    # Bad but informative:
    # - both are low
    # - neither is degenerate zero/background-only
    # - keep around 0.30-0.45 if possible
    bad_examples = select_distinct(
        records,
        condition_fn=lambda r: (
            0.25 <= r["base_iou"] <= 0.45 and
            0.25 <= r["ft_iou"] <= 0.45 and
            r["base_area"] > 0.01 and
            r["ft_area"] > 0.01
        ),
        score_fn=lambda r: -abs(0.35 - 0.5 * (r["base_iou"] + r["ft_iou"])),
        k=3
    )

    # If there are not enough, relax the range a bit
    if len(bad_examples) < 3:
        bad_examples = select_distinct(
            records,
            condition_fn=lambda r: (
                0.20 <= r["base_iou"] <= 0.50 and
                0.20 <= r["ft_iou"] <= 0.50 and
                r["base_area"] > 0.01 and
                r["ft_area"] > 0.01
            ),
            score_fn=lambda r: -abs(0.35 - 0.5 * (r["base_iou"] + r["ft_iou"])),
            k=3
        )

    # Pretrained clearly better than fine-tuned
    pretrained_better = select_distinct(
        records,
        condition_fn=lambda r: (r["base_iou"] - r["ft_iou"]) >= 0.15,
        score_fn=lambda r: (r["base_iou"] - r["ft_iou"]),
        k=3
    )

    return {
        "good_examples": good_examples,
        "bad_examples": bad_examples,
        "pretrained_better_examples": pretrained_better,
    }


def save_selection_summary(selection, save_path):
    out = {}
    for key, recs in selection.items():
        out[key] = []
        for r in recs:
            out[key].append({
                "image_stem": r["image_stem"],
                "class_name": r["class_name"],
                "base_iou": r["base_iou"],
                "ft_iou": r["ft_iou"],
                "base_score": r["base_score"],
                "ft_score": r["ft_score"],
                "base_area": r["base_area"],
                "ft_area": r["ft_area"],
                "gt_area": r["gt_area"],
            })

    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)


def save_examples(example_list, prefix, output_dir):
    for i, rec in enumerate(example_list, start=1):
        save_visualization(
            image=rec["image"],
            gt_mask=rec["gt_mask"],
            base_mask=rec["base_mask"],
            ft_mask=rec["ft_mask"],
            save_path=os.path.join(output_dir, f"{prefix}_{i}.png"),
            title=f"{prefix.replace('_', ' ').title()}: {rec['image_stem']} ({rec['class_name']})",
            subtitle_lines=[
                f"Base IoU: {rec['base_iou']:.3f} | FT IoU: {rec['ft_iou']:.3f}",
                f"Base score: {rec['base_score']:.3f} | FT score: {rec['ft_score']:.3f}",
                f"Base area: {rec['base_area']:.3f} | FT area: {rec['ft_area']:.3f} | GT area: {rec['gt_area']:.3f}",
            ],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results_ds")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OxfordPetDataset(
        root_dir=config["data_dir"],
        split=config.get("eval_split", "test"),
        transform=None,
        label_mode="breed"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=detection_collate_fn
    )

    print(f"Dataset size for qualitative analysis: {len(dataset)}")

    base_wrapper = SAMFineTuneWrapper(
        device=device,
        model_name=config["sam_model_name"]
    )
    print(f"Loaded base model: {config['sam_model_name']}")

    ft_wrapper = SAMFineTuneWrapper(
        device=device,
        model_name=config["sam_model_name"]
    )
    ft_wrapper.load_checkpoint(args.checkpoint)
    print(f"Loaded fine-tuned checkpoint: {args.checkpoint}")

    print("Collecting predictions...")
    records = collect_predictions(base_wrapper, ft_wrapper, dataloader, dataset)
    print(f"Collected {len(records)} records.")

    csv_path = os.path.join(args.output_dir, "iou_distribution.csv")
    hist_path = os.path.join(args.output_dir, "iou_distribution_histogram.png")
    summary_path = os.path.join(args.output_dir, "qualitative_selection_summary.json")

    save_iou_csv(records, csv_path)
    save_iou_histogram(records, hist_path)

    selection = choose_examples(records)
    save_selection_summary(selection, summary_path)

    save_examples(selection["good_examples"], "good_example", args.output_dir)
    save_examples(selection["bad_examples"], "bad_example", args.output_dir)
    save_examples(selection["pretrained_better_examples"], "pretrained_better", args.output_dir)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"- CSV: {csv_path}")
    print(f"- Histogram: {hist_path}")
    print(f"- Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()