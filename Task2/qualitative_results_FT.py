import os
import re
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.dataset import KittiMotsDataset
from src.models.sam_finetune_wrapper import SAMFineTuneWrapper


TRACK_ID_KEYS = [
    "track_id", "instance_id", "inst_id", "obj_id", "object_id", "id", "instance"
]


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def image_to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    if torch.is_tensor(image):
        x = image.detach().cpu()
        if x.ndim == 3 and x.shape[0] in [1, 3]:
            x = x.permute(1, 2, 0).numpy()
        else:
            x = x.numpy()

        if x.dtype != np.uint8:
            if x.max() <= 1.0:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)

        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        return x

    arr = np.array(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def to_uint8_mask(mask):
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8)
    return mask


def mask_iou(pred, gt):
    pred = to_uint8_mask(pred).astype(bool)
    gt = to_uint8_mask(gt).astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def extract_track_id(anno):
    for k in TRACK_ID_KEYS:
        if k in anno:
            return str(anno[k])
    return None


def normalize_seq_frame(text):
    text = text.replace("\\", "/").strip()
    m = re.search(r"(\d{4})/(\d{6})", text)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return text


def sample_seq_frame(sample):
    """
    Tries to recover '0013/000022' from dataset sample metadata/path.
    """
    candidate_strings = []

    for key in ["path", "image_path", "file_name", "img_path"]:
        if key in sample:
            candidate_strings.append(str(sample[key]))

    for s in candidate_strings:
        s = s.replace("\\", "/")
        m = re.search(r"(\d{4})/(\d{6})(?:\.[A-Za-z0-9]+)?$", s)
        if m:
            return f"{m.group(1)}/{m.group(2)}"

    for s in candidate_strings:
        s = s.replace("\\", "/")
        m = re.search(r"(\d{4})/(\d{6})", s)
        if m:
            return f"{m.group(1)}/{m.group(2)}"

    return str(sample.get("path", "unknown"))


def parse_exact_specs(specs_list):
    parsed = []
    seen = set()

    for s in specs_list:
        s = s.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            continue

        track_id = parts[0].strip()
        seq_frame = normalize_seq_frame(parts[1].strip())

        key = (track_id, seq_frame)
        if key not in seen:
            seen.add(key)
            parsed.append({"track_id": track_id, "seq_frame": seq_frame})

    return parsed


def add_top_banner(img_np, text, banner_h=36):
    h, w = img_np.shape[:2]
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8) + 18
    out = np.concatenate([banner, img_np], axis=0)
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    draw.text((10, 10), text, fill=(255, 255, 255))
    return np.array(pil)


def blend_mask(canvas, mask, color, alpha):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return canvas
    color = np.asarray(color, dtype=np.float32)
    canvas = canvas.astype(np.float32)
    canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * color
    return canvas.astype(np.uint8)


def draw_box_and_label(img_np, box, text=None, color=(0, 255, 255), width=3):
    pil = Image.fromarray(img_np)
    draw = ImageDraw.Draw(pil)
    x1, y1, x2, y2 = [float(v) for v in box]
    for i in range(width):
        draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
    if text:
        tx = max(2, int(x1))
        ty = max(2, int(y1) - 16)
        draw.text((tx, ty), text, fill=color)
    return np.array(pil)


def make_overlay_single(image, gt_mask, pred_mask, box, title, out_path):
    img = image_to_numpy(image).copy()
    img = blend_mask(img, gt_mask, color=(0, 255, 0), alpha=0.35)
    img = blend_mask(img, pred_mask, color=(255, 0, 0), alpha=0.35)
    img = draw_box_and_label(img, box, color=(0, 255, 255), text=None)
    img = add_top_banner(img, title)
    Image.fromarray(img).save(out_path)


def make_overlay_all_instances(
    image,
    gt_masks,
    pred_masks,
    boxes,
    focus_idx,
    title,
    out_path
):
    img = image_to_numpy(image).copy()

    for i, (gt, pred, box) in enumerate(zip(gt_masks, pred_masks, boxes)):
        if i == focus_idx:
            gt_alpha = 0.38
            pred_alpha = 0.38
            box_color = (0, 255, 255)
        else:
            gt_alpha = 0.18
            pred_alpha = 0.18
            box_color = (140, 140, 140)

        img = blend_mask(img, gt, color=(0, 255, 0), alpha=gt_alpha)
        img = blend_mask(img, pred, color=(255, 0, 0), alpha=pred_alpha)
        img = draw_box_and_label(img, box, color=box_color, text=None, width=2)

    img = add_top_banner(img, title)
    Image.fromarray(img).save(out_path)


@torch.inference_mode()
def predict_for_target(wrapper, image, target):
    if len(target["boxes"]) == 0:
        return [], []

    prompt_kwargs = {"input_boxes": [[target["boxes"].tolist()]]}
    pred_masks, pred_scores = wrapper.predict(image, prompt_kwargs)
    return pred_masks, pred_scores


def select_unique_by_frame(records, k):
    out = []
    seen = set()
    for r in records:
        frame_key = r["seq_frame"]
        if frame_key in seen:
            continue
        seen.add(frame_key)
        out.append(r)
        if len(out) == k:
            break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output_dir", type=str, default="results_ft")
    parser.add_argument("--poor_threshold", type=float, default=0.5)

    # New arguments
    parser.add_argument("--small_improvement_only", action="store_true")
    parser.add_argument("--small_improvement_target", type=float, default=0.20)
    parser.add_argument("--small_improvement_tol", type=float, default=0.08)
    parser.add_argument("--small_improvement_max", type=int, default=10)

    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["checkpoint_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = config.get("sam_model_name", "facebook/sam-vit-base")

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "better_finetune"))
    ensure_dir(os.path.join(args.output_dir, "poor_both"))
    ensure_dir(os.path.join(args.output_dir, "small_improvement"))
    ensure_dir(os.path.join(args.output_dir, "exact_instances"))
    ensure_dir(os.path.join(args.output_dir, "exact_instances", "isolated"))
    ensure_dir(os.path.join(args.output_dir, "exact_instances", "all_instances"))

    exact_specs_raw = [
        "2009 0013/000022",
        "2409 0014/000082",
        "2522 0016/000089",
        "2409 0014/000082",
        "216 0002/000216",
    ]
    exact_specs = parse_exact_specs(exact_specs_raw)

    print(f"Using device: {device}")
    print(f"Loading dataset split='{args.split}' from: {config['data_dir']}")

    dataset = KittiMotsDataset(
        root_dir=config["data_dir"],
        split=args.split,
        return_masks=True,
        transform=None
    )

    print(f"Loading baseline model: {model_name}")
    baseline = SAMFineTuneWrapper(device=device, model_name=model_name)

    print(f"Loading fine-tuned model: {checkpoint_path}")
    finetuned = SAMFineTuneWrapper(device=device, model_name=model_name)
    finetuned.load_checkpoint(checkpoint_path)

    records = []
    exact_found = OrderedDict(((x["track_id"], x["seq_frame"]), None) for x in exact_specs)

    for sample_idx in range(len(dataset)):
        image, target = dataset[sample_idx]
        sample_meta = dataset.samples[sample_idx]
        seq_frame = sample_seq_frame(sample_meta)

        if len(target["boxes"]) == 0 or len(target["masks"]) == 0:
            continue

        gt_masks = [to_uint8_mask(m) for m in target["masks"]]
        boxes = target["boxes"].detach().cpu().numpy()

        base_masks, _ = predict_for_target(baseline, image, target)
        ft_masks, _ = predict_for_target(finetuned, image, target)

        if len(base_masks) != len(gt_masks) or len(ft_masks) != len(gt_masks):
            print(
                f"[WARN] Sample {seq_frame}: prediction count mismatch "
                f"(gt={len(gt_masks)}, base={len(base_masks)}, ft={len(ft_masks)}). Skipping."
            )
            continue

        annos = sample_meta.get("annos", [])
        if len(annos) != len(gt_masks):
            pass

        for inst_idx, (gt, base_pred, ft_pred, box) in enumerate(zip(gt_masks, base_masks, ft_masks, boxes)):
            base_iou = mask_iou(base_pred, gt)
            ft_iou = mask_iou(ft_pred, gt)

            anno = annos[inst_idx] if inst_idx < len(annos) else {}
            track_id = extract_track_id(anno)

            rec = {
                "sample_idx": sample_idx,
                "instance_idx": inst_idx,
                "seq_frame": seq_frame,
                "track_id": track_id,
                "base_iou": base_iou,
                "ft_iou": ft_iou,
                "delta": ft_iou - base_iou,
            }
            records.append(rec)

            key = (str(track_id), seq_frame)
            if key in exact_found and exact_found[key] is None:
                exact_found[key] = rec

    print(f"Collected {len(records)} instance-level comparisons.")

    better_candidates = [r for r in records if r["delta"] > 0.0]
    better_candidates = sorted(
        better_candidates,
        key=lambda x: (x["delta"], x["ft_iou"], -x["base_iou"]),
        reverse=True
    )
    better_examples = select_unique_by_frame(better_candidates, 5)

    poor_candidates = [
        r for r in records
        if max(r["base_iou"], r["ft_iou"]) <= args.poor_threshold
    ]
    poor_candidates = sorted(
        poor_candidates,
        key=lambda x: (max(x["base_iou"], x["ft_iou"]), (x["base_iou"] + x["ft_iou"]) / 2.0)
    )
    poor_examples = select_unique_by_frame(poor_candidates, 5)

    # New: small-improvement examples around target delta (default 0.20 ± 0.08)
    lo = args.small_improvement_target - args.small_improvement_tol
    hi = args.small_improvement_target + args.small_improvement_tol

    small_improvement_candidates = [
        r for r in records
        if lo <= r["delta"] <= hi
    ]

    # prioritize deltas closest to target
    small_improvement_candidates = sorted(
        small_improvement_candidates,
        key=lambda x: (abs(x["delta"] - args.small_improvement_target), -x["ft_iou"])
    )
    small_improvement_examples = select_unique_by_frame(
        small_improvement_candidates,
        args.small_improvement_max
    )

    summary = {
        "config": args.config,
        "checkpoint": checkpoint_path,
        "split": args.split,
        "model_name": model_name,
        "poor_threshold": args.poor_threshold,
        "small_improvement_target": args.small_improvement_target,
        "small_improvement_tol": args.small_improvement_tol,
        "small_improvement_range": [lo, hi],
        "num_records": len(records),
        "better_examples": better_examples,
        "poor_examples": poor_examples,
        "small_improvement_examples": small_improvement_examples,
        "exact_instances": {},
    }

    def save_pair_for_record(rec, out_dir, prefix):
        image, target = dataset[rec["sample_idx"]]
        gt_masks = [to_uint8_mask(m) for m in target["masks"]]
        boxes = target["boxes"].detach().cpu().numpy()

        base_masks, _ = predict_for_target(baseline, image, target)
        ft_masks, _ = predict_for_target(finetuned, image, target)

        i = rec["instance_idx"]
        seq_frame = rec["seq_frame"]
        track_id = rec["track_id"] if rec["track_id"] is not None else f"idx{i}"

        base_name = f"{prefix}__{seq_frame.replace('/', '_')}__track_{track_id}"

        base_title = (
            f"BASELINE | {seq_frame} | track={track_id} | "
            f"IoU={rec['base_iou']:.4f}"
        )
        ft_title = (
            f"FINETUNED | {seq_frame} | track={track_id} | "
            f"IoU={rec['ft_iou']:.4f} | delta={rec['delta']:+.4f}"
        )

        make_overlay_single(
            image=image,
            gt_mask=gt_masks[i],
            pred_mask=base_masks[i],
            box=boxes[i],
            title=base_title,
            out_path=os.path.join(out_dir, f"{base_name}__baseline.png")
        )
        make_overlay_single(
            image=image,
            gt_mask=gt_masks[i],
            pred_mask=ft_masks[i],
            box=boxes[i],
            title=ft_title,
            out_path=os.path.join(out_dir, f"{base_name}__finetuned.png")
        )

    # New: small improvement only mode
    if args.small_improvement_only:
        print("Saving only small-improvement examples...")
        for idx, rec in enumerate(small_improvement_examples, start=1):
            save_pair_for_record(
                rec,
                os.path.join(args.output_dir, "small_improvement"),
                f"{idx:02d}"
            )

        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print("\nDone.")
        print(f"Saved {len(small_improvement_examples)} small-improvement examples in:")
        print(os.path.join(args.output_dir, "small_improvement"))
        return

    print("Saving top 5 better-finetune examples...")
    for idx, rec in enumerate(better_examples, start=1):
        save_pair_for_record(rec, os.path.join(args.output_dir, "better_finetune"), f"{idx:02d}")

    print("Saving top 5 poor-both examples...")
    for idx, rec in enumerate(poor_examples, start=1):
        save_pair_for_record(rec, os.path.join(args.output_dir, "poor_both"), f"{idx:02d}")

    print("Saving small-improvement examples...")
    for idx, rec in enumerate(small_improvement_examples, start=1):
        save_pair_for_record(
            rec,
            os.path.join(args.output_dir, "small_improvement"),
            f"{idx:02d}"
        )

    print("Saving exact instance requests...")
    not_found = []
    for spec in exact_specs:
        key = (spec["track_id"], spec["seq_frame"])
        rec = exact_found.get(key, None)

        if rec is None:
            not_found.append(f"{spec['track_id']} {spec['seq_frame']}")
            summary["exact_instances"][f"{spec['track_id']} {spec['seq_frame']}"] = "NOT_FOUND"
            continue

        image, target = dataset[rec["sample_idx"]]
        gt_masks = [to_uint8_mask(m) for m in target["masks"]]
        boxes = target["boxes"].detach().cpu().numpy()

        base_masks, _ = predict_for_target(baseline, image, target)
        ft_masks, _ = predict_for_target(finetuned, image, target)

        i = rec["instance_idx"]
        seq_frame = rec["seq_frame"]
        track_id = rec["track_id"] if rec["track_id"] is not None else spec["track_id"]

        tag = f"{seq_frame.replace('/', '_')}__track_{track_id}"

        make_overlay_single(
            image=image,
            gt_mask=gt_masks[i],
            pred_mask=base_masks[i],
            box=boxes[i],
            title=f"BASELINE | isolated | {seq_frame} | track={track_id} | IoU={rec['base_iou']:.4f}",
            out_path=os.path.join(
                args.output_dir, "exact_instances", "isolated",
                f"{tag}__baseline_isolated.png"
            )
        )
        make_overlay_single(
            image=image,
            gt_mask=gt_masks[i],
            pred_mask=ft_masks[i],
            box=boxes[i],
            title=f"FINETUNED | isolated | {seq_frame} | track={track_id} | IoU={rec['ft_iou']:.4f}",
            out_path=os.path.join(
                args.output_dir, "exact_instances", "isolated",
                f"{tag}__finetuned_isolated.png"
            )
        )

        make_overlay_all_instances(
            image=image,
            gt_masks=gt_masks,
            pred_masks=base_masks,
            boxes=boxes,
            focus_idx=i,
            title=f"BASELINE | all instances | {seq_frame} | focus track={track_id}",
            out_path=os.path.join(
                args.output_dir, "exact_instances", "all_instances",
                f"{tag}__baseline_all_instances.png"
            )
        )
        make_overlay_all_instances(
            image=image,
            gt_masks=gt_masks,
            pred_masks=ft_masks,
            boxes=boxes,
            focus_idx=i,
            title=f"FINETUNED | all instances | {seq_frame} | focus track={track_id}",
            out_path=os.path.join(
                args.output_dir, "exact_instances", "all_instances",
                f"{tag}__finetuned_all_instances.png"
            )
        )

        summary["exact_instances"][f"{track_id} {seq_frame}"] = {
            "sample_idx": rec["sample_idx"],
            "instance_idx": rec["instance_idx"],
            "base_iou": rec["base_iou"],
            "ft_iou": rec["ft_iou"],
            "delta": rec["delta"],
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if not_found:
        with open(os.path.join(args.output_dir, "exact_instances", "not_found.txt"), "w") as f:
            for x in not_found:
                f.write(x + "\n")
        print("\n[WARN] Some exact instances were not found:")
        for x in not_found:
            print("  -", x)

    print("\nDone.")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()