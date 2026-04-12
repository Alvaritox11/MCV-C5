# quick_inference.py
import os
import random
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from models import ViTQwen05BModel, ViTQwen15BModel
import evaluate
import argparse

# ── Patch for torch < 2.6 ─────────────────────────────────────────────────────
import transformers.utils.import_utils as _hf_import_utils
_hf_import_utils.check_torch_load_is_safe = lambda: None

meteor_metric = evaluate.load("meteor")
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_meteor(prediction: str, references: list) -> float:
    result = meteor_metric.compute(
        predictions=[prediction],
        references=[references],
    )
    return result["meteor"] * 100


def pick_random_sample(ann_path: str, img_dir: str):
    """Pick a random image + its reference captions from a VizWiz annotation file."""
    with open(ann_path, "r") as f:
        data = json.load(f)

    from collections import defaultdict
    img_to_refs   = defaultdict(list)
    img_id_to_info = {}

    for img in data["images"]:
        img_id_to_info[img["id"]] = img["file_name"]

    for ann in data["annotations"]:
        cap = ann.get("caption", "").strip()
        if cap:
            img_to_refs[ann["image_id"]].append(cap)

    valid_ids  = [k for k, v in img_to_refs.items() if len(v) > 0]
    chosen_id  = random.choice(valid_ids)
    file_name  = img_id_to_info[chosen_id]
    img_path   = os.path.join(img_dir, file_name)
    refs       = img_to_refs[chosen_id]

    return chosen_id, img_path, refs


def load_checkpoint(model, cfg: dict, model_label: str):
    """Load .pth checkpoint into model if checkpoint_path is set in config."""
    ckpt_file = cfg.get("checkpoint_path")
    if ckpt_file:
        ckpt_file = ckpt_file.strip()
    if ckpt_file and os.path.exists(ckpt_file):
        print(f"  Loading checkpoint: {ckpt_file}")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        sd    = state.get("model_state_dict", state)   # handle both formats
        model.load_state_dict(sd)
    else:
        print(f"  [{model_label}] No checkpoint found — using base weights")
    return model


def save_result(output_dir: str, model_name: str, image: Image.Image,
                prediction: str, references: list, meteor: float):
    os.makedirs(output_dir, exist_ok=True)

    # Save raw image
    img_path = os.path.join(output_dir, f"{model_name}_image.png")
    image.save(img_path)

    # Save JSON
    result    = {
        "model":      model_name,
        "prediction": prediction,
        "references": references,
        "meteor":     meteor,
    }
    json_path = os.path.join(output_dir, f"{model_name}_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    # Save figure with overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.axis("off")
    refs_str = "\n".join([f"  • {r}" for r in references[:4]])
    caption  = (
        f"Model: {model_name}\n\n"
        f"Prediction:\n  {prediction}\n\n"
        f"References:\n{refs_str}\n\n"
        f"METEOR: {meteor:.2f}%"
    )
    ax.set_title(caption, fontsize=9, loc="left", wrap=True)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{model_name}_output.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"  Saved → {json_path}")
    print(f"  Saved → {fig_path}")
    return result


# ── Inference: fine-tuned ViT+Qwen models ─────────────────────────────────────
def infer_vit_qwen(model, image: Image.Image, cfg: dict) -> str:
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    pixel_values = transform(image).unsqueeze(0).to(device).to(torch.bfloat16)

    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=cfg.get("max_length", 64),
            num_beams=4,
            repetition_penalty=1.5,       # penalizes repeated tokens
            no_repeat_ngram_size=3,        # forbids repeating any 3-gram

        )

    prediction = model.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    return prediction


# ── Inference: off-the-shelf Qwen2.5-VL-7B ────────────────────────────────────
def infer_qwen7b(model, processor, image: Image.Image, max_length: int = 64) -> str:
    prompt   = "Describe this image concisely."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,
        )

    generated_ids = output[:, inputs["input_ids"].shape[-1]:]
    prediction    = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0].strip()
    return prediction


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_05b", type=str, required=True,
                        help="Config JSON for ViT+Qwen2.5-0.5B model")
    parser.add_argument("--config_15b", type=str, required=True,
                        help="Config JSON for ViT+Qwen2.5-1.5B model")
    parser.add_argument("--ann_path",   type=str, required=True,
                        help="VizWiz annotation JSON (val.json)")
    parser.add_argument("--img_dir",    type=str, required=True,
                        help="VizWiz image directory")
    parser.add_argument("--output_dir", type=str, default="outputs/quick_inference")
    parser.add_argument("--seed",       type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── Pick one random image shared across all models ─────────────────────────
    print("Picking random sample...")
    img_id, img_path, references = pick_random_sample(args.ann_path, args.img_dir)
    image = Image.open(img_path).convert("RGB")
    print(f"  Image ID : {img_id}")
    print(f"  Path     : {img_path}")
    print(f"  Refs     : {references}\n")

    all_results = {}

    # ── Model 1: ViT + Qwen2.5-0.5B ──────────────────────────────────────────
    print("=" * 60)
    print("Loading ViT + Qwen2.5-0.5B ...")
    with open(args.config_05b) as f:
        cfg_05b = json.load(f)

    model_05b = ViTQwen05BModel(cfg_05b).to(device)
    model_05b = load_checkpoint(model_05b, cfg_05b, "Qwen2.5-0.5B")

    pred_05b   = infer_vit_qwen(model_05b, image, cfg_05b)
    meteor_05b = compute_meteor(pred_05b, references)
    print(f"  Prediction : {pred_05b}")
    print(f"  METEOR     : {meteor_05b:.2f}%")
    all_results["qwen_05b"] = save_result(
        args.output_dir, "qwen_05b", image, pred_05b, references, meteor_05b
    )
    del model_05b
    torch.cuda.empty_cache()

    # ── Model 2: ViT + Qwen2.5-1.5B ──────────────────────────────────────────
    print("=" * 60)
    print("Loading ViT + Qwen2.5-1.5B ...")
    with open(args.config_15b) as f:
        cfg_15b = json.load(f)

    model_15b = ViTQwen15BModel(cfg_15b).to(device)
    model_15b = load_checkpoint(model_15b, cfg_15b, "Qwen2.5-1.5B")

    pred_15b   = infer_vit_qwen(model_15b, image, cfg_15b)
    meteor_15b = compute_meteor(pred_15b, references)
    print(f"  Prediction : {pred_15b}")
    print(f"  METEOR     : {meteor_15b:.2f}%")
    all_results["qwen_15b"] = save_result(
        args.output_dir, "qwen_15b", image, pred_15b, references, meteor_15b
    )
    del model_15b
    torch.cuda.empty_cache()

    # ── Model 3: Off-the-shelf Qwen2.5-VL-7B ─────────────────────────────────
    print("=" * 60)
    print("Loading off-the-shelf Qwen2.5-VL-7B-Instruct ...")
    model_7b = AutoModelForImageTextToText.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor_7b = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model_7b.eval()

    pred_7b   = infer_qwen7b(model_7b, processor_7b, image, max_length=64)
    meteor_7b = compute_meteor(pred_7b, references)
    print(f"  Prediction : {pred_7b}")
    print(f"  METEOR     : {meteor_7b:.2f}%")
    all_results["qwen_7b"] = save_result(
        args.output_dir, "qwen_7b", image, pred_7b, references, meteor_7b
    )
    del model_7b
    torch.cuda.empty_cache()

    # ── Summary figure: all 3 side by side ────────────────────────────────────
    print("=" * 60)
    print("Saving summary figure...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    model_keys   = ["qwen_05b",                   "qwen_15b",                   "qwen_7b"]
    model_labels = ["ViT + Qwen2.5-0.5B\n(fine-tuned)",
                    "ViT + Qwen2.5-1.5B\n(fine-tuned)",
                    "Qwen2.5-VL-7B\n(off-the-shelf)"]

    for ax, key, label in zip(axes, model_keys, model_labels):
        r        = all_results[key]
        refs_str = "\n".join([f"• {r_}" for r_ in r["references"][:3]])
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{label}\n\n"
            f"Pred: {r['prediction']}\n\n"
            f"Refs:\n{refs_str}\n\n"
            f"METEOR: {r['meteor']:.2f}%",
            fontsize=8, loc="left",
        )

    plt.suptitle(f"VizWiz Image ID: {img_id}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    summary_path = os.path.join(args.output_dir, "summary.png")
    plt.savefig(summary_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Summary saved → {summary_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary_json = {
        "image_id":   img_id,
        "image_path": img_path,
        "references": references,
        "results": {
            k: {"prediction": v["prediction"], "meteor": v["meteor"]}
            for k, v in all_results.items()
        },
    }
    summary_json_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=4)
    print(f"  Summary JSON → {summary_json_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n===== Final METEOR Scores =====")
    for k, v in all_results.items():
        print(f"  {k:<12}: {v['meteor']:.2f}%")
    print("================================")


if __name__ == "__main__":
    main()