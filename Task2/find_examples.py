import os
import torch
import numpy as np
import json
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset import KittiMotsDataset, detection_collate_fn
from src.models.grounded_sam_wrapper import GroundedSAMWrapper

def show_box(box, ax, edgecolor='green'):
    """Plots a bounding box."""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))


def plot_and_save_comparison(image, basic_boxes, basic_texts, rich_boxes, rich_texts, save_path, title=""):
    """Saves the Basic vs Rich bounding box comparison to disk."""
    plt.figure(figsize=(16, 6))
    if title:
        plt.suptitle(title, fontsize=20, fontweight='bold', y=1.02)
        
    # --- PANEL 1: BASIC PROMPT ---
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Basic Prompt: 'car . person .'\nTotal Objects: {len(basic_boxes)}", fontsize=16)
    plt.axis('off')
    ax1 = plt.gca()
    
    for i in range(len(basic_boxes)):
        show_box(basic_boxes[i], ax1, edgecolor='cyan')
        ax1.text(basic_boxes[i][0], basic_boxes[i][1] - 5, str(basic_texts[i]), 
                 color='cyan', fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    # --- PANEL 2: RICH PROMPT ---
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title(f"Rich Prompt (Expanded Vocab)\nTotal Objects: {len(rich_boxes)}", fontsize=16)
    plt.axis('off')
    ax2 = plt.gca()
    
    for i in range(len(rich_boxes)):
        text = str(rich_texts[i]).lower().strip()
        color = 'orange' if text not in ["car", "person"] else 'cyan'
        
        show_box(rich_boxes[i], ax2, edgecolor=color)
        ax2.text(rich_boxes[i][0], rich_boxes[i][1] - 5, text, 
                 color=color, fontsize=12, fontweight='bold', 
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close() 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = "vocabulary_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Dataset and Grounded SAM...")
    val_dataset = KittiMotsDataset(
        root_dir="/home/mcv/datasets/C5/KITTI-MOTS/", # UPDATE THIS PATH IF NEEDED
        split="val",
        return_masks=False 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=True, 
        collate_fn=detection_collate_fn
    )
    
    wrapper = GroundedSAMWrapper(device=device)

    basic_prompt = "car . person ."
    rich_prompt = "car . suv . pickup truck . child . person ."

    threshold_box = 0.3
    threshold_text = 0.25
    
    num_samples_to_check = 100 
    print(f"Scanning {num_samples_to_check} RANDOM images to compare vocabularies...")
    
    discrepancies = []

    for step, (images, targets) in enumerate(tqdm(val_loader, total=num_samples_to_check)):
        if step >= num_samples_to_check:
            break 
            
        image = images[0]
        target = targets[0]
        original_idx = target["image_id"].item()
        
        # Run Basic Prompt
        _, _, basic_boxes, _, basic_texts = wrapper.predict(
            image, text_prompt=basic_prompt, box_threshold=threshold_box, text_threshold=threshold_text
        )
        
        # Run Rich Prompt
        _, _, rich_boxes, _, rich_texts = wrapper.predict(
            image, text_prompt=rich_prompt, box_threshold=threshold_box, text_threshold=threshold_text
        )
        
        basic_count = len(basic_boxes) if basic_boxes else 0
        rich_count = len(rich_boxes) if rich_boxes else 0
        
        diff = abs(rich_count - basic_count)
        
        if diff > 0:
            discrepancies.append({
                "image_idx": original_idx,
                "basic_count": basic_count,
                "rich_count": rich_count,
                "diff": diff,
                "basic_boxes": basic_boxes,
                "basic_texts": basic_texts,
                "rich_boxes": rich_boxes,
                "rich_texts": rich_texts,
                "image": image 
            })

    discrepancies.sort(key=lambda x: x["diff"], reverse=True)
    
    # ---------------------------------------------------------
    # NEW: Save all discrepancy data to a JSON file
    # ---------------------------------------------------------
    json_ready_data = []
    for d in discrepancies:
        json_ready_data.append({
            "image_idx": int(d["image_idx"]),
            "basic_count": int(d["basic_count"]),
            "rich_count": int(d["rich_count"]),
            "diff": int(d["diff"]),
            "basic_texts": [str(t) for t in d["basic_texts"]],
            "rich_texts": [str(t) for t in d["rich_texts"]],
            # Convert numpy/tensor boxes to standard Python floats for JSON
            "basic_boxes": [[float(coord) for coord in box] for box in d["basic_boxes"]],
            "rich_boxes": [[float(coord) for coord in box] for box in d["rich_boxes"]]
        })

    json_path = os.path.join(output_dir, "discrepancies_log.json")
    with open(json_path, "w") as f:
        json.dump(json_ready_data, f, indent=4)
    print(f"\nSaved raw data log to: {json_path}")
    # ---------------------------------------------------------

    top_n = min(15, len(discrepancies))
    print(f"Saving the Top {top_n} visual comparisons...")
    
    for i in tqdm(range(top_n)):
        d = discrepancies[i]
        idx = d["image_idx"]
        save_path = os.path.join(output_dir, f"vocab_diff_rank{i+1}_img{idx}.png")
        
        title = f"Vocabulary Impact | Image {idx} | Difference: {d['diff']} objects"
        plot_and_save_comparison(
            d["image"], d["basic_boxes"], d["basic_texts"], 
            d["rich_boxes"], d["rich_texts"], save_path, title
        )
        
    print(f"\nDone! Check the '{output_dir}' folder for your presentation images and JSON log.")

if __name__ == "__main__":
    main()