import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import json
from src.dataset import KittiMotsDataset
from src.models.grounded_sam_wrapper import GroundedSAMWrapper

def instances_to_semantic(masks, labels, image_shape):
    semantic_mask = np.zeros(image_shape, dtype=np.uint8)
    if len(masks) == 0: 
        return semantic_mask
        
    areas = [np.sum(m) for m in masks]
    sorted_indices = np.argsort(areas)[::-1]
    
    for idx in sorted_indices:
        semantic_mask[masks[idx] > 0] = labels[idx]
        
    return semantic_mask

def plot_side_by_side(image, gt_mask, pred_mask, class_colors, title="", figsize=(16, 8)):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    def apply_overlay(mask):
        overlay = np.zeros_like(img_cv)
        for class_id, color in class_colors.items():
            if class_id == 0: 
                continue
            overlay[mask == class_id] = color[::-1] 
            
        alpha = 0.5
        mask_indices = mask > 0
        blended = img_cv.copy()
        blended[mask_indices] = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)[mask_indices]
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    gt_rgb = apply_overlay(gt_mask)
    pred_rgb = apply_overlay(pred_mask)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].imshow(gt_rgb)
    axes[0].set_title("Ground Truth", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Prediction", fontsize=16)
    axes[1].axis('off')
    
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    
    return fig

def generate_qualitative_plots(config_path, output_dir="output_plots"):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    os.makedirs(output_dir, exist_ok=True)
    dataset = KittiMotsDataset(root_dir=config["data_dir"], split="val", return_masks=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finetuned_weights = config.get("finetuned_weights", None)
    wrapper = GroundedSAMWrapper(device=device, finetuned_sam_path=finetuned_weights)
    
    targets = [
        {"seq": "0013", "frame_id": 22, "title": "Person", "filename": "persona_0013_000022.png"},
        {"seq": "0014", "frame_id": 82, "title": "Crowded Cars", "filename": "coche_0014_000082.png"},
        {"seq": "0016", "frame_id": 89, "title": "Crowded Pedestrians", "filename": "crowded_pedestrians_0016_000089.png"},
        {"seq": "0002", "frame_id": 216, "title": "Small Cars", "filename": "small_cars_0002_000216.png"}
    ]
    
    class_colors = {
        1: (0, 255, 0),
        3: (0, 0, 255)
    }
    
    for target_info in targets:
        sample_idx = next(
            (i for i, sample in enumerate(dataset.samples) 
             if sample["seq"] == target_info["seq"] and sample["frame_id"] == target_info["frame_id"]), 
            None
        )
        
        if sample_idx is None:
            continue
            
        image, target = dataset[sample_idx]
        gt_masks = target["masks"].numpy()
        gt_labels = target["labels"].numpy()
        
        image_shape = (image.height, image.width)
        gt_semantic_mask = instances_to_semantic(gt_masks, gt_labels, image_shape)
        
        pred_masks, _, _, pred_labels, _ = wrapper.predict(
            image, 
            text_prompt=config["text_prompt"],
            box_threshold=config.get("box_threshold", 0.3),
            text_threshold=config.get("text_threshold", 0.25)
        )
        
        pred_semantic_mask = instances_to_semantic(pred_masks, pred_labels, image_shape)
        
        fig = plot_side_by_side(
            image=image, 
            gt_mask=gt_semantic_mask, 
            pred_mask=pred_semantic_mask,
            class_colors=class_colors, 
            title=target_info["title"]
        )
        
        output_path = os.path.join(output_dir, target_info["filename"])
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    # config_file = "configs/eval_grounded_sam.json" 
    # generate_qualitative_plots(config_path=config_file)
    config_file = "configs/eval_finetuned_sam.json" 
    generate_qualitative_plots(config_path=config_file)