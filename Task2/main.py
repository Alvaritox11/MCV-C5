import json
import time
import torch
import wandb
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, instances_to_semantic

from src.dataset import KittiMotsDataset, detection_collate_fn
from src.metrics import CocoEvaluator, SemanticEvaluator
from src.prompts import get_prompts
from src.models.sam_wrapper import SAMWrapper
from src.models.grounded_sam_wrapper import GroundedSAMWrapper
from src.visualize import create_wandb_image

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

@torch.inference_mode()
def evaluate_sam(model_wrapper, dataloader, dataset, config):
    evaluator = CocoEvaluator(dataset)
    log_interval = config.get("log_image_interval", 50) 
    
    for step, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating SAM")):
        image = images[0]
        target = targets[0]
        
        # 1. Prediction routing
        if config["prompt_type"] == "text":
            # Grounded SAM handles text internally and outputs predicted labels
            pred_masks, pred_scores, pred_boxes, pred_labels, pred_text_labels = model_wrapper.predict(
                image, 
                text_prompt=config["text_prompt"],
                box_threshold=config.get("box_threshold", 0.3),
                text_threshold=config.get("text_threshold", 0.25)
            )
            pts = None
            lbls = None
        else:
            # Base SAM (Task A) uses GT prompts and GT labels
            # if len(target["labels"]) == 0:
            #     continue
            prompt_kwargs, extracted_labels = get_prompts(target, prompt_type=config["prompt_type"], config=config)
            
            if not prompt_kwargs: # Skip if no prompts were generated (e.g. no valid week 1 predictions for this image)
                continue
                
            pred_masks, pred_scores = model_wrapper.predict(image, prompt_kwargs)
            
            # Use the labels we extracted 
            pred_labels = extracted_labels
            
            # FIX: Safely unpack the deeply nested boxes for the W&B visualizer
            pred_boxes = prompt_kwargs.get("input_boxes", None)
            if pred_boxes is not None:
                pred_boxes = pred_boxes[0][0] # Go two levels deep to get the flat list of boxes!
                
            pts = prompt_kwargs.get("input_points", [None])[0]
            lbls = prompt_kwargs.get("input_labels", [None])[0]

        # Filter out "0" labels (unrecognized by Grounding DINO)
        valid_indices = [i for i, lbl in enumerate(pred_labels) if lbl != 0]
        
        if len(valid_indices) == 0:
            continue

        pred_masks = [pred_masks[i] for i in valid_indices]
        pred_scores = [pred_scores[i] for i in valid_indices]
        pred_labels = [pred_labels[i] for i in valid_indices]
        if pred_boxes: pred_boxes = [pred_boxes[i] for i in valid_indices]

        # 2. Log qualitative results to W&B
        if step % log_interval == 0:
            wandb_img = create_wandb_image(
                image=image, masks=pred_masks, points=pts, labels=lbls, boxes=pred_boxes, 
                title=f"Step {step} | Prompt: {config['prompt_type']}"
            )
            wandb.log({"qualitative_results/predictions": wandb_img, "step": step})

        # 3. Format for Evaluator
        prediction = {
            "image_id": target['image_id'].item(),
            "masks": pred_masks, 
            "scores": pred_scores,
            "labels": pred_labels
        }
            
        evaluator.update([prediction])
        
    stats, map_car, map_ped = evaluator.summarize() 
    return stats, map_car, 

@torch.inference_mode()
def evaluate_semantic_sam(model_wrapper, dataloader, config, image_shape=(375, 1242)):
    evaluator = SemanticEvaluator(num_classes=4) 
    
    for images, targets in dataloader:
        image = images[0]
        target = targets[0]
        
        gt_masks = target["masks"].numpy()
        gt_labels = target["labels"].numpy()
        gt_semantic = instances_to_semantic(gt_masks, gt_labels, image_shape)
        
        pred_masks, _, _, pred_labels, _ = model_wrapper.predict(
            image, 
            text_prompt=config["text_prompt"],
            box_threshold=config.get("box_threshold", 0.3),
            text_threshold=config.get("text_threshold", 0.25)
        )
        
        pred_semantic = instances_to_semantic(pred_masks, pred_labels, image_shape)
        evaluator.update(pred_semantic, gt_semantic)
        
    iou_per_class, miou = evaluator.compute_iou()
    return iou_per_class, miou

def main():
    parser = argparse.ArgumentParser(description="C5 Object Segmentation Pipeline")
    parser.add_argument('--config', type=str, default='configs/eval_sam_points.json')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- W&B Initialization ---
    wandb.init(
        project=config["wandb_project"], 
        entity=config.get("wandb_entity", None), 
        name=config["wandb_run_name"], 
        config=config
    )
    
    print(f"--- Starting Pipeline | Mode: {config['mode'].upper()} | Prompt: {config['prompt_type'].upper()} ---")

    val_dataset = KittiMotsDataset(root_dir=config["data_dir"], split="val", return_masks=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=detection_collate_fn)

    if config["model_type"] == "sam":
        wrapper = SAMWrapper(device=device)
    elif config["model_type"] == "grounded_sam":
        wrapper = GroundedSAMWrapper(device=device)
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")
        
    if config["mode"] == "evaluate":
        stats, map_car, map_ped = evaluate_sam(wrapper, val_loader, val_dataset, config)
        
        if stats is not None:
            # Extract standard COCO metrics
            mAP_05_095 = stats[0]
            mAP_05 = stats[1]
            
            # --- W&B Logging ---
            wandb.log({
                "val/mAP_0.50_0.95": mAP_05_095,
                "val/mAP_0.50": mAP_05,
                "val/mAP_Car": map_car,
                "val/mAP_Pedestrian": map_ped
            })
            
            print("\n--- Evaluation Results ---")
            print(f"mAP (0.50:0.95): {mAP_05_095:.4f}")
            print(f"mAP Car:         {map_car:.4f}")
            print(f"mAP Pedestrian:  {map_ped:.4f}")
            
    wandb.finish()

if __name__ == "__main__":
    main()