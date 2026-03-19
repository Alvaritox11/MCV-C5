# TO AVOID COMPLICATIONS WITH THE SWEEPS, ANOTHER MAIN SCRIPT

import os
import json
import time
import torch
import wandb
import argparse
import albumentations as A
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

# Import your custom classes
from src.dataset import KittiMotsDataset, detection_collate_fn
from src.metrics import CocoEvaluator
from src.models.frnn_wrapper import FasterRCNNWrapper
from src.models.detr_wrapper import DetrWrapper
from src.models.yolo_wrapper import YoloWrapper

def load_config(config_path="configs/detr_config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_transforms(aug_type, is_train):
    """Returns Albumentations transforms based on the requested strategy."""
    # If eval mode, or aug_type is explicitly none/false, return no transforms
    if not is_train or aug_type == "none" or aug_type is False:
        return None

    # Strategy 1: Basic Spatial and Lighting
    if aug_type == "basic" or aug_type is True:
        return A.Compose([
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.8, 1.2), p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # Strategy 2: Weather & Domain Robustness
    elif aug_type == "weather":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomFog(p=0.3),                 # Simulates foggy dashcam footage
            A.RandomRain(p=0.3),                               # Simulates rain
            A.MotionBlur(blur_limit=5, p=0.3),  # Simulates camera shake/driving speed
            A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.5), # Occlusion
            A.ColorJitter(brightness=0.1, contrast=0.2, p=0.5)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    return None

def train_one_epoch(model_wrapper, dataloader, optimizer, device, epoch):
    model_wrapper.model.train()
    total_loss, total_loss_ce, total_loss_bbox = 0.0, 0.0, 0.0

    start_time = time.time()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for images, targets in progress_bar:
        optimizer.zero_grad()
        
        # ---------------------------------------------------------
        # FASTER R-CNN FORWARD PASS
        # ---------------------------------------------------------
        if isinstance(model_wrapper, FasterRCNNWrapper):
            # Convert PIL to Tensor [C, H, W] for FRCNN training
            tensor_images = [F.to_tensor(img).to(device) for img in images]
            
            # FRCNN strictly expects only 'boxes' and 'labels' in the target dict
            tensor_targets = [
                {k: v.to(device) for k, v in t.items() if k in ['boxes', 'labels']} 
                for t in targets
            ]
            
            loss_dict = model_wrapper.model(tensor_images, tensor_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Extract separated losses for FRCNN
            loss_ce = loss_dict.get('loss_classifier', torch.tensor(0.0)) + loss_dict.get('loss_objectness', torch.tensor(0.0))
            loss_bbox = loss_dict.get('loss_box_reg', torch.tensor(0.0)) + loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0))
            
        # ---------------------------------------------------------
        # DETR FORWARD PASS
        # ---------------------------------------------------------
        elif isinstance(model_wrapper, DetrWrapper):
            labels_list = []
            for tgt in targets:
                h, w = tgt['orig_size']
                boxes = tgt['boxes'] # Current format: xyxy
                
                # DETR strictly expects boxes in [0, 1] normalized cxcywh format
                if len(boxes) > 0:
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / w
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / h
                    bw = (boxes[:, 2] - boxes[:, 0]) / float(w)
                    bh = (boxes[:, 3] - boxes[:, 1]) / float(h)
                    norm_boxes = torch.stack([cx, cy, bw, bh], dim=1)
                else:
                    norm_boxes = torch.empty((0, 4), dtype=torch.float32)

                labels_list.append({
                    "class_labels": tgt['labels'].to(device),
                    "boxes": norm_boxes.to(device)
                })

            # The processor handles creating the pixel_mask for batched padding
            inputs = model_wrapper.processor(images=list(images), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model_wrapper.model(**inputs, labels=labels_list)
            losses = outputs.loss
            
            # Extract separated losses for DETR
            loss_ce = outputs.loss_dict.get('loss_ce', torch.tensor(0.0))
            loss_bbox = outputs.loss_dict.get('loss_bbox', torch.tensor(0.0)) + outputs.loss_dict.get('loss_giou', torch.tensor(0.0))
            
        else:
            raise ValueError("Unsupported model for manual training loop.")

        # Backward & Optimize
        losses.backward()
        
        # Gradient clipping is highly recommended for DETR
        if isinstance(model_wrapper, DetrWrapper):
            torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), max_norm=0.1)
            
        optimizer.step()
        
        total_loss += losses.item()
        total_loss_ce += loss_ce.item()
        total_loss_bbox += loss_bbox.item()
        progress_bar.set_postfix(loss=losses.item())
        # wandb.log({"train/batch_loss": losses.item()})
    
    epoch_time = time.time() - start_time
    return total_loss / len(dataloader), total_loss_ce / len(dataloader), total_loss_bbox / len(dataloader), epoch_time

@torch.inference_mode()
def evaluate(model_wrapper, dataloader, dataset, config):
    if hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'eval'):
        model_wrapper.model.eval()
        
    evaluator = CocoEvaluator(dataset)
    total_images = 0
    inference_time = 0.0
    
    # Variables for DETR Validation Loss
    total_val_loss, total_val_loss_ce, total_val_loss_bbox = 0.0, 0.0, 0.0
    
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        start_inf = time.time()

        # ---------------------------------------------------------
        # DETR: One-Pass Validation (Loss + Predictions)
        # ---------------------------------------------------------
        if config["model_type"] == "detr" and hasattr(model_wrapper, 'processor'):
            labels_list = []
            target_sizes_list = []
            
            for tgt in targets:
                h, w = tgt['orig_size']
                target_sizes_list.append([h, w])
                boxes = tgt['boxes']
                
                # Convert to normalized cxcywh
                if len(boxes) > 0:
                    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / w
                    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / h
                    bw = (boxes[:, 2] - boxes[:, 0]) / float(w)
                    bh = (boxes[:, 3] - boxes[:, 1]) / float(h)
                    norm_boxes = torch.stack([cx, cy, bw, bh], dim=1)
                else:
                    norm_boxes = torch.empty((0, 4), dtype=torch.float32)

                labels_list.append({
                    "class_labels": tgt['labels'].to(model_wrapper.device),
                    "boxes": norm_boxes.to(model_wrapper.device)
                })

            inputs = model_wrapper.processor(images=list(images), return_tensors="pt")
            inputs = {k: v.to(model_wrapper.device) for k, v in inputs.items()}
            
            # --- THE SINGLE FORWARD PASS ---
            outputs = model_wrapper.model(**inputs, labels=labels_list)
            
            # 1. Extract Losses
            total_val_loss += outputs.loss.item()
            total_val_loss_ce += outputs.loss_dict.get('loss_ce', torch.tensor(0.0)).item()
            total_val_loss_bbox += (outputs.loss_dict.get('loss_bbox', torch.tensor(0.0)).item() + 
                                    outputs.loss_dict.get('loss_giou', torch.tensor(0.0)).item())
            
            # 2. Extract Predictions (Replicating the wrapper's logic safely)
            target_sizes = torch.tensor(target_sizes_list).to(model_wrapper.device)
            processed_results = model_wrapper.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=config['confidence_threshold']
            )
            
            predictions = []
            keep_classes = [1, 3] # Person and Car
            for result in processed_results:
                b, s, l = result['boxes'].cpu().tolist(), result['scores'].cpu().tolist(), result['labels'].cpu().tolist()
                filtered = {'boxes': [], 'scores': [], 'labels': []}
                for box, score, label in zip(b, s, l):
                    if label in keep_classes:
                        filtered['boxes'].append(box)
                        filtered['scores'].append(score)
                        filtered['labels'].append(label)
                predictions.append(filtered)

        # ---------------------------------------------------------
        # FASTER R-CNN: Standard prediction (No Loss)
        # ---------------------------------------------------------
        else:
            predictions = model_wrapper.predict(images, confidence_threshold=config['confidence_threshold'])
        
        inference_time += time.time() - start_inf
        total_images += len(images)
        
        # Inject image_id so CocoEvaluator knows which image this belongs to
        for pred, target in zip(predictions, targets):
            pred['image_id'] = target['image_id'].item()
            
        evaluator.update(predictions)
        
    fps = total_images / inference_time if inference_time > 0 else 0
    stats, map_car, map_ped = evaluator.summarize()
    
    # Average the DETR validation losses
    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_val_loss_ce = total_val_loss_ce / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_val_loss_bbox = total_val_loss_bbox / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return stats, map_car, map_ped, fps, avg_val_loss, avg_val_loss_ce, avg_val_loss_bbox

# Helper function to unpack all 12 COCO metrics safely
def get_all_stats(stats, prefix):
    if stats is None: return {}
    return {
        f"{prefix}/mAP_0.50_0.95_all": stats[0],
        f"{prefix}/mAP_0.50_all": stats[1],
        f"{prefix}/mAP_0.75_all": stats[2],
        f"{prefix}/mAP_0.50_0.95_small": stats[3],
        f"{prefix}/mAP_0.50_0.95_medium": stats[4],
        f"{prefix}/mAP_0.50_0.95_large": stats[5],
        f"{prefix}/Recall_max1_all": stats[6],
        f"{prefix}/Recall_max10_all": stats[7],
        f"{prefix}/Recall_max100_all": stats[8],
        f"{prefix}/Recall_max100_small": stats[9],
        f"{prefix}/Recall_max100_medium": stats[10],
        f"{prefix}/Recall_max100_large": stats[11],
    }

def main():
    parser = argparse.ArgumentParser(description="C5 Object Detection Pipeline")
    parser.add_argument('--config', type=str, default='configs/detr_config.json', help='Path to your JSON configuration file')
    args, unknown = parser.parse_known_args()

    torch.manual_seed(42)
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # wandb_run = wandb.init(project=config["wandb_project"], entity=config['wandb_entity'], name=config["wandb_run_name"], config=config)
    wandb_run = wandb.init(project=config["wandb_project"], entity=config['wandb_entity'], config=config)

    # Unpack W&B sweep parameters into our local config
    if "lora_combo" in wandb.config:
        combo = wandb.config["lora_combo"]
        config["lora_r"] = combo["lora_r"]
        config["lora_alpha"] = combo["lora_alpha"]
        config["learning_rate"] = combo["learning_rate"]
        
        # Give it a clean name on the dashboard
        config["wandb_run_name"] = f"lora_{combo['name']}"
    elif "frnn_combo" in wandb.config:
        combo = wandb.config["frnn_combo"]
        config["freeze_base"] = combo["freeze_base"]
        config["use_partial_unfreeze"] = combo["use_partial_unfreeze"]
        config["learning_rate"] = combo["learning_rate"]
        config["wandb_run_name"] = f"frnn_{combo['name']}"

    else:
        for key, value in wandb.config.items():
            config[key] = value

    run_id = wandb_run.id if wandb_run is not None else str(int(time.time()))
    config["wandb_run_name"] = f"lora_sweep_{run_id}" # Give it a unique name
    
    # Setup local results directory appending the Run ID to avoid overwriting
    # Ensure the run name has the ID appended to avoid folder collisions
    if not config["wandb_run_name"].endswith(run_id):
        config["wandb_run_name"] = f"{config['wandb_run_name']}_{run_id}"
        
    folder_name = f"{config['wandb_run_name']}_{run_id}"
    results_dir = os.path.join("results", folder_name)
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f: json.dump([], f)
    
    print(f"--- Starting Pipeline | Mode: {config['mode'].upper()} | Model: {config['model_type']} ---")
    print(f"--- Results will be saved locally to: {results_dir} ---")

    # 1. Dataset & DataLoader initialization
    train_transforms = get_transforms(config.get("apply_augmentations", "none"), is_train=True)
    val_transforms = get_transforms("none", is_train=False) 

    train_dataset = KittiMotsDataset(root_dir=config["data_dir"], split="train", transforms=train_transforms)
    val_dataset = KittiMotsDataset(root_dir=config["data_dir"], split="val", transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=detection_collate_fn)

    # 2. Model Initialization
    if config["model_type"] == "faster_rcnn":
        wrapper = FasterRCNNWrapper(device=device, freeze_base=config.get("freeze_base", False), use_partial_unfreeze=config.get("use_partial_unfreeze", False))
    elif config["model_type"] == "detr":
        wrapper = DetrWrapper(
            device=device, 
            freeze_base=config.get("freeze_base", False), 
            use_lora=config.get("use_lora", False),
            lora_r=config.get("lora_r", 8), 
            lora_alpha=config.get("lora_alpha", 32)
        )
    elif config["model_type"] == "yolo":
        wrapper = YoloWrapper(device=device)
    else:
        raise ValueError("Invalid model_type in config.")
    
    # Calculate system metrics
    trainable_params = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad) if hasattr(wrapper, 'model') else 0
    print(f"Trainable Parameters: {trainable_params:,}")

    best_map = 0.0

    # 3. Execution based on Mode
    if config["mode"] == "evaluate":
        print("Evaluating pre-trained model...")
        stats, map_car, map_ped, fps, val_loss, val_loss_ce, val_loss_bbox = evaluate(wrapper, val_loader, val_dataset, config)
        
        if stats is not None:
            eval_metrics = {
                **get_all_stats(stats, "val"),
                "val/mAP_Car": map_car, 
                "val/mAP_Pedestrian": map_ped, 
                "system/inference_fps": fps
            }
            
            # FIX: Conditionally log DETR validation loss during step 0 evaluation
            if config["model_type"] == "detr":
                eval_metrics.update({
                    "val/total_loss": val_loss,
                    "val/loss_ce": val_loss_ce,
                    "val/loss_bbox": val_loss_bbox
                })
                
            wandb.log(eval_metrics)
            with open(metrics_file, 'w') as f: json.dump([eval_metrics], f, indent=4)
    elif config["mode"] == "train":
        if config["model_type"] == "yolo":
            raise RuntimeError("Train YOLO using Ultralytics CLI, not this script! Use this script only for YOLO evaluation.")
            
        params = [p for p in wrapper.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])

        for epoch in range(1, config["epochs"] + 1):
            avg_loss, loss_ce, loss_bbox, epoch_time = train_one_epoch(wrapper, train_loader, optimizer, device, epoch)
            
            # print(f"Evaluating Training Set Epoch {epoch}...")
            # train_stats, train_map_car, train_map_ped, _, _, _, _ = evaluate(wrapper, train_loader, train_dataset, config)

            print(f"Validating Epoch {epoch}...")
            val_stats, val_map_car, val_map_ped, fps, val_loss, val_loss_ce, val_loss_bbox = evaluate(wrapper, val_loader, val_dataset, config)

            val_mAP = val_stats[0] if val_stats is not None else 0.0
            max_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            
            # Combine all metrics into one dictionary
            epoch_metrics = {
                "epoch": epoch,
                "train/total_loss": avg_loss,
                "train/loss_ce": loss_ce,
                "train/loss_bbox": loss_bbox,

                # **get_all_stats(train_stats, "train"),
                # # "train/mAP_0.5_0.95": train_mAP,
                # # "train/mAP_0.5": train_stats[1] if train_stats is not None else 0.0,     
                # # "train/Recall_AR100": train_stats[8] if train_stats is not None else 0.0,
                # # "train/mAP_small": train_stats[3] if train_stats is not None else 0.0,
                # # "train/mAP_medium": train_stats[4] if train_stats is not None else 0.0,
                # # "train/mAP_large": train_stats[5] if train_stats is not None else 0.0,
                # "train/mAP_Car": train_map_car,
                # "train/mAP_Pedestrian": train_map_ped,

                **get_all_stats(val_stats, "val"),
                # "val/mAP_0.5_0.95": mAP,
                # "val/mAP_small": stats[3] if stats is not None else 0.0,
                # "val/mAP_medium": stats[4] if stats is not None else 0.0,
                # "val/mAP_large": stats[5] if stats is not None else 0.0,
                "val/mAP_Car": val_map_car,
                "val/mAP_Pedestrian": val_map_ped,

                "system/epoch_time_sec": epoch_time,
                "system/inference_fps": fps,
                "system/max_gpu_mem_MB": max_mem,
                "system/trainable_params": trainable_params
            }

            if config["model_type"] == "detr":
                epoch_metrics.update({
                    "val/total_loss": val_loss,
                    "val/loss_ce": val_loss_ce,
                    "val/loss_bbox": val_loss_bbox
                })
            
            wandb.log(epoch_metrics)
            
            # Append to local JSON
            with open(metrics_file, 'r') as f: all_metrics = json.load(f)
            all_metrics.append(epoch_metrics)
            with open(metrics_file, 'w') as f: json.dump(all_metrics, f, indent=4)
            
            # Checkpoint saving
            if val_mAP > best_map:
                best_map = val_mAP
                checkpoint_path = os.path.join(results_dir, "best_model.pt")
                torch.save(wrapper.model.state_dict(), checkpoint_path)
                print(f"--> Saved new Best Model to {checkpoint_path} (mAP: {best_map:.4f})")

    wandb.finish()

if __name__ == "__main__":
    main()