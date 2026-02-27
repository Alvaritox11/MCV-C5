import json
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
from src.models.rt_detr_wrapper import Rt_DetrWrapper
def load_config(config_path="configs/detr_config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_transforms(apply_aug, is_train):
    """Returns Albumentations transforms formatted for object detection."""
    if is_train and apply_aug:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            # Add more Albumentations here if needed
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    return None

def train_one_epoch(model_wrapper, dataloader, optimizer, device, epoch):
    model_wrapper.model.train()
    total_loss = 0.0
    
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
            
        # ---------------------------------------------------------
        # DETR FORWARD PASS
        # ---------------------------------------------------------
        elif isinstance(model_wrapper, DetrWrapper) or isinstance(model_wrapper, Rt_DetrWrapper):
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
        
        else:
            raise ValueError("Unsupported model for manual training loop.")

        # Backward & Optimize
        losses.backward()
        
        # Gradient clipping is highly recommended for DETR
        if isinstance(model_wrapper, DetrWrapper):
            torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), max_norm=0.1)
            
        optimizer.step()
        
        total_loss += losses.item()
        progress_bar.set_postfix(loss=losses.item())
        wandb.log({"train/batch_loss": losses.item()})

    return total_loss / len(dataloader)

@torch.inference_mode()
def evaluate(model_wrapper, dataloader, dataset, config):
    if hasattr(model_wrapper, 'model') and hasattr(model_wrapper.model, 'eval'):
        model_wrapper.model.eval()
        
    evaluator = CocoEvaluator(dataset)
    
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        # All wrappers take a list of PIL images and output standardized dicts
        predictions = model_wrapper.predict(images, confidence_threshold=config['confidence_threshold'])
        
        # Inject image_id so CocoEvaluator knows which image this belongs to
        for pred, target in zip(predictions, targets):
            pred['image_id'] = target['image_id'].item()
            
        evaluator.update(predictions)
        
    stats = evaluator.summarize()
    return stats

def main():
    parser = argparse.ArgumentParser(description="C5 Object Detection Pipeline")
    parser.add_argument('--config', type=str, default='configs/detr_config.json', help='Path to your JSON configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wandb.init(project=config["wandb_project"], entity=config['wandb_entity'], name=config["wandb_run_name"], config=config)
    print(f"--- Starting Pipeline | Mode: {config['mode'].upper()} | Model: {config['model_type']} ---")

    # 1. Dataset & DataLoader initialization
    train_transforms = get_transforms(config["apply_augmentations"], is_train=True)
    val_transforms = get_transforms(config["apply_augmentations"], is_train=False)

    train_dataset = KittiMotsDataset(root_dir=config["data_dir"], split="train", transforms=train_transforms)
    val_dataset = KittiMotsDataset(root_dir=config["data_dir"], split="val", transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=detection_collate_fn)

    # 2. Model Initialization
    if config["model_type"] == "faster_rcnn":
        wrapper = FasterRCNNWrapper(device=device, freeze_base=config.get("freeze_base", False))
    elif config["model_type"] == "detr":
        wrapper = DetrWrapper(device=device, freeze_base=config.get("freeze_base", False))
    elif config["model_type"] == "yolo":
        wrapper = YoloWrapper(device=device)
    elif config["model_type"] == "rt_detr":
        print('aaaa')
        wrapper = Rt_DetrWrapper(device=device, freeze_base=config.get("freeze_base", False))
    else:
        raise ValueError("Invalid model_type in config.")

    # 3. Execution based on Mode
    if config["mode"] == "evaluate":
        # TASKS C & D: Evaluate pre-trained models
        print("Evaluating pre-trained model...")
        val_stats = evaluate(wrapper, val_loader, val_dataset, config)
        
        mAP = val_stats[0] if val_stats is not None else 0.0
        wandb.log({"val/mAP_0.5_0.95": mAP})
        print(f"Evaluation Complete. mAP (0.5:0.95): {mAP:.4f}")
        
    elif config["mode"] == "train":
        # TASKS E & F: Fine-tune models
        if config["model_type"] == "yolo":
            raise RuntimeError("Train YOLO using Ultralytics CLI, not this script! Use this script only for YOLO evaluation.")
            
        params = [p for p in wrapper.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])

        for epoch in range(1, config["epochs"] + 1):
            avg_train_loss = train_one_epoch(wrapper, train_loader, optimizer, device, epoch)
            
            print(f"Validating Epoch {epoch}...")
            val_stats = evaluate(wrapper, val_loader, val_dataset, config)
            mAP = val_stats[0] if val_stats is not None else 0.0
            
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "val/mAP_0.5_0.95": mAP
            })
            
            # Note: You can add code here to save model checkpoints (e.g., torch.save)

    wandb.finish()

if __name__ == "__main__":
    main()