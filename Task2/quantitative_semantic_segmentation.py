import torch
from src.dataset import KittiMotsDataset, detection_collate_fn, instances_to_semantic
import json
from torch.utils.data import DataLoader
from src.models.grounded_sam_wrapper import GroundedSAMWrapper
from src.metrics import SemanticEvaluator
from tqdm import tqdm

def run_quantitative_evaluation(config_path, data_dir):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config['batch_size']
    dataset = KittiMotsDataset(root_dir=data_dir, split="val", return_masks=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=detection_collate_fn)
    
    finetuned_weights = config.get("finetuned_weights", None)
    wrapper = GroundedSAMWrapper(device=device, finetuned_sam_path=finetuned_weights)
    evaluator = SemanticEvaluator(num_classes=4)
    
    for images, targets in tqdm(dataloader, desc="Evaluating Semantic Seg"):
        batch_masks, _, _, batch_labels, _ = wrapper.predict_batch(
            images, 
            text_prompt=config["text_prompt"],
            box_threshold=config.get("box_threshold", 0.3),
            text_threshold=config.get("text_threshold", 0.25)
        )
        
        for i in range(len(images)):
            image_shape = (images[i].height, images[i].width) 
            
            gt_masks = targets[i]["masks"].numpy()
            gt_labels = targets[i]["labels"].numpy()
            gt_semantic = instances_to_semantic(gt_masks, gt_labels, image_shape)
            
            pred_semantic = instances_to_semantic(batch_masks[i], batch_labels[i], image_shape)
            evaluator.update(pred_semantic, gt_semantic)
        
    iou, miou, dice, mdice = evaluator.compute_metrics()
    
    print("\n--- Semantic Segmentation Results ---")
    print(f"mIoU (Pedestrian & Car): {miou:.4f}")
    print(f"mDice (Pedestrian & Car): {mdice:.4f}")
    print(f"IoU Pedestrian: {iou[1]:.4f} | Car: {iou[3]:.4f}")
    print(f"Dice Pedestrian: {dice[1]:.4f} | Car: {dice[3]:.4f}")

if __name__ == "__main__":
    # Example usage:
    configs = "configs/eval_finetuned_sam.json"
    dataset = "/home/mcv/datasets/C5/KITTI-MOTS"
    run_quantitative_evaluation(configs, dataset)
    