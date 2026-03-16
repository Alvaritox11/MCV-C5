import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import your Week 1 classes 
from src.dataset import KittiMotsDataset, detection_collate_fn
from src.models.yolo_wrapper import YoloWrapper

def generate_predictions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Paths (Update these!)
    data_dir = "/home/mcv/datasets/C5/KITTI-MOTS/"
    checkpoint_path = "/ghome/group05/gerard/MCV-C5/Task1/runs/GRID_freeze20_strong_wd1e-3_img1024/weights/best.pt"
    output_json_path = "week1_yolo_predictions.json"
    
    # 2. Dataset
    # Make sure to use the validation split without augmentations
    val_dataset = KittiMotsDataset(root_dir=data_dir, split="val", transforms=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=detection_collate_fn)
    
    # 3. Load the YOLO Model
    print(f"Loading Week 1 YOLO model from {checkpoint_path}...")
    wrapper = YoloWrapper(model_path=checkpoint_path, device=str(device))
    
    # 4. Run Inference and Save
    all_predictions = []
    
    # We use a very low confidence threshold (e.g., 0.1) to save all potential boxes.
    # You will filter out the bad ones later using "custom_box_threshold" in your Week 2 config!
    confidence_threshold = 0.1 

    for images, targets in tqdm(val_loader, desc="Generating YOLO Predictions"):
        image = images[0]
        target = targets[0]
        img_id = target['image_id'].item()
        
        # The wrapper returns a list of dicts. Since batch_size=1, we take index 0.
        predictions = wrapper.predict([image], confidence_threshold=confidence_threshold)[0]
        
        b = predictions['boxes']
        s = predictions['scores']
        l = predictions['labels']
        
        # Format to COCO JSON: [x_min, y_min, width, height]
        for box, score, label in zip(b, s, l):
            # YoloWrapper returns xyxy format
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            all_predictions.append({
                "image_id": img_id,
                "category_id": label,  # Already mapped to 1 (person) or 3 (car) by the wrapper
                "bbox": [x_min, y_min, width, height],
                "score": score
            })
                
    # 5. Save to JSON
    print(f"Saving {len(all_predictions)} predictions to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(all_predictions, f, indent=4)
        
    print(f"Done! Move '{output_json_path}' to your Week 2 folder and use it for Task C.")

if __name__ == "__main__":
    generate_predictions()