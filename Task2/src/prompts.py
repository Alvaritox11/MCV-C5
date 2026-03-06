import torch
import json

# Cache to avoid reloading the JSON file for every single image
_cached_predictions = None

def load_and_cache_predictions(json_path):
    """Loads a COCO-style prediction JSON and organizes it by image_id."""
    global _cached_predictions
    if _cached_predictions is None:
        print(f"Loading Week 1 predictions from {json_path}...")
        with open(json_path, 'r') as f:
            raw_preds = json.load(f)
            
        _cached_predictions = {}
        for p in raw_preds:
            img_id = p["image_id"]
            if img_id not in _cached_predictions:
                _cached_predictions[img_id] = []
            _cached_predictions[img_id].append(p)
    return _cached_predictions


def generate_point_prompts(masks_tensor, num_points=1):
    """
    Given a tensor of binary masks [N, H, W], extracts random foreground points.
    Returns format expected by HF SamProcessor: [[[x, y]], ...] and [[1], ...]
    """
    input_points = []
    input_labels = []
    
    for mask in masks_tensor:
        # Find all coordinates where the mask is positive (foreground)
        y_indices, x_indices = torch.where(mask > 0)
        
        if len(y_indices) == 0:
            # Fallback if mask is empty
            input_points.append([[0, 0]])
            input_labels.append([0]) 
            continue
            
        # Randomly sample 'num_points' from the foreground
        chosen_idxs = torch.randint(low=0, high=len(y_indices), size=(num_points,))
        
        points_for_object = []
        labels_for_object = []
        
        for idx in chosen_idxs:
            x, y = x_indices[idx].item(), y_indices[idx].item()
            points_for_object.append([x, y])
            labels_for_object.append(1) # 1 indicates foreground point
            
        input_points.append(points_for_object)
        input_labels.append(labels_for_object)
        
    return input_points, input_labels

def get_prompts(target, prompt_type="point", config=None):
    """Router function to extract different types of prompts for a single image."""
    
    if prompt_type == "point":
        points, labels = generate_point_prompts(target["masks"])
        return {"input_points": [points], "input_labels": [labels]}, target["labels"].tolist()
    
    elif prompt_type == "bbox":
        # Extracts ground truth boxes for baseline testing
        boxes = target["boxes"].tolist()
        return {"input_boxes": [[boxes]]}, target["labels"].tolist()
    
    elif prompt_type == "custom_bbox":
        if config is None or "week1_predictions_path" not in config:
            raise ValueError("week1_predictions_path must be provided in config for custom_bbox")
            
        preds_dict = load_and_cache_predictions(config["week1_predictions_path"])

        img_id = target["image_id"].item()
        img_preds = preds_dict.get(img_id, [])

        boxes = []
        labels = []
        threshold = config.get("custom_box_threshold", 0.5)
        
        for p in img_preds:
            # Only use bounding boxes that our Week 1 model was confident about
            if p.get("score", 1.0) >= threshold:
                # Assuming Week 1 saved boxes in COCO format: [x, y, w, h]
                x, y, w, h = p["bbox"]
                # SAM expects [xmin, ymin, xmax, ymax]
                boxes.append([x, y, x + w, y + h])
                labels.append(p["category_id"])
                
        if len(boxes) == 0:
            return {}, []
            
        return {"input_boxes": [[boxes]]}, labels
        
    return {}, []