import torch
import json
import numpy as np
import cv2

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

def generate_random_points(masks_tensor, num_points=1):
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

def generate_sift_points(image_np, masks_tensor, num_points=1):
    """
    Detects SIFT keypoints. If it finds fewer than `num_points` on an object, 
    it pads the rest with random foreground points to prevent Hugging Face tensor crashes.
    """
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
    all_keypoints = sift.detect(gray, None)
    
    input_points = []
    input_labels = []
    for mask in masks_tensor:
        mask_np = mask.cpu().numpy()
        y_indices, x_indices = np.where(mask_np > 0)
        
        if len(y_indices) == 0:
            # Completely empty mask padding
            input_points.append([[0, 0]] * num_points)
            input_labels.append([0] * num_points)
            continue
            
        # Filter keypoints to only keep those INSIDE this specific object's mask
        valid_kps = [kp for kp in all_keypoints if mask_np[int(kp.pt[1]), int(kp.pt[0])] > 0]
        valid_kps = sorted(valid_kps, key=lambda x: x.response, reverse=True)
        
        # Take up to num_points
        points_for_object = [[int(kp.pt[0]), int(kp.pt[1])] for kp in valid_kps[:num_points]]
        labels_for_object = [1] * len(points_for_object)
        
        # FIX: Pad with random foreground points if SIFT didn't find enough
        while len(points_for_object) < num_points:
            idx = np.random.randint(0, len(y_indices))
            points_for_object.append([int(x_indices[idx]), int(y_indices[idx])])
            labels_for_object.append(1)
            
        input_points.append(points_for_object)
        input_labels.append(labels_for_object)
        
    return input_points, input_labels


# def generate_object_grid_points(masks_tensor, grid_points_per_axis=3):
#     """
#     Generates a uniform grid of points. Pads or truncates to ensure exactly 
#     (grid_points_per_axis^2) points are returned per object.
#     """
#     target_num_points = grid_points_per_axis ** 2
#     input_points = []
#     input_labels = []
    
#     for mask in masks_tensor:
#         y_indices, x_indices = torch.where(mask > 0)
        
#         if len(y_indices) == 0:
#             input_points.append([[0, 0]] * target_num_points)
#             input_labels.append([0] * target_num_points)
#             continue
            
#         x_min, x_max = x_indices.min().item(), x_indices.max().item()
#         y_min, y_max = y_indices.min().item(), y_indices.max().item()
        
#         x_steps = np.linspace(x_min, x_max, grid_points_per_axis)
#         y_steps = np.linspace(y_min, y_max, grid_points_per_axis)
        
#         points_for_object = []
#         labels_for_object = []
        
#         for x in x_steps:
#             for y in y_steps:
#                 if mask[int(y), int(x)] > 0:
#                     points_for_object.append([int(x), int(y)])
#                     labels_for_object.append(1)
                    
#         # FIX: Pad or truncate to ensure strict target_num_points length
#         while len(points_for_object) < target_num_points:
#             idx = torch.randint(0, len(y_indices), (1,)).item()
#             points_for_object.append([x_indices[idx].item(), y_indices[idx].item()])
#             labels_for_object.append(1)
            
#         if len(points_for_object) > target_num_points:
#             points_for_object = points_for_object[:target_num_points]
#             labels_for_object = labels_for_object[:target_num_points]
            
#         input_points.append(points_for_object)
#         input_labels.append(labels_for_object)
        
#     return input_points, input_labels


def generate_object_grid_points(masks_tensor, grid_points_per_axis=3):
    """
    Generates a STRICT uniform grid of points across the object's bounding box.
    Points inside the mask are positive (1), points outside are negative (0).
    """
    target_num_points = grid_points_per_axis ** 2
    input_points = []
    input_labels = []
    
    for mask in masks_tensor:
        y_indices, x_indices = torch.where(mask > 0)
        
        if len(y_indices) == 0:
            # Fallback for completely empty masks
            input_points.append([[0, 0]] * target_num_points)
            input_labels.append([0] * target_num_points)
            continue
            
        # Get the tight bounding box of the mask
        x_min, x_max = x_indices.min().item(), x_indices.max().item()
        y_min, y_max = y_indices.min().item(), y_indices.max().item()
        
        # Create the strict geometric grid
        x_steps = np.linspace(x_min, x_max, grid_points_per_axis)
        y_steps = np.linspace(y_min, y_max, grid_points_per_axis)
        
        points_for_object = []
        labels_for_object = []
        
        # Iterate through every intersection of the grid
        for x in x_steps:
            for y in y_steps:
                points_for_object.append([int(x), int(y)])
                
                # YOUR IDEA: Check if the point is actually on the object
                if mask[int(y), int(x)] > 0:
                    labels_for_object.append(1)  # Positive prompt (foreground)
                else:
                    labels_for_object.append(0)  # Negative prompt (background)
                    
        # Because we iterate exactly grid_points_per_axis^2 times,
        # the length is guaranteed to be correct. No random padding needed!
        
        input_points.append(points_for_object)
        input_labels.append(labels_for_object)
        
    return input_points, input_labels

def get_prompts(target, prompt_type="random_point", config=None, image_np=None):
    """Router function to extract different types of prompts for a single image."""
    
    # --- POINT PROMPTS ---
    if prompt_type in ["point", "random_point"]:
        num_pts = config.get("num_points", 1) if config else 1
        points, labels = generate_random_points(target["masks"], num_points=num_pts)
        return {"input_points": [points], "input_labels": [labels]}, target["labels"].tolist()
        
    elif prompt_type == "sift_point":
        if image_np is None:
            raise ValueError("image_np must be provided to use SIFT points.")
        num_pts = config.get("num_points", 1) if config else 1
        points, labels = generate_sift_points(image_np, target["masks"], num_points=num_pts)
        return {"input_points": [points], "input_labels": [labels]}, target["labels"].tolist()
        
    elif prompt_type == "grid_point":
        grid_size = config.get("grid_size", 3) if config else 3
        points, labels = generate_object_grid_points(target["masks"], grid_points_per_axis=grid_size)
        return {"input_points": [points], "input_labels": [labels]}, target["labels"].tolist()
    
    # --- BOX PROMPTS ---
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
            if p.get("score", 1.0) >= threshold:
                x, y, w, h = p["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(p["category_id"])
                
        if len(boxes) == 0:
            return {}, []
            
        return {"input_boxes": [[boxes]]}, labels
        
    return {}, []