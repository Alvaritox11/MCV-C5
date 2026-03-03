import argparse
import os
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import torch
import torchvision
from PIL import Image
from src.models.detr_wrapper import DetrWrapper
from src.models.frnn_wrapper import FasterRCNNWrapper
from src.models.yolo_wrapper import YoloWrapper
from torchvision.transforms import functional as F
from transformers import DetrForObjectDetection, DetrImageProcessor
from typing import List, Tuple
import xml.etree.ElementTree as ET

KITTI = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='detr',
                        choices=['yolo', 'detr', 'fasterrcnn'],
                        help='Model framework to use')
    parser.add_argument('--data_root', type=str, default='/home/mcv/datasets/C5/KITTI-MOTS/',
                        help='Path to KITTI-MOTS dataset')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to load')
    parser.add_argument('--images_paths', type=str, nargs='+', help='Set of images like 0000/0000.png')
    parser.add_argument('--device', type=str, default=None)

    return parser.parse_args()

def parse_txt_annotations(txt_path, original_frame_id):
        """
        Parses KITTI-MOTS txt format:
        frame_id track_id class_id img_height img_width rle_mask
        """
        annotations = {}
        if not os.path.exists(txt_path):
            return annotations
            
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                frame_id = int(parts[0])
                if int(original_frame_id) == int(frame_id):
                    class_id = int(parts[2])
                    rle = parts[5]
                    height = int(parts[3])
                    width = int(parts[4])
                
                    # We only care about Car (1) and Pedestrian (2)
                    if class_id not in [1, 2]:
                        continue

                    if frame_id not in annotations:
                        annotations[frame_id] = []
                
                    # Decode RLE to BBox using pycocotools [cite: 385]
                    rle_obj = {'counts': rle, 'size': [height, width]}
                    bbox = mask_utils.toBbox(rle_obj) # returns [x, y, w, h]
                    
                    annotations[frame_id].append({
                        'bbox': bbox, # xywh
                        'label': class_id
                    })
        return annotations

def parse_xml(xml_path: str, frame_id: int = 0) -> dict:
    """
    Parses a Pascal VOC XML file.
    Returns same format as parse_txt_annotations: {frame_id: [{'bbox': [x,y,w,h], 'label': int}]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {frame_id: []}

    for obj in root.findall("object"):
        name = obj.find("name").text
        print(f"Found object: {name}")
        if name != "person":
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert xyxy -> xywh to match txt format
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            annotations[frame_id].append({
                'bbox': bbox,  # xywh
                'label': 1     # pedestrian/person = 2, matching KITTI-MOTS class_id
            })
        else:
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert xyxy -> xywh to match txt format
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

            annotations[frame_id].append({
                'bbox': bbox,  # xywh
                'label': 2     # pedestrian/person = 2, matching KITTI-MOTS class_id
            })

    return annotations

def load_samples(images, data_root): #[0000/0000.png]
        samples = []
        if KITTI:
            dataset_root = os.path.join(data_root, "training", "image_02")
            label_dir = os.path.join(data_root, 'instances_txt')
        else:
            dataset_root = os.path.join(data_root, "images")
            label_dir = os.path.join(data_root, 'annots_pub')

        for image in images:
            image = Path(image)

            if KITTI:
                frame_id = int(image.stem)
                folder = image.parent.name
                anno_path = os.path.join(label_dir, f"{folder}.txt")
                annos_by_frame = parse_txt_annotations(anno_path, frame_id)
                samples.append(
                    {
                        "path": os.path.join(dataset_root, image),
                        "seq": str(folder),
                        "frame_id": frame_id,
                        "annos": annos_by_frame.get(frame_id,[])
                    }
                )
            else:
                frame_id = image.stem
                anno_path = os.path.join(label_dir, f"{frame_id}.xml")
                annos_by_frame = parse_xml(anno_path, frame_id)
                print('annos by', annos_by_frame)
                samples.append(
                    {
                        "path": os.path.join(dataset_root, image),
                        "seq": str(frame_id),
                        "frame_id": int(frame_id),
                        "annos": annos_by_frame.get(frame_id,[])
                    }
                )
        return samples

def get_model(model_type, device, checkpoint_path):
    if 'detr' in model_type:
        if checkpoint_path:
            return DetrWrapper(device=device, checkpoint_path=checkpoint_path)
        else:
            return DetrWrapper(device=device)
    elif 'yolo' in model_type:
        if checkpoint_path:
            return YoloWrapper(device=device, model_path=checkpoint_path)
        else:
            return YoloWrapper(device=device,)
    elif 'fasterrcnn' in model_type:
        if checkpoint_path:
            return FasterRCNNWrapper(device=device, checkpoint_path=checkpoint_path)
        else:
            return FasterRCNNWrapper(device=device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

def standerize_output(outputs, confidence_threshold = 0.9):
    keep_classes=(1, 3)
    keep_classes = tuple(keep_classes)
    keep_classes_t = torch.tensor(keep_classes, dtype=torch.int64)
    standardized = []
    for out in outputs:
        boxes  = torch.tensor(out["boxes"])
        scores = torch.tensor(out["scores"])
        labels = torch.tensor(out["labels"])


        # score filter
        keep = scores >= confidence_threshold

        # class filter (person=1, car=3)
        try:
            keep = keep & torch.isin(labels, keep_classes_t)
        except AttributeError:
            # fallback if torch.isin not available
            keep2 = labels.new_zeros(labels.shape, dtype=torch.bool)
            for c in keep_classes:
                keep2 |= (labels == c)
            keep = keep & keep2

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        standardized.append(
            {
                "boxes": boxes.detach().cpu().tolist(),
                "scores": scores.detach().cpu().tolist(),
                "labels": labels.detach().cpu().tolist(),
            }
        )

    return standardized

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)

def box_iou_xyxy(a, b):
    """
    a: (N,4) xyxy
    b: (M,4) xyxy
    returns IoU matrix (N,M)
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    # areas
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)

    # intersection
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter_w = (xx2 - xx1).clip(0)
    inter_h = (yy2 - yy1).clip(0)
    inter = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def save_boxes_on_image(pred, gt, save_path,
                        score_thr=None, class_names=None,
                        iou_thr=0.7):
    img_path = Path(gt['image_path'])
    img = Image.open(img_path).convert("RGB")
    filename = img_path.name
    save_file_path = os.path.join(save_path, filename)

    # --- Convert to numpy ---
    pred_boxes = np.asarray(pred["boxes"], dtype=np.float32)
    pred_labels = np.asarray(pred["labels"], dtype=np.int64)
    pred_scores = np.asarray(pred.get("scores", np.ones(len(pred_boxes))), dtype=np.float32)

    gt_boxes = np.asarray(gt["boxes"], dtype=np.float32)
    gt_labels = np.asarray(gt["labels"], dtype=np.int64)

    # Optional score filter (apply once, early)
    if score_thr is not None:
        keep = pred_scores >= float(score_thr)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis("off")

    # -------------------------
    # Matching: same class AND IoU >= iou_thr
    # One-to-one greedy matching by IoU (highest first)
    # -------------------------
    matched_pred = set()
    matched_gt = set()
    matches = []  # list of (p_idx, g_idx, iou)

    if len(pred_boxes) and len(gt_boxes):
        iou = box_iou_xyxy(pred_boxes, gt_boxes)  # (P, G)

        # candidate pairs: same class and IoU >= thr
        cand = []
        for p in range(len(pred_boxes)):
            for g in range(len(gt_boxes)):
                if pred_labels[p] == gt_labels[g] and iou[p, g] >= iou_thr:
                    print(pred_labels[p], gt_labels[g])
                    cand.append((p, g, float(iou[p, g])))

        # sort best IoU first
        cand.sort(key=lambda x: x[2], reverse=True)

        # greedy assign (1 pred ↔ 1 gt)
        for p, g, v in cand:
            if p in matched_pred or g in matched_gt:
                continue
            matched_pred.add(p)
            matched_gt.add(g)
            matches.append((p, g, v))

    # --- Draw GT (green), except those matched (we'll show match in purple) ---
    for g_idx, (box, cls) in enumerate(zip(gt_boxes, gt_labels)):
        if g_idx in matched_gt:
            continue
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor="green", facecolor="none")
        ax.add_patch(rect)

        cls_text = class_names[cls] if class_names else str(int(cls))
        ax.text(x1, y1, f"GT:{cls_text}", color="green", fontsize=10,
                verticalalignment="top",
                bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))

    # --- Draw unmatched predictions (red) ---
    for p_idx, (box, cls, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        if p_idx in matched_pred:
            continue

        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                 edgecolor="red", facecolor="none")
        ax.add_patch(rect)

        cls_text = class_names[cls] if class_names else str(int(cls))
        ax.text(x1, y1, f"P:{cls_text} ({float(score):.2f})", color="red", fontsize=10,
                verticalalignment="bottom",
                bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))

    # --- Draw matches (purple) ---
    for p_idx, g_idx, v in matches:
        # draw the prediction box as the "correct detection"
        box = pred_boxes[p_idx]
        cls = int(pred_labels[p_idx])
        score = float(pred_scores[p_idx])

        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1

        rect = patches.Rectangle((x1, y1), w, h, linewidth=2.5,
                                 edgecolor="purple", facecolor="none")
        ax.add_patch(rect)

        cls_text = class_names[cls] if class_names else str(cls)
        ax.text(x1, y1,
                f"OK:{cls_text} IoU={v:.2f} ({score:.2f})",
                color="purple", fontsize=10,
                verticalalignment="bottom",
                bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))

    Path(save_path).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_file_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)



def draw_pred_boxes(images_paths, model_type, device, checkpoint_path, output_path):
    
    if KITTI:
        data_root = "/home/mcv/datasets/C5/KITTI-MOTS/"
    else:
        data_root = "/ghome/group05/c5_data"
    samples = load_samples(images_paths, data_root)
    if KITTI:
        class_map = {
                    1: 3,  
                    2: 1   
                }
    else:
        class_map = {
                    1: 3,  
                    2:  1  
                }
    targets = []
    images = []
    print(samples, 'Samples')
    for i,sample in enumerate(samples):
        image_path = sample['path']
        image = Image.open(sample['path']).convert("RGB") 
        w, h = image.size
        images.append(image)

        boxes = []
        labels = []

        for anno in sample['annos']:
                # Convert xywh to xyxy for standard detection format
                print(anno['bbox'])
                x, y, w_box, h_box = anno['bbox']
                boxes.append([x, y, x + w_box, y + h_box])
                
                labels.append(class_map.get(anno['label']))

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor(int(Path(image_path).stem)),
            'orig_size': torch.as_tensor([h, w]),
            'seq': sample['seq'], 
            'image_path': image_path
        }
        targets.append(target)

    model = get_model(model_type, device, checkpoint_path)

    outputs = model.predict(images)
    standerized = standerize_output(outputs)
    for pred, target in zip(standerized, targets):
        save_boxes_on_image(pred, target, output_path)