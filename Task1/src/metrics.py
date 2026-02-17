import torch
import json
import copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoEvaluator:
    def __init__(self, dataset):
        """
        dataset: The KittiMotsDataset instance. We need it to build the Ground Truth COCO object.
        """
        self.dataset = dataset
        self.coco_gt = self._build_coco_gt(dataset)
        self.results = []

    def _build_coco_gt(self, dataset):
        """
        Converts the custom KittiMotsDataset into a standard COCO dictionary
        so pycocotools can use it as Ground Truth.
        """
        print("Building COCO Ground Truth object...")
        coco_dict = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 3, "name": "car"}
            ]
        }
        
        anno_id = 1
        for i in range(len(dataset)):
            # We assume dataset[i] returns (image, target_dict)
            # We access the raw sample directly to avoid loading the image (slow)
            sample = dataset.samples[i]
            img_id = i  # Use index as ID
            
            # Add Image Info
            # Note: We need dimensions. If loading images is too slow, 
            # hardcode 375x1242 (standard Kitti) or read from cached metadata.
            # Here we just use the target's orig_size from the dataset __getitem__ 
            # but that requires loading. For speed, let's trust the dataset.
            # (In a perfect world, we'd cache image sizes in dataset.py).
            # For now, we will just use the annotations to build the GT.
            
            coco_dict["images"].append({
                "id": img_id,
                "width": 1242, # Approximate KITTI width, strictly needed only for some metrics
                "height": 375, # Approximate KITTI height
                "file_name": sample['path']
            })

            # Add Annotations
            for anno in sample['annos']:
                # anno['bbox'] is xywh
                # anno['label'] is KITTI ID (1=Car, 2=Ped) -> Map to COCO (3=Car, 1=Person)
                cat_id = dataset.class_map[anno['label']]
                
                coco_dict["annotations"].append({
                    "id": anno_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": anno['bbox'], # xywh is correct for COCO
                    "area": anno['bbox'][2] * anno['bbox'][3],
                    "iscrowd": 0
                })
                anno_id += 1

        # Load into COCO api
        coco_gt = COCO()
        coco_gt.dataset = coco_dict
        coco_gt.createIndex()
        return coco_gt

    def update(self, predictions):
        """
        accumulate predictions from a batch.
        predictions: list of dicts {'boxes': xyxy, 'scores': s, 'labels': l}
        """
        self.results.extend(predictions)

    def synchronize_between_processes(self):
        # For single GPU/CPU this is pass
        pass

    def accumulate(self):
        # For standard cocoeval this is handled in summarize
        pass

    def summarize(self):
        """
        Run the official COCO evaluation.
        """
        print(f"Evaluating on {len(self.results)} predictions...")
        
        if not self.results:
            print("No predictions found!")
            return

        # Convert our "Standard format" (xyxy) to COCO format (xywh)
        coco_results = []
        for res in self.results:
            img_id = res['image_id'] # Added this field in main.py
            
            for box, score, label in zip(res['boxes'], res['scores'], res['labels']):
                # box is [x1, y1, x2, y2] -> convert to [x, y, w, h]
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                coco_results.append({
                    "image_id": img_id,
                    "category_id": label,
                    "bbox": [x1, y1, w, h],
                    "score": score
                })

        if not coco_results:
             print("No valid predictions to evaluate.")
             return

        # Load results into COCO API
        coco_dt = self.coco_gt.loadRes(coco_results)
        
        # Run Evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats 