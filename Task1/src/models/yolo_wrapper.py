import torch
from ultralytics import YOLO

class YoloWrapper:
    def __init__(self, model_path="yolo26x.pt", device=None):
        """
        Initializes an Ultralytics YOLO model.

        Args:
            model_path (str): Path to a .pt model (e.g. yolov8n.pt, yolo26n.pt, runs/.../best.pt)
            device (str): 'cuda' or 'cpu'. If None, detects automatically.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading YOLO model: {model_path} on {self.device}...")

        self.model = YOLO(model_path)

    def predict(self, images, confidence_threshold=0.25, iou_threshold=0.7, imgsz=640):
        """
        Runs inference on a list of images.

        Args:
            images (list[PIL.Image] | PIL.Image): input images
            confidence_threshold (float): conf threshold
            iou_threshold (float): NMS IoU threshold
            imgsz (int): inference size (Ultralytics will resize)

        Returns:             (just entries of people and cars)
            list of dicts:
            [
              {'boxes': [[x1,y1,x2,y2], ...],
               'scores': [..],
               'labels': [..] }   # COCO IDs if model is COCO-trained
            ]
        """
        # Normalize input to list
        if isinstance(images, tuple):
            images = list(images)
        elif not isinstance(images, list):
            images = [images]

        # Get results
        results = self.model.predict(
            source=images,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )

        standardized_predictions = []
        for r in results:
            # r.boxes is ultralytics.engine.results.Boxes
            if r.boxes is None or len(r.boxes) == 0:
                standardized_predictions.append({"boxes": [], "scores": [], "labels": []})
                continue

            boxes_xyxy = r.boxes.xyxy.detach().cpu().tolist()   # Nx4
            scores = r.boxes.conf.detach().cpu().tolist()       # N
            labels = r.boxes.cls.detach().cpu().tolist()        # N (class indices)


            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []

            for box, score, cls_idx in zip(boxes_xyxy, scores, labels):
                cls_idx = int(cls_idx)
                name = r.names.get(cls_idx, str(cls_idx))

                if name == "person":
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_labels.append(1)   # COCO person
                elif name == "car":
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
                    filtered_labels.append(3)   # COCO car
                # else: ignore other classes

            standardized_predictions.append({
                "boxes": filtered_boxes,
                "scores": filtered_scores,
                "labels": filtered_labels
            })
        return standardized_predictions
