import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image
from torch.utils.data import Dataset


class KittiMotsDataset(Dataset):
    """
    KITTI-MOTS -> TorchVision detection-style dataset.

    Returns:
        image: torch.FloatTensor[C,H,W] if you use torchvision transforms / ToTensor,
               or PIL.Image if you don't.
        target: dict with keys:
            - boxes: FloatTensor[N,4] in xyxy
            - labels: Int64Tensor[N]
            - image_id: Int64Tensor[1]
            - area: FloatTensor[N]
            - iscrowd: UInt8Tensor[N] (all zeros)
            - orig_size: Int64Tensor[2] (h, w)
            - size: Int64Tensor[2] (h, w)  # can be updated by some transform pipelines
            - (optional) masks: UInt8Tensor[N,H,W] if return_masks=True
            - (optional) seq: str
            - (optional) frame_id: int

    Notes:
      - KITTI-MOTS txt lines are typically:
        frame_id track_id class_id img_height img_width rle...
        The RLE portion can contain spaces -> we join parts[5:].
      - If you apply geometric transforms, you MUST transform boxes (and masks if used).
        This class supports either:
          A) Albumentations (pass `transforms=` as an albumentations.Compose with bbox_params)
          B) TorchVision v2 transforms (pass a callable that accepts (image, target) and returns both)
          C) Basic torchvision transforms that only take PIL (then boxes won't be changed!)
             -> only safe if you do no geometry (e.g., just ToTensor/Normalize).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        return_masks: bool = False,
        include_seq_in_target: bool = False,
        class_map: Optional[Dict[int, int]] = None,
        seq_ids_train: Optional[List[str]] = None,
        seq_ids_val: Optional[List[str]] = None,
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.return_masks = return_masks
        self.include_seq_in_target = include_seq_in_target

        # Default mapping: KITTI-MOTS: 1=Car, 2=Pedestrian
        # Use your own if you want contiguous labels (e.g. {1:1,2:2})
        self.class_map = class_map or {1: 3, 2: 1}  # Car->COCO car(3), Pedestrian->COCO person(1)

        # Dataset structure (common):
        #   root/training/image_02/<seq>/*.png
        #   root/instances_txt/<seq>.txt
        self.img_dir = os.path.join(root_dir, "training", "image_02")
        self.label_dir = os.path.join(root_dir, "instances_txt")

        # Default split lists (you can override via seq_ids_train/seq_ids_val)
        default_train = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
        default_val = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]

        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        self.seq_ids = (seq_ids_train or default_train) if split == "train" else (seq_ids_val or default_val)

        # Load all samples (each sample = 1 image frame + its annotations)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for seq in self.seq_ids:
            seq_path = os.path.join(self.img_dir, seq)
            if not os.path.exists(seq_path):
                continue

            anno_path = os.path.join(self.label_dir, f"{seq}.txt")
            annos_by_frame = self._parse_txt_annotations(anno_path)

            frames = sorted([f for f in os.listdir(seq_path) if f.endswith(".png")])
            for frame_file in frames:
                frame_id = int(os.path.splitext(frame_file)[0])
                samples.append(
                    {
                        "path": os.path.join(seq_path, frame_file),
                        "seq": seq,
                        "frame_id": frame_id,
                        "annos": annos_by_frame.get(frame_id, []),
                    }
                )

        return samples

    def _parse_txt_annotations(self, txt_path: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        Parse KITTI-MOTS instances_txt format:
          frame_id track_id class_id img_height img_width rle...

        Returns dict: frame_id -> list of objects:
          {
            "bbox_xywh": [x,y,w,h],
            "label": int,  # KITTI label (1 or 2)
            "rle": {"counts": bytes, "size":[h,w]},
            "track_id": int
          }
        """
        annotations: Dict[int, List[Dict[str, Any]]] = {}
        if not os.path.exists(txt_path):
            return annotations

        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()  # robust against variable whitespace
                if len(parts) < 6:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])
                class_id = int(parts[2])
                height = int(parts[3])
                width = int(parts[4])

                # Only keep Car (1) and Pedestrian (2) unless user gave a different class_map
                if class_id not in self.class_map:
                    continue

                # RLE may contain spaces -> join the rest of the line
                rle_str = " ".join(parts[5:])
                rle_obj = {"counts": rle_str.encode("utf-8"), "size": [height, width]}

                # bbox is xywh
                bbox = mask_utils.toBbox(rle_obj).tolist()

                annotations.setdefault(frame_id, []).append(
                    {
                        "bbox": bbox,
                        "label": class_id,
                        "rle": rle_obj,
                        "track_id": track_id,
                    }
                )

        return annotations

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        sample = self.samples[idx]

        image = Image.open(sample["path"]).convert("RGB")
        img_w, img_h = image.size

        boxes_xyxy: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []

        # Build boxes/labels (and optional masks)
        for anno in sample["annos"]:
            x, y, w_box, h_box = anno["bbox"]

            # Filter invalid / tiny boxes
            if w_box <= 1 or h_box <= 1:
                continue

            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w_box)
            y2 = float(y + h_box)

            # Clamp to image bounds (optional but safer)
            x1 = max(0.0, min(x1, img_w - 1.0))
            y1 = max(0.0, min(y1, img_h - 1.0))
            x2 = max(0.0, min(x2, img_w * 1.0))
            y2 = max(0.0, min(y2, img_h * 1.0))

            # Ensure valid after clamp
            if x2 <= x1 or y2 <= y1:
                continue

            mapped_label = self.class_map.get(anno["label"], None)
            if mapped_label is None:
                continue

            boxes_xyxy.append([x1, y1, x2, y2])
            labels.append(int(mapped_label))

            if self.return_masks:
                # Decode RLE to binary mask
                m = mask_utils.decode(anno["rle"])  # HxW uint8 {0,1}
                if m is None:
                    continue
                masks.append(m.astype(np.uint8))

        boxes_t = torch.as_tensor(boxes_xyxy, dtype=torch.float32)
        # --- FIX: Ensure 2D shape for empty boxes --- PARCHEE
        if boxes_t.numel() == 0:
            boxes_t = torch.empty((0, 4), dtype=torch.float32)
        # --------------------------------------------
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "orig_size": torch.tensor([img_h, img_w], dtype=torch.int64),
            "size": torch.tensor([img_h, img_w], dtype=torch.int64),
        }

        # Optional common fields for TorchVision utilities/eval
        if boxes_t.numel() > 0:
            area = (boxes_t[:, 2] - boxes_t[:, 0]).clamp(min=0) * (boxes_t[:, 3] - boxes_t[:, 1]).clamp(min=0)
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((labels_t.shape[0],), dtype=torch.uint8)

        if self.return_masks:
            if len(masks) > 0:
                masks_t = torch.from_numpy(np.stack(masks, axis=0))  # [N,H,W] uint8
            else:
                masks_t = torch.zeros((0, img_h, img_w), dtype=torch.uint8)
            target["masks"] = masks_t

        if self.include_seq_in_target:
            target["seq"] = sample["seq"]
            target["frame_id"] = sample["frame_id"]

        if self.transforms is not None:
            # Case A: Albumentations-style (expects numpy image + bboxes)
            # We detect this by checking for a callable with attribute 'bbox_params' or by try/except.
            # If you use Albumentations, pass a Compose with bbox_params=format='pascal_voc'
            try:
                # If it's Albumentations, it usually wants numpy image
                img_np = np.array(image)  # HWC RGB

                if self.return_masks:
                    # Albumentations supports masks list
                    transformed = self.transforms(
                        image=img_np,
                        bboxes=boxes_xyxy,
                        labels=labels,
                        masks=masks if len(masks) > 0 else [],
                    )
                    img_np = transformed["image"]
                    new_boxes = transformed["bboxes"]
                    new_labels = transformed["labels"]
                    new_masks = transformed.get("masks", [])
                else:
                    transformed = self.transforms(image=img_np, bboxes=boxes_xyxy, labels=labels)
                    img_np = transformed["image"]
                    new_boxes = transformed["bboxes"]
                    new_labels = transformed["labels"]
                    new_masks = None

                # Convert outputs back
                image = img_np
                # Albumentations may output image as numpy or torch depending on ToTensorV2
                if isinstance(image, np.ndarray):
                    # leave as numpy; user can add ToTensor in transform
                    pass

                target["boxes"] = torch.as_tensor(new_boxes, dtype=torch.float32)
                # --- FIX: Ensure 2D shape if augmentation removed all boxes --- PARCHE
                if target["boxes"].numel() == 0:
                    target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                # --------------------------------------------------------------
                target["labels"] = torch.as_tensor(new_labels, dtype=torch.int64)

                # Update size fields if image is numpy/torch with known shape
                if isinstance(image, np.ndarray):
                    h2, w2 = image.shape[:2]
                    target["size"] = torch.tensor([h2, w2], dtype=torch.int64)
                elif torch.is_tensor(image):
                    # expecting CHW
                    target["size"] = torch.tensor([int(image.shape[-2]), int(image.shape[-1])], dtype=torch.int64)

                # Recompute area/iscrowd after transform
                b = target["boxes"]
                if b.numel() > 0:
                    target["area"] = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
                else:
                    target["area"] = torch.zeros((0,), dtype=torch.float32)
                target["iscrowd"] = torch.zeros((target["labels"].shape[0],), dtype=torch.uint8)

                if self.return_masks:
                    if new_masks is None or len(new_masks) == 0:
                        # infer H,W from transformed image
                        if isinstance(image, np.ndarray):
                            h2, w2 = image.shape[:2]
                        elif torch.is_tensor(image):
                            h2, w2 = int(image.shape[-2]), int(image.shape[-1])
                        else:
                            h2, w2 = img_h, img_w
                        target["masks"] = torch.zeros((0, h2, w2), dtype=torch.uint8)
                    else:
                        # new_masks is list of HxW arrays
                        target["masks"] = torch.from_numpy(np.stack(new_masks, axis=0).astype(np.uint8))

                return image, target

            except TypeError:
                # Case B: TorchVision v2-style transform that accepts (image, target)
                # or any custom callable that does so.
                out = self.transforms(image, target)
                if isinstance(out, tuple) and len(out) == 2:
                    image, target = out
                else:
                    # Case C: transform(image) only (WARNING: boxes not updated!)
                    image = self.transforms(image)

        return image, target


def detection_collate_fn(batch):
    """Standard collate_fn for TorchVision detection models."""
    return tuple(zip(*batch))