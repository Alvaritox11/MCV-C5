import os
import torch
import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple

class KittiMotsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        return_masks: bool = True,
        class_map: Optional[Dict[int, int]] = None,
        transform=None,
    ):
        self.root_dir = root_dir
        self.return_masks = return_masks
        self.dataset_type = "kitti_mots"
        self.transform = transform

        # KITTI-MOTS classes: 1->Car, 2->Pedestrian. Mapped to COCO ids: 3->Car, 1->Person
        self.class_map = class_map or {1: 3, 2: 1}

        self.img_dir = os.path.join(root_dir, "training", "image_02")
        self.label_dir = os.path.join(root_dir, "instances_txt")

        # Default KITTI-MOTS splits
        default_train = ["0000", "0001", "0003", "0004", "0005", "0009", "0011", "0012", "0015", "0017", "0019", "0020"]
        default_val = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]

        self.seq_ids = default_train if split == "train" else default_val
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        samples = []
        for seq in self.seq_ids:
            seq_path = os.path.join(self.img_dir, seq)
            if not os.path.exists(seq_path): continue

            anno_path = os.path.join(self.label_dir, f"{seq}.txt")
            annos_by_frame = self._parse_txt_annotations(anno_path)

            frames = sorted([f for f in os.listdir(seq_path) if f.endswith(".png")])
            for frame_file in frames:
                frame_id = int(os.path.splitext(frame_file)[0])
                samples.append({
                    "path": os.path.join(seq_path, frame_file),
                    "seq": seq,
                    "frame_id": frame_id,
                    "annos": annos_by_frame.get(frame_id, []),
                })
        return samples

    def _parse_txt_annotations(self, txt_path: str) -> Dict[int, List[Dict[str, Any]]]:
        annotations = {}
        if not os.path.exists(txt_path): return annotations

        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6: continue

                frame_id, track_id, class_id = int(parts[0]), int(parts[1]), int(parts[2])
                height, width = int(parts[3]), int(parts[4])

                if class_id not in self.class_map: continue

                rle_str = " ".join(parts[5:])
                rle_obj = {"counts": rle_str.encode("utf-8"), "size": [height, width]}
                bbox = mask_utils.toBbox(rle_obj).tolist()

                annotations.setdefault(frame_id, []).append({
                    "bbox": bbox,
                    "label": class_id,
                    "rle": rle_obj,
                    "track_id": track_id,
                })
        return annotations

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        img_w, img_h = image.size

        boxes_xyxy, labels, masks, rles = [], [], [], []

        for anno in sample["annos"]:
            x, y, w_box, h_box = anno["bbox"]
            if w_box <= 1 or h_box <= 1:
                continue

            mapped_label = self.class_map.get(anno["label"], None)
            if mapped_label is None:
                continue

            boxes_xyxy.append([x, y, x + w_box, y + h_box])
            labels.append(int(mapped_label))
            rles.append(anno["rle"])

            if self.return_masks:
                m = mask_utils.decode(anno["rle"])
                if m is not None:
                    masks.append(m.astype(np.uint8))

        # Convert PIL image to numpy for Albumentations
        image_np = np.array(image)

        if self.transform is not None and len(boxes_xyxy) > 0:
            transformed = self.transform(
                image=image_np,
                bboxes=boxes_xyxy,
                masks=masks if self.return_masks else [],
                labels=labels
            )

            image_np = transformed["image"]

            if self.return_masks:
                transformed_masks = transformed["masks"]
                transformed_labels = list(transformed["labels"])

                new_masks = []
                new_boxes = []
                new_labels = []

                for mask, label in zip(transformed_masks, transformed_labels):
                    mask = np.asarray(mask, dtype=np.uint8)
                    ys, xs = np.where(mask > 0)

                    if len(xs) == 0 or len(ys) == 0:
                        continue

                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()

                    if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                        continue

                    new_masks.append(mask)
                    new_boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    new_labels.append(int(label))

                masks = new_masks
                boxes_xyxy = new_boxes
                labels = new_labels

            else:
                boxes_xyxy = list(transformed["bboxes"])
                labels = list(transformed["labels"])

        # ---- SANITY CHECKS HERE ----
        assert len(boxes_xyxy) == len(labels)

        if self.return_masks:
            assert len(boxes_xyxy) == len(masks)

        if idx < 3:  # only print for first few samples
            print("Image size:", image_np.shape)
            print("Num boxes:", len(boxes_xyxy))
            if self.return_masks:
                print("Num masks:", len(masks))
                if len(masks) > 0:
                    print("Mask shape:", masks[0].shape)
            print("-" * 40)

        # Convert back to PIL
        image = Image.fromarray(image_np)

        boxes_t = (
            torch.as_tensor(boxes_xyxy, dtype=torch.float32)
            if len(boxes_xyxy) > 0
            else torch.empty((0, 4), dtype=torch.float32)
        )
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "rles": rles
        }

        if self.return_masks:
            if len(masks) > 0:
                target["masks"] = torch.from_numpy(np.stack(masks, axis=0)).to(torch.uint8)
            else:
                h, w = image_np.shape[:2]
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

        return image, target

def detection_collate_fn(batch):
    return tuple(zip(*batch))