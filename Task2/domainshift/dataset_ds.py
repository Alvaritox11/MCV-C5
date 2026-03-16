import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OxfordPetDataset(Dataset):
    def __init__(self, root_dir, split="test", transform=None, label_mode="species"):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.trimaps_dir = os.path.join(self.annotations_dir, "trimaps")
        self.xmls_dir = os.path.join(self.annotations_dir, "xmls")
        self.transform = transform
        self.label_mode = label_mode

        split_file = os.path.join(self.annotations_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.samples = []
        self.categories = self._build_categories()

        with open(split_file, "r") as f:
            valid_image_id = 0
            for line in f:
                line = line.strip()

                # Skip empty lines, comments, headers
                if not line or line.startswith("#") or line.startswith("Image"):
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                try:
                    image_stem = parts[0]
                    class_id = int(parts[1])   # 1..37
                    species = int(parts[2])    # 1=cat, 2=dog
                    breed_id = int(parts[3])
                except ValueError:
                    continue

                image_path = os.path.join(self.images_dir, f"{image_stem}.jpg")
                mask_path = os.path.join(self.trimaps_dir, f"{image_stem}.png")
                xml_path = os.path.join(self.xmls_dir, f"{image_stem}.xml")

                if not os.path.exists(image_path):
                    continue

                if not os.path.exists(mask_path):
                    continue

                if not self._has_valid_prompt_and_gt(mask_path):
                    print("self._has_valid_prompt_and_gt(mask_path:", self._has_valid_prompt_and_gt(mask_path))
                    continue

                    
                if self.label_mode == "species":
                    label = 1 if species == 1 else 2
                elif self.label_mode == "breed":
                    label = class_id
                else:
                    raise ValueError(f"Unsupported label_mode: {self.label_mode}")

                self.samples.append({
                    "image_id": valid_image_id,
                    "image_stem": image_stem,
                    "class_id": class_id,
                    "species": species,
                    "breed_id": breed_id,
                    "label": label,
                    "path": image_path
                })
                valid_image_id += 1
                
    def _build_categories(self):
        if self.label_mode == "species":
            return [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ]

        class_to_name = {}
        for split_name in ["trainval.txt", "test.txt", "list.txt"]:
            split_path = os.path.join(self.annotations_dir, split_name)
            if not os.path.exists(split_path):
                continue

            with open(split_path, "r") as f:
                for line in f:
                    line = line.strip()

                    # Skip empty lines, comments, and headers
                    if not line or line.startswith("#") or line.startswith("Image"):
                        continue

                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    try:
                        image_stem = parts[0]
                        class_id = int(parts[1])
                    except ValueError:
                        continue

                    breed_name = "_".join(image_stem.split("_")[:-1])
                    class_to_name[class_id] = breed_name

        return [{"id": cid, "name": class_to_name[cid]} for cid in sorted(class_to_name)]



    def __len__(self):
        return len(self.samples)




    def _read_box_from_xml(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Missing XML annotation: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")

        if bndbox is None:
            raise ValueError(f"No bounding box found in {xml_path}")


        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        stem = sample["image_stem"]

        image_path = os.path.join(self.images_dir, f"{stem}.jpg")
        mask_path = os.path.join(self.trimaps_dir, f"{stem}.png")
        xml_path = os.path.join(self.xmls_dir, f"{stem}.xml")

        image = Image.open(image_path).convert("RGB")
        trimap = np.array(Image.open(mask_path), dtype=np.uint8)

        trimap = np.array(Image.open(mask_path), dtype=np.uint8)

        binary_mask = (trimap == 1).astype(np.uint8)
        prompt_mask = (trimap != 2).astype(np.uint8)

        box = self._box_from_mask(prompt_mask)
        if box is None:
            w, h = image.size
            box = [0.0, 0.0, float(w - 1), float(h - 1)]
            
        target = {
            "image_id": torch.tensor(sample["image_id"], dtype=torch.int64),
            "boxes": torch.tensor([box], dtype=torch.float32),
            "labels": torch.tensor([sample["label"]], dtype=torch.int64),
            "masks": torch.tensor(binary_mask[None, ...], dtype=torch.uint8),
            "image_path": image_path,
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def _box_from_mask(self, binary_mask):
        ys, xs = np.where(binary_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        xmin = float(xs.min())
        ymin = float(ys.min())
        xmax = float(xs.max())
        ymax = float(ys.max())
        return [xmin, ymin, xmax, ymax]
    
    def _has_valid_prompt_and_gt(self, mask_path):
        trimap = np.array(Image.open(mask_path), dtype=np.uint8)

        eval_mask = (trimap == 1).astype(np.uint8)
        prompt_mask = (trimap != 2).astype(np.uint8)

        has_gt = eval_mask.sum() > 0
        has_prompt = prompt_mask.sum() > 0
        return has_gt and has_prompt

def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)