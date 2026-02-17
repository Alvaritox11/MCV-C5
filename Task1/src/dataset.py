import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pycocotools.mask as mask_utils

class KittiMotsDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        """
        Args:
            root_dir (str): Path to KITTI-MOTS folder (e.g., /home/mcv/datasets/C5/KITTI-MOTS/)
            split (str): 'train' or 'val' (KITTI-MOTS usually splits by sequence ID)
            transforms (callable, optional): Albumentations or Torchvision transforms
        """
        self.root_dir = root_dir
        self.transforms = transforms
        
        # KITTI-MOTS structure: /instances_txt/ and /image_02/
        self.img_dir = os.path.join(root_dir, 'training', 'image_02')
        self.label_dir = os.path.join(root_dir, 'instances_txt')
        
        # Define sequences for split (Standard KITTI-MOTS splits)
        # You might need to adjust these IDs based on the exact PDF instructions or provided files
        if split == 'train':
            self.seq_ids = ['0000', '0001', '0003', '0004', '0005', '0009', '0011', '0012', '0015', '0017', '0019', '0020']
        else: # val
            self.seq_ids = ['0002', '0006', '0007', '0008', '0010', '0013', '0014', '0016', '0018']

        self.samples = self._load_samples()

        # MAPPING: KITTI-MOTS Class ID -> COCO Class ID
        # KITTI: 1=Car, 2=Pedestrian 
        # COCO: 3=Car, 1=Person
        self.class_map = {
            1: 3,  # Car -> Car
            2: 1   # Pedestrian -> Person
        }

    def _load_samples(self):
        samples = []
        for seq in self.seq_ids:
            seq_path = os.path.join(self.img_dir, seq)
            if not os.path.exists(seq_path): continue
            
            # Load annotations for this sequence
            anno_path = os.path.join(self.label_dir, f"{seq}.txt")
            annos = self._parse_txt_annotations(anno_path)
            
            # List all images in sequence
            frames = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])
            
            for frame in frames:
                frame_id = int(frame.replace('.png', ''))
                samples.append({
                    'path': os.path.join(seq_path, frame),
                    'seq': seq,
                    'frame_id': frame_id,
                    'annos': annos.get(frame_id, [])
                })
        return samples

    def _parse_txt_annotations(self, txt_path):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert("RGB")
        w, h = image.size
        
        boxes = []
        labels = []
        
        for anno in sample['annos']:
            # Convert xywh to xyxy for standard detection format
            x, y, w_box, h_box = anno['bbox']
            boxes.append([x, y, x + w_box, y + h_box])
            
            # Map KITTI label to COCO label [cite: 448]
            labels.append(self.class_map.get(anno['label']))

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx]),
            'orig_size': torch.as_tensor([h, w]),
            'seq': sample['seq'] # Helpful for debugging
        }

        if self.transforms:
            # Note: You'll need to adapt this depending on if you use 
            # torchvision transforms (takes PIL) or Albumentations (takes numpy)
            image = self.transforms(image)

        return image, target