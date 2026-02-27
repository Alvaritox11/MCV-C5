import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO

model = YOLO("yolo26x.pt")
samples = ["0000/000000.png"]
dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")
device = "cuda" if torch.cuda.is_available() else "cpu"
project = "/home/group05/maiol/MCV-C5/Task1/YOLO"
for i,sample in enumerate(samples):
    print("Image", i)
    image_path = os.path.join(dataset_root, sample)
    image = Image.open(image_path).convert("RGB") 
    results = model.predict(
            source=image,
            conf=0.25,
            iou=0.7,
            imgsz=(375, 1242),
            device=device,
            verbose=False,
            visualize = True,
            save = True,
            project = project
        )