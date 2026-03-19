import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO

def yolo_cam(sample, model_path, device, project_path):

    model = YOLO(model_path)

    dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")

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
            project = project_path
        )