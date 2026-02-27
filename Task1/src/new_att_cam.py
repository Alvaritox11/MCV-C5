import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import os 
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.cm as cm 
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    attn_implementation="eager",  
).to(device)

model.eval()
model.config.output_attentions = True  # now allowed

samples = ["0000/000000.png", "0001/000003.png", "0003/000006.png"]
dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")
image_path = os.path.join(dataset_root, samples[0])
img = Image.open(image_path).convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, return_dict=True)

print(len(outputs.cross_attentions), outputs.cross_attentions[0].shape)

attn = outputs.cross_attentions[-1]      # (B=1, H=8, Q=100, S=546)
attn = attn.mean(dim=1)[0]               # (Q=100, S=546)

q = 20                                    # query index you want
a = attn[q]     


S = a.shape[0]
best = None
for h in range(1, int(math.sqrt(S)) + 1):
    if S % h == 0:
        w = S // h
        best = (h, w)
h, w = best
print("grid:", h, w)  # e.g. 21 x 26 = 546
a2 = a.reshape(h, w)

img_h, img_w = inputs["pixel_values"].shape[-2:]

heat = F.interpolate(
    a2[None, None], size=(img_h, img_w),
    mode="bilinear", align_corners=False
)[0, 0]

mean = torch.tensor([0.485, 0.456, 0.406], device=inputs["pixel_values"].device).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225], device=inputs["pixel_values"].device).view(3,1,1)

img_tensor = inputs["pixel_values"][0]
img_unnorm = (img_tensor * std + mean).clamp(0,1)
img_np = img_unnorm.permute(1,2,0).cpu().numpy()

# Convert to uint8
img_uint8 = (img_np * 255).astype(np.uint8)
base_img = Image.fromarray(img_uint8)

# ---- 2. Normalize attention heatmap ----
heat = heat.cpu()
heat = heat - heat.min()
heat = heat / (heat.max() + 1e-8)

heat_np = heat.numpy()

# ---- 3. Convert heatmap to color using colormap ----
colormap = cm.get_cmap("inferno")  # better than jet
heat_color = colormap(heat_np)[:, :, :3]  # drop alpha
heat_color = (heat_color * 255).astype(np.uint8)

heat_img = Image.fromarray(heat_color)

# ---- 4. Resize heatmap to match image (safety) ----
heat_img = heat_img.resize(base_img.size, resample=Image.BILINEAR)

# ---- 5. Alpha blend manually ----
alpha = 0.6
overlay = Image.blend(base_img, heat_img, alpha=alpha)

# ---- 6. Save ----
overlay.save("detr_attention_overlay.png")
     