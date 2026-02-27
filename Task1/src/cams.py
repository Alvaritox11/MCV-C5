import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor


class FasterRCNNGradCAM:
    def __init__(self, model, feature_keys=("0", "1", "2", "3")):
        self.model = model.eval()
        self.feature_keys = feature_keys
        self.activations = {}

    def __call__(self, image_tensor, target_label=1, score_thresh=0.7, combine="sum", keep_first = False):
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device)
        self.model.zero_grad(set_to_none=True)

        images = [image_tensor]
        original_image_sizes = [img.shape[-2:] for img in images]

        with torch.enable_grad():
            img_list, _ = self.model.transform(images)

            feats = self.model.backbone(img_list.tensors)  # OrderedDict
            # retain grads for all FPN levels you care about
            self.activations = {}
            for k in self.feature_keys:
                if k in feats:
                    self.activations[k] = feats[k]
                    self.activations[k].retain_grad()

            proposals, _ = self.model.rpn(img_list, feats, targets=None)
            detections, _ = self.model.roi_heads(feats, proposals, img_list.image_sizes, targets=None)

            # keep raw detections for gradient objective
            det_raw = detections[0]
            # postprocess only for returning boxes in original image coords
            det_pp = self.model.transform.postprocess(detections, img_list.image_sizes, original_image_sizes)[0]

            # optional thresholding (affects objective + returned det)
            keep = det_raw["scores"] >= score_thresh
            det_raw = {k: v[keep] for k, v in det_raw.items()}
            det_pp  = {k: v[keep] for k, v in det_pp.items()}

            if det_raw["scores"].numel() == 0:
                raise RuntimeError(f"No detections above score_thresh={score_thresh}.")

            labels = det_raw["labels"]
            scores = det_raw["scores"]
            print(scores)
            mask = (labels == target_label)

            if keep_first: #if only the best box is wanted to be analyzed
                cumsum = mask.cumsum(dim=0)
                mask = mask & (cumsum == 1)

            if mask.sum() == 0:
                raise RuntimeError(f"No detections of label={target_label} above threshold.")
            
            objective = scores[mask].sum()
            objective.backward(retain_graph=False)

            # build CAM per FPN level and combine
            H, W = image_tensor.shape[-2:]
            cams = []
            for k, acts in self.activations.items():
                grads = acts.grad
                if grads is None:
                    continue
                # skip levels with zero gradient
                if grads.abs().max().item() == 0.0:
                    continue

                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = (weights * acts).sum(dim=1, keepdim=True)
                cam = F.relu(cam)

                cam = cam[0, 0]
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-6)

                cam = F.interpolate(cam[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                cams.append(cam)

            if not cams:
                raise RuntimeError(
                    "All FPN-level gradients were zero. Try different feature_keys or compute CAM on ROI features."
                )

            if combine == "max":
                cam = torch.stack(cams, dim=0).max(dim=0).values
            else:  # "sum"
                cam = torch.stack(cams, dim=0).mean(dim=0)  # mean is usually nicer than sum
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-6)

        return cam.detach().cpu(), {k: v.detach().cpu() for k, v in det_pp.items()}
    
def save_gradcam_overlay(image_tensor, cam, out_path, alpha=0.45):
    """
    image_tensor: (3,H,W) float in [0,1] on CPU or GPU
    cam: (H,W) float in [0,1] on CPU (as returned by your extractor)
    out_path: path to save PNG/JPG
    """
    # Move to CPU
    img = image_tensor.detach().float().cpu()
    cam = cam.detach().float().cpu()

    # To HWC uint8
    img_np = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Apply colormap to CAM -> RGBA in [0,1]
    cam_np = cam.numpy()
    heatmap = cm.get_cmap("jet")(cam_np)[..., :3]  # drop alpha channel, RGB in [0,1]
    heatmap = (heatmap * 255.0).astype(np.uint8)

    # Blend
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = overlay.clip(0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)
def crop_bottom_half(image):
    print(image.size)
    print(int(image.size[1]/2))
    cropped_img = image.crop(((0, 0, image.size[0]//2, image.size[1] )))
    return cropped_img

# --- Example usage ---
ckpt = torch.load(
    "/home/group05/maiol/MCV-C5/Task1/best_model.pt",
    map_location="cpu",
    weights_only=True
)  # ckpt is OrderedDict

# Build the same architecture you trained
model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)

# If you trained with custom num_classes, set the predictor BEFORE loading
# (see the helper below to infer it automatically)
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

missing, unexpected = model.load_state_dict(ckpt, strict=True)
model = model.cuda().eval()
cam_extractor = FasterRCNNGradCAM(model)

# image: PIL -> tensor
# image_tensor = to_tensor(pil_image)  # [0,1], shape (3,H,W)
samples = ["0000/000000.png"]
dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")
for i,sample in enumerate(samples):
    print("Image", i)
    image_path = os.path.join(dataset_root, sample)
    image = Image.open(image_path).convert("RGB") 
    image = crop_bottom_half(image)
    image_tensor = to_tensor(image)
    print(image.size)
    cam, det = cam_extractor(image_tensor, target_label=3, score_thresh = 0.7, keep_first=True)
    #det has boxes/labels/scores for the same forward
    save_gradcam_overlay(image_tensor, cam, f"gradcam_overlay_half_{i}.png")