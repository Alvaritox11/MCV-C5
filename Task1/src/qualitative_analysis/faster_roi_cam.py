import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FasterRCNNROICAM:
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def _postprocess_boxes(self, boxes, img_list_image_sizes, original_sizes):
        # boxes: list[Tensor] in transformed image coords
        dets = [{"boxes": b} for b in boxes]
        dets = self.model.transform.postprocess(dets, img_list_image_sizes, original_sizes)
        return [d["boxes"] for d in dets]

    def __call__(self, image_tensor, target_label=1, score_thresh=0.7, keep_first=False, combine="max"):
        """
        Returns:
          cam_full: (H,W) CAM pasted into image canvas (0..1)
          info: dict with selected boxes/labels/scores (original image coords)
        """
        device = next(self.model.parameters()).device
        self.model.zero_grad(set_to_none=True)

        image_tensor = image_tensor.to(device)
        images = [image_tensor]
        original_sizes = [img.shape[-2:] for img in images]
        H, W = original_sizes[0]

        with torch.enable_grad():
            img_list, _ = self.model.transform(images)

            feats = self.model.backbone(img_list.tensors)
            proposals, _ = self.model.rpn(img_list, feats, targets=None)  # list[T] per image

            # ---- ROIAlign features (this is the key) ----
            roi_feats = self.model.roi_heads.box_roi_pool(feats, proposals, img_list.image_sizes)
            # roi_feats: [num_proposals_total, C, 7, 7]
            roi_feats.retain_grad()

            box_feats = self.model.roi_heads.box_head(roi_feats)
            class_logits, box_regression = self.model.roi_heads.box_predictor(box_feats)
            # class_logits: [N, num_classes]

            # Convert logits -> per-proposal scores/labels (pre-NMS)
            probs = F.softmax(class_logits, dim=1)
            scores, labels = probs.max(dim=1)

            # Filter by target_label + threshold
            mask = (labels == target_label) & (scores >= score_thresh)

            if keep_first:
                # keep only best scoring ROI among those
                if mask.any():
                    idxs = torch.where(mask)[0]
                    best = idxs[scores[idxs].argmax()]
                    mask = torch.zeros_like(mask, dtype=torch.bool)
                    mask[best] = True

            if mask.sum() == 0:
                raise RuntimeError(f"No ROI proposals predicted as label={target_label} above score_thresh={score_thresh}.")

            # Objective: use logit (better gradients) instead of post-NMS score
            objective = class_logits[mask, target_label].sum()
            objective.backward()

            grads = roi_feats.grad  # [N,C,7,7]
            acts  = roi_feats        # [N,C,7,7]

            # Compute CAM per selected ROI
            cams_roi = []
            for i in torch.where(mask)[0]:
                g = grads[i]  # [C,7,7]
                a = acts[i]   # [C,7,7]
                w = g.mean(dim=(1,2), keepdim=True)   # [C,1,1]
                cam = (w * a).sum(dim=0)              # [7,7]
                cam = F.relu(cam)
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-6)
                cams_roi.append(cam)

            # Now we need ROI boxes to paste CAM back.
            # Use the *proposals* for the same indices. Proposals are per-image; we have 1 image,
            # so proposals[0] corresponds to roi_feats rows.
            prop_boxes = proposals[0]  # [num_props, 4] in transformed coords

            # Map proposals to original image coords
            prop_boxes_orig = self._postprocess_boxes([prop_boxes], img_list.image_sizes, original_sizes)[0]

            # Paste ROI CAM(s) into a full image canvas
            cam_full = torch.zeros((H, W), device=device)

            for cam7, i in zip(cams_roi, torch.where(mask)[0]):
                x1, y1, x2, y2 = prop_boxes_orig[i].round().long().tolist()
                x1 = max(0, min(x1, W-1))
                x2 = max(0, min(x2, W))
                y1 = max(0, min(y1, H-1))
                y2 = max(0, min(y2, H))

                if x2 <= x1 or y2 <= y1:
                    continue

                # resize 7x7 -> ROI pixel size
                cam_up = F.interpolate(cam7[None,None], size=(y2-y1, x2-x1),
                                       mode="bilinear", align_corners=False)[0,0]

                if combine == "sum":
                    cam_full[y1:y2, x1:x2] += cam_up
                else:  # "max" usually nicer for multiple ROIs
                    cam_full[y1:y2, x1:x2] = torch.maximum(cam_full[y1:y2, x1:x2], cam_up)

            cam_full = cam_full - cam_full.min()
            cam_full = cam_full / (cam_full.max() + 1e-6)

            # Return some info for visualization
            sel = torch.where(mask)[0]
            info = {
                "boxes": prop_boxes_orig[sel].detach().cpu(),
                "labels": labels[sel].detach().cpu(),
                "scores": scores[sel].detach().cpu(),
            }

        return cam_full.detach().cpu(), info

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


def faster_cam_roi(sample, output_path, model_path=None, class_id = 1):
    if model_path is None:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        # Infer num_classes from checkpoint
        num_classes = state["roi_heads.box_predictor.cls_score.weight"].shape[0]
        in_features = state["roi_heads.box_predictor.cls_score.weight"].shape[1]

        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        missing, unexpected = model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model = model.cuda().eval()
    cam_extractor = FasterRCNNROICAM(model)

    dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")

    image_path = os.path.join(dataset_root, sample)
    image = Image.open(image_path).convert("RGB") 
    image_tensor = to_tensor(image)
    cam, info = cam_extractor(image_tensor, target_label=class_id, score_thresh=0.7, keep_first=True)
    save_gradcam_overlay(image_tensor, cam, os.path.join(output_path, "grad_cam_roi.png"))
    print(info["scores"][:5], info["boxes"][:1])