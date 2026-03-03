import os

import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

KITTI = False

class FasterRCNNGradCAM:
    def __init__(self, model, feature_keys=("0", "1", "2", "3")):
        self.model = model.eval()
        self.feature_keys = feature_keys
        self.activations = {}

    def __call__(self, image_tensor, target_label=1, score_thresh=0.7, combine="mean", box_idx=0):
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
            det_pp = self.model.transform.postprocess(
                detections, img_list.image_sizes, original_image_sizes
            )[0]

            # filter by confidence threshold
            keep = det_raw["scores"] >= score_thresh
            det_raw = {k: v[keep] for k, v in det_raw.items()}
            det_pp  = {k: v[keep] for k, v in det_pp.items()}

            if det_raw["scores"].numel() == 0:
                raise RuntimeError(f"No detections above score_thresh={score_thresh}.")

            labels = det_raw["labels"]
            scores = det_raw["scores"]
            print("Scores:", scores)
            mask = (labels == target_label)

            if mask.sum() == 0:
                raise RuntimeError(f"No detections of label={target_label} above threshold.")

            # Get indices of all detections of this class, pick the one at box_idx
            class_indices = mask.nonzero(as_tuple=False).squeeze(1)
            # scores are already sorted descending by Faster RCNN
            
            selected = class_indices[box_idx]  # 0=highest, -1=lowest, 1=second highest, etc.
            mask = torch.zeros_like(mask)
            mask[selected] = True

            objective = scores[mask].sum()
            objective.backward(retain_graph=False)

            # build CAM per FPN level and combine
            H, W = image_tensor.shape[-2:]
            cams = []
            for k, acts in self.activations.items():
                grads = acts.grad
                if grads is None:
                    continue
                if grads.abs().max().item() == 0.0:
                    print(f"FPN level {k}: zero gradient (skipped)")
                    continue

                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam_k = (weights * acts).sum(dim=1, keepdim=True)
                cam_k = F.relu(cam_k)
                cam_k = cam_k[0, 0]
                cam_k = cam_k - cam_k.min()
                cam_k = cam_k / (cam_k.max() + 1e-6)
                cam_k = F.interpolate(
                    cam_k[None, None], size=(H, W), mode="bilinear", align_corners=False
                )[0, 0]

                print(f"FPN level {k}: contributed (grad max={grads.abs().max().item():.6f}, spatial={acts.shape[-2:]})")
                cams.append((k, cam_k))  # store key alongside cam

            if not cams:
                raise RuntimeError(
                    "All FPN-level gradients were zero. "
                    "Try different feature_keys or compute CAM on ROI features."
                )

            if combine == "max":
                cam = torch.stack([c for _, c in cams], dim=0).max(dim=0).values
            else:
                cam = torch.stack([c for _, c in cams], dim=0).mean(dim=0)
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-6)

        # return per-level cams too for inspection
        per_level = {k: c.detach().cpu() for k, c in cams}
        return cam.detach().cpu(), {k: v.detach().cpu() for k, v in det_pp.items()}, int(selected.item()), per_level



def save_gradcam_overlay(image_tensor, cam, out_path, alpha=0.45, return_pil=False):
    """
    image_tensor: (3, H, W) float in [0, 1]
    cam:          (H, W)    float in [0, 1]
    """
    img = image_tensor.detach().float().cpu()
    cam = cam.detach().float().cpu()

    img_np = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    cam_np = cam.numpy()


    heatmap = cm.get_cmap("jet")(cam_np)[..., :3]   # RGB in [0, 1]
    heatmap = (heatmap * 255.0).astype(np.uint8)

    overlay = ((1 - alpha) * img_np + alpha * heatmap).clip(0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)
    overlay_pil.save(out_path)

    if return_pil:
        return overlay_pil


def draw_bbox_on_pil(pil_img, box_xyxy, text=None, width=3):
    img  = pil_img.copy()
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = box_xyxy
    for t in range(width):
        draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(255, 0, 0))

    if text is not None:
        tx, ty = x1, max(0, y1 - 18)
        draw.rectangle([tx, ty, tx + 160, ty + 18], fill=(255, 0, 0))
        draw.text((tx + 4, ty + 2), text, fill=(255, 255, 255))

    return img


def faster_cam(sample, output_path, model_path=None, class_id=1, feature_keys=("0", "1", "2", "3"), pred_id=0, lora = False):

    if model_path is None:
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        # weights_only=False required for full checkpoint dicts
        ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        if lora:
            finetuned_state = {}
            for k, v in state.items():
                if "modules_to_save.default" in k:
                    new_key = k.replace("base_model.model.", "").replace("modules_to_save.default.", "")
                    finetuned_state[new_key] = v
                elif "original_module" not in k:
                    new_key = k.replace("base_model.model.", "")
                    finetuned_state[new_key] = v
            print("Stripped LoRA keys (first 10):")
            for k in list(finetuned_state.keys())[:10]:
                print(" ", k)
            # infer num_classes from bbox_predictor head
            num_classes = finetuned_state["roi_heads.box_predictor.cls_score.weight"].shape[0]
            in_features  = finetuned_state["roi_heads.box_predictor.cls_score.weight"].shape[1]

            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            missing, unexpected = model.load_state_dict(finetuned_state, strict=False)
            if missing:
                print(f"WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(f"WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
        else:
            for k in state.keys():
                if "predictor" in k or "cls_score" in k or "box_predictor" in k:
                    print(k)
            # Infer num_classes and in_features directly from checkpoint
            num_classes = state["roi_heads.box_predictor.cls_score.weight"].shape[0]
            in_features = state["roi_heads.box_predictor.cls_score.weight"].shape[1]

            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            # strict=False to tolerate minor key mismatches (e.g. from Lightning)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"WARNING: {len(missing)} missing keys (first 5): {missing[:5]}")
            if unexpected:
                print(f"WARNING: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")

    model = model.cuda().eval()
    cam_extractor = FasterRCNNGradCAM(model, feature_keys)

    if KITTI:
        dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")
        image_path   = os.path.join(dataset_root, sample)
    else:
        data_root = "/ghome/group05/c5_data"
        dataset_root = os.path.join(data_root, "images")
        image_path   = os.path.join(dataset_root, sample)

    image        = Image.open(image_path).convert("RGB")
    image_tensor = to_tensor(image)

    cam, det, selected_idx, per_level = cam_extractor(
    image_tensor, target_label=class_id, score_thresh=0.7, box_idx=pred_id
)
    for level_key, level_cam in per_level.items():
        overlay_pil = save_gradcam_overlay(
            image_tensor, level_cam,
            out_path=os.path.join(output_path, f"grad_cam_level_{level_key}.png"),
            alpha=0.45,
            return_pil=True
        )

    # use selected_idx directly instead of re-deriving it
    box   = det["boxes"][selected_idx].tolist()
    score = float(det["scores"][selected_idx].item())
    label = int(det["labels"][selected_idx].item())

    # overlay_pil = save_gradcam_overlay(
    #     image_tensor, cam,
    #     out_path=os.path.join(output_path, "full_grad_cam.png"),
    #     alpha=0.45,
    #     return_pil=True,
    # )

    #box, score, label = get_top_detection(det, target_label=class_id, box_idx=pred_id)

    overlay_boxed = draw_bbox_on_pil(
        overlay_pil, box,
        text=f"cls={label} score={score:.3f}",
        width=3,
    )
    overlay_boxed.save(os.path.join(output_path, "gradcam_with_box.png"))
    print('Done here')