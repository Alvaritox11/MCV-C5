import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

class FasterRCNNWrapper:
    def __init__(self, device: str | None = None, weights: str = "DEFAULT", keep_classes=(1, 3), freeze_base=False, use_partial_unfreeze=False):
        """
        TorchVision Faster R-CNN wrapper using the COCO head (91 classes).

        Args:
            device: 'cuda' or 'cpu'. Auto if None.
            weights: torchvision weights identifier, "DEFAULT" is COCO pretrained.
            keep_classes: which COCO class IDs to keep (default: person=1, car=3)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model WITH COCO head (do not replace predictor)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        
        if use_partial_unfreeze:
            print("Partial Unfreeze: Freezing ResNet layers 1-3. Training layer4, FPN, RPN, and RoI Heads...")
            for name, param in self.model.named_parameters():
                # Freeze the early layers of the backbone
                if any(x in name for x in ["backbone.body.conv1", "backbone.body.bn1", 
                                           "backbone.body.layer1", "backbone.body.layer2", 
                                           "backbone.body.layer3"]):
                    param.requires_grad = False
                else:
                    # Explicitly unfreeze layer4, fpn, rpn, and roi_heads
                    param.requires_grad = True
        elif freeze_base:
            print("Freezing Faster R-CNN base. Training RoI box predictor head only...")
            for name, param in self.model.named_parameters():
                # Freeze everything EXCEPT the final predictor head
                if "roi_heads.box_predictor" not in name:
                    param.requires_grad = False
        else:
            print("Full Fine-Tune: Training entire Faster R-CNN (Backbone, RPN, RoI Heads)...")
            for param in self.model.parameters():
                param.requires_grad = True
        
        self.model.to(self.device)
        self.model.eval()

        self.keep_classes = tuple(keep_classes)
        self.keep_classes_t = torch.tensor(self.keep_classes, device=self.device, dtype=torch.int64)

    @torch.inference_mode()
    def predict(self, images, confidence_threshold: float = 0.7):
        """
        Args:
            images: list/tuple of PIL.Image or torch.Tensor (C,H,W float in [0,1])
            confidence_threshold: filter out low score detections

        Returns:
            list of dicts with 'boxes', 'scores', 'labels' as python lists (CPU).
        """
        if isinstance(images, tuple):
            images = list(images)
        elif not isinstance(images, list):
            images = [images]

        # Convert PIL -> Tensor, and move to device
        tensor_images = []
        for img in images:
            if hasattr(img, "mode"):  # PIL.Image
                img_t = F.to_tensor(img)  # float32 [0,1] CHW
            elif isinstance(img, np.ndarray):
                img_t = F.to_tensor(img)  # F.to_tensor natively converts HWC numpy to CHW tensor!
            elif torch.is_tensor(img):
                img_t = img
                if img_t.dtype != torch.float32:
                    img_t = img_t.float()
                if img_t.max() > 1.5:
                    img_t = img_t / 255.0
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            tensor_images.append(img_t.to(self.device))

        outputs = self.model(tensor_images)

        standardized = []
        for out in outputs:
            boxes = out["boxes"]
            scores = out["scores"]
            labels = out["labels"]

            # score filter
            keep = scores >= confidence_threshold

            # class filter (person=1, car=3)
            try:
                keep = keep & torch.isin(labels, self.keep_classes_t)
            except AttributeError:
                # fallback if torch.isin not available
                keep2 = labels.new_zeros(labels.shape, dtype=torch.bool)
                for c in self.keep_classes:
                    keep2 |= (labels == c)
                keep = keep & keep2

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            standardized.append(
                {
                    "boxes": boxes.detach().cpu().tolist(),
                    "scores": scores.detach().cpu().tolist(),
                    "labels": labels.detach().cpu().tolist(),
                }
            )

        return standardized
