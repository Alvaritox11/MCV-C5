import os
import torch
import torch.nn.functional as F
from transformers import SamModel, SamProcessor

class SAMFineTuneWrapper:
    def __init__(self, device, model_name="facebook/sam-vit-base"):
        self.device = device
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(device)

        # Freeze encoders, train decoder
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.model.train()

    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0, eps=1e-6):
        """
        Binary focal loss on logits.
        logits:  [N, H, W]
        targets: [N, H, W] with values 0/1
        """
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, eps, 1.0 - eps)

        ce_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

        focal = alpha_t * ((1 - p_t) ** gamma) * ce_loss
        return focal.mean()
        
    def get_trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def dice_loss(self, logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice.mean()

    def forward_train(self, image, boxes, gt_masks):
        if boxes.shape[0] == 0:
            return None

        input_boxes = [[boxes.tolist()]]

        inputs = self.processor(
            image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        # Post-process to ORIGINAL image size, but keep as logits/float masks
        processed_masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            binarize=False
        )[0]   # [N, 3, H, W]

        iou_scores = outputs.iou_scores[0]  # [N, 3]
        best_idx = torch.argmax(iou_scores, dim=-1)

        best_logits = torch.stack([
            processed_masks[i, best_idx[i]] for i in range(processed_masks.shape[0])
        ])  # [N, H, W]

        gt_masks = gt_masks.float().to(self.device)  # [N, H, W]

        bce = F.binary_cross_entropy_with_logits(best_logits, gt_masks)
        dice = self.dice_loss(best_logits, gt_masks)
        focal = self.focal_loss(best_logits, gt_masks, alpha=0.25, gamma=2.0)

        loss = dice + bce
        
        return {
            "loss": loss,
            "bce_loss": bce.detach(),
            "dice_loss": dice.detach(),
            "focal_loss": focal.detach()
        }

    @torch.inference_mode()
    def predict(self, image, prompt_kwargs):
        self.model.eval()

        inputs = self.processor(image, **prompt_kwargs, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        processed_masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
            binarize=False
        )[0]

        scores = outputs.iou_scores[0].cpu()
        best_mask_indices = torch.argmax(scores, dim=-1)

        best_masks, best_scores = [], []

        for i in range(processed_masks.shape[0]):
            best_idx = best_mask_indices[i]
            mask_logits = processed_masks[i, best_idx]

            # DEBUG
            # print("\n--- MASK DEBUG ---")
            # print("mask shape:", mask_logits.shape)
            # print("mask logits min:", mask_logits.min().item())
            # print("mask logits max:", mask_logits.max().item())
            # print("positive pixels:", (mask_logits > 0).sum().item())

            # probs = torch.sigmoid(mask_logits)
            # print("prob min/max/mean:", probs.min().item(), probs.max().item(), probs.mean().item())
            
            best_mask = (mask_logits > 0).numpy().astype("uint8")

            best_masks.append(best_mask)
            best_scores.append(scores[i, best_idx].item())

        
        self.model.train()

        return best_masks, best_scores

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)