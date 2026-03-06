import torch
import numpy as np
from transformers import SamModel, SamProcessor

class SAMWrapper:
    def __init__(self, device, model_name="facebook/sam-vit-base"):
        self.device = device
        # Load the SAM processor and model
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image, prompt_kwargs):
        """
        Runs inference for a SINGLE image.
        image: PIL.Image
        prompt_kwargs: dict with 'input_points' or 'input_boxes'
        """
        # 1. Process inputs
        inputs = self.processor(image, **prompt_kwargs, return_tensors="pt").to(self.device)

        # 2. Forward pass
        outputs = self.model(**inputs)

        # 3. Post-process the 256x256 masks back to the original image size
        # This removes padding and scales them correctly!
        processed_masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0] # Extract the 0th batch

        # Extract scores
        scores = outputs.iou_scores[0].cpu() # Shape: [N_objects, 3]

        # Find the index of the highest scoring mask for each object
        best_mask_indices = torch.argmax(scores, dim=-1) # Shape: [N_objects]
        
        N_objects = processed_masks.shape[0]
        best_masks = []
        best_scores = []
        
        for i in range(N_objects):
            best_idx = best_mask_indices[i]
            
            # Extract the best mask and convert logits to a binary mask
            mask_logits = processed_masks[i, best_idx]
            best_mask = (mask_logits > 0).numpy().astype(np.uint8)
            
            best_masks.append(best_mask)
            best_scores.append(scores[i, best_idx].item())

        return best_masks, best_scores