import torch
import numpy as np
from transformers import SamModel, SamProcessor

class SAMWrapper:
    def __init__(self, device, model_name="facebook/sam-vit-base", checkpoint_path=None):
        self.device = device
        # Load the SAM processor and model
        self.processor = SamProcessor.from_pretrained(model_name)
        # 1. Always load the base model architecture and config first
        self.model = SamModel.from_pretrained("facebook/sam-vit-base")
        
        # 2. If a checkpoint is provided, overwrite the base weights with your fine-tuned ones
        if checkpoint_path:
            print(f"Applying fine-tuned weights from {checkpoint_path}...")
            
            # Load the .pth file using PyTorch
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            
            # (Optional) If you saved your weights inside a dictionary (e.g., state_dict['model_state']), 
            # you might need to extract them like this:
            # state_dict = state_dict['model_state'] 
            
            # Inject the weights into the model
            self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image, prompt_kwargs):
        """
        Runs inference for a SINGLE image.
        image: PIL.Image
        prompt_kwargs: dict with 'input_points' or 'input_boxes'
        """

        print("Received prompt_kwargs:", prompt_kwargs)  # Debugging line to check the input format

        if not prompt_kwargs:
            return [], []
            
        if "input_points" in prompt_kwargs:
            # Check if the list is completely empty, e.g., [[]]
            if len(prompt_kwargs["input_points"][0]) == 0:
                return [], []
                
        if "input_boxes" in prompt_kwargs:
            # Check if the list is completely empty, e.g., [[]]
            if len(prompt_kwargs["input_boxes"][0][0]) == 0:
                return [], []

        # 1. Process inputs
        inputs = self.processor(image, **prompt_kwargs, return_tensors="pt").to(self.device)

        print("Processed inputs:", {k: v.shape for k, v in inputs.items()})  # Debugging line to check processed input shapes

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