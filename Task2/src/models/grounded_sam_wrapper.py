import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from src.models.sam_wrapper import SAMWrapper

class GroundedSAMWrapper:
    def __init__(self, device, gd_model_id="IDEA-Research/grounding-dino-base", sam_model_id="facebook/sam-vit-base"):
        self.device = device
        
        print("Loading Grounding DINO...")
        self.gd_processor = AutoProcessor.from_pretrained(gd_model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_model_id).to(device)
        self.gd_model.eval()
        
        print("Loading SAM...")
        self.sam = SAMWrapper(device, sam_model_id)
        
        # Mapping for the Evaluator
        self.class_map = {"person": 1, "car": 3}
        # Mapping for the Visualizations
        self.class_name_map = {1: "Pedestrian", 3: "Car"}

    @torch.inference_mode()
    def predict(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        # 1. Grounding DINO: Text -> Boxes
        inputs = self.gd_processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        outputs = self.gd_model(**inputs)
        
        width, height = image.size
        
        # FIX: Changed 'box_threshold' to 'threshold' to support recent Transformers updates
        gd_results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold, 
            text_threshold=text_threshold,
            target_sizes=[(height, width)]
        )[0]
        
        boxes = gd_results["boxes"]
        gd_scores = gd_results["scores"]
        
        # FIX: Handle the recent transformers API change where text labels moved to "text_labels"
        if "text_labels" in gd_results:
            phrases = gd_results["text_labels"]
        else:
            phrases = gd_results["labels"]
        
        if len(boxes) == 0:
            return [], [], [], [], []

        # Filter and map labels cleanly
        filtered_boxes, predicted_labels, mapped_phrases = [], [], []
        
        for box, phrase in zip(boxes, phrases):
            mapped_label = 0
            # Ensure phrase is a string (older versions returned strings, newest return IDs in "labels")
            phrase_str = str(phrase).lower() 
            
            for key, val in self.class_map.items():
                if key in phrase_str:
                    mapped_label = val
                    break
            
            if mapped_label != 0:
                filtered_boxes.append(box.tolist())
                predicted_labels.append(mapped_label)
                mapped_phrases.append(self.class_name_map[mapped_label])

        if len(filtered_boxes) == 0:
            return [], [], [], [], []

        # 2. SAM: Boxes -> Masks
        sam_prompt = {"input_boxes": [[filtered_boxes]]}
        masks, sam_scores = self.sam.predict(image, sam_prompt)

        return masks, sam_scores, filtered_boxes, predicted_labels, mapped_phrases