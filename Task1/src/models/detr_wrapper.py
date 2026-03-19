import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from peft import LoraConfig, get_peft_model
from PIL import Image

class DetrWrapper:
    def __init__(self, model_name="facebook/detr-resnet-50", device=None, freeze_base=False, use_lora=False, lora_r=8, lora_alpha=32):
        """
        Initializes the DETR model and processor from HuggingFace.
        
        Args:
            model_name (str): HuggingFace model ID. Default is the standard ResNet-50 backbone 
                              used in the original paper[cite: 148].
            device (str): 'cuda' or 'cpu'. If None, detects automatically.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading DETR model: {model_name} on {self.device}...")
        
        # Load the processor (handles resizing, normalization specific to DETR)
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        # Load the model with the correct head for object detection
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        
        # ResNet always freezed to ensure fair comparison across all experiments
        print("Freezing ResNet Backbone...")
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

        # Fine-tuning experiment configurations
        if use_lora:
            print("Applying LoRA to DETR Attention layers...")
            lora_config = LoraConfig(
                r=lora_r, # Sweetspot, aovid memorize data 
                lora_alpha=lora_alpha, # alpha = 4 * r (common choice). Pay heavy attention to the new features without needing to add more params
                target_modules=["q_proj", "v_proj"], # Target the attention matrices
                modules_to_save=["class_labels_classifier", "bbox_predictor"], # " List of modules ... to be set as trainable" as well
                bias="none" # ignore bias
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
        elif freeze_base:
            print("Freezing DETR Transformer. Training heads only...")
            for name, param in self.model.named_parameters():
                if "class_labels_classifier" not in name and "bbox_predictor" not in name:
                    param.requires_grad = False
                    
        else:
            print("Training full Transformer and Heads (ResNet remains frozen)...")
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, images, confidence_threshold=0.7):
        """
        Runs inference on a list of images.

        Args:
            images (list of PIL.Image): List of input images.
            confidence_threshold (float): Filter low-confidence predictions (default 0.7).
                                          DETR always outputs 100 boxes, so this is crucial.

        Returns:
            list of dicts: Standardized results for evaluation.
            [
                {
                    'boxes': [[x1, y1, x2, y2], ...],  # Absolute pixel coordinates
                    'scores': [0.99, 0.85, ...],       # Floats 0-1
                    'labels': [3, 1, ...]              # COCO Class IDs (Ints)
                },
                ...
            ]
        """
        # Ensure input is a list
        if isinstance(images, tuple): images = list(images)
        elif not isinstance(images, list): images = [images]

        # 1. Preprocessing
        # The processor handles resizing and normalization automatically.
        # It accepts a list of PIL images directly.
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Post-processing
        # DETR outputs relative coordinates [0-1]. We must convert them to absolute [x,y,w,h].
        # The processor's helper function does this using the original image sizes.
        target_sizes = torch.tensor([img.size[::-1] for img in images]).to(self.device)
        
        # This converts relative [0,1] center-coords to absolute [x1,y1,x2,y2]
        processed_results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )

        # 4. Standardization
        standardized_predictions = []
        keep_classes = [1, 3] # COCO IDs for Person and Car
        
        for result in processed_results:
            boxes = result['boxes'].cpu().tolist()
            scores = result['scores'].cpu().tolist()
            labels = result['labels'].cpu().tolist()
            
            filtered_boxes, filtered_scores, filtered_labels = [], [], []
            
            # --- NEW: Filter out anything that isn't a Car or Person ---
            for b, s, l in zip(boxes, scores, labels):
                if l in keep_classes:
                    filtered_boxes.append(b)
                    filtered_scores.append(s)
                    filtered_labels.append(l)
            # -----------------------------------------------------------

            pred_dict = {
                'boxes': filtered_boxes,
                'scores': filtered_scores,
                'labels': filtered_labels
            }
            standardized_predictions.append(pred_dict)

        return standardized_predictions