import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)
import os
KITTI = True

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

torch.set_grad_enabled(False)
def detr_decoder_cam(sample, out_file, model_path = "facebook/detr-resnet-50", checkpoint_path=None):
    # --- Load model from HuggingFace (or local path) ---
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path, attn_implementation="eager")
    base_w = model.model.encoder.layers[0].self_attn.v_proj.weight.clone()
    base_cls = model.class_labels_classifier.weight.clone()
    if checkpoint_path is not None:
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_raw = ckpt.get("state_dict", ckpt)
        state = {k.replace("base_model.model.model.", "model."): v for k, v in state_raw.items()}
        for k in state_raw.keys():
            if "lora_A" in k:
                print(k)
                break
        # First load all base_layer weights (the frozen base)
        base_state = {}
        for k, v in state.items():
            if "base_layer" in k:
                new_key = k.replace(".base_layer", "")
                base_state[new_key] = v
            elif "lora_A" not in k and "lora_B" not in k and "original_module" not in k and "modules_to_save" not in k:
                base_state[k] = v
        
        # Then merge LoRA deltas: W = W_base + lora_B @ lora_A
        # find all lora pairs
        lora_keys = set()
        for k in state.keys():
            if "lora_A" in k:
                lora_keys.add(k.replace(".lora_A.default.weight", ""))
        for k in state.keys():
            if "encoder.layers.0" in k and "v_proj" in k:
                print(k)
        lora_rank = None    
        for prefix in lora_keys:
            lora_A = state[f"{prefix}.lora_A.default.weight"]
            lora_B = state[f"{prefix}.lora_B.default.weight"]
            if lora_rank is None:
                lora_rank = lora_A.shape[0]
            base_key = prefix + ".weight"  # <-- fix here
            scaling = 1.0
            if base_key in base_state:
                base_state[base_key] = base_state[base_key] + scaling * (lora_B @ lora_A)
            else:
                print(f"WARNING: base_key not found: {base_key}")

        # handle modules_to_save (fine-tuned heads)
        for k, v in state_raw.items():
            if "modules_to_save.default" in k and "original_module" not in k:
                # base_model.model.class_labels_classifier.modules_to_save.default.weight
                # -> class_labels_classifier.weight
                new_key = k.replace("base_model.model.", "").replace("modules_to_save.default.", "")
                base_state[new_key] = v
                print(f"  head key: {new_key}")

        missing, unexpected = model.load_state_dict(base_state, strict=False)
        print("missing", len(missing), "unexpected", len(unexpected))
        print("First 5 missing:", missing[:5])
        print("base_state keys (first 5):", list(base_state.keys())[:5])
        print("model state_dict keys (first 5):", list(model.state_dict().keys())[:5])
        new_w = model.model.encoder.layers[0].self_attn.v_proj.weight
        print("Weight diff:", (new_w - base_w).abs().max().item())
        print("Classifier diff:", (model.class_labels_classifier.weight - base_cls).abs().max().item())
    model.eval()

    # --- Load image ---
    if KITTI:
        dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")
        image_path = os.path.join(dataset_root, sample)
        im = Image.open(image_path).convert("RGB")
    else:
        dataset_root = "/ghome/group05/c5_data"
        image_path = os.path.join(dataset_root, "images", sample)
        im = Image.open(image_path).convert("RGB")
    print(im)
    # --- Preprocess with HF processor (replaces manual transform) ---
    inputs = processor(images=im, return_tensors="pt")

    # --- First pass to get detections ---
    outputs = model(**inputs)
    print('outputs', outputs)
    # Post-process: filter by confidence
    target_sizes = torch.tensor([im.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]

    keep_scores = results["scores"]
    keep_labels = results["labels"]
    keep_boxes  = results["boxes"]  # already in xyxy, image-scale

    # --- Hooks for attention visualization ---
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [
        model.model.backbone.model.layer3.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.model.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.model.decoder.layers[-1].encoder_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # --- Second pass to capture attention weights ---
    outputs = model(**inputs)

    for hook in hooks:
        hook.remove()
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # --- Get feature map spatial dims ---
    # For HF DETR the encoder input is flattened; recover h, w from the pixel mask
    pixel_mask = inputs["pixel_mask"][0]  # [H, W] of original padded size

    # The feature map is downsampled by 32
    h_attn = (pixel_mask[:, 0].sum() / 32).round().int().item()
    w_attn = (pixel_mask[0, :].sum() / 32).round().int().item()

    print(f"h_attn: {h_attn}, w_attn: {w_attn}, product: {h_attn*w_attn}")

    # --- Plot ---
    # We need the indices of kept queries to index dec_attn_weights
    # Use outputs.pred_logits to find which query slots correspond to detections
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep_query_ids = (probas.max(-1).values > 0.8).nonzero()
    mask = torch.isin(keep_labels, torch.tensor([1, 3]))

    keep_scores = keep_scores[mask]
    keep_labels = keep_labels[mask]
    keep_boxes  = keep_boxes[mask]
    keep_query_ids = keep_query_ids[mask]
    fig, axs = plt.subplots(ncols=len(keep_boxes), nrows=2, figsize=(22, 7))
    if len(keep_boxes) == 1:
        axs = axs[:, None]  # ensure 2D indexing works for single detection
    colors = COLORS * 100

    h_attn = (pixel_mask[:, 0].sum() / 32).round().int().item()
    w_attn = (pixel_mask[0, :].sum() / 32).round().int().item()

    for idx, ax_i, (xmin, ymin, xmax, ymax), label, score in zip(
        keep_query_ids, axs.T, keep_boxes.tolist(), keep_labels, keep_scores
    ):
        ax = ax_i[0]
        print("dec_attn_weights shape:", dec_attn_weights.shape)
        print("conv_features shape:", conv_features.shape)

        # in plot loop:
        attn_map = dec_attn_weights[0, :, idx, :].mean(0).view(h_attn, w_attn)
        ax.imshow(attn_map.cpu())
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')

        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[label])

    fig.tight_layout()
    out = os.path.join(out_file, "decoder.png")
    plt.savefig(out, dpi=200)
    plt.close(fig)
    print("Saved:", out)