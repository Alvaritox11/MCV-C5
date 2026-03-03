
import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import os
import torch
import torchvision
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)
from transformers import DetrForObjectDetection, DetrImageProcessor
KITTI=True

def detr_encoder_cam(samples, device, output_path, model_path = "facebook/detr-resnet-50", checkpoint_path=None):
  transforms=T.Compose([
                        T.Resize(800),
                        T.ToTensor(),
                        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
  ])

  processor = DetrImageProcessor.from_pretrained(model_path)
  model = DetrForObjectDetection.from_pretrained(model_path, attn_implementation="eager")
  base_w = model.model.encoder.layers[0].self_attn.v_proj.weight.clone()

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
  model.eval()
     
  print('BBBBBBBBBBBB')
  print(checkpoint_path)
  #select image of choice from URL
  if KITTI:
        dataset_root = os.path.join('/home/mcv/datasets/C5/KITTI-MOTS/', "training", "image_02")
        image_path = os.path.join(dataset_root, samples)
        im = Image.open(image_path).convert("RGB")
  else:
        dataset_root = "/ghome/group05/c5_data"
        image_path = os.path.join(dataset_root, "images", samples)
        im = Image.open(image_path).convert("RGB")
  inputs = processor(images=im, return_tensors="pt")
     

  #using hooks to extract attention weights
  #extract conv features to derive the shape of the backbone output
  conv_features, enc_attn_weights, dec_attn_weights = [], [], []
  hooks=[
          model.model.backbone.register_forward_hook(
              lambda self, input, output: conv_features.append(output)),

          model.model.encoder.layers[-1].self_attn.register_forward_hook(
              lambda self, input, output: enc_attn_weights.append(output))
  ]

  outputs = model(**inputs)
  print('Outputs AAAA',outputs)
  for hook in hooks:
    hook.remove()

  print('ATT', enc_attn_weights)
  conv_features=conv_features[0]
  enc_attn_weights=enc_attn_weights[0]

  conv_shape = conv_features[0][-1][0].shape
  attn_avg = enc_attn_weights[1][0].mean(0)  # [seq_len, seq_len]

  # Now get h, w from pixel_mask like before
  pixel_mask = inputs["pixel_mask"][0]
  h_attn = (pixel_mask[:, 0].sum() / 32).round().int().item()
  w_attn = (pixel_mask[0, :].sum() / 32).round().int().item()
  shape = (h_attn, w_attn)

  sattn_shape1 = attn_avg.reshape(shape + shape)  # [h, w, h, w]
  sattn_shape2 = attn_avg                          # [seq_len, seq_len

     

  #select a Coordinate (X,Y) of choice
  idxs = (870, 280)

  x=idxs[0]
  y=idxs[1]

  # downsampling factor for the CNN backbone, from input image to output feature map 
  fact = 32

  #create the canvas
  fig = plt.figure(constrained_layout=True, figsize=(18 * 0.7, 8.5 * 0.7))
  gs = fig.add_gridspec(2, 3)

  ax_out = fig.add_subplot(gs[0, 0])
  ax_in  = fig.add_subplot(gs[1, 0])
  ax_img = fig.add_subplot(gs[:, 1:])

  idx = (idxs[0] // fact, idxs[1] // fact)

  # Top-left: where does this pixel attend?
  ax_out.imshow(sattn_shape1[idx[1], idx[0], ...], cmap='cividis', interpolation='nearest')
  ax_out.axis('off')
  ax_out.set_title(f'Outgoing attention from ({idxs[0]}, {idxs[1]})\n(where this pixel looks)', fontsize=9)

  # Bottom-left: who attends to this pixel?
  ax_in.imshow(sattn_shape1[..., idx[1], idx[0]], cmap='cividis', interpolation='nearest')
  ax_in.axis('off')
  ax_in.set_title(f'Incoming attention to ({idxs[0]}, {idxs[1]})\n(who looks at this pixel)', fontsize=9)

  # Center: original image with red dot
  ax_img.imshow(im)
  cx = ((idxs[0] // fact) + 0.5) * fact * (im.width  / (w_attn * fact))
  cy = ((idxs[1] // fact) + 0.5) * fact * (im.height / (h_attn * fact))
  ax_img.add_patch(plt.Circle((cx, cy), fact // 2, color='r'))
  ax_img.axis('off')
  ax_img.set_title(f'Reference point: ({idxs[0]}, {idxs[1]})', fontsize=9)

  output_path = os.path.join(output_path, "detr_encoder.png")
  plt.savefig(output_path)
  plt.close(fig)
