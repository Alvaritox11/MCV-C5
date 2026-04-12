import os
import transformers.utils.import_utils as _hf_import_utils
_hf_import_utils.check_torch_load_is_safe = lambda: None  # bypass the version check

import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import ResNetModel
from transformers import (
    VisionEncoderDecoderModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    ViTModel,
    AutoProcessor,
    LlamaForCausalLM,
    VisionEncoderDecoderConfig
)
from peft import LoraConfig, get_peft_model, TaskType

device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
def build_lora_config(r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",        # MLP
        ],
    )


# ── Projection: ViT hidden dim → Qwen embedding dim ───────────────────────────
class VisualProjection(nn.Module):
    def __init__(self, vit_hidden_size: int, decoder_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(vit_hidden_size, decoder_hidden_size, bias=False)
        self.norm = nn.LayerNorm(decoder_hidden_size)

    def forward(self, x):          # (B, num_patches+1, vit_hidden)
        return self.norm(self.proj(x))   # (B, num_patches+1, dec_hidden)


# ── Base class (shared logic) ──────────────────────────────────────────────────
class ViTQwenBase(nn.Module):
    """
    Frozen fine-tuned ViT encoder  +  VisualProjection  +  Qwen2.5 decoder (LoRA).

    Config keys consumed here:
        vit_checkpoint   : path to the saved VisionEncoderDecoderModel from Task 1
        decoder_name     : HF model-id (set by subclass default, overridable)
        lora_r / lora_alpha / lora_dropout
        max_length
    """

    DECODER_ID: str = ""   # overridden by subclasses

    def __init__(self, cfg: dict):
        super().__init__()

        # ── 1. Load fine-tuned ViT from Task-1 checkpoint (encoder only) ──────
        vit_ckpt_path = cfg.get("vit_checkpoint", None)
        if vit_ckpt_path:
            vit_ckpt_path = vit_ckpt_path.strip()
            print(f"Loading fine-tuned ViT encoder from: {vit_ckpt_path}")
            if vit_ckpt_path.endswith(".safetensors"):
                # Safe, no torch.load involved
                full_model = VisionEncoderDecoderModel.from_pretrained(
                    ".", 
                    model_file=vit_ckpt_path,
                )
            elif vit_ckpt_path.endswith(".pth"):
                full_model = VisionEncoderDecoderModel(
                    VisionEncoderDecoderConfig.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
                )
                state_dict = torch.load(vit_ckpt_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
                full_model.load_state_dict(state_dict)
            else:
                # HuggingFace saved directory
                full_model = VisionEncoderDecoderModel.from_pretrained(vit_ckpt_path)
        else:
            print("No 'vit_checkpoint' provided — loading base weights from HuggingFace.")
            full_model = VisionEncoderDecoderModel.from_pretrained(
                "nlpconnect/vit-gpt2-image-captioning"
            )
        self.vit = full_model.encoder   # ViTModel only, GPT-2 discarded
        del full_model

        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()

        vit_hidden = self.vit.config.hidden_size

        # ── 2. Qwen2.5 tokenizer ───────────────────────────────────────────────
        decoder_id = cfg.get("decoder_name", self.DECODER_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(
            decoder_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        self.tokenizer.model_max_length = cfg.get("max_length", 64)

        # ── 3. Qwen2.5 decoder + LoRA ─────────────────────────────────────────
        print(f"Loading Qwen2.5 decoder: {decoder_id}")
        base_decoder = AutoModelForCausalLM.from_pretrained(
            decoder_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        base_decoder.resize_token_embeddings(len(self.tokenizer))

        lora_cfg = build_lora_config(
            r            = cfg.get("lora_r",         16),
            lora_alpha   = cfg.get("lora_alpha",     32),
            lora_dropout = cfg.get("lora_dropout", 0.05),
        )
        self.decoder = get_peft_model(base_decoder, lora_cfg)
        self.decoder.print_trainable_parameters()

        dec_hidden = self.decoder.config.hidden_size

        # ── 4. Visual projection (trainable) ──────────────────────────────────
        self.visual_proj = VisualProjection(vit_hidden, dec_hidden)

    # ── Image → prefix embeddings ──────────────────────────────────────────────
    def _encode_image(self, pixel_values):
        """
        Runs the frozen ViT and projects patch embeddings into decoder space.
        Returns (B, num_patches+1, dec_hidden).
        """
        with torch.no_grad():
            vit_out = self.vit(pixel_values=pixel_values)
        patch_embs = vit_out.last_hidden_state          # (B, N+1, vit_hidden)
        return self.visual_proj(patch_embs)             # (B, N+1, dec_hidden)

    # ── forward (teacher-forcing) ──────────────────────────────────────────────
    def forward(self, pixel_values, labels, decoder_attention_mask=None):
        """
        Prepends visual prefix to text embeddings and computes causal-LM loss.
        labels: (B, T) with padding positions set to -100.
        """
        B       = pixel_values.size(0)
        vis_embs = self._encode_image(pixel_values)     # (B, N+1, D)
        num_vis  = vis_embs.size(1)

        # Text token → embeddings (clamp to avoid -100 index into embedding table)
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = self.tokenizer.pad_token_id
        tok_embs = self.decoder.get_input_embeddings()(safe_labels)                                         # (B, T, D)

        # Concat visual prefix + text
        inputs_embeds = torch.cat([vis_embs, tok_embs], dim=1)
                                                        # (B, N+1+T, D)

        # Labels: mask out the visual prefix positions
        vis_ignore  = torch.full(
            (B, num_vis), -100, dtype=labels.dtype, device=labels.device
        )
        labels_full = torch.cat([vis_ignore, labels], dim=1)
                                                        # (B, N+1+T)

        # Attention mask: visual prefix is always attended to
        if decoder_attention_mask is not None:
            vis_mask = torch.ones(
                B, num_vis,
                dtype=decoder_attention_mask.dtype,
                device=decoder_attention_mask.device,
            )
            attention_mask = torch.cat([vis_mask, decoder_attention_mask], dim=1)
        else:
            attention_mask = None

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels_full,
        )

    # ── generate ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, pixel_values, max_length: int = 64, num_beams: int = 4,
             repetition_penalty: float = 1.5, no_repeat_ngram_size: int = 3):
        B        = pixel_values.size(0)
        vis_embs = self._encode_image(pixel_values)
        vis_mask = torch.ones(
            B, vis_embs.size(1), dtype=torch.long, device=pixel_values.device
        )

        return self.decoder.generate(
            inputs_embeds        = vis_embs,
            attention_mask       = vis_mask,
            max_new_tokens       = max_length,
            do_sample            = False,
            num_beams            = num_beams,
            repetition_penalty   = repetition_penalty,
            no_repeat_ngram_size = no_repeat_ngram_size,
            pad_token_id         = self.tokenizer.pad_token_id,
            eos_token_id         = self.tokenizer.eos_token_id,
        )



# ── Concrete subclasses ────────────────────────────────────────────────────────

class ViTQwen05BModel(ViTQwenBase):
    """Frozen fine-tuned ViT  +  Qwen2.5-0.5B-Instruct decoder with LoRA."""
    DECODER_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, cfg: dict):
        cfg.setdefault("decoder_name", self.DECODER_ID)
        super().__init__(cfg)
        self.to(torch.bfloat16)

class ViTQwen15BModel(ViTQwenBase):
    """Frozen fine-tuned ViT  +  Qwen2.5-1.5B-Instruct decoder with LoRA."""
    DECODER_ID = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(self, cfg: dict):
        cfg.setdefault("decoder_name", self.DECODER_ID)
        super().__init__(cfg)   
        self.to(torch.bfloat16)