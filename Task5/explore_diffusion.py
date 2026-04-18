"""
Diffusion Model Exploration Pipeline
=====================================
Explores: DDPM vs DDIM | Positive/Negative Prompting | CFG Strength | Denoising Steps
Models: SD 2.1, SD 2.1 Turbo, SDXL, SDXL Turbo, SD 3.5 Medium, SD 3.5 Large Turbo
"""

import os
import json
import time
import argparse
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForText2Image,
    DDPMScheduler,
    DDIMScheduler,
)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    name: str          # human-readable short name
    hf_id: str         # HuggingFace model id
    pipeline_cls: str  # "sd" | "sdxl" | "auto"
    is_turbo: bool = False   # turbo/distilled models need few steps & no neg prompt
    skip_scheduler_swap: bool = False
    native_res: int = 512  


@dataclass
class ExperimentConfig:
    # ── Prompts ──────────────────────────────
    positive_prompt: str = (
        "a blurry photograph of a kitchen counter taken by a visually impaired person, "
        "out-of-focus, motion blur, low quality camera"
    )
    negative_prompt: str = (
        "sharp, in focus, professional photography, high resolution, clear"
    )

    # ── Schedulers ───────────────────────────
    schedulers: list = field(default_factory=lambda: ["ddim", "ddpm"])

    # ── CFG sweep ────────────────────────────
    cfg_values: list = field(default_factory=lambda: [1.0, 3.5, 7.5, 12.0, 20.0])

    # ── Denoising steps sweep ────────────────
    step_values: list = field(default_factory=lambda: [5, 15, 30, 50, 75, 100])

    # ── Image size ───────────────────────────
    height: int = 512
    width: int = 512

    # ── Seed ─────────────────────────────────
    seed: int = 42

    # ── Output root ──────────────────────────
    output_root: str = "outputs"


# ─────────────────────────────────────────────
# Model Zoo
# ─────────────────────────────────────────────
MODEL_ZOO = [
    ModelConfig(
        name="sd-2-1",
        hf_id="stabilityai/stable-diffusion-2-1",              
        pipeline_cls="sd",   
        native_res=512),

    ModelConfig(
        name="sd-2-1-turbo",       
        hf_id="stabilityai/sd-turbo",                          
        pipeline_cls="auto", native_res=512,  
        is_turbo=True),

    ModelConfig(
        name="sdxl-base",          
        hf_id="stabilityai/stable-diffusion-xl-base-1.0",      
        pipeline_cls="sdxl", 
        native_res=1024),

    ModelConfig(
        name="sdxl-turbo",         
        hf_id="stabilityai/sdxl-turbo",                        
        pipeline_cls="auto", 
        native_res=512,  
        is_turbo=True),

    ModelConfig(
        name="sd-3-5-medium",      
        hf_id="stabilityai/stable-diffusion-3.5-medium",       
        pipeline_cls="auto", 
        native_res=1024, 
        skip_scheduler_swap=True),

    ModelConfig(
        name="sd-3-5-large-turbo", 
        hf_id="stabilityai/stable-diffusion-3.5-large-turbo",  
        pipeline_cls="auto", 
        native_res=1024, 
        is_turbo=True, 
        skip_scheduler_swap=True),
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(device: str):
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def load_pipeline(model: ModelConfig, device: str, dtype):
    """Load the appropriate pipeline for the model."""
    log.info(f"  Loading {model.hf_id} ...")
    kwargs = dict(torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None)

    if model.pipeline_cls == "sd":
        pipe = StableDiffusionPipeline.from_pretrained(model.hf_id, **kwargs)
    elif model.pipeline_cls == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(model.hf_id, **kwargs)
    else:  # "auto"
        pipe = AutoPipelineForText2Image.from_pretrained(model.hf_id, **kwargs)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()          # saves VRAM
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass                                 # xformers not installed – fine
    return pipe


def apply_scheduler(pipe, scheduler_name: str, model: ModelConfig):
    """Swap the pipeline scheduler in-place."""
    if model.is_turbo or model.skip_scheduler_swap:
        return   # ← don't touch SD 3.5's scheduler

    base_config = pipe.scheduler.config

    if scheduler_name == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(base_config)
    elif scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(base_config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def make_generator(seed: int, device: str):
    return torch.Generator(device=device).manual_seed(seed)


def save_image(img: Image.Image, path: Path, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"    ✓ Saved → {path.name}")


def turbo_steps(requested_steps: int) -> int:
    """Turbo/distilled models work best with very few steps."""
    return min(requested_steps, 4)


# ─────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────
def run_scheduler_experiment(pipe, model: ModelConfig, cfg: ExperimentConfig,
                              model_dir: Path, device: str):
    """DDPM vs DDIM — fixed CFG=7.5, steps=30."""
    exp_dir = model_dir / "1_scheduler_comparison"
    log.info("  [Exp 1] Scheduler comparison (DDPM vs DDIM)")

    for sched in cfg.schedulers:
        apply_scheduler(pipe, sched, model)
        steps = turbo_steps(30) if model.is_turbo else 30
        neg = "" if model.is_turbo else cfg.negative_prompt

        t0 = time.time()
        result = pipe(
            prompt=cfg.positive_prompt,
            negative_prompt=neg if neg else None,
            num_inference_steps=steps,
            guidance_scale=1.0 if model.is_turbo else 7.5,
            height=model.native_res,
            width=model.native_res,
            generator=make_generator(cfg.seed, device),
        )
        elapsed = time.time() - t0

        img = result.images[0]
        label = "turbo_default" if model.is_turbo else sched
        fname = exp_dir / f"scheduler_{label}.png"
        save_image(img, fname, {
            "experiment": "scheduler_comparison",
            "model": model.name,
            "scheduler": label,
            "steps": steps,
            "cfg": 1.0 if model.is_turbo else 7.5,
            "elapsed_s": round(elapsed, 2),
            "prompt": cfg.positive_prompt,
            "negative_prompt": neg,
        })
        if model.is_turbo:
            break   # only one scheduler variant for turbo


def run_prompting_experiment(pipe, model: ModelConfig, cfg: ExperimentConfig,
                              model_dir: Path, device: str):
    """Positive only vs Positive+Negative prompting."""
    exp_dir = model_dir / "2_prompting"
    log.info("  [Exp 2] Positive vs Positive+Negative prompting")

    apply_scheduler(pipe, "ddim", model)
    steps = turbo_steps(30) if model.is_turbo else 30
    guidance = 1.0 if model.is_turbo else 7.5

    variants = [
        ("positive_only", cfg.positive_prompt, ""),
        ("positive_and_negative", cfg.positive_prompt, cfg.negative_prompt),
    ]
    for label, pos, neg in variants:
        if model.is_turbo and neg:
            # turbo models largely ignore neg prompts; still generate for comparison
            pass
        t0 = time.time()
        result = pipe(
            prompt=pos,
            negative_prompt=neg if neg else None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=model.native_res,
            width=model.native_res,
            generator=make_generator(cfg.seed, device),
        )
        elapsed = time.time() - t0
        fname = exp_dir / f"prompting_{label}.png"
        save_image(result.images[0], fname, {
            "experiment": "prompting",
            "model": model.name,
            "variant": label,
            "steps": steps,
            "cfg": guidance,
            "elapsed_s": round(elapsed, 2),
            "prompt": pos,
            "negative_prompt": neg,
        })


def run_cfg_experiment(pipe, model: ModelConfig, cfg: ExperimentConfig,
                        model_dir: Path, device: str):
    """Sweep CFG values — fixed DDIM, steps=30."""
    exp_dir = model_dir / "3_cfg_sweep"
    log.info("  [Exp 3] CFG strength sweep")

    apply_scheduler(pipe, "ddim", model)
    steps = turbo_steps(30) if model.is_turbo else 30
    neg = "" if model.is_turbo else cfg.negative_prompt

    cfg_list = cfg.cfg_values 
    for g in cfg_list:
        t0 = time.time()
        result = pipe(
            prompt=cfg.positive_prompt,
            negative_prompt=neg if neg else None,
            num_inference_steps=steps,
            guidance_scale=g,
            height=model.native_res,
            width=model.native_res,
            generator=make_generator(cfg.seed, device),
        )
        elapsed = time.time() - t0
        fname = exp_dir / f"cfg_{str(g).replace('.', '_')}.png"
        save_image(result.images[0], fname, {
            "experiment": "cfg_sweep",
            "model": model.name,
            "cfg": g,
            "steps": steps,
            "elapsed_s": round(elapsed, 2),
            "prompt": cfg.positive_prompt,
            "negative_prompt": neg,
        })


def run_steps_experiment(pipe, model: ModelConfig, cfg: ExperimentConfig,
                          model_dir: Path, device: str):
    """Sweep denoising steps — fixed DDIM, CFG=7.5."""
    exp_dir = model_dir / "4_steps_sweep"
    log.info("  [Exp 4] Denoising steps sweep")

    apply_scheduler(pipe, "ddim", model)
    guidance = 1.0 if model.is_turbo else 7.5
    neg = "" if model.is_turbo else cfg.negative_prompt
    step_list = cfg.step_values

    for steps in step_list:
        t0 = time.time()
        result = pipe(
            prompt=cfg.positive_prompt,
            negative_prompt=neg if neg else None,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=model.native_res,
            width=model.native_res,
            generator=make_generator(cfg.seed, device),
        )
        elapsed = time.time() - t0
        fname = exp_dir / f"steps_{steps:03d}.png"
        save_image(result.images[0], fname, {
            "experiment": "steps_sweep",
            "model": model.name,
            "steps": steps,
            "cfg": guidance,
            "elapsed_s": round(elapsed, 2),
            "prompt": cfg.positive_prompt,
            "negative_prompt": neg,
        })


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run_pipeline(
    models_to_run: Optional[list] = None,
    cfg: Optional[ExperimentConfig] = None,
    experiments: Optional[list] = None,
):
    if cfg is None:
        cfg = ExperimentConfig()
    if experiments is None:
        experiments = ["scheduler", "prompting", "cfg", "steps"]

    device = get_device()
    dtype = get_dtype(device)
    log.info(f"Device: {device} | dtype: {dtype}")

    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Save global config
    with open(output_root / "experiment_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    selected_models = [m for m in MODEL_ZOO
                       if models_to_run is None or m.name in models_to_run]

    for model in selected_models:
        log.info(f"\n{'='*60}")
        log.info(f"Model: {model.name}  ({model.hf_id})")
        log.info(f"{'='*60}")
        model_dir = output_root / model.name

        try:
            pipe = load_pipeline(model, device, dtype)
        except Exception as e:
            log.error(f"  Failed to load {model.name}: {e}")
            continue

        exp_map = {
            "scheduler": run_scheduler_experiment,
            "prompting": run_prompting_experiment,
            "cfg": run_cfg_experiment,
            "steps": run_steps_experiment,
        }

        for exp_name in experiments:
            if exp_name in exp_map:
                try:
                    exp_map[exp_name](pipe, model, cfg, model_dir, device)
                except Exception as e:
                    log.error(f"  Experiment '{exp_name}' failed for {model.name}: {e}")

        # Free VRAM before next model
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

    log.info(f"\n✅ Pipeline complete. Results saved to: {output_root.resolve()}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Exploration Pipeline")
    parser.add_argument(
        "--models", nargs="+",
        choices=[m.name for m in MODEL_ZOO],
        default=None,
        help="Which models to run (default: all). Example: --models sd-2-1 sdxl-base",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=["scheduler", "prompting", "cfg", "steps"],
        default=["scheduler", "prompting", "cfg", "steps"],
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Root output directory (default: outputs/)",
    )
    parser.add_argument(
        "--height", type=int, default=512,
        help="Image height in pixels (default: 512)",
    )
    parser.add_argument(
        "--width", type=int, default=512,
        help="Image width in pixels (default: 512)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--positive-prompt", type=str,
        default=(
            "a blurry photograph of a kitchen counter taken by a visually impaired person, "
            "out-of-focus, motion blur, low quality camera"
        ),
        help="Positive prompt for generation",
    )
    parser.add_argument(
        "--negative-prompt", type=str,
        default="sharp, in focus, professional photography, high resolution, clear",
        help="Negative prompt for generation",
    )
    parser.add_argument(
        "--cfg-values", nargs="+", type=float,
        default=[1.0, 3.5, 7.5, 12.0],
        help="CFG scale values to sweep (default: 1.0 3.5 7.5 12.0)",
    )
    parser.add_argument(
        "--step-values", nargs="+", type=int,
        default=[5, 15, 30, 50],
        help="Denoising step counts to sweep (default: 5 15 30 50)",
    )

    args = parser.parse_args()

    exp_cfg = ExperimentConfig(
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        cfg_values=args.cfg_values,
        step_values=args.step_values,
        height=args.height,
        width=args.width,
        seed=args.seed,
        output_root=args.output,
    )

    run_pipeline(
        models_to_run=args.models,
        cfg=exp_cfg,
        experiments=args.experiments,
    )