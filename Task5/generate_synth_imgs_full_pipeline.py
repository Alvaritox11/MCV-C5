"""
End-to-End Synthetic Blurry Image Pipeline
==========================================
Generates novel VizWiz-style captions, formats them into SD3 prompts, 
generates the images, and safely appends them to an existing dataset.

Usage:
  python end_to_end_synthetic_pipeline.py \
      --img-dir /ghome/group05/datasets/synthetic_vizwiz_blurry/images \
      --ann-out /ghome/group05/datasets/synthetic_vizwiz_blurry/synthetic_annotations.json \
      --hf-token YOUR_HF_TOKEN \
      --target-images 1000
"""

import argparse
import json
import logging
import os
import re
import random
import time
from pathlib import Path

import torch
from diffusers import AutoPipelineForText2Image
from huggingface_hub import InferenceClient

# ─────────────────────────────────────────────
# Logging & Setup
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

# ─────────────────────────────────────────────
# Prompt Pools (For Diversity)
# ─────────────────────────────────────────────
PREFIX_POOL = [
    "(extremely blurry:1.4), (out of focus:1.3), motion blur, shaky camera, low resolution, grain, smartphone photo, ",
    "(bad photo quality:1.4), smudged lens, completely out of focus, casual candid photography, ",
    "shaky hands, (severe motion blur:1.3), pixelated, cheap phone camera, dirty lens, ",
    "unfocused, (blurry:1.3), hastily taken picture, noisy image, low quality, ",
    "(extreme macro, out of focus:1.4), lens smear, blurry background, rushed photo, digital noise, ",
    "(camera shake:1.3), poor focus, soft edges, 2000s flip phone camera, highly compressed jpeg, "
]

SUFFIX_POOL = [
    ", harsh direct flash, flat lighting",
    ", underexposed, dimly lit room",
    ", overexposed, bad glare, washed out colors",
    ", taken in the dark, heavy image noise, high ISO",
    ", blown out highlights, bright reflection blocking view"
]

NEGATIVE_POOL = [
    "(masterpiece, best quality, ultra-detailed:1.3), 8k, professional photography, cinematic, DSLR",
    "sharp focus, clear text, perfect composition, 4k, studio lighting, bokeh, award-winning",
    "perfect composition, rule of thirds, centered, symmetrical, clean background, neat, tidy",
    "(crisp edges, sharp text, clear font:1.4), readable text, macro lens, high-definition, noise-free"
]

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def get_starting_id(ann_path):
    """Finds the highest ID in the existing JSON to resume safely."""
    if not os.path.exists(ann_path):
        return 0
    
    with open(ann_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return 0
            
    max_id = -1
    for item in data:
        sid = item.get("synthetic_id", "")
        if isinstance(sid, int):
            val = sid
        else:
            nums = re.findall(r'\d+', str(sid))
            val = int(nums[-1]) if nums else -1
        if val > max_id:
            max_id = val
            
    return max_id + 1 if max_id >= 0 else 0

def generate_caption_batch(client, batch_size=10):
    """Generates a batch of novel captions AND extracts the clean subject in one LLM call."""
    system_prompt = f"""You are an expert AI dataset creator for computer vision. 
    Your task is to generate completely NOVEL and UNIQUE image captions in the style of the VizWiz dataset.

    Domain constraints (subjects must reflect the daily needs of a visually impaired person taking a photo to get assistance):
        1. Technology & Displays: Computer monitors, laptop screens, smartphones, TV menus, thermostat screens, or appliance displays (e.g., microwave clocks, oven panels).
        2. Health & Medical: Prescription bottles, pill organizers, over-the-counter medicine boxes, vitamins, or medical devices (like blood glucose monitors).
        3. Documents & Printed Text: Mail envelopes, formal letters, receipts, utility bills, instruction manuals, greeting cards, or business cards.
        4. Food & Groceries: Canned goods, cereal boxes, spice jars, fresh produce, expiration dates, nutrition labels, restaurant menus, or frozen food packages.
        5. Clothing & Personal Care: Shirts, pants, clothing tags (seeking size or washing instructions), makeup palettes, lotion bottles, or shampoo versus conditioner.
        6. Household & Utility Items: Remote controls, keys, wallets, paper money (bills/coins), credit cards, CDs/DVDs, or thermostats.
        7. Environment & Navigation: Closed doors, window views (checking the weather), street signs, or generic room overviews.

    Rules:
        1. Generate EXACTLY {batch_size} unique captions.
        2. DO NOT copy existing dataset captions. Invent new specific scenarios.
        3. USE THIRD-PERSON DESCRIPTIVE LANGUAGE ONLY. Avoid first-person pronouns (I, me, my) and speculative phrases (I think, it seems to be). 
        4. DESCRIBE THE QUALITY OBJECTIVELY. If an image is blurry, state the visual condition as a fact (e.g., "The image is blurry," "Out of focus view of...").
        5. Output ONLY a valid JSON list of strings. Do not include markdown formatting or conversational text.
        6. ONLY DESCRIBE images with visual problems. For eg. Blurryness, unfocus, not contered...


    EACH OBJECT MUST HAVE TWO KEYS:
    - "caption": The full descriptive caption with the quality issue.
    - "subject": ONLY the physical objects and scene, stripping away all quality/blurriness adjectives.

    Example Output:
    [
        {{
            "caption": "Very blurry view of a white remote control on a brown couch.",
            "subject": "a white remote control on a brown couch"
        }},
        {{
            "caption": "Out of focus image making the label on an orange medicine bottle unreadable.",
            "subject": "an orange medicine bottle"
        }}
    ]"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": system_prompt}],
            max_tokens=1500,
            temperature=0.8,
        )
        output_text = response.choices[0].message.content.strip()
        output_text = re.sub(r"```json\n|\n```|```", "", output_text).strip()
        return json.loads(output_text)
    except Exception as e:
        log.error(f"Failed to generate caption batch: {e}")
        return []

# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def generate(args):
    img_dir = Path(args.img_dir)
    ann_out = Path(args.ann_out)
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Determine Starting ID
    global_id = get_starting_id(ann_out)
    log.info(f"Resuming generation. Next ID will be: syn_novel_{global_id:06d}")

    # Load existing data to append to it
    existing_data = []
    if ann_out.exists():
        with open(ann_out, 'r') as f:
            existing_data = json.load(f)

    # 2. Initialize Models
    device = get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    
    log.info("Initializing Hugging Face LLM Client...")
    llm_client = InferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=args.hf_token)

    log.info(f"Loading SD3 {MODEL_ID} on {device}...")
    sd_pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None
    ).to(device)
    sd_pipe.enable_attention_slicing()

    # 3. Generation Loop
    success_count = 0
    target = args.target_images

    log.info(f"Starting generation loop. Target: {target} new images.")

    while success_count < target:
        # Generate a batch of text prompts
        log.info("Generating a new batch of text concepts from LLM...")
        batch = generate_caption_batch(llm_client, batch_size=10)
        
        if not batch:
            time.sleep(5) # Pause briefly if API rate limit or error hit
            continue

        for item in batch:
            if success_count >= target:
                break
                
            caption = item.get("caption", "").strip()
            subject = item.get("subject", "").strip()
            
            if not caption or not subject:
                continue

            # Construct Prompts
            pos_prompt = f"{random.choice(PREFIX_POOL)}{subject}{random.choice(SUFFIX_POOL)}"
            neg_prompt = random.choice(NEGATIVE_POOL)
            
            synthetic_id_str = f"syn_novel_w{args.worker_id}_{global_id:06d}"
            img_name = f"{synthetic_id_str}.jpg"
            img_path = img_dir / img_name

            log.info(f"[{success_count+1}/{target}] Attempting: {img_name}")
            log.info(f"  > Caption: {caption}")
            
            t0 = time.time()
            try:
                # Generate Image
                result = sd_pipe(
                    prompt=pos_prompt,
                    negative_prompt=neg_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.cfg,
                    height=1024,
                    width=1024,
                    generator=torch.Generator(device=device).manual_seed(42 + global_id),
                )
                
                # Save Image
                img = result.images[0]
                img.save(img_path)
                
                elapsed = time.time() - t0
                log.info(f"  ✓ Success! Saved in {elapsed:.1f}s")

                # ---- CRITICAL: ONLY RUNS IF IMAGE SAVED SUCCESSFULLY ----
                # Append data and save JSON immediately
                new_entry = {
                    "synthetic_id": synthetic_id_str,
                    "caption": caption,
                    "image_path": img_name,
                    "sd3_positive_prompt": pos_prompt,
                    "sd3_negative_prompt": neg_prompt
                }
                
                existing_data.append(new_entry)
                with open(ann_out, "w") as f:
                    json.dump(existing_data, f, indent=4)
                
                # Increment counters
                global_id += 1
                success_count += 1
                # ---------------------------------------------------------

            except Exception as e:
                log.error(f"  ✗ Failed to generate or save {img_name}. Error: {e}")
                log.info("  Skipping to next prompt without incrementing ID.")
                torch.cuda.empty_cache() # Clear VRAM just in case it was an OOM error

    log.info(f"\n🎉 Finished! Successfully generated {success_count} new images.")
    log.info(f"Total dataset size is now {len(existing_data)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Synthetic Data Generation")
    parser.add_argument("--img-dir", required=True, help="Output folder for generated images")
    parser.add_argument("--ann-out", required=True, help="Path to the annotations JSON")
    parser.add_argument("--hf-token", required=True, help="Hugging Face API Token")
    parser.add_argument("--target-images", type=int, default=1000, help="Number of successful images to generate")
    parser.add_argument("--cfg", type=float, default=4.0, help="CFG guidance scale for SD3 (default: 4.0)")
    parser.add_argument("--steps", type=int, default=30, help="Number of denoising steps for SD3 (default: 30)")
    parser.add_argument("--worker-id", type=str, default="0", help="Unique identifier for parallel workers (e.g., 1, 2, A, B)")
    args = parser.parse_args()
    
    generate(args)