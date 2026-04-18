# Task 5 — Synthetic Data Augmentation for VizWiz Image Captioning

This task explores synthetic data generation using diffusion models to augment the
[VizWiz](https://vizwiz.org/) dataset, and evaluates its impact on an image captioning
model trained with encoder–decoder architectures.

---

## 📁 Project Structure

```
Task5/
├── configs/                              # JSON config files for training runs
├── analyse_results.py                    # Generates comparison grids + HTML report
├── calc_fid.py                           # Computes FID score between datasets
├── dataset.py                            # VizWizDataset, SyntheticDataset, dataloaders
├── explore_diffusion.py                  # Diffusion model exploration pipeline
├── generate_synth_imgs_full_pipeline.py  # End-to-end synthetic image + caption generation
├── models.py                             # Encoder–decoder captioning model with attention
├── tokenizer.py                          # Char, Word, and Subword (BERT) tokenizers
├── train.py                              # Standard training loop with W&B logging
├── train_kfold.py                        # K-fold cross-validation training
└── tsne_plot.py                          # t-SNE visualization of feature distributions
```

---

## 🔬 Diffusion Model Exploration

`explore_diffusion.py` systematically benchmarks six Stable Diffusion variants:

| Model | HuggingFace ID |
|---|---|
| SD 2.1 | `stabilityai/stable-diffusion-2-1` |
| SD Turbo | `stabilityai/sd-turbo` |
| SDXL Base | `stabilityai/stable-diffusion-xl-base-1.0` |
| SDXL Turbo | `stabilityai/sdxl-turbo` |
| SD 3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` |
| SD 3.5 Large Turbo | `stabilityai/stable-diffusion-3.5-large-turbo` |

Four experiments are run per model:

1. **Scheduler comparison** — DDPM vs. DDIM at fixed CFG=7.5, steps=30
2. **Prompting** — positive-only vs. positive+negative prompt
3. **CFG sweep** — guidance scale over `[1.0, 3.5, 7.5, 12.0, 20.0]`
4. **Steps sweep** — denoising steps over `[5, 15, 30, 50, 75, 100]`

```bash
python explore_diffusion.py \
    --models sd-2-1 sdxl-base \
    --experiments scheduler cfg steps \
    --output outputs/ \
    --seed 42
```

After running, generate a visual HTML report:

```bash
python analyse_results.py --output outputs/
```

This produces per-model image grids, a timing bar chart, and `outputs/report.html`.

---

## 🖼️ Synthetic Data Generation

`generate_synth_imgs_full_pipeline.py` generates VizWiz-style blurry images using
SD 3.5 Medium, guided by captions produced by a Llama 3 LLM via the HuggingFace
Inference API.

**Pipeline:**

1. LLM generates `(caption, subject)` pairs representing scenes a visually impaired person might photograph.
2. Subject is wrapped in randomised blur/noise prompt prefixes and suffixes.
3. SD 3.5 Medium generates the image.
4. Annotations are appended to a JSON file after each successful save (resume-safe).

```bash
python generate_synth_imgs_full_pipeline.py \
    --img-dir /path/to/output/images \
    --ann-out /path/to/synthetic_annotations.json \
    --hf-token YOUR_HF_TOKEN \
    --target-images 1000 \
    --cfg 4.0 \
    --steps 30 \
    --worker-id 0
```

Multiple workers can run in parallel using distinct `--worker-id` values.

---

## 🧠 Captioning Model

`models.py` implements a configurable encoder–decoder model (`BaselineModel`):

**Encoders:** ResNet-18, ResNet-50, VGG-16, VGG-19

**Decoders:** GRU or LSTM (configurable depth and dropout)

**Attention:** Optional additive (Bahdanau) spatial attention with learnable hidden state initialisation from mean encoder features.

---

## 📝 Tokenizers

Three tokenisation strategies are available, selected via `text_level` in the config:

| `text_level` | Class | Notes |
|---|---|---|
| `char` | `CharTokenizer` | Fixed 84-token character vocabulary |
| `word` | `WordTokenizer` | Built from training captions; configurable min frequency |
| `subword` | `SubwordTokenizer` | Wraps `bert-base-uncased`; 30k vocab |

---

## 🏋️ Training

### Standard training

```bash
python train.py --config configs/resnet50_gru.json
```

### K-Fold cross-validation

```bash
python train_kfold.py --config configs/resnet50_gru_kfold.json
```

Both scripts log metrics and qualitative plots to [Weights & Biases](https://wandb.ai).
Metrics reported: **BLEU-1**, **BLEU-2**, **ROUGE-L**, **METEOR**.

### Example config keys

```json
{
  "run_name": "resnet50_gru_word",
  "encoder": "ResNet-50",
  "decoder": "GRU",
  "decoder_hidden_dim": 512,
  "use_attention": true,
  "text_level": "word",
  "max_len": 50,
  "batch_size": 32,
  "epochs": 20,
  "learning_rate": 1e-4,
  "teacher_forcing": true,
  "freeze_encoder": false,
  "k_folds": 5,
  "early_stopping": true,
  "early_stopping_patience": 3,
  "train_ann_path": "/path/to/train/annotations.json",
  "val_ann_path": "/path/to/val/annotations.json",
  "train_img_dir": "/path/to/train/images",
  "val_img_dir": "/path/to/val/images",
  "synthetic_ann_path": "/path/to/synthetic_annotations.json",
  "synthetic_img_dir": "/path/to/synthetic/images",
  "wandb_project": "mcv-c5",
  "wandb_entity": "your-entity"
}
```

---

## 📊 Dataset Quality Analysis

### FID Score

```bash
python calc_fid.py
```

Computes Fréchet Inception Distance between real VizWiz images and synthetic/deART
datasets using [clean-fid](https://github.com/GaParmar/clean-fid).

### t-SNE Distribution Plot

```bash
python tsne_plot.py
```

Extracts ResNet-50 features from up to 3000 images per dataset and plots a 2D t-SNE
comparing Real VizWiz, Synthetic SD3, and deART distributions. Output saved to
`distribution_comparison_tsne.png`.

---

## ⚙️ Requirements

```
torch
torchvision
transformers
diffusers
Pillow
matplotlib
wandb
evaluate
scikit-learn
clean-fid
huggingface_hub
opencv-python
tqdm
```

---

## 📌 Notes

- Turbo/distilled models (SD Turbo, SDXL Turbo, SD 3.5 Large Turbo) run with ≤4 denoising steps and do not support negative prompts meaningfully.
- SD 3.5 models skip scheduler swapping as their internal schedulers are not interchangeable with DDPM/DDIM.
- Synthetic data is seamlessly merged into training via `ConcatDataset`; the tokenizer vocabulary is built from both real and synthetic captions.
- K-fold validation uses only real data in the held-out fold; synthetic samples are added exclusively to training folds.
