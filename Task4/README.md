
# C5 Project - Week 4: Image Captioning

This repository contains the code and evaluation for **Week 4 of the C5: Multimodal Recognition** project (Master in Computer Vision Barcelona). The objective of this week is to explore, adapt, and evaluate **Vision-Language Transformers and Large Language Models (LLMs)** for image captioning.

## 📌 Project Overview

This project transitions from the CNN-RNN architectures of Week 3 to state-of-the-art Transformer-based models. We systematically evaluate zero-shot capabilities, investigate parameter-freezing fine-tuning strategies, and implement Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA for massive LLMs.

### Key Features & Experiments
* **Baseline Transformer Pipeline:** Zero-shot evaluation using the off-the-shelf pre-trained `nlpconnect/vit-gpt2-image-captioning` model (Vision Transformer encoder + GPT-2 autoregressive decoder).
* **Parameter-Freezing Strategies:** Evaluated the effects of domain adaptation and catastrophic forgetting by testing three fine-tuning configurations on the VizWiz dataset:
  * ViT (Train) + GPT-2 (Frozen)
  * ViT (Frozen) + GPT-2 (Train)
  * ViT (Train) + GPT-2 (Train) [Full Fine-tuning]
* **Advanced LLMs & PEFT (LoRA):** Integration of modern LLMs (e.g., Llama 3.2 1B/3B, Qwen) using Low-Rank Adaptation (LoRA) to adapt large text decoders to specific captioning tasks without updating full model weights, alongside zero-shot evaluation of multimodal models (Llama 3.2-11B Vision).
* **Inference & Decoding Constraints:** Replaced greedy decoding with **Beam Search** (`num_beams=4`), length constraints (`max_length=50`), and repetition penalties (`no_repeat_ngram_size=3`) to mitigate generative collapse.
* **Interpretability & Qualitative Analysis:** Generated cross-attention map overlays to visualize the spatial grounding of generated tokens, allowing us to isolate visual misclassifications from language head hallucinations.
* **Cross-Architecture Benchmarking:** Direct quantitative and qualitative comparison between Week 3's CNN/RNN inductive biases and Week 4's data-hungry Transformer architectures.

## 📂 Repository Structure

The codebase is designed to be modular, driven by JSON configurations, and integrated with Weights & Biases for experiment tracking.

```text
├── configs/                # JSON configurations for each experiment (Zero-shot, Freezing strategies)
│   ├── off_the_shelf.json
│   ├── vit_frozen_gpt2_train.json
│   ├── vit_train_gpt2_frozen.json
│   └── vit_train_gpt2_train.json
├── Qwen_code/              # Codebase specific to Qwen LLM experiments
│   ├── train_qwen.py
│   ├── evaluation_qwen.py
│   └── ...
├── dataset.py              # Custom PyTorch Dataset for loading images and parsing VizWiz captions
├── models.py               # Hugging Face wrappers (VisionEncoderDecoderModel) and LoRA integrations
├── tokenizer.py            # Tokenization logic leveraging pre-trained HF tokenizers
├── train.py                # Main training loop with W&B logging, evaluation, and attention plotting
├── job_high / job_low      # Cluster execution scripts
└── README.md
```

## 🚀 How to Run

The training and evaluation pipelines are controlled via configuration files passed to the main execution scripts.

**1. Run Baseline (Zero-Shot) Evaluation:**
To evaluate the off-the-shelf pre-trained model without fine-tuning:
```bash
python train.py --config configs/off_the_shelf.json
```

**2. Train a ViT-GPT2 Configuration:**
To run a specific parameter-freezing strategy (e.g., training the encoder while keeping the decoder frozen):
```bash
python train.py --config configs/vit_train_gpt2_frozen.json
```

**3. Train LLM Decoders using LoRA:**
To execute the Parameter-Efficient Fine-Tuning pipeline (e.g., using Qwen):
```bash
python Qwen_code/train_qwen.py --config Qwen_code/train_qwen.json
```

## 📊 Results Summary

Our experiments with Vision-Language Transformers yielded critical insights regarding pre-training, domain shift, and architectural limitations:

* **Visual Adaptation Wins (The Domain Shift Bottleneck):** The **ViT (Train) + GPT-2 (Frozen)** configuration yielded the highest overall metrics. Training the ViT allowed the model to map VizWiz's unique visual noise (obscured framing, lighting) into the strictly preserved, robust language latent space of GPT-2.
* **Catastrophic Forgetting in Decoders:** Unfreezing the GPT-2 decoder resulted in a massive quantitative collapse (BLEU-1 dropping from ~53 to ~27). The language head severely overfit the small VizWiz dataset, destroying its pre-trained grammatical structure and falling into repetitive n-gram loops.
* **Inductive Biases vs. Pre-training (W3 vs W4):** Despite the massive scale of pre-trained Transformers, our Week 3 CNN/RNN baseline maintained a slight performance edge across all metrics. This highlights that Vision Transformers lack built-in spatial translation invariance. Without these inductive biases, full block fine-tuning on a small dataset like VizWiz is suboptimal compared to naturally data-efficient CNNs.
* **Interpretability:** Cross-attention visualizations effectively separated vision failures (e.g., misclassifying a curled dog as a cat due to poor lighting) from language failures (hallucinating common MS COCO objects entirely absent from the frame).
