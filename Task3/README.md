# C5 Project - Week 3: Image Captioning

This repository contains the code and evaluation for **Week 3 of the C5: Multimodal Recognition** project (Master in Computer Vision Barcelona). The objective of this week is to design, train, and evaluate **Encoder-Decoder Image Captioning** architectures.

## 📌 Project Overview

This project explores a complete image captioning pipeline, systematically improving upon a basic baseline through architectural modifications, advanced text representation, and training techniques.

### Key Features & Experiments
* **Baseline Pipeline:** Implemented an end-to-end image captioning model using ResNet-18 (Encoder) and a 1-layer GRU (Decoder) with character-level tokenization.
* **Encoder Architectures:** Evaluated and compared various pre-trained CNN feature extractors, including ResNet-18, ResNet-50, VGG-16, and VGG-19.
* **Text Representation Strategy:** Transitioned from Character-level generation to Subword-level and Word-level representations to solve spelling and stuttering issues.
* **Training Enhancements:** Integrated **Teacher Forcing** to stabilize training and accelerate convergence, alongside comprehensive hyperparameter optimization using Weights & Biases (W&B) sweeps.
* **Advanced Decoders:** Compared standard GRU cells against multi-layer LSTMs.
* **Attention Mechanisms:** Implemented additive visual attention to allow the decoder to focus on specific spatial features of the image during text generation.
* **Robust Evaluation:** Computed sentence-level metrics including BLEU (1 and 4), ROUGE-L, and METEOR (focusing heavily on METEOR for semantic evaluation), alongside deep qualitative analysis to identify generation bottlenecks like "safe guess" loops.

## 📂 Repository Structure

The codebase is designed to be modular and is heavily driven by JSON configuration files for reproducible experiments.

```text
├── configs/                # JSON configurations for each experiment (Encoders, Tokenizers, TF)
│   ├── sweeps/             # YAML configurations for W&B hyperparameter optimization
│   └── ...                 
├── dataset.py              # Custom PyTorch Dataset for loading images and captions
├── models.py               # Neural network architectures (CNN Encoders, RNN/LSTM Decoders, Attention)
├── tokenizer.py            # Custom vocabulary builder and tokenization logic (Char/Subword/Word)
├── train.py                # Main training loop with W&B logging and evaluation integration
├── sweep.py                # Entry point for running W&B hyperparameter sweeps
├── evaluate_ckpt.py        # Script to load a .pth checkpoint and generate predictions.json
├── evaluate_offline.py     # Offline HF evaluate script to compute BLEU, ROUGE, and METEOR per image
├── compute_metrics.sh      # Bash script to automate evaluation runs
├── Baseline Model and Metrics.ipynb # Exploratory notebook for baseline results
├── qualitative.ipynb       # Notebook for rendering qualitative predictions and attention maps
└── README.md
```

## 🚀 How to Run

The entire training and evaluation pipeline is controlled via configuration files passed to the main scripts.

**1. Train a Model:**
To train a model from scratch, pass the desired configuration file to `train.py`.
```bash
python train.py --config configs/exp-resnet50-gru-word_lr1e-4_bs64_maxlen50_teacher_forcing.json
```

**2. Generate Predictions from a Checkpoint:**
To generate a `predictions.json` file from a saved checkpoint for offline evaluation.
```bash
python evaluate_ckpt.py --config configs/baseline.json --checkpoint outputs/run_name/checkpoints/best_model.pth --epoch_label epoch_best
```

**3. Compute Offline Metrics:**
To calculate detailed BLEU, ROUGE, and METEOR scores for your generated predictions.
```bash
python evaluate_offline.py --preds_file outputs/run_name/predictions.json --epoch epoch_best --output_dir metrics_computed --run_name resnet50_tf
```

**4. Run a Hyperparameter Sweep:**
```bash
python sweep.py --sweep_config configs/sweeps/sweep.yaml
```

## 📊 Results Summary

Our systematic experiments revealed crucial insights into sequence-to-sequence model behaviors:

* **Encoder Selection:** ResNet-50 proved to be the most capable feature extractor, slightly outperforming VGG-16/19 and ResNet-18.
* **Text Representation:** Moving from Character-level to Word-level tokenization provided a massive qualitative leap, solving spelling failures (e.g., generating "A con" instead of "A computer").
* **Teacher Forcing Impact:** Applying Teacher Forcing yielded the largest quantitative jump in our pipeline (doubling the METEOR score) by preventing cascading sequence errors early in training.
* **The "Safe Guess" Phenomenon:** Despite high BLEU/ROUGE scores on advanced configurations (like the 2-layer LSTM with Teacher Forcing), qualitative analysis revealed the model learned to exploit generic dataset templates (e.g., appending *"is on a table"*). This highlighted the necessity of METEOR and manual visual inspection over purely n-gram based metrics.