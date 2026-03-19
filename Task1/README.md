# C5 Project - Week 1: Object Detection & Recognition

Implementation for Session 1 (Object Detection) of the C5 Project. This pipeline evaluates and fine-tunes Faster R-CNN, DETR, RT-DETR, and YOLO models on the KITTI-MOTS and DEArt datasets.


## 🏗️ Pipeline Structure

Our pipeline is modular and driven by JSON configuration files to standardize experiments across different architectures.

### Core Scripts

- **main.py**: The universal orchestrator used for training and evaluating Faster R-CNN, DETR, and RT-DETR. It handles training loops, validation, and logs metrics to Weights & Biases.

- **main_yolo.py**: A dedicated script for the YOLO pipeline. It utilizes the Ultralytics engine for training and a custom CocoEvaluator for standardized evaluation.

- **main_sweep.py**: Optimized for WandB Hyperparameter Sweeps. It allows for automated testing of different configurations (e.g., LoRA parameters or learning rates) without manual intervention.

- **qualitative_analysis.py**: A tool for generating visual results, including bounding box plots and Grad-CAM visualizations for model backbones and attention maps for DETR.


## 📁 Source Code (`src/`)

- **dataset.py**: Contains `KittiMotsDataset` and `DeArtDataset`. It handles RLE decoding, sequence mapping, and integrates Albumentations for advanced data augmentation.

- **models/**: Wrapper classes (`DetrWrapper`, `FasterRCNNWrapper`, `YoloWrapper`, `Rt_DetrWrapper`) that standardize inputs and outputs to a common `xyxy` format for evaluation.

- **metrics.py**: Implements a `CocoEvaluator` using `pycocotools` to provide consistent mAP (mean Average Precision) calculations across all model types.

- **qualitative_analysis/**: Specialized scripts for Model Interpretability, such as Grad-CAM for YOLO/FRCNN and attention map extractors for DETR encoders/decoders.


## ⚙️ Configuration

Experiments are controlled via JSON files in the `configs/` directory.

### Key Configuration Parameters

- **mode**: Choose between `"train"` or `"evaluate"`.

- **model_type**: `"faster_rcnn"`, `"detr"`, `"rt_detr"`, or `"yolo"`.

- **freeze_base**: Set to `true` to train only the final prediction heads (linear probing) or `false` for full fine-tuning.

- **apply_augmentations**: Supports `"none"`, `"basic"`, or `"weather"` (specialized for simulating adverse driving conditions).

- **use_lora**: (For DETR/RT-DETR) Enables Parameter-Efficient Fine-Tuning using Low-Rank Adaptation.


## 🚀 Usage Instructions

### 1. Training and Evaluation (General)

To run a standard experiment (Faster R-CNN, DETR, RT-DETR):

```bash
python main.py --config configs/FRNN/step1_frnn_head_baseline.json
```

### 2. YOLO Operations

YOLO training utilizes the Ultralytics framework and a dedicated wrapper:

```bash
python main_yolo.py --config configs/yolo_config.json
```

### 3. Hyperparameter Sweeps

To initialize and run a WandB sweep agent:

```bash
# Initialize the sweep to get the SWEEP_ID
wandb sweep configs/sweeps/sweep_lora.yaml

# Run the agent (replace with your specific details)
wandb agent <YOUR_ENTITY>/<PROJECT_NAME>/<SWEEP_ID>
```

### 4. Qualitative Analysis

To generate interpretability visualizations (Grad-CAM, Attention, Bboxes) for specific samples:

```bash
python qualitative_analysis.py \
    --samples "0000/000000.png" \
    --model_type detr \
    --model_paths "path/to/checkpoint.pt"
```

## 📊 Monitoring and Results

- **Weights & Biases**: All runs (loss, mAP per class, system memory, FPS) are logged live to your WandB project.

- **Local Storage**: Metrics and the best model checkpoints (best_model.pt) are saved in the results/ directory, organized by run name and unique ID.

- **Logs**: If running via SLURM (using provided job_* scripts), check the logs/ directory for standard output and error files.