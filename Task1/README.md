Markdown

# C5 Project - Week 1: Object Detection & Recognition

Implementation for **Session 1 (Object Detection)** of the C5 Project. This pipeline evaluates and fine-tunes Faster R-CNN, DETR, and YOLO models on the KITTI-MOTS dataset.



## 🏗️ Pipeline Structure

Our pipeline is driven entirely by JSON configuration files to keep the terminal commands clean.

* `main.py`: The universal orchestrator (handles both inference and fine-tuning).
* `configs/`: Directory containing all experiment configurations.
* `src/dataset.py`: Unified PyTorch `Dataset` (`KittiMotsDataset`) handling mapping, RLE decoding, and Albumentations.
* `src/models/`: Wrappers for Faster R-CNN, DETR, and YOLO standardizing inputs/outputs to `xyxy` format.
* `src/metrics.py`: `CocoEvaluator` for on-the-fly mAP computation using `pycocotools`.

## ⚙️ Configuration

Control hyperparameters, modes, and WandB settings via JSON. 

**Example (`configs/detr_fine_tuning.json`):**
```json
{
    "mode": "train", 
    "model_type": "detr", 
    "data_dir": "/home/mcv/datasets/C5/KITTI-MOTS/",
    "wandb_project": "DETR",
    "run_name": "DETR_fine_tuning_head",
    "freeze_base": true, 
    "batch_size": 4,
    "epochs": 5,
    "learning_rate": 1e-4,
    "confidence_threshold": 0.5,
    "apply_augmentations": true
}
```
Set "freeze_base": true to train only the final prediction heads. Set "mode": "evaluate" for inference only.

## 🚀 Execution (SLURM)


Point the script to your desired config file at the bottom of the job file:
```Bash
#!/bin/bash 
...

python main.py --config configs/some_experiments/detr_fine_tuning.json
```

```Bash
sbatch job.sh
```

Monitor outputs in the logs/ directory and live mAP/loss metrics on your Weights & Biases dashboard.
