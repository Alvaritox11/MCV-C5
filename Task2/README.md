# C5 Project - Week 2: Object Segmentation

This repository contains the code and evaluation for **Week 2 of the C5: Multimodal Recognition** project (Master in Computer Vision Barcelona). The objective of this week is to explore, evaluate, and fine-tune the **Segment Anything Model (SAM)** for instance and semantic segmentation on the KITTI-MOTS dataset.

## 📌 Project Overview

This project evaluates the performance of SAM using different prompting paradigms and compares zero-shot vision-language models against domain-specific detectors. 

### Key Features & Tasks Completed
* **Task A (Point Prompts):** Evaluated SAM using various point-based spatial priors (Single points, SIFT keypoints, and Uniform 3x3 Grids).
* **Task B (Text Prompts):** Implemented **Grounded SAM** (Grounding DINO + SAM) for zero-shot text-to-mask segmentation using rich vocabulary prompts.
* **Task C (Box Prompts):** Integrated our fine-tuned YOLO model (from Week 1) to extract bounding boxes and feed them as spatial prompts to SAM.
* **Task D & G (Analysis):** Conducted deep qualitative and quantitative comparisons focusing on the robustness, spatial ambiguity, and computational efficiency of each prompt type.
* **Task E (Fine-Tuning):** Fine-tuned the Prompt and Mask Decoders of SAM on KITTI-MOTS to adapt to the specific dataset domain.
* **Task F (Domain Shift):** Evaluated the zero-shot generalization of our fine-tuned models on the DEART dataset.
* **Task H (Semantic Segmentation):** Extended the instance segmentation pipeline to perform semantic segmentation.

## 📂 Repository Structure

The code follows a scalable Adapter/Wrapper pattern to standardize inference and evaluation across different frameworks (Hugging Face, Ultralytics).

```text
├── data/                   # (Not uploaded) KITTI-MOTS dataset
├── src/
│   ├── dataset.py          # KittiMotsDataset parser and COCO format mapper
│   ├── metrics.py          # COCO evaluation logic (pycocotools wrappers)
│   └── models/
│       ├── sam_wrapper.py           # Wrapper for pre-trained and fine-tuned SAM
│       ├── grounded_sam_wrapper.py  # Wrapper for Grounding DINO + SAM
│       └── sam_finetuned_wrapper.py          # Wrapper for fine-tuning
├── main.py                 # Main entry point for inference and evaluation
├── requirements.txt        # Python dependencies
└── README.md

```

## 🚀 How to Run

The pipeline is controlled via `main.py`. You can select the experiment to run via arguments.

**1. Run Evaluation with Grounded SAM (Text Prompts):**

```bash
python main.py --config configs/eval_*.json
```

## 📊 Results Summary

Our experiments demonstrate a clear trade-off between semantic generalization and computational efficiency:

* **Point Prompts:** Highly efficient but vulnerable to spatial ambiguity ("part-vs-whole" problem).
* **Grounded SAM (Text):** Highest semantic robustness for zero-shot generalization, but computationally heavy.
* **YOLO + SAM (Boxes):** Offers the best balance of speed and domain-specific precision, effectively resolving spatial ambiguity for narrow objects like pedestrians.

