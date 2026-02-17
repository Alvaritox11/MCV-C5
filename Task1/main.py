import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import KittiMotsDataset
from src.metrics import CocoEvaluator

# After being implemented, import here the modesl
from src.models.detr_wrapper import DetrWrapper
# from src.models.yolo_wrapper import YoloWrapper
# from src.models.rcnn_wrapper import FasterRCNNWrapper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='inference', choices=['inference', 'eval', 'train'], 
                        help='Execution mode: inference only, evaluation (COCO metrics), or training')
    parser.add_argument('--model', type=str, default='detr', choices=['yolo', 'detr', 'fasterrcnn'],
                        help='Model framework to use')
    parser.add_argument('--data_root', type=str, default='/home/mcv/datasets/C5/KITTI-MOTS/', 
                        help='Path to KITTI-MOTS dataset')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()

def get_model(args):
    if args.model == 'detr':
        return DetrWrapper(device=args.device)
    elif args.model == 'yolo':
        pass
        # return YoloWrapper(args)
    elif args.model == 'fasterrcnn':
        pass
        # return FasterRCNNWrapper(args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    args = get_args()
    print(f"--- Starting C5 Project Task 1 ---")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")

    # Dataset initialization
    # We use 'val' split for evaluation/inference as per instructions
    print("Loading Dataset...")
    dataset = KittiMotsDataset(root_dir=args.data_root, split='val')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(f"Dataset Loaded: {len(dataset)} images.")

    # Model initialization
    model = get_model(args)
    
    # Evaluator initialization
    evaluator = None
    if args.mode == 'eval':
        evaluator = CocoEvaluator(dataset)

    # Inference Loop
    print("Starting Inference Loop...")
    results_list = []
    
    # Iterate with a progress bar
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader)):
        
        # Run prediction
        # The wrapper returns: [{'boxes': [], 'scores': [], 'labels': []}, ...]
        batch_preds = model.predict(images)
        
        # Post-process results for Metrics
        # We need to inject the 'image_id' so the evaluator knows which image this is
        for i, pred in enumerate(batch_preds):
            target_id = targets[i]['image_id'].item() # Get ID from the dataset target
            pred['image_id'] = target_id
            
            # If in eval mode, add to evaluator
            if evaluator:
                evaluator.results.append(pred)
            
            # (Optional) Save to JSON/Disk if needed
            # results_list.append(pred)

    # 5. Final Evaluation
    if args.mode == 'eval' and evaluator:
        print("\n--- Running COCO Evaluation ---")
        evaluator.summarize()
        print("-------------------------------")

if __name__ == "__main__":
    main()