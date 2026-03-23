import os
import json
import argparse
import torch
from tqdm import tqdm

# Import your existing pipeline functions
from dataset import get_dataloaders
from models import BaselineModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_predictions(model, dataloader, tokenizer, output_file, epoch_label="epoch_last"):
    """
    Runs inference on the validation set and saves a predictions.json file
    formatted perfectly for evaluate_offline.py.
    """
    model.eval()
    run_predictions = {}

    print("Generating predictions on validation set...")
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for img_ids, imgs, _, all_raw_captions in loop:
            imgs = imgs.to(device)
            
            # Autoregressive generation
            gen_output, _ = model(imgs, return_attention=True)
            pred_ids = torch.argmax(gen_output, dim=1)

            # Reconstruct ground truth lists
            refs_for_batch = []
            for b in range(imgs.size(0)):
                refs_for_batch.append([all_raw_captions[j][b] for j in range(len(all_raw_captions))])

            # Process predictions
            for i in range(pred_ids.shape[0]):
                pred_str = tokenizer.decode(pred_ids[i].tolist())
                current_id = str(img_ids[i].item() if torch.is_tensor(img_ids[i]) else img_ids[i])

                if current_id not in run_predictions:
                    run_predictions[current_id] = {
                        "references": refs_for_batch[i],
                        "predictions": {}
                    }
                
                # Save under the specified epoch label
                run_predictions[current_id]["predictions"][epoch_label] = pred_str

    # Save to disk
    with open(output_file, "w") as f:
        json.dump(run_predictions, f, indent=4)
        
    print(f"✅ Successfully saved predictions to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions.json from a saved .pth model.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config.json file used for this model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the best_model.pth file')
    parser.add_argument('--output_dir', type=str, default='outputs/recovered_preds', help='Where to save the new predictions.json')
    parser.add_argument('--epoch_label', type=str, default='epoch_last', help='The key used in the JSON (e.g., epoch_10)')
    args = parser.parse_args()

    # 1. Load configuration
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = cfg.get("run_name", "experiment")
    name = f"predictions_{base_name}.json"
    output_file = os.path.join(args.output_dir, name)
    
    # 2. Initialize DataLoaders & Tokenizer
    print("Initializing DataLoaders...")
    _, valid_loader, _, tokenizer = get_dataloaders(cfg)

    # 3. Initialize Model
    print("Initializing Model...")
    model = BaselineModel(cfg, tokenizer).to(device)

    # 4. Load weights from the checkpoint
    print(f"Loading weights from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()

    for name, param in state_dict.items():
        if name in model_state and param.shape != model_state[name].shape:
            print(f"⚠️ Auto-resizing {name} from {param.shape} to {model_state[name].shape}")
            new_param = model_state[name].clone()
            
            # Get the overlapping bounds between the old and new weights
            slices = tuple(slice(0, min(d1, d2)) for d1, d2 in zip(param.shape, new_param.shape))
            new_param[slices] = param[slices]
            state_dict[name] = new_param

    model.load_state_dict(state_dict, strict=False)

    # 5. Run generation
    generate_predictions(model, valid_loader, tokenizer, output_file, args.epoch_label)

if __name__ == "__main__":
    main()