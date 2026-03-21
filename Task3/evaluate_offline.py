import json
import argparse
import evaluate
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Compute per-image metrics offline.")
    parser.add_argument('--preds_file', type=str, required=True, help='Path to your predictions.json file')
    parser.add_argument('--epoch', type=str, default='epoch_10', help='Which epoch to evaluate (e.g., epoch_10)')
    parser.add_argument('--output_file', type=str, default='detailed_metrics.json', help='Where to save the per-image results')
    args = parser.parse_args()

    print(f"Loading predictions from {args.preds_file}...")
    with open(args.preds_file, 'r') as f:
        data = json.load(f)

    print("Loading HF evaluate modules...")
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')

    results = []

    print(f"Calculating metrics for {len(data)} images at {args.epoch}...")
    for img_id, item in tqdm(data.items(), desc="Processing Images"):
        refs = item.get("references", [])
        preds_dict = item.get("predictions", {})
        
        # Skip if the epoch doesn't exist for this image
        if args.epoch not in preds_dict:
            continue
            
        pred = preds_dict[args.epoch]
        
        # Hugging Face evaluate expects lists
        pred_list = [pred]
        ref_list = [refs]

        try:
            # Compute individual sentence-level metrics
            b1 = bleu.compute(predictions=pred_list, references=ref_list, max_order=1)['bleu']
            b4 = bleu.compute(predictions=pred_list, references=ref_list, max_order=2)['bleu']
            rl = rouge.compute(predictions=pred_list, references=ref_list)['rougeL']
            met = meteor.compute(predictions=pred_list, references=ref_list)['meteor']
        except ZeroDivisionError:
            # Handle edge cases where the model predicts an empty string
            b1, b4, rl, met = 0.0, 0.0, 0.0, 0.0

        # Create a combined score to rank the images
        avg_score = ((b4 * 100) + (rl * 100) + (met * 100)) / 3.0

        results.append({
            "image_id": img_id,
            "prediction": pred,
            "references": refs,
            "BLEU-1": round(b1 * 100, 2),
            "BLEU-4": round(b4 * 100, 2),
            "ROUGE-L": round(rl * 100, 2),
            "METEOR": round(met * 100, 2),
            "Avg_Score": round(avg_score, 2)
        })

    # Sort the results from highest score to lowest score
    results.sort(key=lambda x: x['Avg_Score'], reverse=True)

    # Save to disk
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Saved detailed metrics to {args.output_file}")
    
    # --- Print Quick Summary ---
    if len(results) > 0:
        best = results[0]
        median = results[len(results) // 2]
        worst = results[-1]
        
        print("\n" + "="*50)
        print("🏆 BEST IMAGE")
        print("="*50)
        print(f"ID: {best['image_id']} | Avg Score: {best['Avg_Score']}")
        print(f"PRED: {best['prediction']}")
        print(f"REF 1: {best['references'][0]}")
        
        print("\n" + "="*50)
        print("⚖️  MEDIAN IMAGE")
        print("="*50)
        print(f"ID: {median['image_id']} | Avg Score: {median['Avg_Score']}")
        print(f"PRED: {median['prediction']}")
        print(f"REF 1: {median['references'][0]}")
        
        print("\n" + "="*50)
        print("📉 WORST IMAGE")
        print("="*50)
        print(f"ID: {worst['image_id']} | Avg Score: {worst['Avg_Score']}")
        print(f"PRED: {worst['prediction']}")
        print(f"REF 1: {worst['references'][0]}")
        print("="*50)

if __name__ == "__main__":
    main()