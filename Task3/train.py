# train.py

import os  
import wandb 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import evaluate

import config
from dataset import get_dataloaders
from models import BaselineModel

# Load the evaluation metrics
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_one_epoch(model, optimizer, crit, dataloader):
    model.train()
    total_loss = 0
    
    # tqdm gives us a nice progress bar
    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, captions, _ in loop: # _ ignores the raw text strings from our dataset
        imgs, captions = imgs.to(device), captions.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        preds = model(imgs)
        
        # Calculate loss (CrossEntropy expects raw logits and class indices)
        loss = crit(preds, captions)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def evaluate_and_log(model, crit, dataloader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for imgs, captions, raw_captions in loop:
            imgs, captions = imgs.to(device), captions.to(device)
            
            # Forward pass
            output = model(imgs) # Shape: (batch, NUM_CHAR, seq_len)
            
            # 1. Calculate Loss
            loss = crit(output, captions)
            total_loss += loss.item()
            
            # 2. Decode Predictions for Metrics
            pred_ids = torch.argmax(output, dim=1)
            
            for i in range(pred_ids.shape[0]):
                pred_str = ""
                for char_id in pred_ids[i]:
                    char = config.IDX2CHAR[char_id.item()]
                    if char == '<EOS>':
                        break
                    if char not in ['<SOS>', '<PAD>']:
                        pred_str += char
                
                all_preds.append(pred_str)
                all_refs.append([raw_captions[i]]) 
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics using Hugging Face evaluate 
    bleu1_score = bleu.compute(predictions=all_preds, references=all_refs, max_order=1)['bleu']
    bleu2_score = bleu.compute(predictions=all_preds, references=all_refs, max_order=2)['bleu']
    rouge_score = rouge.compute(predictions=all_preds, references=all_refs)['rougeL']
    meteor_score = meteor.compute(predictions=all_preds, references=all_refs)['meteor']
    
    metrics = {
        "Valid Loss": avg_loss,
        "BLEU-1": bleu1_score * 100,
        "BLEU-2": bleu2_score * 100,
        "ROUGE-L": rouge_score * 100,
        "METEOR": meteor_score * 100
    }
    
    return metrics

def main():
    # Create a directory to store checkpoints if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    wandb.init(
        project="ImageCaptioning",
        entity="Team5-C5",
        name="baseline-resnet18-gru-char",
        config={
            "encoder": "ResNet-18",
            "decoder": "GRU",
            "level": "Character",
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": 0.0001
        }
    )
    
    print("Initializing DataLoaders...")
    train_loader, valid_loader, test_loader = get_dataloaders()
    
    print("Initializing Baseline Model...")
    model = BaselineModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    crit = nn.CrossEntropyLoss(ignore_index=config.CHAR2IDX['<PAD>'])
    
    # Initialize a variable to keep track of the best validation loss
    best_valid_loss = float('inf')
    
    print("Starting Training Loop...")
    for epoch in range(wandb.config.epochs):
        
        # 1. Train for one epoch
        train_loss = train_one_epoch(model, optimizer, crit, train_loader)
        
        # 2. Evaluate and get all metrics in one efficient pass
        metrics = evaluate_and_log(model, crit, valid_loader)
        valid_loss = metrics["Valid Loss"]
        
        print(f"Epoch {epoch+1}/{wandb.config.epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"Metrics: BLEU-1: {metrics['BLEU-1']:.1f}% | ROUGE-L: {metrics['ROUGE-L']:.1f}%")
        
        # 3. Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics # This unpacks the entire metrics dictionary into W&B
        })
        
        # 4. Save the Checkpoint if it's the best one so far!
        if valid_loss < best_valid_loss:
            print(f"🌟 Validation loss improved from {best_valid_loss:.4f} to {valid_loss:.4f}. Saving checkpoint!")
            best_valid_loss = valid_loss
            
            checkpoint_path = f"checkpoints/best_baseline_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': valid_loss,
            }, checkpoint_path)
            
            # Optional: Save it directly to W&B cloud as well
            wandb.save(checkpoint_path)

    wandb.finish()
    print("Training Complete!")

if __name__ == "__main__":
    main()