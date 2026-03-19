# dataset.py

import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from vizwiz_API.vizwiz_api.vizwiz import VizWiz
import vocab

class VizWizDataset(Dataset):
    def __init__(self, vw_api, img_ids, img_dir):
        self.vw = vw_api
        self.img_dir = img_dir
        self.max_len = vocab.TEXT_MAX_LEN

        self.img_ids = []
        for img_id in img_ids:
            if len(self.vw.imgToAnns.get(img_id, [])) == 0:
                print(f"⚠️ Skipping Image ID {img_id}: No captions found.")
            else:
                self.img_ids.append(img_id)
        
        self.img_proc = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img_info = self.vw.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = f"{self.img_dir}/{img_name}"
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.img_proc(img)
        
        anns = self.vw.imgToAnns[img_id]
        
        all_raw_captions = [ann['caption'] for ann in anns]
        
        if len(all_raw_captions) < 5:
            # Pad with empty strings if there are less than 5
            all_raw_captions += [""] * (5 - len(all_raw_captions))
        else:
            # Truncate if there are more than 5
            all_raw_captions = all_raw_captions[:5]
            
        raw_caption = random.choice([c for c in all_raw_captions if c != ""])
        
        filtered_caption = [c for c in raw_caption if c in vocab.CHAR2IDX]
        final_list = ['<SOS>'] + filtered_caption + ['<EOS>']
        gap = self.max_len - len(final_list)
        
        if gap > 0:
            final_list.extend(['<PAD>'] * gap)
        else:
            final_list = final_list[:self.max_len-1] + ['<EOS>']
            
        cap_idx = torch.tensor([vocab.CHAR2IDX[char] for char in final_list])
        
        return img_tensor, cap_idx, all_raw_captions

def get_dataloaders(cfg):
    print("Loading annotations...")
    vw_train_full = VizWiz(cfg["train_ann_path"])
    vw_test = VizWiz(cfg["val_ann_path"])
    
    all_train_img_ids = vw_train_full.getImgIds()
    random.seed(42)
    random.shuffle(all_train_img_ids)
    split_idx = int(len(all_train_img_ids) * 0.9)
    
    train_ids = all_train_img_ids[:split_idx]
    valid_ids = all_train_img_ids[split_idx:]
    test_ids = vw_test.getImgIds()

    print(f"Train size: {len(train_ids)} | Valid size: {len(valid_ids)} | Test size: {len(test_ids)}")
    
    train_dataset = VizWizDataset(vw_train_full, train_ids, cfg["train_img_dir"])
    valid_dataset = VizWizDataset(vw_train_full, valid_ids, cfg["train_img_dir"])
    test_dataset = VizWizDataset(vw_test, test_ids, cfg["val_img_dir"])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)
    
    return train_loader, valid_loader, test_loader