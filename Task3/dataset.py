# dataset.py

import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

# VizWiz API 
from vizwiz_API.vizwiz_api.vizwiz import VizWiz
import config


class VizWizDataset(Dataset):
    def __init__(self, vw_api, img_ids, img_dir):
        """
        Args:
            vw_api: The initialized VizWiz API object (vw_train_full or vw_test)
            img_ids: List of image IDs for this specific split
            img_dir: Path to the folder containing the actual .jpg images
        """
        self.vw = vw_api
        self.img_dir = img_dir
        self.max_len = config.TEXT_MAX_LEN

        # --- Pre-filter missing captions ---
        self.img_ids = []
        for img_id in img_ids:
            # Check if the dictionary has captions for this ID
            if len(self.vw.imgToAnns.get(img_id, [])) == 0:
                print(f"⚠️ Skipping Image ID {img_id}: No captions found.")
            else:
                self.img_ids.append(img_id)
        
        # Kept the exact image processing from the baseline notebook
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
        
        # --- Image Processing ---
        img_info = self.vw.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = f"{self.img_dir}/{img_name}"
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.img_proc(img)
        
        # --- Caption Processing (Baseline Logic) ---
        # Bypass the API method and access the dictionary directly (faster and safer)
        anns = self.vw.imgToAnns[img_id]
        
        # Quick safety check just to be absolutely sure
        if len(anns) == 0:
            raise ValueError(f"Image ID {img_id} has no captions.")
            
        # Pick a random caption out of the 5 available
        raw_caption = random.choice(anns)['caption']
        
        # Filter out unknown characters to prevent KeyErrors
        filtered_caption = [c for c in raw_caption if c in config.CHAR2IDX]
        
        # Build the final list: <SOS> + chars + <EOS> + <PAD>
        final_list = ['<SOS>'] + filtered_caption + ['<EOS>']
        gap = self.max_len - len(final_list)
        
        if gap > 0:
            final_list.extend(['<PAD>'] * gap)
        else:
            final_list = final_list[:self.max_len-1] + ['<EOS>'] # Truncate if too long
            
        cap_idx = torch.tensor([config.CHAR2IDX[char] for char in final_list])
        
        return img_tensor, cap_idx, raw_caption


def get_dataloaders():
    """Initializes the API, splits the data, and returns DataLoaders."""
    # Initialize API
    print("Loading annotations...")
    vw_train_full = VizWiz(config.TRAIN_ANN_PATH)
    vw_test = VizWiz(config.VAL_ANN_PATH)
    
    # Split the training data 90/10
    all_train_img_ids = vw_train_full.getImgIds()
    random.seed(42)
    random.shuffle(all_train_img_ids)
    split_idx = int(len(all_train_img_ids) * 0.9)
    
    train_ids = all_train_img_ids[:split_idx]
    valid_ids = all_train_img_ids[split_idx:]
    test_ids = vw_test.getImgIds()

    print(f"Train size: {len(train_ids)} | Valid size: {len(valid_ids)} | Test size: {len(test_ids)}")
    
    # Create datasets
    train_dataset = VizWizDataset(vw_train_full, train_ids, config.TRAIN_IMG_DIR)
    valid_dataset = VizWizDataset(vw_train_full, valid_ids, config.TRAIN_IMG_DIR)
    test_dataset = VizWizDataset(vw_test, test_ids, config.VAL_IMG_DIR)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, valid_loader, test_loader