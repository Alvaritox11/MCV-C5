# dataset.py

import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from vizwiz_API.vizwiz_api.vizwiz import VizWiz
from transformers import AutoProcessor
from transformers import ViTImageProcessor, AutoTokenizer

# import vocab
from tokenizer import build_tokenizer


class VizWizDataset(Dataset):
    # def __init__(self, vw_api, img_ids, img_dir):
    def __init__(self, vw_api, img_ids, img_dir, image_processor, tokenizer):
        self.vw = vw_api
        self.img_dir = img_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length


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
        pixel_values = self.image_processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

        anns = self.vw.imgToAnns[img_id]
        all_raw_captions = [ann['caption'] for ann in anns if ann['caption'].strip() != ""]

        if len(all_raw_captions) < 5:
            all_raw_captions += [""] * (5 - len(all_raw_captions))
        else:
            all_raw_captions = all_raw_captions[:5]

        raw_caption = random.choice([c for c in all_raw_captions if c != ""])
        
        labels = self.tokenizer(
            text=raw_caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        labels[labels == self.tokenizer.pad_token_id] = -100 
        print(self.tokenizer.decode(labels[labels != -100]))
        return img_id, pixel_values, labels, all_raw_captions


def extract_captions_from_img_ids(vw_api, img_ids):
    captions = []
    for img_id in img_ids:
        anns = vw_api.imgToAnns.get(img_id, [])
        for ann in anns:
            caption = ann['caption'].strip()
            if caption != "":
                captions.append(caption)
    return captions

def get_qwen_dataloaders(cfg, tokenizer):
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

    # Build tokenizer ONLY from training captions
    # train_captions = extract_captions_from_img_ids(vw_train_full, train_ids)
    # tokenizer = build_tokenizer(cfg, train_captions)


    # processor = AutoProcessor.from_pretrained(cfg.get("model_name", "nlpconnect/vit-gpt2-image-captioning"))
    # processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    image_processor = ViTImageProcessor.from_pretrained(cfg.get("model_name", "nlpconnect/vit-gpt2-image-captioning"))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = cfg.get("max_length", 128)
    print(f"Tokenizer built | text_level={cfg.get('text_level', 'char')} | vocab_size={tokenizer.vocab_size} | max_len={tokenizer.model_max_length}")

    train_dataset = VizWizDataset(vw_train_full, train_ids, cfg["train_img_dir"], image_processor, tokenizer)
    valid_dataset = VizWizDataset(vw_train_full, valid_ids, cfg["train_img_dir"], image_processor, tokenizer)
    test_dataset = VizWizDataset(vw_test, test_ids, cfg["val_img_dir"], image_processor, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4), pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=True)

    return train_loader, valid_loader, test_loader