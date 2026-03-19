# models.py

import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import ResNetModel
import vocab

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaselineModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_name = cfg.get("encoder", "ResNet-18")
        
        if self.encoder_name == "ResNet-50":
            self.resnet = ResNetModel.from_pretrained('microsoft/resnet-50').to(device)
            self.feat_dim = 2048

        elif self.encoder_name == "VGG-16":
            # Load VGG-16 features from standard torchvision
            self.encoder = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features.to(device)
            self.vgg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = 512

        else: # Default Baseline ResNet-18
            self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(device)
            self.feat_dim = 512
            
        # Project ResNet-50's 2048 dimensions down to 512 so the decoder doesn't break
        self.encoder_proj = nn.Linear(self.feat_dim, 512) if self.feat_dim != 512 else nn.Identity()
        
        # 2. Configurable Decoder
        self.decoder_type = cfg.get("decoder", "GRU")
        if self.decoder_type == "LSTM":
            self.rnn = nn.LSTM(512, 512, num_layers=1)
        else: # Default Baseline GRU
            self.rnn = nn.GRU(512, 512, num_layers=1)
            
        self.proj = nn.Linear(512, vocab.NUM_CHAR)
        self.embed = nn.Embedding(vocab.NUM_CHAR, 512)

    def forward(self, img, captions=None):
        batch_size = img.shape[0]
        

        # --- Feature Extraction depending on Encoder Type ---
        if self.encoder_name == "VGG-16":
            feat = self.encoder(img) # Shape: (batch, 512, 7, 7)
            feat = self.vgg_pool(feat).squeeze(-1).squeeze(-1) # Flatten to (batch, 512)
        else:
            feat = self.encoder(img)
            feat = feat.pooler_output.squeeze(-1).squeeze(-1) 

        # Extract and adjust image features
        feat = self.encoder_proj(feat).unsqueeze(0) # Shape: (1, batch, 512)
        
        # Handle LSTM vs GRU hidden states
        if self.decoder_type == "LSTM":
            hidden = (feat, torch.zeros_like(feat))
        else:
            hidden = feat
            
        use_tf = self.cfg.get("teacher_forcing", False)
        
        if use_tf and captions is not None:
            # --- TEACHER FORCING MODE ---
            # Embed the full ground truth (except the last token) and feed it all at once
            inputs = captions[:, :-1]
            embeds = self.embed(inputs).permute(1, 0, 2)
            out, _ = self.rnn(embeds, hidden)
            res = self.proj(out.permute(1, 0, 2))
            return res.permute(0, 2, 1)
            
        else:
            # --- BASELINE AUTOREGRESSIVE MODE ---
            start = torch.tensor(vocab.CHAR2IDX['<SOS>']).to(device)
            start_embed = self.embed(start)
            start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)
            
            inp = start_embeds
            outputs = []
            
            for t in range(vocab.TEXT_MAX_LEN - 1):
                out, hidden = self.rnn(inp, hidden)
                outputs.append(out[-1:])
                inp = out[-1:] # Next input is model's own output
        
            res = torch.cat(outputs, dim=0).permute(1, 0, 2)
            res = self.proj(res) 
            return res.permute(0, 2, 1)