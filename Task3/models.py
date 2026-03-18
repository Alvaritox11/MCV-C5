# models.py

import torch
import torch.nn as nn
from transformers import ResNetModel
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Baseline Encoder
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(device)
        
        # Baseline Decoder (GRU)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, config.NUM_CHAR)
        self.embed = nn.Embedding(config.NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        
        # Extract features with ResNet
        feat = self.resnet(img)
        # Pool and reshape to match GRU expected hidden state format: (1, batch, 512)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0) 
        
        # Prepare the initial <SOS> token
        start = torch.tensor(config.CHAR2IDX['<SOS>']).to(device)
        start_embed = self.embed(start) # Shape: 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0) # Shape: 1, batch, 512
        
        inp = start_embeds
        hidden = feat
        
        # Autoregressive Generation Loop
        for t in range(config.TEXT_MAX_LEN - 1): # -1 to account for <SOS>
            out, hidden = self.gru(inp, hidden)
            # Feed the last output back in as the next input 
            # Note: This is generation WITHOUT Teacher Forcing
            inp = torch.cat((inp, out[-1:]), dim=0) # Concatenate along sequence length
    
        # Final Projections
        res = inp.permute(1, 0, 2) # batch, seq, 512
        res = self.proj(res)       # batch, seq, NUM_CHAR
        res = res.permute(0, 2, 1) # batch, NUM_CHAR, seq (Expected by CrossEntropyLoss)
        
        return res