# models.py

import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import ResNetModel
# import vocab

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BaselineModel(nn.Module):
    # def __init__(self, cfg):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.encoder_name = cfg.get("encoder", "ResNet-18")

        if self.encoder_name == "ResNet-50":
            # self.resnet = ResNetModel.from_pretrained('microsoft/resnet-50').to(device)
            self.encoder = ResNetModel.from_pretrained('microsoft/resnet-50').to(device)
            self.feat_dim = 2048

        elif self.encoder_name == "VGG-16":
            # Load VGG-16 features from standard torchvision
            self.encoder = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features.to(device)
            self.vgg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = 512

        else:  # Default Baseline ResNet-18
            # self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(device)
            self.encoder = ResNetModel.from_pretrained('microsoft/resnet-18').to(device)
            self.feat_dim = 512
            
        if cfg.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Project ResNet-50's 2048 dimensions down to 512 so the decoder doesn't break
        self.encoder_proj = nn.Linear(self.feat_dim, 512) if self.feat_dim != 512 else nn.Identity()

        # 2. Configurable Decoder
        self.decoder_type = cfg.get("decoder", "GRU")
        self.decoder_hidden_dim = cfg.get("decoder_hidden_dim", 512)
        self.decoder_num_layers = cfg.get("decoder_num_layers", 1)
        self.decoder_dropout = cfg.get("decoder_dropout", 0.0)

        if self.decoder_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=512,
                hidden_size=self.decoder_hidden_dim,
                num_layers=self.decoder_num_layers,
                dropout=self.decoder_dropout if self.decoder_num_layers > 1 else 0.0
            )
        else:
            self.rnn = nn.GRU(
                input_size=512,
                hidden_size=self.decoder_hidden_dim,
                num_layers=self.decoder_num_layers,
                dropout=self.decoder_dropout if self.decoder_num_layers > 1 else 0.0
            )

        self.proj = nn.Linear(self.decoder_hidden_dim, self.tokenizer.vocab_size)
        self.embed = nn.Embedding(self.tokenizer.vocab_size, 512)

    def forward(self, img, captions=None):
        batch_size = img.shape[0]

        # --- Feature Extraction depending on Encoder Type ---
        if self.encoder_name == "VGG-16":
            feat = self.encoder(img)  # Shape: (batch, 512, 7, 7)
            feat = self.vgg_pool(feat).squeeze(-1).squeeze(-1)  # Flatten to (batch, 512)
        else:
            feat = self.encoder(img)
            feat = feat.pooler_output.squeeze(-1).squeeze(-1)

        feat = self.encoder_proj(feat)  # (batch, 512)

        if self.decoder_type == "LSTM":
            h0 = feat.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)
        else:
            hidden = feat.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)

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
            # --- AUTOREGRESSIVE MODE ---
            # start = torch.tensor(vocab.CHAR2IDX['<SOS>']).to(device)
            start = torch.full((batch_size,), self.tokenizer.sos_id, dtype=torch.long, device=device)

            # start_embed = self.embed(start)
            # start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)
            inp = self.embed(start).unsqueeze(0)  # (1, batch, 512)

            outputs = []

            # for t in range(vocab.TEXT_MAX_LEN - 1):
            for t in range(self.tokenizer.max_len - 1):
                out, hidden = self.rnn(inp, hidden)            # out: (1, batch, 512)

                # outputs.append(out[-1:])
                # inp = out[-1:]  # Next input is model's own output

                logits = self.proj(out[-1])                    # (batch, vocab_size)
                outputs.append(logits.unsqueeze(1))            # (batch, 1, vocab_size)

                next_ids = torch.argmax(logits, dim=-1)        # (batch,)
                inp = self.embed(next_ids).unsqueeze(0)        # (1, batch, 512)

            # res = torch.cat(outputs, dim=0).permute(1, 0, 2)
            # res = self.proj(res)
            # return res.permute(0, 2, 1)

            res = torch.cat(outputs, dim=1)                    # (batch, seq_len, vocab_size)
            return res.permute(0, 2, 1)                        # (batch, vocab_size, seq_len)