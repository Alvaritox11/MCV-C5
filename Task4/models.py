import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import ResNetModel
from transformers import VisionEncoderDecoderModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        """
        encoder_out: (batch, num_regions, encoder_dim)
        decoder_hidden: (batch, decoder_dim)
        """
        att1 = self.encoder_att(encoder_out)                  # (batch, num_regions, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1) # (batch, 1, attention_dim)
        scores = self.full_att(torch.tanh(att1 + att2)).squeeze(-1)  # (batch, num_regions)

        alpha = torch.softmax(scores, dim=1)                 # (batch, num_regions)
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)      # (batch, encoder_dim)

        return context, alpha


class BaselineModel(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.encoder_name = cfg.get("encoder", "ResNet-18")
        self.use_attention = cfg.get("use_attention", False)
        self.attention_dim = cfg.get("attention_dim", 256)

        # -------------------------
        # Encoder
        # -------------------------
        if self.encoder_name == "ResNet-50":
            self.encoder = ResNetModel.from_pretrained('microsoft/resnet-50').to(device)
            self.feat_dim = 2048

        elif self.encoder_name == "VGG-16":
            self.encoder = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features.to(device)
            self.vgg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = 512

        elif self.encoder_name == "VGG-19":
            self.encoder = tv_models.vgg19(weights=tv_models.VGG19_Weights.DEFAULT).features.to(device)
            self.vgg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feat_dim = 512

        else:  # Default: ResNet-18
            self.encoder = ResNetModel.from_pretrained('microsoft/resnet-18').to(device)
            self.feat_dim = 512

        if cfg.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

        # For ResNet-50 project 2048 -> 512
        self.encoder_proj = nn.Linear(self.feat_dim, 512) if self.feat_dim != 512 else nn.Identity()

        # -------------------------
        # Decoder config
        # -------------------------
        self.decoder_type = cfg.get("decoder", "GRU")
        self.decoder_hidden_dim = cfg.get("decoder_hidden_dim", 512)
        self.decoder_num_layers = cfg.get("decoder_num_layers", 1)
        self.decoder_dropout = cfg.get("decoder_dropout", 0.0)

        self.embed = nn.Embedding(self.tokenizer.vocab_size, 512)

        # -------------------------
        # Attention-specific modules
        # -------------------------
        if self.use_attention:
            self.attention = AdditiveAttention(
                encoder_dim=512,
                decoder_dim=self.decoder_hidden_dim,
                attention_dim=self.attention_dim
            )

            self.init_h = nn.Linear(512, self.decoder_hidden_dim)
            if self.decoder_type == "LSTM":
                self.init_c = nn.Linear(512, self.decoder_hidden_dim)

            rnn_input_dim = 512 + 512  # word embedding + attended image context
        else:
            rnn_input_dim = 512

        # -------------------------
        # RNN
        # -------------------------
        if self.decoder_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=self.decoder_hidden_dim,
                num_layers=self.decoder_num_layers,
                dropout=self.decoder_dropout if self.decoder_num_layers > 1 else 0.0
            )
        else:
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=self.decoder_hidden_dim,
                num_layers=self.decoder_num_layers,
                dropout=self.decoder_dropout if self.decoder_num_layers > 1 else 0.0
            )

        self.proj = nn.Linear(self.decoder_hidden_dim, self.tokenizer.vocab_size)

    # -------------------------------------------------
    # Helpers for attention mode
    # -------------------------------------------------
    def extract_spatial_features(self, img):
        """
        Returns spatial encoder features:
        (batch, num_regions, 512)
        """
        if "VGG" in self.encoder_name:
            feat_map = self.encoder(img)  # (batch, 512, H, W)
        else:
            outputs = self.encoder(img)
            feat_map = outputs.last_hidden_state  # (batch, C, H, W)

        batch_size, c, h, w = feat_map.shape
        feat_map = feat_map.view(batch_size, c, h * w).permute(0, 2, 1)  # (batch, num_regions, c)
        feat_map = self.encoder_proj(feat_map)  # (batch, num_regions, 512)

        return feat_map

    def init_hidden_state_attention(self, encoder_out):
        """
        encoder_out: (batch, num_regions, 512)
        """
        mean_feat = encoder_out.mean(dim=1)  # (batch, 512)

        h0 = self.init_h(mean_feat).unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)

        if self.decoder_type == "LSTM":
            c0 = self.init_c(mean_feat).unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
            return (h0, c0)
        else:
            return h0

    def get_decoder_hidden(self, hidden):
        """
        Returns top-layer hidden state with shape (batch, hidden_dim)
        """
        if self.decoder_type == "LSTM":
            return hidden[0][-1]
        return hidden[-1]

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, img, captions=None, return_attention=False):
        batch_size = img.shape[0]
        use_tf = self.cfg.get("teacher_forcing", False)

        # =========================================
        # CASE 1: NO ATTENTION -> preserve old logic
        # =========================================
        if not self.use_attention:
            if "VGG" in self.encoder_name:
                feat = self.encoder(img)  # (batch, 512, 7, 7)
                feat = self.vgg_pool(feat).squeeze(-1).squeeze(-1)  # (batch, 512)
            else:
                feat = self.encoder(img)
                feat = feat.pooler_output.squeeze(-1).squeeze(-1)   # (batch, feat_dim)

            feat = self.encoder_proj(feat)  # (batch, 512)

            if self.decoder_type == "LSTM":
                h0 = feat.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
                c0 = torch.zeros_like(h0)
                hidden = (h0, c0)
            else:
                hidden = feat.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)

            if use_tf and captions is not None:
                inputs = captions[:, :-1]
                embeds = self.embed(inputs).permute(1, 0, 2)  # (seq_len, batch, 512)
                out, _ = self.rnn(embeds, hidden)
                res = self.proj(out.permute(1, 0, 2))         # (batch, seq_len, vocab)
                res = res.permute(0, 2, 1)                    # (batch, vocab, seq_len)
                if return_attention:
                    return res, None
                return res

            else:
                start = torch.full(
                    (batch_size,),
                    self.tokenizer.sos_id,
                    dtype=torch.long,
                    device=img.device
                )

                inp = self.embed(start).unsqueeze(0)  # (1, batch, 512)
                outputs = []

                for _ in range(self.tokenizer.max_len - 1):
                    out, hidden = self.rnn(inp, hidden)      # out: (1, batch, hidden_dim)
                    logits = self.proj(out[-1])              # (batch, vocab_size)
                    outputs.append(logits.unsqueeze(1))      # (batch, 1, vocab_size)

                    next_ids = torch.argmax(logits, dim=-1)  # (batch,)
                    inp = self.embed(next_ids).unsqueeze(0)  # (1, batch, 512)

                res = torch.cat(outputs, dim=1)              # (batch, seq_len, vocab_size)
                res = res.permute(0, 2, 1)                   # (batch, vocab_size, seq_len)
                if return_attention:
                    return res, None
                return res

        # =========================================
        # CASE 2: ATTENTION
        # =========================================
        else:
            encoder_out = self.extract_spatial_features(img)   # (batch, num_regions, 512)
            hidden = self.init_hidden_state_attention(encoder_out)

            if use_tf and captions is not None:
                inputs = captions[:, :-1]
                seq_len = inputs.size(1)
                outputs = []

                for t in range(seq_len):
                    word_embed = self.embed(inputs[:, t])  # (batch, 512)

                    dec_hidden = self.get_decoder_hidden(hidden)  # (batch, hidden_dim)
                    context, alpha = self.attention(encoder_out, dec_hidden)  # (batch, 512)

                    rnn_input = torch.cat([word_embed, context], dim=-1).unsqueeze(0)  # (1, batch, 1024)

                    out, hidden = self.rnn(rnn_input, hidden)
                    logits = self.proj(out[-1])            # (batch, vocab_size)
                    outputs.append(logits.unsqueeze(1))    # (batch, 1, vocab_size)

                res = torch.cat(outputs, dim=1)            # (batch, seq_len, vocab_size)
                res = res.permute(0, 2, 1)                # (batch, vocab_size, seq_len)
                if return_attention:
                    return res, None
                return res

            else:
                current_tokens = torch.full(
                    (batch_size,),
                    self.tokenizer.sos_id,
                    dtype=torch.long,
                    device=img.device
                )

                outputs = []
                attn_weights_all = []

                for _ in range(self.tokenizer.max_len - 1):
                    word_embed = self.embed(current_tokens)  # (batch, 512)

                    dec_hidden = self.get_decoder_hidden(hidden)
                    context, alpha = self.attention(encoder_out, dec_hidden)

                    rnn_input = torch.cat([word_embed, context], dim=-1).unsqueeze(0)  # (1, batch, 1024)

                    out, hidden = self.rnn(rnn_input, hidden)
                    logits = self.proj(out[-1])              # (batch, vocab_size)
                    outputs.append(logits.unsqueeze(1))
                    attn_weights_all.append(alpha.unsqueeze(1))  # (batch, 1, num_regions)

                    current_tokens = torch.argmax(logits, dim=-1)

                res = torch.cat(outputs, dim=1)              # (batch, seq_len, vocab_size)
                res = res.permute(0, 2, 1)                   # (batch, vocab_size, seq_len)

                attn_weights_all = torch.cat(attn_weights_all, dim=1)  # (batch, seq_len, num_regions)

                if return_attention:
                    return res, attn_weights_all
                return res
            
class HFTransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_name = cfg.get("model_name", "nlpconnect/vit-gpt2-image-captioning")
        # Load config first to set the tying flag
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        config.tie_word_embeddings = False
        
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

        if cfg.get("freeze_encoder", False):
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        if cfg.get("freeze_decoder", False):
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None, decoder_attention_mask=None):
        return self.model(
            pixel_values=pixel_values, 
            labels=labels, 
            decoder_attention_mask=decoder_attention_mask
        )

    def generate(self, pixel_values, **kwargs):
        return self.model.generate(pixel_values, **kwargs)