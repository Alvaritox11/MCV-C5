import torch
import torch.nn as nn
import torchvision.models as tv_models
from transformers import ResNetModel

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
    def forward(self, img, captions=None, return_attention=False, return_loss_logits=False):
        batch_size = img.shape[0]
        use_tf = self.cfg.get("teacher_forcing", False)

        # =========================================
        # CASE 1: NO ATTENTION -> preserve old logic
        # =========================================
        if not self.use_attention:
            # ---- ENCODER ----
            if "VGG" in self.encoder_name:
                feat = self.encoder(img)
                feat = self.vgg_pool(feat).squeeze(-1).squeeze(-1)
            else:
                feat = self.encoder(img)
                feat = feat.pooler_output.squeeze(-1).squeeze(-1)

            feat = self.encoder_proj(feat)

            if self.decoder_type == "LSTM":
                h0 = feat.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
                c0 = torch.zeros_like(h0)
                hidden_tf = (h0, c0)
                hidden_gen = (h0.clone(), c0.clone())
            else:
                hidden_tf = feat.unsqueeze(0).repeat(self.decoder_num_layers, 1, 1)
                hidden_gen = hidden_tf.clone()

            outputs = {}

            # ---- TEACHER FORCING (LOSS) ----
            if return_loss_logits and captions is not None:
                inputs = captions[:, :-1]
                embeds = self.embed(inputs).permute(1, 0, 2)
                out, _ = self.rnn(embeds, hidden_tf)
                res_tf = self.proj(out.permute(1, 0, 2)).permute(0, 2, 1)
                outputs["logits_tf"] = res_tf

            # ---- GENERATION ----
            if not return_loss_logits or return_attention:
                start = torch.full((batch_size,), self.tokenizer.sos_id, dtype=torch.long, device=img.device)
                inp = self.embed(start).unsqueeze(0)
                outputs_gen = []

                hidden = hidden_gen

                for _ in range(self.tokenizer.max_len - 1):
                    out, hidden = self.rnn(inp, hidden)
                    logits = self.proj(out[-1])
                    outputs_gen.append(logits.unsqueeze(1))
                    next_ids = torch.argmax(logits, dim=-1)
                    inp = self.embed(next_ids).unsqueeze(0)

                res_gen = torch.cat(outputs_gen, dim=1).permute(0, 2, 1)
                outputs["logits_gen"] = res_gen
                outputs["attn"] = None

            return outputs

        # =========================================
        # CASE 2: ATTENTION
        # =========================================
        else:
            encoder_out = self.extract_spatial_features(img)

            hidden_tf = self.init_hidden_state_attention(encoder_out)
            hidden_gen = self.init_hidden_state_attention(encoder_out)

            outputs = {}

            # ---- TEACHER FORCING ----
            if return_loss_logits and captions is not None:
                inputs = captions[:, :-1]
                seq_len = inputs.size(1)
                outputs_tf = []

                hidden = hidden_tf

                for t in range(seq_len):
                    word_embed = self.embed(inputs[:, t])
                    dec_hidden = self.get_decoder_hidden(hidden)
                    context, _ = self.attention(encoder_out, dec_hidden)

                    rnn_input = torch.cat([word_embed, context], dim=-1).unsqueeze(0)
                    out, hidden = self.rnn(rnn_input, hidden)

                    logits = self.proj(out[-1])
                    outputs_tf.append(logits.unsqueeze(1))

                res_tf = torch.cat(outputs_tf, dim=1).permute(0, 2, 1)
                outputs["logits_tf"] = res_tf

            # ---- GENERATION ----
            if not return_loss_logits or return_attention:
                current_tokens = torch.full((batch_size,), self.tokenizer.sos_id, dtype=torch.long, device=img.device)

                outputs_gen = []
                attn_weights_all = []

                hidden = hidden_gen

                for _ in range(self.tokenizer.max_len - 1):
                    word_embed = self.embed(current_tokens)

                    dec_hidden = self.get_decoder_hidden(hidden)
                    context, alpha = self.attention(encoder_out, dec_hidden)

                    rnn_input = torch.cat([word_embed, context], dim=-1).unsqueeze(0)
                    out, hidden = self.rnn(rnn_input, hidden)

                    logits = self.proj(out[-1])
                    outputs_gen.append(logits.unsqueeze(1))
                    attn_weights_all.append(alpha.unsqueeze(1))

                    current_tokens = torch.argmax(logits, dim=-1)

                res_gen = torch.cat(outputs_gen, dim=1).permute(0, 2, 1)
                attn_weights_all = torch.cat(attn_weights_all, dim=1)

                outputs["logits_gen"] = res_gen
                outputs["attn"] = attn_weights_all

            return outputs