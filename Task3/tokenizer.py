import re
from collections import Counter


class BaseTokenizer:
    def __init__(self, max_len=201):
        self.max_len = max_len

    @property
    def vocab_size(self):
        return len(self.idx2token)

    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError


class CharTokenizer(BaseTokenizer):
    def __init__(self, max_len=201):
        super().__init__(max_len=max_len)

        chars = [
            '<PAD>', '<SOS>', '<EOS>', '<UNK>',
            ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            ':', ';', '=', '?',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]

        self.idx2token = chars
        self.token2idx = {tok: idx for idx, tok in enumerate(self.idx2token)}

        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

        self.pad_id = self.token2idx[self.pad_token]
        self.sos_id = self.token2idx[self.sos_token]
        self.eos_id = self.token2idx[self.eos_token]
        self.unk_id = self.token2idx[self.unk_token]

    def tokenize(self, text):
        return list(text)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token2idx.get(tok, 'self.unk_id') for tok in tokens]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            tok = self.idx2token[int(idx)]
            if tok == self.eos_token:
                break
            if tok not in [self.sos_token, self.pad_token]:
                tokens.append(tok)
        return "".join(tokens)


class WordTokenizer(BaseTokenizer):
    def __init__(self, captions, max_len=50, min_freq=1, lowercase=True):
        super().__init__(max_len=max_len)

        self.lowercase = lowercase
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

        special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token
        ]

        counter = Counter()
        for caption in captions:
            counter.update(self.tokenize(caption))

        vocab_tokens = [
            tok for tok, freq in counter.items()
            if freq >= min_freq and tok not in special_tokens
        ]

        self.idx2token = special_tokens + sorted(vocab_tokens)
        self.token2idx = {tok: idx for idx, tok in enumerate(self.idx2token)}

        self.pad_id = self.token2idx[self.pad_token]
        self.sos_id = self.token2idx[self.sos_token]
        self.eos_id = self.token2idx[self.eos_token]
        self.unk_id = self.token2idx[self.unk_token]

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()

        # words + punctuation as separate tokens
        return re.findall(r"\w+|[^\w\s]", text)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token2idx.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            tok = self.idx2token[int(idx)]
            if tok == self.eos_token:
                break
            if tok not in [self.sos_token, self.pad_token]:
                tokens.append(tok)

        text = " ".join(tokens)
        # remove spaces before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

from transformers import BertTokenizer

class SubwordTokenizer(BaseTokenizer):
    def __init__(self, max_len=50, model_name="bert-base-uncased"):
        super().__init__(max_len=max_len)

        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        self.pad_token = self.tokenizer.pad_token
        self.sos_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.unk_token = self.tokenizer.unk_token

        self.pad_id = self.tokenizer.pad_token_id
        self.sos_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.sep_token_id
        self.unk_id = self.tokenizer.unk_token_id

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids):
        clean_ids = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            if idx not in [self.sos_id, self.pad_id]:
                clean_ids.append(idx)
        return self.tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
    
def build_tokenizer(cfg, train_captions):
    text_level = cfg.get("text_level", "char").lower()

    if text_level == "char":
        return CharTokenizer(max_len=cfg.get("max_len", 201))

    elif text_level == "word":
        return WordTokenizer(
            captions=train_captions,
            max_len=cfg.get("max_len", 50),
            min_freq=cfg.get("min_word_freq", 1),
            lowercase=cfg.get("lowercase", True)
        )
    
    elif text_level == "subword":
        return SubwordTokenizer(
            max_len=cfg.get("max_len", 50)
        )

    else:
        raise ValueError(f"Unsupported text_level: {text_level}")
    

if __name__ == "__main__":
    import json
    from vizwiz_API.vizwiz_api.vizwiz import VizWiz

    def make_padded_sequence(tokenizer, text):
        ids = [tokenizer.sos_id] + tokenizer.encode(text) + [tokenizer.eos_id]
        if len(ids) < tokenizer.max_len:
            ids = ids + [tokenizer.pad_id] * (tokenizer.max_len - len(ids))
        else:
            ids = ids[:tokenizer.max_len - 1] + [tokenizer.eos_id]
        return ids

    with open("/ghome/group05/gerard/MCV-C5/Task3/configs/resnet50_gru.json", "r") as f:
        cfg = json.load(f)

    vw = VizWiz(cfg["train_ann_path"])
    img_ids = vw.getImgIds()[:1000]

    captions = []
    for img_id in img_ids:
        anns = vw.loadAnns(vw.getAnnIds(imgIds=img_id))
        captions.extend([ann["caption"] for ann in anns if ann["caption"].strip() != ""])

    tokenizer = build_tokenizer(cfg, captions)

    print(f"Text level: {cfg.get('text_level', 'char')}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max len: {tokenizer.max_len}")
    print("-" * 60)

    for cap in captions[:5]:
        tokens = tokenizer.tokenize(cap)
        encoded = tokenizer.encode(cap)
        padded_seq = make_padded_sequence(tokenizer, cap)
        decoded = tokenizer.decode(padded_seq)

        unk_count = sum(1 for i in encoded if i == tokenizer.unk_id)

        print(f"Original: {cap}")
        print(f"Tokens: {tokens}")
        print(f"Encoded: {encoded[:20]}{' ...' if len(encoded) > 20 else ''}")
        print(f"UNKs: {unk_count}/{len(encoded)}")
        print(f"Padded seq len: {len(padded_seq)}")
        print(f"Decoded: {decoded}")
        print("-" * 60)