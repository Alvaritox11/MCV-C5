# config.py

# Paths
TRAIN_IMG_DIR = '/ghome/group05/datasets/VizWiz/train'
VAL_IMG_DIR = '/ghome/group05/datasets/VizWiz/val'
TRAIN_ANN_PATH = '/ghome/group05/datasets/VizWiz/annotations/train.json'
VAL_ANN_PATH = '/ghome/group05/datasets/VizWiz/annotations/val.json'

# Baseline Vocabulary
CHARS = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(CHARS)
IDX2CHAR = {k: v for k, v in enumerate(CHARS)}
CHAR2IDX = {v: k for k, v in enumerate(CHARS)}
TEXT_MAX_LEN = 201

BATCH_SIZE = 32
EPOCHS = 10
