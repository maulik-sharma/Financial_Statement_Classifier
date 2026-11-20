import torch
from torchvision.transforms import v2
from torchinfo import summary

from final import (
    CNNLSTMFusionModel,
    MultiModalDataset,
    SquarePadTransform,
)

# ---------------------------
# Config (same as final.py)
# ---------------------------
ANNOTATIONS_FILE = "dataset//labels.csv"
IMAGE_DIR = "dataset//images"
TEXT_DIR = "dataset//ocrtext_easy"

MAX_LENGTH = 500
MIN_WORD_FREQ = 50
EMBEDDING_DIM = 128
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
NUM_CLASSES = 3
DROPOUT = 0.5
BATCH_SIZE = 1         # your actual batch size

# ---------------------------
# Dataset just to get vocab_size
# ---------------------------
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    SquarePadTransform(target_size=1568, fill=0),  # -> H=W=1568
    v2.Grayscale()                                 # -> C=1
])

import json
with open("vocabulary.json") as file:
    vocab = json.load(file)

dataset = MultiModalDataset(
    annotations_file=ANNOTATIONS_FILE,
    image_dir=IMAGE_DIR,
    text_dir=TEXT_DIR,
    vocab=vocab,
    max_length=MAX_LENGTH,
    min_word_freq=MIN_WORD_FREQ,
    image_transform=train_transforms,
)

vocab_size = dataset.vocab_size
print(f"Vocabulary size: {vocab_size}")

# ---------------------------
# Build model
# ---------------------------
model = CNNLSTMFusionModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    lstm_hidden_size=LSTM_HIDDEN_SIZE,
    lstm_num_layers=LSTM_NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ---------------------------
# Input sizes (with batch dim)
# ---------------------------
image_input_size  = (BATCH_SIZE, 1, 1568, 1568)       # N, C, H, W
text_input_size   = (BATCH_SIZE, MAX_LENGTH)         # N, seq_len
fusion_input_size = (BATCH_SIZE, 512 + 128)          # 512 (CNN) + 128 (LSTM)

# ---------------------------
# 1) CNN backbone summary
# ---------------------------
# print("\n================ CNN BACKBONE SUMMARY ================")
# summary(
#     model.cnn_backbone,
#     input_size=image_input_size,
#     dtypes=[torch.float32],
#     col_names=("input_size", "output_size", "num_params","kernel_size",),
#     col_width=20,
#     depth=4,
# )
#
# ---------------------------
# 2) LSTM backbone summary
# ---------------------------
# print("\n================ LSTM BACKBONE SUMMARY ================")
# summary(
#     model.lstm_backbone,
#     input_size=text_input_size,
#     dtypes=[torch.long],
#     col_names=("input_size", "output_size", "num_params"),
#     col_width=20,
#     depth=4,
# )
#
#
# # (Optional) full model summary with both inputs
print("\n================ FULL MODEL SUMMARY (OPTIONAL) ================")
summary(
    model,
    input_size=[image_input_size, text_input_size],
    dtypes=[torch.float32, torch.long],
    col_names=("input_size", "output_size", "num_params"),
    col_width=20,
    depth=2,
)
