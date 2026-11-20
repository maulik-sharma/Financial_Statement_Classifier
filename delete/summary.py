import torch
from torchinfo import summary
import json

# Import your model classes (assuming they're in final.py)
from final import CNNFeatureExtractor, LSTMFeatureExtractor, CNNLSTMFusionModel

# Configuration from your file
MAX_LENGTH = 500
EMBEDDING_DIM = 128
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
NUM_CLASSES = 3
DROPOUT = 0.5

# Load vocabulary to get vocab_size
with open("vocabulary.json", 'r') as f:
    vocab = json.load(f)
    vocab_size = len(vocab)

print(f"Vocabulary size: {vocab_size}")
print("="*80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 1. CNN Feature Extractor Summary
# ============================================================================
print("\n" + "="*80)
print("CNN FEATURE EXTRACTOR SUMMARY")
print("="*80)

cnn_model = CNNFeatureExtractor().to(device)

# Input: (batch_size, channels, height, width)
# From your code: images are grayscale (1 channel) and padded to 1568x1568
cnn_input_size = (1, 1, 1568, 1568)  # (batch, channels, height, width)

cnn_summary = summary(
    cnn_model,
    input_size=cnn_input_size,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
    device=device,
    verbose=1
)

print(f"\nCNN Input Shape: {cnn_input_size}")
print(f"CNN Output Shape: {cnn_summary.summary_list[-1].output_size}")

# ============================================================================
# 2. LSTM Feature Extractor Summary
# ============================================================================
print("\n" + "="*80)
print("LSTM FEATURE EXTRACTOR SUMMARY")
print("="*80)

lstm_model = LSTMFeatureExtractor(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_size=LSTM_HIDDEN_SIZE,
    num_layers=LSTM_NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# Input: (batch_size, sequence_length)
# From your code: text sequences are padded/truncated to MAX_LENGTH
lstm_input_size = (1, MAX_LENGTH)  # (batch, sequence_length)

lstm_summary = summary(
    lstm_model,
    input_size=lstm_input_size,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
    device=device,
    verbose=1,
    dtypes=[torch.long]  # Text indices are long integers
)

print(f"\nLSTM Input Shape: {lstm_input_size}")
print(f"LSTM Output Shape: {lstm_summary.summary_list[-1].output_size}")

# ============================================================================
# 3. CNN-LSTM Fusion Model Summary
# ============================================================================
print("\n" + "="*80)
print("CNN-LSTM FUSION MODEL SUMMARY")
print("="*80)

fusion_model = CNNLSTMFusionModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    lstm_hidden_size=LSTM_HIDDEN_SIZE,
    lstm_num_layers=LSTM_NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)

# The fusion model takes two inputs: images and text
fusion_input_data = [
    torch.randn(1, 1, 1568, 1568).to(device),  # Image input
    torch.randint(0, vocab_size, (1, MAX_LENGTH)).to(device)  # Text input
]

fusion_summary = summary(
    fusion_model,
    input_data=fusion_input_data,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=["var_names"],
    device=device,
    verbose=1
)

print(f"\nFusion Model Input Shapes:")
print(f"  - Image: (batch_size=1, channels=1, height=1568, width=1568)")
print(f"  - Text: (batch_size=1, sequence_length={MAX_LENGTH})")
print(f"Fusion Model Output Shape: {fusion_summary.summary_list[-1].output_size}")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("MODEL STATISTICS COMPARISON")
print("="*80)

print(f"\nCNN Feature Extractor:")
print(f"  Total Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
print(f"  Trainable Parameters: {sum(p.numel() for p in cnn_model.parameters() if p.requires_grad):,}")
print(f"  Output Features: 512")

print(f"\nLSTM Feature Extractor:")
print(f"  Total Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")
print(f"  Trainable Parameters: {sum(p.numel() for p in lstm_model.parameters() if p.requires_grad):,}")
print(f"  Output Features: 128")

print(f"\nFusion Model:")
total_params = sum(p.numel() for p in fusion_model.parameters())
trainable_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
cnn_params = sum(p.numel() for p in fusion_model.cnn_backbone.parameters())
lstm_params = sum(p.numel() for p in fusion_model.lstm_backbone.parameters())
fusion_params = sum(p.numel() for p in fusion_model.fusion.parameters())

print(f"  Total Parameters: {total_params:,}")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  CNN Backbone: {cnn_params:,} ({100*cnn_params/total_params:.1f}%)")
print(f"  LSTM Backbone: {lstm_params:,} ({100*lstm_params/total_params:.1f}%)")
print(f"  Fusion Layers: {fusion_params:,} ({100*fusion_params/total_params:.1f}%)")
print(f"  Output Classes: {NUM_CLASSES}")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)