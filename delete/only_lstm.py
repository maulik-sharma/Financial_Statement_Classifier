# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision.io import decode_image
from torchvision.transforms import v2
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# ==================== DATASET CLASSES ====================

class SquarePadTransform:
    """Custom transform to pad images to square with specified size"""
    def __init__(self, target_size=1568, fill=0):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, Image.Image):
            w, h = img.size
        else:  # Tensor
            c, h, w = img.shape

        pad_h = max(0, self.target_size - h)
        pad_w = max(0, self.target_size - w)

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        padding_transform = v2.Pad(
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill
        )
        return padding_transform(img)


class MultiModalDataset(Dataset):
    """Dataset that loads both images and OCR text for multimodal learning"""
    
    def __init__(self, annotations_file, image_dir, text_dir, vocab=None, 
                 max_length=500, image_transform=None):
        """
        Args:
            annotations_file: Path to CSV file with columns [image_name, label]
            image_dir: Directory containing images
            text_dir: Directory containing OCR text files
            vocab: Vocabulary dictionary (None for training, provided for val/test)
            max_length: Maximum sequence length for text
            image_transform: Transforms to apply to images
        """
        self.labels = pd.read_csv(annotations_file, header=None, 
                                  names=['image_name', 'label'])
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.max_length = max_length
        self.image_transform = image_transform
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocabulary()
        else:
            self.vocab = vocab
        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
    
    def build_vocabulary(self):
        """Build vocabulary from all text files"""
        print("Building vocabulary from OCR text files...")
        word_counter = Counter()
        
        for idx in range(len(self.labels)):
            image_name = self.labels.iloc[idx, 0]
            # Convert image name to text filename (e.g., "1_0.png" -> "1_0.txt")
            text_name = image_name.replace('.png', '.txt')
            text_path = os.path.join(self.text_dir, text_name)
            
            if os.path.exists(text_path):
                try:
                    with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read().lower()
                        words = text.split()
                        word_counter.update(words)
                except Exception as e:
                    print(f"Error reading {text_path}: {e}")
                    continue
        
        # Create vocabulary: word -> index
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(word_counter.most_common(), start=2):
            vocab[word] = idx
        
        print(f"Built vocabulary with {len(vocab)} words")
        return vocab
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        words = text.lower().split()
        sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence = sequence + [self.vocab['<PAD>']] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return sequence
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        # Get image
        image_name = self.labels.iloc[index, 0]
        img_path = os.path.join(self.image_dir, image_name)
        img = decode_image(img_path)
        
        if self.image_transform:
            img = self.image_transform(img)
        
        # Get text
        text_name = image_name.replace('.png', '.txt')
        text_path = os.path.join(self.text_dir, text_name)
        
        if os.path.exists(text_path):
            try:
                with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading {text_path}: {e}")
                text = ""
        else:
            text = ""  # Empty text if file doesn't exist
        
        text_sequence = self.text_to_sequence(text)
        
        # Get label (as class index, not one-hot)
        label = self.labels.iloc[index, 1]
        
        return img, torch.tensor(text_sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# %%
# ==================== MODEL ARCHITECTURES ====================

class ResidualBlock(nn.Module):
    """Residual block for CNN"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class CNNFeatureExtractor(nn.Module):
    """CNN backbone for extracting visual features from images"""
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()

        # Stage 1: Aggressive Initial Downsampling
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=3, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 2: Early Feature Extraction
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1)
        )

        # Stage 3: Mid-level Patterns
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1)
        )

        # Stage 4: High-level Document Structure
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1)
        )

        # Stage 5: Abstract Layout Features
        self.stage5 = nn.Sequential(
            ResidualBlock(512, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1)
        )

        # Global pooling to get fixed-size feature vector
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to [batch, 512]
        return x


class LSTMFeatureExtractor(nn.Module):
    """LSTM backbone for extracting features from OCR text"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, 
                 num_layers=2, dropout=0.5):
        super(LSTMFeatureExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM (unidirectional)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Unidirectional LSTM
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Feature reduction layers
        self.fc1 = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(dropout)
        
    def attention_layer(self, lstm_output):
        """Apply attention mechanism to focus on important parts of text"""
        # lstm_output shape: (batch_size, seq_length, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: [batch, seq_len, hidden_size]
        
        # Apply attention to get context vector
        context = self.attention_layer(lstm_out)  # [batch, hidden_size]
        
        # Reduce to 128 dimensions
        out = self.dropout(context)
        out = F.relu(self.fc1(out))  # [batch, 128]
        
        return out


class CNNLSTMFusionModel(nn.Module):
    """Fusion model combining CNN and LSTM for multimodal classification"""
    def __init__(self, vocab_size, embedding_dim=128, lstm_hidden_size=256, 
                 lstm_num_layers=2, num_classes=3, dropout=0.5):
        super(CNNLSTMFusionModel, self).__init__()
        
        # CNN branch for visual features
        self.cnn_backbone = CNNFeatureExtractor()
        
        # LSTM branch for text features
        self.lstm_backbone = LSTMFeatureExtractor(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
        
        # Fusion layers
        # CNN outputs 512-dim, LSTM outputs 128-dim -> concatenate to 640-dim
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, text):
        # Extract features from both modalities
        cnn_features = self.cnn_backbone(image)      # [batch, 512]
        lstm_features = self.lstm_backbone(text)     # [batch, 128]
        
        # Concatenate features
        combined = torch.cat([cnn_features, lstm_features], dim=1)  # [batch, 640]
        
        # Final classification
        output = self.fusion(combined)  # [batch, num_classes]
        
        return output


# %%
# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, texts, labels) in enumerate(train_loader):
        images = images.to(device)
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'  Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, texts, labels in val_loader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, test_loader, num_epochs=50, device='cuda'):
    """
    Complete training loop for the fusion model
    """
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Mixed precision training
    scaler = GradScaler()
    
    best_acc = 0.0
    patience_counter = 0
    patience = 10
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'{"="*60}')
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, scaler, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print('-' * 60)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, 'best_cnn_lstm_fusion_model.pth')
            print(f'✓ Best model saved with validation accuracy: {best_acc:.2f}%\n')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    print(f'\n{"="*60}')
    print(f'Training completed! Best validation accuracy: {best_acc:.2f}%')
    print(f'{"="*60}')
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(val_accs, label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('fusion_model_training_history.png', dpi=300)
    print('Training history saved as fusion_model_training_history.png')
    plt.close()


def evaluate_model(model, test_loader, device):
    """Evaluate model and generate detailed metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            texts = texts.to(device)
            
            outputs = model(images, texts)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print('\n' + '='*60)
    print('EVALUATION RESULTS')
    print('='*60)
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, 
                                target_names=['Class 0', 'Class 1', 'Class 2']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title('Confusion Matrix - CNN-LSTM Fusion Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('fusion_model_confusion_matrix.png', dpi=300)
    print('\nConfusion matrix saved as fusion_model_confusion_matrix.png')
    plt.close()
    
    return all_predictions, all_labels


# %%
# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Configuration
    ANNOTATIONS_FILE = "dataset_small/labels.csv"
    IMAGE_DIR = "dataset_small/images"
    TEXT_DIR = "dataset_small/text"
    
    # Hyperparameters
    MAX_LENGTH = 500
    EMBEDDING_DIM = 128
    LSTM_HIDDEN_SIZE = 256
    LSTM_NUM_LAYERS = 2
    NUM_CLASSES = 3
    DROPOUT = 0.5
    BATCH_SIZE = 10
    NUM_EPOCHS = 50
    
    # Image transforms
    image_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        SquarePadTransform(target_size=1568, fill=0),
        v2.Grayscale()
    ])
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f'CNN-LSTM FUSION MODEL FOR DOCUMENT CLASSIFICATION')
    print(f'{"="*60}')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'{"="*60}\n')
    
    # Load dataset
    print("Loading multimodal dataset...")
    full_dataset = MultiModalDataset(
        annotations_file=ANNOTATIONS_FILE,
        image_dir=IMAGE_DIR,
        text_dir=TEXT_DIR,
        vocab=None,  # Build vocabulary from training data
        max_length=MAX_LENGTH,
        image_transform=image_transforms
    )
    
    vocab = full_dataset.vocab
    vocab_size = full_dataset.vocab_size
    
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Split dataset (80% train, 20% test)
    train_len = int(0.8 * len(full_dataset))
    test_len = len(full_dataset) - train_len
    train_dataset, test_dataset = random_split(
        full_dataset, [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Train samples: {train_len}")
    print(f"Test samples: {test_len}")
    
    # Create test dataset with same vocabulary
    test_dataset_with_vocab = MultiModalDataset(
        annotations_file=ANNOTATIONS_FILE,
        image_dir=IMAGE_DIR,
        text_dir=TEXT_DIR,
        vocab=vocab,  # Use training vocabulary
        max_length=MAX_LENGTH,
        image_transform=image_transforms
    )
    # Use same test indices
    test_dataset.dataset = test_dataset_with_vocab
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=2)
    
    # Initialize model
    print("\nInitializing CNN-LSTM Fusion Model...")
    model = CNNLSTMFusionModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_num_layers=LSTM_NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    )
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Count parameters for each component
    cnn_params = sum(p.numel() for p in model.cnn_backbone.parameters())
    lstm_params = sum(p.numel() for p in model.lstm_backbone.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"\nComponent breakdown:")
    print(f"CNN backbone: {cnn_params:,} parameters")
    print(f"LSTM backbone: {lstm_params:,} parameters")
    print(f"Fusion layers: {fusion_params:,} parameters")
    
    # Train the model
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, 
        num_epochs=NUM_EPOCHS, 
        device=device
    )
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load('best_cnn_lstm_fusion_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Final evaluation
    predictions, labels = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("Training and evaluation completed successfully!")
    print("="*60)
    print("\nSaved files:")
    print("  - best_cnn_lstm_fusion_model.pth (model checkpoint)")
    print("  - fusion_model_training_history.png (training curves)")
    print("  - fusion_model_confusion_matrix.png (confusion matrix)")

# %%
