"""
LSTM Text Classification for OCR-based Document Classification
This script implements multiple LSTM variants for text classification
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm


class TextDataset(Dataset):
    """Custom Dataset for loading text files and their labels"""
    
    def __init__(self, csv_file, text_dir, vocab=None, max_length=500):
        """
        Args:
            csv_file: Path to the labels CSV file
            text_dir: Directory containing text files
            vocab: Vocabulary dictionary (None for training, provided for validation/test)
            max_length: Maximum sequence length for padding/truncating
        """
        self.labels_df = pd.read_csv(csv_file, header=None, names=['image_name', 'label'], dtype={'image_name': str, 'label': int})
        self.text_dir = text_dir
        self.max_length = max_length
        
        # Convert image names to text file names
        self.labels_df['text_name'] = self.labels_df['image_name'].astype(str).str.replace('.png', '.txt')
        
        # Filter out rows where text files don't exist
        self.labels_df = self.labels_df[self.labels_df['text_name'].apply(
            lambda x: os.path.exists(os.path.join(text_dir, x))
        )].reset_index(drop=True)
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = self.build_vocabulary()
        else:
            self.vocab = vocab
            
        self.vocab_size = len(self.vocab)
        
    def build_vocabulary(self):
        """Build vocabulary from all text files"""
        word_counter = Counter()
        
        for text_name in self.labels_df['text_name']:
            text_path = os.path.join(self.text_dir, text_name)
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
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
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        text_name = self.labels_df.iloc[idx]['text_name']
        label = self.labels_df.iloc[idx]['label']
        
        text_path = os.path.join(self.text_dir, text_name)
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading {text_path}: {e}")
            text = ""
        
        sequence = self.text_to_sequence(text)
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with Attention for text classification"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout=0.5):
        super(BiLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(128)
        
    def attention_layer(self, lstm_output):
        """Apply attention mechanism"""
        # lstm_output shape: (batch_size, seq_length, hidden_size*2)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector
        
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention
        context = self.attention_layer(lstm_out)
        
        # Fully connected layers with batch normalization
        out = self.dropout(context)
        out = F.relu(self.batch_norm1(self.fc1(out)))
        out = self.dropout(out)
        out = F.relu(self.batch_norm2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


class StackedLSTMClassifier(nn.Module):
    """Stacked LSTM with residual connections"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout=0.5):
        super(StackedLSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Stacked LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        last_hidden = hidden[-1, :, :]
        
        out = self.dropout(last_hidden)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


class GRUClassifier(nn.Module):
    """GRU-based classifier (alternative to LSTM)"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes, dropout=0.5):
        super(GRUClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        
        # Concatenate forward and backward hidden states
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        out = self.dropout(hidden_concat)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total*100})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(val_loader, desc='Validating', leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct/total*100})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name):
    """Train the model with learning rate scheduling"""
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining {model_name}...")
    
    for epoch in tqdm(range(num_epochs), desc=f'{model_name} Training', position=0):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Update progress bar description
        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'best_{model_name}_model.pth')
            tqdm.write(f'  ✓ Best model saved with validation accuracy: {val_acc:.2f}%')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            tqdm.write(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return train_losses, val_losses, train_accs, val_accs, best_val_acc


def evaluate_model(model, test_loader, device, model_name):
    """Evaluate the model on test set"""
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    progress_bar = tqdm(test_loader, desc=f'Evaluating {model_name}', leave=False)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f'\n{model_name} Test Results:')
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Weighted F1 Score: {f1:.4f}')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(all_labels, all_predictions, 
                                target_names=['Class 0', 'Class 1', 'Class 2']))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    print(f'\nConfusion matrix saved as {model_name}_confusion_matrix.png')
    
    return accuracy, f1, all_predictions, all_labels


def plot_training_history(histories, model_names):
    """Plot training history for multiple models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (history, model_name) in enumerate(zip(histories, model_names)):
        train_losses, val_losses, train_accs, val_accs = history
        
        # Plot losses
        axes[0, 0].plot(train_losses, label=f'{model_name} - Train', alpha=0.7)
        axes[0, 0].plot(val_losses, label=f'{model_name} - Val', alpha=0.7, linestyle='--')
        
        # Plot accuracies
        axes[0, 1].plot(train_accs, label=f'{model_name} - Train', alpha=0.7)
        axes[0, 1].plot(val_accs, label=f'{model_name} - Val', alpha=0.7, linestyle='--')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Model comparison - validation accuracy
    best_val_accs = [max(history[3]) for history in histories]
    axes[1, 0].bar(model_names, best_val_accs, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_ylabel('Best Validation Accuracy (%)')
    axes[1, 0].set_title('Model Comparison - Best Validation Accuracy')
    axes[1, 0].grid(True, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(best_val_accs):
        axes[1, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    
    # Hide the last subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('all_models_training_history.png', dpi=300)
    print('\nTraining history comparison saved as all_models_training_history.png')
    plt.close()


def save_model_package(model, vocab, config, model_name):
    """Save complete model package with all necessary components"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = f'saved_models/{model_name}_{timestamp}'
    os.makedirs(package_dir, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), f'{package_dir}/model_weights.pth')
    
    # Save vocabulary
    with open(f'{package_dir}/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    
    # Save configuration
    with open(f'{package_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save model architecture as text
    with open(f'{package_dir}/model_architecture.txt', 'w') as f:
        f.write(str(model))
    
    print(f'\nComplete model package saved to: {package_dir}/')
    return package_dir


def load_model_package(package_dir, model_class, device):
    """Load complete model package"""
    
    # Load configuration
    with open(f'{package_dir}/config.json', 'r') as f:
        config = json.load(f)
    
    # Load vocabulary
    with open(f'{package_dir}/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    # Initialize model
    model = model_class(**config['model_params']).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(f'{package_dir}/model_weights.pth'))
    
    print(f'Model package loaded from: {package_dir}/')
    return model, vocab, config


def main():
    """Main function to train and evaluate multiple LSTM models"""
    
    # Configuration
    CSV_FILE = 'dataset_small/labels.csv'
    TEXT_DIR = 'dataset_small/text'
    MAX_LENGTH = 500
    EMBEDDING_DIM = 128
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    NUM_CLASSES = 3
    DROPOUT = 0.5
    BATCH_SIZE = 32
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Load dataset and build vocabulary
    print('\nLoading dataset and building vocabulary...')
    full_dataset = TextDataset(CSV_FILE, TEXT_DIR, vocab=None, max_length=MAX_LENGTH)
    vocab = full_dataset.vocab
    vocab_size = full_dataset.vocab_size
    print(f'Vocabulary size: {vocab_size}')
    print(f'Total samples: {len(full_dataset)}')
    
    # Split dataset
    labels_df = full_dataset.labels_df
    
    # Check class distribution
    class_counts = labels_df['label'].value_counts().sort_index()
    print('\nClass distribution:')
    for cls, count in class_counts.items():
        print(f'  Class {cls}: {count} samples ({count/len(labels_df)*100:.2f}%)')
    
    # Split data
    train_val_df, test_df = train_test_split(
        labels_df, test_size=TEST_SIZE, random_state=42, stratify=labels_df['label']
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42, 
        stratify=train_val_df['label']
    )
    
    print(f'\nTrain samples: {len(train_df)}')
    print(f'Validation samples: {len(val_df)}')
    print(f'Test samples: {len(test_df)}')
    
    # Save split dataframes (only image_name and label columns)
    train_df[['image_name', 'label']].to_csv('temp_train.csv', index=False, header=False)
    val_df[['image_name', 'label']].to_csv('temp_val.csv', index=False, header=False)
    test_df[['image_name', 'label']].to_csv('temp_test.csv', index=False, header=False)
    
    # Create datasets
    train_dataset = TextDataset('temp_train.csv', TEXT_DIR, vocab=vocab, max_length=MAX_LENGTH)
    val_dataset = TextDataset('temp_val.csv', TEXT_DIR, vocab=vocab, max_length=MAX_LENGTH)
    test_dataset = TextDataset('temp_test.csv', TEXT_DIR, vocab=vocab, max_length=MAX_LENGTH)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Configuration for saving
    config = {
        'model_params': {
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'dropout': DROPOUT
        },
        'training_params': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'max_length': MAX_LENGTH
        }
    }
    
    # Define models to train
    models_config = [
        ('BiLSTM_Attention', BiLSTMClassifier),
        ('Stacked_LSTM', StackedLSTMClassifier),
        ('BiGRU', GRUClassifier)
    ]
    
    all_histories = []
    all_results = []
    model_names = []
    
    # Train each model
    for model_name, model_class in tqdm(models_config, desc='Training Models'):
        print(f'\n{"="*80}')
        print(f'Training {model_name}')
        print(f'{"="*80}')
        
        # Initialize model
        model = model_class(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=NUM_CLASSES,
            dropout=DROPOUT
        ).to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Train model
        history = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            scheduler, NUM_EPOCHS, device, model_name
        )
        
        # Evaluate on test set
        print(f'\nLoading best {model_name} model for evaluation...')
        checkpoint = torch.load(f'best_{model_name}_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        accuracy, f1, predictions, labels = evaluate_model(model, test_loader, device, model_name)
        
        # Save complete model package
        package_dir = save_model_package(model, vocab, config, model_name)
        
        # Store results
        all_histories.append(history[:4])  # train_losses, val_losses, train_accs, val_accs
        all_results.append({
            'model_name': model_name,
            'best_val_acc': history[4],
            'test_accuracy': accuracy,
            'test_f1': f1,
            'package_dir': package_dir
        })
        model_names.append(model_name)
    
    # Plot comparison
    plot_training_history(all_histories, model_names)
    
    # Print final comparison
    print(f'\n{"="*80}')
    print('FINAL MODEL COMPARISON')
    print(f'{"="*80}')
    print(f'{"Model":<20} {"Best Val Acc":<15} {"Test Acc":<15} {"Test F1":<15}')
    print(f'{"-"*80}')
    for result in all_results:
        print(f'{result["model_name"]:<20} '
              f'{result["best_val_acc"]:<15.2f} '
              f'{result["test_accuracy"]*100:<15.2f} '
              f'{result["test_f1"]:<15.4f}')
    
    # Save results to JSON
    with open('model_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print('\nResults saved to model_comparison_results.json')
    
    # Clean up temporary files
    os.remove('temp_train.csv')
    os.remove('temp_val.csv')
    os.remove('temp_test.csv')
    
    print('\n✓ Training completed successfully!')
    print('✓ All models saved with complete packages')
    print('✓ Visualizations and reports generated')


if __name__ == '__main__':
    main()
