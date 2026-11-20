import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms import v2
from PIL import Image
import easyocr
from pathlib import Path
import json
import time
import warnings

warnings.filterwarnings('ignore')


class SquarePadTransform:
    """Pad images to square with specified size"""

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
    """CNN backbone for visual features"""

    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=3, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 128, stride=1)
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 256, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1),
        )
        self.stage5 = nn.Sequential(
            ResidualBlock(512, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
            ResidualBlock(512, 512, stride=1)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x


class LSTMFeatureExtractor(nn.Module):
    """LSTM backbone for text features"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256,
                 num_layers=2, dropout=0.5):
        super(LSTMFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout = nn.Dropout(dropout)

    def attention_layer(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        context = self.attention_layer(lstm_out)
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        return out


class CNNLSTMFusionModel(nn.Module):
    """Fusion model combining CNN and LSTM"""

    def __init__(self, vocab_size, embedding_dim=128, lstm_hidden_size=256,
                 lstm_num_layers=2, num_classes=3, dropout=0.5):
        super(CNNLSTMFusionModel, self).__init__()
        self.cnn_backbone = CNNFeatureExtractor()
        self.lstm_backbone = LSTMFeatureExtractor(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout
        )
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
        cnn_features = self.cnn_backbone(image)
        lstm_features = self.lstm_backbone(text)
        combined = torch.cat([cnn_features, lstm_features], dim=1)
        output = self.fusion(combined)
        return output



class FinancialDocumentClassifier:
    """
    Production-ready classifier for financial documents
    """

    CLASS_NAMES = {
        0: "Balance Sheet",
        1: "Cash Flow",
        2: "Income Statement",
        3: "Others"  # Low confidence predictions
    }

    def __init__(self, model_path, vocab_path, device=None, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")

        # Initialize model
        print("Initializing model...")
        self.model = CNNLSTMFusionModel(
            vocab_size=self.vocab_size,
            embedding_dim=128,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            num_classes=3,
            dropout=0.5
        )

        # Load model weights
        print(f"Loading model weights from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()


        # Initialize OCR reader
        print("Initializing EasyOCR (this may take a moment)...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("EasyOCR ready!")

        # Image preprocessing pipeline
        self.image_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            SquarePadTransform(target_size=1568, fill=0),
            v2.Grayscale()
        ])

        self.max_length = 500


    def extract_text_from_image(self, image_path):
        """
        Extract text from image using EasyOCR

        Args:
            image_path: Path to image file

        Returns:
            Extracted text as string
        """
        try:
            result = self.reader.readtext(image_path, detail=0, paragraph=True)
            text = ' '.join(result)
            return text
        except Exception as e:
            print(f"Warning: Error extracting text from {image_path}: {e}")
            return ""

    def preprocess_image(self, image_path):
        """
        Preprocess image for model input

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor
        """
        try:
            img = decode_image(str(image_path))
            img = self.image_transform(img)
            return img
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            # Return blank image as fallback
            return torch.zeros((1, 1568, 1568), dtype=torch.float32)

    def text_to_sequence(self, text):
        """
        Convert text to sequence of vocabulary indices

        Args:
            text: Input text string

        Returns:
            List of vocabulary indices (padded/truncated to max_length)
        """
        text = text.lower().replace('\n', ' ').replace('\t', ' ')
        words = text.split()
        sequence = [self.vocab.get(word, self.vocab.get('<UNK>', 1)) for word in words]

        # Pad or truncate
        if len(sequence) < self.max_length:
            sequence = sequence + [self.vocab.get('<PAD>', 0)] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]

        return sequence

    def predict(self, image_path):
        """
        Classify a financial document image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary containing:
                - predicted_class: Class ID (0-3)
                - class_name: Human-readable class name
        """
        start_time = time.time()

        # Convert to Path object
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"\nProcessing: {image_path.name}")
        print("-" * 60)

        # Extract text using OCR
        print("Extracting text with EasyOCR...")
        ocr_start = time.time()
        text = self.extract_text_from_image(str(image_path))
        ocr_time = time.time() - ocr_start
        print(f"OCR completed in {ocr_time:.2f}s")
        print(f"Extracted text length: {len(text)} characters")

        # Preprocess image
        print("Preprocessing image...")
        img = self.preprocess_image(image_path)
        img = img.unsqueeze(0).to(self.device)

        # Preprocess text
        print("Preprocessing text...")
        text_sequence = self.text_to_sequence(text)
        text_tensor = torch.tensor([text_sequence], dtype=torch.long).to(self.device)

        # Run inference
        print("Running inference...")
        with torch.no_grad():
            outputs = self.model(img, text_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            max_prob, predicted_class = torch.max(probabilities, dim=0)

            predicted_class = predicted_class.item()
            confidence = max_prob.item()

        is_confident = confidence >= self.confidence_threshold

        if not is_confident:
            final_class = 3  # "Others" class
            final_name = self.CLASS_NAMES[3]
        else:
            final_class = predicted_class
            final_name = self.CLASS_NAMES[predicted_class]


        # Prepare result
        result = {
            'predicted_class': final_class,
            'class_name': final_name,
        }


        return result


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "best_model.pth"
    VOCAB_PATH = "vocabulary.json"  # You need to save this from training
    CONFIDENCE_THRESHOLD = 0.8

    # Initialize classifier
    classifier = FinancialDocumentClassifier(
        model_path=MODEL_PATH,
        vocab_path=VOCAB_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # Single prediction example
    print("\n" + "=" * 60)
    print("SINGLE PREDICTION EXAMPLE")
    print("=" * 60)
    image_path = r"test//test9.jpg"

    # Make prediction
    result = classifier.predict(image_path)

    # Access results
    print(f"\nFinal Classification: {result['class_name']}")

