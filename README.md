# Invoice-Information-Extraction-Project

google colob - https://colab.research.google.com/drive/1cjSrd62_bTLQE0R_7sGTmiC69aJaHkHx?usp=sharing
```
# 📄 Invoice Data Extractor

An AI-powered invoice processing system that extracts structured data from invoice images using LayoutLMv3 and OCR technology.

## 🚀 Features

- **Automated Invoice Processing**: Extract key information from invoice images
- **Multi-field Detection**: Invoice number, dates, amounts, vendor details, and addresses
- **Custom Model Training**: Train on your own invoice dataset for improved accuracy  
- **OCR Integration**: Built-in text detection using Tesseract OCR
- **Confidence Scoring**: Quality metrics for extracted data
- **Flexible Input**: Supports various image formats (PNG, JPG, PDF)

## 📋 Extracted Fields

- Invoice Number
- Invoice Date & Due Date
- Vendor/Biller Name & Address
- Total Amount, Subtotal, Tax
- Line Items (configurable)

## 🛠️ Installation

### Prerequisites
```bash
# Install Python dependencies
pip install torch torchvision transformers
pip install layoutlmv3 pillow opencv-python
pip install pytesseract scikit-learn numpy pandas

# Install Tesseract OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Setup
```bash
git clone <repository-url>
cd invoice-extractor
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Usage
```python
from invoice_extractor import CustomInvoiceExtractor

# Initialize extractor
extractor = CustomInvoiceExtractor()

# Extract data from invoice
result = extractor.extract_from_image("invoice.jpg")

# Print results
for field, value in result.items():
    if value:
        print(f"{field}: {value}")
```

### Training Your Own Model
```python
# Prepare your dataset
dataset = load_your_invoice_dataset()

# Train model
trainer = CustomInvoiceTrainer()
train_data, val_data = prepare_training_data(dataset)
trained_model = trainer.train(train_data, val_data)
```

## 📊 Dataset Format

Your training data should follow this structure:
```python
{
    "image_path": "invoice1.jpg",
    "words": ["Invoice", "#", "12345", "Date:", "2024-01-15"],
    "boxes": [[10, 20, 50, 30], [55, 20, 65, 30], ...],
    "labels": ["O", "O", "B-INVOICE_NUMBER", "O", "B-INVOICE_DATE"]
}
```

### Label Format (BIO Tagging)
- `B-FIELD_NAME`: Beginning of entity
- `I-FIELD_NAME`: Inside/continuation of entity  
- `O`: Outside (not part of any entity)

**Supported Entity Types**:
- `INVOICE_NUMBER`, `INVOICE_DATE`, `DUE_DATE`
- `BILLER_NAME`, `BILLER_ADDRESS`
- `TOTAL`, `SUBTOTAL`, `TAX`

## 🔧 Configuration

### Model Settings
```python
# Custom model path
extractor = CustomInvoiceExtractor(
    model_path="/path/to/your/model"
)

# Training parameters
training_args = TrainingArguments(
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=1
)
```

### OCR Settings
- Confidence threshold: 30% (adjustable)
- Supports multiple languages via Tesseract
- Automatic image preprocessing

## 📈 Performance

- **Base Model**: General document understanding
- **Fine-tuned**: 85-95% accuracy on domain-specific invoices
- **Processing Speed**: ~2-5 seconds per invoice
- **Supported Formats**: JPG, PNG, PDF (first page)

## 🐛 Troubleshooting

### Common Issues

**No text detected**:
```bash
# Check Tesseract installation
tesseract --version

# Verify image quality and format
```

**WANDB API Key Error**:
```python
import os
os.environ["WANDB_DISABLED"] = "true"
```

**Low extraction accuracy**:
- Ensure high-quality input images
- Fine-tune model on your specific invoice types
- Adjust OCR confidence threshold

### Debug Mode
```python
# Enable detailed logging
result = test_extractor("invoice.jpg")  # Shows processing steps
```

## 🏗️ Project Structure

```
invoice-extractor/
├── models/
│   └── custom_invoice_model/    # Trained model files
├── data/
│   ├── training/               # Training images
│   └── validation/            # Validation images
├── src/
│   ├── extractor.py          # Main extraction logic
│   ├── trainer.py            # Model training
│   └── processor.py          # OCR and preprocessing
├── examples/
│   └── sample_invoices/      # Example images
└── README.md
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-field`)
3. Commit changes (`git commit -am 'Add new field extraction'`)
4. Push to branch (`git push origin feature/new-field`)
5. Create Pull Request

## 📞 Support

- **Issues**: [GitHub Issues](link-to-issues)
- **Documentation**: [Full Documentation](link-to-docs)
- **Examples**: See `examples/` directory

---

**Built with**: LayoutLMv3, PyTorch, Transformers, Tesseract OCR
```
