# 🔬 Doctane by Zeross

> **Multimodal Intelligent Document Analysis and Understanding System**

<p align="center">
  <a href="https://github.com/Purushothaman-natarajan/doctane/stargazers">
    <img src="https://img.shields.io/github/stars/Purushothaman-natarajan/doctane?style=flat-square" alt="Stars">
  </a>
  <a href="https://github.com/Purushothaman-natarajan/doctane/releases">
    <img src="https://img.shields.io/github/v/release/Purushothaman-natarajan/doctane?style=flat-square" alt="Version">
  </a>
  <a href="https://github.com/Purushothaman-natarajan/doctane/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Purushothaman-natarajan/doctane?style=flat-square" alt="License">
  </a>
  <a href="https://pypi.org/project/doctane/">
    <img src="https://img.shields.io/pypi/dm/doctane?style=flat-square" alt="PyPI Downloads">
  </a>
</p>

---

**Doctane** is a vision-language-based pipeline built to automate document analysis, including OCR, layout understanding, semantic extraction, and captioning — all under one modular framework. Designed for business documents, forms, invoices, and scanned reports.

---

## ✨ Key Features

- **End-to-End OCR**: Unified pipeline for text detection and recognition
- **Robust Detection**: Handles both straight and **rotated** text using segmentation-based models
- **Accurate Recognition**: High-accuracy text recognition with support for custom vocabularies
- **Synthetic Data Generation**: Includes `WordGenerator` to create training data on-the-fly
- **Layout-Aware**: Automatically detects page orientation and can straighten pages before analysis
- **Structured Output**: Exports results as rich, hierarchical `Document` objects (Page, Block, Line, Word) and supports **hOCR** format
- **Distributed Training**: Comes with `DDP` (Distributed Data Parallel) scripts for faster training on multiple GPUs
- **Extensible**: Modular design with a `DocumentBuilder` and hooks for custom post-processing
- **Multi-lingual Support**: Features language detection and allows training with different vocabularies

---

## 🧬 Processing Pipeline

The pipeline processes documents through the following stages:

1. **Preprocessing**: Handles page orientation detection and optional straightening
2. **Text Detection**: A segmentation-based model (e.g., LinkNet) identifies word or line bounding boxes (both straight and rotated)
3. **Text Recognition**: Crops from the detection stage are fed into a recognition model to transcribe the text
4. **Document Assembly**: The final output is a structured `Document` object containing pages, blocks, lines, and words with their geometry and confidence scores. The system also supports Key Information Extraction (KIE)

---

## 🏗️ Supported Models

### Detection Models
| Model | Description |
|-------|-------------|
| LinkNet | Fast and lightweight segmentation |
| DeepLabV3+ | High accuracy for complex layouts |
| SegFormer | State-of-the-art transformer-based |
| Faster R-CNN | Region-based detection |

### Recognition Models
| Model | Description |
|-------|-------------|
| SAR | Sequence Approximation Recognition |
| ViTSTR | Vision Transformer for STR |
| CRNN | CNN + RNN + CTC |
| TrBA | Transformer-based Recognition |
| MASTER | Multi-Aspect Self-Attention |
| ABINet | Attention-based Iterative Network |
| VisionLAN | Vision-Language Awareness Network |
| DiT | Diffusion Transformer |

---

## 🚀 Quick Start

### One-Click Setup (Windows)

```batch
git clone https://github.com/Purushothaman-natarajan/doctane.git
cd doctane
run.bat
```

### Manual Setup

```bash
# Clone the repository
git clone https://github.com/Purushothaman-natarajan/doctane.git
cd doctane

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run the web interface
streamlit run app.py
```

Open **http://localhost:8501** in your browser to use Doctane.

---

## 📖 Documentation

- [Documentation](docs.html) - Comprehensive API and usage guide
- [Architecture](architecture.html) - System design and component details
- [Developer Guide](developer-guide.html) - Setup, training, and contribution guide
- [Profile](profile.html) - Developer information

---

## 📂 Project Structure

```
Doctane/
├── index.html              # Landing page
├── docs.html               # Documentation
├── architecture.html       # System architecture
├── developer-guide.html    # Developer guide
├── profile.html            # Developer profile
├── app.py                  # Streamlit web interface
├── run.bat                 # Windows launcher
│
├── doctane/                # Main package
│   ├── predictor/          # Predictor classes
│   ├── ocr_pipeline/      # OCR pipeline
│   ├── models/             # Model definitions
│   │   ├── detection/      # Detection models
│   │   ├── recognition/    # Recognition models
│   │   └── classification/ # Classification models
│   ├── datasets/           # Dataset loaders
│   ├── train/              # Training scripts
│   ├── utils/              # Utilities
│   └── evaluate/           # Evaluation
│
├── configs/                 # YAML configurations
├── data/                    # Data directory
├── scripts/                 # Utility scripts
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks
└── requirements.txt         # Dependencies
```

---

## 💻 Usage Examples

### Python API

```python
import numpy as np
from PIL import Image
from doctane.ocr_pipeline.ocr_predictor import OCRPredictor

# Load image
image = Image.open("document.jpg").convert("RGB")
np_image = np.array(image)

# Run OCR
predictor = OCRPredictor()
output = predictor([np_image])

# Access results
for page in output.pages:
    print(page)
```

### Command Line

```bash
python scripts/run_inference.py \
    --input_dir data/raw \
    --output_dir results/outputs \
    --model_type seg_linknet_resnet50
```

---

## 📊 Supported Datasets

- **Custom Datasets**: Easily train on your own data using the `DetectionDataset` and `RecognitionDataset` classes
- **Synthetic Data**: The `WordGenerator` can create arbitrary amounts of training data
- **Public Datasets (Planned)**: FUNSD, PubLayNet, DocVQA

---

## 🛠️ Training

### Text Detection

```bash
python train/text_detection/train_detection.py \
    --config configs/detection.yaml \
    --epochs 100
```

### Text Recognition

```bash
python train/text_recognition/train_recognition.py \
    --config configs/recognition.yaml \
    --epochs 50
```

### Distributed Training (DDP)

```bash
torchrun --nproc_per_node=4 \
    train/text_detection/train_detection_ddp.py \
    --config configs/detection.yaml
```

---

## 🔧 Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BUILD_VERSION` | Package version | 0.1.0a0 |
| `CHECKPOINT_DIR` | Model checkpoints | ./checkpoints |
| `DATA_DIR` | Training data | ./data |
| `LOG_DIR` | Training logs | ./logs |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make changes and add tests
4. Run linting and type checking
5. Submit a pull request

---

## 📈 Future Roadmap

- [ ] End-to-end document captioning
- [ ] Visual layout annotation
- [ ] Web Dashboard for interaction
- [ ] Multi-language support expansion
- [ ] PDF export functionality

---

## 🙏 Acknowledgments

- Hugging Face `LayoutLM`, `Donut`, `doctr`
- Tesseract OCR, EasyOCR
- pytorch-segmentation-models
- PyMuPDF, PDFPlumber

---

## 📜 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## 👨‍💻 Developer

**Purushothaman Natarajan**

- GitHub: [@Purushothaman-natarajan](https://github.com/Purushothaman-natarajan)
- Portfolio: [purushothaman-natarajan.github.io](https://purushothaman-natarajan.github.io/)
- Email: purushothamanprt@gmail.com

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Doctane-ai">Doctane</a>
</p>