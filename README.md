# 🧠 Doctane by Zeross

> **Multimodal Intelligent Document Analysis and Understanding System**  
> *(Under Development — 🚧 Final details will be added post-deployment)*

<p align="center">
  <img src="assets/doctane_logo_placeholder.png" alt="Doctane Logo" width="200"/>
</p>

**Doctane** is a vision-language-based pipeline built to automate document analysis, including OCR, layout understanding, semantic extraction, and captioning — all under one modular framework. Designed for business documents, forms, invoices, and scanned reports.

---

## 🧬 Visual Overview

<p align="center">
  <img src="assets/doctane_architecture_placeholder.png" alt="Doctane Architecture Diagram" width="700"/>
</p>

> 🔧 TODO: Replace the above placeholder with the finalized Doctane architecture.
The pipeline processes documents through the following stages:
> 1.  **Preprocessing**: Handles page orientation detection and optional straightening.
> 2.  **Text Detection**: A segmentation-based model (e.g., LinkNet) identifies word or line bounding boxes (both straight and rotated).
> 3.  **Text Recognition**: Crops from the detection stage are fed into a recognition model to transcribe the text.
> 4.  **Document Assembly**: The final output is a structured `Document` object containing pages, blocks, lines, and words with their geometry and confidence scores. The system also supports Key Information Extraction (KIE).

---

## 📂 Project Structure

```
Doctane/
├── assets/                  # Logos, diagrams, figures
├── configs/                 # Config files for training, model params, paths
├── data/                    # Dataset loaders and preparation scripts
├── models/                  # Model definitions
├── train/                   # Training scripts for each module
├── test/                    # Test scripts and evaluations
├── inference/               # Inference scripts and helpers
├── utils/                   # Shared utilities
├── notebooks/               # Jupyter notebooks for exploration and testing
│   └── demo.ipynb
├── scripts/                 # Shell scripts or automation
├── requirements.txt         # Dependancies
├── setup.py
└── README.md

```

---

## 🔍 Key Features

- **End-to-End OCR**: Unified pipeline for text detection and recognition.
- **Robust Detection**: Handles both straight and **rotated** text using segmentation-based models.
- **Accurate Recognition**: High-accuracy text recognition with support for custom vocabularies.
- **Synthetic Data Generation**: Includes `WordGenerator` to create training data on-the-fly, reducing the need for large pre-existing datasets.
- **Layout-Aware**: Automatically detects page orientation and can straighten pages before analysis.
- **Structured Output**: Exports results as rich, hierarchical `Document` objects (`Page`, `Block`, `Line`, `Word`) and supports **hOCR** format.
- **Distributed Training**: Comes with `DDP` (Distributed Data Parallel) scripts for faster training on multiple GPUs.
- **Extensible**: Modular design with a `DocumentBuilder` and hooks for custom post-processing.
- **Multi-lingual Support**: Features language detection and allows training with different vocabularies (e.g., French).

---

## 📊 Supported Datasets

- **Custom Datasets**: Easily train on your own data using the `DetectionDataset` and `RecognitionDataset` classes, which expect a simple folder structure with images and a `detection_labels.json` or `recognition_labels.json` file.
- **Synthetic Data**: The `WordGenerator` can create an arbitrary amount of training data for the recognition model, perfect for bootstrapping or fine-tuning.
- **Public Datasets (Planned)**: Integration with FUNSD, PubLayNet, and DocVQA is on the roadmap.

---

## ⚙️ Installation

```bash
git clone https://github.com/zeross-ai/doctane.git
cd doctane
pip install -r requirements.txt
```

---

## 🚀 Running Inference

```bash
python scripts/run_inference.py \
    --input_dir data/raw \
    --output_dir results/outputs \
    --model_type layoutlm
```

---

## 📈 Results & Metrics

🔧 TODO: Add quantitative performance (Precision, Recall, F1)  
🔧 TODO: Add qualitative visual output samples

---

## 💡 Future Roadmap

- [x] Modular code base
- [x] Baseline OCR and layout engine
- [ ] 🔧 End-to-end document captioning
- [ ] 🔧 Visual layout annotation
- [ ] 🔧 Web Dashboard for interaction

---

## 🤝 Credits

- Hugging Face `LayoutLM`, `Donut`, `doctr`
- Tesseract OCR, EasyOCR
- pytorch-segmentation-models
- PyMuPDF, PDFPlumber

---

## 📜 License

🔧 TODO: Add final license 

---
