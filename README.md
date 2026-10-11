# 🌊 ReefLens — Marine Species AI Detection & Vision Platform

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-88.84%25-brightgreen.svg)](#model-performance)

**ReefLens** is a computer vision and machine learning platform for underwater biodiversity monitoring and marine species identification. It features both a desktop OpenCV detection engine and an interactive local web dashboard with real-time video streaming, motion subtraction, and instant image classification.

---

## 🪸 Key Features

- **11 Marine Species Classified**: Puffers, Sea Rays, Sea Urchins, Seahorse, Seal, Sharks, Shrimp, Squid, Starfish, Turtle/Tortoise, and Whale.
- **88.84% Test Accuracy**: Evaluated on 1,362 test specimens with precision up to 100%.
- **Live Video Detection**: Real-time object tracking on underwater footage (`marine_video.mp4`) using MOG2 background subtraction and KNN classification at 35+ FPS.
- **SIFT Feature Matcher**: Alternative keypoint matching engine utilizing FLANN descriptors.
- **Interactive Web App**: Modern oceanic glassmorphism UI with one-click test specimens, drag-and-drop classification, and candidate probability distributions.

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/RaziyaChikkale/ReefLens.git
cd ReefLens

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Local Web App

```bash
python app.py
```
Open **[http://127.0.0.1:5001](http://127.0.0.1:5001)** in your browser.

---

## 🖥️ Running CLI Scripts Directly

```bash
# Single image prediction with OpenCV display
python predict.py

# Real-time video detection on marine_video.mp4 (Press 'q' to quit)
python video.py

# SIFT feature-matching detection (Press 'q' to quit)
python main1.py

# Evaluate test-split metrics (Accuracy, Precision, Recall, F1)
python evaluate.py

# Retrain KNN model on the dataset
python train.py
```

---

## 📊 Model Performance

| Species Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Puffers** | 1.00 | 0.89 | 0.94 | 57 |
| **Sea Rays** | 0.91 | 0.84 | 0.87 | 103 |
| **Sea Urchins** | 0.95 | 0.88 | 0.91 | 116 |
| **Seahorse** | 0.99 | 0.86 | 0.92 | 95 |
| **Seal** | 0.76 | 0.84 | 0.80 | 83 |
| **Sharks** | 0.83 | 0.90 | 0.86 | 118 |
| **Shrimp** | 0.95 | 0.79 | 0.86 | 98 |
| **Squid** | 0.85 | 0.91 | 0.88 | 97 |
| **Starfish** | 0.93 | 0.82 | 0.87 | 100 |
| **Turtle / Tortoise** | 0.88 | 0.95 | 0.91 | 381 |
| **Whale** | 0.85 | 0.90 | 0.88 | 114 |
| **Overall Accuracy** | **88.84%** | | | **1,362** |

---

## 📁 Repository Structure

```
ReefLens/
├── app.py                 # Flask server & REST API
├── train.py               # Model training script
├── evaluate.py            # Evaluation & metrics generator
├── predict.py             # Single image inference CLI
├── video.py               # Motion MOG2 + KNN detection on video
├── main1.py               # SIFT + FLANN keypoint matching on video
├── check_dataset.py       # Dataset health check utility
├── dataset/               # 6,808 curated marine images across 11 classes
├── models/
│   └── label_encoder.pkl  # Class encoder artifact
├── marine_video.mp4       # Sample underwater footage
├── templates/
│   └── index.html         # Frontend web application template
├── static/
│   ├── css/style.css      # Oceanic dark design system
│   └── js/app.js          # Interactive UI & client logic
└── requirements.txt       # Project dependencies
```

---

## 📜 License
MIT License. Created for marine biodiversity conservation and research.
