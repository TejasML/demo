# 🫁 LungLens AI — Pneumonia Detection from Chest X-Rays

[![HuggingFace](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/Tejas-ML/pneumonia-detection-app)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B)](https://streamlit.io/)
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle%20Dual%20T4%20GPUs-20BEFF)](https://www.kaggle.com/)

A deep learning project that detects **Pneumonia from pediatric chest X-ray images** using two independently developed CNN models — a **Custom CNN built from scratch** and a **DenseNet121 Transfer Learning model** — both deployed in a production-quality Streamlit web application named **LungLens AI**.

> 🚀 **Live Demo** → [huggingface.co/spaces/Tejas-ML/pneumonia-detection-app](https://huggingface.co/spaces/Tejas-ML/pneumonia-detection-app)

---

## 📌 Project Overview

Pneumonia is a life-threatening lung infection diagnosed by identifying cloudy opacities and infiltrates in chest X-rays. This project trains two CNN architectures to automate that detection and compares their real-world performance on an unseen test set.

**Binary Classification Task:**

| Label | Description |
|-------|-------------|
| `NORMAL` | Healthy lungs — dark, clear X-ray |
| `PNEUMONIA` | Infected lungs — cloudy white opacities present |

> ⚠️ **Clinical Design Priority**: In medical AI, a **False Negative** (missed pneumonia) is far more dangerous than a False Positive. Both models are therefore evaluated with heavy focus on **Recall (Sensitivity)** rather than raw accuracy alone.

---

## 🗂️ Repository Structure

```
pneumonia-detection/
│
├── notebooks/
│   ├── custom_cnn.ipynb                      # Custom CNN — EDA, training, evaluation, inference demo
│   └── transfer_learning_densenet.ipynb      # DenseNet121 — two-phase training & threshold tuning
│
├── app/
│   └── app.py                                # LungLens AI — Streamlit deployment app
│
├── models/
│   ├── pneumonia_detection_model.keras       # Saved Custom CNN weights
│   └── transfer_learning_model.keras         # Saved DenseNet121 weights
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle, Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Total | Normal | Pneumonia |
|-------|-------|--------|-----------|
| Train | 5,216 | 1,341 | 3,875 |
| Validation | 16 | — | — |
| Test | 624 | 234 | 390 |

The dataset has a **~3:1 class imbalance** (Pneumonia vs Normal). Each model handles this differently — see model sections below.

---

## 🧠 Model 1 — Custom CNN (Built from Scratch)

A Sequential CNN trained entirely from random initialization on **grayscale** X-rays.

### Why Grayscale?
X-rays carry no color information. Loading in grayscale reduces memory usage, speeds up training, and avoids feeding meaningless color channels into the network.

### Architecture

```
Input (224×224×1)
→ Rescaling (1/255)
→ RandomFlip | RandomRotation(0.1) | RandomZoom(0.1)
→ Conv2D(32)  + MaxPooling2D   # edges, rib curvature
→ Conv2D(64)  + MaxPooling2D   # basic lung structures
→ Conv2D(128) + MaxPooling2D   # complex textures
→ Conv2D(128) + MaxPooling2D   # high-level opacity patterns
→ Flatten → Dense(512) → Dropout(0.5)
→ Dense(1, activation='sigmoid')
```

**Total Parameters: 13,086,337 (~49.92 MB)**

### Training Strategy
- **Optimizer**: Adam (`lr=1e-4`, `clipnorm=1.0`)
- **Loss**: Binary Crossentropy
- **GPU**: Dual NVIDIA T4 via `MirroredStrategy` (batch size = 64)
- **Callbacks**: `EarlyStopping(patience=5)` + `ReduceLROnPlateau(factor=0.3, patience=3)`
- **Class imbalance**: Handled via in-model data augmentation (flip, rotation, zoom)
- Trained for **11 epochs** before early stopping

### Results on Test Set (624 images)

| Metric | Normal | Pneumonia | Overall |
|--------|--------|-----------|---------|
| Precision | 0.84 | **0.94** | — |
| Recall | 0.91 | 0.89 | — |
| F1-Score | 0.87 | 0.92 | — |
| **Accuracy** | — | — | **90.06%** |

---

## 🧠 Model 2 — Transfer Learning (DenseNet121)

Fine-tuned **DenseNet121** pretrained on ImageNet, adapted for chest X-ray classification using the Keras Functional API.

### Why RGB for a Grayscale X-Ray?
DenseNet121 was pretrained on ImageNet (3 channels). Loading X-rays in RGB satisfies the backbone's tensor shape requirements without any architectural modification.

### Architecture

```
Input (224×224×3)
→ Rescaling + RandomFlip + RandomRotation(0.1) + RandomZoom(0.1)
→ DenseNet121 backbone (pretrained on ImageNet, frozen in Phase 1)
→ GlobalAveragePooling2D
→ BatchNormalization
→ Dense(256, relu) → Dropout(0.4)
→ Dense(1, activation='sigmoid')
```

**Total Parameters: 7,304,257** — Trainable in Phase 1: 264,705 (custom head only)

### Two-Phase Training Strategy

**Phase 1 — Head Training (frozen backbone)**
- DenseNet121 base fully frozen (`trainable=False`, `training=False` keeps BN layers frozen)
- Only the custom classification head trained
- `lr = 1e-3`, monitored on `val_auc` — more stable than `val_loss` for imbalanced data
- Stopped at epoch 6 via early stopping

**Phase 2 — Fine-Tuning (partial unfreeze)**
- Last 50 layers of DenseNet121 unfrozen, all earlier layers remain frozen
- Lower `lr = 1e-5` to avoid destroying pretrained ImageNet weights
- `clipnorm=1.0` for gradient stability
- Stopped at epoch 9, best weights from epoch 4 restored

**Class imbalance** handled via computed class weights applied in both phases:
```
{ Normal: 1.94,  Pneumonia: 0.67 }
```

### Threshold Optimization
Rather than using the default 0.5 cutoff, thresholds from 0.20 → 0.80 were swept to maximize **Macro F1**:

| Threshold | Normal Recall | Pneumonia Recall | Macro F1 |
|-----------|--------------|------------------|----------|
| 0.30 | 0.7350 | 0.9487 | 0.8539 |
| 0.40 | 0.7991 | 0.9282 | 0.8695 |
| 0.50 | 0.8333 | 0.9154 | 0.8763 |
| 0.55 | 0.8590 | 0.9077 | 0.8824 |
| **0.57** ✅ | **0.87** | **0.91** | **0.8877** |

### Results on Test Set (624 images, threshold = 0.57)

| Metric | Normal | Pneumonia | Overall |
|--------|--------|-----------|---------|
| Precision | 0.85 | **0.92** | — |
| Recall | 0.87 | 0.91 | — |
| F1-Score | 0.86 | 0.91 | — |
| **Accuracy** | — | — | **89%** |

---

## 📊 Model Comparison

| | Custom CNN | DenseNet121 (TL) |
|---|---|---|
| Input format | Grayscale (1 ch) | RGB (3 ch) |
| Parameters | 13M | 7.3M |
| Training approach | From scratch | Two-phase fine-tuning |
| Imbalance handling | Data augmentation | Class weights |
| Decision threshold | 0.50 | 0.57 (optimized) |
| Test Accuracy | **90.06%** | 89% |
| Pneumonia Recall | 89.49% | **91%** |
| Pneumonia Precision | **94%** | 92% |
| Pneumonia F1 | **0.92** | 0.91 |

> The Custom CNN achieves slightly higher overall accuracy. DenseNet121 edges ahead on Pneumonia Recall — making it the safer choice in clinical contexts where missing a positive case carries the greatest risk.

---

## 🖥️ LungLens AI — Streamlit App

The web app lets users upload a chest X-ray and get an instant AI diagnosis with a confidence score.

### Features
- **Model selector** — switch between Custom CNN and DenseNet121 at runtime
- **Auto preprocessing** — grayscale vs RGB conversion handled automatically per model
- **Confidence score** — raw sigmoid probability + animated visual confidence bar
- **Precaution panel** — when Pneumonia is detected, displays 6 actionable precautions (seek medical help, rest, hydration, isolation, breathing monitoring, avoid irritants)
- **Correct thresholds applied** — CNN uses 0.50, DenseNet uses optimized 0.57
- **Medical disclaimer** — clearly marks the tool as research/educational only, not a clinical substitute
- **Models served from HF Hub** — loaded at runtime via `hf_hub_download` from `Tejas-ML/pneumonia-detection-models`

### Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/pneumonia-detection.git
cd pneumonia-detection
pip install -r requirements.txt
streamlit run app/app.py
```

---

## ⚙️ Requirements

```
tensorflow>=2.12
streamlit
numpy
matplotlib
seaborn
scikit-learn
Pillow
huggingface_hub
```

Install all:

```bash
pip install -r requirements.txt
```

---

## 🚀 Deployment

Both model `.keras` files are hosted on Hugging Face Hub and loaded dynamically at runtime:

- **Model Hub repo**: `Tejas-ML/pneumonia-detection-models`
- **Live Space**: [huggingface.co/spaces/Tejas-ML/pneumonia-detection-app](https://huggingface.co/spaces/Tejas-ML/pneumonia-detection-app)

> Model weight files are excluded from this GitHub repo due to size. They are stored and served via Hugging Face Hub.

---

## 👤 Author

**Tejas** — [Hugging Face](https://huggingface.co/Tejas-ML) · [Kaggle Notebook](https://www.kaggle.com/code/tejas5112/pnuemonia-detection)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset by Paul Mooney
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) — Huang et al., 2017
- Kaggle for dual NVIDIA T4 GPU compute
- Hugging Face for free model hosting and Spaces deployment
