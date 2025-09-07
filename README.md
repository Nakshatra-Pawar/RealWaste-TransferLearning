# RealWaste — Transfer Learning for 9-Class Waste Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

A compact Keras/TensorFlow project that classifies waste images into **nine** categories using transfer learning (VGG16, ResNet, EfficientNet). The workflow is notebook-driven and includes preprocessing, augmentation, model training, and evaluation with Precision/Recall/F1/AUC.

---

## Table of Contents

- [Features](#features)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
  - [Requirements](#requirements)  
  - [Install](#install)  
- [Data Layout](#data-layout)  
- [Usage](#usage)  
  - [Run the Notebook](#run-the-notebook)  
  - [Key Config](#key-config)  
  - [Training](#training)  
  - [Evaluation & Artifacts](#evaluation--artifacts)  
- [Tips & Troubleshooting](#tips--troubleshooting)  
- [.gitignore (recommended)](#gitignore-recommended)  
- [`requirements.txt` (suggested)](#requirementstxt-suggested)  
- [Contributing](#contributing)  
- [License](#license)

---

## Features

- Transfer learning with popular image backbones: **VGG16**, **ResNet50/101**, **EfficientNetB0**  
- Reproducible, notebook-first pipeline (easy to inspect and modify)  
- Standard image **augmentation** (flip/rotate/zoom/translate/contrast)  
- Clear **metrics**: micro/macro Precision, Recall, F1, AUC, confusion matrix  
- Lightweight and GPU-friendly; configurable batch size and image size

---

## Project Structure

```
RealWaste-TransferLearning/
├─ Data/                     # (not tracked in git; keep datasets here)
│  ├─ RealWaste/             # optional master copy of the dataset
│  ├─ RealWaste_train/       # train split: class folders with images
│  └─ RealWaste_test/        # test  split: class folders with images
├─ Notebook/
│  └─ Pawar_Nakshatra_Final_Project.ipynb
├─ README.md
├─ requirements.txt
└─ .gitignore
```

> ⚠️ The dataset is large and **should not be committed**. Keep it under `Data/`.

---

## Getting Started

### Requirements
- **Python 3.10+**
- TensorFlow 2.x / Keras
- See the suggested `requirements.txt` below.

### Install

Create and activate a virtual environment (optional but recommended), then:

```bash
pip install -r requirements.txt
```

---

## Data Layout

Place your images in class-named folders for both splits:

```
Data/RealWaste_train/<class_name>/*.jpg|png
Data/RealWaste_test/<class_name>/*.jpg|png
```

If you only have a single folder `Data/RealWaste/`, you can create an 80/20 split inside the notebook (a helper cell is provided there).

---

## Usage

### Run the Notebook

Open:

```
Notebook/Pawar_Nakshatra_Final_Project.ipynb
```

In the first configuration cell, set:

```python
DATA_DIR = "../Data"             # adjust if needed
TRAIN_DIR = f"{DATA_DIR}/RealWaste_train"
TEST_DIR  = f"{DATA_DIR}/RealWaste_test"
```

### Key Config

Typical variables you can tweak in the notebook:

```python
BACKBONE = "ResNet50"            # "VGG16", "ResNet50", "ResNet101", "EfficientNetB0"
IMG_SIZE = (224, 224)            # (height, width)
BATCH    = 5
EPOCHS   = 50                    # EarlyStopping will cap this if val loss plateaus
SEED     = 42
```

### Training

The notebook does the following:

- **Preprocessing:** resize/center-pad; label encoding  
- **Augmentation:** random flip/rotate/zoom/translate/contrast  
- **Feature extractor:** chosen backbone **frozen** initially  
- **Classification head:** BatchNorm → Dropout(0.2) → Dense(ReLU) → Dense(softmax) + L2  
- **Optimization:** Adam + categorical cross-entropy  
- **Callbacks:** EarlyStopping, ModelCheckpoint

Just execute the training cell—loss/accuracy curves and best weights will be saved into `./artifacts/` (path can be changed).

### Evaluation & Artifacts

The notebook computes:

- Micro/Macro **Precision, Recall, F1**, and **AUC**  
- **Confusion matrix** and sample predictions  
- Curves: training/validation **loss** and **accuracy**

You can export figures to `./artifacts/` for inclusion in reports or the repo (avoid committing raw datasets).

---

## Tips & Troubleshooting

- Start with **EfficientNetB0** for faster runs; use **ResNet50** for a stronger baseline.  
- If you hit OOM (out-of-memory), **lower** `BATCH` or reduce `IMG_SIZE`.  
- Set a **fixed `SEED`** and enable deterministic ops (if needed) for strict reproducibility.  
- Keep the **Data/** folder **out of git**. Consider symlinks if you store data elsewhere.  
- GPU available? Verify with:

  ```python
  import tensorflow as tf
  tf.config.list_physical_devices('GPU')
  ```

---

## .gitignore (recommended)

```gitignore
# data & artifacts
Data/
artifacts/
*.h5
*.ckpt*
*.tfrecord
*.npz
*.npy

# notebooks & caches
.ipynb_checkpoints/
__pycache__/
*.DS_Store

# OS/editor files
Thumbs.db
.vscode/
```

---

## `requirements.txt` (suggested)

```
tensorflow>=2.13
keras>=2.13
opencv-python
numpy
pandas
matplotlib
scikit-learn
tqdm
```

---

## Contributing

Small improvements are welcome—typo fixes, better docs, or extra backbone configs.  
Open an issue or a PR describing the change.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
