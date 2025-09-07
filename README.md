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
- [Training](#training)    
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
├─ Data/                     
│  ├─ RealWaste/             
│  ├─ RealWaste_train/       
│  └─ RealWaste_test/        
├─ Notebook/
│  └─ Pawar_Nakshatra_Final_Project.ipynb
├─ README.md
├─ requirements.txt
└─ .gitignore
```

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

### Training

The notebook does the following:

- **Preprocessing:** resize/center-pad; label encoding  
- **Augmentation:** random flip/rotate/zoom/translate/contrast  
- **Feature extractor:** chosen backbone **frozen** initially  
- **Classification head:** BatchNorm → Dropout(0.2) → Dense(ReLU) → Dense(softmax) + L2  
- **Optimization:** Adam + categorical cross-entropy  
- **Callbacks:** EarlyStopping, ModelCheckpoint

Just execute the training cell—loss/accuracy curves and best weights will be saved into `./artifacts/` (path can be changed).

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
