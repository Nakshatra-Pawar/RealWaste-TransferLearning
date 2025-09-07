# RealWaste-TransferLearning

Lightweight Keras/TensorFlow project that classifies waste images into nine categories using transfer learning (VGG16, ResNet, EfficientNet). Trains a small classification head on frozen backbones with standard image augmentation and reports Precision/Recall/F1/AUC.

Repo Structure
RealWaste-TransferLearning/
├─ Data/                     # (not tracked in git; see .gitignore)
│  ├─ RealWaste/             # optional master copy of the dataset
│  ├─ RealWaste_train/       # train split: class folders with images
│  └─ RealWaste_test/        # test  split: class folders with images
├─ Notebook/
│  └─ Pawar_Nakshatra_Final_Project.ipynb
├─ README.md
├─ requirements.txt
└─ .gitignore


The dataset is large and should not be committed. Keep it under Data/.

Setup

Create an environment (Python 3.10+ recommended) and install dependencies:

pip install -r requirements.txt


requirements.txt (suggested)

tensorflow>=2.13
keras>=2.13
opencv-python
numpy
pandas
matplotlib
scikit-learn
tqdm


Place your dataset as:

Data/RealWaste_train/<class>/*.jpg|png
Data/RealWaste_test/<class>/*.jpg|png


(If you only have Data/RealWaste/, the notebook can create an 80/20 split.)

Open the notebook:

Notebook/Pawar_Nakshatra_Final_Project.ipynb


In the first config cell, set:

DATA_DIR = "../Data"      # adjust if needed
TRAIN_DIR = f"{DATA_DIR}/RealWaste_train"
TEST_DIR  = f"{DATA_DIR}/RealWaste_test"

What the Notebook Does

Preprocessing: resize/center-pad; label encoding

Augmentation: random flip/rotate/zoom/translate/contrast

Backbones: VGG16, ResNet50/101, EfficientNetB0 (frozen feature extractor)

Head: BatchNorm → Dropout(0.2) → Dense(ReLU) → Dense(softmax) with L2

Training: Adam, categorical cross-entropy, early stopping & model checkpoint

Evaluation: micro/macro Precision, Recall, F1, AUC; confusion matrix and curves

Quick Start (example)

In the notebook, pick a backbone and train:

BACKBONE = "ResNet50"    # options: "VGG16", "ResNet50", "ResNet101", "EfficientNetB0"
IMG_SIZE = (224, 224)
BATCH    = 5
EPOCHS   = 50            # early stopping will cap this

# run the provided training cell; metrics & plots will be saved under ./artifacts/

Tips

Start with EfficientNetB0 for speed; try ResNet50 for stronger baseline.

If GPU memory is tight, reduce batch size or image size.

Add your final metric table and sample predictions as screenshots to the repo (but keep raw images out of git).

.gitignore (recommended)
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

License

This project is licensed under the MIT License. See LICENSE for details.