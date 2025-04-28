# Facial Expression Recognition with Deep Learning

## Overview

This project aims to build a deep learning-based system for classifying facial expressions using convolutional neural networks (CNNs) in PyTorch. The final product will allow users to upload an image of a face and receive a predicted facial expression. The focus is on creating a proof of concept (PoC) within **3 weeks** by two software engineers.

## Goals

- Build and train a CNN to classify facial expressions with high accuracy  
- Use the [FERPlus dataset](https://github.com/microsoft/FERPlus) (potentially extended with other datasets — see planning docs for details)  
- Create a minimal user interface for image upload and real-time inference  
- Prepare a demo-ready prototype for presentations  

## Emotion Classes

We support the 7 FER-2013 categories:

- **Anger**
- **Disgust**
- **Fear**
- **Happiness**
- **Sadness**
- **Surprise**
- **Neutral**

## Directory Structure

See the sitemap below. Key areas include:

- `data/`: Preprocessing datasets into training-ready format with a standard schema across datasets  
- `models/`: PyTorch CNN architectures  
- `train/`: Training pipeline and config management  
- `eval/`: Evaluation tools and confusion matrix  
- `demo/`: Minimal UI using Flask or Streamlit  

```
facial-expression-recognition/
│
├── data/              # Stores datasets and preprocessing scripts
│   ├── raw/           # Original datasets (e.g., FERPlus)
│   ├── processed/     # Preprocessed and resized datasets
│   └── preprocess.py  # Preprocessing pipeline (resize, normalize, split)
│
├── models/            # Model architectures
│   └── cnn.py         # CNN model for expression recognition
│
├── train/             # Training scripts and configs
│   ├── train.py       # Training loop
│   ├── config.yaml    # Model and training configuration
│   └── utils.py       # Metrics, checkpointing, etc.
│
├── eval/              # Evaluation scripts
│   └── evaluate.py    # Model evaluation and confusion matrix
│
├── demo/              # Simple UI for uploading and predicting images
│   ├── app.py         # Web or CLI interface (Flask or Streamlit)
│   └── utils.py       # Image loading, preprocessing, and prediction
│
├── checkpoints/       # Saved models
├── logs/              # Training logs (for TensorBoard or custom)
├── requirements.txt   # Python dependencies
├── README.md          # Project overview and instructions
└── .gitignore
```

## Authors

Sebastian Lueders and Sid Dutta
