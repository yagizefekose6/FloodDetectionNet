# Flood Detection using U-Net with Attention Mechanism

This project implements a deep learning solution for flood detection using satellite imagery. The model uses a U-Net architecture enhanced with attention mechanisms to accurately segment and identify flooded areas in images.

## Features

- U-Net architecture with attention gates for improved segmentation
- Data augmentation pipeline for robust model training
- Binary segmentation for flood detection
- Comprehensive evaluation metrics including IoU and Dice coefficient
- Support for custom dataset integration

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- scikit-learn
- Matplotlib
```

## Dataset Structure

The project expects the following dataset structure:
```
Dataset/
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
└── labels/
    ├── 1.png
    ├── 2.png
    └── ...
```

## Model Architecture

The model implements a U-Net architecture with the following key components:
- Encoder path with multiple convolutional blocks
- Attention gates for feature refinement
- Decoder path with skip connections
- Binary segmentation output

## Training

To train the model:

1. Place your dataset in the appropriate directories
2. Adjust the hyperparameters in the notebook if needed
3. Run the training cells in the Jupyter notebook

## Evaluation

The model is evaluated using:
- IoU (Intersection over Union)
- Dice coefficient
- Accuracy metrics
- Confusion matrix

## Results

The model achieves high accuracy in flood detection with:
- Precise segmentation of flooded areas
- Robust performance across different image conditions
- Efficient processing of satellite imagery

