# Iris Flower Classification with PyTorch

A complete deep learning example using a multi-layer neural network to classify Iris flowers using the famous Iris dataset.

This repository contains a Jupyter/Colab notebook demonstrating modern PyTorch best practices on a small but classic classification problem.

## Features

- Data loading & preparation using `scikit-learn`
- Train / Validation / Test split (~70% / 15% / 15%)
- Feature standardization with `StandardScaler`
- Mini-batch training using `DataLoader`
- Neural network with:
  - Batch Normalization
  - Dropout (30%)
  - Two hidden layers (24 → 16 neurons)
  - ReLU activations
- Adam optimizer + L2 weight decay
- Learning rate scheduling (`ReduceLROnPlateau`)
- Early stopping based on validation loss
- Loss & accuracy curve visualization
- Final evaluation with:
  - Accuracy
  - Classification report (precision/recall/F1)
  - Confusion matrix heatmap
  - Misclassified sample inspection
  - 2D decision boundary visualization (petal length vs petal width)

## Results (example run)

- **Final Test Accuracy**: ~95–100% (typically 22–23/23 correct)
- Early stopping usually triggers around epoch 20–35
- One common borderline misclassification: sample [6.7, 3.0, 5.0, 1.7] (often predicted as virginica instead of versicolor)


## Requirements

```bash
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib seaborn numpy pandas
