# Emotion Recognition from Speech

This project implements a deep learning model to recognize human emotions (e.g., happy, angry, sad) from speech audio using MFCC features and a Convolutional Neural Network (CNN) built with PyTorch.

## Features
- **Feature Extraction**: Extracts MFCC, Chroma, and Mel features from audio files using `librosa`.
- **Deep Learning Model**: A 1D CNN architecture designed for audio feature classification.
- **Dataset Support**: Prepared for the RAVDESS dataset structure.

## Requirements
- Python 3.14 (Supports PyTorch)
- Libraries: `torch`, `librosa`, `scikit-learn`, `pandas`, `numpy`, `joblib`

## How to Use

### 1. Data Preparation
Place your RAVDESS dataset in a folder named `data/` in the project root. The structure should look like `data/Actor_01/03-01-01-...wav`.

### 2. Training
Run the training script to extract features and train the model:
```bash
python train.py [cnn|lstm]
```
Example: `python train.py lstm`

### 3. Prediction
...
```bash
python predict.py path/to/your/audio.wav
```

### 4. Desktop GUI (Recommended)
Launch the interactive desktop app to upload audio or record voice:
```bash
python gui.py
```

### 5. Visualization
...
```bash
python visualize.py path/to/your/audio.wav
```

### 6. Real-time Prediction (CLI)
...
```bash
python record_and_predict.py
```

## Project Structure
- `feature_extraction.py`: Functions to process audio files.
- `data_loader.py`: Handles dataset walking and label extraction.
- `model.py`: PyTorch CNN and LSTM architectures.
- `train.py`: Main training pipeline with history plotting.
- `predict.py`: Inference script for new audio files.
- `visualize.py`: Script to plot audio features.
- `record_and_predict.py`: Script for microphone-based prediction.
- `gui.py`: Desktop application for the project.
