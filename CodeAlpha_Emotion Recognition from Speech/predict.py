import torch
import numpy as np
import librosa
import joblib
import json
from feature_extraction import extract_features
from model import EmotionCNN, EmotionLSTM

def predict(audio_file):
    # Load model architecture
    # Assuming we have 180 features and we know the number of classes from training
    # For this demo, we'll load the encoder to get number of classes
    try:
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        num_classes = config['num_classes']
        model_type = config['model_type']
        input_size = config['input_size']
        
    except FileNotFoundError:
        print("Model configuration or artifacts not found. Please train the model first.")
        return

    if model_type == 'lstm':
        model = EmotionLSTM(input_size=input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
    else:
        model = EmotionCNN(num_classes)
    
    # Load weights
    try:
        model.load_state_dict(torch.load('emotion_model.pth'))
    except FileNotFoundError:
        print("Model weights not found. Please train the model first.")
        return
        
    model.eval()

    # Extract features
    features = extract_features(audio_file)
    if features is None:
        return
    
    # Preprocess
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_t = torch.tensor(features_scaled, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        outputs = model(features_t)
        _, predicted = torch.max(outputs, 1)
        prediction = encoder.inverse_transform([predicted.item()])[0]
        
    return prediction

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file>")
    else:
        result = predict(sys.argv[1])
        if result:
            print(f"Predicted Emotion: {result}")
