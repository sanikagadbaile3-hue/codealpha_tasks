import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_loader import load_ravdess_data
from model import EmotionCNN, EmotionLSTM
import joblib
import matplotlib.pyplot as plt
import numpy as np

def prepare_data(features, labels):
    """
    Scale features and encode labels.
    """
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    features_scaled = scaler.fit_transform(features)
    labels_encoded = encoder.fit_transform(labels)
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    
    return features_scaled, labels_encoded, encoder.classes_

def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")
    # plt.show() # Commented out to avoid blocking in non-interactive environments

def train(model_type='cnn'):
    data_path = 'data' 
    
    print(f"Attempting to load data... (Model: {model_type.upper()})")
    features, labels = load_ravdess_data(data_path)
    
    if features is None or len(features) == 0:
        print(f"No data found in {data_path}. Please place RAVDESS data in 'data/' folder.")
        print("Falling back to dummy data for demonstration...")
        features = np.random.randn(100, 180)
        labels = ['happy', 'sad', 'angry', 'neutral', 'calm', 'fearful', 'disgust', 'surprised'] * 12 + ['happy', 'sad', 'angry', 'neutral']
    
    features_scaled, labels_encoded, classes = prepare_data(features, labels)
    num_classes = len(classes)
    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_encoded, test_size=0.2, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if model_type.lower() == 'lstm':
        model = EmotionLSTM(input_size=180, hidden_size=128, num_layers=2, num_classes=num_classes)
    else:
        model = EmotionCNN(num_classes)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 50
    history = {'loss': [], 'accuracy': []}
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%")
            
    config = {
        'model_type': model_type.lower(),
        'num_classes': num_classes,
        'input_size': 180,
        'classes': [str(c) for c in classes]
    }
    with open('model_config.json', 'w') as f:
        json.dump(config, f)
        
    torch.save(model.state_dict(), 'emotion_model.pth')
    print("Model and config saved to emotion_model.pth and model_config.json")
    
    plot_history(history)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    import sys
    m_type = 'cnn'
    if len(sys.argv) > 1:
        m_type = sys.argv[1]
    train(m_type)
