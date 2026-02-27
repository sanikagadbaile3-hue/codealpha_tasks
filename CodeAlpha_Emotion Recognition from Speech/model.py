import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        # Input shape expected: (Batch, 1, Features)
        # Assuming features = 40 (MFCC) + 12 (Chroma) + 128 (Mel) = 180 total
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(64, 128, 5, 1, 2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.dropout2 = nn.Dropout(0.3)
        
        self.conv3 = nn.Conv1d(128, 256, 5, 1, 2)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Calculate flattened size
        # Starting with 180
        # After pool1 (180 / 4) = 45
        # After pool2 (45 / 4) = 11
        self.fc1 = nn.Linear(256 * 11, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, features) -> add channel dimension
        x = x.unsqueeze(1) 
        
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class EmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Features) -> Treat as (Batch, 1, Features) or (Batch, Seq, Feature_dim)
        # For simplicity, if we pass (Batch, 180), we'll treat it as (Batch, 1, 180)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    # Test models with dummy input
    num_emotions = 8
    
    print("Testing CNN...")
    cnn_model = EmotionCNN(num_emotions)
    dummy_input = torch.randn(1, 180)
    cnn_output = cnn_model(dummy_input)
    print(f"CNN output shape: {cnn_output.shape}")

    print("\nTesting LSTM...")
    lstm_model = EmotionLSTM(input_size=180, hidden_size=128, num_layers=2, num_classes=num_emotions)
    lstm_output = lstm_model(dummy_input)
    print(f"LSTM output shape: {lstm_output.shape}")
