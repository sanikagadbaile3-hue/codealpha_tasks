import os
import pandas as pd
import numpy as np
from feature_extraction import extract_features

# Emotion mapping for RAVDESS
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_ravdess_data(data_path):
    """
    Walks through the RAVDESS dataset directory and extracts features.
    Expected structure: data_path/Actor_01/03-01-01-...wav
    """
    features = []
    labels = []
    
    if not os.path.exists(data_path):
        print(f"Directory {data_path} does not exist.")
        return None, None

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                # Extract label from filename
                parts = file.split('-')
                if len(parts) >= 3:
                    emotion_id = parts[2]
                    emotion = EMOTIONS.get(emotion_id)
                    
                    if emotion:
                        file_path = os.path.join(root, file)
                        feature = extract_features(file_path)
                        if feature is not None:
                            features.append(feature)
                            labels.append(emotion)
    
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    print("Data loader module loaded.")
