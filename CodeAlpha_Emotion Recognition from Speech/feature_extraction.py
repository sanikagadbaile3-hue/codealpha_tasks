import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract MFCC, Chroma, and Mel features from an audio file.
    """
    try:
        # Load audio file
        # 'y' is the audio time series, 'sr' is the sampling rate
        y, sr = librosa.load(file_path, res_type='kaiser_fast')
        
        # Result sequence
        result = np.array([])
        
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        
        # Chroma
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma))
        
        # Mel Spectrogram
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        result = np.hstack((result, mel))
        
        return result
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Test with a dummy path or a real file if available
    print("Feature extraction module loaded.")
