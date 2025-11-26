import librosa
import pandas as pd

pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
feature_names = ['rms', 'zcr', 'spectral_centroid', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_C', 'chroma_C#', 'chroma_D', 'chroma_D#', 'chroma_E', 'chroma_F', 'chroma_F#', 'chroma_G', 'chroma_G#', 'chroma_A', 'chroma_A#', 'chroma_B', 'bpm']

def extract_features(y, sr):
    """
    Extract various audio features from an audio signal.
    
    Parameters:
    y (np.ndarray): Audio time series
    sr (int): Sampling rate
    
    Returns:
    dict: Dictionary of extracted features
    """
    features = {}
    
    # Time-domain features
    features['rms'] = librosa.feature.rms(y=y).mean()
    features['zcr'] = librosa.feature.zero_crossing_rate(y).mean()
    
    # Frequency-domain features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = spectral_centroid.mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff'] = spectral_rolloff.mean()
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i+1}'] = mfcc.mean()
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, pitch_class in enumerate(pitch_classes):
        features[f'chroma_{pitch_class}'] = chroma[i].mean()
    
    # BPM
    bpm, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    features['bpm'] = bpm

    return features


if __name__ == "__main__":

    song_features = {}
    for i in feature_names:
        song_features[i] = []

    # Load existing CSV
    df = pd.read_csv("data.csv")

    songs = df["id"]
    numOfSongs = len(songs)
    count = 0

    for song in songs:
        
        # Load File
        y, sr = librosa.load(f"./data/{song}.mp3")

        # Extract features from the audio file
        audio_features = extract_features(y, sr)

        for key in audio_features.keys():
            song_features[key].append(audio_features[key])
        
        print(f"  {i / numOfSongs * 100:.2f}% is done!\r", end="")
        count += 1

    # add features as a column to csv file
    for feature in feature_names:
        df[feature] = song_features[feature]
    

    # Save back to CSV
    df.to_csv("data.csv", index=False)