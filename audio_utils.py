import os
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from os.path import join

pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
feature_names = ['rms', 'zcr', 'spectral_centroid', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_C', 'chroma_C#', 'chroma_D', 'chroma_D#', 'chroma_E', 'chroma_F', 'chroma_F#', 'chroma_G', 'chroma_G#', 'chroma_A', 'chroma_A#', 'chroma_B', 'bpm']

# ================= CONFIG =================
DATASET_DIR = "data"
MEL_OUTPUT_DIR = "mel_spectrograms_segmented"

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
# ==========================================

class AudioUtils:

    def audio_to_segments(self, audio_path, class_name, segment_seconds, num_segments):
        y, sr = librosa.load(audio_path, sr=SR)
        samples_per_segment = sr * segment_seconds

        # Safety check (should already be 30s)
        expected_length = samples_per_segment * num_segments
        if len(y) < expected_length:
            return 0, 0 # skip corrupted/short files

        segments = list()
        for i in range(num_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            segment = y[start:end]
            segments.append(segment)
            
        return segments, sr
    
    def create_mel_spectrogram(self, name):
        mel = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        out_dir = Path(MEL_OUTPUT_DIR) / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{name}.npy"
        np.save(out_path, mel_db)

    def load_file(self, file_path):
        return librosa.load(file_path)


    def extract_features(self, y, sr):
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
        features['bpm'] = bpm[0]

        return features

    def write_features_to_dataset(self):

        song_features = {}
        for i in feature_names:
            song_features[i] = []

        # Load existing CSV
        df = pd.read_csv("data.csv")

        songs = df["id"]
        numOfSongs = len(songs)
        count = 1

        for song in songs:
            
            # Load File
            y, sr = load_file(join("data", f"{song}.mp3"))

            # Extract features from the audio file
            audio_features = extract_features(y, sr)

            for key in audio_features.keys():
                song_features[key].append(audio_features[key])
            
            print(f"  {count / numOfSongs * 100:.2f}% is done!\r", end="")
            count += 1

        # add features as a column to csv file
        for feature in feature_names:
            df[feature] = song_features[feature]
        
        # Save back to CSV
        df.to_csv("data.csv", index=False)
    
    def write_segmented_features_to_dataset(self):

        song_features = {"class": []}
        for i in feature_names:
            song_features[i] = []



        dataset_path = Path(DATASET_DIR)

        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue

            print(f"Processing class: {class_dir.name}")

            for audio_file in tqdm(list(class_dir.glob("*.mp3"))):
                segments, sr = self.audio_to_segments(audio_file, class_dir.name, 6, 5)
                if segments == 0:
                    continue
                for i in range(len(segments)):
                    audio_features = self.extract_features(segments[i], sr)

                    song_features["class"].append(class_dir.name)
                    for key in audio_features.keys():
                        song_features[key].append(audio_features[key])


        df = pd.DataFrame(song_features)

        # Export to CSV
        df.to_csv("data_segmented.csv", index=False)


    def audio_to_mel_segments(audio_path, class_name):
        y, sr = librosa.load(audio_path, sr=SR)

        # Safety check (should already be 30s)
        expected_length = SAMPLES_PER_SEGMENT * NUM_SEGMENTS
        if len(y) < expected_length:
            return  # skip corrupted/short files

        for i in range(NUM_SEGMENTS):
            start = i * SAMPLES_PER_SEGMENT
            end = start + SAMPLES_PER_SEGMENT
            segment = y[start:end]

            mel = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS
            )

            mel_db = librosa.power_to_db(mel, ref=np.max)

            out_dir = Path(OUTPUT_DIR) / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{audio_path.stem}_seg{i}.npy"
            np.save(out_path, mel_db)


    def process_dataset():
        dataset_path = Path(DATASET_DIR)

        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue

            print(f"Processing class: {class_dir.name}")

            for audio_file in tqdm(list(class_dir.glob("*.mp3"))):
                audio_to_mel_segments(audio_file, class_dir.name)



if __name__ == "__main__":
    util = AudioUtils()
    util.write_segmented_features_to_dataset()
