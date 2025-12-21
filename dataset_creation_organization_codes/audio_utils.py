import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from os.path import join

# ================= CONSTANTS =================
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

DATASET_DIR = "data"
MEL_OUTPUT_DIR = "mel_spectrograms_segmented"

SEGMENT_SECONDS = 6
NUM_SEGMENTS = 5
SAMPLES_PER_SEGMENT = SR * SEGMENT_SECONDS

pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']

feature_names = [
    'rms', 'zcr', 'spectral_centroid', 'spectral_rolloff',
    *[f'mfcc_{i}' for i in range(1, 14)],
    *[f'chroma_{p}' for p in pitch_classes],
    'bpm'
]
# ============================================


# ================= AUDIO UTILS =================

def load_audio(file_path):
    return librosa.load(file_path, sr=SR)


def audio_to_segments(audio_path, segment_seconds=6, num_segments=5):
    y, sr = load_audio(audio_path)
    samples_per_segment = sr * segment_seconds

    expected_length = samples_per_segment * num_segments
    if len(y) < expected_length:
        return None, None

    segments = []
    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segments.append(y[start:end])

    return segments, sr


# ================= FEATURE EXTRACTION =================

def extract_features(y, sr):
    features = {}

    # Time domain
    features['rms'] = librosa.feature.rms(y=y).mean()
    features['zcr'] = librosa.feature.zero_crossing_rate(y).mean()

    # Frequency domain
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        features[f'mfcc_{i+1}'] = mfcc.mean()

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, pitch in enumerate(pitch_classes):
        features[f'chroma_{pitch}'] = chroma[i].mean()

    # BPM
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['bpm'] = float(bpm)

    return features


# ================= MEL SPECTROGRAM =================

def save_mel_segment(segment, sr, out_path):
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    np.save(out_path, mel_db)


def audio_to_mel_segments(audio_path, class_name):
    y, sr = load_audio(audio_path)

    if len(y) < SAMPLES_PER_SEGMENT * NUM_SEGMENTS:
        return

    out_dir = Path(MEL_OUTPUT_DIR) / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(NUM_SEGMENTS):
        start = i * SAMPLES_PER_SEGMENT
        end = start + SAMPLES_PER_SEGMENT
        segment = y[start:end]

        out_path = out_dir / f"{audio_path.stem}_seg{i}.npy"
        save_mel_segment(segment, sr, out_path)


# ================= DATASET WRITERS =================

def write_segmented_features_to_dataset():
    song_features = {"class": []}
    for f in feature_names:
        song_features[f] = []

    dataset_path = Path(DATASET_DIR)

    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue

        print(f"Processing class: {class_dir.name}")

        for audio_file in tqdm(list(class_dir.glob("*.mp3"))):
            segments, sr = audio_to_segments(audio_file)

            if segments is None:
                continue

            for segment in segments:
                features = extract_features(segment, sr)
                song_features["class"].append(class_dir.name)

                for key in feature_names:
                    song_features[key].append(features[key])

    df = pd.DataFrame(song_features)
    df.to_csv("data_segmented.csv", index=False)


def process_dataset_to_mel():
    dataset_path = Path(DATASET_DIR)

    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue

        print(f"Processing class: {class_dir.name}")

        for audio_file in tqdm(list(class_dir.glob("*.mp3"))):
            audio_to_mel_segments(audio_file, class_dir.name)


