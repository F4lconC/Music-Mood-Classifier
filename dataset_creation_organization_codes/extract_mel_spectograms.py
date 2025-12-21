from os import makedirs
from os.path import join
import re

import cv2
import librosa
import numpy as np
import pandas as pd


def get_mood(id):
    mood = re.match(r"[a-zA-Z]+", id).group()
    return mood


def make_mood_dirs(moods):
    for mood in moods:
        makedirs(join(save_dir, mood), exist_ok=True)


df = pd.read_csv("data.csv")
songs = df["id"]

numOfSongs = len(songs)
count = 1

save_dir = "mel_spectrograms"
moods = ["angry", "happy", "relaxed", "sad"]

makedirs(save_dir, exist_ok=True)
make_mood_dirs(moods)

for song in songs:
    
    # Load the song file
    y, sr = librosa.load(join("data", f"{song}.mp3"))

    # Compute mel-spectogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)

    # min-max normalize
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())

    # Create an 128x128 image
    img = (cv2.resize(S_norm, (128, 128)) * 128).astype(np.uint8)

    # Save the image
    cv2.imwrite(join(save_dir, get_mood(song), f"{song}.png"), img)

    print(f"Progress: {count / numOfSongs * 100:.2f}%\r", end="")
    count += 1

print("Finished extracting mel-spectograms.")