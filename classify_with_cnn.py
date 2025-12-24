import os
import shutil
from pathlib import Path

import yt_dlp
import numpy as np
import torch

from dataset_creation_organization_codes.audio_utils import audio_to_mel_segments
from training_codes.train_torch_cnn import MusicCNN

# ===================== CONFIG =====================
TMP_MEL_DIR = "tmp_mel"
TMP_DIR = "tmp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
MODEL_PATH = "music_cnn_best_song_level70_41.pth"

MOODS = ["happy", "sad", "angry", "calm"]  # adjust to your labels
# =================================================


def download_song(query: str) -> str:
    """
    Downloads audio from YouTube search.
    Returns path to mp3 file.
    """
    os.makedirs(TMP_DIR, exist_ok=True)

    file_path = os.path.join(TMP_DIR, "song")
    query = f"ytsearch1:{query}"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": file_path,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
        ],
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([query])

    return file_path + ".mp3"

def cut_song_to_30_sec(file_path):
    """
        Takes the file path and creates a 30 second part with the same name and path
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(file_path)
    duration = len(audio)

    if duration <= 30000:
        return

    start = (duration - 30000) // 2
    clip = audio[start:start + 30000]
    clip.export(file_path, format="mp3")

def prepare_mel_segments(audio_path: str) -> torch.Tensor:
    """
    Converts audio to mel spectrogram segments.
    Returns tensor of shape (N, 1, n_mels, time)
    """
    

    audio_to_mel_segments(audio_path=Path(audio_path), out_dir=TMP_MEL_DIR)

    segments = []
    for mel_file in sorted(Path(TMP_MEL_DIR).glob("*.npy")):
        mel = np.load(mel_file)

        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        segments.append(mel)

    segments = torch.stack(segments)  # (N, 1, H, W)
    return segments


def predict_song(model, segments):
    """
    segments: (N, 1, H, W)
    """
    segments = segments.to(DEVICE)

    with torch.no_grad():
        logits = model(segments)          # (N, num_classes)
        probs = torch.softmax(logits, 1)
        mean_probs = probs.mean(dim=0)

        pred = mean_probs.argmax().item()

    return pred, mean_probs.cpu()


# ===================== MAIN =====================
if __name__ == "__main__":
    song_query = input("Enter song name and artist: ")

    # 1️⃣ Download
    audio_path = download_song(song_query)

    cut_song_to_30_sec(audio_path)

    # 2️⃣ Convert to mel segments
    segments = prepare_mel_segments(audio_path)

    # 3️⃣ Load model
    model = MusicCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 4️⃣ Predict
    pred, probs = predict_song(model, segments)

    print("\n🎵 Prediction Results")
    print("--------------------")
    print(f"Predicted Mood: {MOODS[pred]}")
    print("Class Probabilities:")
    for mood, p in zip(MOODS, probs):
        print(f"  {mood}: {p:.3f}")
