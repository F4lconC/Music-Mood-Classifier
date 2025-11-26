import os
import joblib
import yt_dlp
import pandas as pd
from extract_features import extract_features, load_file, feature_names
from sklearn.model_selection import GridSearchCV, train_test_split


def download_song(query):
    """
    Args: 
        query (str): search query to find the song 
    """

    file_path = f"tmp/{query}"
    query = f"ytsearch1:{query}"

    ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': file_path,
    'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    'quiet': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
    except Exception as e:
        raise ValueError("Song could not be downloaded. Please check your query.")

    return file_path + ".mp3"

def cut_song_to_30_sec(file_path):
    """
        Takes the file path and creates a 30 second part with the same name and path
    """
    
    pass

def classify(features: dict):
    """
        Args:
            features (dict): extracted features of song
    """
    X = pd.DataFrame([features])[feature_names].drop(columns=["bpm"])
    model = joblib.load("voting_model.pkl")
    pred = model.predict(X)
    print("Predicted mood:", pred)


if __name__ == "__main__":
    song_query = input("Please enter name and artist of the song you want to classify: ")
    
    # Download
    file_path = download_song(song_query)

    # Cut
    cut_song_to_30_sec(file_path)

    # Extract features
    y, sr = load_file(file_path)
    features = extract_features(y, sr)

    # Classify
    classify(features)