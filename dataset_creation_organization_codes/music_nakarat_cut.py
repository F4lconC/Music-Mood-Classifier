from pydub import AudioSegment
import os

src_root = "/content/drive/MyDrive/Colab Notebooks/NJU_Musics"
dst_root = "/content/drive/MyDrive/Colab Notebooks/Nakarat_NJU_Musics"

os.makedirs(dst_root, exist_ok=True)

for genre in os.listdir(src_root):
    genre_path = os.path.join(src_root, genre)
    if not os.path.isdir(genre_path):
        continue

    out_genre_path = os.path.join(dst_root, genre)
    os.makedirs(out_genre_path, exist_ok=True)

    for split in os.listdir(genre_path):
        split_path = os.path.join(genre_path, split)
        if not os.path.isdir(split_path):
            continue

        out_split_path = os.path.join(out_genre_path, split)
        os.makedirs(out_split_path, exist_ok=True)

        for file in os.listdir(split_path):
            if not file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                continue

            input_path = os.path.join(split_path, file)
            output_path = os.path.join(out_split_path, file)

            try:
                audio = AudioSegment.from_file(input_path)
                duration = len(audio)
                if duration > 60000:
                    start = (duration - 30000) // 2
                    clip = audio[start:start + 30000]
                    clip.export(output_path, format="mp3")
                    print(f"{file} → tamamlandı")
                else:
                    print(f"{file} → kısa, atlandı")
            except Exception as e:
                print(f"Hata ({file}): {e}")

print("Tüm kesimler bitti.")
