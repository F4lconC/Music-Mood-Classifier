import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import random

# ================= CONFIG =================
DATA_DIR = "mel_spectrograms_segmented"
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPEC_AUG_PROB = 0.3
TIME_MASK = 20
FREQ_MASK = 6
# ==========================================


# --------- SPEC AUGMENT ---------
def spec_augment(mel):
    mel = mel.copy()

    # Time masking
    t = mel.shape[1]
    t0 = random.randint(0, max(0, t - TIME_MASK))
    mel[:, t0:t0 + TIME_MASK] = 0

    # Frequency masking
    f = mel.shape[0]
    f0 = random.randint(0, max(0, f - FREQ_MASK))
    mel[f0:f0 + FREQ_MASK, :] = 0

    return mel


# --------- DATASET ---------
class MelDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.samples = []
        self.class_to_idx = {}
        self.train = train

        for idx, class_dir in enumerate(sorted(Path(root_dir).iterdir())):
            if not class_dir.is_dir():
                continue

            self.class_to_idx[class_dir.name] = idx
            for file in class_dir.glob("*.npy"):
                self.samples.append((file, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        mel = np.load(path)

        # Apply SpecAugment ONLY during training
        if self.train and random.random() < SPEC_AUG_PROB:
            mel = spec_augment(mel)

        # Normalize per sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)

        return mel, label


# --------- MODEL ---------
class MusicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------- MAIN ---------
def main():
    full_dataset = MelDataset(DATA_DIR, train=True)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # IMPORTANT: disable SpecAugment for validation
    val_ds.dataset.train = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MusicCNN(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "music_cnn_best_weak_spec2.pth")

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {train_loss:.3f} | "
            f"Val Acc: {acc:.2f}%"
        )

    print(f"✅ Best Val Accuracy: {best_acc:.2f}%")
    print("✅ Best model saved as music_cnn_best_weak_spec.pth")


if __name__ == "__main__":
    main()
