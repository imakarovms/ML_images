from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------- CONFIG --------------------

TRAIN_CSV = Path("training_data/train_pairs.csv")
VAL_CSV = Path("training_data/val_pairs.csv")
MODELS_DIR = Path("models")

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
EMBED_DIM = 128
MARGIN = 1.0
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# -------------------- UTILS --------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            rows.append(row)
    return rows


def read_image(path: str, image_size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def random_augment(img: np.ndarray) -> np.ndarray:
    # легкие аугментации
    if random.random() < 0.5:
        angle = random.uniform(-20, 20)
        h, w = img.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            img,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    if random.random() < 0.5:
        alpha = random.uniform(0.9, 1.1)   # contrast
        beta = random.uniform(-0.05, 0.05) # brightness
        img = np.clip(img * alpha + beta, 0.0, 1.0)

    if random.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def to_tensor(img: np.ndarray) -> torch.Tensor:
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


# -------------------- DATASET --------------------

class PairDataset(Dataset):
    def __init__(self, rows: list[dict], image_size: int, train: bool):
        self.rows = rows
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        img1 = read_image(row["path1"], self.image_size)
        img2 = read_image(row["path2"], self.image_size)

        if self.train:
            img1 = random_augment(img1)
            img2 = random_augment(img2)

        img1 = to_tensor(img1)
        img2 = to_tensor(img2)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return img1, img2, label


# -------------------- MODEL --------------------

class SmallEmbeddingNet(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        dist = F.pairwise_distance(emb1, emb2)

        positive_loss = label * torch.pow(dist, 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)

        loss = torch.mean(positive_loss + negative_loss)
        return loss, dist


# -------------------- TRAIN / EVAL --------------------

@dataclass
class Metrics:
    loss: float
    accuracy: float


def compute_accuracy(dist: torch.Tensor, label: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (dist < threshold).float()
    acc = (pred == label).float().mean().item()
    return acc


def run_epoch(model, loader, criterion, optimizer=None) -> Metrics:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for img1, img2, label in loader:
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        label = label.to(DEVICE)

        emb1 = model(img1)
        emb2 = model(img2)

        loss, dist = criterion(emb1, emb2, label)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = img1.size(0)
        total_loss += loss.item() * batch_size
        total_acc += compute_accuracy(dist.detach(), label.detach()) * batch_size
        total_count += batch_size

    return Metrics(
        loss=total_loss / max(1, total_count),
        accuracy=total_acc / max(1, total_count),
    )


def main():
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(TRAIN_CSV)
    val_rows = load_rows(VAL_CSV)

    train_ds = PairDataset(train_rows, image_size=IMAGE_SIZE, train=True)
    val_ds = PairDataset(val_rows, image_size=IMAGE_SIZE, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = SmallEmbeddingNet(embed_dim=EMBED_DIM).to(DEVICE)
    criterion = ContrastiveLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    history = []

    print(f"Device: {DEVICE}")
    print(f"Train pairs: {len(train_ds)}")
    print(f"Val pairs: {len(val_ds)}")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embed_dim": EMBED_DIM,
                    "image_size": IMAGE_SIZE,
                },
                MODELS_DIR / "best_embedding.pt",
            )

    with (MODELS_DIR / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("Готово")
    print(f"Лучшая модель сохранена: {MODELS_DIR / 'best_embedding.pt'}")


if __name__ == "__main__":
    main()