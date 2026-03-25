#from future import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- CONFIG --------------------

IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("embeddings")
MODEL_PATH = Path("models/best_embedding.pt")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


# -------------------- UTILS --------------------

def find_images(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )


def read_image(path: Path, image_size: int) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    embed_dim = checkpoint["embed_dim"]
    image_size = checkpoint["image_size"]

    model = SmallEmbeddingNet(embed_dim=embed_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    image_paths = find_images(IMAGES_DIR)
    if not image_paths:
        raise RuntimeError("Не найдено изображений")

    embeddings = []
    paths = []

    with torch.no_grad():
        for idx, path in enumerate(image_paths, start=1):
            x = read_image(path, image_size).unsqueeze(0).to(DEVICE)
            emb = model(x).cpu().numpy()[0]
            embeddings.append(emb)
            paths.append(str(path))

            if idx % 100 == 0 or idx == len(image_paths):
                print(f"[EMB] {idx}/{len(image_paths)}")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    with (OUTPUT_DIR / "paths.json").open("w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "count": len(paths),
                "embed_dim": int(embeddings.shape[1]),
                "device": DEVICE,
                "model_path": str(MODEL_PATH),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Готово")
    print(f"Эмбеддинги сохранены в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()