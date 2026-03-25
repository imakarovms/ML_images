#from future import annotations

import csv
import json
from pathlib import Path

import numpy as np


EMB_DIR = Path("embeddings")
OUTPUT_CSV = Path("embedding_candidates.csv")

TOP_K = 20
MAX_DISTANCE = 0.60


def main():
    embeddings = np.load(EMB_DIR / "embeddings.npy")
    with (EMB_DIR / "paths.json").open("r", encoding="utf-8") as f:
        paths = json.load(f)

    if len(embeddings) != len(paths):
        raise RuntimeError("Число эмбеддингов не совпадает с числом путей")

    # cosine distance для нормализованных эмбеддингов:
    # dist = 1 - cosine_similarity
    sim = embeddings @ embeddings.T
    dist = 1.0 - sim

    rows = []
    used_pairs = set()

    n = len(paths)
    for i in range(n):
        nearest = np.argsort(dist[i])

        taken = 0
        for j in nearest:
            if i == j:
                continue

            d = float(dist[i, j])
            if d > MAX_DISTANCE:
                continue

            pair = (min(i, j), max(i, j))
            if pair in used_pairs:
                continue

            used_pairs.add(pair)

            rows.append({
                "i": i,
                "j": j,
                "path1": paths[i],
                "path2": paths[j],
                "embedding_distance": d,
            })

            taken += 1
            if taken >= TOP_K:
                break

        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"[NN] {i + 1}/{n}")

    rows = sorted(rows, key=lambda x: x["embedding_distance"])

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["i", "j", "path1", "path2", "embedding_distance"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Готово")
    print(f"Кандидаты сохранены: {OUTPUT_CSV}")
    print(f"Всего candidate pairs: {len(rows)}")


if __name__ == "__main__":
    main()