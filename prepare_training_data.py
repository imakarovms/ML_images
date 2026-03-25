#from future import annotations

import csv
import json
import random
from pathlib import Path

RANDOM_SEED = 42
VAL_SPLIT = 0.2

REVIEW_DATASET_DIR = Path("review_dataset")
SORTED_POSITIVE_DIR = REVIEW_DATASET_DIR / "sorted" / "positive"
SORTED_NEGATIVE_DIR = REVIEW_DATASET_DIR / "sorted" / "negative"
OUTPUT_DIR = Path("training_data")


def find_pair_images(pair_dir: Path) -> tuple[Path, Path]:
    a_candidates = sorted(pair_dir.glob("A.*"))
    b_candidates = sorted(pair_dir.glob("B.*"))

    if not a_candidates:
        raise FileNotFoundError(f"Не найден файл A.* в {pair_dir}")
    if not b_candidates:
        raise FileNotFoundError(f"Не найден файл B.* в {pair_dir}")

    return a_candidates[0], b_candidates[0]


def load_pairs_from_dir(base_dir: Path, label: int) -> list[dict]:
    rows = []

    if not base_dir.exists():
        return rows

    pair_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])

    for pair_dir in pair_dirs:
        meta_path = pair_dir / "meta.json"
        if not meta_path.exists():
            print(f"[WARN] Нет meta.json: {pair_dir}")
            continue

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        try:
            current_a, current_b = find_pair_images(pair_dir)
        except Exception as e:
            print(f"[WARN] Пропуск {pair_dir}: {e}")
            continue

        rows.append({
            "path1": str(current_a),
            "path2": str(current_b),
            "label": label,
            "pair_type": meta.get("pair_type", ""),
            "source": meta.get("source", ""),
            "group_a": meta.get("group_a", ""),
            "group_b": meta.get("group_b", ""),
        })

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"Нет данных для сохранения в {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["path1", "path2", "label", "pair_type", "source", "group_a", "group_b"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(RANDOM_SEED)

    positive_rows = load_pairs_from_dir(SORTED_POSITIVE_DIR, label=1)
    negative_rows = load_pairs_from_dir(SORTED_NEGATIVE_DIR, label=0)

    all_rows = positive_rows + negative_rows
    random.shuffle(all_rows)

    if not all_rows:
        raise RuntimeError("Не найдено ни одной размеченной пары")

    val_size = max(1, int(len(all_rows) * VAL_SPLIT))
    val_rows = all_rows[:val_size]
    train_rows = all_rows[val_size:]

    save_csv(train_rows, OUTPUT_DIR / "train_pairs.csv")
    save_csv(val_rows, OUTPUT_DIR / "val_pairs.csv")

    stats = {
        "total": len(all_rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "positive_total": sum(r["label"] == 1 for r in all_rows),
        "negative_total": sum(r["label"] == 0 for r in all_rows),
        "positive_train": sum(r["label"] == 1 for r in train_rows),
        "negative_train": sum(r["label"] == 0 for r in train_rows),
        "positive_val": sum(r["label"] == 1 for r in val_rows),
        "negative_val": sum(r["label"] == 0 for r in val_rows),
    }

    with (OUTPUT_DIR / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Готово")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()