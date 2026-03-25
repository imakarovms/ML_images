from __future__ import annotations

import csv
import json
import random
import shutil
from itertools import combinations
from pathlib import Path


GROUPS_FOLDER = Path("grouped_images")
PAIR_RESULTS_CSV = Path("pair_results.csv")
OUTPUT_FOLDER = Path("review_dataset")

RANDOM_SEED = 42

# сколько максимум positive-пар брать из одной группы
MAX_POSITIVE_PAIRS_PER_GROUP = 30

# сколько negative-пар брать на одну positive-пару
NEGATIVE_PER_POSITIVE = 1

# сколько hard negative-пар брать максимум
MAX_HARD_NEGATIVES = 3000


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_group_dirs(groups_folder: Path) -> list[Path]:
    return sorted([p for p in groups_folder.iterdir() if p.is_dir()])


def load_groups(groups_folder: Path) -> list[list[Path]]:
    groups = []
    for group_dir in find_group_dirs(groups_folder):
        files = sorted([p for p in group_dir.iterdir() if p.is_file()])
        if files:
            groups.append(files)
    return groups


def sample_pairs_from_group(files: list[Path], max_pairs: int) -> list[tuple[Path, Path]]:
    pairs = list(combinations(files, 2))
    if len(pairs) <= max_pairs:
        return pairs
    return random.sample(pairs, max_pairs)


def build_positive_pairs(groups: list[list[Path]]) -> list[dict]:
    rows = []
    for group_idx, files in enumerate(groups, start=1):
        if len(files) < 2:
            continue

        pairs = sample_pairs_from_group(files, MAX_POSITIVE_PAIRS_PER_GROUP)
        for a, b in pairs:
            rows.append({
                "pair_type": "positive",
                "group_a": group_idx,
                "group_b": group_idx,
                "path1": str(a),
                "path2": str(b),
                "source": "grouped_images",
            })
    return rows


def build_negative_pairs(groups: list[list[Path]], target_count: int) -> list[dict]:
    nonempty_groups = [g for g in groups if len(g) > 0]
    if len(nonempty_groups) < 2 or target_count <= 0:
        return []

    rows = []
    attempts = 0
    max_attempts = target_count * 20

    while len(rows) < target_count and attempts < max_attempts:
        attempts += 1

        gi, gj = random.sample(range(len(nonempty_groups)), 2)
        group_i = nonempty_groups[gi]
        group_j = nonempty_groups[gj]

        a = random.choice(group_i)
        b = random.choice(group_j)

        rows.append({
            "pair_type": "negative",
            "group_a": gi + 1,
            "group_b": gj + 1,
            "path1": str(a),
            "path2": str(b),
            "source": "cross_group_sampling",
        })

    return rows


def load_hard_negatives(pair_results_csv: Path) -> list[dict]:
    if not pair_results_csv.exists():
        return []

    rows = []
    with pair_results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            same_raw = str(row.get("same", "")).strip().lower()
            if same_raw not in {"false", "0"}:
                continue

            rows.append({
                "pair_type": "hard_negative",
                "group_a": "",
                "group_b": "",
                "path1": row["path1"],
                "path2": row["path2"],
                "source": "orb_failed_candidate",
                "good_matches": row.get("good_matches", ""),
                "inliers": row.get("inliers", ""),
                "inlier_ratio": row.get("inlier_ratio", ""),
                "reason": row.get("reason", ""),
                "angle": row.get("angle", ""),
            })

    if len(rows) > MAX_HARD_NEGATIVES:
        rows = random.sample(rows, MAX_HARD_NEGATIVES)

    return rows


def dedupe_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for row in rows:
        p1 = str(Path(row["path1"]).resolve())
        p2 = str(Path(row["path2"]).resolve())
        key = tuple(sorted((p1, p2)))

        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


def export_pair_folder(base_dir: Path, pair_id: int, row: dict) -> dict:
    pair_dir = base_dir / f"pair_{pair_id:06d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    src1 = Path(row["path1"])
    src2 = Path(row["path2"])

    dst1 = pair_dir / f"A{src1.suffix.lower()}"
    dst2 = pair_dir / f"B{src2.suffix.lower()}"

    shutil.copy2(src1, dst1)
    shutil.copy2(src2, dst2)

    meta = dict(row)
    meta["pair_id"] = pair_id
    meta["copied_a"] = str(dst1)
    meta["copied_b"] = str(dst2)

    with (pair_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "pair_id": pair_id,
        **row,
        "pair_folder": str(pair_dir),
        "copied_a": str(dst1),
        "copied_b": str(dst2),
    }


def save_manifest(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(RANDOM_SEED)

    if not GROUPS_FOLDER.exists():
        raise FileNotFoundError(f"Не найдена папка групп: {GROUPS_FOLDER}")

    clear_dir(OUTPUT_FOLDER)

    pairs_dir = ensure_dir(OUTPUT_FOLDER / "pairs")
    ensure_dir(OUTPUT_FOLDER / "sorted" / "positive")
    ensure_dir(OUTPUT_FOLDER / "sorted" / "negative")
    ensure_dir(OUTPUT_FOLDER / "sorted" / "skip")

    groups = load_groups(GROUPS_FOLDER)

    positive_rows = build_positive_pairs(groups)
    negative_rows = build_negative_pairs(
        groups,
        target_count=len(positive_rows) * NEGATIVE_PER_POSITIVE,
    )
    hard_negative_rows = load_hard_negatives(PAIR_RESULTS_CSV)

    all_rows = positive_rows + negative_rows + hard_negative_rows
    all_rows = dedupe_rows(all_rows)
    random.shuffle(all_rows)

    manifest_rows = []
    for idx, row in enumerate(all_rows, start=1):
        manifest_rows.append(export_pair_folder(pairs_dir, idx, row))

    save_manifest(manifest_rows, OUTPUT_FOLDER / "manifest.csv")

    stats = {
        "total_pairs": len(manifest_rows),
        "positive": sum(1 for r in manifest_rows if r["pair_type"] == "positive"),
        "negative": sum(1 for r in manifest_rows if r["pair_type"] == "negative"),
        "hard_negative": sum(1 for r in manifest_rows if r["pair_type"] == "hard_negative"),
    }

    with (OUTPUT_FOLDER / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Готово")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Папка датасета: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()