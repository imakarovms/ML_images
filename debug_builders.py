from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from config import SUPPORTED_EXTENSIONS
from hashing import phash, dhash


# -------------------- PATHS --------------------

IMAGES_FOLDER = Path("images")
GROUPS_FOLDER = Path("grouped_images")
PAIR_RESULTS_CSV = Path("pair_results.csv")
OUTPUT_FOLDER = Path("review_dataset")


# -------------------- SETTINGS --------------------

RANDOM_SEED = 42

# сколько candidate positive брать максимум
MAX_POSITIVE_CANDIDATES = 2500

# сколько candidate negative брать на одну positive
NEGATIVE_PER_POSITIVE = 1

# минимальное число negative-кандидатов даже при малом числе positive
MIN_NEGATIVE_CANDIDATES = 400

# ограничение пула hard negative из результатов ORB
MAX_HARD_NEGATIVES = 3000

# хэш-пороги
SIMILAR_PHASH_MAX = 10
SIMILAR_DHASH_MAX = 10
NEGATIVE_PHASH_MIN = 20
NEGATIVE_DHASH_MIN = 20

# если grouped_images совсем маленький/плохой — не используем его как основной источник
MIN_GROUPS_FOR_TRUSTED_GROUP_MODE = 20

# ограничение внутреннего пула дальних пар (reservoir sampling)
MAX_NEGATIVE_POOL = 30000


@dataclass
class ImageHash:
    path: Path
    phash: int
    dhash: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_images(folder: Path) -> list[Path]:
    return sorted(
        [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def find_group_dirs(groups_folder: Path) -> list[Path]:
    if not groups_folder.exists():
        return []
    return sorted([p for p in groups_folder.iterdir() if p.is_dir()])


def load_groups(groups_folder: Path) -> list[list[Path]]:
    groups = []
    for group_dir in find_group_dirs(groups_folder):
        files = sorted([p for p in group_dir.iterdir() if p.is_file()])
        if files:
            groups.append(files)
    return groups


def build_hashes(images: list[Path]) -> list[ImageHash]:
    out: list[ImageHash] = []

    for idx, path in enumerate(images, start=1):
        try:
            out.append(ImageHash(path=path, phash=phash(path), dhash=dhash(path)))
        except Exception:
            continue

        if idx % 200 == 0 or idx == len(images):
            print(f"[HASH] {idx}/{len(images)}")

    return out


def build_similar_and_negative_candidates(items: list[ImageHash]) -> tuple[list[dict], list[dict]]:
    similar_rows: list[dict] = []
    negative_pool: list[dict] = []
    seen_negative = 0

    for left, right in combinations(items, 2):
        p_dist = (left.phash ^ right.phash).bit_count()
        d_dist = (left.dhash ^ right.dhash).bit_count()

        if p_dist <= SIMILAR_PHASH_MAX and d_dist <= SIMILAR_DHASH_MAX:
            similar_rows.append({
                "pair_type": "positive_candidate",
                "group_a": "",
                "group_b": "",
                "path1": str(left.path),
                "path2": str(right.path),
                "source": "images_hash_near",
                "phash_distance": p_dist,
                "dhash_distance": d_dist,
                "hash_score": p_dist + d_dist,
            })
            continue

        if p_dist >= NEGATIVE_PHASH_MIN and d_dist >= NEGATIVE_DHASH_MIN:
            seen_negative += 1
            row = {
                "pair_type": "negative_candidate",
                "group_a": "",
                "group_b": "",
                "path1": str(left.path),
                "path2": str(right.path),
                "source": "images_hash_far",
                "phash_distance": p_dist,
                "dhash_distance": d_dist,
                "hash_score": p_dist + d_dist,
            }

            if len(negative_pool) < MAX_NEGATIVE_POOL:
                negative_pool.append(row)
            else:
                # reservoir sampling
                idx = random.randint(0, seen_negative - 1)
                if idx < MAX_NEGATIVE_POOL:
                    negative_pool[idx] = row

    similar_rows = sorted(similar_rows, key=lambda x: (x["hash_score"], x["phash_distance"], x["dhash_distance"]))

    if len(similar_rows) > MAX_POSITIVE_CANDIDATES:
        # берём лучшие + небольшая случайная примесь для разнообразия
        head = similar_rows[: int(MAX_POSITIVE_CANDIDATES * 0.8)]
        tail_pool = similar_rows[int(MAX_POSITIVE_CANDIDATES * 0.8):]
        tail_size = MAX_POSITIVE_CANDIDATES - len(head)
        tail = random.sample(tail_pool, tail_size) if len(tail_pool) > tail_size else tail_pool
        similar_rows = head + tail

    target_negative = max(len(similar_rows) * NEGATIVE_PER_POSITIVE, MIN_NEGATIVE_CANDIDATES)
    target_negative = min(target_negative, len(negative_pool))
    negative_rows = random.sample(negative_pool, target_negative) if target_negative > 0 else []

    return similar_rows, negative_rows


def sample_pairs_from_group(files: list[Path], max_pairs: int = 30) -> list[tuple[Path, Path]]:
    pairs = list(combinations(files, 2))
    if len(pairs) <= max_pairs:
        return pairs
    return random.sample(pairs, max_pairs)


def build_candidates_from_groups(groups: list[list[Path]]) -> tuple[list[dict], list[dict]]:
    positive_rows: list[dict] = []
    for group_idx, files in enumerate(groups, start=1):
        if len(files) < 2:
            continue

        for a, b in sample_pairs_from_group(files, max_pairs=30):
            positive_rows.append({
                "pair_type": "positive_candidate",
                "group_a": group_idx,
                "group_b": group_idx,
                "path1": str(a),
                "path2": str(b),
                "source": "grouped_images",
            })

    if len(groups) < 2 or not positive_rows:
        return positive_rows, []

    negative_target = len(positive_rows) * NEGATIVE_PER_POSITIVE
    negative_rows = []
    attempts = 0
    max_attempts = max(negative_target * 20, 500)

    while len(negative_rows) < negative_target and attempts < max_attempts:
        attempts += 1
        gi, gj = random.sample(range(len(groups)), 2)
        a = random.choice(groups[gi])
        b = random.choice(groups[gj])

        negative_rows.append({
            "pair_type": "negative_candidate",
            "group_a": gi + 1,
            "group_b": gj + 1,
            "path1": str(a),
            "path2": str(b),
            "source": "cross_group_sampling",
        })

    return positive_rows, negative_rows


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


def main() -> None:
    random.seed(RANDOM_SEED)

    if not IMAGES_FOLDER.exists():
        raise FileNotFoundError(f"Не найдена папка изображений: {IMAGES_FOLDER}")

    clear_dir(OUTPUT_FOLDER)

    pairs_dir = ensure_dir(OUTPUT_FOLDER / "pairs")
    ensure_dir(OUTPUT_FOLDER / "sorted" / "positive")
    ensure_dir(OUTPUT_FOLDER / "sorted" / "negative")
    ensure_dir(OUTPUT_FOLDER / "sorted" / "skip")

    # 1) Всегда анализируем images — это основной источник
    images = find_images(IMAGES_FOLDER)
    print(f"[IMAGES] found={len(images)}")
    hashed_items = build_hashes(images)
    print(f"[IMAGES] hash_ok={len(hashed_items)}")

    positive_rows, negative_rows = build_similar_and_negative_candidates(hashed_items)
    print(f"[IMAGES] positive_candidates={len(positive_rows)}")
    print(f"[IMAGES] negative_candidates={len(negative_rows)}")

    # 2) Если группировка выглядит адекватной — добавляем кандидаты из grouped_images
    groups = load_groups(GROUPS_FOLDER)
    if len(groups) >= MIN_GROUPS_FOR_TRUSTED_GROUP_MODE:
        gp, gn = build_candidates_from_groups(groups)
        positive_rows.extend(gp)
        negative_rows.extend(gn)
        print(f"[GROUPS] trusted mode: groups={len(groups)} +{len(gp)} pos +{len(gn)} neg")
    else:
        print(f"[GROUPS] skipped: groups={len(groups)} < {MIN_GROUPS_FOR_TRUSTED_GROUP_MODE}")

    hard_negative_rows = load_hard_negatives(PAIR_RESULTS_CSV)

    all_rows = dedupe_rows(positive_rows + negative_rows + hard_negative_rows)
    random.shuffle(all_rows)

    manifest_rows = []
    for idx, row in enumerate(all_rows, start=1):
        manifest_rows.append(export_pair_folder(pairs_dir, idx, row))

    save_manifest(manifest_rows, OUTPUT_FOLDER / "manifest.csv")

    positive_types = {"positive", "positive_candidate"}
    negative_types = {"negative", "negative_candidate"}

    stats = {
        "total_pairs": len(manifest_rows),
        "positive": sum(1 for r in manifest_rows if r["pair_type"] in positive_types),
        "negative": sum(1 for r in manifest_rows if r["pair_type"] in negative_types),
        "hard_negative": sum(1 for r in manifest_rows if r["pair_type"] == "hard_negative"),
        "images_found": len(images),
        "images_hashed": len(hashed_items),
        "groups_found": len(groups),
    }

    with (OUTPUT_FOLDER / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("Готово")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Папка датасета: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()
