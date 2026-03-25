from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

from config import (
    INPUT_FOLDER,
    NN_OUTPUT_FOLDER,
    NN_EMBEDDING_CANDIDATES_CSV,
    LOG_LEVEL,
    LOG_EVERY_N,
    MAX_WORKERS,
    SUPPORTED_EXTENSIONS,
    PASS1_MIN_KEYPOINTS,
    PASS1_MIN_GOOD_MATCHES,
    PASS1_MIN_INLIERS,
    PASS1_MIN_INLIER_RATIO,
    PASS1_RANSAC_REPROJ_THRESHOLD,
    PASS2_MIN_KEYPOINTS,
    PASS2_MIN_GOOD_MATCHES,
    PASS2_MIN_INLIERS,
    PASS2_MIN_INLIER_RATIO,
    PASS2_RANSAC_REPROJ_THRESHOLD,
)
from hashing import sha1_file
from orb_verify import extract_orb_features, verify_pair_orb
from dsu import DSU
from export_groups import export_groups


def setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def find_images(input_folder: Path) -> list[Path]:
    return sorted(
        [
            p for p in input_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def build_sha1_groups(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for idx, p in enumerate(paths, start=1):
        h = sha1_file(p)
        groups.setdefault(h, []).append(p)
        if idx % max(1, LOG_EVERY_N) == 0 or idx == len(paths):
            logging.info(f"SHA1: {idx}/{len(paths)}")
    return groups


def choose_unique_representatives(sha1_groups: dict[str, list[Path]]) -> list[Path]:
    return [group[0] for group in sha1_groups.values()]


def load_embedding_candidates(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Не найден файл кандидатов: {csv_path}")

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "path1": row["path1"],
                "path2": row["path2"],
                "embedding_distance": float(row["embedding_distance"]),
            })
    return rows


def filter_candidates_to_unique(
    candidates: list[dict],
    rep_set: set[str],
) -> list[dict]:
    def normalize_path(raw_path: str) -> str:
        p = Path(raw_path)
        if p.is_absolute():
            try:
                return str(p.resolve().relative_to(Path.cwd().resolve()))
            except ValueError:
                return str(p.resolve())
        return str(p)

    out = []
    seen = set()

    for row in candidates:
        p1 = normalize_path(row["path1"])
        p2 = normalize_path(row["path2"])

        if p1 not in rep_set or p2 not in rep_set:
            continue

        key = tuple(sorted((p1, p2)))
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "path1": p1,
            "path2": p2,
            "embedding_distance": row["embedding_distance"],
        })

    return out


def compute_orb_for_used_paths(paths: list[Path]) -> dict[str, dict]:
    orb_cache: dict[str, dict] = {}
    for idx, path in enumerate(paths, start=1):
        orb_cache[str(path)] = extract_orb_features(path)
        if idx % max(1, LOG_EVERY_N) == 0 or idx == len(paths):
            logging.info(f"ORB-признаки: {idx}/{len(paths)}")
    return orb_cache


def verify_pair_task(args: tuple[dict, dict, dict]) -> dict:
    feat1, feat2, verify_cfg = args
    return verify_pair_orb(feat1, feat2, **verify_cfg)


def save_pair_results_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_orb_pass(
    *,
    pass_name: str,
    candidate_rows: list[dict],
    verify_cfg: dict,
    path_to_index: dict[str, int],
    dsu: DSU,
) -> int:
    if not candidate_rows:
        logging.info(f"{pass_name}: нет кандидатных пар")
        return 0

    used_paths = sorted(
        {Path(row["path1"]) for row in candidate_rows}.union(
            {Path(row["path2"]) for row in candidate_rows}
        )
    )

    logging.info(f"{pass_name}: файлов для ORB: {len(used_paths)}")
    orb_cache = compute_orb_for_used_paths(used_paths)

    tasks = []
    for row in candidate_rows:
        tasks.append((
            orb_cache[row["path1"]],
            orb_cache[row["path2"]],
            verify_cfg,
        ))

    verified_results: list[dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(verify_pair_task, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                verified_results.append(future.result())
            except Exception as e:
                logging.warning(f"{pass_name}: ошибка ORB-проверки пары: {e}")

            if idx % max(1, LOG_EVERY_N) == 0 or idx == len(futures):
                logging.info(f"{pass_name}: ORB-проверка {idx}/{len(futures)}")

    save_pair_results_csv(verified_results, Path(f"pair_results_{pass_name.lower()}.csv"))

    merged_count = 0
    for res in verified_results:
        if not res["same"]:
            continue

        i = path_to_index[res["path1"]]
        j = path_to_index[res["path2"]]

        if dsu.union(i, j):
            merged_count += 1

    logging.info(f"{pass_name}: подтвержденных объединений: {merged_count}")
    return merged_count


def get_groups_from_dsu(indexed_items: list[dict], dsu: DSU) -> dict[int, list[Path]]:
    root_to_rep_paths: dict[int, list[Path]] = {}
    for item in indexed_items:
        root = dsu.find(item["index"])
        root_to_rep_paths.setdefault(root, []).append(Path(item["path"]))
    return root_to_rep_paths


def expand_final_groups(indexed_items: list[dict], dsu: DSU, sha1_groups: dict[str, list[Path]]) -> list[list[Path]]:
    root_to_rep_paths = get_groups_from_dsu(indexed_items, dsu)

    sha1_to_all_paths = sha1_groups
    path_to_sha1 = {
        str(item["path"]): item["sha1"]
        for item in indexed_items
    }

    final_groups: list[list[Path]] = []

    for rep_group in root_to_rep_paths.values():
        expanded_group: list[Path] = []

        for rep_path in rep_group:
            rep_sha1 = path_to_sha1[str(rep_path)]
            expanded_group.extend(sha1_to_all_paths[rep_sha1])

        seen = set()
        deduped_group = []
        for p in expanded_group:
            key = str(p)
            if key not in seen:
                seen.add(key)
                deduped_group.append(Path(p))

        final_groups.append(deduped_group)

    final_groups = sorted(final_groups, key=lambda g: (-len(g), [str(p) for p in g]))
    return final_groups


def main() -> None:
    setup_logging()
    t0 = time.time()

    input_folder = Path(INPUT_FOLDER)
    output_folder = Path(NN_OUTPUT_FOLDER)
    candidates_path = Path(NN_EMBEDDING_CANDIDATES_CSV)

    if not input_folder.exists():
        logging.error(f"Папка не существует: {input_folder.resolve()}")
        return

    image_paths = find_images(input_folder)
    logging.info(f"Найдено файлов: {len(image_paths)}")

    if not image_paths:
        logging.error("Изображения не найдены")
        return

    if not candidates_path.exists():
        logging.error(f"Не найден файл кандидатных пар: {candidates_path.resolve()}")
        return

    logging.info("Считаю SHA1")
    sha1_groups = build_sha1_groups(image_paths)
    exact_duplicate_count = sum(len(v) - 1 for v in sha1_groups.values() if len(v) > 1)
    logging.info(f"Точных дублей по SHA1: {exact_duplicate_count}")

    unique_paths = choose_unique_representatives(sha1_groups)
    logging.info(f"Уникальных представителей после SHA1: {len(unique_paths)}")

    indexed_items = []
    for idx, p in enumerate(unique_paths):
        indexed_items.append({
            "index": idx,
            "path": p,
            "sha1": sha1_file(p),
        })

    dsu = DSU(len(indexed_items))
    path_to_index = {str(item["path"]): item["index"] for item in indexed_items}

    logging.info("Читаю нейросеточные candidate pairs")
    nn_candidates = load_embedding_candidates(candidates_path)

    rep_set = {str(p) for p in unique_paths}
    nn_candidates = filter_candidates_to_unique(
        nn_candidates,
        rep_set,
    )

    logging.info(f"Кандидатов после фильтра по unique representatives: {len(nn_candidates)}")

    # PASS1: более строгая ORB-проверка по всем nn-кандидатам
    pass1_verify = {
        "min_keypoints": PASS1_MIN_KEYPOINTS,
        "min_good_matches": PASS1_MIN_GOOD_MATCHES,
        "min_inliers": PASS1_MIN_INLIERS,
        "min_inlier_ratio": PASS1_MIN_INLIER_RATIO,
        "ransac_reproj_threshold": PASS1_RANSAC_REPROJ_THRESHOLD,
    }

    run_orb_pass(
        pass_name="NN_PASS1",
        candidate_rows=nn_candidates,
        verify_cfg=pass1_verify,
        path_to_index=path_to_index,
        dsu=dsu,
    )

    # PASS2: только по одиночкам, более мягкая ORB-проверка
    root_to_paths = get_groups_from_dsu(indexed_items, dsu)
    singleton_paths = {
        str(paths[0])
        for paths in root_to_paths.values()
        if len(paths) == 1
    }

    nn_candidates_pass2 = [
        row for row in nn_candidates
        if row["path1"] in singleton_paths and row["path2"] in singleton_paths
    ]

    logging.info(f"NN_PASS2: кандидатов только среди одиночек: {len(nn_candidates_pass2)}")

    pass2_verify = {
        "min_keypoints": PASS2_MIN_KEYPOINTS,
        "min_good_matches": PASS2_MIN_GOOD_MATCHES,
        "min_inliers": PASS2_MIN_INLIERS,
        "min_inlier_ratio": PASS2_MIN_INLIER_RATIO,
        "ransac_reproj_threshold": PASS2_RANSAC_REPROJ_THRESHOLD,
    }

    run_orb_pass(
        pass_name="NN_PASS2",
        candidate_rows=nn_candidates_pass2,
        verify_cfg=pass2_verify,
        path_to_index=path_to_index,
        dsu=dsu,
    )

    final_groups = expand_final_groups(indexed_items, dsu, sha1_groups)
    export_groups(final_groups, output_folder)

    singleton_count = sum(1 for g in final_groups if len(g) == 1)

    logging.info(f"Готово. Групп найдено: {len(final_groups)}")
    logging.info(f"Одиночных групп: {singleton_count}")
    logging.info(f"Общее время: {time.time() - t0:.2f} сек")
    logging.info(f"Результат сохранен в: {output_folder.resolve()}")


if __name__ == "__main__":
    print("START")
    main()
    print("END")
