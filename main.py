from __future__ import annotations

import logging
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

from config import (
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    SUPPORTED_EXTENSIONS,
    MAX_WORKERS,
    PHASH_MAX_DISTANCE,
    DHASH_MAX_DISTANCE,
    MAX_MASK_AREA_RATIO,
    MAX_BBOX_ASPECT_DIFF,
    LOG_LEVEL,
    LOG_EVERY_N_PAIRS,
    SAVE_DEBUG_MATCHES,
    DEBUG_FOLDER,
)
from hashing import sha1_file, phash, dhash
from masks import extract_mask_features_from_image
from candidates import generate_candidate_pairs
from orb_verify import extract_orb_features, verify_pair_orb
from dsu import DSU
from export_groups import export_groups


def setup_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def find_images(input_folder: Path) -> list[Path]:
    """
    Ищет все изображения в папке рекурсивно.
    """
    return sorted(
        [
            p
            for p in input_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def compute_fast_features_for_path(path: str | Path) -> dict:
    """
    Считает быстрые признаки для одного файла:
    - SHA1
    - pHash
    - dHash
    - mask features
    """
    path = Path(path)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")

    mask_data = extract_mask_features_from_image(img)

    return {
        "path": path,
        "sha1": sha1_file(path),
        "phash": phash(path),
        "dhash": dhash(path),
        "mask_features": mask_data["mask_features"],
    }


def build_sha1_groups(items: list[dict]) -> dict[str, list[dict]]:
    """
    Группирует элементы по точному файловому SHA1.
    """
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["sha1"], []).append(item)
    return groups


def choose_unique_representatives(sha1_groups: dict[str, list[dict]]) -> list[dict]:
    """
    Берем по одному представителю из каждой группы SHA1.
    """
    representatives = []
    for _, group in sha1_groups.items():
        representatives.append(group[0])
    return representatives


def compute_orb_for_used_paths(paths: list[Path]) -> dict[str, dict]:
    """
    Считает ORB признаки только для тех файлов, которые реально участвуют в candidate pairs.
    """
    orb_cache: dict[str, dict] = {}
    for path in paths:
        orb_cache[str(path)] = extract_orb_features(path)
    return orb_cache


def verify_pair_task(args: tuple[dict, dict, bool, str | None]) -> dict:
    feat1, feat2, save_debug, debug_path = args
    return verify_pair_orb(
        feat1,
        feat2,
        save_debug=save_debug,
        debug_path=debug_path,
    )


def main() -> None:
    setup_logging()
    t0 = time.time()

    input_folder = Path(INPUT_FOLDER)
    output_folder = Path(OUTPUT_FOLDER)
    debug_folder = Path(DEBUG_FOLDER)

    if not input_folder.exists():
        logging.error(f"Папка не существует: {input_folder.resolve()}")
        return

    image_paths = find_images(input_folder)
    logging.info(f"Найдено файлов: {len(image_paths)}")

    if not image_paths:
        logging.error("Изображения не найдены")
        return

    # -------------------- 1. Быстрые признаки --------------------
    logging.info("Считаю быстрые признаки: SHA1, pHash, dHash, mask features")
    fast_items: list[dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(compute_fast_features_for_path, path): path
            for path in image_paths
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                item = future.result()
                fast_items.append(item)
            except Exception as e:
                logging.warning(f"Пропуск {path}: {e}")

            if idx % max(1, LOG_EVERY_N_PAIRS) == 0 or idx == len(futures):
                logging.info(f"Быстрые признаки: {idx}/{len(futures)}")

    if not fast_items:
        logging.error("Не удалось обработать ни одного изображения")
        return

    # -------------------- 2. Точные дубли по SHA1 --------------------
    sha1_groups = build_sha1_groups(fast_items)
    exact_duplicate_count = sum(len(v) - 1 for v in sha1_groups.values() if len(v) > 1)
    logging.info(f"Точных дублей по SHA1: {exact_duplicate_count}")

    unique_items = choose_unique_representatives(sha1_groups)
    logging.info(f"Уникальных представителей после SHA1: {len(unique_items)}")

    # Пронумеруем для DSU/candidate generation
    indexed_items = []
    for idx, item in enumerate(unique_items):
        indexed_items.append({
            "index": idx,
            "path": item["path"],
            "phash": item["phash"],
            "dhash": item["dhash"],
            "mask_features": item["mask_features"],
            "sha1": item["sha1"],
        })

    # -------------------- 3. Генерация candidate pairs --------------------
    logging.info("Генерирую candidate pairs")
    candidate_pairs = generate_candidate_pairs(
        items=indexed_items,
        phash_max_distance=PHASH_MAX_DISTANCE,
        dhash_max_distance=DHASH_MAX_DISTANCE,
        max_mask_area_ratio=MAX_MASK_AREA_RATIO,
        max_bbox_aspect_diff=MAX_BBOX_ASPECT_DIFF,
    )

    total_full_pairs = len(indexed_items) * (len(indexed_items) - 1) // 2
    logging.info(f"Всего пар без prefilter: {total_full_pairs}")
    logging.info(f"Кандидатных пар после prefilter: {len(candidate_pairs)}")

    if not candidate_pairs:
        logging.info("Кандидатных пар нет, экспортирую уникальные группы")
        final_groups = []
        for group in sha1_groups.values():
            final_groups.append([item["path"] for item in group])

        final_groups = sorted(final_groups, key=lambda g: (-len(g), [str(p) for p in g]))
        export_groups(final_groups, output_folder)
        logging.info(f"Готово. Групп: {len(final_groups)}")
        logging.info(f"Время: {time.time() - t0:.2f} сек")
        return

    # -------------------- 4. ORB только для файлов-кандидатов --------------------
    used_paths = sorted(
        {
            Path(pair["path_i"]) for pair in candidate_pairs
        }.union(
            {Path(pair["path_j"]) for pair in candidate_pairs}
        )
    )

    logging.info(f"Считаю ORB только для нужных файлов: {len(used_paths)}")
    orb_cache = compute_orb_for_used_paths(used_paths)

    # -------------------- 5. Точная ORB-проверка --------------------
    dsu = DSU(len(indexed_items))

    tasks = []
    for pair in candidate_pairs:
        path_i = pair["path_i"]
        path_j = pair["path_j"]

        tasks.append((
            orb_cache[path_i],
            orb_cache[path_j],
            False,
            None,
        ))


    logging.info("Запускаю точную ORB-проверку кандидатных пар")
    verified_results: list[dict] = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(verify_pair_task, task): idx
            for idx, task in enumerate(tasks)
        }

        for done_idx, future in enumerate(as_completed(futures), start=1):
            try:
                res = future.result()
                verified_results.append(res)
            except Exception as e:
                logging.warning(f"Ошибка ORB-проверки пары: {e}")
                continue

            if done_idx % max(1, LOG_EVERY_N_PAIRS) == 0 or done_idx == len(futures):
                logging.info(f"ORB-проверка: {done_idx}/{len(futures)}")

    # -------------------- 6. Объединение через DSU --------------------
    path_to_index = {str(item["path"]): item["index"] for item in indexed_items}

    matches_count = 0
    for res in verified_results:
        if not res["same"]:
            continue

        i = path_to_index[res["path1"]]
        j = path_to_index[res["path2"]]

        if dsu.union(i, j):
            matches_count += 1

    logging.info(f"Подтвержденных ORB-совпадений: {matches_count}")

    # -------------------- 7. Сборка групп уникальных представителей --------------------
    root_to_rep_paths: dict[int, list[Path]] = {}
    for item in indexed_items:
        root = dsu.find(item["index"])
        root_to_rep_paths.setdefault(root, []).append(Path(item["path"]))

    # -------------------- 8. Возвращаем SHA1-дубли обратно в группы --------------------
    sha1_to_all_paths = {
        sha1: [x["path"] for x in group]
        for sha1, group in sha1_groups.items()
    }

    path_to_sha1 = {
        str(item["path"]): item["sha1"]
        for item in indexed_items
    }

    final_groups: list[list[Path]] = []
    for _, rep_group in root_to_rep_paths.items():
        expanded_group: list[Path] = []

        for rep_path in rep_group:
            rep_sha1 = path_to_sha1[str(rep_path)]
            expanded_group.extend(sha1_to_all_paths[rep_sha1])

        # убрать повторы, сохранив порядок
        seen = set()
        deduped_group = []
        for p in expanded_group:
            key = str(p)
            if key not in seen:
                seen.add(key)
                deduped_group.append(Path(p))

        final_groups.append(deduped_group)

    final_groups = sorted(final_groups, key=lambda g: (-len(g), [str(p) for p in g]))

    # -------------------- 9. Экспорт --------------------
    export_groups(final_groups, output_folder)

    elapsed = time.time() - t0
    logging.info(f"Готово. Групп найдено: {len(final_groups)}")
    logging.info(f"Общее время: {elapsed:.2f} сек")


if __name__ == "__main__":
    print("Begin")
    main()
    print("End")