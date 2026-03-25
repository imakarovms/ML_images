from __future__ import annotations

from itertools import combinations


def _safe_ratio(a: float, b: float) -> float:
    """
    Возвращает отношение большего к меньшему.
    Если одно из значений <= 0, возвращает inf.
    """
    if a <= 0 or b <= 0:
        return float("inf")
    return max(a, b) / min(a, b)


def _aspect_diff(a: float, b: float) -> float:
    """
    Относительная разница aspect ratio.
    """
    if a <= 0 or b <= 0:
        return float("inf")
    return abs(a - b) / max(a, b)


def is_candidate_pair(
    item1: dict,
    item2: dict,
    phash_max_distance: int,
    dhash_max_distance: int,
    max_mask_area_ratio: float,
    max_bbox_aspect_diff: float,
) -> tuple[bool, dict]:
    """
    Решает, стоит ли пару отправлять на точную ORB-проверку.

    Ожидает, что item содержит:
    - index
    - path
    - phash
    - dhash
    - mask_features
    """
    p_dist = (item1["phash"] ^ item2["phash"]).bit_count()
    d_dist = (item1["dhash"] ^ item2["dhash"]).bit_count()

    if p_dist > phash_max_distance:
        return False, {
            "reason": "phash_too_far",
            "phash_distance": p_dist,
            "dhash_distance": d_dist,
        }

    if d_dist > dhash_max_distance:
        return False, {
            "reason": "dhash_too_far",
            "phash_distance": p_dist,
            "dhash_distance": d_dist,
        }

    f1 = item1["mask_features"]
    f2 = item2["mask_features"]

    if not f1["has_object"] or not f2["has_object"]:
        return False, {
            "reason": "no_object_in_mask",
            "phash_distance": p_dist,
            "dhash_distance": d_dist,
        }

    area_ratio = _safe_ratio(f1["mask_area"], f2["mask_area"])
    if area_ratio > max_mask_area_ratio:
        return False, {
            "reason": "mask_area_ratio_too_large",
            "phash_distance": p_dist,
            "dhash_distance": d_dist,
            "mask_area_ratio": area_ratio,
        }

    aspect_diff = _aspect_diff(f1["bbox_aspect_ratio"], f2["bbox_aspect_ratio"])
    if aspect_diff > max_bbox_aspect_diff:
        return False, {
            "reason": "bbox_aspect_too_different",
            "phash_distance": p_dist,
            "dhash_distance": d_dist,
            "mask_area_ratio": area_ratio,
            "bbox_aspect_diff": aspect_diff,
        }

    return True, {
        "reason": "candidate",
        "phash_distance": p_dist,
        "dhash_distance": d_dist,
        "mask_area_ratio": area_ratio,
        "bbox_aspect_diff": aspect_diff,
    }


def generate_candidate_pairs(
    items: list[dict],
    phash_max_distance: int,
    dhash_max_distance: int,
    max_mask_area_ratio: float,
    max_bbox_aspect_diff: float,
) -> list[dict]:
    """
    Генерирует список candidate-пар для точной ORB-проверки.

    Каждый item в items должен содержать:
    - index
    - path
    - phash
    - dhash
    - mask_features

    Возвращает список словарей:
    {
        "i": int,
        "j": int,
        "path_i": str,
        "path_j": str,
        "phash_distance": int,
        "dhash_distance": int,
        "mask_area_ratio": float,
        "bbox_aspect_diff": float,
    }
    """
    candidates = []

    for item1, item2 in combinations(items, 2):
        ok, info = is_candidate_pair(
            item1=item1,
            item2=item2,
            phash_max_distance=phash_max_distance,
            dhash_max_distance=dhash_max_distance,
            max_mask_area_ratio=max_mask_area_ratio,
            max_bbox_aspect_diff=max_bbox_aspect_diff,
        )

        if not ok:
            continue

        candidates.append({
            "i": item1["index"],
            "j": item2["index"],
            "path_i": str(item1["path"]),
            "path_j": str(item2["path"]),
            "phash_distance": info["phash_distance"],
            "dhash_distance": info["dhash_distance"],
            "mask_area_ratio": info["mask_area_ratio"],
            "bbox_aspect_diff": info["bbox_aspect_diff"],
        })

    return candidates