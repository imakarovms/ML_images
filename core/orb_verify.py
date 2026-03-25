#from future import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import (
    MAX_IMAGE_SIZE,
    ORB_NFEATURES,
    ORB_SCALE_FACTOR,
    ORB_NLEVELS,
    ORB_EDGE_THRESHOLD,
    ORB_PATCH_SIZE,
    ORB_FAST_THRESHOLD,
    MIN_KEYPOINTS,
    MIN_GOOD_MATCHES,
    MIN_INLIERS,
    RANSAC_REPROJ_THRESHOLD,
)
from core.masks import create_foreground_mask


def load_image(path: str | Path, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть файл: {path}")

    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)

    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    return img


def build_orb():
    return cv2.ORB_create(
        nfeatures=ORB_NFEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_NLEVELS,
        edgeThreshold=ORB_EDGE_THRESHOLD,
        patchSize=ORB_PATCH_SIZE,
        fastThreshold=ORB_FAST_THRESHOLD,
    )


def extract_orb_features(path: str | Path) -> dict:
    """
    Возвращает только pickle-friendly данные.
    """
    path = Path(path)
    img = load_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = create_foreground_mask(img)

    orb = build_orb()
    keypoints, descriptors = orb.detectAndCompute(gray, mask)

    points = None
    if keypoints is not None and len(keypoints) > 0:
        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return {
        "path": str(path),
        "points": points,
        "descriptors": descriptors,
        "kp_count": 0 if keypoints is None else len(keypoints),
    }


def verify_pair_orb(
    feat1: dict,
    feat2: dict,
    min_keypoints: int = MIN_KEYPOINTS,
    min_good_matches: int = MIN_GOOD_MATCHES,
    min_inliers: int = MIN_INLIERS,
    min_inlier_ratio: float = 0.0,
    ransac_reproj_threshold: float = RANSAC_REPROJ_THRESHOLD,
    save_debug: bool = False,
    debug_path: str | Path | None = None,
) -> dict:
    pts1 = feat1["points"]
    pts2 = feat2["points"]
    des1 = feat1["descriptors"]
    des2 = feat2["descriptors"]

    result = {
        "same": False,
        "reason": "",
        "path1": feat1["path"],
        "path2": feat2["path"],
        "kp1": feat1["kp_count"],
        "kp2": feat2["kp_count"],
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "angle": None,
        "homography": None,
    }

    if des1 is None or des2 is None or pts1 is None or pts2 is None:
        result["reason"] = "Нет дескрипторов"
        return result

    if feat1["kp_count"] < min_keypoints or feat2["kp_count"] < min_keypoints:
        result["reason"] = "Слишком мало ключевых точек"
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    src_points = []
    dst_points = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            src_points.append(pts1[m.queryIdx])
            dst_points.append(pts2[m.trainIdx])

    result["good_matches"] = len(good_matches)

    if len(good_matches) < min_good_matches:
        result["reason"] = "Слишком мало хороших совпадений"
        return result

    src_points = np.array(src_points, dtype=np.float32).reshape(-1, 1, 2)
    dst_points = np.array(dst_points, dtype=np.float32).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        ransac_reproj_threshold,
    )

    if H is None or inlier_mask is None:
        result["reason"] = "Не удалось построить homography"
        return result

    inliers = int(inlier_mask.ravel().sum())
    angle = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))

    inlier_ratio = inliers / len(good_matches) if good_matches else 0.0
    result["inliers"] = inliers
    result["inlier_ratio"] = inlier_ratio
    result["angle"] = angle
    result["homography"] = H
    result["same"] = inliers >= min_inliers and inlier_ratio >= min_inlier_ratio
    if inliers < min_inliers:
        result["reason"] = "Недостаточно inliers"
    elif inlier_ratio < min_inlier_ratio:
        result["reason"] = "Низкий inlier ratio"
    else:
        result["reason"] = "OK"

    return result
