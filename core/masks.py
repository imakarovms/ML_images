from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.config import BACKGROUND_THRESHOLD


def create_foreground_mask(
    img: np.ndarray,
    background_threshold: int = BACKGROUND_THRESHOLD,
) -> np.ndarray:
    """
    Строит маску переднего плана.
    Все, что НЕ почти белый фон, считаем объектом.

    Возвращает uint8-маску со значениями 0 и 255.
    """
    if img is None:
        raise ValueError("img is None")

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    mask = (gray < background_threshold).astype(np.uint8) * 255

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def _largest_contour(mask: np.ndarray):
    """
    Возвращает самый большой контур на маске.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def compute_mask_features(mask: np.ndarray) -> dict:
    """
    Считает дешевые признаки маски объекта.

    Возвращает:
    - mask_area: площадь объекта в пикселях
    - image_area: площадь изображения
    - area_fraction: доля объекта
    - bbox_x, bbox_y, bbox_w, bbox_h
    - bbox_area
    - bbox_aspect_ratio
    - centroid_x, centroid_y
    - has_object
    """
    if mask is None:
        raise ValueError("mask is None")

    h, w = mask.shape[:2]
    image_area = int(h * w)

    nonzero = cv2.countNonZero(mask)
    has_object = nonzero > 0

    features = {
        "has_object": has_object,
        "mask_area": int(nonzero),
        "image_area": image_area,
        "area_fraction": float(nonzero / image_area) if image_area > 0 else 0.0,
        "bbox_x": 0,
        "bbox_y": 0,
        "bbox_w": 0,
        "bbox_h": 0,
        "bbox_area": 0,
        "bbox_aspect_ratio": 0.0,
        "centroid_x": 0.0,
        "centroid_y": 0.0,
    }

    if not has_object:
        return features

    contour = _largest_contour(mask)
    if contour is None:
        return features

    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = int(bw * bh)
    bbox_aspect_ratio = float(bw / bh) if bh > 0 else 0.0

    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(x + bw / 2.0)
        cy = float(y + bh / 2.0)

    features.update({
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
        "bbox_area": bbox_area,
        "bbox_aspect_ratio": bbox_aspect_ratio,
        "centroid_x": cx,
        "centroid_y": cy,
    })

    return features


def extract_mask_features_from_image(img: np.ndarray) -> dict:
    """
    Удобная обертка:
    картинка -> маска -> признаки маски
    """
    mask = create_foreground_mask(img)
    features = compute_mask_features(mask)
    return {
        "mask": mask,
        "mask_features": features,
    }


if __name__ == "__main__":
    path = Path("/home/ubuntu/testlexa/image_40.jpg")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Не удалось открыть {path}")
    else:
        result = extract_mask_features_from_image(img)
        print(result["mask_features"])
        cv2.imwrite("debug_mask.jpg", result["mask"])