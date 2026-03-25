import cv2
import numpy as np


def load_image(path, max_size=800):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть файл: {path}")

    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


def create_foreground_mask(img):
    """
    Убираем почти белый/светлый фон, чтобы круглая область анализировалась лучше.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # маска всего, что не слишком светлое
    mask = (gray < 245).astype(np.uint8) * 255

    # чуть сгладим маску
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def orb_similarity(img1, img2, debug=False):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mask1 = create_foreground_mask(img1)
    mask2 = create_foreground_mask(img2)

    orb = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10
    )

    kp1, des1 = orb.detectAndCompute(gray1, mask1)
    kp2, des2 = orb.detectAndCompute(gray2, mask2)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return {
            "same": False,
            "reason": "Слишком мало ключевых точек",
            "kp1": 0 if kp1 is None else len(kp1),
            "kp2": 0 if kp2 is None else len(kp2),
            "good_matches": 0,
            "inliers": 0,
            "angle": None,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # kNN matching
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return {
            "same": False,
            "reason": "Слишком мало хороших совпадений",
            "kp1": len(kp1),
            "kp2": len(kp2),
            "good_matches": len(good),
            "inliers": 0,
            "angle": None,
        }

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    if H is None or inlier_mask is None:
        return {
            "same": False,
            "reason": "Не удалось построить преобразование",
            "kp1": len(kp1),
            "kp2": len(kp2),
            "good_matches": len(good),
            "inliers": 0,
            "angle": None,
        }

    inliers = int(inlier_mask.ravel().sum())

    # Оценка угла поворота из матрицы преобразования
    angle = np.degrees(np.arctan2(H[1, 0], H[0, 0]))

    # Простое правило принятия решения
    same = inliers >= 15

    result = {
        "same": same,
        "reason": "OK" if same else "Недостаточно inliers",
        "kp1": len(kp1),
        "kp2": len(kp2),
        "good_matches": len(good),
        "inliers": inliers,
        "angle": angle,
    }

    if debug:
        draw = cv2.drawMatches(
            img1, kp1, img2, kp2, good[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite("debug_matches.jpg", draw)

    return result


if __name__ == "__main__":
    path1 = "/home/ubuntu/testlexa/image_40.jpg"
    path2 = "/home/ubuntu/testlexa/image_69.jpg"

    img1 = load_image(path1)
    img2 = load_image(path2)

    result = orb_similarity(img1, img2, debug=True)

    print("Результат проверки:")
    print(f"same         = {result['same']}")
    print(f"reason       = {result['reason']}")
    print(f"kp1          = {result['kp1']}")
    print(f"kp2          = {result['kp2']}")
    print(f"good_matches = {result['good_matches']}")
    print(f"inliers      = {result['inliers']}")
    print(f"angle        = {result['angle']}")
    if result["same"]:
        print("Картинки одинаковые")
    else:
        print("Картинки разные")