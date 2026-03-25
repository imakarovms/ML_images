import cv2
import numpy as np
from pathlib import Path
import shutil
import logging
import time


# -------------------- НАСТРОЙКИ --------------------

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "grouped_images"
DEBUG_FOLDER = "debug_matches"

MAX_SIZE = 800
MIN_KEYPOINTS = 10
MIN_GOOD_MATCHES = 8
MIN_INLIERS = 15

# если True — сохранять картинки с отрисованными совпадениями
SAVE_DEBUG_MATCHES = False


# -------------------- ЛОГИРОВАНИЕ --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# -------------------- DSU / UNION-FIND --------------------

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# -------------------- ЗАГРУЗКА И ПРЕДОБРАБОТКА --------------------

def load_image(path, max_size=MAX_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть файл: {path}")

    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return img


def create_foreground_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # все, что не почти белый фон
    mask = (gray < 245).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = create_foreground_mask(img)

    orb = cv2.ORB_create(
        nfeatures=5000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
        fastThreshold=10
    )

    kp, des = orb.detectAndCompute(gray, mask)
    return gray, mask, kp, des


# -------------------- СРАВНЕНИЕ 2 КАРТИНОК --------------------

def compare_features(img1, kp1, des1, img2, kp2, des2, debug_path=None):
    if des1 is None or des2 is None or kp1 is None or kp2 is None:
        return {
            "same": False,
            "reason": "Нет дескрипторов",
            "kp1": 0 if kp1 is None else len(kp1),
            "kp2": 0 if kp2 is None else len(kp2),
            "good_matches": 0,
            "inliers": 0,
            "angle": None,
        }

    if len(kp1) < MIN_KEYPOINTS or len(kp2) < MIN_KEYPOINTS:
        return {
            "same": False,
            "reason": "Слишком мало ключевых точек",
            "kp1": len(kp1),
            "kp2": len(kp2),
            "good_matches": 0,
            "inliers": 0,
            "angle": None,
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
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
    angle = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))

    same = inliers >= MIN_INLIERS

    if SAVE_DEBUG_MATCHES and debug_path is not None:
        draw = cv2.drawMatches(
            img1, kp1, img2, kp2, good[:50], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(str(debug_path), draw)

    return {
        "same": same,
        "reason": "OK" if same else "Недостаточно inliers",
        "kp1": len(kp1),
        "kp2": len(kp2),
        "good_matches": len(good),
        "inliers": inliers,
        "angle": angle,
    }


# -------------------- ПОИСК ФАЙЛОВ --------------------

def find_images(input_folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


# -------------------- ГРУППИРОВКА --------------------

def group_images(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    start_time = time.time()

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    debug_folder = Path(DEBUG_FOLDER)

    if not input_folder.exists():
        logging.error(f"Папка не существует: {input_folder.resolve()}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_MATCHES:
        debug_folder.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_folder)
    if not image_paths:
        logging.error("В папке нет изображений")
        return

    logging.info(f"Найдено изображений: {len(image_paths)}")

    # загружаем и считаем признаки один раз
    data = []
    for path in image_paths:
        try:
            img = load_image(path)
            gray, mask, kp, des = extract_features(img)
            data.append({
                "path": path,
                "img": img,
                "kp": kp,
                "des": des
            })
            logging.info(
                f"Загружено: {path.name} | keypoints={0 if kp is None else len(kp)}"
            )
        except Exception as e:
            logging.warning(f"Пропуск {path}: {e}")

    n = len(data)
    if n == 0:
        logging.error("Не удалось загрузить ни одного изображения")
        return

    total_pairs = n * (n - 1) // 2
    logging.info(f"Всего пар для проверки: {total_pairs}")

    dsu = DSU(n)
    checked = 0
    matched = 0

    for i in range(n):
        for j in range(i + 1, n):
            checked += 1

            result = compare_features(
                data[i]["img"], data[i]["kp"], data[i]["des"],
                data[j]["img"], data[j]["kp"], data[j]["des"],
                debug_path=(
                    Path(DEBUG_FOLDER) / f"{data[i]['path'].stem}__{data[j]['path'].stem}.jpg"
                    if SAVE_DEBUG_MATCHES else None
                )
            )

            logging.info(
                f"[{checked}/{total_pairs}] "
                f"{data[i]['path'].name} vs {data[j]['path'].name} | "
                f"same={result['same']} | "
                f"reason={result['reason']} | "
                f"kp=({result['kp1']},{result['kp2']}) | "
                f"good={result['good_matches']} | "
                f"inliers={result['inliers']} | "
                f"angle={result['angle']}"
            )

            if result["same"]:
                dsu.union(i, j)
                matched += 1

    # собираем группы
    groups = {}
    for idx in range(n):
        root = dsu.find(idx)
        groups.setdefault(root, []).append(data[idx]["path"])

    # сортируем: сначала большие группы
    group_list = sorted(groups.values(), key=lambda g: (-len(g), [p.name for p in g]))

    # сохраняем
    for group_idx, group in enumerate(group_list, start=1):
        group_dir = output_folder / f"group_{group_idx:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        for file_path in group:
            shutil.copy2(file_path, group_dir / file_path.name)

    elapsed = time.time() - start_time

    logging.info("Готово")
    logging.info(f"Проверено пар: {checked}")
    logging.info(f"Найдено совпадений: {matched}")
    logging.info(f"Найдено групп: {len(group_list)}")
    logging.info(f"Время: {elapsed:.2f} сек")


if __name__ == "__main__":
    group_images(
        input_folder="images",
        output_folder="grouped_images"
    )