import cv2
import numpy as np
from pathlib import Path
import shutil
import logging
import time
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


# -------------------- НАСТРОЙКИ --------------------

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "parallel_grouped_images"

MAX_SIZE = 800
MIN_KEYPOINTS = 10
MIN_GOOD_MATCHES = 8
MIN_INLIERS = 15

MAX_WORKERS = max(1, os.cpu_count() - 1)


# -------------------- ЛОГИРОВАНИЕ --------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# -------------------- DSU --------------------

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


# -------------------- ФАЙЛОВЫЙ ХЭШ --------------------

def sha1_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# -------------------- ЗАГРУЗКА --------------------

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

    mask = (gray < 245).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def extract_features(path_str):
    path = Path(path_str)
    img = load_image(path)
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

    # KeyPoint нельзя удобно сериализовать между процессами,
    # поэтому сохраняем только координаты
    pts = None
    if kp is not None:
        pts = np.array([k.pt for k in kp], dtype=np.float32)

    return {
        "path": str(path),
        "img": img,
        "pts": pts,
        "des": des,
        "kp_count": 0 if kp is None else len(kp),
    }


# -------------------- СРАВНЕНИЕ ПАРЫ --------------------

def compare_pair(args):
    i, j, data_i, data_j = args

    des1 = data_i["des"]
    des2 = data_j["des"]
    pts1 = data_i["pts"]
    pts2 = data_j["pts"]

    kp1_count = data_i["kp_count"]
    kp2_count = data_j["kp_count"]

    result = {
        "i": i,
        "j": j,
        "same": False,
        "reason": "",
        "kp1": kp1_count,
        "kp2": kp2_count,
        "good_matches": 0,
        "inliers": 0,
        "angle": None,
    }

    if des1 is None or des2 is None or pts1 is None or pts2 is None:
        result["reason"] = "Нет дескрипторов"
        return result

    if kp1_count < MIN_KEYPOINTS or kp2_count < MIN_KEYPOINTS:
        result["reason"] = "Слишком мало ключевых точек"
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    src_pts = []
    dst_pts = []

    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
            src_pts.append(pts1[m.queryIdx])
            dst_pts.append(pts2[m.trainIdx])

    result["good_matches"] = len(good)

    if len(good) < MIN_GOOD_MATCHES:
        result["reason"] = "Слишком мало хороших совпадений"
        return result

    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or inlier_mask is None:
        result["reason"] = "Не удалось построить преобразование"
        return result

    inliers = int(inlier_mask.ravel().sum())
    angle = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))

    result["inliers"] = inliers
    result["angle"] = angle
    result["same"] = inliers >= MIN_INLIERS
    result["reason"] = "OK" if result["same"] else "Недостаточно inliers"

    return result


# -------------------- ПОИСК ФАЙЛОВ --------------------

def find_images(input_folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in input_folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])


# -------------------- ОСНОВНАЯ ЛОГИКА --------------------

def group_images(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    t0 = time.time()

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not input_folder.exists():
        logging.error(f"Папка не существует: {input_folder.resolve()}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_folder)
    logging.info(f"Найдено изображений: {len(image_paths)}")

    if not image_paths:
        logging.error("Изображения не найдены")
        return

    # -------- 1. Сначала ищем точные дубликаты файлов --------
    hash_groups = {}
    for p in image_paths:
        try:
            h = sha1_file(p)
            hash_groups.setdefault(h, []).append(p)
        except Exception as e:
            logging.warning(f"Не удалось посчитать SHA1 для {p}: {e}")

    exact_duplicate_count = sum(len(v) - 1 for v in hash_groups.values() if len(v) > 1)
    logging.info(f"Точных дубликатов файлов найдено: {exact_duplicate_count}")

    # Для ORB оставляем по одному представителю каждого sha1
    unique_paths = [group[0] for group in hash_groups.values()]
    logging.info(f"Уникальных файлов для ORB: {len(unique_paths)}")

    # -------- 2. Считаем признаки один раз --------
    t_feat0 = time.time()
    data = []

    for p in unique_paths:
        try:
            item = extract_features(str(p))
            data.append(item)
            logging.info(f"Признаки: {p.name} | kp={item['kp_count']}")
        except Exception as e:
            logging.warning(f"Пропуск {p}: {e}")

    n = len(data)
    if n == 0:
        logging.error("Не удалось извлечь признаки ни для одного изображения")
        return

    t_feat1 = time.time()
    logging.info(f"Извлечение признаков заняло: {t_feat1 - t_feat0:.2f} сек")

    # -------- 3. Параллельное сравнение всех пар --------
    total_pairs = n * (n - 1) // 2
    logging.info(f"Всего пар для ORB-сравнения: {total_pairs}")
    logging.info(f"Число worker-процессов: {MAX_WORKERS}")

    dsu = DSU(n)

    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((i, j, data[i], data[j]))

    t_cmp0 = time.time()
    done = 0
    matched = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(compare_pair, task) for task in tasks]

        for future in as_completed(futures):
            res = future.result()
            done += 1

            logging.info(
                f"[{done}/{total_pairs}] "
                f"{Path(data[res['i']]['path']).name} vs {Path(data[res['j']]['path']).name} | "
                f"same={res['same']} | reason={res['reason']} | "
                f"kp=({res['kp1']},{res['kp2']}) | "
                f"good={res['good_matches']} | inliers={res['inliers']} | angle={res['angle']}"
            )

            if res["same"]:
                dsu.union(res["i"], res["j"])
                matched += 1

    t_cmp1 = time.time()
    logging.info(f"Сравнение пар заняло: {t_cmp1 - t_cmp0:.2f} сек")

    # -------- 4. Собираем группы по уникальным файлам --------
    unique_groups = {}
    for idx in range(n):
        root = dsu.find(idx)
        unique_groups.setdefault(root, []).append(Path(data[idx]["path"]))

    # Возвращаем точные дубликаты обратно в группы
    final_groups = []

    for group in unique_groups.values():
        expanded = []
        for rep in group:
            rep_hash = sha1_file(rep)
            expanded.extend(hash_groups[rep_hash])
        final_groups.append(expanded)

    final_groups = sorted(final_groups, key=lambda g: (-len(g), [p.name for p in g]))

    # -------- 5. Сохраняем группы --------
    for idx, group in enumerate(final_groups, start=1):
        group_dir = output_folder / f"group_{idx:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        for file_path in group:
            shutil.copy2(file_path, group_dir / file_path.name)

    t1 = time.time()

    logging.info("Готово")
    logging.info(f"Групп найдено: {len(final_groups)}")
    logging.info(f"Совпавших пар: {matched}")
    logging.info(f"Общее время: {t1 - t0:.2f} сек")


if __name__ == "__main__":
    group_images("images", "grouped_images")