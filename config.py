from pathlib import Path
import os


# -------------------- PATHS --------------------

INPUT_FOLDER = Path("images")
OUTPUT_FOLDER = Path("grouped_images")
DEBUG_FOLDER = Path("debug")


# -------------------- PARALLELISM --------------------

CPU_COUNT = os.cpu_count() or 1
MAX_WORKERS = max(1, CPU_COUNT - 1)


# -------------------- IMAGE LOADING --------------------

MAX_IMAGE_SIZE = 800
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# -------------------- HASHING --------------------

# Максимально допустимое расстояние Хэмминга для pHash
PHASH_MAX_DISTANCE = 20

# Максимально допустимое расстояние Хэмминга для dHash
DHASH_MAX_DISTANCE = 20

# Размер уменьшенного изображения для pHash
PHASH_SIZE = 32
PHASH_HASH_SIZE = 8

# Размер уменьшенного изображения для dHash
DHASH_HASH_SIZE = 8


# -------------------- MASK / OBJECT FILTERS --------------------

# Порог "почти белого" фона
BACKGROUND_THRESHOLD = 245

# Насколько может отличаться площадь объекта
# Например 2.0 значит один объект может быть максимум в 4 раза больше другого
MAX_MASK_AREA_RATIO = 4.0

# Насколько может отличаться aspect ratio bounding box
# Например 0.35 означает допустимую относительную разницу 60%
MAX_BBOX_ASPECT_DIFF = 0.60


# -------------------- ORB --------------------

ORB_NFEATURES = 5000
ORB_SCALE_FACTOR = 1.2
ORB_NLEVELS = 8
ORB_EDGE_THRESHOLD = 15
ORB_PATCH_SIZE = 31
ORB_FAST_THRESHOLD = 10

MIN_KEYPOINTS = 10
MIN_GOOD_MATCHES = 7
MIN_INLIERS = 10

# Порог RANSAC для homography
RANSAC_REPROJ_THRESHOLD = 5.0


# -------------------- LOGGING --------------------

LOG_LEVEL = "INFO"
LOG_EVERY_N_PAIRS = 100
SAVE_DEBUG_MATCHES = False