import hashlib
from pathlib import Path

import cv2
import numpy as np

from config import (
    PHASH_SIZE,
    PHASH_HASH_SIZE,
    DHASH_HASH_SIZE,
)


def sha1_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Считает SHA1 для файла.
    Нужен для поиска точных дублей байт-в-байт.
    """
    path = Path(path)
    h = hashlib.sha1()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def _load_gray(path: str | Path) -> np.ndarray:
    """
    Загружает изображение в grayscale.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось открыть изображение: {path}")
    return img


def _binary_array_to_uint64(bits: np.ndarray) -> int:
    """
    Преобразует массив битов (0/1, bool) в int.
    """
    bits = np.asarray(bits).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash(path: str | Path, hash_size: int = PHASH_HASH_SIZE, highfreq_factor: int = 4) -> int:
    """
    Перцептивный хэш pHash.

    Алгоритм:
    1. grayscale
    2. resize до (hash_size * highfreq_factor)
    3. DCT
    4. берем левый верхний low-frequency блок hash_size x hash_size
    5. сравниваем с медианой
    """
    img = _load_gray(path)

    size = hash_size * highfreq_factor
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = np.float32(img)

    dct = cv2.dct(img)
    dct_lowfreq = dct[:hash_size, :hash_size]

    # часто DC-компонент исключают из медианы, чтобы не искажать порог
    flat = dct_lowfreq.flatten()
    median = np.median(flat[1:]) if flat.size > 1 else np.median(flat)

    bits = dct_lowfreq > median
    return _binary_array_to_uint64(bits)


def dhash(path: str | Path, hash_size: int = DHASH_HASH_SIZE) -> int:
    """
    Difference hash (dHash).

    Алгоритм:
    1. grayscale
    2. resize до (hash_size + 1, hash_size)
    3. сравниваем соседние пиксели по горизонтали
    """
    img = _load_gray(path)

    resized = cv2.resize(
        img,
        (hash_size + 1, hash_size),
        interpolation=cv2.INTER_AREA
    )

    diff = resized[:, 1:] > resized[:, :-1]
    return _binary_array_to_uint64(diff)


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Расстояние Хэмминга между двумя int-хэшами.
    """
    return int((hash1 ^ hash2).bit_count())


if __name__ == "__main__":
    # Пример использования
    file1 = "images/example1.jpg"
    file2 = "images/example2.jpg"

    try:
        sha1_1 = sha1_file(file1)
        sha1_2 = sha1_file(file2)

        ph1 = phash(file1)
        ph2 = phash(file2)

        dh1 = dhash(file1)
        dh2 = dhash(file2)

        print("SHA1 equal:", sha1_1 == sha1_2)
        print("pHash distance:", hamming_distance(ph1, ph2))
        print("dHash distance:", hamming_distance(dh1, dh2))
    except Exception as e:
        print("Ошибка:", e)