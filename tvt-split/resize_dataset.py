#!/usr/bin/env python3
"""
Скрипт для паралельного стиснення зображень датасету з роздільної здатності 1280x720 до 640x360.
Обробка відбувається "на місці" (overwrite) для знайдених зображень у вказаній папці.
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("DatasetResizer")


def safe_imread(path: Path) -> np.ndarray:
    """
    Безпечне зчитування зображення (усуває проблеми з кирилицею або спецсимволами в шляху на Windows).
    """
    try:
        with open(path, 'rb') as f:
            bytes_data = f.read()
        arr = np.frombuffer(bytes_data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"Не вдалося зчитати файл {path}: {e}")
        return None


def safe_imwrite(path: Path, img: np.ndarray) -> bool:
    """
    Безпечний запис зображення.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            with open(path, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        return False
    except Exception as e:
        logger.warning(f"Не вдалося записати файл {path}: {e}")
        return False


def resize_image_task(task_args: Dict[str, Any]) -> str:
    """
    Воркер для паралельної обробки одного зображення.
    """
    img_path = Path(task_args['path'])
    img = safe_imread(img_path)
    if img is None:
        return "failed"

    h, w = img.shape[:2]

    # Змінюємо розмір лише якщо поточна роздільна здатність рівно 1280х720
    if w == 1280 and h == 720:
        # cv2.INTER_AREA є оптимальним вибором для зменшення зображень
        resized_img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
        if safe_imwrite(img_path, resized_img):
            return "resized"
        else:
            return "failed"
    elif w == 640 and h == 360:
        return "already_resized"
    else:
        return "skipped"


def main():
    parser = argparse.ArgumentParser(description="Паралельне стиснення зображень 1280x720 до 640x360")
    parser.add_argument("--dataset_dir", type=str, default="dataset_v25_640", help="Шлях до папки з датасетом")
    parser.add_argument("--workers", type=int, default=None, help="Кількість паралельних процесів (за замовчуванням автоматично)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        logger.error(f"Папка датасету не знайдена: {dataset_dir}")
        return

    logger.info(f"Пошук зображень у папці: {dataset_dir.name}...")
    
    # Підтримувані розширення зображень
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []

    for file_path in dataset_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_paths.append(file_path)

    total_images = len(image_paths)
    if total_images == 0:
        logger.warning("Не знайдено жодного зображення в указаній папці.")
        return

    logger.info(f"Знайдено зображень для аналізу: {total_images}")
    logger.info("Запуск паралельного стиснення...")

    # Формуємо завдання для пулу процесів (передаємо рядки шляхів, оскільки вони легше серіалізуються)
    tasks = [{'path': str(p)} for p in image_paths]

    stats = {
        "resized": 0,
        "already_resized": 0,
        "skipped": 0,
        "failed": 0
    }

    # Використання ProcessPoolExecutor для оминання GIL під час декодування та стиснення зображень
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resize_image_task, task): task for task in tasks}

        for fut in tqdm(as_completed(futures), total=total_images, desc="Обробка кадрів", unit="кадр"):
            result = fut.result()
            stats[result] = stats.get(result, 0) + 1

    print("\n" + "="*50)
    print("             СТАТИСТИКА ОБРОБКИ")
    print("="*50)
    print(f"Знайдено зображень всього       : {total_images}")
    print(f"Успішно стиснуто (1280 -> 640)  : {stats['resized']}")
    print(f"Уже мали розмір 640x360         : {stats['already_resized']}")
    print(f"Пропущено (інша роздільна здатність): {stats['skipped']}")
    print(f"Помилки при обробці             : {stats['failed']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()