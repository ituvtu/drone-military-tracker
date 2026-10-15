#!/usr/bin/env python3
"""
Об'єднаний скрипт для підготовки датасету YOLO:
1. Зчитує дані з вихідного сплітованого датасету.
2. Об'єднує всі класи в один (ID 0: military_object).
3. Паралельно змінює роздільну здатність зображень з 1280x720 на 640x360.
4. Оновлює конфігураційний файл data.yaml, прописуючи шлях для Kaggle.
"""

import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть залежності: pip install opencv-python numpy tqdm Pillow")

from concurrent.futures import ProcessPoolExecutor, as_completed

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("DatasetConverter")


def safe_imread(path: Path) -> np.ndarray:
    """
    Безпечне зчитування зображення (зокрема для усунення проблем із кодуванням шляхів).
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


def process_image_and_label_task(task: Dict[str, Any]) -> str:
    """
    Воркер для обробки однієї пари файлів (зображення + анотація).
    """
    try:
        src_img = Path(task['src_img'])
        dst_img = Path(task['dst_img'])
        src_lbl = Path(task['src_lbl'])
        dst_lbl = Path(task['dst_lbl'])
        
        # Створення цільових директорій
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        
        # Зчитування вихідного зображення
        img = safe_imread(src_img)
        if img is None:
            return "failed"
            
        h, w = img.shape[:2]
        status = "copied"
        
        # Масштабування, якщо роздільна здатність відповідає 1280x720
        if w == 1280 and h == 720:
            resized_img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            if safe_imwrite(dst_img, resized_img):
                status = "resized"
            else:
                return "failed"
        else:
            # Для зображень з іншою роздільною здатністю (наприклад, вже 640x360)
            if safe_imwrite(dst_img, img):
                status = "already_640" if (w == 640 and h == 360) else "skipped_other_res"
            else:
                return "failed"
                
        # Обробка та об'єднання класів у файлі анотації
        if src_lbl.exists():
            with open(src_lbl, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    parts[0] = '0'  # Заміна ID класу на 0 (military_object)
                    new_lines.append(" ".join(parts) + "\n")
                    
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        else:
            # Якщо розмітки немає, створюємо пустий файл для відповідності
            dst_lbl.touch()
            
        return status
    except Exception as e:
        logger.error(f"Помилка обробки пари файлів для {task.get('src_img')}: {e}")
        return "failed"


def update_yaml_to_single_class(src_yaml: Path, dst_yaml: Path):
    """
    Оновлює конфігурацію YAML для роботи з єдиним класом та прописує статичний шлях для Kaggle.
    """
    if not src_yaml.exists():
        logger.warning(f"Файл конфігурації {src_yaml} не знайдено.")
        return
    try:
        with open(src_yaml, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(dst_yaml, 'w', encoding='utf-8') as f:
            in_names_block = False
            for line in lines:
                # Оновлення шляху до папки датасету на Kaggle
                if line.strip().startswith("path:"):
                    f.write("path: /kaggle/input/datasets/ituvtu/military-dataset/dataset_v25_640\n")
                    continue
                
                # Оновлення кількості класів
                if line.strip().startswith("nc:"):
                    f.write("nc: 1\n")
                    continue
                
                # Заміна назв класів
                if line.strip().startswith("names:"):
                    f.write("names:\n  0: military_object\n")
                    in_names_block = True
                    continue
                
                # Пропуск старого списку назв
                if in_names_block:
                    if line.strip() and not line.startswith(" ") and not line.startswith("-"):
                        in_names_block = False
                    else:
                        continue
                
                f.write(line)
        logger.info(f"Оновлено конфігурацію YAML у: {dst_yaml}")
    except Exception as e:
        logger.error(f"Не вдалося оновити YAML конфігурацію: {e}")


def main():
    parser = argparse.ArgumentParser(description="Створення зменшеного датасету (640x360) з об'єднанням усіх класів в один")
    parser.add_argument("--src_dir", type=str, default="../dataset_v25_splitted", help="Шлях до оригінального датасету")
    parser.add_argument("--dst_dir", type=str, default="../dataset_v25_640", help="Шлях для створення нового датасету")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Кількість паралельних процесів (автоматично за замовчуванням)")
    
    args = parser.parse_args()
    
    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    
    if src_dir == dst_dir:
        logger.error("Джерело та призначення збігаються. Вкажіть різні директорії.")
        return
        
    if not src_dir.exists():
        logger.error(f"Папка вихідного датасету не знайдена: {src_dir}")
        return

    splits = ["train", "val", "test"]
    
    logger.info(f"Формування структури папок...")
    for split in splits:
        (dst_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Оновлення конфігурації YAML
    src_yaml = src_dir / "data.yaml"
    dst_yaml = dst_dir / "data.yaml"
    update_yaml_to_single_class(src_yaml, dst_yaml)
    
    # Копіювання файлу метаданих, якщо він є
    src_meta = src_dir / "dataset-metadata.json"
    if src_meta.exists():
        shutil.copy(src_meta, dst_dir / "dataset-metadata.json")

    tasks = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Сканування вихідних зображень та формування задач для паралельної обробки
    for split in splits:
        img_split_dir = src_dir / split / "images"
        if not img_split_dir.exists():
            logger.warning(f"Розділ {split.upper()} відсутній у вихідній директорії.")
            continue
            
        img_files = sorted([
            f for f in img_split_dir.rglob("*") 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ])
        
        for img_path in img_files:
            rel_path = img_path.relative_to(img_split_dir)
            dst_img_path = dst_dir / split / "images" / rel_path
            
            src_lbl_path = src_dir / split / "labels" / rel_path.with_suffix(".txt")
            dst_lbl_path = dst_dir / split / "labels" / rel_path.with_suffix(".txt")
            
            tasks.append({
                'src_img': str(img_path),
                'dst_img': str(dst_img_path),
                'src_lbl': str(src_lbl_path),
                'dst_lbl': str(dst_lbl_path)
            })

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.warning("Не знайдено кадрів для обробки.")
        return

    logger.info(f"Кількість кадрів для обробки: {total_tasks}")
    logger.info("Запуск паралельного процесу стиснення та конвертації класів...")

    stats = {
        "resized": 0,
        "already_640": 0,
        "skipped_other_res": 0,
        "failed": 0
    }

    # Використання пулу процесів для паралельних обчислень
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_image_and_label_task, task): task for task in tasks}
        
        for fut in tqdm(as_completed(futures), total=total_tasks, desc="Конвертація датасету", unit="кадр"):
            result = fut.result()
            stats[result] = stats.get(result, 0) + 1

    print("\n" + "="*50)
    print("             СТАТИСТИКА ОБРОБКИ")
    print("="*50)
    print(f"Оброблено файлів всього          : {total_tasks}")
    print(f"Змінено розмір (1280 -> 640)     : {stats.get('resized', 0)}")
    print(f"Мали роздільну здатність 640x360 : {stats.get('already_640', 0)}")
    print(f"Пропущено (інша роздільна здатність): {stats.get('skipped_other_res', 0)}")
    print(f"Помилки обробки                  : {stats.get('failed', 0)}")
    print("="*50 + "\n")
    logger.info(f"Очищений датасет збережено в: {dst_dir}")


if __name__ == "__main__":
    main()