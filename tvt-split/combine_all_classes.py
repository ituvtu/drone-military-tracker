#!/usr/bin/env python3
"""
Скрипт для створення нового датасету з об'єднанням усіх класів в один єдиний (ID 0: military_object).
Адаптований під структуру: dataset_dir/split/images та dataset_dir/split/labels.
Використовує паралельні процеси та tqdm для прискорення копіювання і конвертації.
"""

import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть залежність tqdm для відображення прогресу: pip install tqdm")

from concurrent.futures import ProcessPoolExecutor, as_completed

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("ClassMerger")


def process_file_pair_task(task: Dict[str, Any]) -> bool:
    """
    Воркер для копіювання одного зображення та конвертації відповідного файлу розмітки.
    Повертає True у разі успіху.
    """
    try:
        src_img = Path(task['src_img'])
        dst_img = Path(task['dst_img'])
        src_lbl = Path(task['src_lbl'])
        dst_lbl = Path(task['dst_lbl'])
        
        # Створення батьківських директорій у новому датасеті
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        
        # Копіювання зображення
        shutil.copy2(src_img, dst_img)
        
        # Якщо розмітка існує — переписуємо всі класи на 0, якщо ні — створюємо порожній файл
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
                    parts[0] = '0'  # Заміна ID класу на 0
                    new_lines.append(" ".join(parts) + "\n")
                    
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        else:
            dst_lbl.touch()
            
        return True
    except Exception as e:
        logger.error(f"Помилка обробки файлу {task.get('src_img')}: {e}")
        return False


def update_yaml_to_single_class(src_yaml: Path, dst_yaml: Path, new_dataset_path: Path):
    """
    Оновлює конфігураційний файл YAML для роботи з єдиним класом та новим шляхом.
    """
    if not src_yaml.exists():
        logger.warning(f"Файл {src_yaml} не знайдено. Конфігурацію YAML не оновлено.")
        return
    try:
        with open(src_yaml, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(dst_yaml, 'w', encoding='utf-8') as f:
            in_names_block = False
            for line in lines:
                # Оновлення кореневого шляху до нової папки
                if line.strip().startswith("path:"):
                    f.write(f"path: {new_dataset_path.as_posix()}\n")
                    continue
                
                # Оновлення кількості класів
                if line.strip().startswith("nc:"):
                    f.write("nc: 1\n")
                    continue
                
                # Оновлення списку назв класів
                if line.strip().startswith("names:"):
                    f.write("names:\n  0: military_object\n")
                    in_names_block = True
                    continue
                
                # Пропускаємо старі класи зі списку
                if in_names_block:
                    if line.strip() and not line.startswith(" ") and not line.startswith("-"):
                        in_names_block = False
                    else:
                        continue
                
                f.write(line)
        logger.info(f"Оновлено конфігурацію YAML у: {dst_yaml}")
    except Exception as e:
        logger.error(f"Не вдалося оновити YAML файл: {e}")


def main():
    parser = argparse.ArgumentParser(description="Паралельне об'єднання класів під структуру dataset_dir/split/images")
    parser.add_argument("--src_dir", type=str, default="../dataset_v25_640", help="Шлях до папки з оригінальним датасетом")
    parser.add_argument("--dst_dir", type=str, default="../dataset_v25_640_merged", help="Шлях для створення нового датасету")
    parser.add_argument("--workers", type=int, default=None, help="Кількість паралельних процесів")
    
    args = parser.parse_args()
    
    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    
    if src_dir == dst_dir:
        logger.error("Джерело та призначення збігаються. Вкажіть різні директорії.")
        return
        
    if not src_dir.exists():
        logger.error(f"Папка джерела не існує: {src_dir}")
        return

    splits = ["train", "val", "test"]
    
    logger.info(f"Формування завдань для конвертації. Джерело: {src_dir} -> Призначення: {dst_dir}")
    
    # Створення структури папок для нового датасету
    for split in splits:
        (dst_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Оновлення yaml конфігурації
    src_yaml = src_dir / "data.yaml"
    dst_yaml = dst_dir / "data.yaml"
    update_yaml_to_single_class(src_yaml, dst_yaml, dst_dir)
    
    # Копіювання метаданих за наявності
    src_meta = src_dir / "dataset-metadata.json"
    if src_meta.exists():
        shutil.copy(src_meta, dst_dir / "dataset-metadata.json")

    tasks = []
    valid_extensions = {'.jpg', '.jpeg', '.png'}

    # Сканування файлів та побудова завдань
    for split in splits:
        img_split_dir = src_dir / split / "images"
        if not img_split_dir.exists():
            logger.warning(f"Розділ {split.upper()} не знайдено в {src_dir}")
            continue
            
        img_files = sorted([
            f for f in img_split_dir.rglob("*") 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ])
        
        for img_path in img_files:
            # Отримання відносного шляху для збереження вкладеності (якщо вона є)
            rel_path = img_path.relative_to(img_split_dir)
            
            dst_img_path = dst_dir / split / "images" / rel_path
            
            # Визначення шляхів для розмітки (.txt) відповідно до назви зображення
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
        logger.warning("Не знайдено жодного кадру для обробки.")
        return

    logger.info(f"Загальна кількість кадрів до обробки: {total_tasks}")
    logger.info("Запуск паралельного перенесення та конвертації даних...")

    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file_pair_task, task): task for task in tasks}
        
        for fut in tqdm(as_completed(futures), total=total_tasks, desc="Конвертація вибірок", unit="файл"):
            if fut.result():
                success_count += 1

    logger.info(f"Процес завершено. Успішно конвертовано {success_count} з {total_tasks} кадрів.")
    logger.info(f"Новий датасет створено у: {dst_dir}")


if __name__ == "__main__":
    main()