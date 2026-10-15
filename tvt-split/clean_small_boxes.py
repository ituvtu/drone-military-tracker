#!/usr/bin/env python3
"""
Скрипт для фільтрації занадто малих рамок анотації в YOLO форматі.
Створює копію датасету (наприклад, v23 з v22) та виконує фільтрацію в ній,
не змінюючи оригінальні файли.
"""

import os
import logging
import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть залежність tqdm для відображення прогресу: pip install tqdm")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("BoxCleaner")


def get_threshold_for_resolution(width: int, height: int) -> int:
    """
    Повертає поріг відповідно до роздільної здатності.
    """
    if width == 1280 and height == 720:
        return 20
    elif width == 800 and height == 450:
        return 12.5
    elif width == 640 and height == 360:
        return 10
    else:
        raise ValueError(
            f"Непідтримувана роздільна здатність: {width}x{height}. "
            f"Підтримувані варіанти: 1280x720 (поріг 20px), 800x450 (поріг 12.5px), 640x360 (поріг 10px)."
        )


def auto_detect_resolution(dataset_dir: Path) -> Optional[Tuple[int, int]]:
    """
    Шукає перше зображення у папці images та визначає його роздільну здатність.
    """
    images_dir = dataset_dir / "images"
    search_dirs = [images_dir] if images_dir.exists() else []
    for split in ["train", "val", "test"]:
        split_img_dir = dataset_dir / split / "images"
        if split_img_dir.exists():
            search_dirs.append(split_img_dir)

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
    for s_dir in search_dirs:
        for ext in image_extensions:
            for img_path in s_dir.rglob(ext):
                if img_path.is_file():
                    try:
                        from PIL import Image
                        with Image.open(img_path) as img:
                            return img.width, img.height
                    except Exception:
                        pass
                    try:
                        import cv2
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            return img.shape[1], img.shape[0]
                    except Exception:
                        pass
    return None


def get_image_path(lbl_path: Path) -> Path:
    """
    Знаходить шлях до відповідного файлу зображення для файлу анотації.
    """
    parts = list(lbl_path.parts)
    try:
        labels_idx = len(parts) - 1 - parts[::-1].index("labels")
        parts[labels_idx] = "images"
        img_base_path = Path(*parts).with_suffix("")
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]:
            img_path = img_base_path.with_suffix(ext)
            if img_path.exists():
                return img_path
    except ValueError:
        pass
    return lbl_path


def delete_corresponding_image(lbl_path: Path) -> bool:
    """
    Видаляє файл зображення, що відповідає файлу анотації.
    """
    img_path = get_image_path(lbl_path)
    if img_path != lbl_path and img_path.exists():
        try:
            img_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Не вдалося видалити зображення {img_path}: {e}")
    return False


def has_boxes(lines_list: List[str]) -> bool:
    """
    Перевіряє, чи містить список рядків хоча б одну валідну розмітку YOLO.
    """
    for line in lines_list:
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                return True
            except ValueError:
                pass
    return False


def detect_dataset_structure(dataset_dir: Path) -> List[Tuple[str, Path]]:
    """
    Аналізує директорію та повертає список знайдених папок розмітки: (назва_спліту, шлях).
    """
    splits = ["train", "val", "test"]
    detected = []

    # 1. Сценарій А: labels/train, labels/val, labels/test
    labels_dir = dataset_dir / "labels"
    if labels_dir.exists():
        has_subsplits = False
        for split in splits:
            split_dir = labels_dir / split
            if split_dir.exists():
                detected.append((split, split_dir))
                has_subsplits = True
        
        # 2. Сценарій Б: пласка структура (всі анотації безпосередньо в labels/)
        if not has_subsplits:
            detected.append(("dataset", labels_dir))

    # 3. Сценарій В: train/labels, val/labels, test/labels
    if not detected:
        for split in splits:
            split_labels_dir = dataset_dir / split / "labels"
            if split_labels_dir.exists():
                detected.append((split, split_labels_dir))

    return detected


def clean_labels_in_split(
    split_name: str,
    split_dir: Path, 
    img_width: int, 
    img_height: int, 
    threshold: int, 
    dry_run: bool,
    edge_case_candidates: List[Path]
) -> Tuple[int, int, int, int, int]:
    """
    Обробляє окрему директорію з анотаціями.
    """
    total_files = 0
    total_boxes = 0
    deleted_boxes = 0
    deleted_frames_count = 0
    deleted_images_count = 0

    label_files = sorted([
        p for p in split_dir.rglob("*") 
        if p.is_file() and p.suffix.lower() == ".txt"
    ])
    
    if not label_files:
        return 0, 0, 0, 0, 0

    for lbl_path in tqdm(label_files, desc=f"Фільтрація {split_name.upper()}", unit="файл"):
        total_files += 1
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not has_boxes(lines):
                continue

            kept_lines = []
            has_edge_case_box = False

            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                
                parts = line_str.split()
                if len(parts) >= 5:
                    try:
                        float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        total_boxes += 1
                        
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])

                        w_px = w_norm * img_width
                        h_px = h_norm * img_height

                        if w_px < threshold or h_px < threshold:
                            deleted_boxes += 1
                        else:
                            kept_lines.append(line)
                            if round(w_px) == (threshold + 1) or round(h_px) == (threshold + 1):
                                has_edge_case_box = True
                    except ValueError:
                        kept_lines.append(line)
                else:
                    kept_lines.append(line)

            became_empty = not has_boxes(kept_lines)

            if became_empty:
                deleted_frames_count += 1
                if not dry_run:
                    try:
                        lbl_path.unlink()
                    except Exception as e:
                        logger.error(f"Помилка видалення анотації {lbl_path}: {e}")
                    
                    img_deleted = delete_corresponding_image(lbl_path)
                    if img_deleted:
                        deleted_images_count += 1
            else:
                if not dry_run:
                    with open(lbl_path, 'w', encoding='utf-8') as f:
                        f.writelines(kept_lines)
                
                if has_edge_case_box:
                    edge_case_candidates.append(lbl_path)

        except Exception as e:
            logger.error(f"Помилка обробки файлу {lbl_path}: {e}")

    return total_files, total_boxes, deleted_boxes, deleted_frames_count, deleted_images_count


def main():
    parser = argparse.ArgumentParser(description="Очищення занадто малих рамок розмітки в YOLO за довжиною сторін із копіюванням датасету")
    parser.add_argument("--src_dir", type=str, default="../dataset_v22", help="Шлях до вихідного папки з датасетом")
    parser.add_argument("--dst_dir", type=str, default="../dataset_v23_no_small_boxes", help="Шлях до нової папки, де буде створено очищений датасет")
    parser.add_argument("--img_width", type=int, default=0, help="Ширина зображення (якщо не вдалося визначити автоматично)")
    parser.add_argument("--img_height", type=int, default=0, help="Висота зображення (якщо не вдалося визначити автоматично)")
    parser.add_argument("--dry_run", action="store_true", help="Режим тестування без копіювання файлів та внесення фізичних змін")
    parser.add_argument("--show_samples", action="store_true", help="Показати до 10 випадкових кадрів з межовим розміром")

    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    
    if not src_dir.exists():
        logger.error(f"Вихідна директорія не існує: {src_dir}")
        return

    # Автовизначення роздільної здатності на основі оригінальних файлів
    detected_res = auto_detect_resolution(src_dir)
    if detected_res:
        img_width, img_height = detected_res
        logger.info(f"Автовизначено роздільну здатність датасету: {img_width}x{img_height}")
    else:
        img_width = args.img_width
        img_height = args.img_height
        logger.info(f"Не вдалося автовизначити роздільну здатність. Використовуємо задану вручну: {img_width}x{img_height}")

    try:
        threshold = get_threshold_for_resolution(img_width, img_height)
    except ValueError as e:
        logger.error(e)
        return

    # Перевірка структури вихідного датасету
    src_targets = detect_dataset_structure(src_dir)
    if not src_targets:
        logger.error(f"Не знайдено папок розмітки (labels) у вихідній директорії {src_dir}")
        return

    # Виконання копіювання у разі повноцінного запуску
    if not args.dry_run:
        logger.info(f"Копіювання датасету з {src_dir} до {dst_dir}...")
        if dst_dir.exists():
            logger.warning(f"Цільова директорія {dst_dir} вже існує. Файли можуть бути перезаписані.")
        try:
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            logger.info("Копіювання завершено успішно.")
        except Exception as e:
            logger.error(f"Не вдалося скопіювати директорію: {e}")
            return

    # Визначаємо робочу директорію залежно від режиму
    working_dir = src_dir if args.dry_run else dst_dir
    targets = detect_dataset_structure(working_dir)

    global_files = 0
    global_boxes = 0
    global_deleted = 0
    global_deleted_frames = 0
    global_deleted_images = 0
    edge_case_candidates = []

    logger.info("="*60)
    logger.info(f"ЗАПУСК ОЧИЩЕННЯ РАМОК")
    logger.info(f"Вихідна директорія (без змін): {src_dir.name}")
    logger.info(f"Цільова директорія            : {dst_dir.name}")
    logger.info(f"Визначена роздільна здатність : {img_width}x{img_height}")
    logger.info(f"Поріг фільтрації              : сторона < {threshold} px")
    if args.dry_run:
        logger.info("!!! ЗАПУЩЕНО В РЕЖИМІ DRY RUN (КОПІЮВАННЯ ТА ЗМІНИ НЕ ВИКОНУЮТЬСЯ) !!!")
    logger.info("="*60)

    for split_name, split_dir in targets:
        f_cnt, b_cnt, d_cnt, df_cnt, di_cnt = clean_labels_in_split(
            split_name=split_name,
            split_dir=split_dir,
            img_width=img_width,
            img_height=img_height,
            threshold=threshold,
            dry_run=args.dry_run,
            edge_case_candidates=edge_case_candidates
        )
        
        global_files += f_cnt
        global_boxes += b_cnt
        global_deleted += d_cnt
        global_deleted_frames += df_cnt
        global_deleted_images += di_cnt

    print("\n" + "="*50)
    print("             СТАТИСТИКА ОЧИЩЕННЯ")
    print("="*50)
    print(f"Оброблено текстових файлів        : {global_files}")
    print(f"Всього рамок проаналізовано       : {global_boxes}")
    print(f"Видалено занадто малих рамок      : {global_deleted}")
    print(f"Видалено текстових файлів (кадрів): {global_deleted_frames}")
    print(f"Видалено відповідних зображень     : {global_deleted_images}")
    print(f"Залишилося рамок у новому датасеті: {global_boxes - global_deleted}")
    print("="*50 + "\n")

    if args.show_samples:
        if edge_case_candidates:
            print("="*60)
            print("  ВИПАДКОВІ КАДРИ З МЕЖОВИМИ РОЗМІРАМИ (хоча б одна сторона = поріг + 1 px)")
            print("="*60)
            sample_size = min(10, len(edge_case_candidates))
            sampled_files = random.sample(edge_case_candidates, sample_size)
            for idx, lbl_p in enumerate(sampled_files, 1):
                img_p = get_image_path(lbl_p)
                print(f"{idx}. {img_p.resolve()}")
            print("="*60 + "\n")
        else:
            logger.info("Не знайдено жодного кадру, який би відповідав умові (сторона = поріг + 1 px).")


if __name__ == "__main__":
    main()