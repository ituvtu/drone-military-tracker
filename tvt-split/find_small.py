#!/usr/bin/env python3
"""
Скрипт для пошуку та генерації посилань на зображення JPG,
які містять хоча б одну рамку площею < 80 px² (або заданого порогу).
Результати сортуються за спаданням площі знайденої рамки.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть залежність tqdm для відображення прогресу: pip install tqdm")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("LinkGenerator")


def find_corresponding_image(label_path: Path, dataset_dir: Path, split: str) -> Optional[Path]:
    """
    Знаходить шлях до зображення за відповідним файлом розмітки,
    враховуючи можливі підпапки та різні розширення файлів.
    """
    try:
        # Відносний шлях лейбла відносно labels/[split]
        rel_path = label_path.relative_to(dataset_dir / "labels" / split)
        base_img_path = dataset_dir / "images" / split / rel_path.with_suffix("")
        
        # Перевірка можливих розширень графічних файлів
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            img_path = base_img_path.with_suffix(ext)
            if img_path.exists():
                return img_path
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Пошук зображень із рамками малої площі")
    parser.add_argument("--dataset_dir", type=str, default="dataset_v17", help="Шлях до папки з датасетом")
    parser.add_argument("--img_width", type=int, default=640, help="Ширина зображення (за замовчуванням 640)")
    parser.add_argument("--img_height", type=int, default=360, help="Висота зображення (за замовчуванням 360)")
    parser.add_argument("--threshold_area_px", type=float, default=40.0, help="Поріг площі рамки в пікселях (шукає все, що < цього значення)")
    parser.add_argument("--output_file", type=str, default="small_boxes_links.txt", help="Файл для збереження списку посилань")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    labels_dir = dataset_dir / "labels"

    if not labels_dir.exists():
        logger.error(f"Директорія розмітки не знайдена: {labels_dir}")
        return

    splits = ["train", "val", "test"]
    # Зберігатимемо кортеж: (шлях_до_зображення, площа_найбільшої_з_малих_рамок_у_файлі)
    found_images: List[Tuple[Path, float]] = []

    logger.info(f"Пошук зображень із рамками площею < {args.threshold_area_px} px² у {dataset_dir.name}...")

    for split in splits:
        split_lbl_dir = labels_dir / split
        if not split_lbl_dir.exists():
            continue

        lbl_files = sorted(list(split_lbl_dir.rglob("*.txt")))
        
        for lbl_path in lbl_files:
            try:
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                has_small_box = False
                max_qualifying_area = -1.0

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            w = float(parts[3])
                            h = float(parts[4])

                            # Розрахунок площі в пікселях
                            w_px = w * args.img_width
                            h_px = h * args.img_height
                            area_px = w_px * h_px

                            # Перевіряємо, чи площа менша за поріг
                            if area_px < args.threshold_area_px:
                                has_small_box = True
                                # Шукаємо найбільшу рамку серед тих, які пройшли поріг
                                if area_px > max_qualifying_area:
                                    max_qualifying_area = area_px
                        except ValueError:
                            continue

                if has_small_box:
                    img_path = find_corresponding_image(lbl_path, dataset_dir, split)
                    if img_path:
                        found_images.append((img_path, max_qualifying_area))

            except Exception as e:
                logger.error(f"Помилка обробки файлу {lbl_path}: {e}")

    if not found_images:
        logger.info(f"Не знайдено жодного зображення з рамками площею < {args.threshold_area_px} px².")
        return

    # Сортування знайдених зображень за спаданням площі найбільшої рамки
    found_images.sort(key=lambda x: x[1], reverse=True)

    # Створення файлу звіту з посиланнями
    output_path = Path(args.output_file).resolve()
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Список зображень із рамками площею < {args.threshold_area_px} px² (Роздільна здатність: {args.img_width}x{args.img_height})\n")
            f.write(f"# Відсортовано за спаданням площі знайденої рамки (від найбільшої до найменшої)\n")
            f.write(f"# Всього знайдено файлів: {len(found_images)}\n\n")
            for img_p, area_px in found_images:
                file_url = img_p.as_uri()
                f.write(f"{file_url}  (Площа: {area_px:.1f} px²)\n")
                
        logger.info(f"Успішно знайдено {len(found_images)} зображень.")
        logger.info(f"Список клікабельних посилань збережено у файл: {output_path}")
        
        # Виводимо перші 15 посилань у консоль для швидкого доступу
        print("\n" + "="*70)
        print("          ПРИКЛАДИ ЗНАЙДЕНИХ ЗОБРАЖЕНЬ (Перші 15 шт. за спаданням площі):")
        print("="*70)
        for img_p, area_px in found_images[:15]:
            print(f"{img_p.as_uri()}  [{area_px:.1f} px²]")
        if len(found_images) > 15:
            print(f"... та ще {len(found_images) - 15} зображень (повний список у файлі {args.output_file})")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Не вдалося зберегти результати у файл: {e}")


if __name__ == "__main__":
    main()