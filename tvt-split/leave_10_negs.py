#!/usr/bin/env python3
"""
Скрипт для оптимізації фонових (негативних) кадрів в існуючому датасеті (наприклад, dataset_v18).
Очищення виконується безпосередньо в папці датасету (in-place).
Залишає цільову частку негативних кадрів 10% із розподілом 70/15/15 (train/val/test).
Використовує метод диверсифікації dHash (Farthest Point Sampling).
"""

import os
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

try:
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть необхідні залежності: pip install Pillow numpy tqdm")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("DatasetPruner")


class ImageMetrics:
    @staticmethod
    def calculate_dhash(image_path: Path) -> int:
        """Обчислює dHash (різницевий хеш) зображення."""
        try:
            with Image.open(image_path) as img:
                img_gray = img.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
                arr = np.array(img_gray, dtype=np.uint8)
            diff = arr[:, :-1] > arr[:, 1:]
            flat_diff = diff.ravel()
            decimal_value = 0
            for bit in flat_diff:
                decimal_value = (decimal_value << 1) | int(bit)
            return decimal_value
        except Exception as e:
            logger.warning(f"Помилка обчислення хешу для {image_path}: {e}")
            return 0

    @staticmethod
    def get_blur_variance(image_path: Path) -> float:
        """Оцінює розмитість зображення через дисперсію Лапласіана."""
        try:
            with Image.open(image_path) as img:
                img_gray = img.convert('L')
                arr = np.array(img_gray, dtype=np.float32)
            laplacian = (
                arr[1:-1, 2:] + arr[1:-1, :-2] +
                arr[2:, 1:-1] + arr[:-2, 1:-1] -
                4 * arr[1:-1, 1:-1]
            )
            return float(laplacian.var())
        except Exception:
            return 1.0

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        return bin(hash1 ^ hash2).count('1')


def is_frame_annotated(label_path: Path) -> bool:
    """Перевіряє, чи містить файл розмітки YOLO хоча б один об'єкт."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    return True
    except Exception:
        pass
    return False


def select_diverse_negatives(
    candidates: List[Tuple[Path, Path, int, float]], 
    num_to_keep: int,
    blur_ratio: float = 0.15,
    blur_pool_percentile: float = 0.30
) -> List[Tuple[Path, Path, int, float]]:
    """
    Обирає найбільш різноманітні негативні кадри за стратифікованим алгоритмом FPS.
    Гарантує збереження заданої частки (blur_ratio) розмитих кадрів, які
    при цьому є візуально унікальними відносно чітких.
    """
    if len(candidates) <= num_to_keep or num_to_keep <= 0:
        return candidates[:num_to_keep]

    # 1. Сортуємо кандидатів за зростанням чіткості
    sorted_by_blur = sorted(candidates, key=lambda x: x[3])
    
    # 2. Визначаємо межу для поділу на "розмитий" та "чіткий" пули
    split_idx = int(len(sorted_by_blur) * blur_pool_percentile)
    blurry_pool = sorted_by_blur[:split_idx]
    sharp_pool = sorted_by_blur[split_idx:]
    
    # 3. Розраховуємо цільову кількість кадрів для кожного пулу
    num_blurry = int(num_to_keep * blur_ratio)
    num_sharp = num_to_keep - num_blurry
    
    # Запобіжні заходи на випадок недостатньої кількості кадрів у пулах
    if num_sharp > len(sharp_pool):
        num_sharp = len(sharp_pool)
        num_blurry = num_to_keep - num_sharp
        
    if num_blurry > len(blurry_pool):
        num_blurry = len(blurry_pool)
        num_sharp = num_to_keep - num_blurry

    # Внутрішня універсальна функція FPS-відбору
    def run_fps(
        pool: List[Tuple[Path, Path, int, float]], 
        count: int, 
        seed_hashes: Optional[List[int]] = None
    ) -> List[Tuple[Path, Path, int, float]]:
        if count <= 0 or not pool:
            return []
        if len(pool) <= count:
            return pool
            
        remaining = list(pool)
        kept = []
        
        if seed_hashes:
            min_distances = [
                min(ImageMetrics.hamming_distance(c[2], h) for h in seed_hashes)
                for c in remaining
            ]
        else:
            first_item = remaining.pop(-1)
            kept.append(first_item)
            min_distances = [
                ImageMetrics.hamming_distance(c[2], first_item[2])
                for c in remaining
            ]
            count -= 1
            
        for _ in range(count):
            if not remaining:
                break
            max_idx = max(range(len(remaining)), key=lambda i: min_distances[i])
            new_member = remaining.pop(max_idx)
            min_distances.pop(max_idx)
            kept.append(new_member)
            
            new_hash = new_member[2]
            for i in range(len(remaining)):
                dist = ImageMetrics.hamming_distance(remaining[i][2], new_hash)
                if dist < min_distances[i]:
                    min_distances[i] = dist
                    
        return kept

    # 4. Спочатку відбираємо унікальні чіткі кадри
    selected_sharp = run_fps(sharp_pool, num_sharp)
    
    # 5. Отримуємо хеші відібраних чітких кадрів для забезпечення крос-унікальності
    sharp_hashes = [item[2] for item in selected_sharp]
    
    # 6. Відбираємо розмиті кадри
    selected_blurry = run_fps(blurry_pool, num_blurry, seed_hashes=sharp_hashes)
    
    return selected_sharp + selected_blurry


def main():
    parser = argparse.ArgumentParser(description="Видалення зайвих фонових кадрів з існуючого датасету")
    parser.add_argument("--dataset_dir", type=str, default="../dataset_v18", help="Шлях до теки з існуючим датасетом (наприклад, dataset_v18)")
    parser.add_argument("--neg_ratio", type=float, default=0.10, help="Цільова частка негативів від загальної кількості кадрів (за замовчуванням 0.10, тобто 10%)")
    parser.add_argument("--dry_run", action="store_true", help="Режим тестування без фактичного видалення файлів з диска")
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir).resolve()
    
    if not dataset_dir.exists():
        logger.error(f"Вказану директорію датасету не знайдено: {dataset_dir}")
        return
        
    if not (0.0 <= args.neg_ratio < 1.0):
        logger.error("Параметр --neg_ratio має бути в діапазоні [0.0, 1.0).")
        return

    splits = ["train", "val", "test"]
    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    
    positives_by_split: Dict[str, List[Tuple[Path, Path]]] = {s: [] for s in splits}
    negatives_by_split: Dict[str, List[Tuple[Path, Path]]] = {s: [] for s in splits}
    
    logger.info(f"Аналіз структури та файлів у: {dataset_dir}")
    
    for split in splits:
        img_split_dir = dataset_dir / "images" / split
        if not img_split_dir.exists():
            continue
            
        img_files = sorted([f for f in img_split_dir.rglob("*") if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        
        for img_path in img_files:
            try:
                rel_path = img_path.relative_to(dataset_dir / "images" / split)
                label_path = dataset_dir / "labels" / split / rel_path.with_suffix(".txt")
            except Exception:
                label_path = dataset_dir / "labels" / split / img_path.with_suffix(".txt").name
                
            if is_frame_annotated(label_path):
                positives_by_split[split].append((img_path, label_path))
            else:
                negatives_by_split[split].append((img_path, label_path))

    total_positives = sum(len(positives_by_split[s]) for s in splits)
    if total_positives == 0:
        logger.error("У датасеті не знайдено анотованих (позитивних) кадрів. Перевірте структуру теки.")
        return
        
    # Цільові розрахунки негативів
    target_neg_total = int(total_positives * args.neg_ratio / (1.0 - args.neg_ratio))
    target_neg_by_split = {
        "train": int(target_neg_total * split_ratios["train"]),
        "val": int(target_neg_total * split_ratios["val"]),
        "test": target_neg_total - int(target_neg_total * split_ratios["train"]) - int(target_neg_total * split_ratios["val"])
    }
    
    print("\n" + "="*60)
    print(f"             ПЛАН ОЧИЩЕННЯ ДЛЯ {dataset_dir.name.upper()}")
    print("="*60)
    print(f"Анотовані кадри (залишаються)      : {total_positives}")
    print(f"Цільова к-сть негативів ({args.neg_ratio*100:.0f}% від усього): {target_neg_total}")
    print("-"*60)
    
    total_to_delete = 0
    for s in splits:
        curr_pos = len(positives_by_split[s])
        curr_neg = len(negatives_by_split[s])
        targ_neg = target_neg_by_split[s]
        reduction = max(0, curr_neg - targ_neg)
        total_to_delete += reduction
        print(f"Спліт {s.upper()}:")
        print(f"  -> Анотовані  : {curr_pos}")
        print(f"  -> Негативні  : {curr_neg} (планується залишити: {min(curr_neg, targ_neg)}, видалити: {reduction})")
        
    print("="*60)
    if args.dry_run:
        logger.info("!!! ЗАПУЩЕНО В РЕЖИМІ DRY RUN (БЕЗ ВИДАЛЕННЯ ФАЙЛІВ) !!!")
    print("="*60 + "\n")

    # Основний процес обробки
    deleted_images_count = 0
    deleted_labels_count = 0

    for split in splits:
        neg_list = negatives_by_split[split]
        targ_count = target_neg_by_split[split]
        
        if not neg_list:
            continue
            
        if len(neg_list) <= targ_count:
            logger.info(f"Спліт {split.upper()}: поточна кількість негативів ({len(neg_list)}) менша або рівна цільовій ({targ_count}). Очищення не потрібне.")
            continue

        # Збір dHash та чіткості для негативів спліту
        neg_candidates = []
        for img_p, lbl_p in tqdm(neg_list, desc=f"Аналіз dHash фону для {split.upper()}", unit="кадр"):
            h = ImageMetrics.calculate_dhash(img_p)
            b = ImageMetrics.get_blur_variance(img_p)
            neg_candidates.append((img_p, lbl_p, h, b))
            
        # Відбір негативів, які треба залишити
        selected_negatives = select_diverse_negatives(neg_candidates, targ_count)
        # Створюємо множину шляхів збережених зображень для швидкого пошуку
        kept_img_paths: Set[Path] = {item[0] for item in selected_negatives}
        
        # Видалення невідібраних кадрів
        for img_p, lbl_p, _, _ in tqdm(neg_candidates, desc=f"Очищення негативів {split.upper()}", unit="файл"):
            if img_p not in kept_img_paths:
                if not args.dry_run:
                    try:
                        if img_p.exists():
                            img_p.unlink()
                            deleted_images_count += 1
                        if lbl_p.exists():
                            lbl_p.unlink()
                            deleted_labels_count += 1
                    except Exception as e:
                        logger.error(f"Помилка видалення файлу {img_p.name}: {e}")
                else:
                    deleted_images_count += 1
                    if lbl_p.exists():
                        deleted_labels_count += 1

    if args.dry_run:
        logger.info(f"[DRY RUN] Очищення завершено. Було б видалено: {deleted_images_count} зображень та {deleted_labels_count} файлів розмітки.")
    else:
        logger.info(f"Очищення успішно завершено. Видалено: {deleted_images_count} зображень та {deleted_labels_count} порожніх файлів розмітки.")


if __name__ == "__main__":
    main()