#!/usr/bin/env python3
"""
Скрипт оптимізації фонових (негативних) кадрів для створення dataset_v16.
Досягає цільової частки негативних кадрів 10% із розподілом 70/15/15 (train/val/test).
Використовує метод диверсифікації dHash (Farthest Point Sampling).
"""

import os
import re
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
    raise ImportError("Встановіть залежності: pip install Pillow numpy tqdm")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("DatasetBalancer")


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

    :param candidates: Список кортежів (img_path, label_path, hash_val, blur_variance)
    :param num_to_keep: Скільки всього кадрів потрібно відібрати для вибірки
    :param blur_ratio: Яку частку (від 0.0 до 1.0) мають складати розмиті кадри у фінальному наборі
    :param blur_pool_percentile: Який відсоток найменш чітких кадрів вважати пулом "розмитих"
    """
    if len(candidates) <= num_to_keep or num_to_keep <= 0:
        return candidates[:num_to_keep]

    # 1. Сортуємо кандидатів за зростанням чіткості (від найменш чітких до найчіткіших)
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
            # Обчислюємо початкові відстані кандидатів до вже відібраних кадрів з іншого пулу
            min_distances = [
                min(ImageMetrics.hamming_distance(c[2], h) for h in seed_hashes)
                for c in remaining
            ]
        else:
            # Якщо початкових хешів немає, ініціалізуємо першим елементом.
            # Оскільки pool відсортований за зростанням чіткості, останній елемент (-1) є найчіткішим у своєму пулі.
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
            # Обираємо найбільш віддалений кадр за відстанню Хеммінга
            max_idx = max(range(len(remaining)), key=lambda i: min_distances[i])
            new_member = remaining.pop(max_idx)
            min_distances.pop(max_idx)
            kept.append(new_member)
            
            # Оновлюємо таблицю мінімальних відстаней для кандидатів, що залишилися
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
    
    # 6. Відбираємо розмиті кадри, унікальні як між собою, так і відносно вже відібраних чітких
    selected_blurry = run_fps(blurry_pool, num_blurry, seed_hashes=sharp_hashes)
    
    # Поєднуємо результати
    return selected_sharp + selected_blurry

def update_yaml_path(src_yaml: Path, dst_yaml: Path, new_dataset_path: Path):
    """Копіює конфігураційний yaml-файл із оновленням кореневого шляху."""
    try:
        with open(src_yaml, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(dst_yaml, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.strip().startswith("path:"):
                    f.write(f"path: {new_dataset_path.as_posix()}\n")
                else:
                    f.write(line)
    except Exception as e:
        logger.error(f"Не вдалося оновити конфігураційний YAML: {e}")


def main():
    parser = argparse.ArgumentParser(description="Балансування негативних кадрів для створення v16")
    parser.add_argument("--src_dir", type=str, required=True, help="Шлях до початкового датасету (наприклад, dataset_v15)")
    parser.add_argument("--dst_dir", type=str, required=True, help="Шлях для створення нового датасету (наприклад, dataset_v16)")
    parser.add_argument("--dry_run", action="store_true", help="Режим аналізу та проектування без фактичних змін на диску")
    parser.add_argument("--neg_ratio", type=float, default=0.10, help="Цільова частка негативів від загальної кількості кадрів (наприклад, 0.10)")
    
    args = parser.parse_args()
    
    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()
    
    if src_dir == dst_dir:
        logger.error("Джерело та призначення збігаються. Вкажіть новий шлях для збереження v16.")
        return
    if not (0.0 <= args.neg_ratio < 1.0):
        logger.error("Параметр --neg_ratio має бути в діапазоні [0.0, 1.0).")
        return
    splits = ["train", "val", "test"]
    split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    
    # Структура збору даних
    positives_by_split: Dict[str, List[Tuple[Path, Path]]] = {s: [] for s in splits}
    negatives_by_split: Dict[str, List[Tuple[Path, Path]]] = {s: [] for s in splits}
    
    logger.info(f"Аналіз вихідного датасету за адресою: {src_dir}")
    
    for split in splits:
        img_split_dir = src_dir / "images" / split
        if not img_split_dir.exists():
            continue
            
        img_files = sorted([f for f in img_split_dir.rglob("*") if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        
        for img_path in img_files:
            try:
                rel_path = img_path.relative_to(src_dir / "images" / split)
                label_path = src_dir / "labels" / split / rel_path.with_suffix(".txt")
            except Exception:
                label_path = src_dir / "labels" / split / img_path.with_suffix(".txt").name
                
            if is_frame_annotated(label_path):
                positives_by_split[split].append((img_path, label_path))
            else:
                negatives_by_split[split].append((img_path, label_path))

    # Кількісні підрахунки
    total_positives = sum(len(positives_by_split[s]) for s in splits)
    if total_positives == 0:
        logger.error("У вихідному датасеті не знайдено анотованих (позитивних) кадрів. Перевірте шляхи.")
        return
        
    # Цільові розрахунки
    target_neg_total = int(total_positives * args.neg_ratio / (1.0 - args.neg_ratio))
    target_neg_by_split = {
        "train": int(target_neg_total * split_ratios["train"]),
        "val": int(target_neg_total * split_ratios["val"]),
        "test": target_neg_total - int(target_neg_total * split_ratios["train"]) - int(target_neg_total * split_ratios["val"])
    }
    
    print("\n" + "="*55)
    print("             ПЛАН БАЛАНСУВАННЯ ДЛЯ V16")
    print("="*55)
    print(f"Анотовані кадри (залишаються без змін): {total_positives}")
    print(f"Цільова к-сть негативів ({args.neg_ratio*100:.0f}% від датасету): {target_neg_total}")
    print("-"*55)
    
    for s in splits:
        curr_pos = len(positives_by_split[s])
        curr_neg = len(negatives_by_split[s])
        targ_neg = target_neg_by_split[s]
        reduction = curr_neg - targ_neg
        print(f"Вибірка {s.upper()}:")
        print(f"  -> Анотовані  : {curr_pos}")
        print(f"  -> Негативні  : {curr_neg} (планується залишити: {targ_neg}, видалити: {reduction})")
        
    print("="*55 + "\n")
    
    if args.dry_run:
        logger.info("Виконано запуск у режимі dry_run. Зміни не записано.")
        return

    # Створення структури папок для v16
    logger.info(f"Створення нового каталогу датасету за шляхом: {dst_dir}")
    for split in splits:
        (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Копіювання конфігурації
    src_yaml = src_dir / "data.yaml"
    if src_yaml.exists():
        update_yaml_path(src_yaml, dst_dir / "data.yaml", dst_dir)
    
    # Копіювання метаданих за наявності
    src_meta = src_dir / "dataset-metadata.json"
    if src_meta.exists():
        shutil.copy(src_meta, dst_dir / "dataset-metadata.json")

    # Основний процес обробки
    for split in splits:
        logger.info(f"Обробка вибірки {split.upper()}...")
        
        # 1. Перенесення позитивних файлів
        pos_list = positives_by_split[split]
        for img_p, lbl_p in tqdm(pos_list, desc=f"Копіювання позитивів {split.upper()}", unit="файл"):
            rel_img = img_p.relative_to(src_dir / "images" / split)
            dst_img = dst_dir / "images" / split / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_p, dst_img)
            
            if lbl_p.exists():
                dst_lbl = dst_dir / "labels" / split / rel_img.with_suffix(".txt")
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(lbl_p, dst_lbl)

        # 2. Оцінка та диверсифіковане копіювання негативів
        neg_list = negatives_by_split[split]
        targ_count = target_neg_by_split[split]
        
        if not neg_list:
            continue
            
        # Збір хешів та метрик для негативів
        neg_candidates = []
        for img_p, lbl_p in tqdm(neg_list, desc=f"Аналіз dHash фону для {split.upper()}", unit="кадр"):
            h = ImageMetrics.calculate_dhash(img_p)
            b = ImageMetrics.get_blur_variance(img_p)
            neg_candidates.append((img_p, lbl_p, h, b))
            
        # Відбір диверсифікованих негативів
        selected_negatives = select_diverse_negatives(neg_candidates, targ_count)
        
        # Перенесення відібраних негативів
        for img_p, lbl_p, _, _ in tqdm(selected_negatives, desc=f"Запис негативів {split.upper()}", unit="файл"):
            rel_img = img_p.relative_to(src_dir / "images" / split)
            dst_img = dst_dir / "images" / split / rel_img
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_p, dst_img)
            
            # Створюємо чистий пустий файл анотації в YOLO
            dst_lbl = dst_dir / "labels" / split / rel_img.with_suffix(".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if lbl_p.exists():
                shutil.copy(lbl_p, dst_lbl)
            else:
                dst_lbl.touch()

    logger.info("Процес балансування успішно завершено. Створено v16.")


if __name__ == "__main__":
    main()