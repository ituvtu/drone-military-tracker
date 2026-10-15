#!/usr/bin/env python3
"""
Комплексна система дедуплікації та балансування YOLO-датасетів з підтримкою паралельних обчислень.
1. Паралельне пре-обчислення візуальних метрик (dHash, Laplacian variance).
2. Внутрішньоепізодне очищення послідовностей з урахуванням специфічних порогів для класів та карт.
3. Проріджування фонових кадрів методом FPS до досягнення цільової частки у 10% (0.1111 * анотовані).
Працює шляхом копіювання вихідного датасету в нову директорію, зберігаючи оригінал без змін.
"""

import os
import re
import logging
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть залежності: pip install Pillow numpy tqdm")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("YoloBalancer")

# Мапінг класів YOLO відповідно до специфікації
CLASS_MAPPING = {
    0: "antiair",
    1: "bmp",
    2: "cars",
    3: "howitzer",
    4: "tank",
    5: "truck"
}


class ImageProcessor:
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
            logger.warning(f"Помилка обчислення dHash для {image_path.name}: {e}")
            return 0

    @staticmethod
    def get_blur_variance(image_path: Path) -> float:
        """Оцінює чіткість зображення через дисперсію Лапласіана."""
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


def compute_metrics_worker(img_path: Path) -> Tuple[Path, int, float]:
    """Робоча функція для обчислення метрик в окремому процесі."""
    h_val = ImageProcessor.calculate_dhash(img_path)
    b_val = ImageProcessor.get_blur_variance(img_path)
    return img_path, h_val, b_val


def precompute_metrics_parallel(img_paths: List[Path], max_workers: Optional[int] = None) -> Dict[Path, Tuple[int, float]]:
    """Паралельно обчислює dHash та показник розмитості для списку зображень."""
    cache = {}
    if not img_paths:
        return cache
    
    logger.info(f"Запуск паралельного аналізу {len(img_paths)} кадрів на {max_workers or 'усіх доступних'} ядрах...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_metrics_worker, p): p for p in img_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Аналіз зображень", unit="кадр"):
            img_path = futures[future]
            try:
                path, h_val, b_val = future.result()
                cache[path] = (h_val, b_val)
            except Exception as e:
                logger.error(f"Помилка обробки файлу {img_path.name}: {e}")
                cache[img_path] = (0, 1.0)
    return cache


def parse_filename(stem: str) -> Tuple[str, int]:
    """Розбирає назву файлу типу m1_s1_ep_001_f_001 на назву епізоду та номер кадру."""
    match = re.match(r'^(.+)_f_(\d+)$', stem, re.IGNORECASE)
    if match:
        return match.group(1), int(match.group(2))
    
    match = re.search(r'(\d+)$', stem)
    if match:
        frame_num = int(match.group(1))
        ep_name = stem[:match.start()].rstrip('_f-')
        if ep_name:
            return ep_name, frame_num
            
    return "default_episode", 0


def get_map_name(filename: str) -> str:
    """Визначає назву карти з префіксу назви файлу."""
    match = re.match(r'^(m|map)_?(\d+)', filename, re.IGNORECASE)
    if match:
        return f"m{match.group(2)}"
    return "default"


def get_classes_from_yolo(label_path: Path) -> Set[str]:
    """Зчитує класи об'єктів, наявних на кадрі."""
    classes = set()
    if not label_path.exists() or label_path.stat().st_size == 0:
        return classes
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        if cls_id in CLASS_MAPPING:
                            classes.add(CLASS_MAPPING[cls_id])
                    except ValueError:
                        pass
    except Exception:
        pass
    return classes


def is_frame_annotated(label_path: Path) -> bool:
    """Перевіряє, чи містить файл розмітки хоча б один об'єкт."""
    return len(get_classes_from_yolo(label_path)) > 0


def select_diverse_negatives(
    candidates: List[Tuple[Path, Path, int, float]], 
    num_to_keep: int,
    blur_ratio: float = 0.15,
    blur_pool_percentile: float = 0.30
) -> List[Tuple[Path, Path, int, float]]:
    """Обирає візуально унікальні негативні кадри за стратифікованим алгоритмом FPS."""
    if len(candidates) <= num_to_keep or num_to_keep <= 0:
        return candidates[:num_to_keep]

    sorted_by_blur = sorted(candidates, key=lambda x: x[3])
    split_idx = int(len(sorted_by_blur) * blur_pool_percentile)
    blurry_pool = sorted_by_blur[:split_idx]
    sharp_pool = sorted_by_blur[split_idx:]
    
    num_blurry = int(num_to_keep * blur_ratio)
    num_sharp = num_to_keep - num_blurry
    
    if num_sharp > len(sharp_pool):
        num_sharp = len(sharp_pool)
        num_blurry = num_to_keep - num_sharp
    if num_blurry > len(blurry_pool):
        num_blurry = len(blurry_pool)
        num_sharp = num_to_keep - num_blurry

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
                min(ImageProcessor.hamming_distance(c[2], h) for h in seed_hashes)
                for c in remaining
            ]
        else:
            first_item = remaining.pop(-1)
            kept.append(first_item)
            min_distances = [
                ImageProcessor.hamming_distance(c[2], first_item[2])
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
                dist = ImageProcessor.hamming_distance(remaining[i][2], new_hash)
                if dist < min_distances[i]:
                    min_distances[i] = dist
                    
        return kept

    selected_sharp = run_fps(sharp_pool, num_sharp)
    sharp_hashes = [item[2] for item in selected_sharp]
    selected_blurry = run_fps(blurry_pool, num_blurry, seed_hashes=sharp_hashes)
    
    return selected_sharp + selected_blurry


class DatasetProcessor:
    def __init__(self, dataset_dir: Path, pos_threshold: int = 3, neg_threshold: int = 14):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        
        self.metrics_cache: Dict[Path, Tuple[int, float]] = {}

        self.map_thresholds_internal = {
            "m2": 1,
        }
        self.class_thresholds_internal = {
            "howitzer": 1,
            "bmp": 2,
        }

    def get_label_path(self, img_path: Path) -> Path:
        """Повертає відповідний текстовий файл розмітки для зображення."""
        parts = list(img_path.parts)
        try:
            images_idx = len(parts) - 1 - parts[::-1].index("images")
            parts[images_idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        except ValueError:
            return img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name

    def resolve_threshold(self, img_path: Path) -> int:
        """Динамічно визначає поріг порівняння dHash для кадру."""
        filename = img_path.name
        map_name = get_map_name(filename)
        
        label_path = self.get_label_path(img_path)
        classes = get_classes_from_yolo(label_path)
        is_neg = len(classes) == 0

        resolved = self.neg_threshold if is_neg else self.pos_threshold

        if map_name == "m2":
            resolved = 1
        elif map_name in self.map_thresholds_internal:
            resolved = self.map_thresholds_internal[map_name]

        if not is_neg:
            class_thresholds = []
            for cls_name in classes:
                if cls_name in self.class_thresholds_internal:
                    class_thresholds.append(self.class_thresholds_internal[cls_name])
            if class_thresholds:
                resolved = min(resolved, min(class_thresholds))

        return resolved

    def detect_dataset_dirs(self) -> List[Tuple[str, Path, Path]]:
        """Знаходить папки images та labels, підтримує спліти та пласку структуру."""
        targets = []
        splits = ["train", "val", "test"]
        
        is_split = False
        for s in splits:
            img_dir = self.dataset_dir / s / "images"
            lbl_dir = self.dataset_dir / s / "labels"
            if img_dir.exists() and lbl_dir.exists():
                targets.append((s, img_dir, lbl_dir))
                is_split = True
                
        if not is_split:
            img_dir = self.dataset_dir / "images"
            lbl_dir = self.dataset_dir / "labels"
            if img_dir.exists() and lbl_dir.exists():
                targets.append(("dataset", img_dir, lbl_dir))
                
        return targets

    def run_deduplication(self, img_files: List[Path], split_name: str, dry_run: bool) -> int:
        """Етап 1: Внутрішньоепізодне очищення послідовних кадрів."""
        if not img_files:
            return 0

        episodes: Dict[str, List[Tuple[int, Path]]] = {}
        for path in img_files:
            ep_name, frame_idx = parse_filename(path.stem)
            if ep_name not in episodes:
                episodes[ep_name] = []
            episodes[ep_name].append((frame_idx, path))

        deleted_count = 0
        total_files = len(img_files)
        
        pbar = tqdm(total=total_files, desc=f"Дедуплікація {split_name.upper()}", unit="кадр")

        for ep_name, frames in episodes.items():
            sorted_frames = sorted(frames, key=lambda x: x[0])
            
            active_path: Optional[Path] = None
            active_hash: int = 0
            active_blur: float = 0.0

            for _, img_path in sorted_frames:
                h_val, b_val = self.metrics_cache.get(img_path, (0, 1.0))
                
                if active_path is None:
                    active_path = img_path
                    active_hash = h_val
                    active_blur = b_val
                    pbar.update(1)
                    continue

                dist = ImageProcessor.hamming_distance(active_hash, h_val)
                threshold = self.resolve_threshold(img_path)

                if dist <= threshold:
                    deleted_count += 1
                    if b_val > active_blur:
                        lbl_to_del = self.get_label_path(active_path)
                        if not dry_run:
                            try:
                                active_path.unlink()
                                if lbl_to_del.exists():
                                    lbl_to_del.unlink()
                            except FileNotFoundError:
                                pass
                        
                        active_path = img_path
                        active_hash = h_val
                        active_blur = b_val
                    else:
                        lbl_to_del = self.get_label_path(img_path)
                        if not dry_run:
                            try:
                                img_path.unlink()
                                if lbl_to_del.exists():
                                    lbl_to_del.unlink()
                            except FileNotFoundError:
                                pass
                else:
                    active_path = img_path
                    active_hash = h_val
                    active_blur = b_val
                    
                pbar.update(1)
                
        pbar.close()
        return deleted_count

    def run_negative_balancing(self, img_dir: Path, split_name: str, target_ratio: float, dry_run: bool) -> Tuple[int, int, int, int]:
        """Етап 2: Оптимізація негативних кадрів за FPS алгоритмом."""
        img_files = sorted([
            f for f in img_dir.rglob("*") 
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        positives = []
        negatives = []

        for img_path in img_files:
            lbl_path = self.get_label_path(img_path)
            if is_frame_annotated(lbl_path):
                positives.append(img_path)
            else:
                negatives.append(img_path)

        total_pos = len(positives)
        total_neg_initial = len(negatives)
        
        target_neg_count = int(round(target_ratio * total_pos))
        
        if total_neg_initial <= target_neg_count:
            logger.info(f"[{split_name.upper()}] Негативів менше/рівно норми ({total_neg_initial} <= {target_neg_count}). Пропускаємо.")
            return total_pos, total_neg_initial, total_neg_initial, 0

        candidates = []
        for img_p in negatives:
            lbl_p = self.get_label_path(img_p)
            h, b = self.metrics_cache.get(img_p, (0, 1.0))
            candidates.append((img_p, lbl_p, h, b))

        selected_negatives = select_diverse_negatives(candidates, target_neg_count)
        kept_paths = {item[0] for item in selected_negatives}

        deleted_neg_count = 0
        for img_p, lbl_p, _, _ in candidates:
            if img_p not in kept_paths:
                deleted_neg_count += 1
                if not dry_run:
                    try:
                        if img_p.exists():
                            img_p.unlink()
                        if lbl_p.exists():
                            lbl_p.unlink()
                    except Exception as e:
                        logger.error(f"Помилка видалення негативного кадру {img_p.name}: {e}")

        return total_pos, total_neg_initial, target_neg_count, deleted_neg_count


def main():
    parser = argparse.ArgumentParser(description="Паралельна дедуплікація та балансування YOLO датасетів із копіюванням результату")
    parser.add_argument("--src_dir", type=str, default="../dataset_v23_no_small_boxes", help="Шлях до вихідної теки датасету")
    parser.add_argument("--dst_dir", type=str, default="../dataset_v24_deduplicated", help="Шлях до нової теки для збалансованого датасету")
    parser.add_argument("--pos_threshold", type=int, default=3, help="Глобальний поріг dHash для позитивних кадрів")
    parser.add_argument("--neg_threshold", type=int, default=14, help="Глобальний поріг dHash для негативних кадрів")
    parser.add_argument("--neg_ratio", type=float, default=0.1111, help="Цільова пропорція негативів від позитивних кадрів")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Кількість паралельних процесів (за замовчуванням автоматично)")
    parser.add_argument("--dry_run", action="store_true", help="Режим тестування (без копіювання файлів та внесення фізичних змін)")

    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    dst_dir = Path(args.dst_dir).resolve()

    if not src_dir.exists():
        logger.error(f"Вихідну директорію {src_dir} не знайдено.")
        return

    # Визначаємо робочу директорію для аналізу
    working_dir = src_dir if args.dry_run else dst_dir

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

    processor = DatasetProcessor(
        dataset_dir=working_dir,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold
    )

    targets = processor.detect_dataset_dirs()
    if not targets:
        logger.error(f"Робоча папка не містить потрібної структури YOLO (images/labels). Перевірте {working_dir}")
        return

    logger.info("=" * 65)
    logger.info(f"ЗАПУСК ПАРАЛЕЛЬНОЇ СИСТЕМИ ДЕДУПЛІКАЦІЇ ТА БАЛАНСУВАННЯ")
    logger.info(f"Вихідна директорія (без змін): {src_dir.name}")
    logger.info(f"Цільова директорія            : {dst_dir.name}")
    logger.info(f"Конфігурація:")
    logger.info(f"  * Базові пороги dHash   : Позитивні = {args.pos_threshold} | Негативні = {args.neg_threshold}")
    logger.info(f"  * Поріг мапи m2 (захист): 1")
    logger.info(f"  * Клас howitzer (поріг) : 1")
    logger.info(f"  * Клас bmp (поріг)      : 2")
    logger.info(f"  * Частка негативів      : {args.neg_ratio * 100:.2f}% ({args.neg_ratio} від анотованих)")
    logger.info(f"  * Паралельні процеси    : {args.workers or 'Всі доступні ядра CPU'}")
    if args.dry_run:
        logger.info("!!! ЗАПУЩЕНО В РЕЖИМІ DRY RUN (БЕЗ ЗАПИСУ НА ДИСК ТА КОПІЮВАННЯ) !!!")
    logger.info("=" * 65)

    for split_name, img_dir, lbl_dir in targets:
        img_files = sorted([
            f for f in img_dir.rglob("*") 
            if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        if not img_files:
            continue

        # Фаза 0: Швидке паралельне обчислення метрик у робочій папці
        processor.metrics_cache = precompute_metrics_parallel(img_files, max_workers=args.workers)

        print(f"\n[Крок 1/2] Початок дедуплікації для {split_name.upper()}...")
        deleted_dup = processor.run_deduplication(img_files, split_name, args.dry_run)
        
        print(f"\n[Крок 2/2] Балансування негативних кадрів для {split_name.upper()}...")
        total_pos, init_neg, final_neg, deleted_neg = processor.run_negative_balancing(
            img_dir, split_name, args.neg_ratio, args.dry_run
        )

        total_final = total_pos + final_neg
        actual_ratio = (final_neg / total_final * 100) if total_final > 0 else 0.0

        print("\n" + "=" * 55)
        print(f"             СТАТИСТИКА ДЛЯ {split_name.upper()}")
        print("=" * 55)
        print(f" Збережено анотованих (позитивних) : {total_pos}")
        print(f" Видалено дублікатів (Етап 1)       : {deleted_dup}")
        print(f" Вихідна кількість негативів       : {init_neg}")
        print(f" Вилучено фону за балансом (Етап 2): {deleted_neg}")
        print(f" Фінальна кількість негативів       : {final_neg}")
        print(f" Загальний обсяг вибірки            : {total_final}")
        print(f" Частка фону у вибірці              : {actual_ratio:.2f}%")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()