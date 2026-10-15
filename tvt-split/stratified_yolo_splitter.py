#!/usr/bin/env python3
"""
Система стратифікованого розподілу YOLO-датасетів (70/15/15 за кадрами).
Забезпечує цілісність епізодів, паралельне сканування/копіювання та балансування класів і фону.
"""

import os
import re
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть залежність tqdm для відображення прогресу: pip install tqdm")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("YoloSplitter")

CLASS_MAPPING = {
    0: "antiair",
    1: "bmp",
    2: "cars",
    3: "howitzer",
    4: "tank",
    5: "truck"
}


def parse_filename(stem: str) -> Tuple[str, int]:
    """Визначає назву епізоду та індекс кадру за назвою файлу."""
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
    """Визначає карту (географію) за назвою файлу."""
    match = re.match(r'^(m|map)_?(\d+)', filename, re.IGNORECASE)
    if match:
        return f"m{match.group(2)}"
    return "default"


def get_label_path_helper(img_path: Path) -> Path:
    """Визначає шлях до відповідного файлу анотації для зображення."""
    parts = list(img_path.parts)
    try:
        images_idx = len(parts) - 1 - parts[::-1].index("images")
        parts[images_idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name


def scan_image_worker(img_path: Path) -> Tuple[str, Path, int, Set[int]]:
    """Робоча функція для багатопроцесорного сканування та аналізу анотацій кадру."""
    stem = img_path.stem
    ep_name, _ = parse_filename(stem)
    lbl_path = get_label_path_helper(img_path)
    
    classes = set()
    is_neg = 1
    
    if lbl_path.exists() and lbl_path.stat().st_size > 0:
        try:
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_parts = line.strip().split()
                    if line_parts:
                        try:
                            cls_id = int(line_parts[0])
                            classes.add(cls_id)
                            is_neg = 0
                        except ValueError:
                            pass
        except Exception:
            pass
            
    return ep_name, img_path, is_neg, classes


def copy_file_worker(task: Tuple[Path, Path]) -> None:
    """Швидке копіювання файлів у пулі потоків."""
    src, dst = task
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:
        logger.error(f"Помилка копіювання {src.name} -> {dst.name}: {e}")


class DatasetSplitter:
    def __init__(self, src_dir: Path, dst_dir: Path, ratios: List[float], max_workers: Optional[int] = None):
        self.src_dir = Path(src_dir).resolve()
        self.dst_dir = Path(dst_dir).resolve()
        self.ratios = ratios
        self.max_workers = max_workers
        
        self.images_dir = self.src_dir / "images"
        self.labels_dir = self.src_dir / "labels"
        
        self.episodes_data: Dict[str, Dict] = {}
        self.maps: Set[str] = set()

    def get_label_path(self, img_path: Path) -> Path:
        """Повертає відповідний шлях до файлу анотації."""
        return get_label_path_helper(img_path)

    def scan_dataset(self):
        """Паралельно сканує зображення та збирає інформацію про класи."""
        logger.info(f"Сканування вихідної директорії: {self.images_dir}")
        img_files = sorted([
            p for p in self.images_dir.rglob("*") 
            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        ])
        
        if not img_files:
            raise FileNotFoundError(f"Не знайдено зображень у папці {self.images_dir}")
            
        logger.info("Паралельний аналіз анотацій кадрів...")
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(scan_image_worker, p): p for p in img_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Аналіз", unit="кадр"):
                try:
                    ep_name, img_path, is_neg, classes = future.result()
                    
                    if ep_name not in self.episodes_data:
                        self.episodes_data[ep_name] = {
                            "images": [],
                            "total_frames": 0,
                            "neg_frames": 0,
                            "pos_frames": 0,
                            "class_frames": {c: 0 for c in CLASS_MAPPING.keys()},
                            "map": get_map_name(img_path.name)
                        }
                        
                    ep = self.episodes_data[ep_name]
                    ep["images"].append(img_path)
                    ep["total_frames"] += 1
                    self.maps.add(ep["map"])
                    
                    if is_neg:
                        ep["neg_frames"] += 1
                    else:
                        ep["pos_frames"] += 1
                        for c in classes:
                            ep["class_frames"][c] += 1
                            
                except Exception as e:
                    logger.error(f"Помилка аналізу майстра: {e}")

    def execute_stratified_split(self) -> Dict[str, List[str]]:
        """Виконує двохетапний стратифікований розподіл епізодів за критерієм глобальної системної похибки."""
        splits: Dict[int, List[str]] = {0: [], 1: [], 2: []}
        
        # Позитивні та негативні епізоди
        pos_episodes = [k for k, v in self.episodes_data.items() if v["pos_frames"] > 0]
        neg_episodes = [k for k, v in self.episodes_data.items() if v["pos_frames"] == 0]
        
        # Глобальні лічильники
        global_class_counts = {c: sum(ep["class_frames"][c] for ep in self.episodes_data.values()) for c in CLASS_MAPPING.keys()}
        total_pos_frames = sum(ep["pos_frames"] for ep in self.episodes_data.values())
        
        # Сортування класів за рідкістю (rare-first)
        sorted_classes = sorted([c for c, count in global_class_counts.items() if count > 0], key=lambda c: global_class_counts[c])
        
        unallocated_pos = set(pos_episodes)
        
        # Лічильники стану розподілу
        split_class_frames = {c: [0, 0, 0] for c in CLASS_MAPPING.keys()}
        split_total_frames = [0, 0, 0]
        
        # Етап 1: Розподіл позитивних епізодів
        for target_class in sorted_classes:
            candidates = [k for k in unallocated_pos if self.episodes_data[k]["class_frames"][target_class] > 0]
            candidates.sort(key=lambda k: self.episodes_data[k]["class_frames"][target_class], reverse=True)
            
            for ep_key in candidates:
                if ep_key not in unallocated_pos:
                    continue
                    
                ep_data = self.episodes_data[ep_key]
                best_split = 0
                min_error = float('inf')
                
                # Пошук спліту, який мінімізує похибку всієї системи
                for candidate_s in range(3):
                    error = 0.0
                    
                    # 1. Покласова похибка по ВСІХ сплітах
                    for c in CLASS_MAPPING.keys():
                        global_c = global_class_counts[c]
                        if global_c == 0:
                            continue
                        for s in range(3):
                            curr = split_class_frames[c][s]
                            if s == candidate_s:
                                curr += ep_data["class_frames"][c]
                            ratio = curr / global_c
                            error += (ratio - self.ratios[s]) ** 2
                            
                    # 2. Похибка загального обсягу кадрів по ВСІХ сплітах
                    for s in range(3):
                        curr_pos = split_total_frames[s]
                        if s == candidate_s:
                            curr_pos += ep_data["total_frames"]
                        ratio_pos = curr_pos / total_pos_frames if total_pos_frames > 0 else 1.0
                        error += (ratio_pos - self.ratios[s]) ** 2
                        
                    if error < min_error:
                        min_error = error
                        best_split = candidate_s
                        
                splits[best_split].append(ep_key)
                unallocated_pos.remove(ep_key)
                
                # Оновлення метрик
                split_total_frames[best_split] += ep_data["total_frames"]
                for c in CLASS_MAPPING.keys():
                    split_class_frames[c][best_split] += ep_data["class_frames"][c]

        # Етап 2: Розподіл негативних епізодів (коригування географії та об'єму фону)
        global_neg_frames = sum(ep["neg_frames"] for ep in self.episodes_data.values())
        global_map_counts = {m: sum(1 for ep in self.episodes_data.values() if ep["map"] == m) for m in self.maps}
        
        split_neg_frames = [0, 0, 0]
        split_map_count = {m: [0, 0, 0] for m in self.maps}
        
        for s in range(3):
            for ep_key in splits[s]:
                m = self.episodes_data[ep_key]["map"]
                split_map_count[m][s] += 1
                split_neg_frames[s] += self.episodes_data[ep_key]["neg_frames"]

        for ep_key in neg_episodes:
            ep_data = self.episodes_data[ep_key]
            best_split = 0
            min_error = float('inf')
            
            for candidate_s in range(3):
                error = 0.0
                # Похибка негативного фону по ВСІХ сплітах
                for s in range(3):
                    curr_neg = split_neg_frames[s]
                    if s == candidate_s:
                        curr_neg += ep_data["neg_frames"]
                    ratio_neg = curr_neg / global_neg_frames if global_neg_frames > 0 else 1.0
                    error += (ratio_neg - self.ratios[s]) ** 2
                
                # Похибка карт (географії) по ВСІХ сплітах
                m_name = ep_data["map"]
                global_m = global_map_counts[m_name]
                for s in range(3):
                    curr_map = split_map_count[m_name][s]
                    if s == candidate_s:
                        curr_map += 1
                    ratio_map = curr_map / global_m if global_m > 0 else 1.0
                    error += (ratio_map - self.ratios[s]) ** 2
                    
                if error < min_error:
                    min_error = error
                    best_split = candidate_s
                    
            splits[best_split].append(ep_key)
            split_neg_frames[best_split] += ep_data["neg_frames"]
            split_map_count[ep_data["map"]][best_split] += 1

        return {
            "train": splits[0],
            "val": splits[1],
            "test": splits[2]
        }

    def apply_physical_transfer(self, splits: Dict[str, List[str]], dry_run: bool):
        """Фізичне копіювання файлів за допомогою ThreadPoolExecutor."""
        if dry_run:
            logger.info("[DRY RUN] Фізичне перенесення активоване у тестовому режимі. Запис не виконується.")
            return

        logger.info(f"Копіювання файлів у цільову директорію: {self.dst_dir}")
        copy_tasks: List[Tuple[Path, Path]] = []
        create_empty_labels: List[Path] = []
        
        for split_name, ep_list in splits.items():
            for ep_key in ep_list:
                ep_data = self.episodes_data[ep_key]
                for img_path in ep_data["images"]:
                    # Цільові шляхи зображень
                    dst_img = self.dst_dir / split_name / "images" / img_path.name
                    copy_tasks.append((img_path, dst_img))
                    
                    lbl_path = self.get_label_path(img_path)
                    dst_lbl = self.dst_dir / split_name / "labels" / img_path.with_suffix(".txt").name
                    
                    if lbl_path.exists():
                        copy_tasks.append((lbl_path, dst_lbl))
                    else:
                        create_empty_labels.append(dst_lbl)

        # Запуск багатопотокового копіювання зображень та існуючих анотацій
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
            list(tqdm(executor.map(copy_file_worker, copy_tasks), total=len(copy_tasks), desc="Копіювання", unit="файл"))

        # Безпечне створення порожніх анотацій для фонових кадрів на місці призначення
        if create_empty_labels:
            logger.info(f"Створення {len(create_empty_labels)} порожніх текстових файлів для фонових кадрів...")
            for path in create_empty_labels:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                except Exception as e:
                    logger.error(f"Не вдалося створити файл фону {path.name}: {e}")

        # Генерація YAML-файлу конфігурації
        self.generate_yolo_yaml()

    def generate_yolo_yaml(self):
        """Генерує класичний файл конфігурації data.yaml."""
        yaml_data = {
            "path": self.dst_dir.as_posix(),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(CLASS_MAPPING),
            "names": CLASS_MAPPING
        }
        
        yaml_path = self.dst_dir / "data.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Конфігурацію YOLO збережено в: {yaml_path}")

    def print_report(self, splits: Dict[str, List[str]]):
        """Розрахунок та друк детального статистичного звіту."""
        print("\n" + "=" * 60)
        print("          АНАЛІТИКА РОЗПОДІЛУ (FRAME-LEVEL STRATIFIED)")
        print("=" * 60)
        
        total_all_frames = sum(ep["total_frames"] for ep in self.episodes_data.values())
        total_pos_frames = sum(ep["pos_frames"] for ep in self.episodes_data.values())
        total_neg_frames = sum(ep["neg_frames"] for ep in self.episodes_data.values())

        print(f"Всього оброблено кадрів : {total_all_frames}")
        print(f"З них анотованих (pos) : {total_pos_frames}")
        print(f"З них фонових (neg)    : {total_neg_frames}")
        print("-" * 60)

        for s_name, ep_list in splits.items():
            s_frames = sum(self.episodes_data[ep]["total_frames"] for ep in ep_list)
            s_pos = sum(self.episodes_data[ep]["pos_frames"] for ep in ep_list)
            s_neg = sum(self.episodes_data[ep]["neg_frames"] for ep in ep_list)
            
            ratio_total = s_frames / total_all_frames * 100 if total_all_frames > 0 else 0
            
            print(f"Вибірка {s_name.upper()}:")
            print(f"  * Кадрів всього   : {s_frames} ({ratio_total:.2f}%)")
            print(f"  * Анотованих (pos): {s_pos}")
            print(f"  * Фонових (neg)   : {s_neg}")
            
        print("\n" + "-" * 30 + " Баланс по класах " + "-" * 30)
        for c_id, c_name in CLASS_MAPPING.items():
            global_c = sum(ep["class_frames"][c_id] for ep in self.episodes_data.values())
            if global_c == 0:
                continue
            
            shares = []
            for split_name in ["train", "val", "test"]:
                split_eps = splits[split_name]
                split_c = sum(self.episodes_data[ep]["class_frames"][c_id] for ep in split_eps)
                shares.append(f"{split_c / global_c * 100:.1f}%")
                
            print(f"  * Class [{c_name:10}] : {global_c:5} кадрів | Розподіл (Train/Val/Test): {' / '.join(shares)}")
            
        print("\n" + "-" * 30 + " Географія карт " + "-" * 30)
        for m in sorted(list(self.maps)):
            global_m = sum(1 for ep in self.episodes_data.values() if ep["map"] == m)
            shares = []
            for split_name in ["train", "val", "test"]:
                split_eps = splits[split_name]
                split_m = sum(1 for ep in split_eps if self.episodes_data[ep]["map"] == m)
                shares.append(f"{split_m}")
            print(f"  * Map [{m:8}] : Всього {global_m:3} епізодів | Розподіл (Train/Val/Test): {' / '.join(shares)}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Автоматичне покадрове розділення YOLO датасету 70/15/15")
    parser.add_argument("--src_dir", type=str, default="../dataset_v24_deduplicated", help="Шлях до вихідного YOLO датасету (де є папки images та labels)")
    parser.add_argument("--dst_dir", type=str, default="../dataset_v25_splitted", help="Шлях до нової папки для створення збалансованого датасету")
    parser.add_argument("--workers", type=int, default=None, help="Кількість паралельних процесів процесора для сканування")
    parser.add_argument("--dry_run", action="store_true", help="Запуск у режимі аналізу без фізичного копіювання")

    args = parser.parse_args()

    ratios = [0.70, 0.15, 0.15]

    splitter = DatasetSplitter(
        src_dir=Path(args.src_dir),
        dst_dir=Path(args.dst_dir),
        ratios=ratios,
        max_workers=args.workers
    )

    # Крок 1: Паралельний аналіз структури кадрів
    splitter.scan_dataset()

    # Крок 2: Оптимальний покадровий розподіл епізодів
    splits = splitter.execute_stratified_split()

    # Крок 3: Вивід статистики та аналітичного звіту
    splitter.print_report(splits)

    # Крок 4: Фізичний експорт даних з урахуванням режиму dry_run
    splitter.apply_physical_transfer(splits, args.dry_run)


if __name__ == "__main__":
    main()