#!/usr/bin/env python3
"""
Комплексна система аналізу та порівняння вибірок (train, val, test) YOLO-датасетів.
Проводить дослідження схожості розподілів класів, витоку епізодів/карт та 
візуальної близькості кадрів за допомогою dHash (з обмеженням порівняння в межах однієї карти).
Працює виключно в режимі читання (без модифікації даних).
"""

import os
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
except ImportError:
    raise ImportError("Встановіть залежності: pip install Pillow numpy tqdm")

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("DatasetComparator")

# Стандартний мапінг класів (якщо відсутній data.yaml)
DEFAULT_CLASS_MAPPING = {
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
    """Розбирає назву файлу на назву епізоду та номер кадру."""
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


def load_class_mapping_from_yaml(yaml_path: Path) -> Dict[int, str]:
    """Зчитує мапінг класів з data.yaml, інакше повертає стандартний."""
    mapping = DEFAULT_CLASS_MAPPING.copy()
    if not yaml_path.exists():
        return mapping
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.splitlines()
        in_names = False
        parsed_mapping = {}
        for line in lines:
            line_strip = line.strip()
            if line_strip.startswith("names:"):
                suffix = line_strip[6:].strip()
                if suffix.startswith("[") and suffix.endswith("]"):
                    elements = [x.strip("'\" ") for x in suffix[1:-1].split(",")]
                    for idx, name in enumerate(elements):
                        parsed_mapping[idx] = name
                    break
                elif suffix.startswith("{") and suffix.endswith("}"):
                    elements = suffix[1:-1].split(",")
                    for el in elements:
                        if ":" in el:
                            k, v = el.split(":")
                            parsed_mapping[int(k.strip())] = v.strip("'\" ")
                    break
                else:
                    in_names = True
                    continue
            
            if in_names:
                if line.startswith(" ") or line.startswith("\t") or line_strip.startswith("-"):
                    match_dict = re.match(r'^\s*(\d+)\s*:\s*(.+)$', line_strip)
                    if match_dict:
                        parsed_mapping[int(match_dict.group(1))] = match_dict.group(2).strip("'\" ")
                    else:
                        match_list = re.match(r'^\s*-\s*(.+)$', line_strip)
                        if match_list:
                            idx = len(parsed_mapping)
                            parsed_mapping[idx] = match_list.group(1).strip("'\" ")
                else:
                    in_names = False
        if parsed_mapping:
            return parsed_mapping
    except Exception as e:
        logger.warning(f"Не вдалося зчитати класи з YAML ({e}), використовується стандартний мапінг.")
    return mapping


def get_label_classes(label_path: Path, class_mapping: Dict[int, str]) -> List[str]:
    """Повертає список класів об'єктів з YOLO файлу розмітки."""
    classes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return classes
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cls_id = int(parts[0])
                        if cls_id in class_mapping:
                            classes.append(class_mapping[cls_id])
                    except ValueError:
                        pass
    except Exception:
        pass
    return classes


def calculate_cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Обчислює косинусну схожість двох розподілів."""
    all_keys = set(v1.keys()).union(set(v2.keys()))
    vec1 = np.array([v1.get(k, 0.0) for k in all_keys])
    vec2 = np.array([v2.get(k, 0.0) for k in all_keys])
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def analyze_split(
    img_dir: Path, 
    lbl_dir: Path, 
    class_mapping: Dict[int, str]
) -> Tuple[List[Path], Dict[str, Any]]:
    """Повний збір статистичних характеристик для одного спліту."""
    img_files = sorted([
        f for f in img_dir.rglob("*") 
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])
    
    stats = {
        "total_images": len(img_files),
        "annotated_count": 0,
        "background_count": 0,
        "class_counts": {name: 0 for name in class_mapping.values()},
        "total_boxes": 0,
        "episodes": set(),
        "maps": set(),
        "files_by_episode": {}
    }
    
    for img_path in img_files:
        rel_path = img_path.relative_to(img_dir)
        lbl_path = lbl_dir / rel_path.with_suffix(".txt")
        
        # Аналіз класів
        classes = get_label_classes(lbl_path, class_mapping)
        if classes:
            stats["annotated_count"] += 1
            stats["total_boxes"] += len(classes)
            for cls in classes:
                stats["class_counts"][cls] = stats["class_counts"].get(cls, 0) + 1
        else:
            stats["background_count"] += 1
            
        # Епізоди та карти
        ep_name, _ = parse_filename(img_path.stem)
        map_name = get_map_name(img_path.name)
        
        stats["episodes"].add(ep_name)
        stats["maps"].add(map_name)
        
        if ep_name not in stats["files_by_episode"]:
            stats["files_by_episode"][ep_name] = []
        stats["files_by_episode"][ep_name].append(img_path)
        
    return img_files, stats


def evaluate_map_constrained_similarity(
    files1: List[Path],
    hashes1: List[int],
    files2: List[Path],
    hashes2: List[int],
    near_dup_threshold: int = 3,
    batch_size: int = 500
) -> Tuple[int, int, float, int]:
    """
    Порівнює візуальну схожість ЛИШЕ між кадрами однієї й тієї самої карти.
    Повертає:
      - Кількість точних дублікатів (відстань = 0)
      - Кількість близьких кадрів (відстань <= threshold)
      - Середню мінімальну відстань Хеммінга (лише для кадрів спільних карт)
      - Кількість кадрів з першої вибірки, що мають спільну карту в другій вибірці
    """
    # Групуємо хеші за назвами карт
    map_to_hashes1 = {}
    for img_path, h_val in zip(files1, hashes1):
        m_name = get_map_name(img_path.name)
        if m_name not in map_to_hashes1:
            map_to_hashes1[m_name] = []
        map_to_hashes1[m_name].append(h_val)
        
    map_to_hashes2 = {}
    for img_path, h_val in zip(files2, hashes2):
        m_name = get_map_name(img_path.name)
        if m_name not in map_to_hashes2:
            map_to_hashes2[m_name] = []
        map_to_hashes2[m_name].append(h_val)
        
    common_maps = set(map_to_hashes1.keys()).intersection(set(map_to_hashes2.keys()))
    
    if not common_maps:
        return 0, 0, 0.0, 0
        
    total_exact = 0
    total_near = 0
    all_min_distances = []
    total_compared_frames = 0
    
    for m_name in common_maps:
        h1_list = map_to_hashes1[m_name]
        h2_list = map_to_hashes2[m_name]
        
        # Конвертація у бінарні матриці
        arr1 = np.array([[int(x) for x in f"{h:064b}"] for h in h1_list], dtype=bool)
        arr2 = np.array([[int(x) for x in f"{h:064b}"] for h in h2_list], dtype=bool)
        
        for i in range(0, len(arr1), batch_size):
            batch = arr1[i:i+batch_size]
            diffs = (batch[:, None, :] != arr2[None, :, :]).sum(axis=2)
            
            batch_mins = diffs.min(axis=1)
            all_min_distances.extend(batch_mins.tolist())
            
            total_exact += int(np.sum(batch_mins == 0))
            total_near += int(np.sum((batch_mins > 0) & (batch_mins <= near_dup_threshold)))
            
        total_compared_frames += len(h1_list)
        
    avg_min_dist = float(np.mean(all_min_distances)) if all_min_distances else 0.0
    return total_exact, total_near, avg_min_dist, total_compared_frames


def main():
    parser = argparse.ArgumentParser(description="Порівняльний аналіз вибірок train, val та test у YOLO датасеті з урахуванням карт")
    parser.add_argument("--dataset_dir", type=str, default="../dataset_v25_splitted", help="Шлях до теки датасету")
    parser.add_argument("--near_threshold", type=int, default=3, help="Поріг dHash для віднесення до майже-дублікатів (за замовчуванням 3)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Кількість паралельних процесів для обчислення dHash")
    
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    
    if not dataset_dir.exists():
        logger.error(f"Директорію {dataset_dir} не знайдено.")
        return

    splits = ["train", "val", "test"]
    split_paths = {}
    
    # Перевірка наявності розділів
    for s in splits:
        img_dir = dataset_dir / s / "images"
        lbl_dir = dataset_dir / s / "labels"
        if img_dir.exists() and lbl_dir.exists():
            split_paths[s] = (img_dir, lbl_dir)
            
    if len(split_paths) < 2:
        logger.error(f"Для порівняння знайдено замало розділів ({list(split_paths.keys())}). Необхідно мінімум 2 розділи.")
        return

    # Завантаження класів
    class_mapping = load_class_mapping_from_yaml(dataset_dir / "data.yaml")
    
    logger.info("=" * 80)
    logger.info(f"ДОСЛІДЖЕННЯ ТА ПОРІВНЯННЯ ВИБІРОК: TRAIN / VAL / TEST (ОБМЕЖЕНО КАРТАМИ)")
    logger.info(f"Директорія датасету: {dataset_dir.name}")
    logger.info(f"Знайдені розділи   : {', '.join(split_paths.keys())}")
    logger.info("=" * 80)
    
    # Збір статистичних даних
    files = {}
    stats = {}
    for s, (img_d, lbl_d) in split_paths.items():
        logger.info(f"Аналіз структури та анотацій вибірки {s.upper()}...")
        f_list, f_stats = analyze_split(img_d, lbl_d, class_mapping)
        files[s] = f_list
        stats[s] = f_stats

    # Обчислення візуальних метрик для всіх кадрів разом
    logger.info("\nПаралельний аналіз візуальних метрик для всіх кадрів...")
    all_files = []
    for s in split_paths.keys():
        all_files.extend(files[s])
        
    metrics_cache = precompute_metrics_parallel(all_files, max_workers=args.workers)
    
    # Розподіл хешів та показників розмитості по вибірках
    hashes = {}
    blurs = {}
    for s in split_paths.keys():
        hashes[s] = [metrics_cache[f][0] for f in files[s] if f in metrics_cache]
        blurs[s] = [metrics_cache[f][1] for f in files[s] if f in metrics_cache]

    # Визначення пар для порівняння
    pairs_to_compare = []
    available = list(split_paths.keys())
    if "train" in available and "val" in available:
        pairs_to_compare.append(("train", "val"))
    if "val" in available and "test" in available:
        pairs_to_compare.append(("val", "test"))
    if "train" in available and "test" in available:
        pairs_to_compare.append(("train", "test"))

    # 1. ОБСЯГИ ТА СТРУКТУРА ВИБІРОК
    print("\n" + "=" * 80)
    print("                1. ОБСЯГИ ТА СТРУКТУРА ВИБІРОК")
    print("=" * 80)
    header = f" Метрика                      "
    for s in splits:
        if s in stats:
            header += f"| {s.upper():<15} "
    print(header)
    print("-" * 80)
    
    def get_row_string(label, key, is_pct=False, custom_val_fn=None):
        row = f" {label:<28} "
        for s in splits:
            if s in stats:
                if custom_val_fn:
                    val = custom_val_fn(s)
                else:
                    val = stats[s][key]
                if is_pct:
                    row += f"| {val:0.2f}%{'':<10} "
                else:
                    row += f"| {val:<15} "
        return row

    print(get_row_string("Загальна кількість кадрів", "total_images"))
    print(get_row_string("Анотовані (позитивні)", "annotated_count"))
    print(get_row_string("Фонові (негативні)", "background_count"))
    print(get_row_string("Частка фону у вибірці", "", is_pct=True, 
                         custom_val_fn=lambda s: (stats[s]["background_count"] / max(1, stats[s]["total_images"]) * 100.0)))
    print(get_row_string("Кількість bounding boxes", "total_boxes"))

    # 2. РОЗПОДІЛ КЛАСІВ
    print("\n" + "=" * 80)
    print("                2. РОЗПОДІЛ КЛАСІВ (Bounding Boxes)")
    print("=" * 80)
    header_cls = f" Клас            "
    for s in splits:
        if s in stats:
            header_cls += f"| {s.upper():<17} "
    print(header_cls)
    print("-" * 80)
    
    for cls in sorted(class_mapping.values()):
        row = f" {cls:<15} "
        for s in splits:
            if s in stats:
                cnt = stats[s]["class_counts"].get(cls, 0)
                pct = (cnt / stats[s]["total_boxes"] * 100) if stats[s]["total_boxes"] > 0 else 0.0
                row += f"| {cnt:<5} ({pct:5.2f}%)     "
        print(row)
    print("-" * 80)
    
    for p1, p2 in pairs_to_compare:
        p1_ratio = {k: (v / stats[p1]["total_boxes"] if stats[p1]["total_boxes"] > 0 else 0) for k, v in stats[p1]["class_counts"].items()}
        p2_ratio = {k: (v / stats[p2]["total_boxes"] if stats[p2]["total_boxes"] > 0 else 0) for k, v in stats[p2]["class_counts"].items()}
        sim = calculate_cosine_similarity(p1_ratio, p2_ratio)
        print(f" Косинусна схожість розподілів ({p1.upper()} <-> {p2.upper()}): {sim:.4f}")

    # 3. АНАЛІЗ ПЕРЕТИНУ МЕТАДАНИХ (ЕПІЗОДИ ТА КАРТИ)
    print("\n" + "=" * 80)
    print("                3. АНАЛІЗ ПЕРЕТИНУ МЕТАДАНИХ (ЕПІЗОДИ ТА КАРТИ)")
    print("=" * 80)
    for s in splits:
        if s in stats:
            print(f" {s.upper():<5} | Унікальних карт: {len(stats[s]['maps']):<3} | Унікальних епізодів: {len(stats[s]['episodes'])}")
    print("-" * 80)
    
    for p1, p2 in pairs_to_compare:
        map_inter = stats[p1]["maps"].intersection(stats[p2]["maps"])
        ep_inter = stats[p1]["episodes"].intersection(stats[p2]["episodes"])
        print(f" Перетин {p1.upper()} <-> {p2.upper()}:")
        print(f"  * Спільні карти (кількість: {len(map_inter)})      : {sorted(list(map_inter))}")
        print(f"  * Спільні епізоди (кількість: {len(ep_inter)})   : {sorted(list(ep_inter)) if ep_inter else 'немає (чистий розподіл)'}")
        print("-" * 80)

    # 4. ВІЗУАЛЬНА СХОЖІСТЬ ТА ДУБЛЮВАННЯ (З ОБМЕЖЕННЯМ НА СПІЛЬНІ КАРТИ)
    print("\n" + "=" * 80)
    print("                4. ВІЗУАЛЬНА СХОЖІСТЬ (Крос-аналіз dHash, обмежений спільними картами)")
    print("=" * 80)
    for p1, p2 in pairs_to_compare:
        logger.info(f"Обчислення близькості кадрів для пари {p1.upper()} <-> {p2.upper()} в межах однакових локацій...")
        exact_dups, near_dups, avg_min_dist, compared_cnt = evaluate_map_constrained_similarity(
            files[p1], hashes[p1], files[p2], hashes[p2], near_dup_threshold=args.near_threshold
        )
        print(f" Порівняння: {p1.upper()} <-> {p2.upper()}")
        if compared_cnt > 0:
            print(f"  * Оброблено кадрів спільних карт                   : {compared_cnt} (з {len(files[p1])})")
            print(f"  * Точні дублікати на спільних картах (dist = 0)     : {exact_dups}")
            print(f"  * Близькі кадри на спільних картах (dist <= {args.near_threshold})      : {near_dups}")
            print(f"  * Середня мін. відстань Хеммінга на спільних картах: {avg_min_dist:.2f} бітів")
        else:
            print("  * Не знайдено спільних карт для зіставлення. Візуальний витік виключений.")
        print("-" * 80)
    print(" *Примітка: Обмеження порівняння межами однієї карти виключає вплив")
    print("            різних фонів і показує реальну схожість дублів або схожих")
    print("            епізодів у межах однієї локації.")

    # 5. ОЦІНКА ОДНОРІДНОСТІ ЯКОСТІ ЗОБРАЖЕНЬ
    print("\n" + "=" * 80)
    print("                5. ОЦІНКА ОДНОРІДНОСТІ ЯКОСТІ ЗОБРАЖЕНЬ")
    print("=" * 80)
    for s in splits:
        if s in blurs:
            mean_val = np.mean(blurs[s]) if blurs[s] else 0.0
            print(f" Середня чіткість у {s.upper():<5} (дисперсія Лапласіана)  : {mean_val:.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()