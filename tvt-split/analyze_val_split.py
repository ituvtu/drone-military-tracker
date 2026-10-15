#!/usr/bin/env python3
"""
Аналітична система дослідження валідаційної вибірки (val) YOLO-датасетів.
1. Оцінює обсяги та розподіл 6 класів (0-5).
2. Знаходить найменш та найбільш різноманітні епізоди за допомогою ентропії Шеннона.
3. Шукає специфічні часові патерни перекриття цілей (наявність -> відсутність -> наявність -> відсутність).
"""

import os
import re
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import Counter


# Стандартний мапінг класів
DEFAULT_CLASS_MAPPING = {
    0: "antiair",
    1: "bmp",
    2: "cars",
    3: "howitzer",
    4: "tank",
    5: "truck"
}


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


def load_class_mapping(yaml_path: Path) -> Dict[int, str]:
    """Зчитує класи з файлу конфігурації data.yaml."""
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
    except Exception:
        pass
    return mapping


def read_labels(label_path: Path) -> List[int]:
    """Зчитує ID класів з YOLO файлу розмітки."""
    class_ids = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return class_ids
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.append(int(parts[0]))
                    except ValueError:
                        pass
    except Exception:
        pass
    return class_ids


def calculate_entropy(class_counts: Dict[int, int], total: int) -> float:
    """Обчислює ентропію Шеннона для оцінки різноманітності класів."""
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in class_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def detect_occlusion_patterns(frames_data: List[Tuple[int, bool]], min_phase_len: int = 1) -> List[Dict[str, Any]]:
    """
    Аналізує хронологічну послідовність кадрів на наявність патерну перекриття цілі:
    Є об'єкт (1) -> Немає об'єкта (0) -> Є об'єкт (1) -> Немає об'єкта (0)
    
    min_phase_len: мінімальна тривалість фази в кадрах, щоб відсіяти одиничні шуми розмітки.
    """
    # Сортуємо кадри за їхнім порядковим номером
    sorted_frames = sorted(frames_data, key=lambda x: x[0])
    if len(sorted_frames) < 4:
        return []
    
    # Групуємо послідовні кадри в однорідні фази (Run-Length Encoding)
    phases = []
    current_state = sorted_frames[0][1]
    current_len = 1
    start_frame = sorted_frames[0][0]
    
    for frame_num, state in sorted_frames[1:]:
        if state == current_state:
            current_len += 1
        else:
            phases.append({
                "state": current_state,
                "length": current_len,
                "start": start_frame,
                "end": frame_num - 1
            })
            current_state = state
            current_len = 1
            start_frame = frame_num
            
    phases.append({
        "state": current_state,
        "length": current_len,
        "start": start_frame,
        "end": sorted_frames[-1][0]
    })
    
    # Фільтруємо фази за мінімальною тривалістю (якщо вказано) для уникнення надмірного шуму
    filtered_phases = [p for p in phases if p["length"] >= min_phase_len]
    
    matches = []
    # Шукаємо ланцюжок: True (є) -> False (немає) -> True (є) -> False (немає)
    for i in range(len(filtered_phases) - 3):
        p1, p2, p3, p4 = filtered_phases[i], filtered_phases[i+1], filtered_phases[i+2], filtered_phases[i+3]
        if p1["state"] is True and p2["state"] is False and p3["state"] is True and p4["state"] is False:
            matches.append({
                "start_frame": p1["start"],
                "end_frame": p4["end"],
                "durations": [p1["length"], p2["length"], p3["length"], p4["length"]]
            })
            
    return matches


def main():
    parser = argparse.ArgumentParser(description="Аналіз епізодів та пошук патернів перекриття цілей у валідаційній вибірці YOLO")
    parser.add_argument("--dataset_dir", type=str, default="../dataset_v25_splitted", help="Шлях до теки датасету")
    parser.add_argument("--split", type=str, default="train", help="Частина датасету для аналізу (за замовчуванням val)")
    parser.add_argument("--min_phase", type=int, default=1, help="Мінімальна довжина фази в кадрах для аналізу перекриття (за замовчуванням 1)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    split_dir = dataset_dir / args.split
    
    if not split_dir.exists():
        print(f"[Помилка] Директорію {split_dir} не знайдено.")
        return

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"[Помилка] Перевірте наявність папок 'images' та 'labels' у {split_dir}")
        return

    class_mapping = load_class_mapping(dataset_dir / "data.yaml")
    
    # Структура для агрегації загальних даних та хронології кадрів
    episodes_data = {}
    episodes_timeline = {}  # episode_name -> list of (frame_number, has_objects)

    image_files = sorted([
        f for f in images_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])

    for img_path in image_files:
        ep_name, frame_num = parse_filename(img_path.stem)
        
        rel_path = img_path.relative_to(images_dir)
        lbl_path = labels_dir / rel_path.with_suffix(".txt")
        
        class_ids = read_labels(lbl_path)
        has_objects = len(class_ids) > 0
        
        # Запис для загальної статистики
        if ep_name not in episodes_data:
            episodes_data[ep_name] = {
                "total_frames": 0,
                "class_counts": Counter(),
                "total_objects": 0
            }
            episodes_timeline[ep_name] = []
            
        episodes_data[ep_name]["total_frames"] += 1
        episodes_timeline[ep_name].append((frame_num, has_objects))
        
        for cid in class_ids:
            if cid in class_mapping:
                episodes_data[ep_name]["class_counts"][cid] += 1
                episodes_data[ep_name]["total_objects"] += 1

    if not episodes_data:
        print("Не знайдено епізодів або розмітки для аналізу.")
        return

    # Розрахунок метрик різноманітності
    analysis_results = []
    for ep_name, data in episodes_data.items():
        total_obj = data["total_objects"]
        class_counts = dict(data["class_counts"])
        unique_classes = len(class_counts)
        entropy = calculate_entropy(class_counts, total_obj)
        concentration_score = total_obj / (unique_classes if unique_classes > 0 else 1)

        analysis_results.append({
            "episode": ep_name,
            "total_frames": data["total_frames"],
            "total_objects": total_obj,
            "unique_classes": unique_classes,
            "entropy": entropy,
            "concentration_score": concentration_score,
            "class_distribution": class_counts
        })

    # Сортування
    most_populated_least_diverse = sorted(analysis_results, key=lambda x: (x["concentration_score"], -x["entropy"]), reverse=True)
    most_diverse = sorted(analysis_results, key=lambda x: (x["unique_classes"], x["entropy"], x["total_objects"]), reverse=True)

    # Пошук патернів перекриття/миготіння
    occlusion_results = {}
    for ep_name, timeline in episodes_timeline.items():
        patterns = detect_occlusion_patterns(timeline, min_phase_len=args.min_phase)
        if patterns:
            occlusion_results[ep_name] = patterns

    print("=" * 90)
    print(f" КОМПЛЕКСНИЙ АНАЛІЗ ВИБІРКИ: {args.split.upper()}")
    print(f" Оброблено унікальних епізодів: {len(analysis_results)}")
    print("=" * 90)

    # 1. Найбільша концентрація (багато техніки, мало класів)
    print("\n[1] ТОП ЕПІЗОДИ З НАЙБІЛЬШОЮ КІЛЬКІСТЮ ТЕХНІКИ ТА НИЗЬКОЮ РІЗНОМАНІТНІСТЮ:")
    print("-" * 90)
    for i, res in enumerate(most_populated_least_diverse[:2], 1):
        dist_str = ", ".join([f"{class_mapping[k]}: {v}" for k, v in sorted(res["class_distribution"].items())])
        print(f" {i}. Епізод: '{res['episode']}'")
        print(f"    * Об'єктів: {res['total_objects']} | Кадрів: {res['total_frames']} | Класів: {res['unique_classes']}/6")
        print(f"    * Склад:    {{{dist_str}}}")
        print("-" * 90)

    # 2. Найбільша різноманітність
    print("\n[2] ТОП НАЙБІЛЬШ РІЗНОМАНІТНИХ ЕПІЗОДІВ ЗА СКЛАДОМ ТЕХНІКИ:")
    print("-" * 90)
    for i, res in enumerate(most_diverse[:2], 1):
        dist_str = ", ".join([f"{class_mapping[k]}: {v}" for k, v in sorted(res["class_distribution"].items())])
        print(f" {i}. Епізод: '{res['episode']}'")
        print(f"    * Об'єктів: {res['total_objects']} | Кадрів: {res['total_frames']} | Класів: {res['unique_classes']}/6")
        print(f"    * Склад:    {{{dist_str}}}")
        print("-" * 90)

    # 3. Аналіз часових патернів перекриття цілей (Нова функція)
    print("\n[3] ВИЯВЛЕННЯ ПАТЕРНІВ ПЕРЕКРИТТЯ / МИГОТІННЯ ЦІЛІ (Є -> Немає -> Є -> Немає):")
    print("-" * 90)
    if not occlusion_results:
        print(f" У жодному епізоді не виявлено чітких патернів зникнення об'єктів з параметром min_phase={args.min_phase}.")
    else:
        for ep_name, matches in sorted(occlusion_results.items(), key=lambda x: len(x[1]), reverse=True):
            print(f" Епізод: '{ep_name}'")
            print(f"  * Знайдено аномалій перекриття: {len(matches)}")
            for idx, match in enumerate(matches[:3], 1):  # Показуємо перші 3 збіги для компактності
                print(f"    - Випадок #{idx}: кадри з f_{match['start_frame']} по f_{match['end_frame']}")
                print(f"      Тривалість фаз (кадрів): [Є: {match['durations'][0]} -> Немає: {match['durations'][1]} -> Є: {match['durations'][2]} -> Немає: {match['durations'][3]}]")
            if len(matches) > 3:
                print(f"    ... та ще {len(matches) - 3} аналогічних переходів у цьому відео.")
            print("-" * 90)


if __name__ == "__main__":
    main()