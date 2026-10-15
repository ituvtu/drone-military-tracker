#!/usr/bin/env python3
"""
Скрипт для виявлення "витоку даних" (data leakage) між вибірками (train/val/test).
Шукає суміжні епізоди (наприклад, ep_001 та ep_002) з однієї мапи та сегменту,
які були розподілені у різні спліти.
"""

import os
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("SplitLeakChecker")

# Регулярний вираз для пошуку структури: m1_s1_ep_001
PATTERN = re.compile(r"^(m\d+)_(s\d+)_ep_(\d+)", re.IGNORECASE)
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    parser = argparse.ArgumentParser(description="Пошук сусідніх епізодів у різних вибірках (витік даних)")
    parser.add_argument("--dataset_dir", type=str, default="dataset_v25_640_merged", help="Шлях до папки з датасетом")
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        logger.error(f"Директорія датасету не знайдена: {dataset_dir}")
        return
        
    splits = ["train", "val", "test"]
    
    # Словник для збереження вибірок: (map, segment, episode_num) -> set of splits
    episode_to_splits: Dict[Tuple[str, str, int], Set[str]] = {}
    
    # Збереження оригінального рядкового формату номеру (наприклад, "001" для гарного виведення)
    episode_formats: Dict[Tuple[str, str, int], str] = {}
    
    logger.info(f"Сканування зображень у {dataset_dir.name}...")
    
    for split in splits:
        img_dir = dataset_dir / split / "images"
        if not img_dir.exists():
            continue
            
        for img_path in img_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTENSIONS:
                stem = img_path.stem
                match = PATTERN.match(stem)
                if match:
                    map_id = match.group(1).lower()
                    seg_id = match.group(2).lower()
                    ep_str = match.group(3)
                    ep_num = int(ep_str)
                    
                    key = (map_id, seg_id, ep_num)
                    if key not in episode_to_splits:
                        episode_to_splits[key] = set()
                    episode_to_splits[key].add(split)
                    
                    episode_formats[key] = ep_str
                    
    if not episode_to_splits:
        logger.warning("Не знайдено жодного файлу з назвою, що відповідає шаблону mX_sX_ep_XXX.")
        return
        
    # Групування унікальних номерів епізодів за комбінацією (map, segment)
    groups: Dict[Tuple[str, str], Set[int]] = {}
    for (map_id, seg_id, ep_num) in episode_to_splits.keys():
        g_key = (map_id, seg_id)
        if g_key not in groups:
            groups[g_key] = set()
        groups[g_key].add(ep_num)
        
    leaks_count = 0
    
    print("\n" + "="*75)
    print("          АНАЛІЗ СУСІДНІХ ЕПІЗОДІВ У РІЗНИХ ВИБІРКАХ")
    print("="*75)
    
    # Прохід по кожній групі (мапа + сегмент)
    for (map_id, seg_id), ep_nums in sorted(groups.items()):
        sorted_eps = sorted(list(ep_nums))
        group_leaks = []
        
        for i in range(len(sorted_eps) - 1):
            ep1 = sorted_eps[i]
            ep2 = sorted_eps[i+1]
            
            # Якщо епізоди є безпосередніми сусідами (різниця дорівнює 1)
            if ep2 - ep1 == 1:
                key1 = (map_id, seg_id, ep1)
                key2 = (map_id, seg_id, ep2)
                
                splits1 = episode_to_splits[key1]
                splits2 = episode_to_splits[key2]
                
                # Якщо набори вибірок для сусідніх епізодів відрізняються
                if splits1 != splits2:
                    ep1_str = episode_formats[key1]
                    ep2_str = episode_formats[key2]
                    
                    s1_str = "/".join(sorted(list(splits1)))
                    s2_str = "/".join(sorted(list(splits2)))
                    
                    group_leaks.append((ep1_str, s1_str, ep2_str, s2_str))
                    leaks_count += 1
                    
        if group_leaks:
            print(f"\n📍 Мапа: {map_id.upper()}, Сегмент: {seg_id.upper()}")
            print("-" * 55)
            for ep1_str, s1, ep2_str, s2 in group_leaks:
                print(f"  • ep_{ep1_str} ({s1:<5})  <--->  ep_{ep2_str} ({s2:<5})")
                
    print("\n" + "="*75)
    if leaks_count > 0:
        print(f"⚠️  Увага! Виявлено суміжних епізодів у різних вибірках: {leaks_count}")
        print("Рекомендовано перевірити логіку розділення (splitter), щоб уникнути витоку даних.")
    else:
        print("✅  Чудово! Суміжних епізодів у різних вибірках не виявлено.")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()