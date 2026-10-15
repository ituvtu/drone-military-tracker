#!/usr/bin/env python3
"""
Система аналізу стану та метрик датасету (Dataset Post-Processing Analyzer)
Аналізує структуру, обсяг кадрів, географію та покласовий баланс після дедуплікації.
Підтримує як класичну структуру з CVAT XML, так і YOLO-формати (обидва варіанти розміщення images/labels).
"""

import os
import json
import logging
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from tqdm import tqdm

# Налаштування логування
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("DatasetAnalyzer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()


class DatasetAnalyzer:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.version = self.get_version()
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.classes: List[str] = []
        self.maps: List[str] = []
        self.splits: List[str] = []

    def get_version(self) -> str:
        """
        Визначає версію датасету з назви кореневої папки.
        Шукає патерни на кшталт v15, v16, v1.5 тощо.
        """
        match = re.search(r"v\d+(?:\.\d+)*", self.dataset_dir.name, re.IGNORECASE)
        if match:
            return match.group(0).lower()
        return ""

    def detect_dataset_type(self) -> str:
        """
        Визначає тип структури датасету: 'yolo' або 'cvat_xml'.
        """
        if (self.dataset_dir / "data.yaml").exists():
            return "yolo"
        return "cvat_xml"

    def load_data_yaml(self, yaml_path: Path) -> Dict[int, str]:
        """
        Зчитує назви класів з конфігураційного файлу data.yaml.
        """
        class_mapping = {}
        if not yaml_path.exists():
            logger.warning(f"Файл {yaml_path} не знайдено.")
            return class_mapping
        
        try:
            import yaml
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data and 'names' in data:
                    names = data['names']
                    if isinstance(names, dict):
                        return {int(k): str(v) for k, v in names.items()}
                    elif isinstance(names, list):
                        return {i: str(v) for i, v in enumerate(names)}
        except ImportError:
            logger.info("Бібліотеку PyYAML не знайдено, виконується альтернативний парсинг data.yaml")
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    names_section = re.search(r'names:\s*\n((?:\s+[-0-9]+:?.*\n?)+)', content)
                    if names_section:
                        lines = names_section.group(1).split('\n')
                        for line in lines:
                            dict_match = re.search(r'(\d+):\s*["\']?([a-zA-Z0-9_-]+)["\']?', line)
                            if dict_match:
                                class_mapping[int(dict_match.group(1))] = dict_match.group(2)
                            else:
                                list_match = re.search(r'-\s*["\']?([a-zA-Z0-9_-]+)["\']?', line)
                                if list_match:
                                    idx = len(class_mapping)
                                    class_mapping[idx] = list_match.group(1)
            except Exception as e:
                logger.error(f"Помилка парсингу data.yaml: {e}")
        return class_mapping

    def parse_yolo_label(self, label_path: Path, class_mapping: Dict[int, str]) -> Dict[str, int]:
        """
        Парсить YOLO текстовий файл анотації та повертає кількість об'єктів для кожного класу.
        """
        counts = {}
        if not label_path.exists() or label_path.stat().st_size == 0:
            return counts  # Порожній файл або відсутній -> негативний кадр

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        try:
                            class_idx = int(parts[0])
                            class_name = class_mapping.get(class_idx, f"class_{class_idx}")
                            counts[class_name] = counts.get(class_name, 0) + 1
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"Не вдалося обробити YOLO-файл анотації {label_path}: {e}")
        return counts

    def parse_episode_from_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        Намагається виділити назву карти, датасету та епізоду з назви файлу зображення.
        Наприклад: map_1_ds_1_ep_013_a_frame_0001.jpg -> map_1, ds_1, ep_013_a
        """
        stem = Path(filename).stem
        
        map_match = re.search(r"map_\d+", stem, re.IGNORECASE)
        ds_match = re.search(r"ds_\d+", stem, re.IGNORECASE)
        ep_match = re.search(r"ep_\d+(?:_[a-zA-Z0-9]+)?", stem, re.IGNORECASE)
        
        map_name = map_match.group(0).lower() if map_match else "map_unknown"
        ds_name = ds_match.group(0).lower() if ds_match else "ds_unknown"
        ep_name = ep_match.group(0).lower() if ep_match else "ep_unknown"
        
        return map_name, ds_name, ep_name

    def parse_cvat_xml(self, xml_path: Path, img_files: List[str]) -> Tuple[Dict[str, int], int]:
        """
        Парсить CVAT annotations.xml, повертаючи покласовий розподіл та кількість унікальних анотованих кадрів.
        """
        counts: Dict[str, Set[Any]] = {}
        existing_filenames = set(img_files)
        existing_frame_numbers = set()
        for f in img_files:
            name_without_ext = Path(f).stem
            digits = ''.join(c for c in name_without_ext if c.isdigit())
            if digits:
                try:
                    existing_frame_numbers.add(int(digits))
                except ValueError:
                    pass

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 1. Track-based CVAT формат (відео-послідовності)
            tracks = root.findall('.//track')
            if tracks:
                for track in tracks:
                    label = track.get('label')
                    if not label:
                        continue
                    if label not in counts:
                        counts[label] = set()
                    for box in track.findall('box'):
                        outside = box.get('outside')
                        if outside in ('1', 'true'):
                            continue
                        frame = box.get('frame')
                        if frame is not None:
                            frame_idx = int(frame)
                            if frame_idx in existing_frame_numbers:
                                counts[label].add(frame_idx)
                
                all_annotated_frames = set().union(*counts.values()) if counts else set()
                return {label: len(frames) for label, frames in counts.items()}, len(all_annotated_frames)
                
            # 2. Image-based CVAT формат (набір окремих зображень)
            images = root.findall('.//image')
            if images:
                img_counts: Dict[str, Set[str]] = {}
                for img in images:
                    img_name = img.get('name') or ""
                    img_id = img.get('id')
                    
                    exists = False
                    if img_name in existing_filenames:
                        exists = True
                    else:
                        if any(Path(f).name == img_name or Path(f).stem == img_name for f in img_files):
                            exists = True
                            
                    if not exists and img_id:
                        try:
                            if int(img_id) in existing_frame_numbers:
                                exists = True
                        except ValueError:
                            pass
                            
                    if not exists:
                        continue
                        
                    for box in img.findall('box'):
                        label = box.get('label')
                        if label:
                            if label not in img_counts:
                                img_counts[label] = set()
                            img_counts[label].add(img_name or img_id)
                
                all_annotated_frames = set().union(*img_counts.values()) if img_counts else set()
                return {label: len(imgs) for label, imgs in img_counts.items()}, len(all_annotated_frames)
                
        except ET.ParseError as e:
            logger.warning(f"Помилка структури XML у файлі {xml_path}: {e}")
        except Exception as e:
            logger.warning(f"Не вдалося обробити XML-файл {xml_path}: {e}")
        return {}, 0

    def scan_dataset(self) -> Dict[str, Dict[str, Any]]:
        """
        Сканує директорію та автоматично визначає формат даних (YOLO або CVAT XML).
        Підтримує різні варіанти структури папок для YOLO.
        """
        self.episodes.clear()
        
        if not self.dataset_dir.exists():
            logger.error(f"Вказана директорія не існує: {self.dataset_dir}")
            return self.episodes

        dataset_type = self.detect_dataset_type()

        # Сценарій 1: YOLO Структура
        if dataset_type == "yolo":
            logger.info("Виявлено структуру YOLO. Початок аналізу за YAML-конфігурацією та мітками.")
            yaml_path = self.dataset_dir / "data.yaml"
            class_mapping = self.load_data_yaml(yaml_path)
            self.classes = sorted(list(class_mapping.values()))
            
            # Списки для збереження шляхів сканування: (директорія_зображень, директорія_міток, назва_спліту)
            scan_targets: List[Tuple[Path, Path, str]] = []
            images_dir = self.dataset_dir / "images"

            # Перевірка Варіанту А: images/ у корені датасету (images/train, images/val і т.д.)
            if images_dir.exists() and images_dir.is_dir():
                logger.info("Виявлено структуру YOLO з кореневою папкою 'images'.")
                splits = [d.name for d in images_dir.iterdir() if d.is_dir()]
                if not splits:
                    logger.warning("Не знайдено піддиректорій у папці images, сканується корінь.")
                    splits = ["dataset"]

                for split in splits:
                    split_img_dir = images_dir / split if split != "dataset" else images_dir
                    split_lbl_dir = self.dataset_dir / "labels" / split if split != "dataset" else self.dataset_dir / "labels"
                    scan_targets.append((split_img_dir, split_lbl_dir, split))
            
            # Перевірка Варіанту Б: спліти в корені (train/images, test/images і т.д.)
            else:
                logger.info("Кореневу папку 'images' не знайдено. Перевірка структури зі сплітами в корені (напр. train/images).")
                # Шукаємо всі папки в корені, що містять підпапку "images"
                possible_splits = [d.name for d in self.dataset_dir.iterdir() if d.is_dir() and (d / "images").exists()]
                for split in possible_splits:
                    split_img_dir = self.dataset_dir / split / "images"
                    split_lbl_dir = self.dataset_dir / split / "labels"
                    scan_targets.append((split_img_dir, split_lbl_dir, split))

            if not scan_targets:
                logger.error("Не вдалося знайти структуру зображень та міток у вказаній директорії.")
                return self.episodes

            # Пошук файлів за налаштованими цілями
            scan_jobs = []
            for split_img_dir, split_lbl_dir, split in scan_targets:
                for root_dir, _, files in os.walk(split_img_dir):
                    current_path = Path(root_dir).resolve()
                    img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if img_files:
                        scan_jobs.append((current_path, split_img_dir, split_lbl_dir, split, img_files))

            grouped_episodes = {}
            for current_path, split_img_dir, split_lbl_dir, split, img_files in tqdm(scan_jobs, desc="Аналіз YOLO зображень", unit="папка"):
                for img_file in img_files:
                    img_path = current_path / img_file
                    map_name, ds_name, ep_name = self.parse_episode_from_filename(img_file)
                    
                    try:
                        # Зберігаємо відносну структуру вкладеності всередині папки зображень спліту
                        rel_to_split = img_path.relative_to(split_img_dir)
                        label_path = split_lbl_dir / rel_to_split.with_suffix(".txt")
                    except (ValueError, Exception):
                        label_path = split_lbl_dir / img_path.with_suffix(".txt").name
                    
                    frame_classes = self.parse_yolo_label(label_path, class_mapping)
                    is_frame_neg = len(frame_classes) == 0
                    
                    episode_key = f"{split}/{map_name}/{ds_name}/{ep_name}"
                    
                    if episode_key not in grouped_episodes:
                        grouped_episodes[episode_key] = {
                            "split": split,
                            "map": map_name,
                            "ds": ds_name,
                            "ep": ep_name,
                            "is_neg": True,
                            "total_frames": 0,
                            "annotated_frames": 0,
                            "negative_frames": 0,
                            "class_counts": {c: 0 for c in self.classes}
                        }
                        
                    grouped_episodes[episode_key]["total_frames"] += 1
                    if not is_frame_neg:
                        grouped_episodes[episode_key]["is_neg"] = False
                        grouped_episodes[episode_key]["annotated_frames"] += 1
                        for cls_name, count in frame_classes.items():
                            if cls_name not in grouped_episodes[episode_key]["class_counts"]:
                                grouped_episodes[episode_key]["class_counts"][cls_name] = 0
                            grouped_episodes[episode_key]["class_counts"][cls_name] += count
                    else:
                        grouped_episodes[episode_key]["negative_frames"] += 1

            self.episodes = grouped_episodes

        # Сценарій 2: Оригінальна структура з CVAT XML
        else:
            logger.info("Виявлено структуру CVAT XML. Початок сканування директорій.")
            scan_targets = []
            for root_dir, _, files in os.walk(self.dataset_dir):
                current_path = Path(root_dir).resolve()
                try:
                    rel_path = current_path.relative_to(self.dataset_dir)
                except ValueError:
                    continue
                    
                parts = rel_path.parts
                if len(parts) in (3, 4):
                    img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if img_files:
                        scan_targets.append((current_path, parts, img_files))

            for current_path, parts, img_files in tqdm(scan_targets, desc="Аналіз структури XML", unit="еп"):
                if len(parts) == 4:
                    split_name, map_name, ds_name, ep_name = parts
                else:
                    split_name = "dataset"
                    map_name, ds_name, ep_name = parts
                    
                episode_key = f"{split_name}/{map_name}/{ds_name}/{ep_name}" if len(parts) == 4 else f"{map_name}/{ds_name}/{ep_name}"
                total_frames = len(img_files)
                
                xml_path = current_path / "annotations.xml"
                is_neg = "neg" in ep_name.lower() or not xml_path.exists()
                
                class_counts = {}
                annotated_frames_count = 0
                if not is_neg and xml_path.exists():
                    class_counts, annotated_frames_count = self.parse_cvat_xml(xml_path, img_files)
                
                self.episodes[episode_key] = {
                    "split": split_name,
                    "map": map_name,
                    "ds": ds_name,
                    "ep": ep_name,
                    "is_neg": is_neg,
                    "total_frames": total_frames,
                    "annotated_frames": annotated_frames_count,
                    "negative_frames": total_frames - annotated_frames_count,
                    "class_counts": class_counts
                }

        # Оновлення метаданих для звіту
        detected_classes = set()
        detected_maps = set()
        detected_splits = set()
        for ep_info in self.episodes.values():
            detected_classes.update(ep_info["class_counts"].keys())
            detected_maps.add(ep_info["map"])
            detected_splits.add(ep_info["split"])
            
        self.classes = sorted(list(detected_classes))
        self.maps = sorted(list(detected_maps))
        self.splits = sorted(list(detected_splits))
        
        logger.info(f"Аналіз завершено. Оброблено епізодів: {len(self.episodes)}")
        return self.episodes

    def generate_report(self, markdown_path: Path, json_path: Path):
        """
        Розраховує метрики та генерує фінальний аналітичний звіт.
        """
        if not self.episodes:
            logger.error("Відсутні дані для формування звіту.")
            return

        # Додавання версії до назв файлів звітів (наприклад, report.md -> report_v15.md)
        if self.version:
            if self.version not in markdown_path.stem:
                markdown_path = markdown_path.with_name(f"{markdown_path.stem}_{self.version}{markdown_path.suffix}")
            if self.version not in json_path.stem:
                json_path = json_path.with_name(f"{json_path.stem}_{self.version}{json_path.suffix}")

        total_episodes_by_split = {s: 0 for s in self.splits}
        total_frames_by_split = {s: 0 for s in self.splits}
        negatives_by_split = {s: 0 for s in self.splits}
        
        # Статистика кадрів (анотовані vs негативні)
        total_annotated_frames_by_split = {s: 0 for s in self.splits}
        total_negative_frames_by_split = {s: 0 for s in self.splits}
        
        class_distribution = {c: {s: 0 for s in self.splits} for c in self.classes}
        map_episodes_distribution = {m: {s: 0 for s in self.splits} for m in self.maps}
        map_frames_distribution = {m: {s: 0 for s in self.splits} for m in self.maps}
        
        for ep in self.episodes.values():
            s = ep["split"]
            total_episodes_by_split[s] += 1
            total_frames_by_split[s] += ep["total_frames"]
            total_annotated_frames_by_split[s] += ep.get("annotated_frames", 0)
            total_negative_frames_by_split[s] += ep.get("negative_frames", 0)
            
            if ep["is_neg"]:
                negatives_by_split[s] += 1
                
            for c, count in ep["class_counts"].items():
                if c in class_distribution:
                    class_distribution[c][s] += count
                
            map_episodes_distribution[ep["map"]][s] += 1
            map_frames_distribution[ep["map"]][s] += ep["total_frames"]

        total_eps_global = len(self.episodes)
        total_frames_global = sum(total_frames_by_split.values())
        
        total_annotated_frames_global = sum(total_annotated_frames_by_split.values())
        total_negative_frames_global = sum(total_negative_frames_by_split.values())
        
        is_split_mode = len(self.splits) > 1 or "dataset" not in self.splits

        # Генерація тексту Markdown
        report = []
        
        version_title = f" ({self.version})" if self.version else ""
        report.append(f"# Фактичний аналітичний звіт стану структури датасету{version_title}\n")
        
        version_meta = f" | Версія: `{self.version}`" if self.version else ""
        report.append(f"> :file_folder: Аналіз проведено для каталогу: `{self.dataset_dir.name}`{version_meta}\n")
        report.append("Звіт відображає дійсний обсяг даних після виконання етапів очищення, дедуплікації та фільтрації.\n")
        
        # Розділ 1: Загальний кількісний розподіл
        if is_split_mode:
            report.append("## 1. Загальний кількісний розподіл епізодів та кадрів\n")
            report.append("| Вибірка | Кількість епізодів | Частка епізодів (%) | Кількість кадрів | Частка кадрів (%) |")
            report.append("| :--- | :---: | :---: | :---: | :---: |")
            for s in self.splits:
                e_cnt = total_episodes_by_split[s]
                e_pct = (e_cnt / total_eps_global * 100) if total_eps_global > 0 else 0
                f_cnt = total_frames_by_split[s]
                f_pct = (f_cnt / total_frames_global * 100) if total_frames_global > 0 else 0
                report.append(f"| **{s.capitalize()}** | {e_cnt} | {e_pct:.1f}% | {f_cnt} | {f_pct:.1f}% |")
            report.append(f"| **Всього** | **{total_eps_global}** | **100.0%** | **{total_frames_global}** | **100.0%** |\n")
        else:
            report.append("## 1. Загальні метрики обсягу датасету\n")
            report.append(f"- **Загальна кількість епізодів**: {total_eps_global}")
            report.append(f"- **Загальна кількість кадрів (зображень)**: {total_frames_global}")
            report.append(f"- **Загальна кількість фонових (негативних) епізодів**: {sum(negatives_by_split.values())}\n")

        # Розділ 2: Покласовий розподіл об'єктів
        report.append("## 2. Покласовий розподіл об'єктів (Class Instance Distribution)\n")
        total_annotated_instances = sum(sum(class_distribution[c].values()) for c in self.classes)
        
        if is_split_mode:
            headers = ["Клас", "Всього об'єктів", "Частка від загальної к-сті (%)"] + [f"{s.capitalize()} (кількість / %)" for s in self.splits]
            report.append("| " + " | ".join(headers) + " |")
            report.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
            
            for c in self.classes:
                vals = class_distribution[c]
                tot_c = sum(vals.values())
                pct_of_total = (tot_c / total_annotated_instances * 100) if total_annotated_instances > 0 else 0
                row = [f"**{c}**", str(tot_c), f"{pct_of_total:.1f}%"]
                for s in self.splits:
                    cnt_s = vals[s]
                    pct_s = (cnt_s / tot_c * 100) if tot_c > 0 else 0
                    row.append(f"{cnt_s} ({pct_s:.1f}%)")
                report.append("| " + " | ".join(row) + " |")
        else:
            report.append("| Клас | Кількість об'єктів | Частка від загальної к-сті (%) |")
            report.append("| :--- | :---: | :---: |")
            for c in self.classes:
                tot_c = sum(class_distribution[c].values())
                pct_of_total = (tot_c / total_annotated_instances * 100) if total_annotated_instances > 0 else 0
                report.append(f"| **{c}** | {tot_c} | {pct_of_total:.1f}% |")
        report.append("\n")

        # Розділ 3: Розподіл за картами
        report.append("## 3. Розподіл за картами (Map Geographic Distribution)\n")
        if is_split_mode:
            headers = ["Карта", "Всього відео/еп", "Всього кадрів"] + [f"{s.capitalize()} (відео/кадри)" for s in self.splits]
            report.append("| " + " | ".join(headers) + " |")
            report.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
            
            for m in self.maps:
                tot_m_eps = sum(map_episodes_distribution[m].values())
                tot_m_fms = sum(map_frames_distribution[m].values())
                row = [f"**{m}**", str(tot_m_eps), str(tot_m_fms)]
                for s in self.splits:
                    eps_s = map_episodes_distribution[m][s]
                    fms_s = map_frames_distribution[m][s]
                    row.append(f"{eps_s} / {fms_s}")
                report.append("| " + " | ".join(row) + " |")
        else:
            report.append("| Назва карти | Кількість відео/еп | Кількість кадрів | Частка кадрів (%) |")
            report.append("| :--- | :---: | :---: | :---: |")
            for m in self.maps:
                tot_m_eps = map_episodes_distribution[m]["dataset"]
                tot_m_fms = map_frames_distribution[m]["dataset"]
                pct = (tot_m_fms / total_frames_global * 100) if total_frames_global > 0 else 0
                report.append(f"| **{m}** | {tot_m_eps} | {tot_m_fms} | {pct:.1f}% |")
        report.append("\n")

        # Розділ 4: Стан фонових даних
        report.append("## 4. Стан фонових даних (Чисті негативні епізоди)\n")
        total_negatives = sum(negatives_by_split.values())
        report.append(f"Загальна кількість чистих негативних епізодів (без анотованих об'єктів): **{total_negatives}**.\n")
        if total_negatives > 0 and is_split_mode:
            report.append("| Вибірка | Кількість фонових відео | Частка від загальної кількості (%) |")
            report.append("| :--- | :---: | :---: |")
            for s in self.splits:
                s_neg = negatives_by_split[s]
                s_pct = (s_neg / total_negatives * 100) if total_negatives > 0 else 0
                report.append(f"| **{s.capitalize()}** | {s_neg} | {s_pct:.1f}% |")
            report.append("\n")

        # Розділ 5: Співвідношення анотованих та негативних кадрів
        report.append("## 5. Співвідношення анотованих та негативних кадрів (Annotated vs Negative Frames)\n")
        report.append("Звіт відображає баланс між кадрами, що містять розмічені об'єкти (анотовані), та чистими фоновими кадрами (негативні).\n")
        
        if is_split_mode:
            headers = ["Вибірка", "Анотовані кадри", "Частка анотованих (%)", "Негативні кадри", "Частка негативних (%)", "Всього кадрів"]
            report.append("| " + " | ".join(headers) + " |")
            report.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
            
            for s in self.splits:
                ann_s = total_annotated_frames_by_split[s]
                neg_s = total_negative_frames_by_split[s]
                tot_s = ann_s + neg_s
                ann_pct = (ann_s / tot_s * 100) if tot_s > 0 else 0
                neg_pct = (neg_s / tot_s * 100) if tot_s > 0 else 0
                report.append(f"| **{s.capitalize()}** | {ann_s} | {ann_pct:.1f}% | {neg_s} | {neg_pct:.1f}% | {tot_s} |")
                
            ann_g = total_annotated_frames_global
            neg_g = total_negative_frames_global
            tot_g = total_frames_global
            ann_g_pct = (ann_g / tot_g * 100) if tot_g > 0 else 0
            neg_g_pct = (neg_g / tot_g * 100) if tot_g > 0 else 0
            report.append(f"| **Всього** | **{ann_g}** | **{ann_g_pct:.1f}%** | **{neg_g}** | **{neg_g_pct:.1f}%** | **{tot_g}** |\n")
        else:
            report.append("| Тип кадру | Кількість кадрів | Відносна частка (%) |")
            report.append("| :--- | :---: | :---: |")
            ann_g = total_annotated_frames_global
            neg_g = total_negative_frames_global
            tot_g = total_frames_global
            ann_g_pct = (ann_g / tot_g * 100) if tot_g > 0 else 0
            neg_g_pct = (neg_g / tot_g * 100) if tot_g > 0 else 0
            report.append(f"| **Анотовані кадри (з об'єктами)** | {ann_g} | {ann_g_pct:.1f}% |")
            report.append(f"| **Негативні кадри (фонові)** | {neg_g} | {neg_g_pct:.1f}% |")
            report.append(f"| **Всього кадрів** | **{tot_g}** | **100.0%** |\n")

        # Запис результатів
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
            
        output_json_data = {
            "metadata": {
                "dataset_root_dir": str(self.dataset_dir),
                "dataset_version": self.version if self.version else "unknown",
                "detected_classes": self.classes,
                "detected_maps": self.maps,
                "detected_splits": self.splits,
                "total_episodes": total_eps_global,
                "total_frames": total_frames_global
            },
            "statistics": {
                "episodes_by_split": total_episodes_by_split,
                "frames_by_split": total_frames_by_split,
                "annotated_frames_by_split": total_annotated_frames_by_split,
                "negative_frames_by_split": total_negative_frames_by_split,
                "negatives_by_split": negatives_by_split,
                "class_distribution": class_distribution,
                "map_episodes_distribution": map_episodes_distribution,
                "map_frames_distribution": map_frames_distribution
            }
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_json_data, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Аналітичний звіт збережено у Markdown: {markdown_path}")
        logger.info(f"Статистику збережено у JSON: {json_path}")
        
        print("\n" + "="*45)
        print("             КОНСОЛЬНЕ РЕЗЮМЕ")
        print("="*45)
        if self.version:
            print(f"Версія датасету : {self.version}")
        print(f"Всього епізодів : {total_eps_global}")
        print(f"Всього кадрів   : {total_frames_global}")
        print(f" -> Анотовані  : {total_annotated_frames_global} кадри ({total_annotated_frames_global/total_frames_global*100:.1f}%)")
        print(f" -> Негативні   : {total_negative_frames_global} кадри ({total_negative_frames_global/total_frames_global*100:.1f}%)")
        if is_split_mode:
            for s in self.splits:
                print(f" -> Вибірка '{s}': {total_episodes_by_split[s]} еп. | {total_frames_by_split[s]} кадрів")
        print("="*45 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Автоматичний системний аналіз метрик датасету")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Шлях до теки датасету")
    parser.add_argument("--output_report", type=str, default="dataset_analysis_report.md", help="Назва файлу вихідного Markdown звіту")
    parser.add_argument("--output_json", type=str, default="dataset_analysis_stats.json", help="Назва вихідного JSON файлу зі статистикою")
    
    args = parser.parse_args()
    
    try:
        analyzer = DatasetAnalyzer(dataset_dir=Path(args.dataset_dir))
        ep_dict = analyzer.scan_dataset()
        
        if not ep_dict:
            logger.error("Не знайдено валідних даних для аналізу. Перевірте структуру директорії.")
            return
            
        analyzer.generate_report(Path(args.output_report), Path(args.output_json))
        logger.info("Процес аналізу успішно завершено.")
        
    except Exception as e:
        logger.error(f"Виникла помилка під час роботи аналізатора: {e}", exc_info=True)


if __name__ == "__main__":
    main()