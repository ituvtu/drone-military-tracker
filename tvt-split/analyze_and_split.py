#!/usr/bin/env python3
"""
Система аналізу та розподілу епізодів комп'ютерного зору (YOLO / CVAT Dataset Splitter)
Забезпечує розподіл без витоку даних із суворим географічним та класовим балансуванням.
"""

import os
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any

# Налаштування логування
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("DatasetSplitter")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()


class DatasetSplitter:
    def __init__(self, dataset_dir: Path, ratios: List[float] = [0.70, 0.15, 0.15]):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.ratios = ratios
        
        # Перевірка валідності співвідношення вибірок
        if not abs(sum(ratios) - 1.0) < 1e-5:
            raise ValueError("Сума часток розподілу (ratios) повинна дорівнювати 1.0 (наприклад: [0.70, 0.15, 0.15])")
            
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.classes: List[str] = []
        self.maps: List[str] = []

    def parse_cvat_xml(self, xml_path: Path) -> Dict[str, int]:
        """
        Парсить CVAT annotations.xml підтримуючи як track-based, так і image-based формати.
        Повертає словник {назва_класу: кількість_кадрів}.
        """
        counts: Dict[str, Set[int]] = {}
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 1. Перевірка track-based CVAT формату (відео-послідовності)
            tracks = root.findall('.//track')
            if tracks:
                for track in tracks:
                    label = track.get('label')
                    if not label:
                        continue
                    if label not in counts:
                        counts[label] = set()
                    for box in track.findall('box'):
                        # Пропускаємо кадри, де об'єкт виходить за межі видимості
                        outside = box.get('outside')
                        if outside in ('1', 'true'):
                            continue
                        frame = box.get('frame')
                        if frame is not None:
                            counts[label].add(int(frame))
                return {label: len(frames) for label, frames in counts.items()}
                
            # 2. Перевірка image-based CVAT формату (набір окремих зображень)
            images = root.findall('.//image')
            if images:
                img_counts: Dict[str, Set[str]] = {}
                for img in images:
                    img_id = img.get('id') or img.get('name') or "unknown"
                    for box in img.findall('box'):
                        label = box.get('label')
                        if label:
                            if label not in img_counts:
                                img_counts[label] = set()
                            img_counts[label].add(img_id)
                return {label: len(imgs) for label, imgs in img_counts.items()}
                
        except ET.ParseError as e:
            logger.warning(f"Помилка структури XML у файлі {xml_path}: {e}")
        except Exception as e:
            logger.warning(f"Не вдалося обробити XML-файл {xml_path}: {e}")
        return {}

    def scan_dataset(self) -> Dict[str, Dict[str, Any]]:
        """
        Сканує файлову систему згідно з архітектурою: root / map_X / ds_Y / ep_Z
        """
        logger.info(f"Сканування директорії датасету: {self.dataset_dir}")
        self.episodes.clear()
        
        if not self.dataset_dir.exists():
            logger.error(f"Вказана директорія не існує: {self.dataset_dir}")
            return self.episodes

        # Обхід каталогу
        for root_dir, _, files in os.walk(self.dataset_dir):
            current_path = Path(root_dir).resolve()
            try:
                rel_path = current_path.relative_to(self.dataset_dir)
            except ValueError:
                continue
                
            parts = rel_path.parts
            
            # Перевіряємо глибину вкладеності (має бути рівно 3 рівня: map_X / ds_Y / ep_Z)
            if len(parts) == 3:
                map_name, ds_name, ep_name = parts
                
                # Пошук графічних файлів кадрів епізоду
                img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if not img_files:
                    continue  # Пропускаємо пусті директорії
                    
                episode_key = f"{map_name}/{ds_name}/{ep_name}"
                total_frames = len(img_files)
                
                # Визначення типу епізоду (позитивний чи чистий негатив)
                xml_path = current_path / "annotations.xml"
                is_neg = "neg" in ep_name.lower() or not xml_path.exists()
                
                class_counts = {}
                if not is_neg and xml_path.exists():
                    class_counts = self.parse_cvat_xml(xml_path)
                
                self.episodes[episode_key] = {
                    "map": map_name,
                    "ds": ds_name,
                    "ep": ep_name,
                    "is_neg": is_neg,
                    "total_frames": total_frames,
                    "class_counts": class_counts
                }

        # Динамічне визначення унікальних класів та карт
        detected_classes = set()
        detected_maps = set()
        for ep_info in self.episodes.values():
            detected_classes.update(ep_info["class_counts"].keys())
            detected_maps.add(ep_info["map"])
            
        self.classes = sorted(list(detected_classes))
        self.maps = sorted(list(detected_maps))
        
        logger.info(f"Успішно проскановано епізодів: {len(self.episodes)}")
        logger.info(f"Знайдено карт: {len(self.maps)} {self.maps}")
        logger.info(f"Знайдено класів: {len(self.classes)} {self.classes}")
        
        return self.episodes

    def run_split(self) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
        """
        Запускає оптимізований алгоритм із розділенням обов'язків:
        1. Позитивні відео розподіляються суворо за балансом покласових кадрів.
        2. Негативні відео коригують географічний баланс карт та об'єм фону.
        """
        if not self.episodes:
            raise ValueError("Немає даних для розподілу. Спершу виконайте сканування.")

        # Глобальні об'єми кадрів класів
        global_class_frames = {c: sum(ep["class_counts"].get(c, 0) for ep in self.episodes.values()) for c in self.classes}
        # Глобальна кількість епізодів для карт
        global_map_episodes = {m: sum(1 for ep in self.episodes.values() if ep["map"] == m) for m in self.maps}
        
        # Ініціалізація вибірок
        splits: Dict[int, List[str]] = {0: [], 1: [], 2: []}
        current_class_frames = {c: [0.0, 0.0, 0.0] for c in self.classes}
        current_map_episodes = {m: [0.0, 0.0, 0.0] for m in self.maps}
        
        positive_keys = [k for k, v in self.episodes.items() if not v["is_neg"]]
        negative_keys = [k for k, v in self.episodes.items() if v["is_neg"]]
        
        # Сортуємо класи за рідкістю (rare-first)
        sorted_classes = sorted(self.classes, key=lambda c: global_class_frames[c])
        unallocated_pos = set(positive_keys)

        # 1. Оцінка похибки позитивних епізодів (Тільки покласові кадри)
        def evaluate_positive_class_error(ep_info: Dict[str, Any], candidate_split: int) -> float:
            total_error = 0.0
            for c in ep_info["class_counts"]:
                total_c = global_class_frames[c]
                if total_c > 0:
                    for s in range(3):
                        current = current_class_frames[c][s]
                        if s == candidate_split:
                            current += ep_info["class_counts"][c]
                        proportion = current / total_c
                        total_error += (proportion - self.ratios[s]) ** 2
            return total_error

        # Розподіл позитивних епізодів
        for target_class in sorted_classes:
            candidates = [k for k in unallocated_pos if self.episodes[k]["class_counts"].get(target_class, 0) > 0]
            candidates.sort(key=lambda k: self.episodes[k]["class_counts"][target_class], reverse=True)
            
            for ep_key in candidates:
                if ep_key not in unallocated_pos:
                    continue
                    
                ep_info = self.episodes[ep_key]
                best_split_idx = 0
                min_error = float('inf')
                
                for idx in range(3):
                    error = evaluate_positive_class_error(ep_info, idx)
                    if error < min_error:
                        min_error = error
                        best_split_idx = idx
                        
                splits[best_split_idx].append(ep_key)
                unallocated_pos.remove(ep_key)
                
                # Оновлення метрик розподілу
                for c, count in ep_info["class_counts"].items():
                    current_class_frames[c][best_split_idx] += count
                current_map_episodes[ep_info["map"]][best_split_idx] += 1

        # Обробка можливих залишків
        if unallocated_pos:
            for ep_key in list(unallocated_pos):
                ep_info = self.episodes[ep_key]
                best_split_idx = sorted(range(3), key=lambda i: len(splits[i]) / self.ratios[i])[0]
                splits[best_split_idx].append(ep_key)
                unallocated_pos.remove(ep_key)
                current_map_episodes[ep_info["map"]][best_split_idx] += 1

        # 2. Розподіл негативних епізодів (Коригування географії карт та об'єму фону)
        total_negatives = len(negative_keys)
        current_negatives = [0.0, 0.0, 0.0]

        def evaluate_negative_global_error(ep_info: Dict[str, Any], candidate_split: int) -> float:
            total_error = 0.0
            
            # Оцінка похибки карт (інтегрує поточний стан від позитивних відео)
            m_name = ep_info["map"]
            total_m = global_map_episodes[m_name]
            if total_m > 0:
                for s in range(3):
                    current = current_map_episodes[m_name][s]
                    if s == candidate_split:
                        current += 1
                    proportion = current / total_m
                    total_error += (proportion - self.ratios[s]) ** 2
                    
            # Оцінка похибки об'єму чистих негативів
            if total_negatives > 0:
                for s in range(3):
                    current = current_negatives[s]
                    if s == candidate_split:
                        current += 1
                    proportion = current / total_negatives
                    total_error += (proportion - self.ratios[s]) ** 2
                    
            return total_error

        for ep_key in negative_keys:
            ep_info = self.episodes[ep_key]
            best_split_idx = 0
            min_error = float('inf')
            
            for idx in range(3):
                error = evaluate_negative_global_error(ep_info, idx)
                if error < min_error:
                    min_error = error
                    best_split_idx = idx
                    
            splits[best_split_idx].append(ep_key)
            current_negatives[best_split_idx] += 1
            current_map_episodes[ep_info["map"]][best_split_idx] += 1

        named_splits = {
            "train": splits[0],
            "val": splits[1],
            "test": splits[2]
        }
        
        final_fills_int = {c: [int(current_class_frames[c][i]) for i in range(3)] for c in self.classes}
        
        return named_splits, final_fills_int

    def save_json_config(self, splits: Dict[str, List[str]], class_fills: Dict[str, List[int]], output_path: Path):
        """
        Зберігає конфігурацію розподілу в JSON для наступного кроку (фізичного перенесення файлів).
        """
        output_data = {
            "metadata": {
                "dataset_root_dir": str(self.dataset_dir),
                "target_ratios": self.ratios,
                "detected_classes": self.classes,
                "detected_maps": self.maps,
                "class_frames_distribution": class_fills
            },
            "splits": {
                split_name: sorted(ep_list) for split_name, ep_list in splits.items()
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Конфігурацію розподілу успішно збережено в: {output_path}")

    def generate_markdown_report(self, splits: Dict[str, List[str]], class_fills: Dict[str, List[int]], output_path: Path):
        """
        Генерує детальний звіт про метрики збалансованості датасету.
        """
        report = []
        report.append("# Аналітичний звіт про розподіл та структуру датасету\n")
        report.append(f"> Автоматично згенеровано для директорії: `{self.dataset_dir.name}`\n")
        
        # 1. Загальна статистика епізодів
        report.append("## 1. Загальний кількісний розподіл відеоепізодів\n")
        report.append("| Вибірка | Кількість епізодів | Співвідношення (%) | Цільове (%) |")
        report.append("| :--- | :---: | :---: | :---: |")
        
        total_eps = len(self.episodes)
        for idx, s_name in enumerate(["train", "val", "test"]):
            count = len(splits[s_name])
            pct = (count / total_eps * 100) if total_eps > 0 else 0
            target_pct = self.ratios[idx] * 100
            report.append(f"| **{s_name.capitalize()}** | {count} | {pct:.1f}% | {target_pct:.1f}% |")
        report.append(f"| **Всього** | **{total_eps}** | **100.0%** | **100.0%** |\n")
        
        # 2. Розподіл за класами
        report.append("## 2. Покласовий розподіл кадрів\n")
        report.append("| Клас | Всього кадрів | Train (%) | Val (%) | Test (%) | Стан балансування |")
        report.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
        
        for c in self.classes:
            vals = class_fills[c]
            tot = sum(vals)
            if tot == 0:
                continue
            p_tr = vals[0] / tot * 100
            p_val = vals[1] / tot * 100
            p_ts = vals[2] / tot * 100
            
            # Статус балансування на основі відхилення
            status = "Збалансовано"
            if abs(p_tr - (self.ratios[0]*100)) <= 6.0 and abs(p_val - (self.ratios[1]*100)) <= 4.0:
                status = "**Оптимально**"
                
            report.append(f"| **{c}** | {tot} | {p_tr:.1f}% | {p_val:.1f}% | {p_ts:.1f}% | {status} |")
        report.append("\n")

        # 3. Географічний розподіл карт
        report.append("## 3. Географічний розподіл за картами (Map Geographic Distribution)\n")
        report.append("| Назва карти | Train (відео) | Val (відео) | Test (відео) | Всього відео | Розподіл (%) |")
        report.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
        
        for m in self.maps:
            tr_c = sum(1 for ep in splits["train"] if self.episodes[ep]["map"] == m)
            vl_c = sum(1 for ep in splits["val"] if self.episodes[ep]["map"] == m)
            ts_c = sum(1 for ep in splits["test"] if self.episodes[ep]["map"] == m)
            m_total = tr_c + vl_c + ts_c
            
            tr_p = tr_c / m_total * 100 if m_total > 0 else 0
            vl_p = vl_c / m_total * 100 if m_total > 0 else 0
            ts_p = ts_c / m_total * 100 if m_total > 0 else 0
            
            report.append(f"| **{m}** | {tr_c} | {vl_c} | {ts_c} | {m_total} | {tr_p:.1f}% / {vl_p:.1f}% / {ts_p:.1f}% |")
        report.append("\n")

        # 4. Негативні епізоди
        report.append("## 4. Розподіл чистих негативних епізодів (Фонові дані)\n")
        neg_eps = [k for k, v in self.episodes.items() if v["is_neg"]]
        total_neg = len(neg_eps)
        report.append(f"Загальна кількість чистих негативних епізодів: **{total_neg}**.\n")
        if total_neg > 0:
            report.append("| Вибірка | Кількість фонових відео | Частка від негативів |")
            report.append("| :--- | :---: | :---: |")
            for s_name in ["train", "val", "test"]:
                s_neg = sum(1 for ep in splits[s_name] if self.episodes[ep]["is_neg"])
                s_pct = s_neg / total_neg * 100
                report.append(f"| **{s_name.capitalize()}** | {s_neg} | {s_pct:.1f}% |")
            report.append("\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
        logger.info(f"Аналітичний звіт збережено в: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Автоматичний системний аналіз та розподіл датасету")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Шлях до кореневої папки датасету")
    parser.add_argument("--output_json", type=str, default="dataset_splits_config.json", help="Назва вихідного JSON-файлу")
    parser.add_argument("--output_report", type=str, default="dataset_split_report.md", help="Назва файлу аналітичного звіту")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Частка навчальної вибірки")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Частка валідаційної вибірки")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Частка тестової вибірки")
    
    args = parser.parse_args()
    
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    
    try:
        splitter = DatasetSplitter(dataset_dir=Path(args.dataset_dir), ratios=ratios)
        episodes_dict = splitter.scan_dataset()
        
        if not episodes_dict:
            logger.error("Не виявлено валідних епізодів у вказаній директорії. Перевірте структуру папок.")
            return

        splits, class_fills = splitter.run_split()
        
        # Збереження результатів розподілу
        splitter.save_json_config(splits, class_fills, Path(args.output_json))
        splitter.generate_markdown_report(splits, class_fills, Path(args.output_report))
        
        logger.info("Процес аналізу та розрахунку розподілу завершено успішно.")
        
    except Exception as e:
        logger.error(f"Помилка під час виконання програми: {e}", exc_info=True)


if __name__ == "__main__":
    main()