#!/usr/bin/env python3
"""
Система аналізу стану та метрик датасету (Dataset Post-Processing Analyzer)
Аналізує структуру, обсяг кадрів, географію та покласовий баланс після дедуплікації.
"""

import os
import json
import logging
import argparse
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
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.classes: List[str] = []
        self.maps: List[str] = []
        self.splits: List[str] = []

    def parse_cvat_xml(self, xml_path: Path, img_files: List[str]) -> Dict[str, int]:
        """
        Парсить CVAT annotations.xml, враховуючи лише ті кадри, які фізично існують на диску
        (актуально після дедуплікації, коли частину кадрів було видалено).
        """
        counts: Dict[str, Set[Any]] = {}
        
        # Підготовка списку наявних файлів для валідації
        existing_filenames = set(img_files)
        existing_frame_numbers = set()
        for f in img_files:
            name_without_ext = Path(f).stem
            # Витягуємо цифри з назви файлу для зіставлення з індексом кадру
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
                            # Зараховуємо клас лише якщо кадр існує фізично після дедуплікації
                            if frame_idx in existing_frame_numbers:
                                counts[label].add(frame_idx)
                return {label: len(frames) for label, frames in counts.items()}
                
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
                        continue  # Пропускаємо анотацію, якщо файл видалено під час дедуплікації
                        
                    for box in img.findall('box'):
                        label = box.get('label')
                        if label:
                            if label not in img_counts:
                                img_counts[label] = set()
                            img_counts[label].add(img_name or img_id)
                return {label: len(imgs) for label, imgs in img_counts.items()}
                
        except ET.ParseError as e:
            logger.warning(f"Помилка структури XML у файлі {xml_path}: {e}")
        except Exception as e:
            logger.warning(f"Не вдалося обробити XML-файл {xml_path}: {e}")
        return {}

    def scan_dataset(self) -> Dict[str, Dict[str, Any]]:
        """
        Сканує директорію та автоматично визначає структуру:
        - Звичайна: root / map_X / ds_Y / ep_Z
        - Розділена: root / [train|val|test] / map_X / ds_Y / ep_Z
        """
        logger.info(f"Сканування директорії для аналізу: {self.dataset_dir}")
        self.episodes.clear()
        
        if not self.dataset_dir.exists():
            logger.error(f"Вказана директорія не існує: {self.dataset_dir}")
            return self.episodes

        # Попередній збір шляхів для коректного відображення прогрес-бару
        scan_targets = []
        for root_dir, _, files in os.walk(self.dataset_dir):
            current_path = Path(root_dir).resolve()
            try:
                rel_path = current_path.relative_to(self.dataset_dir)
            except ValueError:
                continue
                
            parts = rel_path.parts
            # Перевіряємо глибину структури
            if len(parts) in (3, 4):
                img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if img_files:
                    scan_targets.append((current_path, parts, img_files))

        # Аналіз зібраних епізодів з прогрес-баром
        for current_path, parts, img_files in tqdm(scan_targets, desc="Аналіз структури", unit="еп"):
            if len(parts) == 4:
                split_name, map_name, ds_name, ep_name = parts
            else:
                split_name = "dataset"
                map_name, ds_name, ep_name = parts
                
            episode_key = f"{split_name}/{map_name}/{ds_name}/{ep_name}" if len(parts) == 4 else f"{map_name}/{ds_name}/{ep_name}"
            total_frames = len(img_files)
            
            # Визначення типу епізоду
            xml_path = current_path / "annotations.xml"
            is_neg = "neg" in ep_name.lower() or not xml_path.exists()
            
            class_counts = {}
            if not is_neg and xml_path.exists():
                class_counts = self.parse_cvat_xml(xml_path, img_files)
            
            self.episodes[episode_key] = {
                "split": split_name,
                "map": map_name,
                "ds": ds_name,
                "ep": ep_name,
                "is_neg": is_neg,
                "total_frames": total_frames,
                "class_counts": class_counts
            }

        # Збір унікальних метаданих
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

        # Підготовка статистичних агрегатів
        total_episodes_by_split = {s: 0 for s in self.splits}
        total_frames_by_split = {s: 0 for s in self.splits}
        negatives_by_split = {s: 0 for s in self.splits}
        
        class_distribution = {c: {s: 0 for s in self.splits} for c in self.classes}
        map_episodes_distribution = {m: {s: 0 for s in self.splits} for m in self.maps}
        map_frames_distribution = {m: {s: 0 for s in self.splits} for m in self.maps}
        
        for ep in self.episodes.values():
            s = ep["split"]
            total_episodes_by_split[s] += 1
            total_frames_by_split[s] += ep["total_frames"]
            if ep["is_neg"]:
                negatives_by_split[s] += 1
                
            for c, count in ep["class_counts"].items():
                class_distribution[c][s] += count
                
            map_episodes_distribution[ep["map"]][s] += 1
            map_frames_distribution[ep["map"]][s] += ep["total_frames"]

        total_eps_global = len(self.episodes)
        total_frames_global = sum(total_frames_by_split.values())
        is_split_mode = len(self.splits) > 1 or "dataset" not in self.splits

        # 1. Створення Markdown звіту
        report = []
        report.append("# Фактичний аналітичний звіт стану структури датасету\n")
        report.append(f"> Аналіз проведено для каталогу: `{self.dataset_dir.name}`\n")
        report.append("Звіт відображає дійсний обсяг даних після виконання етапів очищення, дедуплікації та фільтрації.\n")
        
        # Розділ 1: Загальна статистика
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
            report.append(f"- **Загальна кількість фонових (негативних) відео**: {sum(negatives_by_split.values())}\n")

        # Розділ 2: Покласовий розподіл
        report.append("## 2. Покласовий розподіл кадрів (Class Frame Distribution)\n")
        if is_split_mode:
            headers = ["Клас", "Всього кадрів"] + [f"{s.capitalize()} (кадри / %)" for s in self.splits]
            report.append("| " + " | ".join(headers) + " |")
            report.append("| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |")
            
            for c in self.classes:
                vals = class_distribution[c]
                tot_c = sum(vals.values())
                if tot_c == 0:
                    continue
                row = [f"**{c}**", str(tot_c)]
                for s in self.splits:
                    cnt_s = vals[s]
                    pct_s = (cnt_s / tot_c * 100) if tot_c > 0 else 0
                    row.append(f"{cnt_s} ({pct_s:.1f}%)")
                report.append("| " + " | ".join(row) + " |")
        else:
            report.append("| Клас | Кількість дійсних кадрів | Відносна частка (%) |")
            report.append("| :--- | :---: | :---: |")
            total_annotated_frames = sum(sum(class_distribution[c].values()) for c in self.classes)
            for c in self.classes:
                tot_c = sum(class_distribution[c].values())
                pct = (tot_c / total_annotated_frames * 100) if total_annotated_frames > 0 else 0
                report.append(f"| **{c}** | {tot_c} | {pct:.1f}% |")
        report.append("\n")

        # Розділ 3: Географічний розподіл
        report.append("## 3. Розподіл за картами (Map Geographic Distribution)\n")
        if is_split_mode:
            headers = ["Карта", "Всього відео", "Всього кадрів"] + [f"{s.capitalize()} (відео/кадри)" for s in self.splits]
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
            report.append("| Назва карти | Кількість відео | Кількість кадрів | Частка кадрів (%) |")
            report.append("| :--- | :---: | :---: | :---: |")
            for m in self.maps:
                tot_m_eps = map_episodes_distribution[m]["dataset"]
                tot_m_fms = map_frames_distribution[m]["dataset"]
                pct = (tot_m_fms / total_frames_global * 100) if total_frames_global > 0 else 0
                report.append(f"| **{m}** | {tot_m_eps} | {tot_m_fms} | {pct:.1f}% |")
        report.append("\n")

        # Розділ 4: Негативи
        report.append("## 4. Стан фонових даних (Чисті негативні епізоди)\n")
        total_negatives = sum(negatives_by_split.values())
        report.append(f"Загальна кількість чистих негативних епізодів: **{total_negatives}**.\n")
        if total_negatives > 0 and is_split_mode:
            report.append("| Вибірка | Кількість фонових відео | Частка від загальної кількості (%) |")
            report.append("| :--- | :---: | :---: |")
            for s in self.splits:
                s_neg = negatives_by_split[s]
                s_pct = (s_neg / total_negatives * 100) if total_negatives > 0 else 0
                report.append(f"| **{s.capitalize()}** | {s_neg} | {s_pct:.1f}% |")
            report.append("\n")

        # Запис у файли
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report))
            
        output_json_data = {
            "metadata": {
                "dataset_root_dir": str(self.dataset_dir),
                "detected_classes": self.classes,
                "detected_maps": self.maps,
                "detected_splits": self.splits,
                "total_episodes": total_eps_global,
                "total_frames": total_frames_global
            },
            "statistics": {
                "episodes_by_split": total_episodes_by_split,
                "frames_by_split": total_frames_by_split,
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
        
        # Виведення консольного резюме
        print("\n" + "="*45)
        print("             КОНСОЛЬНЕ РЕЗЮМЕ")
        print("="*45)
        print(f"Всього епізодів : {total_eps_global}")
        print(f"Всього кадрів   : {total_frames_global}")
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
            logger.error("Не знайдено валідних епізодів за структурою map/ds/ep. Перевірте вказаний шлях.")
            return
            
        analyzer.generate_report(Path(args.output_report), Path(args.output_json))
        logger.info("Процес аналізу успішно завершено.")
        
    except Exception as e:
        logger.error(f"Виникла помилка під час роботи аналізатора: {e}", exc_info=True)


if __name__ == "__main__":
    main()