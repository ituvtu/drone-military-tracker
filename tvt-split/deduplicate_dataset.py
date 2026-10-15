#!/usr/bin/env python3
"""
Система дедуплікації з гнучким налаштуванням порогів dHash для кожної карти, 
класу та типу дедуплікації (внутрішня / крос-вибіркова).
"""

import os
import re
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise ImportError("Встановіть залежності: pip install Pillow numpy")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Deduplicator")


def frame_idx_from_name(name: str) -> int:
    match = re.search(r'\d+', name)
    if match:
        return int(match.group()) - 1
    raise ValueError(f"Неможливо визначити індекс кадру з імені файлу: {name}")


class ImageProcessor:
    @staticmethod
    def calculate_dhash(image_path: Path) -> int:
        with Image.open(image_path) as img:
            img_gray = img.convert('L').resize((9, 8), Image.Resampling.LANCZOS)
            arr = np.array(img_gray, dtype=np.uint8)
            
        diff = arr[:, :-1] > arr[:, 1:]
        flat_diff = diff.ravel()
        decimal_value = 0
        for bit in flat_diff:
            decimal_value = (decimal_value << 1) | int(bit)
        return decimal_value

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        return bin(hash1 ^ hash2).count('1')

    @staticmethod
    def get_blur_variance(image_path: Path) -> float:
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


class AnnotationManager:
    @staticmethod
    def get_episode_classes(xml_path: Path) -> Set[str]:
        """Парсить XML та повертає унікальну множину класів в епізоді."""
        if not xml_path.exists():
            return set()
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            labels = set()
            for track in root.findall('.//track'):
                label = track.get('label')
                if label:
                    labels.add(label)
            for img in root.findall('.//image'):
                label = img.get('label')
                if label:
                    labels.add(label)
                for box in img.findall('box'):
                    label = box.get('label')
                    if label:
                        labels.add(label)
            return labels
        except Exception as e:
            logger.warning(f"Не вдалося зчитати класи з XML {xml_path}: {e}")
            return set()

    @staticmethod
    def update_cvat_xml(xml_path: Path, kept_frame_names: Set[str]):
        if not xml_path.exists():
            return
            
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            sorted_kept = sorted(list(kept_frame_names))
            orig_to_new = {}
            for new_idx, name in enumerate(sorted_kept):
                try:
                    orig_idx = frame_idx_from_name(name)
                    orig_to_new[orig_idx] = str(new_idx)
                except ValueError:
                    continue
                    
            images_found = root.findall('.//image')
            if images_found:
                for img_elem in images_found:
                    img_name = img_elem.get("name")
                    if img_name not in kept_frame_names:
                        root.remove(img_elem)
                    else:
                        img_elem.set("id", orig_to_new[frame_idx_from_name(img_name)])
                        
            tracks_found = root.findall('.//track')
            if tracks_found:
                for track_elem in tracks_found:
                    boxes = track_elem.findall('box')
                    for box_elem in boxes:
                        orig_frame = int(box_elem.get("frame"))
                        if orig_frame in orig_to_new:
                            box_elem.set("frame", orig_to_new[orig_frame])
                        else:
                            track_elem.remove(box_elem)
                            
                    if len(track_elem.findall('box')) == 0:
                        root.remove(track_elem)
                        
            ET.indent(tree, space="  ")
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            
        except Exception as e:
            logger.error(f"Помилка оновлення розмітки XML {xml_path}: {e}")


class DatasetDeduplicator:
    def __init__(self, dataset_dir: Path, pos_threshold: int = 3, neg_threshold: int = 14):
        self.dataset_dir = Path(dataset_dir).resolve()
        
        # Базові глобальні конфігурації
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        
        # --- СПЕЦИФІЧНІ НАЛАШТУВАННЯ ПОРОГІВ ---
        
        # Пороги для внутрішньої (Intra-episode) дедуплікації
        self.map_thresholds_internal = {
            "map_2": 3,  # Нормальне очищення всередині відео на map_2
        }
        self.class_thresholds_internal = {
            # Базовий поріг 3 зазвичай підходить для всіх класів
        }
        
        # Пороги для крос-вибіркової (Inter-split) дедуплікації (усунення витоків)
        self.map_thresholds_cross = {
            "map_2": 1,  # Знижуємо до 1, щоб фон map_2 не спричиняв зникнення епізодів
        }
        self.class_thresholds_cross = {
            "howitzer": 1,  # Послаблюємо поріг для howitzer, щоб зберегти кадри
            "bmp": 2,       # Злегка послаблюємо для bmp
        }
        
        # --------------------------------------

        self.train_dir = self.dataset_dir / "train"
        self.val_dir = self.dataset_dir / "val"
        self.test_dir = self.dataset_dir / "test"
        
        self.is_split = self.train_dir.exists() or self.val_dir.exists() or self.test_dir.exists()
        
        self.registries: Dict[str, Dict[str, List[Tuple[int, str]]]] = {
            "train": {}, "val": {}, "test": {}, "dataset": {}
        }
        
        self.stats = {
            "train_processed": 0, "train_removed_duplicates": 0,
            "val_processed": 0, "val_removed_duplicates": 0, "val_removed_leaks": 0,
            "test_processed": 0, "test_removed_duplicates": 0, "test_removed_leaks": 0,
            "dataset_processed": 0, "dataset_removed_duplicates": 0,
            "leaks_details": []
        }

    def resolve_threshold(self, ep_path: Path, is_cross: bool = False) -> int:
        """Динамічно визначає поріг dHash на основі типу епізоду, карти та класів."""
        # 1. Негативні епізоди
        is_neg = "neg" in ep_path.name.lower() or not (ep_path / "annotations.xml").exists()
        if is_neg:
            return self.neg_threshold
            
        # 2. Визначаємо назву карти з шляху
        map_name = "default"
        for part in ep_path.parts:
            if part.startswith("map_"):
                map_name = part
                break
                
        # 3. Визначаємо наявні класи
        classes = AnnotationManager.get_episode_classes(ep_path / "annotations.xml")
        
        # Вибираємо базові конфігурації залежно від типу дедуплікації (внутрішня/крос)
        map_rules = self.map_thresholds_cross if is_cross else self.map_thresholds_internal
        class_rules = self.class_thresholds_cross if is_cross else self.class_thresholds_internal
        
        # 4. Розрахунок порогу: спочатку перевіряємо карту
        resolved_threshold = map_rules.get(map_name, self.pos_threshold)
        
        # 5. Перевіряємо класи: якщо хоча б один клас в епізоді має специфічний ліміт,
        # використовуємо найбільш консервативний (мінімальний) поріг для захисту рідкісних даних
        class_thresholds_found = []
        for cls in classes:
            if cls in class_rules:
                class_thresholds_found.append(class_rules[cls])
                
        if class_thresholds_found:
            resolved_threshold = min(class_thresholds_found)
            
        return resolved_threshold

    def _scan_episodes(self, split_dir: Path) -> List[Path]:
        episodes = []
        if not split_dir.exists():
            return episodes
        for root, dirs, _ in os.walk(split_dir):
            p = Path(root)
            if p.name.startswith("ep_") or "_neg" in p.name.lower():
                episodes.append(p)
        return sorted(episodes)

    def _deduplicate_single_split(self, split_dir: Path, split_name: str) -> Dict[str, List[Tuple[int, str]]]:
        logger.info(f"Внутрішня дедуплікація вибірки {split_name.upper()}...")
        episodes = self._scan_episodes(split_dir)
        
        total_files = 0
        for ep_path in episodes:
            total_files += sum(1 for f in ep_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
            
        pbar = tqdm(total=total_files, desc=f"Дедуплікація {split_name.upper()}", unit="кадр") if tqdm else None
        split_registry: Dict[str, List[Tuple[int, str]]] = {}
        processed_count = 0
        removed_count = 0
        
        for ep_path in episodes:
            try:
                map_name = ep_path.relative_to(split_dir).parts[0]
            except Exception:
                map_name = "unknown"
                
            if map_name not in split_registry:
                split_registry[map_name] = []
                
            # Вирішення динамічного порогу
            current_threshold = self.resolve_threshold(ep_path, is_cross=False)
            
            img_files = sorted([f for f in ep_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if not img_files:
                continue
                
            kept_frames: List[Path] = []
            active_frame: Optional[Path] = None
            active_hash: int = 0
            active_blur: float = 0.0
            
            for img_file in img_files:
                processed_count += 1
                h_val = ImageProcessor.calculate_dhash(img_file)
                b_val = ImageProcessor.get_blur_variance(img_file)
                
                if active_frame is None:
                    active_frame = img_file
                    active_hash = h_val
                    active_blur = b_val
                    kept_frames.append(img_file)
                    if pbar:
                        pbar.update(1)
                    continue
                    
                dist = ImageProcessor.hamming_distance(active_hash, h_val)
                if dist <= current_threshold:
                    removed_count += 1
                    if b_val > active_blur:
                        try:
                            active_frame.unlink()
                        except FileNotFoundError:
                            pass
                        if active_frame in kept_frames:
                            kept_frames.remove(active_frame)
                        
                        active_frame = img_file
                        active_hash = h_val
                        active_blur = b_val
                        kept_frames.append(img_file)
                    else:
                        img_file.unlink()
                else:
                    active_frame = img_file
                    active_hash = h_val
                    active_blur = b_val
                    kept_frames.append(img_file)
                    
                if pbar:
                    pbar.update(1)
                    
            kept_names = {f.name for f in kept_frames}
            AnnotationManager.update_cvat_xml(ep_path / "annotations.xml", kept_names)
            
            for f in kept_frames:
                f_hash = ImageProcessor.calculate_dhash(f)
                rel_path = f.relative_to(self.dataset_dir).as_posix()
                split_registry[map_name].append((f_hash, rel_path))
                
        if pbar:
            pbar.close()
            
        self.stats[f"{split_name.lower()}_processed"] = processed_count
        self.stats[f"{split_name.lower()}_removed_duplicates"] = removed_count
        return split_registry

    def remove_cross_split_leakage(self, split_dir: Path, split_name: str, compare_against_splits: List[str]):
        logger.info(f"Крос-дедуплікація: Перевірка {split_name.upper()} проти {', '.join([s.upper() for s in compare_against_splits])}...")
        episodes = self._scan_episodes(split_dir)
        
        total_files = 0
        for ep_path in episodes:
            total_files += sum(1 for f in ep_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
            
        pbar = tqdm(total=total_files, desc=f"Крос-очищення {split_name.upper()}", unit="кадр") if tqdm else None
        removed_key = f"{split_name.lower()}_removed_leaks"
        updated_registry_entries: Dict[str, List[Tuple[int, str]]] = {}
        
        for ep_path in episodes:
            try:
                map_name = ep_path.relative_to(split_dir).parts[0]
            except Exception:
                map_name = "unknown"
                
            if map_name not in updated_registry_entries:
                updated_registry_entries[map_name] = []
                
            # Вирішення динамічного порогу для крос-спліту
            current_threshold = self.resolve_threshold(ep_path, is_cross=True)
            
            img_files = sorted([f for f in ep_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            if not img_files:
                continue
                
            kept_frames: List[Path] = []
            
            for img_file in img_files:
                h_val = ImageProcessor.calculate_dhash(img_file)
                is_leak = False
                matched_file = ""
                min_dist = current_threshold + 1
                
                for base_split in compare_against_splits:
                    registry = self.registries[base_split]
                    map_registry = registry.get(map_name, [])
                    
                    for t_hash, t_rel in map_registry:
                        dist = ImageProcessor.hamming_distance(h_val, t_hash)
                        if dist <= current_threshold:
                            is_leak = True
                            matched_file = t_rel
                            min_dist = dist
                            break
                    if is_leak:
                        break
                        
                if is_leak:
                    self.stats[removed_key] += 1
                    rel_self = img_file.relative_to(self.dataset_dir).as_posix()
                    self.stats["leaks_details"].append({
                        "leak_file": rel_self,
                        "matched_train": matched_file,
                        "distance": min_dist
                    })
                    img_file.unlink()
                else:
                    kept_frames.append(img_file)
                    rel_self = img_file.relative_to(self.dataset_dir).as_posix()
                    updated_registry_entries[map_name].append((h_val, rel_self))
                    
                if pbar:
                    pbar.update(1)
                    
            kept_names = {f.name for f in kept_frames}
            AnnotationManager.update_cvat_xml(ep_path / "annotations.xml", kept_names)
            
        if pbar:
            pbar.close()
            
        self.registries[split_name.lower()] = updated_registry_entries

    def write_markdown_report(self):
        report_path = self.dataset_dir / "deduplication_report.md"
        md = []
        md.append("# Системний звіт про дедуплікацію та очищення витоків (Параметризований)\n")
        md.append("> Звіт сформовано автоматично модулем `Deduplicator`.\n")
        
        if not self.is_split:
            p = self.stats.get("dataset_processed", 0)
            rem = self.stats.get("dataset_removed_duplicates", 0)
            kept = p - rem
            md.append("## 1. Загальна статистика дедуплікації (Нерозділений датасет)\n")
            md.append("| Кількість оброблених кадрів | Видалено внутрішніх дублікатів | Збережено унікальних кадрів |")
            md.append("| :---: | :---: | :---: |")
            md.append(f"| {p} | {rem} | **{kept}** |\n")
            reduction = (rem / p * 100) if p > 0 else 0.0
            md.append(f"Загальне зменшення об'єму: **{reduction:.2f}%**\n")
        else:
            tr_p = self.stats["train_processed"]
            tr_rem_dup = self.stats["train_removed_duplicates"]
            tr_kept = tr_p - tr_rem_dup

            vl_p = self.stats["val_processed"]
            vl_rem_dup = self.stats["val_removed_duplicates"]
            vl_rem_leak = self.stats["val_removed_leaks"]
            vl_kept = vl_p - vl_rem_dup - vl_rem_leak

            ts_p = self.stats["test_processed"]
            ts_rem_dup = self.stats["test_removed_duplicates"]
            ts_rem_leak = self.stats["test_removed_leaks"]
            ts_kept = ts_p - ts_rem_dup - ts_rem_leak

            total_p = tr_p + vl_p + ts_p
            total_dup = tr_rem_dup + vl_rem_dup + ts_rem_dup
            total_leak = vl_rem_leak + ts_rem_leak
            total_kept = tr_kept + vl_kept + ts_kept
            
            md.append("## 1. Загальна статистика дедуплікації\n")
            md.append("| Вибірка | Початково кадрів | Видалено дублікатів (внутрішніх) | Видалено витоків (міжвибіркових) | Збережено унікальних кадрів |")
            md.append("| :--- | :---: | :---: | :---: | :---: |")
            md.append(f"| **Train** | {tr_p} | {tr_rem_dup} | - | **{tr_kept}** |")
            md.append(f"| **Val** | {vl_p} | {vl_rem_dup} | {vl_rem_leak} | **{vl_kept}** |")
            md.append(f"| **Test** | {ts_p} | {ts_rem_dup} | {ts_rem_leak} | **{ts_kept}** |")
            md.append(f"| **Всього** | **{total_p}** | **{total_dup}** | **{total_leak}** | **{total_kept}** |\n")
            
            reduction = ((total_dup + total_leak) / total_p * 100) if total_p > 0 else 0.0
            md.append(f"Загальне зменшення об'єму вибірок: **{reduction:.2f}%**\n")
            
        report_path.write_text("\n".join(md), encoding="utf-8")

    def run(self):
        logger.info("=" * 60)
        logger.info("ЗАПУСК ПРОЦЕСУ ДЕДУПЛІКАЦІЇ (ЛОКАЛІЗОВАНІ ПОРОГИ)")
        logger.info("=" * 60)
        logger.info(f"Базові параметри: Глобальний об'єктний = {self.pos_threshold} | Глобальний фоновий = {self.neg_threshold}")
        logger.info(f"Специфічні внутрішні пороги: за картами = {self.map_thresholds_internal} | за класами = {self.class_thresholds_internal}")
        logger.info(f"Специфічні крос-пороги: за картами = {self.map_thresholds_cross} | за класами = {self.class_thresholds_cross}")
        
        if not self.is_split:
            logger.info("ВИЯВЛЕНО НЕРОЗДІЛЕНИЙ ДАТАСЕТ. ЗАПУСК ДЕДУПЛІКАЦІЇ...")
            self.registries["dataset"] = self._deduplicate_single_split(self.dataset_dir, "dataset")
            self.write_markdown_report()
            
            p = self.stats.get("dataset_processed", 0)
            rem = self.stats.get("dataset_removed_duplicates", 0)
            kept = p - rem
            
            print("\n" + "=" * 50)
            print(" РЕЗУЛЬТАТИ ВНУТРІШНЬОЇ ДЕДУПЛІКАЦІЇ")
            print("=" * 50)
            print(f" Всього оброблено кадрів: {p}")
            print(f" Усунено дублікатів: {rem}")
            print(f" Збережено кадрів: {kept}")
            print("=" * 50 + "\n")
            return

        logger.info("ФАЗА 1: ВНУТРІШНЬОВИБІРКОВА ДЕДУПЛІКАЦІЯ")
        self.registries["train"] = self._deduplicate_single_split(self.train_dir, "train")
        self.registries["val"] = self._deduplicate_single_split(self.val_dir, "val")
        self.registries["test"] = self._deduplicate_single_split(self.test_dir, "test")
        
        logger.info("ФАЗА 2: КРОС-ВИБІРКОВА ДЕДУПЛІКАЦІЯ (УСУНЕННЯ ВИТОКІВ)")
        self.remove_cross_split_leakage(self.val_dir, "val", compare_against_splits=["train"])
        self.remove_cross_split_leakage(self.test_dir, "test", compare_against_splits=["train", "val"])
        
        self.write_markdown_report()
        
        tr_p = self.stats["train_processed"]
        tr_kept = tr_p - self.stats["train_removed_duplicates"]
        vl_p = self.stats["val_processed"]
        vl_kept = vl_p - self.stats["val_removed_duplicates"] - self.stats["val_removed_leaks"]
        ts_p = self.stats["test_processed"]
        ts_kept = ts_p - self.stats["test_removed_duplicates"] - self.stats["test_removed_leaks"]
        
        total_dup = self.stats["train_removed_duplicates"] + self.stats["val_removed_duplicates"] + self.stats["test_removed_duplicates"]
        total_leak = self.stats["val_removed_leaks"] + self.stats["test_removed_leaks"]
        
        print("\n" + "=" * 50)
        print(" РЕЗУЛЬТАТИ ДЕДУПЛІКАЦІЇ")
        print("=" * 50)
        print(f" Train (Початково / Залишено): {tr_p} -> {tr_kept}")
        print(f" Val   (Початково / Залишено): {vl_p} -> {vl_kept}")
        print(f" Test  (Початково / Залишено): {ts_p} -> {ts_kept}")
        print("-" * 50)
        print(f" Усунено внутрішніх дублікатів:  {total_dup}")
        print(f" Усунено крос-вибіркових витоків: {total_leak}")
        print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Очищення та дедуплікація розділеного або нерозділеного датасету")
    parser.add_argument("--dataset_dir", type=str, default="dataset_v12_cleaned", help="Шлях до папки з датасетом")
    parser.add_argument("--pos_threshold", type=int, default=3, help="Базовий поріг dHash для позитивних кадрів")
    parser.add_argument("--neg_threshold", type=int, default=14, help="Базовий поріг dHash для негативних кадрів")
    
    args = parser.parse_args()
    
    try:
        deduplicator = DatasetDeduplicator(
            dataset_dir=Path(args.dataset_dir),
            pos_threshold=args.pos_threshold,
            neg_threshold=args.neg_threshold
        )
        deduplicator.run()
    except Exception as e:
        logger.error(f"Критична помилка під час виконання дедуплікації: {e}", exc_info=True)


if __name__ == "__main__":
    main()