#!/usr/bin/env python3
"""
Система візуалізації розмітки датасету CVAT XML для структури dataset_v21 з підтримкою паралелізму.
Використовує метадані для точного мапінгу кадрів у разі наявності видалених файлів.
"""

import os
import re
import json
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("DatasetVisualizer")


def safe_imread(path: Path) -> np.ndarray:
    try:
        with open(path, 'rb') as f:
            bytes_data = f.read()
        arr = np.frombuffer(bytes_data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"Не вдалося зчитати файл {path}: {e}")
        return None


def safe_imwrite(path: Path, img: np.ndarray) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix
        success, encoded_img = cv2.imencode(ext, img)
        if success:
            with open(path, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        return False
    except Exception as e:
        logger.warning(f"Не вдалося записати файл {path}: {e}")
        return False


def label_to_color(label: str) -> Tuple[int, int, int]:
    hash_val = sum(ord(char) * (i + 1) for i, char in enumerate(label))
    r = int(60 + (hash_val * 73) % 180)
    g = int(60 + (hash_val * 139) % 180)
    b = int(60 + (hash_val * 197) % 180)
    return b, g, r


def load_metadata_mapping(episode_path: Path) -> Dict[int, str]:
    """
    Зчитує metadata.txt та будує точний мапінг: frame_index (int) -> filename (str).
    Дозволяє уникнути десинхронізації через видалені кадри.
    """
    mapping = {}
    meta_path = episode_path / "metadata.txt"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                frames = data.get("frames", [])
                for idx, frame_info in enumerate(frames):
                    name = frame_info.get("name")
                    if name:
                        mapping[idx] = name
        except Exception as e:
            logger.warning(f"Не вдалося зчитати metadata.txt у {episode_path}: {e}")
    return mapping


def parse_cvat_xml_boxes(xml_path: Path, 
                         img_files: List[str], 
                         frame_to_filename: Dict[int, str] = None) -> Dict[str, List[Tuple[str, int, int, int, int]]]:
    """
    Парсить CVAT XML та зіставляє анотації з конкретними зображеннями в папці епізоду.
    """
    boxes_by_image = {f: [] for f in img_files}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        tracks = root.findall('.//track')
        if tracks:
            # Якщо метаданих немає, готуємо стандартний порядковий fallback
            sorted_img_files = []
            if not frame_to_filename:
                def natural_sort_key(s):
                    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
                sorted_img_files = sorted(img_files, key=natural_sort_key)
            
            for track in tracks:
                label = track.get('label', 'unknown')
                for box in track.findall('box'):
                    outside = box.get('outside')
                    if outside in ('1', 'true'):
                        continue
                    frame = box.get('frame')
                    if frame is not None:
                        try:
                            frame_idx = int(frame)
                            img_name = None
                            
                            # Намагаємось отримати ім'я кадру через метадані
                            if frame_to_filename:
                                img_name = frame_to_filename.get(frame_idx)
                            # Якщо метаданих немає, використовуємо порядковий fallback
                            elif 0 <= frame_idx < len(sorted_img_files):
                                img_name = sorted_img_files[frame_idx]
                                
                            # Додаємо бокс тільки якщо кадр знайдено і він є серед реальних файлів на диску
                            if img_name and img_name in boxes_by_image:
                                xtl = int(float(box.get('xtl', 0)))
                                ytl = int(float(box.get('ytl', 0)))
                                xbr = int(float(box.get('xbr', 0)))
                                ybr = int(float(box.get('ybr', 0)))
                                boxes_by_image[img_name].append((label, xtl, ytl, xbr, ybr))
                        except ValueError:
                            pass
            return boxes_by_image
            
        # Сценарій 2: Image-based CVAT формат
        images = root.findall('.//image')
        if images:
            for img_elem in images:
                img_name = img_elem.get('name') or ""
                matched_file = None
                if img_name in boxes_by_image:
                    matched_file = img_name
                else:
                    for f in img_files:
                        if Path(f).name == img_name or Path(f).stem == img_name:
                            matched_file = f
                            break
                if not matched_file:
                    continue
                    
                for box in img_elem.findall('box'):
                    label = box.get('label', 'unknown')
                    xtl = int(float(box.get('xtl', 0)))
                    ytl = int(float(box.get('ytl', 0)))
                    xbr = int(float(box.get('xbr', 0)))
                    ybr = int(float(box.get('ybr', 0)))
                    boxes_by_image[matched_file].append((label, xtl, ytl, xbr, ybr))
            return boxes_by_image
            
    except Exception as e:
        logger.warning(f"Не вдалося розпарсити CVAT XML {xml_path}: {e}")
    return boxes_by_image


def process_single_image(task: Dict[str, Any], thickness: int) -> bool:
    try:
        img_path = Path(task['img_path'])
        out_path = Path(task['out_path'])
        
        img = safe_imread(img_path)
        if img is None:
            return False
            
        img_h, img_w = img.shape[:2]
        boxes_to_draw = task.get('boxes', [])
            
        for label, xmin, ymin, xmax, ymax in boxes_to_draw:
            color = label_to_color(label)
            xmin_c = max(0, min(xmin, img_w - 1))
            ymin_c = max(0, min(ymin, img_h - 1))
            xmax_c = max(0, min(xmax, img_w - 1))
            ymax_c = max(0, min(ymax, img_h - 1))
            
            cv2.rectangle(img, (xmin_c, ymin_c), (xmax_c, ymax_c), color, thickness)
            
        return safe_imwrite(out_path, img)
    except Exception as e:
        logger.warning(f"Помилка під час обробки кадру {task.get('img_path')}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Паралельна візуалізація анотацій CVAT XML для структури dataset_v21")
    parser.add_argument("--input-dir", type=str, default="../dataset_v25_640_merged", help="Шлях до оригінального датасету dataset_v21")
    parser.add_argument("--output-dir", type=str, default="../dataset_v25_640_merged_visualized", help="Шлях для збереження результатів візуалізації")
    parser.add_argument("--thickness", type=int, default=2, help="Товщина ліній обмежувальної рамки")
    parser.add_argument("--workers", type=int, default=None, help="Кількість потоків процесора")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not input_dir.exists():
        logger.error(f"Оригінальний датасет не знайдено: {input_dir}")
        return

    logger.info("Сканування структури dataset_v21 та формування завдань...")
    tasks = []
    episode_count = 0
    
    for root_dir, _, files in os.walk(input_dir):
        current_path = Path(root_dir)
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if img_files:
            xml_path = current_path / "annotations.xml"
            if xml_path.exists():
                episode_count += 1
                
                # Завантажуємо мапінг метаданих
                frame_to_filename = load_metadata_mapping(current_path)
                
                # Парсимо XML файл анотацій з урахуванням мапінгу метаданих
                boxes_by_image = parse_cvat_xml_boxes(xml_path, img_files, frame_to_filename)
                
                for img_file in img_files:
                    img_path = current_path / img_file
                    rel_path = img_path.relative_to(input_dir)
                    out_img_path = output_dir / rel_path
                    
                    tasks.append({
                        'img_path': str(img_path),
                        'out_path': str(out_img_path),
                        'boxes': boxes_by_image.get(img_file, [])
                    })

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.warning("Не виявлено епізодів для обробки.")
        return

    logger.info(f"Знайдено епізодів: {episode_count}")
    logger.info(f"Загальна кількість кадрів до обробки: {total_tasks}")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_image, task, args.thickness): task for task in tasks}
        
        for fut in tqdm(as_completed(futures), total=total_tasks, desc="Візуалізація", unit="кадр"):
            if fut.result():
                success_count += 1
                
    logger.info(f"Процес завершено. Візуалізовано та успішно збережено {success_count} з {total_tasks} кадрів.")


if __name__ == "__main__":
    main()