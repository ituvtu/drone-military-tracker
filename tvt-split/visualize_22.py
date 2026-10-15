#!/usr/bin/env python3
"""
Система візуалізації розмітки датасету (YOLO або CVAT XML) з підтримкою паралелізму.
Малює тонкі межі обмежувальних рамок виразних кольорів без текстових підписів.
"""

import os
import re
import yaml
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Налаштування логування
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("DatasetVisualizer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()


def safe_imread(path: Path) -> np.ndarray:
    """
    Безпечно зчитує зображення за допомогою OpenCV, підтримуючи Unicode-символи у шляху.
    """
    try:
        with open(path, 'rb') as f:
            bytes_data = f.read()
        arr = np.frombuffer(bytes_data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f"Не вдалося зчитати файл {path}: {e}")
        return None


def safe_imwrite(path: Path, img: np.ndarray) -> bool:
    """
    Безпечно записує зображення за допомогою OpenCV, створюючи необхідні папки.
    """
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
    """
    Генерує унікальний виразний колір (BGR) на основі текстового імені або індексу класу.
    """
   
    return 0, 255, 0


def detect_dataset_type(dataset_dir: Path) -> str:
    """
    Визначає тип структури датасету.
    """
    if (dataset_dir / "data.yaml").exists():
        return "yolo"
    return "cvat_xml"


def load_data_yaml(yaml_path: Path) -> Dict[int, str]:
    """
    Зчитує конфігурацію класів з YAML файлу.
    """
    class_mapping = {}
    if not yaml_path.exists():
        return class_mapping
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if data and 'names' in data:
                names = data['names']
                if isinstance(names, dict):
                    return {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, list):
                    return {i: str(v) for i, v in enumerate(names)}
    except Exception as e:
        logger.warning(f"Помилка зчитування YAML конфігурації: {e}")
    return class_mapping


def parse_cvat_xml_boxes(xml_path: Path, img_files: List[str]) -> Dict[str, List[Tuple[str, int, int, int, int]]]:
    """
    Парсить CVAT XML та зіставляє анотації з конкретними зображеннями.
    """
    boxes_by_image = {f: [] for f in img_files}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Сценарій 1: Track-based CVAT формат (відео)
        tracks = root.findall('.//track')
        if tracks:
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
                            if 0 <= frame_idx < len(sorted_img_files):
                                img_name = sorted_img_files[frame_idx]
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
    """
    Обробляє одне зображення: зчитує, малює рамки та зберігає в нову директорію.
    """
    try:
        img_path = Path(task['img_path'])
        out_path = Path(task['out_path'])
        
        img = safe_imread(img_path)
        if img is None:
            return False
            
        img_h, img_w = img.shape[:2]
        boxes_to_draw = []
        
        if task['type'] == 'yolo':
            label_path_str = task.get('label_path')
            class_mapping = task.get('class_mapping', {})
            if label_path_str:
                label_path = Path(label_path_str)
                if label_path.exists() and label_path.stat().st_size > 0:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    class_idx = int(parts[0])
                                    class_name = class_mapping.get(class_idx, f"class_{class_idx}")
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    w = float(parts[3])
                                    h = float(parts[4])
                                    
                                    xmin = int((x_center - w / 2) * img_w)
                                    ymin = int((y_center - h / 2) * img_h)
                                    xmax = int((x_center + w / 2) * img_w)
                                    ymax = int((y_center + h / 2) * img_h)
                                    
                                    boxes_to_draw.append((class_name, xmin, ymin, xmax, ymax))
                                except ValueError:
                                    pass
        else:
            boxes_to_draw = task.get('boxes', [])
            
        # Малюємо тонкі рамки
        for label, xmin, ymin, xmax, ymax in boxes_to_draw:
            color = label_to_color(label)
            # Обмежуємо координати рамками зображення
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
    parser = argparse.ArgumentParser(description="Паралельна візуалізація анотацій")
    parser.add_argument("--input_dir", type=str, default="../dataset_v22", help="Шлях до оригінального датасету")
    parser.add_argument("--output_dir", type=str, default="../dataset_v22_visualized", help="Шлях для збереження результатів")
    parser.add_argument("--thickness", type=int, default=2, help="Товщина ліній рамки")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Кількість потоків процесора")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not input_dir.exists():
        logger.error(f"Оригінальний датасет не знайдено: {input_dir}")
        return

    logger.info(f"Вхідний каталог: {input_dir}")
    logger.info(f"Вихідний каталог: {output_dir}")
    
    dataset_type = detect_dataset_type(input_dir)
    tasks = []
    
    if dataset_type == "yolo":
        logger.info("Виявлено структуру YOLO. Формування переліку завдань...")
        yaml_path = input_dir / "data.yaml"
        class_mapping = load_data_yaml(yaml_path)
        
        # Копіюємо файл налаштувань у вихідну теку, якщо він є
        if yaml_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(output_dir / "data.yaml", 'w', encoding='utf-8') as f:
                    yaml.safe_dump(yaml.safe_load(open(yaml_path, 'r', encoding='utf-8')), f)
            except Exception as e:
                logger.warning(f"Не вдалося скопіювати data.yaml: {e}")

        images_dir = input_dir / "images"
        splits = [d.name for d in images_dir.iterdir() if d.is_dir()] if images_dir.exists() else []
        if not splits:
            splits = [""]
            
        for split in splits:
            split_img_dir = images_dir / split if split else images_dir
            for root_dir, _, files in os.walk(split_img_dir):
                current_path = Path(root_dir)
                img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img_file in img_files:
                    img_path = current_path / img_file
                    rel_path = img_path.relative_to(input_dir)
                    out_img_path = output_dir / rel_path
                    
                    try:
                        rel_to_split = img_path.relative_to(images_dir / split) if split else img_path.relative_to(images_dir)
                        label_path = input_dir / "labels" / split / rel_to_split.with_suffix(".txt")
                    except Exception:
                        label_path = input_dir / "labels" / img_path.with_suffix(".txt").name
                        
                    tasks.append({
                        'type': 'yolo',
                        'img_path': str(img_path),
                        'out_path': str(out_img_path),
                        'label_path': str(label_path) if label_path.exists() else None,
                        'class_mapping': class_mapping
                    })
    else:
        logger.info("Виявлено структуру CVAT XML. Формування переліку завдань...")
        for root_dir, _, files in os.walk(input_dir):
            current_path = Path(root_dir)
            img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if img_files:
                xml_path = current_path / "annotations.xml"
                boxes_by_image = {}
                if xml_path.exists():
                    boxes_by_image = parse_cvat_xml_boxes(xml_path, img_files)
                    
                for img_file in img_files:
                    img_path = current_path / img_file
                    rel_path = img_path.relative_to(input_dir)
                    out_img_path = output_dir / rel_path
                    
                    tasks.append({
                        'type': 'cvat',
                        'img_path': str(img_path),
                        'out_path': str(out_img_path),
                        'boxes': boxes_by_image.get(img_file, [])
                    })

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.warning("Не виявлено файлів зображень для обробки.")
        return

    logger.info(f"Знайдено зображень до обробки: {total_tasks}")
    
    # Виконання з використанням паралельного пулу процесів
    success_count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_image, task, args.thickness): task for task in tasks}
        
        for fut in tqdm(as_completed(futures), total=total_tasks, desc="Візуалізація", unit="кадр"):
            if fut.result():
                success_count += 1
                
    logger.info(f"Візуалізацію успішно завершено. Успішно збережено {success_count} з {total_tasks} кадрів.")


if __name__ == "__main__":
    main()