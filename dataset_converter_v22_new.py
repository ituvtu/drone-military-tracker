#!/usr/bin/env python3
"""
Скрипт для реструктуризації датасету зі структури v21 у формат YOLO (v22).
Отримує актуальні анотації безпосередньо з сервера CVAT (у форматі CVAT for images 1.1),
копіює локальні кадри з префіксом епізоду та конвертує розмітку в формат YOLO (.txt).
"""

import os
import re
import sys
import time
import shutil
import zipfile
import tempfile
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, urlunparse, urljoin
from typing import Dict, List, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DatasetConverterV22")

try:
    import requests
except ImportError:
    logger.error("The 'requests' library is required. Install it using: pip install requests")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    logger.error("The 'Pillow' library is required. Install it using: pip install Pillow")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Заглушка, якщо tqdm не встановлено
    def tqdm(iterable, **kwargs):
        return iterable


DEFAULT_CLASSES = ["howitzer", "antiair", "bmp", "tank", "cars", "truck"]

# =====================================================================
# ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ СОРТУВАННЯ ТА CVAT API
# =====================================================================

def get_canonical_task_prefix(task_name: str) -> str:
    pattern = r"^(m\d+)[_](s\d+)[_](ep_\d+)(_[a-zA-Z0-9\u0400-\u04FF]+)?$"
    match = re.match(pattern, task_name)
    if match:
        m_part = match.group(1)
        s_part = match.group(2)
        ep_part = match.group(3)
        suffix_part = match.group(4) if match.group(4) else ""
        return f"{m_part}_{s_part}_{ep_part}{suffix_part}"
    return task_name


def parse_task_path(task_name: str) -> bool:
    if "_del" in task_name:
        return False
    pattern = r"^(m\d+)[_](s\d+)[_](ep_\d+)(_[a-zA-Z0-9\u0400-\u04FF]+)?$"
    return bool(re.match(pattern, task_name))


def get_expected_local_path(task_name: str, input_dir: Path) -> Path:
    """
    Визначає очікуваний локальний шлях епізоду на основі імені таски в CVAT.
    Враховує суфікс-літерал безпосередньо в назві папки епізоду.
    """
    pattern = r"^(m\d+)[_](s\d+)[_](ep_\d+)(_[a-zA-Z0-9\u0400-\u04FF]+)?$"
    match = re.match(pattern, task_name)
    if match:
        m_part = match.group(1)
        s_part = match.group(2)
        ep_base = match.group(3)
        suffix_part = match.group(4) if match.group(4) else ""
        
        ep_folder = f"{ep_base}{suffix_part}"
        primary_path = input_dir / m_part / s_part / ep_folder
        if primary_path.exists():
            return primary_path
            
        # Запобігання помилкам через змішування латинської 'c' та кириличної 'с' в іменах папок
        fallback_suffix = suffix_part
        if "c" in fallback_suffix:
            fallback_suffix = fallback_suffix.replace("c", "с")
        elif "с" in fallback_suffix:
            fallback_suffix = fallback_suffix.replace("с", "c")
            
        fallback_path = input_dir / m_part / s_part / f"{ep_base}{fallback_suffix}"
        if fallback_path.exists():
            return fallback_path
            
        return primary_path
    return None


def get_authenticated_session(username, password) -> requests.Session:
    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({"Accept": "application/vnd.cvat+json"})
    return session


def make_request_with_retries(session, method, url, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            response = session.request(method, url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2**attempt)


def get_download_url(server_url: str, result_url: str) -> str:
    parsed_server = urlparse(server_url)
    parsed_result = urlparse(result_url)
    
    if not parsed_result.netloc:
        return urljoin(server_url, result_url)
        
    reconstructed = urlunparse((
        parsed_server.scheme,
        parsed_server.netloc,
        parsed_result.path,
        parsed_result.params,
        parsed_result.query,
        parsed_result.fragment
    ))
    return reconstructed


def initiate_export(session, server_url, task_id) -> str:
    export_url = f"{server_url}/api/tasks/{task_id}/dataset/export"
    params = {
        "format": "CVAT for images 1.1",
        "save_images": "false"
    }
    
    response = make_request_with_retries(session, "POST", export_url, params=params)
    
    if response.status_code in [201, 202, 409]:
        return response.json().get("rq_id")
        
    response.raise_for_status()
    return None


def convert_to_yolo(xtl, ytl, xbr, ybr, img_w, img_h):
    xtl = max(0.0, min(float(xtl), img_w))
    xbr = max(0.0, min(float(xbr), img_w))
    ytl = max(0.0, min(float(ytl), img_h))
    ybr = max(0.0, min(float(ybr), img_h))
    
    x_center = ((xtl + xbr) / 2.0) / img_w
    y_center = ((ytl + ybr) / 2.0) / img_h
    w = (xbr - xtl) / img_w
    h = (ybr - ytl) / img_h
    
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return x_center, y_center, w, h


def get_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size


# =====================================================================
# ПАРСИНГ XML ТА ФОРМУВАННЯ ДАТАСЕТУ (CVAT FOR IMAGES)
# =====================================================================

def process_xml_and_copy_data(xml_path: Path, task_name: str, local_ep_dir: Path, 
                              output_dir: Path, classes_list: List[str], dry_run: bool = False) -> Tuple[int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    canonical_prefix = get_canonical_task_prefix(task_name).lower()
    
    frame_map = {}
    images_in_xml = root.findall(".//image")
    
    for img_el in images_in_xml:
        img_name = img_el.get("name")
        if img_name:
            frame_map[img_name] = img_name

    dataset_labels_accumulator = {}

    # Обробка виключно формату CVAT for images
    for img_el in images_in_xml:
        img_name = img_el.get("name")
        if not img_name:
            continue
            
        frame_base, _ = os.path.splitext(img_name)
        src_img_path = local_ep_dir / img_name
        
        if not src_img_path.exists():
            continue
            
        try:
            img_w, img_h = get_image_size(src_img_path)
        except Exception:
            img_w = float(img_el.get("width", 1280))
            img_h = float(img_el.get("height", 720))
            
        boxes = img_el.findall("box")
        if boxes:
            if frame_base not in dataset_labels_accumulator:
                dataset_labels_accumulator[frame_base] = []
                
            for box in boxes:
                label = box.get("label")
                if label in classes_list:
                    class_idx = classes_list.index(label)
                    x, y, w, h = convert_to_yolo(box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr"), img_w, img_h)
                    dataset_labels_accumulator[frame_base].append(f"{class_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    success_count = 0
    error_count = 0

    # Копіювання кадрів та запис YOLO-файлів
    for img_name in frame_map.keys():
        frame_base, ext = os.path.splitext(img_name)
        src_img_path = local_ep_dir / img_name
        
        if not src_img_path.exists():
            continue
            
        target_img_name = f"{canonical_prefix}_{img_name}"
        target_lbl_name = f"{canonical_prefix}_{frame_base}.txt"
        
        target_img_path = output_dir / "images" / target_img_name
        target_lbl_path = output_dir / "labels" / target_lbl_name
        
        try:
            yolo_lines = dataset_labels_accumulator.get(frame_base, [])
            
            if not dry_run:
                with open(target_lbl_path, "w", encoding="utf-8") as lf:
                    if yolo_lines:
                        lf.write("\n".join(yolo_lines) + "\n")
                
                shutil.copy2(src_img_path, target_img_path)
                
            success_count += 1
        except Exception as e:
            logger.error(f"Помилка копіювання/розмітки кадру '{src_img_path}': {e}")
            error_count += 1
            
    return success_count, error_count


# =====================================================================
# ОБРОБКА КЛІЄНТСЬКОГО ПОТОКУ
# =====================================================================

def process_task_dataset(task: Dict[str, Any], local_ep_dir: Path, 
                         args: argparse.Namespace, classes_list: List[str]) -> Tuple[int, int]:
    task_id = task["id"]
    task_name = task["name"]
    output_dir = Path(args.output_dir).resolve()
    
    session = get_authenticated_session(args.username, args.password)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "export.zip").replace("\\", "/")
            
            logger.info(f"[{task_name}] Запит на експорт XML з CVAT (for images 1.1)...")
            rq_id = initiate_export(session, args.server_url, task_id)
            if not rq_id:
                logger.error(f"[{task_name}] Не вдалося отримати rq_id.")
                return 0, 0
                
            request_url = f"{args.server_url}/api/requests/{rq_id}"
            while True:
                status_response = make_request_with_retries(session, "GET", request_url)
                status_response.raise_for_status()
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "finished":
                    result_url = status_data.get("result_url")
                    break
                elif status == "failed":
                    logger.error(f"[{task_name}] Фонове завдання експорту на сервері завершилося помилкою.")
                    return 0, 0
                else:
                    time.sleep(3)
                    
            download_url = get_download_url(args.server_url, result_url)
            download_response = make_request_with_retries(session, "GET", download_url, stream=True)
            download_response.raise_for_status()
            
            with open(zip_path, "wb") as f:
                for chunk in download_response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    
            extract_dir = os.path.join(temp_dir, "extracted").replace("\\", "/")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
                
            xml_filename = "annotations.xml"
            xml_path = os.path.join(extract_dir, xml_filename).replace("\\", "/")
            if not os.path.exists(xml_path):
                found_xmls = [f for f in os.listdir(extract_dir) if f.lower().endswith(".xml")]
                if found_xmls:
                    xml_path = os.path.join(extract_dir, found_xmls[0]).replace("\\", "/")
                else:
                    logger.error(f"[{task_name}] XML-анотації в ZIP не знайдено.")
                    return 0, 0
                    
            success_count, error_count = process_xml_and_copy_data(
                xml_path=Path(xml_path),
                task_name=task_name,
                local_ep_dir=local_ep_dir,
                output_dir=output_dir,
                classes_list=classes_list,
                dry_run=args.dry_run
            )
            
            action_str = "[DRY-RUN] Оброблено б" if args.dry_run else "Успішно імпортовано"
            logger.info(f"[{task_name}] {action_str} {success_count} кадрів.")
            return success_count, error_count
            
    except Exception as e:
        logger.error(f"Помилка виконання завдання '{task_name}': {e}")
        return 0, 0


# =====================================================================
# ГОЛОВНИЙ ЦИКЛ (MAIN)
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Конвертація структури v21 у YOLO v22 із завантаженням розмітки з CVAT")
    parser.add_argument("--server-url", default="http://localhost:8080", help="CVAT сервер URL")
    parser.add_argument("--username", default="ituvtu", help="CVAT користувач")
    parser.add_argument("--password", default="3151333Vcvat", help="CVAT пароль")
    parser.add_argument("--input-dir", type=str, default="dataset_v21", help="Папка оригінального датасету v21")
    parser.add_argument("--output-dir", type=str, default="dataset_v22", help="Папка збереження структури v22")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Кількість паралельних потоків")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Список класів для фільтрації та мапінгу")
    parser.add_argument("--dry-run", action="store_true", help="Симуляція процесу без запису файлів на диск")
    
    args = parser.parse_args()
    args.server_url = args.server_url.rstrip("/")
    
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    if not input_dir.exists():
        logger.error(f"Вхідну папку оригінального датасету v21 не знайдено: {input_dir}")
        sys.exit(1)
        
    session = get_authenticated_session(args.username, args.password)
    
    logger.info("Підключення до CVAT сервера та отримання списку завдань...")
    all_tasks = []
    next_url = f"{args.server_url}/api/tasks"
    
    try:
        while next_url:
            response = make_request_with_retries(session, "GET", next_url)
            response.raise_for_status()
            data = response.json()
            all_tasks.extend(data.get("results", []))
            next_url = data.get("next")
    except Exception as e:
        logger.error(f"Не вдалося завантажити список завдань з сервера: {e}")
        sys.exit(1)
        
    logger.info(f"Загалом знайдено завдань на сервері: {len(all_tasks)}")
    
    # Співставлення завдань на сервері із локальними папками
    scheduled_tasks: List[Tuple[Dict[str, Any], Path]] = []
    skipped_count = 0
    
    for task in all_tasks:
        task_name = task["name"]
        
        if not parse_task_path(task_name):
            skipped_count += 1
            continue
            
        expected_folder = get_expected_local_path(task_name, input_dir)
        if expected_folder and expected_folder.exists():
            # Перевіримо, чи містить папка якісь зображення
            img_files = [f for f in os.listdir(expected_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if img_files:
                scheduled_tasks.append((task, expected_folder))
            else:
                logger.warning(f"Папка '{expected_folder}' для завдання '{task_name}' порожня. Пропускаємо.")
        else:
            logger.debug(f"Локальну папку для завдання '{task_name}' не знайдено за шляхом '{expected_folder}'.")
            
    logger.info("======================= АНАЛІЗ МАПІНГУ =======================")
    logger.info(f"Пропущено завдань (невірний формат імені / '_del'): {skipped_count}")
    logger.info(f"Завдань успішно пов'язано з локальними папками:   {len(scheduled_tasks)}")
    logger.info("==============================================================")
    
    if not scheduled_tasks:
        logger.info("Немає завдань для обробки. Вихід.")
        return
        
    if not args.dry_run:
        os.makedirs(output_dir / "images", exist_ok=True)
        os.makedirs(output_dir / "labels", exist_ok=True)
        
        # Створення конфігураційних файлів класів
        logger.info("Створення конфігураційних файлів класів...")
        with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
            for name in args.classes:
                f.write(f"{name}\n")
                
        yaml_lines = [
            "path: ./",
            "train: images",
            "val: images",
            "",
            "names:"
        ]
        for idx, name in enumerate(args.classes):
            yaml_lines.append(f"  {idx}: {name}")
            
        with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_lines) + "\n")
            
        logger.info(f"Створено: '{output_dir}/classes.txt' та '{output_dir}/data.yaml'")
        
    # Паралельна обробка
    logger.info(f"Запуск реструктуризації датасету за допомогою {args.workers} паралельних потоків...")
    total_images_processed = 0
    total_errors = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_task_dataset, task_info[0], task_info[1], args, args.classes): task_info
            for task_info in scheduled_tasks
        }
        
        for fut in tqdm(as_completed(futures), total=len(scheduled_tasks), desc="Конвертація епізодів", unit="епізод"):
            task, local_folder = futures[fut]
            try:
                success_count, error_count = fut.result()
                total_images_processed += success_count
                total_errors += error_count
            except Exception as e:
                logger.error(f"Потік завершився помилкою під час обробки завдання '{task['name']}': {e}")
                
    logger.info("======================= РЕЗУЛЬТАТИ =======================")
    logger.info("Конвертацію та реструктуризацію завершено.")
    logger.info(f"Оброблено кадрів успішно:             {total_images_processed}")
    logger.info(f"Кадрів, під час обробки яких були помилки: {total_errors}")
    logger.info(f"Локація збереженого датасету v22:       '{output_dir}'")
    logger.info("==========================================================")


if __name__ == "__main__":
    main()