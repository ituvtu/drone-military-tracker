import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import requests
from huggingface_hub import HfApi, CommitOperationAdd

# Завантаження .env
load_dotenv()

# ================= CONFIGURATION =================
REPO_ID = "ituvtu/annotations"
CVAT_HOST = "localhost"
CVAT_PORT = "8080"
LOCAL_DIR = "D:/labels"

CVAT_USER = os.getenv("CVAT_USER")      
CVAT_PASS = os.getenv("CVAT_PASS")   
# =================================================

api = HfApi()
auth_cvat = (CVAT_USER, CVAT_PASS)

LABELS = [
    {"name": "howitzer"}, {"name": "antiair"}, {"name": "bmp"}, {"name": "tank"},
    {"name": "cars"}, {"name": "truck"}, {"name": "other"}
]

def run_cmd(cmd):
    """Запуск термінальних команд (для cvat-cli)"""
    venv_cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "cvat-cli.exe")
    if os.path.exists(venv_cli) and cmd.startswith("cvat-cli"):
        cmd = f'"{venv_cli}"' + cmd[len("cvat-cli"):]

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Помилка виконання: {cmd}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")
        sys.exit(1)
    return result.stdout + "\n" + result.stderr

def get_cvat_prefix(split):
    """Перетворює map_1/ds_1 у m1_s1_"""
    prefix = re.sub(r'map_(\d+)/ds_(\d+)', r'm\1_s\2_', split)
    if not prefix.endswith('_') and prefix != split:
        prefix += '_'
    elif prefix == split:
        prefix = split.replace('/', '_').replace('\\', '_') + '_'
    return prefix

# ================= ФУНКЦІЇ СКАНИРУВАННЯ ТА СКІПУ =================
def find_all_splits(dataset_root):
    """Шукає всі теки сплітів (які закінчуються на _a, _b і т.д.) рекурсивно"""
    root_path = Path(dataset_root).resolve()
    splits_found = []
    
    # Регулярний вираз для пошуку сплітів (наприклад: ep_013_a, ep_023_neg_b)
    pattern = re.compile(r"^ep_.*_[a-z]$")
    
    for ep_dir in root_path.glob("map_*/ds_*/ep_*"):
        if ep_dir.is_dir() and pattern.match(ep_dir.name):
            rel_path = ep_dir.relative_to(root_path)
            split = "/".join(rel_path.parts[:2])  # map_X/ds_Y
            splits_found.append((ep_dir, split, ep_dir.name))
            
    return sorted(splits_found, key=lambda x: str(x[0]))

def get_all_cvat_tasks(cvat_host, cvat_port, auth):
    """Отримує список назв усіх задач з CVAT за один запит (швидко)"""
    url = f"http://{cvat_host}:{cvat_port}/api/tasks?page_size=1000"
    try:
        r = requests.get(url, auth=auth)
        r.raise_for_status()
        results = r.json().get('results', [])
        return {t['name'] for t in results}
    except Exception as e:
        print(f"[ERROR] Не вдалося отримати список задач з CVAT: {e}")
        return set()

# ================= ФУНКЦІЯ 1: ВИДАЛЕННЯ З HF =================
def delete_hf_assets(repo_id, split, ep_name, hf_token, dry_run=False):
    """Видаляє кадри (.zip) та папку анотацій з HF"""
    print(f"[HF] Початок видалення активів для {ep_name} ({split})...")
    
    hf_frames_path = f"data/{split}/{ep_name}.zip"
    hf_ann_folder = f"annotations/{split}/{ep_name}"

    if dry_run:
        print(f"  [Dry-Run] Було б видалено файл з HF: {hf_frames_path}")
        print(f"  [Dry-Run] Було б видалено теку з HF: {hf_ann_folder}/")
        return True

    # 1. Видалення кадрів
    try:
        api.delete_file(
            path_in_repo=hf_frames_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            commit_message=f"Delete frames zip for {ep_name} (cleanup)"
        )
        print(f"  ✅ [SUCCESS] Видалено файл кадрів: {hf_frames_path}")
    except Exception as e:
        print(f"  ⚠️  [WARNING] Файл {hf_frames_path} не знайдено або помилка: {e}")

    # 2. Видалення папки анотацій
    try:
        api.delete_folder(
            path_in_repo=hf_ann_folder,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
            commit_message=f"Delete annotations folder for {ep_name} (cleanup)"
        )
        print(f"  ✅ [SUCCESS] Видалено теку анотацій: {hf_ann_folder}")
    except Exception as e:
        print(f"  ⚠️  [WARNING] Теку {hf_ann_folder} не знайдено або помилка: {e}")

# ================= ФУНКЦІЯ 2: ПУШ НА HF =================
def push_hf_assets(repo_id, split, ep_dir, local_temp_dir, hf_token, dry_run=False):
    """Пакує та завантажує локальний епізод на HF"""
    ep_dir = Path(ep_dir).resolve()
    if not ep_dir.exists():
        print(f"[ERROR] Локальна папка не існує: {ep_dir}")
        return False

    video_name = ep_dir.name
    img_files = sorted([f for f in ep_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not img_files:
        print(f"[WARNING] Немає кадрів у {ep_dir}. Пропуск пушу.")
        return False

    xml_file = ep_dir / "annotations.xml"

    if dry_run:
        print(f"  [Dry-Run] Було б завантажено епізод {video_name} ({len(img_files)} кадрів) на HF.")
        if xml_file.exists():
            print(f"  [Dry-Run] Було б завантажено XML анотації {xml_file.name} на HF.")
        return True

    os.makedirs(local_temp_dir, exist_ok=True)
    
    # Створення ZIP кадрів
    temp_frames_zip = Path(local_temp_dir) / f"temp_{video_name}_frames.zip"
    if temp_frames_zip.exists():
        temp_frames_zip.unlink()

    print(f"  -> Пакування кадрів для {video_name}...")
    with zipfile.ZipFile(temp_frames_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img in img_files:
            zipf.write(img, arcname=img.name)

    # Створення ZIP анотації
    temp_ann_zip = Path(local_temp_dir) / f"temp_{video_name}_ann.zip"
    if temp_ann_zip.exists():
        temp_ann_zip.unlink()

    has_xml = xml_file.exists()
    if has_xml:
        print(f"  -> Пакування XML анотації для {video_name}...")
        with zipfile.ZipFile(temp_ann_zip, 'w', zipfile.ZIP_DEFLATED) as zipa:
            zipa.write(xml_file, arcname="annotations.xml")

    # Пуш одним комітом
    print(f"  -> Відправка на HF: {repo_id}...")
    operations = []
    
    hf_frames_path = f"data/{split}/{video_name}.zip"
    operations.append(CommitOperationAdd(
        path_in_repo=hf_frames_path,
        path_or_fileobj=str(temp_frames_zip)
    ))

    if has_xml:
        hf_ann_path = f"annotations/{split}/{video_name}/cvat/{video_name}.zip"
        operations.append(CommitOperationAdd(
            path_in_repo=hf_ann_path,
            path_or_fileobj=str(temp_ann_zip)
        ))

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Upload direct package for {video_name} under {split} (multitool)",
            token=hf_token
        )
        print(f"  ✅ [SUCCESS] Епізод {video_name} успішно залито на HF.")
    except Exception as e:
        print(f"  ❌ [ERROR] Помилка пушу {video_name} на HF: {e}")
    finally:
        if temp_frames_zip.exists():
            temp_frames_zip.unlink()
        if temp_ann_zip.exists():
            temp_ann_zip.unlink()
    return True

# ================= ФУНКЦІЯ 3: ВИДАЛЕННЯ З CVAT =================
def get_cvat_task_id(cvat_host, cvat_port, auth, task_name):
    """Шукає Task ID за назвою завдання"""
    url = f"http://{cvat_host}:{cvat_port}/api/tasks?search={task_name}"
    try:
        r = requests.get(url, auth=auth)
        r.raise_for_status()
        results = r.json().get('results', [])
        for t in results:
            if t['name'] == task_name:
                return t['id']
    except Exception as e:
        print(f"[ERROR] Помилка пошуку Task ID для {task_name}: {e}")
    return None

def delete_cvat_task_by_name(cvat_host, cvat_port, auth, task_name, dry_run=False):
    """Знаходить та видаляє задачу в CVAT за точною назвою"""
    print(f"[CVAT] Пошук задачі '{task_name}' для видалення...")
    task_id = get_cvat_task_id(cvat_host, cvat_port, auth, task_name)
    
    if not task_id:
        print(f"  ℹ️  Задача '{task_name}' не знайдена в CVAT. Пропуск.")
        return False

    if dry_run:
        print(f"  [Dry-Run] Було б видалено задачу '{task_name}' (ID: {task_id}) з CVAT.")
        return True

    url = f"http://{cvat_host}:{cvat_port}/api/tasks/{task_id}"
    try:
        r = requests.delete(url, auth=auth)
        r.raise_for_status()
        print(f"  ✅ [SUCCESS] Задача '{task_name}' (ID: {task_id}) успішно видалена з CVAT.")
        return True
    except Exception as e:
        print(f"  ❌ [ERROR] Помилка видалення задачі {task_name} з CVAT: {e}")
        return False

# ================= ФУНКЦІЯ 4: СТВОРЕННЯ В CVAT =================
def create_cvat_task_from_local(cvat_host, cvat_port, auth, task_name, ep_dir, local_temp_dir, labels, dry_run=False):
    """Створює задачу в CVAT та імпортує кадри/локальні анотації"""
    ep_dir = Path(ep_dir).resolve()
    if not ep_dir.exists():
        print(f"[ERROR] Локальна папка не існує: {ep_dir}")
        return False

    img_files = sorted([f for f in ep_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not img_files:
        print(f"[WARNING] Немає кадрів у {ep_dir}. Пропуск створення задачі.")
        return False

    xml_file = ep_dir / "annotations.xml"

    if dry_run:
        print(f"  [Dry-Run] Було б створено задачу '{task_name}' в CVAT ({len(img_files)} кадрів).")
        if xml_file.exists():
            print(f"  [Dry-Run] Було б імпортовано локальний XML {xml_file.name} до задачі.")
        return True

    os.makedirs(local_temp_dir, exist_ok=True)
    temp_zip_path = Path(local_temp_dir) / f"{task_name}_temp_frames.zip"

    # 1. Тимчасовий ZIP кадрів
    with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for img in img_files:
            zipf.write(img, arcname=img.name)

    # 2. CLI створення задачі
    print(f"  -> Створення задачі '{task_name}' у CVAT...")
    labels_str = json.dumps(labels).replace('"', '\\"')
    create_cmd = f'cvat-cli --server-host {cvat_host} --server-port {cvat_port} --auth {auth[0]}:{auth[1]} create "{task_name}" --labels "{labels_str}" local "{temp_zip_path}"'
    output = run_cmd(create_cmd)

    if temp_zip_path.exists():
        temp_zip_path.unlink()

    match = re.search(r'(?i)ID[\s:]*(\d+)', output)
    if not match:
        print(f"  ❌ [ERROR] Не вдалося розпарсити ID задачі для {task_name}. Вивід: {output}")
        return False

    task_id = match.group(1)
    print(f"  ✅ [SUCCESS] Задача створена. ID: {task_id}")

    # 3. XML анотація
    if xml_file.exists():
        print(f"  -> Завантаження локальних анотацій для {task_name}...")
        temp_ann_zip = Path(local_temp_dir) / f"{task_name}_temp_ann.zip"
        with zipfile.ZipFile(temp_ann_zip, 'w', zipfile.ZIP_DEFLATED) as zipa:
            zipa.write(xml_file, arcname="annotations.xml")

        upload_cmd = f'cvat-cli --server-host {cvat_host} --server-port {cvat_port} --auth {auth[0]}:{auth[1]} upload {task_id} "{temp_ann_zip}" --format "CVAT 1.1"'
        run_cmd(upload_cmd)

        if temp_ann_zip.exists():
            temp_ann_zip.unlink()
        print(f"  ✅ [SUCCESS] Анотації імпортовані у CVAT.")

    print(f"  🔗 Задача готова: http://{cvat_host}:{cvat_port}/tasks/{task_id}")
    return True

# ================= МАКРО-СИНХРОНІЗАЦІЯ ВСЬОГО ДАТАСЕТУ =================
def sync_all_splits(repo_id, dataset_root, local_temp_dir, hf_token, dry_run=False):
    """Шукає всі спліти у локальному датасеті та пушить нові (яких ще немає в CVAT)"""
    print("🔍 Початок сканування датасету на наявність сплітів...")
    splits = find_all_splits(dataset_root)
    
    if not splits:
        print("[INFO] Сплітів (папок, що закінчуються на _a, _b тощо) не знайдено.")
        return

    print(f"\n📋 Знайдено сплітів локально ({len(splits)}):")
    for _, split, ep_name in splits:
        print(f"  - {split}/{ep_name}")

    print("\n🌐 Отримання списку існуючих задач з CVAT...")
    existing_tasks = get_all_cvat_tasks(CVAT_HOST, CVAT_PORT, auth_cvat)
    print(f"  [CVAT] Знайдено {len(existing_tasks)} активних задач в системі.")

    print("\n🚀 Початок синхронізації...")
    processed_count = 0
    skipped_count = 0

    for ep_dir, split, ep_name in splits:
        prefix = get_cvat_prefix(split)
        task_name = f"{prefix}{ep_name}"

        # Головна умова: якщо таска вже є в CVAT - пропускаємо
        if task_name in existing_tasks:
            print(f"\n[SKIP] Задача '{task_name}' вже існує в CVAT. Пропуск.")
            skipped_count += 1
            continue

        print(f"\n🆕 Знайдено новий спліт: {split}/{ep_name} -> Задача: {task_name}")
        
        # 1. Пуш на HF
        push_success = push_hf_assets(repo_id, split, ep_dir, local_temp_dir, hf_token, dry_run)
        
        # 2. Створення в CVAT
        if push_success or dry_run:
            create_cvat_task_from_local(
                CVAT_HOST, CVAT_PORT, auth_cvat, task_name,
                ep_dir, local_temp_dir, LABELS, dry_run
            )
            processed_count += 1

    print(f"\n🎉 Синхронізацію завершено. Оброблено нових: {processed_count}, Пропущено існуючих: {skipped_count}")

# ================= МЕНЮ ТА ОРКЕСТРАЦІЯ =================
def main():
    parser = argparse.ArgumentParser(description="Мультитул міст-синхронізатор CVAT та Hugging Face")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Видалення з HF
    p_del_hf = subparsers.add_parser("delete-hf", help="Видалити архіви епізоду з Hugging Face")
    p_del_hf.add_argument("split", help="Наприклад, map_1/ds_1")
    p_del_hf.add_argument("ep_name", help="Наприклад, ep_013")

    # Пуш на HF
    p_push_hf = subparsers.add_parser("push-hf", help="Запакувати та завантажити епізод на Hugging Face")
    p_push_hf.add_argument("split", help="Наприклад, map_1/ds_1")
    p_push_hf.add_argument("ep_dir", help="Шлях до локальної папки")

    # Видалення з CVAT
    p_del_cvat = subparsers.add_parser("delete-cvat", help="Видалити задачу з CVAT за назвою")
    p_del_cvat.add_argument("task_name", help="Наприклад, m1_s1_ep_013")

    # Створення в CVAT
    p_create_cvat = subparsers.add_parser("create-cvat", help="Створити задачу в CVAT з локальної папки")
    p_create_cvat.add_argument("task_name", help="Наприклад, m1_s1_ep_013_a")
    p_create_cvat.add_argument("ep_dir", help="Шлях до локальної папки")

    # СУПЕР-КОМАНДА 1: Повний цикл сплітування одного епізоду
    p_sync = subparsers.add_parser("sync-split", help="Заміна старого епізоду на нові спліти всюди (HF + CVAT)")
    p_sync.add_argument("--split", required=True, help="Наприклад, map_1/ds_1")
    p_sync.add_argument("--parent", required=True, help="Оригінальний епізод, наприклад, ep_013")
    p_sync.add_argument("--children", required=True, nargs="+", help="Нові спліти, наприклад, ep_013_a ep_013_b")
    p_sync.add_argument("--dataset_root", default="dataset_v11", help="Шлях до dataset_v11")

    # СУПЕР-КОМАНДА 2: Сканування і масова синхронізація всього датасету
    p_sync_all = subparsers.add_parser("sync-all-splits", help="Сканувати датасет та синхронізувати всі нові спліти")
    p_sync_all.add_argument("--dataset_root", default="dataset_v11", help="Шлях до dataset_v11")

    # Флаг Dry-run для всіх
    for sub in [p_del_hf, p_push_hf, p_del_cvat, p_create_cvat, p_sync, p_sync_all]:
        sub.add_argument("--dry-run", action="store_true", help="Запуск без фізичного внесення змін")

    args = parser.parse_args()
    hf_token = os.getenv("HF_TOKEN")

    if args.command == "delete-hf":
        delete_hf_assets(REPO_ID, args.split, args.ep_name, hf_token, args.dry_run)
        
    elif args.command == "push-hf":
        push_hf_assets(REPO_ID, args.split, args.ep_dir, LOCAL_DIR, hf_token, args.dry_run)
        
    elif args.command == "delete-cvat":
        delete_cvat_task_by_name(CVAT_HOST, CVAT_PORT, auth_cvat, args.task_name, args.dry_run)
        
    elif args.command == "create-cvat":
        create_cvat_task_from_local(CVAT_HOST, CVAT_PORT, auth_cvat, args.task_name, args.ep_dir, LOCAL_DIR, LABELS, args.dry_run)
        
    elif args.command == "sync-split":
        prefix = get_cvat_prefix(args.split)
        parent_task_name = f"{prefix}{args.parent}"

        print(f"🔄 === ПОЧАТОК ЦИКЛУ СИНХРОНІЗАЦІЇ СПЛІТІВ ДЛЯ {args.parent} ===")
        if args.dry_run:
            print("[Dry-Run] Тестовий прохід активовано.")

        delete_hf_assets(REPO_ID, args.split, args.parent, hf_token, args.dry_run)
        delete_cvat_task_by_name(CVAT_HOST, CVAT_PORT, auth_cvat, parent_task_name, args.dry_run)

        for child in args.children:
            child_dir = Path(args.dataset_root) / args.split / child
            child_task_name = f"{prefix}{child}"

            print(f"\n👉 Обробка спліту: {child}...")
            push_success = push_hf_assets(REPO_ID, args.split, child_dir, LOCAL_DIR, hf_token, args.dry_run)
            
            if push_success or args.dry_run:
                create_cvat_task_from_local(
                    CVAT_HOST, CVAT_PORT, auth_cvat, child_task_name, 
                    child_dir, LOCAL_DIR, LABELS, args.dry_run
                )
        print("\n🎉 === ЦИКЛ СИНХРОНІЗАЦІЇ СПЛІТІВ ЗАВЕРШЕНО ===")

    elif args.command == "sync-all-splits":
        sync_all_splits(REPO_ID, args.dataset_root, LOCAL_DIR, hf_token, args.dry_run)

if __name__ == "__main__":
    main()