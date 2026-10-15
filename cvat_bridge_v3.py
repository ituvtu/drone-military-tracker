import os 
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
import sys
import subprocess
import re
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import requests

# Load environment variables
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

def run_cmd(cmd):
    """Executes a command in the terminal and returns the combined stdout and stderr"""
    # Resolve cvat-cli to the virtualenv path if available
    venv_cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "cvat-cli.exe")
    if os.path.exists(venv_cli) and cmd.startswith("cvat-cli"):
        cmd = f'"{venv_cli}"' + cmd[len("cvat-cli"):]

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command execution error: {cmd}\nSTDERR: {result.stderr}\nSTDOUT: {result.stdout}")
        sys.exit(1)
    return result.stdout + "\n" + result.stderr

def pull_batch(split):
    """Downloads a batch from HF and automatically creates CVAT tasks for all videos"""
    print(f"[INFO] Searching for files in split '{split}' on Hugging Face...")
    
    batch_prefix = f"annotations/{split}/"
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # Find all ZIP archives in this batch (looking for sam3 annotations)
    zip_files = [f for f in files if f.startswith(batch_prefix) and "/sam3/" in f and f.endswith('.zip')]
    
    if not zip_files:
        print("[ERROR] No annotation archives found in this batch.")
        return

    os.makedirs(LOCAL_DIR, exist_ok=True)

    # Process each ZIP archive in the batch
    for zip_file_repo_path in zip_files:
        # Extract the clean video name
        zip_filename = os.path.basename(zip_file_repo_path)
        base_name = zip_filename.replace('_mot.zip', '').replace('_cvat.zip', '').replace('.zip', '')
        
        # Look for the frame ZIP in the repository
        video_repo_path = next((f for f in files if f.startswith(f"data/{split}/{base_name}.") and f.endswith('.zip')), None)
        
        if "_neg" in base_name.lower():
            print(f"[INFO] Skipping negative episode {base_name} from CVAT task creation.")
            continue
        
        if not video_repo_path:
            print(f"[WARNING] Skipping {base_name}: could not find the frame ZIP on HF.")
            continue
            
        print(f"\n[INFO] Downloading {base_name}...")
        local_video_path = hf_hub_download(repo_id=REPO_ID, filename=video_repo_path, repo_type="dataset", local_dir=LOCAL_DIR)
        local_zip_path = hf_hub_download(repo_id=REPO_ID, filename=zip_file_repo_path, repo_type="dataset", local_dir=LOCAL_DIR)

        print(f"[INFO] Creating a CVAT task for {base_name}...")
        task_name = f"{base_name}_{split}"
        create_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} create "{task_name}" --labels "[{{\\"name\\": \\"howitzer\\\"}}, {{\\"name\\": \\"antiair\\\"}}, {{\\"name\\": \\"bmp\\\"}}, {{\\"name\\": \\"tank\\\"}}, {{\\"name\\": \\"cars\\\"}}, {{\\"name\\": \\"truck\\\"}}]" local "{local_video_path}"'
        output = run_cmd(create_cmd)
        
        # Flexible regex to catch the Task ID regardless of formatting/case
        match = re.search(r'(?i)ID[\s:]*(\d+)', output)
        if not match:
            print(f"[ERROR] Failed to extract Task ID for {base_name}.")
            print(f"[DEBUG] Raw CVAT Output:\n{output.strip()}")
            continue
            
        task_id = match.group(1)
        print(f"[SUCCESS] Task created! Task ID: {task_id}")

        print("[INFO] Uploading SAM 3 annotations to task...")
        # FIX: Changed 'import' to 'upload' for annotations
        upload_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} upload {task_id} "{local_zip_path}" --format "CVAT 1.1"'
        run_cmd(upload_cmd)
        
        print(f"[SUCCESS] Done: http://{CVAT_HOST}:{CVAT_PORT}/tasks/{task_id}")

def push_batch(task_id, split):
    """Exports annotations from CVAT and pushes them to labels on HF with proper video names"""
    print(f"[INFO] Exporting annotation from CVAT (Task ID: {task_id})...")
    
    # 1. Ask CVAT for the task name to get the correct video name
    try:
        response = requests.get(
            f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks/{task_id}",
            auth=(CVAT_USER, CVAT_PASS)
        )
        response.raise_for_status()
        task_info = response.json()
        task_name = task_info.get("name", f"task_{task_id}")
        # Clean up the name: try to find 'ep_XXX' pattern, otherwise fallback
        match_ep = re.search(r'ep_\d+', task_name)
        if match_ep:
            video_name = match_ep.group(0)
        else:
            video_name = task_name.replace(f"_{split}", "")
    except Exception as e:
        print(f"[WARNING] Could not fetch task name from CVAT: {e}. Using Task ID instead.")
        video_name = f"task_{task_id}"

    os.makedirs(LOCAL_DIR, exist_ok=True)
    export_filename = f"{video_name}.zip"
    export_path = os.path.join(LOCAL_DIR, export_filename)
    if os.path.exists(export_path):
        os.remove(export_path)
    
    # 2. Dump annotations
    dump_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} dump {task_id} "{export_path}" --format "CVAT for video 1.1"'
    run_cmd(dump_cmd)
    
    # 3. Upload to Hugging Face into the correct folder structure
    print(f"[INFO] Sending the corrected annotation to Hugging Face as {export_filename}...")
    path_in_repo = f"annotations/{split}/{video_name}/cvat/{export_filename}"
    
    api.upload_file(
        path_or_fileobj=export_path,
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Upload manually validated CVAT annotations for {video_name}"
    )
    
    print(f"[SUCCESS] Success! The verified archive has been saved to HF: {path_in_repo}")

def complete_tasks(task_ids_str):
    """Marks task(s) as completed in CVAT by completing their jobs"""
    task_ids = [t.strip() for t in task_ids_str.split(",") if t.strip()]
    if not task_ids:
        print("[ERROR] No task IDs provided.")
        return

    print(f"[INFO] Marking tasks {task_ids} as completed...")
    auth = (CVAT_USER, CVAT_PASS)
    for task_id in task_ids:
        try:
            # 1. Fetch jobs for the task
            url = f"http://{CVAT_HOST}:{CVAT_PORT}/api/jobs?task_id={task_id}"
            response = requests.get(url, auth=auth)
            response.raise_for_status()
            jobs = response.json().get("results", [])
            if not jobs:
                print(f"[WARNING] No jobs found for task {task_id}")
                continue

            for job in jobs:
                job_id = job["id"]
                # 2. Patch each job to completed/acceptance
                patch_url = f"http://{CVAT_HOST}:{CVAT_PORT}/api/jobs/{job_id}"
                patch_resp = requests.patch(
                    patch_url,
                    json={"state": "completed", "stage": "acceptance"},
                    auth=auth
                )
                patch_resp.raise_for_status()
                print(f"[SUCCESS] Task {task_id}, Job {job_id} marked as completed")
        except Exception as e:
            print(f"[ERROR] Failed to complete task {task_id}: {e}")

def push_dataset(task_id, split):
    """Exports dataset with images from CVAT, packages flatly, and pushes both frames and labels to HF"""
    import zipfile
    import shutil

    print(f"[INFO] Exporting dataset with images from CVAT (Task ID: {task_id})...")
    
    # 1. Fetch task name to get the correct video name
    try:
        response = requests.get(
            f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks/{task_id}",
            auth=(CVAT_USER, CVAT_PASS)
        )
        response.raise_for_status()
        task_info = response.json()
        task_name = task_info.get("name", f"task_{task_id}")
        
        match_ep = re.search(r'ep_\d+', task_name)
        if match_ep:
            video_name = match_ep.group(0)
        else:
            video_name = task_name.replace(f"_{split}", "")
    except Exception as e:
        print(f"[WARNING] Could not fetch task name: {e}. Using Task ID instead.")
        video_name = f"task_{task_id}"

    os.makedirs(LOCAL_DIR, exist_ok=True)
    temp_dataset_zip = os.path.join(LOCAL_DIR, f"task_{task_id}_dataset_temp.zip")
    if os.path.exists(temp_dataset_zip):
        os.remove(temp_dataset_zip)

    # 2. Export dataset from CVAT (CVAT for video 1.1 with images)
    export_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} task export-dataset --format "CVAT for video 1.1" --with-images True {task_id} "{temp_dataset_zip}"'
    run_cmd(export_cmd)

    if not os.path.exists(temp_dataset_zip):
        print(f"[ERROR] Dataset export ZIP not found for task {task_id}")
        return

    # Extract temporary folders
    temp_extract_dir = os.path.join(LOCAL_DIR, f"task_{task_id}_extract")
    if os.path.exists(temp_extract_dir):
        shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir)

    print("[INFO] Extracting exported archive...")
    with zipfile.ZipFile(temp_dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    images_dir = os.path.join(temp_extract_dir, "images")
    ann_xml_path = os.path.join(temp_extract_dir, "annotations.xml")

    if not os.path.exists(images_dir) or not os.path.exists(ann_xml_path):
        print("[ERROR] Exported ZIP structure is invalid (missing images/ or annotations.xml)")
        shutil.rmtree(temp_extract_dir)
        os.remove(temp_dataset_zip)
        return

    # 3. Zip frames flatly
    frames_zip_path = os.path.join(LOCAL_DIR, f"{video_name}_frames.zip")
    if os.path.exists(frames_zip_path):
        os.remove(frames_zip_path)
    
    print(f"[INFO] Packaging frames into flat ZIP...")
    with zipfile.ZipFile(frames_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_f:
        for f_name in sorted(os.listdir(images_dir)):
            if f_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                f_path = os.path.join(images_dir, f_name)
                zip_f.write(f_path, arcname=f_name)

    # 4. Zip annotations flatly
    ann_zip_path = os.path.join(LOCAL_DIR, f"{video_name}_ann.zip")
    if os.path.exists(ann_zip_path):
        os.remove(ann_zip_path)

    print(f"[INFO] Packaging annotations.xml into flat ZIP...")
    with zipfile.ZipFile(ann_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_a:
        zip_a.write(ann_xml_path, arcname="annotations.xml")

    # 5. Upload frames to HF
    hf_frames_path = f"data/{split}/{video_name}.zip"
    print(f"[INFO] Uploading frames to HF: {hf_frames_path}...")
    api.upload_file(
        path_or_fileobj=frames_zip_path,
        path_in_repo=hf_frames_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Update frames for {video_name} (Task {task_id}) - removed duplicates"
    )

    # 6. Upload annotations to HF
    hf_ann_path = f"annotations/{split}/{video_name}/cvat/{video_name}.zip"
    print(f"[INFO] Uploading annotations to HF: {hf_ann_path}...")
    api.upload_file(
        path_or_fileobj=ann_zip_path,
        path_in_repo=hf_ann_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Update CVAT annotations for {video_name} (Task {task_id}) - removed duplicates"
    )

    # Clean up
    print("[INFO] Cleaning up local temporary files...")
    shutil.rmtree(temp_extract_dir)
    os.remove(temp_dataset_zip)
    if os.path.exists(frames_zip_path):
        os.remove(frames_zip_path)
    if os.path.exists(ann_zip_path):
        os.remove(ann_zip_path)

    print(f"[SUCCESS] Success! Dataset push completed for task {task_id}")

def create_local_tasks(dataset_path, split, labels_type="standard"):
    """Creates CVAT tasks from a local dataset folder containing ep_* folders, and pulls annotations from HF if they exist"""
    import zipfile
    import json
    
    # 1. Determine labels
    if labels_type == "burn":
        labels = [
            {"name": "howitzer"}, {"name": "antiair"}, {"name": "bmp"}, {"name": "tank"},
            {"name": "cars"}, {"name": "truck"}, {"name": "burn_howitzer"}, {"name": "burn_antiair"},
            {"name": "burn_bmp"}, {"name": "burn_tank"}, {"name": "burn_cars"}, {"name": "burn_truck"},
            {"name": "burn_other"}, {"name": "other"}
        ]
    else: # standard
        labels = [
            {"name": "howitzer"}, {"name": "antiair"}, {"name": "bmp"}, {"name": "tank"},
            {"name": "cars"}, {"name": "truck"}, {"name": "other"}
        ]
        
    # Get existing tasks
    print("[INFO] Fetching existing tasks from CVAT...")
    try:
        url = f"http://{CVAT_HOST}:{CVAT_PORT}/api/tasks?page_size=200"
        response = requests.get(url, auth=(CVAT_USER, CVAT_PASS))
        response.raise_for_status()
        results = response.json().get('results', [])
        existing_tasks = {t['name']: t['id'] for t in results}
    except Exception as e:
        print(f"[ERROR] Could not fetch tasks from CVAT: {e}")
        existing_tasks = {}

    # Find local episodes
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path {dataset_path} does not exist.")
        return
        
    episodes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and re.match(r'ep_\d+', d)]
    episodes = sorted(episodes)
    print(f"[INFO] Found {len(episodes)} episodes: {episodes}")

    # Normalize split to task name prefix, e.g. map_1/ds_1 -> m1_s1_
    prefix = re.sub(r'map_(\d+)/ds_(\d+)', r'm\1_s\2_', split)
    if not prefix.endswith('_') and prefix != split:
        prefix += '_'
    elif prefix == split:
        # Fallback if split doesn't match map_X/ds_Y format
        prefix = split.replace('/', '_').replace('\\', '_') + '_'

    # Get HF token if available
    hf_token = os.getenv("HF_TOKEN")
    print("[INFO] Fetching list of files from Hugging Face...")
    try:
        hf_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=hf_token)
    except Exception as e:
        print(f"[WARNING] Could not list HF files: {e}. Skipping annotation downloading.")
        hf_files = []

    for ep in episodes:
        task_name = f"{prefix}{ep}"
        if task_name in existing_tasks:
            print(f"[INFO] Skipping {task_name} (already exists with ID {existing_tasks[task_name]})")
            continue

        print(f"\n========================================\n[INFO] Creating task: {task_name}")
        ep_dir = os.path.join(dataset_path, ep)
        
        # 1. Zip local frames
        os.makedirs(LOCAL_DIR, exist_ok=True)
        temp_zip_path = os.path.join(LOCAL_DIR, f"{task_name}_frames.zip")
        
        print(f"[INFO] Zipping {ep_dir} to {temp_zip_path}...")
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(ep_dir):
                for file in sorted(files):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, arcname=file)

        # 2. Create task via cvat-cli
        print(f"[INFO] Uploading frames to CVAT...")
        labels_str = json.dumps(labels).replace('"', '\\"')
        create_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} create "{task_name}" --labels "{labels_str}" local "{temp_zip_path}"'
        output = run_cmd(create_cmd)
        
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

        match = re.search(r'(?i)ID[\s:]*(\d+)', output)
        if not match:
            print(f"[ERROR] Could not parse task ID from CVAT output for {task_name}")
            continue

        task_id = match.group(1)
        print(f"[SUCCESS] Created task {task_name} with ID {task_id}")

        # 3. Download/upload annotations if they exist on HF
        cvat_ann_path = f"annotations/{split}/{ep}/cvat/{ep}.zip"
        sam3_ann_path = f"annotations/{split}/{ep}/sam3/{ep}.zip"

        ann_path_in_repo = None
        if cvat_ann_path in hf_files:
            ann_path_in_repo = cvat_ann_path
            print(f"[INFO] Found CVAT annotations on HF: {cvat_ann_path}")
        elif sam3_ann_path in hf_files:
            ann_path_in_repo = sam3_ann_path
            print(f"[INFO] Found SAM3 annotations on HF: {sam3_ann_path}")

        if ann_path_in_repo:
            print(f"[INFO] Downloading annotations from HF...")
            try:
                local_ann_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=ann_path_in_repo,
                    repo_type="dataset",
                    local_dir=LOCAL_DIR,
                    token=hf_token
                )
                print(f"[INFO] Uploading annotations to CVAT task {task_id}...")
                upload_cmd = f'cvat-cli --server-host {CVAT_HOST} --server-port {CVAT_PORT} --auth {CVAT_USER}:{CVAT_PASS} upload {task_id} "{local_ann_path}" --format "CVAT 1.1"'
                run_cmd(upload_cmd)
                print(f"[SUCCESS] Annotations uploaded for {task_name}")
                
                if os.path.exists(local_ann_path):
                    os.remove(local_ann_path)
            except Exception as e:
                print(f"[ERROR] Failed to download/upload annotations: {e}")

def download_dataset_from_hf(split=None):
    """Downloads files (data zip and cvat annotations zip) from HF, ignoring any 'sam3' directory"""
    hf_token = os.getenv("HF_TOKEN")
    print(f"[INFO] Listing repository files on Hugging Face (Repo: {REPO_ID})...")
    
    try:
        files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=hf_token)
    except Exception as e:
        print(f"[ERROR] Failed to read repository: {e}")
        return

    # Фільтрація: викидаємо 'sam3', за потреби фільтруємо по спліту
    download_queue = []
    for f in files:
        if "/sam3/" in f:
            continue
            
        if split:
            # Нормалізація шляхів
            norm_f = f.replace("\\", "/")
            prefix_data = f"data/{split}/"
            prefix_ann = f"annotations/{split}/"
            if not (norm_f.startswith(prefix_data) or norm_f.startswith(prefix_ann)):
                continue
                
        download_queue.append(f)

    if not download_queue:
        print("[WARNING] No matching files found to download.")
        return

    print(f"[INFO] Found {len(download_queue)} files to download (excluding 'sam3').")
    os.makedirs(LOCAL_DIR, exist_ok=True)

    for idx, path_in_repo in enumerate(download_queue, 1):
        print(f"[{idx}/{len(download_queue)}] Downloading: {path_in_repo}")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=path_in_repo,
                repo_type="dataset",
                local_dir=LOCAL_DIR,
                token=hf_token
            )
        except Exception as e:
            print(f"[ERROR] Failed to download {path_in_repo}: {e}")

    print("[SUCCESS] All files downloaded successfully.")

def push_v11_dataset(dataset_path):
    """Directly packages and uploads dataset_v11 episodes to Hugging Face without CVAT tasks.
    Creates 1 commit on HF per episode."""
    from pathlib import Path
    import zipfile

    dataset_root = Path(dataset_path).resolve()
    if not dataset_root.exists():
        print(f"[ERROR] Local dataset path does not exist: {dataset_root}")
        return

    print(f"[INFO] Scanning {dataset_root} for episodes...")
    # Шукаємо структури map_*/ds_*/ep_*
    episode_dirs = sorted(list(dataset_root.glob("map_*/ds_*/ep_*")))

    if not episode_dirs:
        print("[WARNING] No episodes found matching 'map_*/ds_*/ep_*'.")
        return

    print(f"[INFO] Found {len(episode_dirs)} episodes to package.")
    os.makedirs(LOCAL_DIR, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")

    for idx, ep_dir in enumerate(episode_dirs, 1):
        rel_path = ep_dir.relative_to(dataset_root)
        split = "/".join(rel_path.parts[:2])  # map_X/ds_Y
        video_name = ep_dir.name  # ep_ZZZ

        print(f"\n[{idx}/{len(episode_dirs)}] Processing {video_name} ({split})...")

        # Перевірка наявності кадрів
        img_files = sorted([f for f in ep_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if not img_files:
            print(f"[WARNING] No images found in {ep_dir}. Skipping.")
            continue

        xml_file = ep_dir / "annotations.xml"

        # 1. Створення тимчасового архіву кадрів
        temp_frames_zip = Path(LOCAL_DIR) / f"temp_{video_name}_frames.zip"
        if temp_frames_zip.exists():
            temp_frames_zip.unlink()

        print("  -> Packaging frames...")
        with zipfile.ZipFile(temp_frames_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img in img_files:
                zipf.write(img, arcname=img.name)

        # 2. Створення тимчасового архіву анотацій
        temp_ann_zip = Path(LOCAL_DIR) / f"temp_{video_name}_ann.zip"
        if temp_ann_zip.exists():
            temp_ann_zip.unlink()

        has_xml = xml_file.exists()
        if has_xml:
            print("  -> Packaging annotations...")
            with zipfile.ZipFile(temp_ann_zip, 'w', zipfile.ZIP_DEFLATED) as zipa:
                zipa.write(xml_file, arcname="annotations.xml")

        # 3. Коміт та пуш на Hugging Face
        print(f"  -> Uploading to HF: {REPO_ID}...")
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
                repo_id=REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Upload direct package for {video_name} under {split} (dataset_v11)",
                token=hf_token
            )
            print(f"[SUCCESS] Committed and pushed {video_name} successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to push {video_name}: {e}")

        # Очищення тимчасових архівів
        if temp_frames_zip.exists():
            temp_frames_zip.unlink()
        if temp_ann_zip.exists():
            temp_ann_zip.unlink()

    print("\n🎉 Direct push of v11 completed.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Import to CVAT:         python cvat_bridge.py pull map_1/ds_1")
        print("  Export XML to HF:       python cvat_bridge.py push <TASK_ID> map_1/ds_1")
        print("  Export dataset+pics:    python cvat_bridge.py push-dataset <TASK_ID> map_1/ds_1")
        print("  Mark task(s) complete:  python cvat_bridge.py complete <TASK_ID_1,TASK_ID_2...>")
        print("  Create local tasks:     python cvat_bridge.py create-tasks <DATASET_PATH> <SPLIT> [standard|burn]")
        print("  Download from HF:       python cvat_bridge.py download [SPLIT]")
        print("  Push v11 direct to HF:  python cvat_bridge.py push-v11 <DATASET_PATH>")
        sys.exit(1)

    command = sys.argv[1]
    if command == "pull" and len(sys.argv) == 3:
        pull_batch(sys.argv[2])
    elif command == "push" and len(sys.argv) == 4:
        push_batch(sys.argv[2], sys.argv[3])
    elif command == "push-dataset" and len(sys.argv) == 4:
        push_dataset(sys.argv[2], sys.argv[3])
    elif command == "complete" and len(sys.argv) == 3:
        complete_tasks(sys.argv[2])
    elif command == "create-tasks" and (len(sys.argv) == 4 or len(sys.argv) == 5):
        labels_type = sys.argv[4] if len(sys.argv) == 5 else "standard"
        create_local_tasks(sys.argv[2], sys.argv[3], labels_type)
    elif command == "download":
        split_arg = sys.argv[2] if len(sys.argv) == 3 else None
        download_dataset_from_hf(split_arg)
    elif command == "push-v11" and len(sys.argv) == 3:
        push_v11_dataset(sys.argv[2])
    else:
        print("[ERROR] Incorrect number of arguments.")

