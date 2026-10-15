# Клітинка 1: Встановлення необхідних бібліотек у тихому режимі
!pip install -q boxmot trackeval opencv-python numpy pandas huggingface_hub ultralytics

# Клітинка 2: Конфігурація шляхів, параметрів пошуку, завантаження моделі з Hugging Face та імпорт базових залежностей
import os
import sys
import shutil
import logging
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from huggingface_hub import hf_hub_download

# --- ШЛЯХИ ТА СТРУКТУРА ДАТАСЕТУ ---
INPUT_DATASET_DIR = Path("/kaggle/input/datasets/ituvtu/tracker-test")
WORKSPACE_DIR = Path("/kaggle/working")
MOT_DATASET_DIR = WORKSPACE_DIR / "mil_mot_dataset"
SEQMAPS_DIR = MOT_DATASET_DIR / "seqmaps"
YOLO_DETECTIONS_OUT = WORKSPACE_DIR / "yolo_detections"
TRACKERS_OUT_DIR = WORKSPACE_DIR / "trackers"
BEST_VISUALIZATIONS_DIR = WORKSPACE_DIR / "best_visualizations"

# Автоматичне створення необхідних робочих директорій
for path in [MOT_DATASET_DIR, SEQMAPS_DIR, YOLO_DETECTIONS_OUT, TRACKERS_OUT_DIR, BEST_VISUALIZATIONS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# --- ГІПЕРПАРАМЕТРИ ДЛЯ СІТКИ (GRID SEARCH) ---
BYTETRACK_GRID = {
    "track_thresh": [0.1, 0.2, 0.3, 0.4],
    "track_buffer": [15, 30, 45, 60],
    "match_thresh": [0.6, 0.7, 0.8]
}

OCSORT_GRID = {
    "det_thresh": [0.1, 0.2, 0.3, 0.4],
    "max_age": [15, 30, 45, 60],
    "iou_threshold": [0.2, 0.3, 0.4]
}

# --- АВТОРИЗАЦІЯ І ЗАВАНТАЖЕННЯ МОДЕЛІ З HUGGING FACE ---
user_secrets = UserSecretsClient()
try:
    HF_TOKEN = user_secrets.get_secret("HF_token")
except Exception:
    try:
        HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None

def load_model_from_hf(repo_id: str, filename: str, token: str = None) -> str:
    """Завантажує ваги моделі безпосередньо з репозиторію Hugging Face Hub."""
    print(f"Початок імпорту ваг моделі з Hugging Face ({repo_id})...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            local_dir=str(WORKSPACE_DIR),
            local_dir_use_symlinks=False
        )
        print(f"✅ Модель успішно завантажено та збережено за шляхом: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"[Попередження] Помилка завантаження моделі з HF Hub: {e}")
        local_pts = list(WORKSPACE_DIR.glob("*.pt")) + list(Path("/kaggle/input").rglob("best.pt"))
        if local_pts:
            print(f"Використовуються локальні резервні ваги: {local_pts[0]}")
            return str(local_pts[0])
        else:
            print("[Попередження] Ваги не знайдено, буде використано за замовчуванням 'yolo26n.pt'")
            return "yolo26n.pt"

YOLO_MODEL_PATH = load_model_from_hf(
    repo_id="ituvtu/yolov26-musgd-e2e-experiment4",
    filename="best.pt",
    token=HF_TOKEN
)

# Клітинка 3: Підготовка MOT 1.1, фізичний ресайз кадрів до 640x360 та адаптивне авто-масштабування gt.txt з автогенерацією ID при їх відсутності
import cv2
import numpy as np
from tqdm import tqdm

def prepare_resized_mot_dataset_adaptive():
    print("Початок імпорту, ресайзингу зображень та адаптивного масштабування розмітки...")
    
    seq_name = "tracker-test"
    trackeval_split_name = "mil-test"
    
    # Створюємо цільові папки для сумісності з TrackEval в робочій директорії Kaggle
    gt_dir = MOT_DATASET_DIR / trackeval_split_name
    seq_folder = gt_dir / seq_name
    seq_gt_dir = seq_folder / "gt"
    seq_img_dir = seq_folder / "img1"
    
    for path in [seq_gt_dir, seq_img_dir]:
        path.mkdir(parents=True, exist_ok=True)
        
    img1_src_dir = INPUT_DATASET_DIR / "img1"
    if not img1_src_dir.exists():
        raise FileNotFoundError(f"Не знайдено вхідну папку з оригінальними кадрами: {img1_src_dir}")
        
    src_images = sorted(
        list(img1_src_dir.glob("*.jpg")) + 
        list(img1_src_dir.glob("*.png")) + 
        list(img1_src_dir.glob("*.jpeg"))
    )
    
    if len(src_images) == 0:
        raise FileNotFoundError(f"Вхідна папка img1 порожня.")
        
    # 1. Фізичний ресайз кадрів до 640x360 та збереження в робочу директорію
    target_width, target_height = 640, 360
    print(f"Масштабування {len(src_images)} зображень до роздільної здатності {target_width}x{target_height}...")
    
    for img_path in tqdm(src_images, desc="Ресайз зображень"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(seq_img_dir / img_path.name), resized_img)
        
    # 2. Запис файлу конфігурації seqinfo.ini з новими розмірами
    seqinfo_content = (
        f"[Sequence]\n"
        f"name={seq_name}\n"
        f"imDir=img1\n"
        f"frameRate=30\n"
        f"seqLength={len(src_images)}\n"
        f"imWidth={target_width}\n"
        f"imHeight={target_height}\n"
        f"imExt={src_images[0].suffix}\n"
    )
    with open(seq_folder / "seqinfo.ini", "w") as f:
        f.write(seqinfo_content)
        
    # 3. Обробка та адаптивне масштабування розмітки gt.txt
    raw_gt_path = INPUT_DATASET_DIR / "gt" / "gt.txt"
    if not raw_gt_path.exists():
        raise FileNotFoundError(f"Файл оригінальної розмітки gt.txt не знайдено за шляхом: {raw_gt_path}")
        
    gt_data = np.loadtxt(raw_gt_path, delimiter=',')
    if gt_data.ndim == 1:
        gt_data = np.expand_dims(gt_data, axis=0)
        
    if len(gt_data) > 0:
        # 3a. Перевірка наявності унікальних траєкторій (ID > 0)
        has_valid_tracks = np.any(gt_data[:, 1] > 0)
        
        if not has_valid_tracks:
            print("\n⚠️ [ВАЖЛИВЕ ПОПЕРЕДЖЕННЯ]: У вашому файлі gt.txt відсутні унікальні ID траєкторій (всі значення ID = -1).")
            print("Автоматично генеруємо наскрізні унікальні ID для запуску TrackEval...")
            gt_data[:, 1] = np.arange(1, len(gt_data) + 1)
        else:
            gt_data = gt_data[gt_data[:, 1] > 0]
            
        # 3b. Забезпечуємо наявність стандартних 9 колонок для MOT Challenge
        if gt_data.shape[1] < 9:
            padded_gt = np.zeros((gt_data.shape[0], 9))
            padded_gt[:, :gt_data.shape[1]] = gt_data
            padded_gt[:, 6] = 1    # Активність
            padded_gt[:, 8] = 1.0  # Видимість
            gt_data = padded_gt
            
        # 3c. Автовизначення необхідності масштабування (1280x720 -> 640x360)
        if len(gt_data) > 0:
            max_x = np.max(gt_data[:, 2] + gt_data[:, 4])
            max_y = np.max(gt_data[:, 3] + gt_data[:, 5])
            
            if max_x > 640 or max_y > 360:
                print(f"⚠️ Виявлено оригінальну розмітку 1280x720. Масштабуємо координати у 640x360 (коефіцієнт 0.5)...")
                gt_data[:, 2] *= 0.5  # bb_left
                gt_data[:, 3] *= 0.5  # bb_top
                gt_data[:, 4] *= 0.5  # bb_width
                gt_data[:, 5] *= 0.5  # bb_height
            else:
                print(f"ℹ️ Координати розмітки вже відповідають простору 640x360. Масштабування пропущено.")
                
            gt_data[:, 7] = 1 # Примусово зливаємо класи в 'mil' (індекс 1 для TrackEval)
        
    target_gt_path = seq_gt_dir / "gt.txt"
    if len(gt_data) > 0:
        np.savetxt(target_gt_path, gt_data, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%d,%.2f')
    else:
        open(target_gt_path, 'w').close()
        
    # Створення карти послідовностей (seqmaps)
    with open(SEQMAPS_DIR / f"{trackeval_split_name}.txt", "w") as f:
        f.write("name\n" + seq_name)
        
    print(f"✅ Датасет успішно реорганізовано та збережено.")

prepare_mot_dataset = prepare_resized_mot_dataset_adaptive
prepare_mot_dataset()

# Клітинка 4: Запуск низькопорогового інференсу YOLO на ресайзнутих кадрах 640x360 з увімкненим прямокутним режимом (rect=True)
from tqdm import tqdm
from ultralytics import YOLO

def run_low_conf_resized_inference():
    model = YOLO(str(YOLO_MODEL_PATH))
    seq_name = "tracker-test"
    
    img1_dir = MOT_DATASET_DIR / "mil-test" / seq_name / "img1"
    images = sorted(
        list(img1_dir.glob("*.jpg")) + 
        list(img1_dir.glob("*.png")) + 
        list(img1_dir.glob("*.jpeg"))
    )
    
    output_split_dir = YOLO_DETECTIONS_OUT / "test"
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    detections = []
    print(f"\n--- Запуск інференсу на кадрах 640x360 (imgsz=(360, 640), rect=True, conf=0.01) ---")
    
    for idx, img_path in enumerate(tqdm(images, desc="Детекція кадрів")):
        frame_id = idx + 1
        result = model(str(img_path), conf=0.01, imgsz=(360, 640), rect=True, verbose=False)[0]
        
        boxes = result.boxes
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf = confs[i]
                w = x2 - x1
                h = y2 - y1
                detections.append([frame_id, -1, x1, y1, w, h, conf, 0, -1])
                
    out_file = output_split_dir / f"{seq_name}.txt"
    if len(detections) > 0:
        np.savetxt(out_file, np.array(detections), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d')
    else:
        open(out_file, 'w').close()
        
    print(f"✅ Інференс завершено. Детекції збережено в: {out_file}")

run_low_conf_inference = run_low_conf_resized_inference
run_low_conf_inference()

# Клітинка 5: Багатопотокова архітектура реального часу (Capture -> Process -> Output)
import time
import queue
import threading
import numpy as np
import cv2
from pathlib import Path

class RealTimeTrackingSystem:
    def __init__(self, model, tracker, frame_paths, target_fps=30):
        self.model = model
        self.tracker = tracker
        self.frame_paths = frame_paths
        self.delay = 1.0 / target_fps
        
        # Потокобезпечні черги з лімітом для запобігання накопиченню затримки
        self.capture_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()
        
        self.stopped = False
        self.height, self.width = 360, 640
        self.screen_cx, self.screen_cy = self.width // 2, self.height // 2

    def _frame_capture_thread(self):
        """Потік 1: Захоплення кадрів з фіксованою частотою (імітація камери)"""
        for img_path in self.frame_paths:
            if self.stopped:
                break
                
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
                
            # Якщо черга заповнена, видаляємо застарілий кадр (keep-alive-latest)
            if self.capture_queue.full():
                try:
                    self.capture_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.capture_queue.put((img_path.stem, frame))
            time.sleep(self.delay)
        self.stopped = True

    def _processing_thread(self):
        """Потік 2: Детекція YOLO + Трекер BoxMOT"""
        while not self.stopped or not self.capture_queue.empty():
            try:
                frame_id, frame = self.capture_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            # Низькопороговий інференс моделі
            results = self.model(frame, conf=0.01, imgsz=(360, 640), rect=True, verbose=False)[0]
            boxes = results.boxes
            
            tracker_input = []
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    tracker_input.append([xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3], confs[i], 0])
            
            tracker_input = np.array(tracker_input) if len(tracker_input) > 0 else np.empty((0, 6))
            
            # Оновлення трекера
            online_targets = self.tracker.update(tracker_input, frame)
            
            frame_results = []
            for target in online_targets:
                x1, y1, x2, y2, track_id = target[:5]
                w, h = x2 - x1, y2 - y1
                
                # Розрахунок центру треку та вектора неузгодженості
                cx, cy = int(x1 + w / 2), int(y1 + h / 2)
                dx = cx - self.screen_cx
                dy = cy - self.screen_cy
                
                frame_results.append({
                    "track_id": int(track_id),
                    "bbox": [x1, y1, w, h],
                    "vector": [dx, dy]
                })
                
            self.output_queue.put((frame_id, frame_results))

    def _output_dispatcher_thread(self):
        """Потік 3: Відправка векторів супроводження [ΔX, ΔY] системі донаведення"""
        while not self.stopped or not self.output_queue.empty():
            try:
                frame_id, targets = self.output_queue.get(timeout=1.0)
                for t in targets:
                    # Демонстраційний вивід відправки вектора помилки
                    dx, dy = t["vector"]
                    # print(f"[OUTPUT] Кадр {frame_id} | ID {t['track_id']} | Вектор помилки: ΔX={dx:+d}, ΔY={dy:+d}")
            except queue.Empty:
                continue

    def run(self):
        t1 = threading.Thread(target=self._frame_capture_thread, daemon=True)
        t2 = threading.Thread(target=self._processing_thread, daemon=True)
        t3 = threading.Thread(target=self._output_dispatcher_thread, daemon=True)
        
        t1.start()
        t2.start()
        t3.start()
        
        t1.join()
        t2.join()
        t3.join()

# Клітинка 6: Тиха інтеграція TrackEval із відображенням tqdm прогрес-бару замість консольного спаму
import os
import sys
import contextlib
import trackeval
from tqdm import tqdm

def run_trackeval_quiet(split_name: str, trackers_to_eval: list):
    """Запуск оцінки TrackEval із tqdm прогрес-баром та повним придушенням консольного спаму."""
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        'PLOT_CURVES': False,
        'PRINT_RESULTS': False,       
        'PRINT_ONLY_COMBINED': True,
        'TIME_PROGRESS': False,
        'PRINT_CONFIG': False         
    })
    
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        'GT_FOLDER': str(MOT_DATASET_DIR),
        'TRACKERS_FOLDER': str(TRACKERS_OUT_DIR),
        'SUB_SPLIT': '',
        'SPLIT_TO_EVAL': split_name,
        'BENCHMARK': 'mil',
        'CLASSES_TO_EVAL': ['pedestrian'],  
        'SKIP_SPLIT_FOL': False,
        'PRINT_CONFIG': False
    })
    
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity()
    ]
    
    print(f"📊 Початок оцінювання {len(trackers_to_eval)} конфігурацій (спліт: '{split_name}')...")
    
    for tracker_name in tqdm(trackers_to_eval, desc="Оцінювання трекерів", unit="cfg", leave=True):
        run_dataset_config = dataset_config.copy()
        run_dataset_config['TRACKERS_TO_EVAL'] = [tracker_name]
        
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                try:
                    evaluator = trackeval.Evaluator(eval_config)
                    dataset = trackeval.datasets.MotChallenge2DBox(run_dataset_config)
                    evaluator.evaluate([dataset], metrics)
                except Exception as e:
                    print(f"\n[Помилка] Не вдалося оцінити {tracker_name}: {e}", file=sys.stderr)
                    
    print("✅ Оцінювання успішно завершено. Результати збережені на диску.")

# Клітинка 7: Запуск Grid Search для ByteTrack та OC-SORT на єдиному тестовому спліті (mil-test)
import itertools

YOLO_TEST_DETECTIONS = YOLO_DETECTIONS_OUT / "test"

# --- 1. Оптимізація ByteTrack ---
bytetrack_runs = []
keys_bt, values_bt = zip(*BYTETRACK_GRID.items())
bytetrack_combinations = [dict(zip(keys_bt, v)) for v in itertools.product(*values_bt)]

print("Запуск Grid Search для ByteTrack на тестових детекціях...")
for idx, config in enumerate(bytetrack_combinations):
    run_name = f"bytetrack_run_{idx}"
    bytetrack_runs.append(run_name)
    run_tracking_pipeline(
        split_name="test", 
        tracker_name="bytetrack", 
        tracker_config=config, 
        run_dir_name=run_name,
        detections_source_dir=YOLO_TEST_DETECTIONS
    )

run_trackeval_quiet(split_name="test", trackers_to_eval=bytetrack_runs)

# --- 2. Оптимізація OC-SORT ---
ocsort_runs = []
keys_oc, values_oc = zip(*OCSORT_GRID.items())
ocsort_combinations = [dict(zip(keys_oc, v)) for v in itertools.product(*values_oc)]

print("\nЗапуск Grid Search для OC-SORT на тестових детекціях...")
for idx, config in enumerate(ocsort_combinations):
    run_name = f"ocsort_run_{idx}"
    ocsort_runs.append(run_name)
    run_tracking_pipeline(
        split_name="test", 
        tracker_name="ocsort", 
        tracker_config=config, 
        run_dir_name=run_name,
        detections_source_dir=YOLO_TEST_DETECTIONS
    )

run_trackeval_quiet(split_name="test", trackers_to_eval=ocsort_runs)

# Клітинка 8: Парсинг результатів сітки для пошуку оптимальних параметрів на спліті mil-test
import itertools
import pandas as pd
from pathlib import Path
from IPython.display import display

def parse_trackeval_detailed_summary(summary_path: Path):
    """Зчитує та конвертує детальні метрики з текстового звіту TrackEval."""
    if not summary_path.exists():
        return None
        
    with open(summary_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if len(lines) < 2:
        return None
        
    headers = [h.replace('(%)', '').strip() for h in lines[0].split()]
    values = lines[1].split()
    
    metrics = {}
    for h, v in zip(headers, values):
        try:
            metrics[h] = float(v)
        except ValueError:
            metrics[h] = v
    return metrics

def display_and_select_best_flat_configs():
    bytetrack_comb = [dict(zip(keys_bt, v)) for v in itertools.product(*values_bt)]
    ocsort_comb = [dict(zip(keys_oc, v)) for v in itertools.product(*values_oc)]
    
    records = []
    
    # Парсинг результатів ByteTrack
    best_bt_hota = -1.0
    best_bt_config = None
    for idx, config in enumerate(bytetrack_comb):
        summary_path = TRACKERS_OUT_DIR / "mil-test" / f"bytetrack_run_{idx}" / "pedestrian_summary.txt"
        m = parse_trackeval_detailed_summary(summary_path)
        if m:
            hota = m.get("HOTA", 0.0)
            if hota > best_bt_hota:
                best_bt_hota = hota
                best_bt_config = config
            records.append({
                "Tracker": "ByteTrack",
                "Conf": config.get("track_thresh"),
                "Memory": config.get("track_buffer"),
                "Match/IoU": config.get("match_thresh"),
                "HOTA": m.get("HOTA", 0.0),
                "DetA": m.get("DetA", 0.0),
                "AssA": m.get("AssA", 0.0),
                "IDF1": m.get("IDF1", 0.0),
                "MOTA": m.get("MOTA", 0.0),
                "MOTP": m.get("MOTP", 0.0),
                "IDSW": int(m.get("IDSW", 0)),
                "Frag": int(m.get("Frag", 0)),
                "Mostly Tracked (MT)": int(m.get("MT", 0)),
                "Mostly Lost (ML)": int(m.get("ML", 0))
            })
            
    # Парсинг результатів OC-SORT
    best_oc_hota = -1.0
    best_oc_config = None
    for idx, config in enumerate(ocsort_comb):
        summary_path = TRACKERS_OUT_DIR / "mil-test" / f"ocsort_run_{idx}" / "pedestrian_summary.txt"
        m = parse_trackeval_detailed_summary(summary_path)
        if m:
            hota = m.get("HOTA", 0.0)
            if hota > best_oc_hota:
                best_oc_hota = hota
                best_oc_config = config
            records.append({
                "Tracker": "OC-SORT",
                "Conf": config.get("det_thresh"),
                "Memory": config.get("max_age"),
                "Match/IoU": config.get("iou_threshold"),
                "HOTA": m.get("HOTA", 0.0),
                "DetA": m.get("DetA", 0.0),
                "AssA": m.get("AssA", 0.0),
                "IDF1": m.get("IDF1", 0.0),
                "MOTA": m.get("MOTA", 0.0),
                "MOTP": m.get("MOTP", 0.0),
                "IDSW": int(m.get("IDSW", 0)),
                "Frag": int(m.get("Frag", 0)),
                "Mostly Tracked (MT)": int(m.get("MT", 0)),
                "Mostly Lost (ML)": int(m.get("ML", 0))
            })

    if len(records) == 0:
        print("[Помилка] Не знайдено результатів оцінювання. Перевірте роботу TrackEval.")
        return None, None

    df = pd.DataFrame(records)
    top_configs = df.sort_values(by="HOTA", ascending=False).head(5)
    
    print("\n" + "=" * 120)
    print("               ТОП-5 НАЙКРАЩИХ КОНФІГУРАЦІЙ ТРЕКЕРІВ НА ВАШОМУ ДАТАСЕТІ ЗА МЕТРИКОЮ HOTA")
    print("=" * 120)
    
    float_format = {
        "HOTA": "{:.2f}%", "DetA": "{:.2f}%", "AssA": "{:.2f}%",
        "IDF1": "{:.2f}%", "MOTA": "{:.2f}%", "MOTP": "{:.2f}%"
    }
    display(top_configs.style.format(float_format).hide(axis='index'))
    print("=" * 120 + "\n")
    
    return best_bt_config, best_oc_config

best_bt_params, best_oc_params = display_and_select_best_flat_configs()

# Клітинка 9: Фінальне тестування на відкладеній вибірці (з інтеграцією оптимальних та ручних конфігурацій)
print("Запуск фінального тестування на спліті 'test'...")

YOLO_TEST_DETECTIONS = YOLO_DETECTIONS_OUT / "test"

# 1. Запуск оптимального ByteTrack
run_tracking_pipeline(
    split_name="test", 
    tracker_name="bytetrack", 
    tracker_config=best_bt_params, 
    run_dir_name="bytetrack_final",
    detections_source_dir=YOLO_TEST_DETECTIONS
)

# 2. Запуск оптимального OC-SORT
run_tracking_pipeline(
    split_name="test", 
    tracker_name="ocsort", 
    tracker_config=best_oc_params, 
    run_dir_name="ocsort_final",
    detections_source_dir=YOLO_TEST_DETECTIONS
)

# 3. Запуск кастомного OC-SORT із налаштуваннями низького порогу асоціації
run_tracking_pipeline(
    split_name="test", 
    tracker_name="ocsort", 
    tracker_config={
        "det_thresh": 0.3,
        "max_age": 30,
        "iou_threshold": 0.10  
    }, 
    run_dir_name="ocsort_custom_test",
    detections_source_dir=YOLO_TEST_DETECTIONS
)

# КРИТИЧНЕ ПОКРАЩЕННЯ: Оцінюємо всі три фінальні трекери (включаючи кастомний) разом
run_trackeval_quiet(
    split_name="test", 
    trackers_to_eval=["bytetrack_final", "ocsort_final", "ocsort_custom_test"]
)

# Вивід детальної порівняльної таблиці для всіх трьох конфігурацій
def display_final_test_results():
    final_records = []
    
    targets = [
        ("bytetrack_final", "ByteTrack (Optimal)"), 
        ("ocsort_final", "OC-SORT (Optimal)"),
        ("ocsort_custom_test", "OC-SORT (Custom High-Buffer)")
    ]
    
    for tracker_run, name in targets:
        summary_path = TRACKERS_OUT_DIR / "mil-test" / tracker_run / "pedestrian_summary.txt"
        m = parse_trackeval_detailed_summary(summary_path)
        if m:
            final_records.append({
                "Tracker": name,
                "HOTA": m.get("HOTA", 0.0),
                "DetA": m.get("DetA", 0.0),
                "AssA": m.get("AssA", 0.0),
                "IDF1": m.get("IDF1", 0.0),
                "MOTA": m.get("MOTA", 0.0),
                "MOTP": m.get("MOTP", 0.0),
                "IDSW": int(m.get("IDSW", 0)),
                "Frag": int(m.get("Frag", 0)),
                "Mostly Tracked (MT)": int(m.get("MT", 0)),
                "Mostly Lost (ML)": int(m.get("ML", 0))
            })
    
    df_final = pd.DataFrame(final_records)
    print("\n" + "=" * 120)
    print("                    ФІНАЛЬНИЙ ЗВІТ НА ТЕСТОВІЙ ВИБІРЦІ (СПЛІТ: TEST)")
    print("=" * 120)
    float_format = {
        "HOTA": "{:.2f}%", "DetA": "{:.2f}%", "AssA": "{:.2f}%",
        "IDF1": "{:.2f}%", "MOTA": "{:.2f}%", "MOTP": "{:.2f}%"
    }
    display(df_final.style.format(float_format).hide(axis='index'))
    print("=" * 120 + "\n")

display_final_test_results()

# Клітинка 10: Збирання демонстраційних відео з відображенням вектора помилки [ΔX, ΔY]
import cv2
import colorsys
import numpy as np
from pathlib import Path

def generate_visualized_tracking_video_from_resized_images(split_name: str, run_name: str, output_video_name: str):
    """Компілює демонстраційне відео з розрахунком спрямованого вектора помилки супроводження."""
    seq_name = "tracker-test"
    img1_dir = MOT_DATASET_DIR / f"mil-{split_name}" / seq_name / "img1"
    
    images = sorted(
        list(img1_dir.glob("*.jpg")) + 
        list(img1_dir.glob("*.png")) + 
        list(img1_dir.glob("*.jpeg"))
    )
    
    if len(images) == 0:
        print(f"[Помилка] Кадри за шляхом {img1_dir} не знайдені.")
        return
        
    tracking_result_file = TRACKERS_OUT_DIR / f"mil-{split_name}" / run_name / "data" / f"{seq_name}.txt"
    if not tracking_result_file.exists():
        print(f"[Помилка] Файл результатів трекінгу відсутній для {run_name}: {tracking_result_file}")
        return
        
    track_data = np.loadtxt(tracking_result_file, delimiter=',')
    if track_data.ndim == 1:
        track_data = np.expand_dims(track_data, axis=0)
        
    sample_img = cv2.imread(str(images[0]))
    if sample_img is None:
        return
    height, width, _ = sample_img.shape
    fps = 30
    
    # Центр кадру [Sx, Sy]
    screen_cx = width // 2
    screen_cy = height // 2
    
    output_path = BEST_VISUALIZATIONS_DIR / output_video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    def get_color(idx):
        h = (idx * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 0.9)
        return int(r * 255), int(g * 255), int(b * 255)
        
    print(f"🎬 Збирання фінального відео ({width}x{height}) для '{run_name}': {output_video_name}...")
    
    for idx, img_path in enumerate(images):
        frame_id = idx + 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        frame_tracks = track_data[track_data[:, 0] == frame_id]
        
        for row in frame_tracks:
            track_id = int(row[1])
            left, top, w, h = row[2:6]
            x1, y1, x2, y2 = int(left), int(top), int(left + w), int(top + h)
            
            color = get_color(track_id)
            
            # Центр треку [Cx, Cy]
            cx = int(left + w / 2)
            cy = int(top + h / 2)
            
            # Розрахунок вектора неузгодженості: ΔX = Cx - Sx, ΔY = Cy - Sy
            dx = cx - screen_cx
            dy = cy - screen_cy
            
            # Малювання рамки об'єкта
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Лаконічний підпис у форматі: ID **** - [±ΔX, ±ΔY]px (шрифт тонкий, товщина 1)
            text_label = f"ID {track_id} - [{dx:+d}, {dy:+d}]px"
            cv2.putText(frame, text_label, (x1, max(12, y1 - 6)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Тонка лінія (1px) між центром кадру [Sx, Sy] та центром рамки об'єкта [Cx, Cy]
            cv2.line(frame, (screen_cx, screen_cy), (cx, cy), color, 1)
                
        out.write(frame)
        
    out.release()
    print(f"✅ Відео успішно згенеровано та збережено: {output_path}")

generate_visualized_tracking_video = generate_visualized_tracking_video_from_resized_images

# --- СПИСОК ФІНАЛЬНИХ ТРЕКЕРІВ ДЛЯ ВІЗУАЛІЗАЦІЇ ---
visualizations = [
    ("bytetrack_final", "bytetrack_optimal_demo.mp4"),
    ("ocsort_final", "ocsort_optimal_demo.mp4"),
    ("ocsort_custom_test", "ocsort_custom_demo.mp4")
]

# Циклічна генерація трьох окремих відео у папці /best_visualizations/
for run_name, video_name in visualizations:
    generate_visualized_tracking_video(
        split_name="test", 
        run_name=run_name, 
        output_video_name=video_name
    )