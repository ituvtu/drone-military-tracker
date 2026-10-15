# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:13:13.777824Z","iopub.execute_input":"2026-06-03T19:13:13.778396Z","iopub.status.idle":"2026-06-03T19:13:33.180732Z","shell.execute_reply.started":"2026-06-03T19:13:13.778363Z","shell.execute_reply":"2026-06-03T19:13:33.180007Z"}}
!pip install -q transformers 'kernels<0.12' accelerate huggingface_hub decord av -U

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:13:33.182571Z","iopub.execute_input":"2026-06-03T19:13:33.182820Z","iopub.status.idle":"2026-06-03T19:13:51.807973Z","shell.execute_reply.started":"2026-06-03T19:13:33.182793Z","shell.execute_reply":"2026-06-03T19:13:51.807002Z"}}
import os
import cv2
import numpy as np
import torch
import gc
import shutil
from accelerate import Accelerator
from huggingface_hub import login, HfApi, hf_hub_download
from transformers import Sam3VideoModel, Sam3VideoProcessor
from tqdm.auto import tqdm

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:13:51.809081Z","iopub.execute_input":"2026-06-03T19:13:51.809662Z","iopub.status.idle":"2026-06-03T19:13:51.815736Z","shell.execute_reply.started":"2026-06-03T19:13:51.809606Z","shell.execute_reply":"2026-06-03T19:13:51.814924Z"}}
# --- Configuration ---
DATASET_PATH = "/kaggle/input/datasets/ituvtu/mil-dataset/dataset_v5"
PROCESS_ALL_EPISODES = True  # True: process all episodes in DATASET_PATH; False: process only SINGLE_EPISODE
SINGLE_EPISODE = "ep_001"     # Only used if PROCESS_ALL_EPISODES is False
START_FROM_EPISODE = "ep_001"    # Set to e.g. "ep_005" to resume processing from a specific episode

USE_HF = True  # Set to True to process files from Hugging Face dataset
HF_REPO_ID = "ituvtu/annotations"
HF_PATH = "map_1/ds_3"  # Path in HF dataset containing episode zip files
SKIP_EPISODES = []  # List of episodes to skip, e.g. ["map_1\\ds_1\\ep_021.zip"]
NEGATIVE_EPISODES = []  # List of episodes to skip tracking and generate empty annotations for, e.g. ["ep_003.zip"]

USE_MANUAL_BOX = False 
EXPORT_FORMAT = "cvat"        # "cvat" for CVAT XML 1.1 video format, "mot" for MOT Challenge format 

PROMPTS = [
    ("howitzer", "howitzer, artillery"),
    ("antiair", "antiair vehicle, anti-aircraft system"),
    ("bmp", "bmp, apc, ifv, btr"),
    ("tank", "tank, armored combat vehicle"),
    ("cars", "car, military car"),
    ("truck", "military truck, cargo truck"),
]

manual_box = (1042, 569, 1069, 591) 

CLASS_PRIORITY = {
    "tank": 60,
    "bmp": 55,
    "howitzer": 50,
    "antiair": 45,
    "truck": 40,
    "cars": 30,
}

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:13:51.817598Z","iopub.execute_input":"2026-06-03T19:13:51.817931Z","iopub.status.idle":"2026-06-03T19:13:51.835340Z","shell.execute_reply.started":"2026-06-03T19:13:51.817907Z","shell.execute_reply":"2026-06-03T19:13:51.834637Z"}}
# --- Helper Functions ---
def load_input(path):
    if os.path.isdir(path):
        valid_extensions = ('.jpg', '.jpeg', '.png')
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
        files.sort()
        if not files: raise RuntimeError(f"No images found in {path}")
        frames = []
        for f in files:
            frame = cv2.imread(f)
            if frame is None: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        return frames, 1
    else:
        frame = cv2.imread(path)
        if frame is None: raise RuntimeError(f"Error opening {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return [frame], 1

def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except: return os.environ.get("HF_TOKEN")

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_class_priority(class_idx):
    if 1 <= class_idx <= len(PROMPTS):
        label = PROMPTS[class_idx - 1][0]
        return CLASS_PRIORITY.get(label, 0)
    return 0

def apply_nms(boxes, class_ids, iou_threshold=0.45):
    if len(boxes) == 0:
        return [], []
    keep = []
    order = list(range(len(boxes)))
    order.sort(key=lambda i: (get_class_priority(class_ids[i]), (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])), reverse=True)
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1: break
        remaining = []
        for j in order[1:]:
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold: continue
            remaining.append(j)
        order = remaining
    return [boxes[i] for i in keep], [class_ids[i] for i in keep]

def create_empty_mot(base_dir):
    gt_dir = os.path.join(base_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    with open(os.path.join(base_dir, "labels.txt"), "w") as f_lbl:
        for label, _ in PROMPTS: f_lbl.write(f"{label}\n")
    gt_path = os.path.join(gt_dir, "gt.txt")
    with open(gt_path, "w") as f_gt:
        pass

def create_empty_annotations(episode_name, task_name, num_frames, base_dir):
    if EXPORT_FORMAT == "cvat":
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        annotations_el = ET.Element("annotations")
        version_el = ET.SubElement(annotations_el, "version")
        version_el.text = "1.1"
        
        meta_el = ET.SubElement(annotations_el, "meta")
        task_el = ET.SubElement(meta_el, "task")
        
        name_el = ET.SubElement(task_el, "name")
        name_el.text = task_name
        
        size_el = ET.SubElement(task_el, "size")
        size_el.text = str(num_frames)
        
        mode_el = ET.SubElement(task_el, "mode")
        mode_el.text = "interpolation"
        
        labels_el = ET.SubElement(task_el, "labels")
        for label, _ in PROMPTS:
            label_el = ET.SubElement(labels_el, "label")
            label_name_el = ET.SubElement(label_el, "name")
            label_name_el.text = label
            ET.SubElement(label_el, "attributes")
            
        xml_str = ET.tostring(annotations_el, encoding="utf-8")
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent="  ")
        
        xml_path = os.path.join(base_dir, "annotations.xml")
        with open(xml_path, "w", encoding="utf-8") as f_xml:
            f_xml.write(pretty_xml)
    else:
        create_empty_mot(base_dir)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:13:51.836174Z","iopub.execute_input":"2026-06-03T19:13:51.836409Z","iopub.status.idle":"2026-06-03T19:13:51.851367Z","shell.execute_reply.started":"2026-06-03T19:13:51.836387Z","shell.execute_reply":"2026-06-03T19:13:51.850652Z"}}
# --- Tracking Classes ---
class TrackMapper:
    def __init__(self):
        self.next_track_id = 1
        self.prev_tracks = [] 
    def assign_ids(self, boxes, class_ids):
        assigned_ids = []
        used_prev = set()
        for box, cid in zip(boxes, class_ids):
            best_iou = -1
            best_id = None
            for p_box, p_cid, p_id in self.prev_tracks:
                if p_id in used_prev or p_cid != cid: continue
                iou = calculate_iou(box, p_box)
                if iou > 0.25 and iou > best_iou:
                    best_iou = iou
                    best_id = p_id
            if best_id is not None:
                assigned_ids.append(best_id)
                used_prev.add(best_id)
            else:
                assigned_ids.append(self.next_track_id)
                self.next_track_id += 1
        self.prev_tracks = list(zip(boxes, class_ids, assigned_ids))
        return assigned_ids

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:13:51.852132Z","iopub.execute_input":"2026-06-03T19:13:51.852306Z","iopub.status.idle":"2026-06-03T19:14:11.528973Z","shell.execute_reply.started":"2026-06-03T19:13:51.852288Z","shell.execute_reply":"2026-06-03T19:14:11.528342Z"}}
# --- Model Setup ---
gc.collect()
torch.cuda.empty_cache()
hf_token = get_hf_token()
if hf_token: login(token=hf_token)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = Sam3VideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:14:11.529939Z","iopub.execute_input":"2026-06-03T19:14:11.530197Z","iopub.status.idle":"2026-06-03T19:14:11.556895Z","shell.execute_reply.started":"2026-06-03T19:14:11.530172Z","shell.execute_reply":"2026-06-03T19:14:11.556038Z"}}
# --- Episode Processing Function ---
def process_episode(episode_dir, base_dir, rel_path=None):
    episode_name = os.path.basename(episode_dir)
    task_name = episode_name
    if rel_path:
        parts = rel_path.split("/")
        if len(parts) >= 2:
            ds_part = parts[-2].replace("_", "")
            ep_part = parts[-1]
            task_name = f"{ds_part}/{ep_part}"
            
    print(f"\n--- Starting processing: {task_name} ---")
    video_frames, frame_step = load_input(episode_dir)
    print(f"Loaded {len(video_frames)} frames.")
    inference_session = processor.init_video_session(video=video_frames, inference_device=device, processing_device="cpu", video_storage_device="cpu", dtype=dtype)
    with torch.inference_mode():
        if USE_MANUAL_BOX:
            box_x1, box_y1, box_x2, box_y2 = map(int, manual_box)
            processor.add_inputs_to_inference_session(inference_session, frame_idx=0, obj_ids=1, input_boxes=[[[box_x1, box_y1, box_x2, box_y2]]])
        else:
            prompts_to_add = [p[1] for p in PROMPTS]
            processor.add_text_prompt(inference_session, text=prompts_to_add)
        outputs_per_frame = {}
        pbar = tqdm(total=len(video_frames), desc=f"Tracking {episode_name}")
        for model_outputs in model.propagate_in_video_iterator(inference_session=inference_session):
            processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
            
            # Оптимізація пам'яті: копіюємо тільки потрібне на CPU/RAM, видаляємо маски
            boxes = processed_outputs.get("boxes", [])
            if hasattr(boxes, "tolist"):
                boxes = boxes.tolist()
            
            obj_ids = processed_outputs.get("object_ids", [])
            if hasattr(obj_ids, "tolist"):
                obj_ids = obj_ids.tolist()
                
            prompt_to_obj_ids = {}
            for p, ids in processed_outputs.get("prompt_to_obj_ids", {}).items():
                if hasattr(ids, "tolist"):
                    prompt_to_obj_ids[p] = ids.tolist()
                else:
                    prompt_to_obj_ids[p] = list(ids)
                    
            outputs_per_frame[model_outputs.frame_idx] = {
                "boxes": boxes,
                "object_ids": obj_ids,
                "prompt_to_obj_ids": prompt_to_obj_ids
            }
            pbar.update(1)
        pbar.close()
    
    os.makedirs(base_dir, exist_ok=True)
    
    mapper = TrackMapper()
    tracks_data = {}
    
    for frame_idx, data in sorted(outputs_per_frame.items()):
        raw_boxes = data.get("boxes", [])
        if len(raw_boxes) == 0: continue
        obj_to_class = {}
        prompt_to_obj_ids = data.get("prompt_to_obj_ids", {})
        for prompt, obj_ids in prompt_to_obj_ids.items():
            class_idx = 1
            prompt_clean = str(prompt).strip().lower()
            for idx, (label, query) in enumerate(PROMPTS, start=1):
                if query.strip().lower() == prompt_clean or label.strip().lower() == prompt_clean:
                    class_idx = idx
                    break
            for o_id in obj_ids:
                o_id_int = int(o_id.item()) if hasattr(o_id, "item") else int(o_id)
                obj_to_class[o_id_int] = class_idx
        track_ids = data.get("object_ids", [])
        if hasattr(track_ids, "tolist"):
            track_ids = track_ids.tolist()
        raw_class_ids = []
        for i in range(len(raw_boxes)):
            t_id = track_ids[i] if i < len(track_ids) else 1
            t_id_int = int(t_id.item()) if hasattr(t_id, "item") else int(t_id)
            raw_class_ids.append(obj_to_class.get(t_id_int, 1))
        clean_boxes, clean_class_ids = apply_nms(raw_boxes, raw_class_ids)
        assigned_track_ids = mapper.assign_ids(clean_boxes, clean_class_ids)
        
        for box, track_id, class_id in zip(clean_boxes, assigned_track_ids, clean_class_ids):
            if track_id not in tracks_data:
                tracks_data[track_id] = []
            tracks_data[track_id].append({
                "frame": frame_idx,
                "box": box,
                "class_id": class_id
            })

    if EXPORT_FORMAT == "cvat":
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        annotations_el = ET.Element("annotations")
        version_el = ET.SubElement(annotations_el, "version")
        version_el.text = "1.1"
        
        meta_el = ET.SubElement(annotations_el, "meta")
        task_el = ET.SubElement(meta_el, "task")
        
        name_el = ET.SubElement(task_el, "name")
        name_el.text = task_name
        
        size_el = ET.SubElement(task_el, "size")
        size_el.text = str(len(video_frames))
        
        mode_el = ET.SubElement(task_el, "mode")
        mode_el.text = "interpolation"
        
        labels_el = ET.SubElement(task_el, "labels")
        for label, _ in PROMPTS:
            label_el = ET.SubElement(labels_el, "label")
            label_name_el = ET.SubElement(label_el, "name")
            label_name_el.text = label
            ET.SubElement(label_el, "attributes")
            
        for track_id, boxes_list in sorted(tracks_data.items()):
            boxes_list.sort(key=lambda x: x["frame"])
            if not boxes_list:
                continue
            
            class_id = boxes_list[0]["class_id"]
            label_name = PROMPTS[class_id - 1][0] if 1 <= class_id <= len(PROMPTS) else "unknown"
            
            track_el = ET.SubElement(annotations_el, "track", {
                "id": str(track_id),
                "label": label_name,
                "source": "manual"
            })
            
            min_f = boxes_list[0]["frame"]
            max_f = boxes_list[-1]["frame"]
            active_boxes = {x["frame"]: x["box"] for x in boxes_list}
            
            last_box = None
            was_outside = True
            
            for f in range(min_f, max_f + 2):
                if f >= len(video_frames):
                    if not was_outside and last_box is not None:
                        x1, y1, x2, y2 = last_box
                        ET.SubElement(track_el, "box", {
                            "frame": str(f - 1),
                            "keyframe": "1",
                            "outside": "1",
                            "occluded": "0",
                            "xtl": f"{x1:.2f}",
                            "ytl": f"{y1:.2f}",
                            "xbr": f"{x2:.2f}",
                            "ybr": f"{y2:.2f}",
                            "z_order": "0"
                        })
                    break
                
                if f in active_boxes:
                    box = active_boxes[f]
                    x1, y1, x2, y2 = box
                    ET.SubElement(track_el, "box", {
                        "frame": str(f),
                        "keyframe": "1",
                        "outside": "0",
                        "occluded": "0",
                        "xtl": f"{x1:.2f}",
                        "ytl": f"{y1:.2f}",
                        "xbr": f"{x2:.2f}",
                        "ybr": f"{y2:.2f}",
                        "z_order": "0"
                    })
                    last_box = box
                    was_outside = False
                else:
                    if not was_outside:
                        x1, y1, x2, y2 = last_box
                        ET.SubElement(track_el, "box", {
                            "frame": str(f),
                            "keyframe": "1",
                            "outside": "1",
                            "occluded": "0",
                            "xtl": f"{x1:.2f}",
                            "ytl": f"{y1:.2f}",
                            "xbr": f"{x2:.2f}",
                            "ybr": f"{y2:.2f}",
                            "z_order": "0"
                        })
                        was_outside = True
                        
        xml_str = ET.tostring(annotations_el, encoding="utf-8")
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent="  ")
        
        xml_path = os.path.join(base_dir, "annotations.xml")
        with open(xml_path, "w", encoding="utf-8") as f_xml:
            f_xml.write(pretty_xml)
            
    else:
        mot_rows = []
        for track_id, boxes_list in tracks_data.items():
            for item in boxes_list:
                frame_idx = item["frame"]
                box = item["box"]
                class_id = item["class_id"]
                mot_frame_idx = (frame_idx * frame_step) + 1
                x1, y1, x2, y2 = box
                mot_rows.append((mot_frame_idx, track_id, x1, y1, x2 - x1, y2 - y1, class_id))
        
        mot_rows.sort(key=lambda x: (x[0], x[1]))
        
        gt_dir = os.path.join(base_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)
        with open(os.path.join(base_dir, "labels.txt"), "w") as f_lbl:
            for label, _ in PROMPTS: f_lbl.write(f"{label}\n")
        gt_path = os.path.join(gt_dir, "gt.txt")
        with open(gt_path, "w") as f_gt:
            for row in mot_rows:
                f_gt.write(f"{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,{row[6]},1\n")
    # Free up memory
    del inference_session, video_frames, outputs_per_frame, tracks_data
    gc.collect()
    torch.cuda.empty_cache()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-06-03T19:14:11.557763Z","iopub.execute_input":"2026-06-03T19:14:11.558108Z","execution_failed":"2026-06-03T19:22:18.249Z"}}
# --- Run tracking ---
if USE_HF:
    import time

    POLL_INTERVAL_SEC = 60   # how long to wait before re-checking HF for new episodes
    POLL_MAX_WAIT_SEC = 3600 # stop after 1h of no new episodes (safety exit)

    api = HfApi()

    temp_download_dir = "./temp_hf_download"
    temp_extract_dir  = "./temp_hf_extract"
    os.makedirs(temp_download_dir, exist_ok=True)

    normalized_skips     = {s.replace('\\', '/').lower() for s in SKIP_EPISODES}
    normalized_negatives = {n.replace('\\', '/').lower() for n in NEGATIVE_EPISODES}

    processed_zips = set()   # track what we already annotated

    # Helper: apply START_FROM_EPISODE filter (only on first poll)
    first_poll = True

    idle_since = None  # timestamp when we first found nothing new

    while True:
        print(f"\n[POLL] Checking HF for new episodes under data/{HF_PATH}/ ...")
        all_files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
        hf_zips = sorted(
            f for f in all_files
            if f.startswith(f"data/{HF_PATH}/") and f.endswith(".zip")
        )

        # On the very first poll, apply START_FROM_EPISODE offset
        if first_poll and START_FROM_EPISODE:
            start_index = None
            for idx, hf_zip in enumerate(hf_zips):
                ep_name = os.path.basename(hf_zip).replace(".zip", "")
                if (ep_name.lower() == START_FROM_EPISODE.lower() or
                        START_FROM_EPISODE.lower() in hf_zip.lower()):
                    start_index = idx
                    break
            if start_index is not None:
                hf_zips = hf_zips[start_index:]
                print(f"[POLL] Resuming from: {hf_zips[0]}")
        first_poll = False

        # Find new (not yet processed) episodes
        new_zips = [z for z in hf_zips if z not in processed_zips]

        if not new_zips:
            elapsed_idle = time.time() - idle_since if idle_since else 0
            if idle_since is None:
                idle_since = time.time()
                print(f"[POLL] No new episodes. Waiting {POLL_INTERVAL_SEC}s before retry...")
            elif elapsed_idle >= POLL_MAX_WAIT_SEC:
                print(f"[POLL] No new episodes for {POLL_MAX_WAIT_SEC}s. Exiting.")
                break
            else:
                print(f"[POLL] Still no new episodes (idle {int(elapsed_idle)}s / {POLL_MAX_WAIT_SEC}s). Retrying in {POLL_INTERVAL_SEC}s...")
            time.sleep(POLL_INTERVAL_SEC)
            continue

        idle_since = None  # reset idle timer when new episodes found
        print(f"[POLL] Found {len(new_zips)} new episode(s) to process.")

        for hf_zip_path in new_zips:
            if hf_zip_path.lower() in normalized_skips:
                print(f"Skipping HF episode {hf_zip_path} (in SKIP_EPISODES).")
                processed_zips.add(hf_zip_path)
                continue

            ep_filename  = os.path.basename(hf_zip_path)
            episode_name = ep_filename.replace(".zip", "")

            is_negative = "_neg" in hf_zip_path.lower()
            if not is_negative:
                for neg in normalized_negatives:
                    if neg in hf_zip_path.lower():
                        is_negative = True
                        break

            # 1. Download frame zip from HF
            print(f"\n[EP] {episode_name} — downloading...")
            local_zip_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_zip_path,
                repo_type="dataset",
                local_dir=temp_download_dir,
            )

            # 2. Extract ZIP locally
            ep_extract_dir = os.path.join(temp_extract_dir, episode_name)
            shutil.rmtree(ep_extract_dir, ignore_errors=True)
            os.makedirs(ep_extract_dir, exist_ok=True)
            shutil.unpack_archive(local_zip_path, ep_extract_dir)

            # 3. Output dir
            local_out_dir = os.path.join("./temp_hf_out", episode_name)
            shutil.rmtree(local_out_dir, ignore_errors=True)
            os.makedirs(local_out_dir, exist_ok=True)

            # 4. Track or create empty annotations
            rel_path = f"{HF_PATH}/{episode_name}"
            if is_negative:
                print(f"[EP] {episode_name} — NEGATIVE, creating empty annotations...")
                valid_extensions = ('.jpg', '.jpeg', '.png')
                num_frames = len([f for f in os.listdir(ep_extract_dir) if f.lower().endswith(valid_extensions)])
                create_empty_annotations(episode_name, rel_path, num_frames, local_out_dir)
            else:
                process_episode(ep_extract_dir, local_out_dir, rel_path)

            # 5. Zip annotations
            annotation_zip_name = f"./temp_hf_out/{episode_name}"
            annotation_zip_path = shutil.make_archive(annotation_zip_name, 'zip', local_out_dir)

            # 6. Upload to HF
            hf_upload_path = f"annotations/{HF_PATH}/{episode_name}/sam3/{episode_name}.zip"
            print(f"[EP] {episode_name} — uploading annotations -> {hf_upload_path}")
            api.upload_file(
                path_or_fileobj=annotation_zip_path,
                path_in_repo=hf_upload_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"SAM3 annotations: {episode_name}",
            )
            print(f"[EP] {episode_name} -> done")

            # 7. Mark processed + cleanup
            processed_zips.add(hf_zip_path)
            shutil.rmtree(ep_extract_dir, ignore_errors=True)
            shutil.rmtree(local_out_dir, ignore_errors=True)
            if os.path.exists(annotation_zip_path):
                os.remove(annotation_zip_path)
            if os.path.exists(local_zip_path):
                os.remove(local_zip_path)

    # Final cleanup
    shutil.rmtree(temp_download_dir, ignore_errors=True)
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    print("\nDone processing all HF episodes!")

else:
    if PROCESS_ALL_EPISODES:
        if os.path.exists(DATASET_PATH):
            episode_dirs = []
            for root, dirs, files in os.walk(DATASET_PATH):
                has_images = any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files)
                if has_images and os.path.basename(root).lower().startswith("ep_"):
                    episode_dirs.append(root)
            episode_dirs.sort()
            
            if START_FROM_EPISODE:
                start_index = None
                for idx, ep_dir in enumerate(episode_dirs):
                    rel_path = os.path.relpath(ep_dir, DATASET_PATH).replace("\\", "/")
                    if (os.path.basename(ep_dir).lower() == START_FROM_EPISODE.lower() or
                        START_FROM_EPISODE.lower() in rel_path.lower()):
                        start_index = idx
                        break
                if start_index is None:
                    print(f"❌ Error: START_FROM_EPISODE '{START_FROM_EPISODE}' not found in dataset!")
                    print("Available paths:")
                    for ep_dir in episode_dirs:
                        print(f"  - {os.path.relpath(ep_dir, DATASET_PATH).replace('\\', '/')}")
                    raise SystemExit("Aborting run due to invalid START_FROM_EPISODE.")
                
                episode_dirs = episode_dirs[start_index:]
                print(f"Resuming run. Filtered to start from: {os.path.relpath(episode_dirs[0], DATASET_PATH).replace('\\', '/')}")
            
            if not episode_dirs:
                print(f"No episode directories found under {DATASET_PATH}!")
            else:
                print(f"Found {len(episode_dirs)} episodes to process.")
                dataset_name = os.path.basename(os.path.abspath(DATASET_PATH))
                master_base_dir = f"/kaggle/working/cvat_mot_{dataset_name}"
                # Only clean output dir if starting fresh
                if os.path.exists(master_base_dir) and not START_FROM_EPISODE:
                    shutil.rmtree(master_base_dir)
                os.makedirs(master_base_dir, exist_ok=True)
                
                normalized_skips = {s.replace('\\', '/').lower() for s in SKIP_EPISODES}
                normalized_negatives = {n.replace('\\', '/').lower() for n in NEGATIVE_EPISODES}
                for ep_dir in episode_dirs:
                    rel_path = os.path.relpath(ep_dir, DATASET_PATH).replace("\\", "/")
                    check_path = rel_path + ".zip"
                    if check_path.lower() in normalized_skips:
                        print(f"Skipping local episode {rel_path} as requested in SKIP_EPISODES.")
                        continue
                    
                    is_negative = "_neg" in rel_path.lower()
                    if not is_negative:
                        for neg in normalized_negatives:
                            if neg in rel_path.lower():
                                is_negative = True
                                break
                            
                    ep_base_dir = os.path.join(master_base_dir, rel_path)
                    
                    if is_negative:
                        print(f"Generating EMPTY annotations for NEGATIVE local episode {rel_path}...")
                        valid_extensions = ('.jpg', '.jpeg', '.png')
                        num_frames = len([f for f in os.listdir(ep_dir) if f.lower().endswith(valid_extensions)])
                        os.makedirs(ep_base_dir, exist_ok=True)
                        create_empty_annotations(os.path.basename(ep_dir), rel_path, num_frames, ep_base_dir)
                    else:
                        process_episode(ep_dir, ep_base_dir, rel_path)
                    
                final_output_dir = master_base_dir
                print(f"\n✅ All episodes processed! Target folder: {master_base_dir}")
        else:
            print(f"Dataset path {DATASET_PATH} not found!")
    else:
        single_ep_dir = os.path.join(DATASET_PATH, SINGLE_EPISODE)
        base_dir = f"/kaggle/working/cvat_mot_{SINGLE_EPISODE}"
        
        is_negative = False
        if "_neg" in SINGLE_EPISODE.lower():
            is_negative = True
        else:
            normalized_negatives = {n.replace('\\', '/').lower() for n in NEGATIVE_EPISODES}
            for neg in normalized_negatives:
                if neg in SINGLE_EPISODE.lower():
                    is_negative = True
                    break
                
        if is_negative:
            print(f"Generating EMPTY annotations for NEGATIVE single episode {SINGLE_EPISODE}...")
            valid_extensions = ('.jpg', '.jpeg', '.png')
            num_frames = len([f for f in os.listdir(single_ep_dir) if f.lower().endswith(valid_extensions)])
            os.makedirs(base_dir, exist_ok=True)
            create_empty_annotations(SINGLE_EPISODE, SINGLE_EPISODE, num_frames, base_dir)
        else:
            process_episode(single_ep_dir, base_dir)
        final_output_dir = base_dir
        print(f"✅ Single episode processed! Target folder: {base_dir}")
    
    # --- Zip and Download ---
    if 'final_output_dir' in globals() and os.path.exists(final_output_dir):
        dir_name = os.path.basename(os.path.abspath(final_output_dir))
        archive_basename = dir_name
        if archive_basename.startswith("cvat_mot_"):
            archive_basename = archive_basename[len("cvat_mot_"):]
        
        master_archive_name = f"/kaggle/working/{archive_basename}_annotations"
        zip_path = shutil.make_archive(master_archive_name, 'zip', final_output_dir)
        print(f"Zip created: {zip_path}")
        
        # Clean up folder to save space
        shutil.rmtree(final_output_dir, ignore_errors=True)
        
        # Generate download link
        display(FileLink(f"{archive_basename}_annotations.zip"))
    else:
        # Fallback to hardcoded paths if ran out-of-order
        fallback_dir = '/kaggle/working/cvat_mot_dataset_v5'
        if os.path.exists(fallback_dir):
            zip_path = shutil.make_archive('/kaggle/working/dataset_v5_annotations', 'zip', fallback_dir)
            print(f"Zip created from fallback: {zip_path}")
            shutil.rmtree(fallback_dir, ignore_errors=True)
            display(FileLink('dataset_v5_annotations.zip'))
        else:
            print("Error: No folder to archive. Make sure tracking step completed successfully.")