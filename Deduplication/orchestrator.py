import os
import shutil
import logging
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Спробуємо імпортувати tqdm для прогрес-бару
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .version_manager import DatasetVersionManager
from .frame_loader import Frame, FrameLoader, frame_idx_from_name
from .perceptual_hasher import PerceptualHasher, FrameRegistry
from .geometry_evaluator import GeometryEvaluator
from .blur_detector import BlurDetector

logger = logging.getLogger(__name__)

class DeduplicationOrchestrator:
    """Orchestrates the hybrid cross-episode deduplication pipeline."""
    
    @staticmethod
    def _generate_stats_markdown(stats: dict) -> str:
        """Generates a professional markdown report from the final statistics."""
        total_processed = stats["total_processed"]
        total_unique = stats["total_unique"]
        total_skipped = stats["total_skipped_duplicates"]
        total_replacements = stats["total_replacements"]
        
        absolute_reduction = total_processed - total_unique
        reduction_pct = (absolute_reduction / total_processed * 100) if total_processed > 0 else 0.0
        
        md = []
        md.append("# Dataset Deduplication Report\n")
        
        md.append("## Executive Summary\n")
        md.append("| Metric | Value (Frames) | Percentage |")
        md.append("| :--- | :---: | :---: |")
        md.append(f"| **Total Source Frames (Processed)** | {total_processed} | 100.0% |")
        md.append(f"| **Total Target Frames (Retained)** | {total_unique} | {100.0 - reduction_pct:.2f}% |")
        md.append(f"| **Absolute Reduction (Deduplicated)** | {absolute_reduction} | {reduction_pct:.2f}% |")
        md.append(f"| &nbsp;&nbsp;&nbsp;&nbsp;*↳ Skipped Duplicates* | {total_skipped} | {total_skipped / total_processed * 100 if total_processed > 0 else 0.0:.2f}% |")
        md.append(f"| &nbsp;&nbsp;&nbsp;&nbsp;*↳ Quality Upgraded (Replaced)* | {total_replacements} | {total_replacements / total_processed * 100 if total_processed > 0 else 0.0:.2f}% |\n")
        
        md.append("## Frame Category Breakdown\n")
        md.append("| Category | Processed | Retained (Unique) | Replaced (Upgraded) | Skipped (Duplicates) |")
        md.append("| :--- | :---: | :---: | :---: | :---: |")
        md.append(f"| **Negative (Background)** | {stats['total_processed_neg']} | {stats['total_unique_neg']} | {stats['total_replacements_neg']} | {stats['total_skipped_neg']} |")
        md.append(f"| **Positive (With Objects)** | {stats['total_processed_pos']} | {stats['total_unique_pos']} | {stats['total_replacements_pos']} | {stats['total_skipped_pos']} |")
        md.append(f"| **Total** | {total_processed} | {total_unique} | {total_replacements} | {total_skipped} |\n")
        
        md.append("## Map-Level Statistics\n")
        md.append("| Map Name | Processed | Retained (Unique) | Replaced | Skipped | Reduction % |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        
        for m_name, m_stats in sorted(stats["maps"].items()):
            m_proc = m_stats["processed"]
            m_uniq = m_stats["unique"]
            m_repl = m_stats["replacements"]
            m_skip = m_stats["skipped"]
            m_red = m_proc - m_uniq
            m_red_pct = (m_red / m_proc * 100) if m_proc > 0 else 0.0
            md.append(f"| `{m_name}` | {m_proc} | {m_uniq} | {m_repl} | {m_skip} | {m_red_pct:.2f}% |")
            
        md.append("\n## Detailed Map Category Breakdown\n")
        md.append("| Map Name | Category | Processed | Retained (Unique) | Replaced | Skipped |")
        md.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
        for m_name, m_stats in sorted(stats["maps"].items()):
            md.append(f"| `{m_name}` | Negative (Background) | {m_stats['processed_neg']} | {m_stats['unique_neg']} | {m_stats['replacements_neg']} | {m_stats['skipped_neg']} |")
            md.append(f"| `{m_name}` | Positive (With Objects) | {m_stats['processed_pos']} | {m_stats['unique_pos']} | {m_stats['replacements_pos']} | {m_stats['skipped_pos']} |")
            md.append("| | | | | | |")  # Row separator
            
        # Clean up last separator row
        if md[-1] == "| | | | | | |":
            md.pop()
            
        md.append("\n*Report generated automatically by the DeduplicationOrchestrator pipeline.*")
        return "\n".join(md)
    
    @staticmethod
    def process_balanced_migration(source_dataset: str, target_dataset: str, config: dict) -> bool:
        """
        Runs the balanced cross-episode deduplication migration from source_dataset to target_dataset.
        
        Args:
            source_dataset: Path to the source dataset directory (e.g. dataset_v12).
            target_dataset: Path to the target dataset directory (e.g. dataset_v13).
            config: Configuration dictionary containing:
                - 'beta' (float): Balancing coefficient (default 1.0).
                - 'dhash_threshold' (int): Hamming distance threshold for duplicates (default 8).
                - 'img_width' (float): Default image width (default 1280.0).
                - 'img_height' (float): Default image height (default 720.0).
                
        Returns:
            bool: True if migration succeeded, False otherwise.
        """
        source_path = Path(source_dataset).resolve()
        target_path = Path(target_dataset).resolve()
        
        # Configure local logging to dataset_actions.log
        target_path.mkdir(parents=True, exist_ok=True)
        log_file_path = target_path / "dataset_actions.log"
        
        action_logger = logging.getLogger("dataset_actions")
        action_logger.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicates
        action_logger.handlers = []
        
        # Відключаємо поширення логів до батьківського логера, щоб уникнути засмічення консолі
        action_logger.propagate = False
        
        file_handler = logging.FileHandler(str(log_file_path), mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        action_logger.addHandler(file_handler)
        
        try:
            action_logger.info(f"Starting migration from {source_path} to {target_path}")
            
            # 1. Recreate directory structure
            try:
                DatasetVersionManager.initialize_next_version(str(source_path), str(target_path))
                action_logger.info("Mirror folder structure successfully created.")
            except Exception as e:
                action_logger.error(f"Failed to initialize next version directory structure: {e}")
                return False
                
            beta = float(config.get("beta", 1.0))
            dhash_threshold = int(config.get("dhash_threshold", 8))
            img_w = float(config.get("img_width", 1280.0))
            img_h = float(config.get("img_height", 720.0))
            
            # Initialize registries
            registries: Dict[str, FrameRegistry] = {}
            # active_counters mapping: episode relative path -> count of active frames in target_dir
            active_counters: Dict[str, int] = {}
            # active_frames_per_episode mapping: episode relative path -> list of Frame objects
            active_frames_per_episode: Dict[str, List[Frame]] = {}
            # episode_meta mapping: episode relative path -> (meta_element, fmt_type)
            episode_meta: Dict[str, Tuple[Optional[ET.Element], Optional[str]]] = {}
            
            # Initialize structured statistics
            stats = {
                "total_processed": 0,
                "total_processed_neg": 0,
                "total_processed_pos": 0,
                
                "total_unique": 0,
                "total_unique_neg": 0,
                "total_unique_pos": 0,
                
                "total_replacements": 0,
                "total_replacements_neg": 0,
                "total_replacements_pos": 0,
                
                "total_skipped_duplicates": 0,
                "total_skipped_neg": 0,
                "total_skipped_pos": 0,
                
                "maps": {}
            }
            
            # Scan all map_X/ds_Y/ep_Z directories
            episode_paths: List[str] = []
            for root, dirs, files in os.walk(source_path):
                root_path = Path(root)
                try:
                    rel = root_path.relative_to(source_path)
                except ValueError:
                    continue
                parts = rel.parts
                if len(parts) == 3 and parts[0].startswith("map_") and parts[1].startswith("ds_") and parts[2].startswith("ep_"):
                    rel_str = rel.as_posix()
                    episode_paths.append(rel_str)
                    active_counters[rel_str] = 0
                    active_frames_per_episode[rel_str] = []
                    
                    # Pre-initialize map statistics
                    normalized_path = rel_str.replace("\\", "/").strip("/")
                    path_parts = [p for p in normalized_path.split("/") if p]
                    map_name = path_parts[0] if path_parts else "unknown_map"
                    if map_name not in stats["maps"]:
                        stats["maps"][map_name] = {
                            "processed": 0, "processed_neg": 0, "processed_pos": 0,
                            "unique": 0, "unique_neg": 0, "unique_pos": 0,
                            "replacements": 0, "replacements_neg": 0, "replacements_pos": 0,
                            "skipped": 0, "skipped_neg": 0, "skipped_pos": 0,
                        }
                    
            # Sort episode paths for deterministic processing order
            episode_paths.sort()
            action_logger.info(f"Scanned {len(episode_paths)} episodes to process.")
            
            # Quick pre-scan to count total frames for the progress bar
            total_frames_count = 0
            valid_exts = {".jpg", ".jpeg", ".png"}
            for ep_rel in episode_paths:
                ep_source = source_path / ep_rel
                try:
                    for file in ep_source.iterdir():
                        if file.is_file() and file.suffix.lower() in valid_exts:
                            try:
                                _ = frame_idx_from_name(file.name)
                                total_frames_count += 1
                            except ValueError:
                                continue
                except Exception:
                    continue
            
            # Initialize progress bar
            pbar = None
            if tqdm is not None:
                pbar = tqdm(total=total_frames_count, desc="Deduplicating frames", unit="frame")
            else:
                print(f"Deduplicating total of {total_frames_count} frames...")
            
            # 2. Iterate episodes
            for ep_rel in episode_paths:
                ep_source = source_path / ep_rel
                action_logger.info(f"Processing episode: {ep_rel}")
                if pbar:
                    pbar.set_description(f"Processing {ep_rel}")
                
                try:
                    frames, meta, fmt_type = FrameLoader.load_episode(str(ep_source))
                    episode_meta[ep_rel] = (meta, fmt_type)
                except Exception as e:
                    action_logger.error(f"Error loading episode {ep_rel}: {e}")
                    continue
                    
                for frame in frames:
                    try:
                        img = frame.load_image()
                        hash_val = PerceptualHasher.calculate_dhash(img)
                        var_val = BlurDetector.get_blur_variance(img)
                    except Exception as e:
                        action_logger.warning(f"Failed to process frame {frame.image_path.name} in {ep_rel}: {e}")
                        if pbar:
                            pbar.update(1)
                        continue
                        
                    # Get map name (e.g. 'map_1') from episode relative path:
                    normalized_path = ep_rel.replace("\\", "/").strip("/")
                    path_parts = [p for p in normalized_path.split("/") if p]
                    map_name = path_parts[0] if path_parts else "unknown_map"
                    
                    if map_name not in registries:
                        registries[map_name] = FrameRegistry(threshold=dhash_threshold)
                    registry = registries[map_name]
                    
                    # Double check map initialization in stats
                    if map_name not in stats["maps"]:
                        stats["maps"][map_name] = {
                            "processed": 0, "processed_neg": 0, "processed_pos": 0,
                            "unique": 0, "unique_neg": 0, "unique_pos": 0,
                            "replacements": 0, "replacements_neg": 0, "replacements_pos": 0,
                            "skipped": 0, "skipped_neg": 0, "skipped_pos": 0,
                        }
                    
                    is_neg = frame.is_background
                    stats["total_processed"] += 1
                    stats["maps"][map_name]["processed"] += 1
                    if is_neg:
                        stats["total_processed_neg"] += 1
                        stats["maps"][map_name]["processed_neg"] += 1
                    else:
                        stats["total_processed_pos"] += 1
                        stats["maps"][map_name]["processed_pos"] += 1

                    # Search for duplicates in registry
                    duplicate_match = None
                    matched_entry_idx = -1
                    for idx, (reg_hash, reg_info) in enumerate(registry.entries):
                        dist = PerceptualHasher.hamming_distance(hash_val, reg_hash)
                        if dist <= dhash_threshold:
                            # Check Level 2 static geometry redundancy
                            is_geom_match = False
                            if frame.is_background and reg_info["is_background"]:
                                is_geom_match = True
                            elif not frame.is_background and not reg_info["is_background"]:
                                is_geom_match = GeometryEvaluator.evaluate_geometry(
                                    reg_info["bboxes"], frame.bboxes, img_w, img_h
                                )
                                
                            if is_geom_match:
                                duplicate_match = (reg_hash, reg_info)
                                matched_entry_idx = idx
                                break
                                
                    # Score values
                    score_new = var_val / ((active_counters[ep_rel] + 1) ** beta)
                    
                    if duplicate_match is not None:
                        reg_hash, reg_info = duplicate_match
                        stored_ep = reg_info["episode_rel_path"]
                        score_stored = reg_info["var"] / ((active_counters[stored_ep] + 1) ** beta)
                        
                        if score_new > score_stored:
                            # Replace
                            action_logger.info(
                                f"REPLACE: {ep_rel}/{frame.image_path.name} (Score: {score_new:.2f}) "
                                f"replaces {stored_ep}/{reg_info['image_name']} (Score: {score_stored:.2f})"
                            )
                            
                            stats["total_replacements"] += 1
                            stats["maps"][map_name]["replacements"] += 1
                            if is_neg:
                                stats["total_replacements_neg"] += 1
                                stats["maps"][map_name]["replacements_neg"] += 1
                            else:
                                stats["total_replacements_pos"] += 1
                                stats["maps"][map_name]["replacements_pos"] += 1
                            
                            # Delete physically
                            old_file_path = target_path / stored_ep / reg_info["image_name"]
                            if old_file_path.exists():
                                old_file_path.unlink()
                                
                            # Remove from active lists & counters defensively
                            try:
                                active_frames_per_episode[stored_ep].remove(reg_info["frame_obj"])
                            except ValueError:
                                # Fallback: search by image name if object identity was lost
                                found = False
                                for f in active_frames_per_episode[stored_ep]:
                                    if f.image_path.name == reg_info["image_name"]:
                                        active_frames_per_episode[stored_ep].remove(f)
                                        found = True
                                        action_logger.warning(
                                            f"Frame object identity mismatch for {reg_info['image_name']}. "
                                            f"Removed by matching image name instead."
                                        )
                                        break
                                if not found:
                                    action_logger.warning(
                                        f"Frame {reg_info['image_name']} not found in active_frames_per_episode[{stored_ep}] "
                                        f"during replacement. Skipping list removal."
                                    )
                            
                            active_counters[stored_ep] -= 1
                            
                            # Write new file
                            new_file_path = target_path / ep_rel / frame.image_path.name
                            shutil.copy2(frame.image_path, new_file_path)
                            
                            # Add to active lists & counters
                            active_frames_per_episode[ep_rel].append(frame)
                            active_counters[ep_rel] += 1
                            
                            # Update registry entry directly at the matched index to prevent collision bugs
                            new_reg_info = {
                                "frame_obj": frame,
                                "image_name": frame.image_path.name,
                                "episode_rel_path": ep_rel,
                                "is_background": frame.is_background,
                                "bboxes": frame.bboxes,
                                "var": var_val
                            }
                            registry.entries[matched_entry_idx] = (hash_val, new_reg_info)
                        else:
                            # Skip
                            action_logger.info(
                                f"SKIP: Duplicate of {stored_ep}/{reg_info['image_name']} rejected. "
                                f"Score new: {score_new:.2f} <= Score stored: {score_stored:.2f}"
                            )
                            
                            stats["total_skipped_duplicates"] += 1
                            stats["maps"][map_name]["skipped"] += 1
                            if is_neg:
                                stats["total_skipped_neg"] += 1
                                stats["maps"][map_name]["skipped_neg"] += 1
                            else:
                                stats["total_skipped_pos"] += 1
                                stats["maps"][map_name]["skipped_pos"] += 1
                    else:
                        # Unique frame (silently written, stats analyzed at the end)
                        new_file_path = target_path / ep_rel / frame.image_path.name
                        shutil.copy2(frame.image_path, new_file_path)
                        
                        # Add to active lists & counters
                        active_frames_per_episode[ep_rel].append(frame)
                        active_counters[ep_rel] += 1
                        
                        # Register entry
                        new_reg_info = {
                            "frame_obj": frame,
                            "image_name": frame.image_path.name,
                            "episode_rel_path": ep_rel,
                            "is_background": frame.is_background,
                            "bboxes": frame.bboxes,
                            "var": var_val
                        }
                        registry.entries.append((hash_val, new_reg_info))
                        
                    if pbar:
                        pbar.update(1)
                    elif stats["total_processed"] % 50 == 0:
                        print(f"Processed {stats['total_processed']}/{total_frames_count} frames...", end="\r")

            if pbar:
                pbar.close()
                
            # Compute final unique/retained counts from the registries
            for m_name, registry in registries.items():
                if m_name not in stats["maps"]:
                    continue
                for _, reg_info in registry.entries:
                    is_bg = reg_info["is_background"]
                    stats["total_unique"] += 1
                    stats["maps"][m_name]["unique"] += 1
                    if is_bg:
                        stats["total_unique_neg"] += 1
                        stats["maps"][m_name]["unique_neg"] += 1
                    else:
                        stats["total_unique_pos"] += 1
                        stats["maps"][m_name]["unique_pos"] += 1
                        
            # 3. Finalize and write annotations.xml for positive episodes
            action_logger.info("Finalizing annotations XML files...")
            for ep_rel, (meta, fmt_type) in episode_meta.items():
                if fmt_type is None:
                    # Negative or empty annotations episode, skip writing XML
                    continue
                    
                active_frames = active_frames_per_episode[ep_rel]
                # Ensure they are sorted by original frame index to maintain order
                active_frames.sort(key=lambda x: x.frame_idx)
                
                new_root = ET.Element("annotations")
                
                # Version
                ver = ET.SubElement(new_root, "version")
                ver.text = "1.1"
                
                # Meta
                if meta is not None:
                    new_root.append(copy.deepcopy(meta))
                    
                if fmt_type == "track":
                    # Reconstruct track elements
                    track_groups = {}
                    for new_idx, frame_obj in enumerate(active_frames):
                        for box in frame_obj.bboxes:
                            tid = box.get("track_id")
                            if tid not in track_groups:
                                track_groups[tid] = {
                                    "label": box["label"],
                                    "boxes": []
                                }
                            track_groups[tid]["boxes"].append((new_idx, box))
                            
                    for tid, track_info in sorted(track_groups.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0]):
                        track_elem = ET.SubElement(new_root, "track", {
                            "id": str(tid),
                            "label": track_info["label"],
                            "source": "manual"
                        })
                        for new_idx, box in track_info["boxes"]:
                            ET.SubElement(track_elem, "box", {
                                "frame": str(new_idx),
                                "outside": str(box.get("outside", "0")),
                                "keyframe": str(box.get("keyframe", "0")),
                                "occluded": str(box.get("occluded", "0")),
                                "xtl": f"{box['xtl']:.2f}",
                                "ytl": f"{box['ytl']:.2f}",
                                "xbr": f"{box['xbr']:.2f}",
                                "ybr": f"{box['ybr']:.2f}",
                                "z_order": str(box.get("z_order", "0"))
                            })
                elif fmt_type == "image":
                    # Reconstruct image elements
                    for new_idx, frame_obj in enumerate(active_frames):
                        try:
                            img_loaded = frame_obj.load_image()
                            h, w = img_loaded.shape[:2]
                        except Exception:
                            w, h = int(img_w), int(img_h)
                            
                        img_elem = ET.SubElement(new_root, "image", {
                            "id": str(new_idx),
                            "name": frame_obj.image_path.name,
                            "width": str(w),
                            "height": str(h)
                        })
                        for box in frame_obj.bboxes:
                            ET.SubElement(img_elem, "box", {
                                "label": box["label"],
                                "xtl": f"{box['xtl']:.2f}",
                                "ytl": f"{box['ytl']:.2f}",
                                "xbr": f"{box['xbr']:.2f}",
                                "ybr": f"{box['ybr']:.2f}",
                                "occluded": str(box.get("occluded", "0")),
                                "z_order": str(box.get("z_order", "0"))
                            })
                            
                # Indent and write to file
                tree = ET.ElementTree(new_root)
                ET.indent(tree, space="  ")
                xml_out = target_path / ep_rel / "annotations.xml"
                tree.write(xml_out, encoding="utf-8", xml_declaration=True)
                action_logger.info(f"Saved {fmt_type}-based XML: {ep_rel}/annotations.xml with {len(active_frames)} frames.")
                
            # Generate markdown report
            stats_out = target_path / "stats.md"
            md_content = DeduplicationOrchestrator._generate_stats_markdown(stats)
            stats_out.write_text(md_content, encoding="utf-8")
            action_logger.info(f"Saved deduplication statistics report to: {stats_out}")
            
            # Write summary stats to log file
            action_logger.info("Migration summary statistics:")
            action_logger.info(f"  Total frames processed: {stats['total_processed']}")
            action_logger.info(f"  Total unique frames written: {stats['total_unique']}")
            action_logger.info(f"  Total frame replacements: {stats['total_replacements']}")
            action_logger.info(f"  Total duplicates skipped: {stats['total_skipped_duplicates']}")
            
            action_logger.info("Final episode frame distribution:")
            for ep_rel, count in sorted(active_counters.items()):
                action_logger.info(f"  {ep_rel}: {count} frames")
                
            action_logger.info("Migration complete!")
            
            # Print beautiful summary to console
            absolute_reduction = stats["total_processed"] - stats["total_unique"]
            reduction_pct = (absolute_reduction / stats["total_processed"] * 100) if stats["total_processed"] > 0 else 0.0
            
            print("\n" + "="*50)
            print(" MIGRATION SUMMARY STATS")
            print("="*50)
            print(f"Total Frames Processed: {stats['total_processed']}")
            print(f"  ↳ Negative (Background): {stats['total_processed_neg']}")
            print(f"  ↳ Positive (With Objects): {stats['total_processed_pos']}")
            print(f"Total Retained (Unique): {stats['total_unique']}")
            print(f"  ↳ Negative: {stats['total_unique_neg']}")
            print(f"  ↳ Positive: {stats['total_unique_pos']}")
            print(f"Total Replaced (Upgraded): {stats['total_replacements']}")
            print(f"  ↳ Negative: {stats['total_replacements_neg']}")
            print(f"  ↳ Positive: {stats['total_replacements_pos']}")
            print(f"Total Skipped (Duplicates): {stats['total_skipped_duplicates']}")
            print(f"  ↳ Negative: {stats['total_skipped_neg']}")
            print(f"  ↳ Positive: {stats['total_skipped_pos']}")
            print(f"Overall Absolute Reduction: {absolute_reduction} frames")
            print(f"Overall Reduction Percentage: {reduction_pct:.2f}%")
            print("="*50)
            print(f"Detailed report saved to: {stats_out}")
            print("="*50 + "\n")
            
            return True
        finally:
            file_handler.close()
            action_logger.removeHandler(file_handler)