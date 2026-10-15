import os
import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)

def frame_idx_from_name(name: str) -> int:
    """
    Extracts frame index from filename, e.g. 'f_011.jpg' -> 11.
    Finds the last sequence of digits in the filename.
    """
    match = re.findall(r'\d+', name)
    if match:
        return int(match[-1])
    raise ValueError(f"Cannot extract frame index from filename: {name}")


class Frame:
    """Represents a single frame in an episode with its image path and annotations."""
    
    def __init__(self, image_path: Path, frame_idx: int, bboxes: list[dict], is_background: bool):
        self.image_path = Path(image_path)
        self.frame_idx = frame_idx
        # List of bounding boxes: [{'label': str, 'xtl': float, 'ytl': float, 'xbr': float, 'ybr': float}]
        self.bboxes = bboxes
        self.is_background = is_background

    def load_image(self):
        """
        Loads the image from disk and returns it as a numpy array in BGR format.
        
        Returns:
            np.ndarray: The image array.
            
        Raises:
            IOError: If image loading fails.
        """
        import cv2
        img = cv2.imread(str(self.image_path))
        if img is None:
            raise IOError(f"Failed to load image: {self.image_path}")
        return img
        
    def __repr__(self) -> str:
        return (f"Frame(path={self.image_path.name}, idx={self.frame_idx}, "
                f"boxes={len(self.bboxes)}, bg={self.is_background})")


class FrameLoader:
    """Loads frames and annotations for episodes."""
    
    @staticmethod
    def parse_cvat_xml(xml_path: Path, image_files: List[Path]) -> Tuple[Dict[int, list], Optional[ET.Element], Optional[str]]:
        """
        Parses CVAT XML annotations.xml.
        Supports both track-based and image-based formats.
        
        Args:
            xml_path: Path to the XML file.
            image_files: Sorted list of image file paths in the episode.
            
        Returns:
            Tuple[Dict[int, list], Optional[ET.Element], Optional[str]]:
                - Mapping of frame_idx (int) -> list of bbox dicts.
                - The <meta> element from CVAT.
                - The CVAT format type ('track' or 'image').
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        meta = root.find("meta")
        frame_boxes = {}
        fmt_type = None
        
        # 1. Track-based format
        tracks = root.findall(".//track")
        if tracks:
            fmt_type = "track"
            for track in tracks:
                label = track.get("label")
                for box in track.findall("box"):
                    cvat_frame_idx = int(box.get("frame"))
                    # Map the 0-based CVAT frame index to the actual filename's index
                    if 0 <= cvat_frame_idx < len(image_files):
                        actual_file = image_files[cvat_frame_idx]
                        try:
                            frame_idx = frame_idx_from_name(actual_file.name)
                        except ValueError:
                            continue
                    else:
                        continue
                        
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))
                    
                    # Validate box bounds to skip degenerate ones
                    if xbr <= xtl or ybr <= ytl:
                        continue
                        
                    if frame_idx not in frame_boxes:
                        frame_boxes[frame_idx] = []
                    frame_boxes[frame_idx].append({
                        "label": label,
                        "xtl": xtl,
                        "ytl": ytl,
                        "xbr": xbr,
                        "ybr": ybr,
                        "track_id": track.get("id"),
                        "occluded": box.get("occluded", "0"),
                        "z_order": box.get("z_order", "0"),
                        "keyframe": box.get("keyframe", "0"),
                        "outside": box.get("outside", "0")
                    })
            return frame_boxes, meta, fmt_type
            
        # 2. Image-based format
        images = root.findall(".//image")
        if images:
            fmt_type = "image"
            for img in images:
                name = img.get("name") or img.get("id")
                try:
                    frame_idx = frame_idx_from_name(name)
                except ValueError:
                    continue
                    
                for box in img.findall("box"):
                    label = box.get("label")
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))
                    
                    if xbr <= xtl or ybr <= ytl:
                        continue
                        
                    if frame_idx not in frame_boxes:
                        frame_boxes[frame_idx] = []
                    frame_boxes[frame_idx].append({
                        "label": label,
                        "xtl": xtl,
                        "ytl": ytl,
                        "xbr": xbr,
                        "ybr": ybr,
                        "occluded": box.get("occluded", "0"),
                        "z_order": box.get("z_order", "0"),
                        "keyframe": box.get("keyframe", "0"),
                        "outside": box.get("outside", "0")
                    })
            return frame_boxes, meta, fmt_type
            
        return frame_boxes, meta, fmt_type

    @staticmethod
    def load_episode(ep_dir: str) -> Tuple[List[Frame], Optional[ET.Element], Optional[str]]:
        """
        Loads all frames in an episode directory.
        Reads annotations from annotations.xml if positive episode.
        
        Args:
            ep_dir: Path to the episode folder.
            
        Returns:
            Tuple[List[Frame], Optional[ET.Element], Optional[str]]:
                - Sorted list of Frame objects.
                - The <meta> element from CVAT annotations.
                - The format type ('track' or 'image').
            
        Raises:
            ValueError: If ep_dir does not exist or is not a directory.
        """
        ep_path = Path(ep_dir).resolve()
        if not ep_path.exists():
            raise ValueError(f"Episode directory does not exist: {ep_dir}")
        if not ep_path.is_dir():
            raise ValueError(f"Episode path is not a directory: {ep_dir}")
            
        is_neg = ep_path.name.endswith("_neg")
        
        # Collect all image files
        valid_exts = {".jpg", ".jpeg", ".png"}
        image_files = []
        for file in ep_path.iterdir():
            if file.is_file() and file.suffix.lower() in valid_exts:
                try:
                    idx = frame_idx_from_name(file.name)
                    image_files.append((idx, file))
                except ValueError:
                    # Skip files where frame index cannot be parsed
                    continue
                    
        # Sort images by frame index
        image_files.sort(key=lambda x: x[0])
        sorted_paths = [path for _, path in image_files]
        
        # Load annotations if positive episode
        annotations = {}
        meta = None
        fmt_type = None
        
        if not is_neg:
            xml_path = ep_path / "annotations.xml"
            if xml_path.exists():
                try:
                    annotations, meta, fmt_type = FrameLoader.parse_cvat_xml(xml_path, sorted_paths)
                except Exception as e:
                    logger.warning(f"Error parsing XML {xml_path}: {e}")
            else:
                logger.warning(f"Missing annotations.xml for positive episode: {ep_path}")
                
        # Construct Frame objects
        frames = []
        for idx, img_path in image_files:
            if is_neg:
                bboxes = []
                is_bg = True
            else:
                bboxes = annotations.get(idx, [])
                is_bg = all(b.get("outside") in ("1", "true") for b in bboxes)
                
            frames.append(Frame(
                image_path=img_path,
                frame_idx=idx,
                bboxes=bboxes,
                is_background=is_bg
            ))
            
        return frames, meta, fmt_type
