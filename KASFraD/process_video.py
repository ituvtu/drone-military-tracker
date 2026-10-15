from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from decord import VideoReader, cpu, gpu
except ImportError as exc:
    raise SystemExit("Missing dependency: decord. Install it before running this script.") from exc

try:
    import albumentations as A
except ImportError as exc:
    raise SystemExit("Missing dependency: albumentations. Install it before running this script.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart video decimation for YOLO dataset building")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output_dir", "--output", dest="output_dir", required=True, help="Output dataset root")
    parser.add_argument("--hist_threshold", type=float, default=0.35, help="Bhattacharyya threshold")
    parser.add_argument("--rapid_scale_on", type=float, default=1.03, help="Enable aggressive keep mode")
    parser.add_argument("--rapid_scale_off", type=float, default=1.01, help="Disable aggressive keep mode")
    parser.add_argument("--stride", type=int, default=2, help="Dense processing stride (check every Nth frame)")
    parser.add_argument("--grid_size", type=int, default=3, help="Grid split size (3 means 3x3 grid)")
    parser.add_argument("--max_gap_secs", type=float, default=1.0, help="Maximum allowed time gap (in seconds) between saved frames")
    return parser.parse_args()


def resolve_output_root(raw_output: str) -> Path:
    output_path = Path(raw_output)
    if output_path.is_absolute() and output_path.drive == "" and raw_output[:1] in {"/", "\\"}:
        output_path = Path.cwd() / raw_output.lstrip("/\\")
    elif not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    return output_path


def open_reader(path: str) -> VideoReader:
    try:
        return VideoReader(path, ctx=gpu(0))
    except Exception:
        return VideoReader(path, ctx=cpu(0))


def hsv_hist(rgb_frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, None).flatten()


def get_block_hists(rgb_frame: np.ndarray, grid_size: int) -> list[np.ndarray]:
    """
    Splits the frame into non-overlapping grid blocks and calculates normalized HSV histograms.
    """
    rows = np.array_split(rgb_frame, grid_size, axis=0)
    hists = []
    for row in rows:
        cols = np.array_split(row, grid_size, axis=1)
        for block in cols:
            hists.append(hsv_hist(block))
    return hists


def scale_factor(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=250, qualityLevel=0.01, minDistance=10, blockSize=7)
    if pts is None or len(pts) < 8:
        return 1.0
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None)
    if next_pts is None or status is None:
        return 1.0
    ok_prev = pts[status.ravel() == 1]
    ok_next = next_pts[status.ravel() == 1]
    if len(ok_prev) < 6:
        return 1.0
    matrix, _ = cv2.estimateAffinePartial2D(ok_prev, ok_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return 1.0
    return float((np.hypot(matrix[0, 0], matrix[0, 1]) + np.hypot(matrix[1, 0], matrix[1, 1])) * 0.5)


def build_augmenter() -> A.Compose:
    return A.Compose([
        A.Resize(720, 1280, interpolation=cv2.INTER_AREA),
    ])


def main() -> None:
    args = parse_args()
    video_path = Path(args.input)
    output_root = resolve_output_root(args.output_dir)
    output_images = output_root / "images"

    if output_images.exists():
        for file in output_images.iterdir():
            if file.is_file():
                file.unlink()
    output_images.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise SystemExit(f"Input video not found: {video_path}")

    vr = open_reader(str(video_path))
    fps = vr.get_avg_fps() or 30.0
    
    # Calculate the maximum frame gap in terms of video frames
    max_gap_frames = int(fps * args.max_gap_secs)
    
    augment = build_augmenter()

    saved = 0
    skipped = 0
    aggressive = False
    prev_block_hists = None
    prev_gray = None
    frames_since_last_save = 0

    for idx in range(len(vr)):
        frame = vr[idx].asnumpy()
        frames_since_last_save += 1

        if idx % args.stride != 0:
            continue

        current_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        scale = 1.0
        if prev_gray is not None:
            scale = scale_factor(prev_gray, current_gray)

        prev_gray = current_gray
        aggressive = scale >= args.rapid_scale_on or (aggressive and scale > args.rapid_scale_off)

        # Extract local histograms for the grid blocks
        current_block_hists = get_block_hists(frame, args.grid_size)
        
        # Check if we should force save due to maximum allowed time gap
        force_save = frames_since_last_save >= max_gap_frames

        if prev_block_hists is not None and not aggressive and not force_save:
            # Compare each block to its counterpart in the previous frame
            max_dist = 0.0
            for p_hist, c_hist in zip(prev_block_hists, current_block_hists):
                dist = cv2.compareHist(p_hist.astype(np.float32), c_hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
                if dist > max_dist:
                    max_dist = dist
            
            # If the most modified block still has low variance, skip the frame
            if max_dist < args.hist_threshold:
                skipped += 1
                continue

        prev_block_hists = current_block_hists
        frames_since_last_save = 0  # Reset counter on save
        
        augmented = augment(image=frame)["image"]
        out_file = output_images / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(out_file), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"Video: {video_path}")
    print(f"Total processed frames: {len(vr) // args.stride}")
    print(f"Saved: {saved}")
    print(f"Skipped: {skipped}")
    print(f"Output: {output_images.resolve()}")


if __name__ == "__main__":
    main()