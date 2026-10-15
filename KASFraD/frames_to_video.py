from __future__ import annotations

import argparse
import platform
import re
from pathlib import Path

import cv2

FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild video from sparse frame set")
    parser.add_argument("--frames_dir", required=True, help="Directory with frame_#### images")
    parser.add_argument("--input_video", required=True, help="Original video for fps and duration")
    parser.add_argument("--output", required=True, help="Output video path")
    return parser.parse_args()


def read_video_meta(video_path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open input video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    if frame_count <= 0:
        raise SystemExit(f"Cannot read frame count from: {video_path}")
    return fps, frame_count


def load_frames(frames_dir: Path) -> list[tuple[int, Path]]:
    items: list[tuple[int, Path]] = []
    for path in frames_dir.iterdir():
        if not path.is_file():
            continue
        match = FRAME_RE.match(path.name)
        if match is None:
            continue
        items.append((int(match.group(1)), path))

    items.sort(key=lambda item: item[0])
    if not items:
        raise SystemExit(f"No frame_#### images found in: {frames_dir}")
    return items


def build_frame_schedule(indices: list[int], total_frames: int) -> list[tuple[int, int, int]]:
    schedule: list[tuple[int, int, int]] = []
    if not indices:
        return schedule

    if indices[0] > 0:
        schedule.append((indices[0], 0, indices[0]))

    for pos, source_index in enumerate(indices):
        start_frame = source_index
        next_index = indices[pos + 1] if pos + 1 < len(indices) else total_frames
        end_frame = min(next_index, total_frames)
        if start_frame < end_frame:
            schedule.append((source_index, start_frame, end_frame))

    return schedule


def open_video_writer(output_video: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    codec_order = ["mp4v", "avc1"] if platform.system() == "Windows" else ["avc1", "mp4v"]

    for codec in codec_order:
        fourcc = getattr(cv2, "VideoWriter_fourcc")(*codec)
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, size)
        if writer.isOpened():
            print(f"Using codec: {codec}")
            return writer

    raise SystemExit(f"Cannot open output video for writing: {output_video}")


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    input_video = Path(args.input_video)
    output_video = Path(args.output)

    if not frames_dir.exists():
        raise SystemExit(f"Frames directory not found: {frames_dir}")
    if not input_video.exists():
        raise SystemExit(f"Input video not found: {input_video}")

    fps, total_frames = read_video_meta(input_video)
    frames = load_frames(frames_dir)
    frame_map = {index: path for index, path in frames}
    indices = [index for index, _ in frames]

    first_image = cv2.imread(str(frames[0][1]))
    if first_image is None:
        raise SystemExit(f"Cannot read image: {frames[0][1]}")

    height, width = first_image.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = open_video_writer(output_video, fps, (width, height))

    schedule = build_frame_schedule(indices, total_frames)
    written = 0
    last_frame = None

    for source_index, start_frame, end_frame in schedule:
        image_path = frame_map.get(source_index)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise SystemExit(f"Cannot read image: {image_path}")
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        last_frame = image

        repeat_count = max(0, end_frame - start_frame)
        for _ in range(repeat_count):
            writer.write(image)
            written += 1

    if written < total_frames and last_frame is not None:
        for _ in range(total_frames - written):
            writer.write(last_frame)
            written += 1

    writer.release()

    print(f"Input video: {input_video}")
    print(f"FPS: {fps:.3f}")
    print(f"Input frames: {total_frames}")
    print(f"Source images: {len(frames)}")
    print(f"Output frames: {written}")
    print(f"Output video: {output_video.resolve()}")


if __name__ == "__main__":
    main()