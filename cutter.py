import json
import os
import subprocess
import re

def load_lossless_cut_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as file:
        content = file.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        content = re.sub(r'([{,]\s*)(\w+)\s*:', r'\1"\2":', content)
        content = re.sub(r"'([^']*)'", r'"\1"', content)
        content = re.sub(r',\s*([\]}])', r'\1', content)
        return json.loads(content)


def process_segments(
    segments: list, fps: float, min_duration: float = 1.0
) -> list:
    selected_segments = [seg for seg in segments if seg.get("selected", False)]
    selected_segments.sort(key=lambda x: x["start"])

    selected_segments = [seg for seg in selected_segments if seg["start"] < seg["end"]]

    frame_delta = 1.0 / fps
    valid_segments = []

    idx = 1
    for seg in selected_segments:
        start = seg["start"]
        end = seg["end"]

        if valid_segments:
            prev_end = valid_segments[-1]["end"]
            if start <= prev_end:
                start = prev_end + frame_delta
                if start >= end:
                    continue

        duration = end - start
        is_short = duration < min_duration

        suffix = "_short" if is_short else ""
        expected_name = f"ep_{idx:03d}{suffix}"

        if not seg.get("name"):
            seg["name"] = expected_name

        valid_segments.append({
            "start": start,
            "end": end,
            "name": seg["name"]
        })
        idx += 1

    return valid_segments


def cut_video_segments(
    input_video: str, segments: list, output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base_name, ext = os.path.splitext(os.path.basename(input_video))

    for idx, seg in enumerate(segments, start=1):
        start = seg["start"]
        end = seg["end"]
        name = seg.get("name", "segment")
        output_file = os.path.join(
            output_dir, f"{name}{ext}"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-to",
            f"{end:.6f}",
            "-i",
            input_video,
            "-c",
            "copy",
            output_file,
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            error_message = error.stderr.decode()
            print(f"Error cutting segment {idx}: {error_message}")
        except FileNotFoundError:
            print("Error: ffmpeg executable not found. Please install ffmpeg and add it to system PATH.")
            break


if __name__ == "__main__":

    PROJECT_JSON_PATH = r"videos\map_3\2026-05-24 15-09-06-proj.txt"
    INPUT_VIDEO_PATH = r"videos\map_3\2026-05-24 15-09-06.mp4"
    TARGET_FPS = 30.0
    MIN_SEGMENT_DURATION = 3
    OUTPUT_DIRECTORY = r"videos\map_3\ds_1"

    if os.path.exists(PROJECT_JSON_PATH):
        project_data = load_lossless_cut_json(PROJECT_JSON_PATH)

        raw_segments = project_data.get("cutSegments", [])

        cleaned_segments = process_segments(
            raw_segments, TARGET_FPS, MIN_SEGMENT_DURATION
        )

        with open(PROJECT_JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(project_data, file, indent=2, ensure_ascii=False)

        if os.path.exists(INPUT_VIDEO_PATH):
            cut_video_segments(
                INPUT_VIDEO_PATH, cleaned_segments, OUTPUT_DIRECTORY
            )
        else:
            print(f"Error: Video file {INPUT_VIDEO_PATH} not found.")
    else:
        print(f"Error: Project file {PROJECT_JSON_PATH} not found.")