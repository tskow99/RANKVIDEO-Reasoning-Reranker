#!/usr/bin/env python3
"""
Generate video captions using Qwen VLMs.

Part of the RANKVIDEO data synthesis pipeline for creating
perception-grounded training data.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import (
        Qwen3OmniMoeProcessor,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def index_video_paths(video_root: str, extensions=(".mp4", ".mov", ".mkv", ".webm", ".avi")):
    mapping = {}
    for dirpath, _, filenames in os.walk(video_root):
        for name in filenames:
            if not any(name.lower().endswith(ext) for ext in extensions):
                continue
            key = os.path.splitext(name)[0]
            mapping[key] = os.path.join(dirpath, name)
    return mapping


def evenly_spaced_indices(frame_count: int, target: int) -> List[int]:
    if frame_count <= target:
        return list(range(frame_count))
    step = frame_count / float(target)
    return [min(frame_count - 1, int(math.floor(step * idx))) for idx in range(target)]


def load_video_frames(path: str, num_frames: int, resize: Optional[Tuple[int, int]] = None):
    if cv2 is None:
        raise RuntimeError("opencv-python required. Install: pip install opencv-python")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = evenly_spaced_indices(frame_count, num_frames)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame_rgb = cv2.resize(frame_rgb, resize, interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(frame_rgb))
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode frames from video: {path}")
    return frames


def get_video_duration(path: str) -> Optional[float]:
    try:
        if cv2 is not None:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                fps_native = cap.get(cv2.CAP_PROP_FPS) or 0
                cap.release()
                if frame_count > 0 and fps_native > 0:
                    return float(frame_count) / float(fps_native)
    except Exception:
        pass
    return None


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds <= 0:
        return "0s"
    total = int(max(1, round(seconds)))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


DEFAULT_CAPTION_PROMPT = (
    "You will see a short video. Write a concise, factual 3-6 sentence "
    "caption that describes the main event(s), people, actions, visible "
    "objects, locations, and any readable text. If the time or place is not "
    "clear, do not guess. Avoid speculation and avoid first-person language."
)


class QwenVideoCaptioner:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        device_map: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
    ):
        if device == "auto":
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_map = "auto"
        else:
            resolved_device = torch.device(device)
            device_map = None if resolved_device.type == "cpu" else device
        
        self.device = resolved_device
        dtype = getattr(torch, torch_dtype)

        if not QWEN3_AVAILABLE:
            raise ImportError("Qwen3 models not available. Update transformers.")

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=cache_dir
        )
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device_map, cache_dir=cache_dir
        )
        if device_map is None:
            self.model.to(self.device)

    @torch.inference_mode()
    def caption_video(self, video_path: str, prompt: str, fps: float = 1.0, 
                      max_new_tokens: int = 196) -> str:
        chat = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        features = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            num_frames=16,
            video_backend="decord",
            load_audio_from_video=False,
            use_audio_in_video=False,
        )
        inputs = features.to(self.device) if hasattr(features, "to") else {
            k: v.to(self.device) for k, v in features.items()
        }
        
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            use_cache=True,
        )
        
        inp_len = int(inputs["input_ids"].shape[1])
        gen_only = output_ids[0, inp_len:]
        return self.processor.decode(gen_only, skip_special_tokens=True).strip()


def load_id_list(path: str) -> List[str]:
    ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video captions")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", help="File with video IDs to process (one per line)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Omni-8B")
    parser.add_argument("--cache-dir", default=config.CACHE_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=196)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--caption-prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    video_index = index_video_paths(args.video_root)
    if not video_index:
        raise RuntimeError(f"No videos found under {args.video_root}")

    if args.ids:
        id_list = load_id_list(args.ids)
    else:
        id_list = sorted(video_index.keys())

    captioner = QwenVideoCaptioner(
        args.model_name, args.device, args.torch_dtype, cache_dir=args.cache_dir
    )

    written = 0
    skipped = 0
    total = len(id_list)
    start_time = time.perf_counter()

    with open(args.output, "w", encoding="utf-8") as out:
        for i, doc_id in enumerate(id_list):
            path = video_index.get(doc_id)
            if path is None:
                skipped += 1
                continue
            
            try:
                caption = captioner.caption_video(
                    path, args.caption_prompt, fps=args.fps, max_new_tokens=args.max_new_tokens
                )
                record = {
                    "doc_id": doc_id,
                    "text": caption,
                    "meta": {"video_id": doc_id, "source": "qwen_caption"},
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                logging.warning("Failed %s: %s", doc_id, e)
                skipped += 1

            if args.log_every and (i + 1) % args.log_every == 0:
                elapsed = time.perf_counter() - start_time
                eta = elapsed / (i + 1) * (total - i - 1)
                logging.info(
                    "Progress %d/%d (%.1f%%) written=%d skipped=%d ETA=%s",
                    i + 1, total, (i + 1) / total * 100, written, skipped, format_seconds(eta)
                )

    print(json.dumps({"written": written, "skipped": skipped, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
