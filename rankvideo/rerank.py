#!/usr/bin/env python3
"""
RANKVIDEO reranking script.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rankvideo.inference import VLMReranker


def parse_args():
    parser = argparse.ArgumentParser(description="Rerank videos with RANKVIDEO")
    parser.add_argument("--model", default=config.HF_MODEL_PATH)
    parser.add_argument("--cache-dir", default=config.CACHE_DIR)
    parser.add_argument("--video2queries", required=True)
    parser.add_argument("--query-mapping", default=config.QUERY_MAPPING_PATH)
    parser.add_argument("--video-dir", default=config.VIDEO_DIR)
    parser.add_argument("--output-dir", default="outputs/reranking")
    parser.add_argument("--id-start", type=int, default=None)
    parser.add_argument("--id-end", type=int, default=None)
    parser.add_argument("--fps", type=float, default=config.FPS)
    parser.add_argument("--max-frames", type=int, default=config.MAX_FRAMES)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    return parser.parse_args()


def video_id_to_path(video_id, video_dir):
    shard_int = (int(video_id) // 100) + 1
    shard_str = str(shard_int).zfill(6)
    return os.path.join(video_dir, shard_str, f"{video_id}.mp4")


def load_query_mapping(path):
    query2text = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                query2text[parts[0]] = parts[1]
    return query2text


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    video2query = json.load(open(args.video2queries))
    query2text = load_query_mapping(args.query_mapping)

    all_videos = list(video2query.keys())
    if args.id_start is not None and args.id_end is not None:
        batch_videos = all_videos[args.id_start:args.id_end]
    else:
        batch_videos = all_videos

    missing_videos = []
    model_name = os.path.basename(args.model).replace('/', '-')
    for video_id in batch_videos:
        output_file = f"{video_id}_reranking_{model_name}.json"
        if not os.path.exists(os.path.join(args.output_dir, output_file)):
            missing_videos.append(video_id)

    if not missing_videos:
        print("All videos already reranked")
        return

    print(f"Reranking {len(missing_videos)} videos")

    reranker = VLMReranker(
        model_path=args.model,
        cache_dir=args.cache_dir,
        gpu_memory_utilization=args.gpu_memory,
        tensor_parallel_size=args.tensor_parallel,
    )

    failed_log = os.path.join(args.output_dir, "failed.log")

    for video_id in tqdm(missing_videos, desc="Processing"):
        try:
            video_path = video_id_to_path(video_id, args.video_dir)
            if not os.path.exists(video_path):
                continue

            queries = []
            query_ids = video2query[video_id]
            for qid in query_ids:
                if qid in query2text:
                    queries.append(query2text[qid])

            if not queries:
                continue

            scores = reranker.score_batch(
                queries, [video_path] * len(queries),
                fps=args.fps, max_frames=args.max_frames,
            )

            results = {}
            for qid, score in zip(query_ids, scores):
                results[qid] = score

            output_file = f"{video_id}_reranking_{model_name}.json"
            with open(os.path.join(args.output_dir, output_file), 'w') as f:
                json.dump({video_id: results}, f, indent=2)

        except Exception as e:
            with open(failed_log, "a") as f:
                f.write(f"{video_id}: {e}\n")


if __name__ == "__main__":
    main()
