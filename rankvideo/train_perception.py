"""
Stage 1: Perception-grounded supervised fine-tuning.

Trains the VLM to generate captions grounded in video content.
"""

import argparse
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Set

import torch
from datasets import load_dataset
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from rankvideo.utils import (
    set_seed, timestamp, timestamp_for_name, safe_serialize, 
    maybe_to_dict, valid_example, find_last_subsequence
)

logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in video and text understanding. "
    "Given a video, your task is to produce an accurate caption. "
    "Respond within <think></think>"
)

USER_PROMPT = "Caption this video. Respond within <think></think>."


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Perception-grounded SFT")
    parser.add_argument("--model", default=config.BASE_MODEL_ID)
    parser.add_argument("--train-data", default=config.TRAIN_DATA_PATH)
    parser.add_argument("--eval-data", default=config.EVAL_DATA_PATH)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=config.CACHE_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--qlora", action="store_true", default=config.USE_QLORA)
    parser.add_argument("--no-qlora", action="store_false", dest="qlora")
    parser.add_argument("--wandb", action="store_true", default=config.LOG_TO_WANDB)
    parser.add_argument("--fps", type=float, default=config.FPS)
    parser.add_argument("--max-frames", type=int, default=config.MAX_FRAMES)
    return parser.parse_args()


def extract_caption(ex):
    ev = ex.get("evidence") or {}
    caption = ex.get("caption") or ev.get("caption") or ""
    return caption.strip()


def has_caption_and_video(ex):
    if not extract_caption(ex):
        return False
    vids = ex.get("videos") or []
    vp_raw = vids[0] if vids else None
    return bool(vp_raw and isinstance(vp_raw, str) and os.path.isfile(vp_raw))


def extract_doc_id(ex):
    doc_id = ex.get("doc_id")
    if doc_id is not None:
        return str(doc_id)
    meta = ex.get("meta") or {}
    if isinstance(meta, dict) and meta.get("doc_id") is not None:
        return str(meta["doc_id"])
    vids = ex.get("videos") or []
    vp_raw = vids[0] if vids else None
    if isinstance(vp_raw, str) and vp_raw:
        return f"video:{vp_raw}"
    return None


def dedupe_by_doc_id(dataset, label):
    seen = set()
    before = len(dataset)
    
    def keep(ex):
        doc_id = extract_doc_id(ex)
        if doc_id is None:
            return True
        if doc_id in seen:
            return False
        seen.add(doc_id)
        return True
    
    deduped = dataset.filter(keep)
    logging.info("[dedupe] %s dropped=%d kept=%d", label, before - len(deduped), len(deduped))
    return deduped


def build_messages(ex, fps, max_frames):
    caption = extract_caption(ex)
    if not caption:
        return {}

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": USER_PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"<think>{caption}</think>"}]},
    ]

    vids = ex.get("videos") or []
    vp_raw = vids[0] if vids else None
    if vp_raw and isinstance(vp_raw, str) and os.path.isfile(vp_raw):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        messages[last_user]["content"] = [
            {"type": "video", "video": vp_raw, "fps": fps, "max_frames": max_frames},
            *messages[last_user]["content"],
        ]
    else:
        return {}
    
    out = {"messages": messages}
    meta = ex.get("meta") or {}
    for k in ("query_id", "doc_id"):
        if k in ex:
            out[k] = ex[k]
        elif k in meta:
            out[k] = meta[k]
    return out


class VideoCollator:
    def __init__(self, processor, assistant_prefix_ids, fps, num_frames):
        self.processor = processor
        self.assistant_prefix_ids = list(assistant_prefix_ids)
        self.fps = fps
        self.num_frames = num_frames

    def __call__(self, batch):
        msgs_raw = [ex["messages"] for ex in batch]
        msgs = []
        
        for ex, conv in zip(batch, msgs_raw):
            new_conv = []
            for msg in conv:
                content = msg.get("content")
                if isinstance(content, list):
                    new_parts = []
                    for ele in content:
                        if not isinstance(ele, dict):
                            new_parts.append(ele)
                            continue
                        etype = ele.get("type")
                        if etype == "video":
                            v = ele.get("video")
                            if isinstance(v, (str, list, tuple)):
                                new_parts.append(ele)
                        else:
                            if ele.get("video", "__absent__") is None:
                                ele = dict(ele)
                                ele.pop("video", None)
                            new_parts.append(ele)
                    nm = dict(msg)
                    nm["content"] = new_parts
                    new_conv.append(nm)
                else:
                    new_conv.append(msg)
            msgs.append(new_conv)

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
            for m in msgs
        ]

        images, videos, video_kwargs = process_vision_info(
            msgs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        if video_kwargs is None:
            video_kwargs = {}

        video_meta = None
        if videos:
            videos, metas = zip(*videos)
            videos, video_meta = list(videos), list(metas)
        else:
            videos = None

        enc = self.processor(
            text=texts, images=images, videos=videos, video_metadata=video_meta,
            do_resize=False, padding=True, return_tensors="pt", **video_kwargs
        )

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        labels = input_ids.clone()
        
        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            pos = find_last_subsequence(ids, self.assistant_prefix_ids)
            if pos == -1:
                labels[i, :] = -100
            else:
                cutoff = pos + len(self.assistant_prefix_ids)
                labels[i, :cutoff] = -100
                labels[i, attn_mask[i] == 0] = -100
        
        enc["labels"] = labels
        return enc


def main():
    args = parse_args()
    
    if args.wandb:
        import wandb
        if config.WANDB_API_KEY:
            wandb.login(key=config.WANDB_API_KEY, relogin=True)
    
    run_ts = timestamp_for_name()
    model_save_dir = os.path.join(args.output_dir, f"perception-{run_ts}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    set_seed(args.seed)
    os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)
    os.environ.setdefault("WANDB_MODE", "online" if args.wandb else "disabled")

    processor = AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    raw_train = load_dataset("json", data_files=args.train_data, split="train")
    raw_eval = load_dataset("json", data_files=args.eval_data, split="train")

    raw_train = raw_train.filter(has_caption_and_video)
    raw_eval = raw_eval.filter(has_caption_and_video)
    raw_train = dedupe_by_doc_id(raw_train, label="train")
    raw_eval = dedupe_by_doc_id(raw_eval, label="eval")

    mapped_train = raw_train.map(
        lambda ex: build_messages(ex, args.fps, args.max_frames),
        remove_columns=[c for c in raw_train.column_names if c not in {"messages", "query_id", "doc_id"}],
    )
    mapped_eval = raw_eval.map(
        lambda ex: build_messages(ex, args.fps, args.max_frames),
        remove_columns=[c for c in raw_eval.column_names if c not in {"messages", "query_id", "doc_id"}],
    )

    train_dataset = mapped_train.filter(valid_example)
    eval_dataset = mapped_eval.filter(valid_example)

    assistant_prefix = "<|im_start|>assistant\n"
    assistant_prefix_ids = processor.tokenizer.encode(assistant_prefix, add_special_tokens=False)

    data_collator = VideoCollator(processor, assistant_prefix_ids, args.fps, config.NFRAMES)

    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.LORA_TARGET_MODULES,
    )

    model_init_kwargs = {"trust_remote_code": True, "cache_dir": args.cache_dir}
    if args.qlora:
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_init_kwargs["torch_dtype"] = torch.bfloat16

    training_args = SFTConfig(
        output_dir=model_save_dir,
        run_name=f"rankvideo-perception-{run_ts}",
        report_to=["wandb"] if args.wandb else [],
        logging_steps=1,
        save_steps=100,
        save_total_limit=5,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        remove_unused_columns=False,
        packing=False,
        model_init_kwargs=model_init_kwargs,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    processor.tokenizer.truncation_side = "left"

    run_config = {
        "run": {"id": run_ts, "timestamp_utc": timestamp(), "output_dir": model_save_dir},
        "model": {"model_id": args.model, "cache_dir": args.cache_dir},
        "data": {
            "train_path": args.train_data,
            "eval_path": args.eval_data,
            "train_size": safe_serialize(len(train_dataset)),
            "eval_size": safe_serialize(len(eval_dataset)),
        },
        "stage": "perception",
    }
    
    with open(os.path.join(model_save_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        processing_class=processor,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    metrics = trainer.evaluate()
    try:
        metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
    except Exception:
        pass
    logging.info("metrics: %s", metrics)
    
    with open(os.path.join(model_save_dir, "eval_metrics.json"), "w") as f:
        json.dump({"timestamp": timestamp(), **metrics}, f, indent=2)
    
    trainer.save_model()
    logging.info("Model saved to %s", model_save_dir)


if __name__ == "__main__":
    main()
