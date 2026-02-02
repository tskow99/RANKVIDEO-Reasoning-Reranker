"""
Stage 2: Reranker training with pairwise ranking and teacher distillation.

Combines pointwise, pairwise, and teacher confidence distillation objectives.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Sampler
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
    "Given a text query and a video, determine if the video is relevant."
)

USER_PROMPT = "Query: {query}\nIs the video relevant? Answer with yes or no."


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: Reranker training")
    parser.add_argument("--model", default=config.STAGE1_CHECKPOINT,
                       help="Path to Stage 1 checkpoint or base model")
    parser.add_argument("--train-data", default=config.TRAIN_DATA_PATH)
    parser.add_argument("--eval-data", default=config.EVAL_DATA_PATH)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=config.CACHE_DIR)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--num-negs", type=int, default=1, help="Negatives per positive")
    parser.add_argument("--qlora", action="store_true", default=config.USE_QLORA)
    parser.add_argument("--no-qlora", action="store_false", dest="qlora")
    parser.add_argument("--wandb", action="store_true", default=config.LOG_TO_WANDB)
    parser.add_argument("--fps", type=float, default=config.FPS)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--tau-rank", type=float, default=config.TAU_RANK)
    parser.add_argument("--tau-distill", type=float, default=config.TAU_DISTILL)
    parser.add_argument("--tau-point", type=float, default=config.TAU_POINT)
    parser.add_argument("--lambda-distill", type=float, default=config.LAMBDA_DISTILL)
    parser.add_argument("--lambda-point", type=float, default=config.LAMBDA_POINT)
    return parser.parse_args()


def get_label(ex):
    y = ex.get("true_label", None)
    if y is None:
        y = ex.get("teacher_label", 0)
    return 1 if (y or 0) >= 1 else 0


def get_teacher_p_yes(ex):
    if "teacher_p_yes" in ex and ex["teacher_p_yes"] is not None:
        try:
            return float(ex["teacher_p_yes"])
        except Exception:
            return None
    meta = ex.get("meta") or {}
    if "teacher_p_yes" in meta and meta["teacher_p_yes"] is not None:
        try:
            return float(meta["teacher_p_yes"])
        except Exception:
            return None
    ex["teacher_p_yes"] = None
    return None


def build_messages(ex, fps, max_frames):
    query = ex.get("query", "")
    gold = get_label(ex)
    gold_txt = "yes" if gold == 1 else "no"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": USER_PROMPT.format(query=query)}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"<answer>{gold_txt}</answer>"}]},
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
    for k in ("query_id", "doc_id", "teacher_label", "true_label"):
        if k in ex:
            out[k] = ex[k]
        else:
            meta = ex.get("meta") or {}
            if k in meta:
                out[k] = meta[k]

    tp = get_teacher_p_yes(ex)
    if tp is not None:
        out["teacher_p_yes"] = float(tp)

    return out


def build_query_index(dataset):
    qpos = defaultdict(list)
    qneg = defaultdict(list)

    for idx in range(len(dataset)):
        ex = dataset[idx]
        qid = ex["query_id"]
        y = get_label(ex)
        (qpos[qid] if y == 1 else qneg[qid]).append(idx)

    qids = [qid for qid in qpos.keys() if len(qpos[qid]) > 0 and len(qneg.get(qid, [])) > 0]
    return qids, qpos, qneg


class QueryGroupedBatchSampler(Sampler):
    def __init__(self, qids, qpos, qneg, num_negs=2, seed=42, shuffle=True, drop_last=True):
        self.qids = list(qids)
        self.qpos = qpos
        self.qneg = qneg
        self.num_negs = int(num_negs)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rank = 0
        self.world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = int(epoch)

    def __len__(self):
        n = len(self.qids)
        if self.world_size > 1 and self.drop_last:
            n = (n // self.world_size) * self.world_size
        return n // self.world_size if self.world_size > 1 else n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)

        qids = self.qids
        if self.shuffle:
            perm = torch.randperm(len(qids), generator=g).tolist()
            qids = [qids[i] for i in perm]

        if self.world_size > 1:
            if self.drop_last:
                n_total = (len(qids) // self.world_size) * self.world_size
                qids = qids[:n_total]
            qids = qids[self.rank::self.world_size]

        for qid in qids:
            pos_list = self.qpos[qid]
            neg_list = self.qneg[qid]

            pos_idx = pos_list[int(torch.randint(0, len(pos_list), (1,), generator=g).item())]

            if len(neg_list) >= self.num_negs:
                neg_perm = torch.randperm(len(neg_list), generator=g).tolist()
                neg_idxs = [neg_list[j] for j in neg_perm[:self.num_negs]]
            else:
                neg_idxs = [
                    neg_list[int(torch.randint(0, len(neg_list), (1,), generator=g).item())]
                    for _ in range(self.num_negs)
                ]

            batch = [pos_idx] + neg_idxs
            if self.shuffle:
                bperm = torch.randperm(len(batch), generator=g).tolist()
                batch = [batch[j] for j in bperm]

            yield batch

        self._epoch += 1


class RerankerCollator:
    def __init__(self, processor, assistant_prefix_ids, fps, num_frames, enable_thinking=False):
        self.processor = processor
        self.assistant_prefix_ids = list(assistant_prefix_ids)
        self.fps = fps
        self.num_frames = num_frames
        self.enable_thinking = enable_thinking

        self.yes_token_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_token_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

        self._answer_open_alts = [
            processor.tokenizer.encode(p, add_special_tokens=False)
            for p in ["<answer>", " <answer>", "\n<answer>", "\n <answer>"]
        ]

    def _find_any_last(self, hay_ids, patterns):
        best = (-1, 0)
        for pat in patterns:
            if not pat:
                continue
            Lh, Lp = len(hay_ids), len(pat)
            j = -1
            for i in range(max(0, Lh - Lp), -1, -1):
                if hay_ids[i:i+Lp] == pat:
                    j = i
                    break
            if j != -1 and j > best[0]:
                best = (j, len(pat))
        return best

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

        qids = [ex["query_id"] for ex in batch]
        assert len(set(qids)) == 1, f"Batch has multiple queries: {set(qids)}"

        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False, enable_thinking=self.enable_thinking
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

        B, L = input_ids.size()
        binary_labels = torch.tensor([get_label(ex) for ex in batch], dtype=torch.float32)

        tp_list = []
        for ex in batch:
            tp = ex.get("teacher_p_yes", None)
            tp_list.append(float(tp) if tp is not None else -1.0)
        teacher_p_yes = torch.tensor(tp_list, dtype=torch.float32)

        answer_logit_pos = torch.full((B,), -1, dtype=torch.long)
        for i in range(B):
            ids = input_ids[i].tolist()
            pos_assist = find_last_subsequence(ids, self.assistant_prefix_ids)
            if pos_assist == -1:
                continue
            search_ids = ids[pos_assist:]
            o_pos_rel, o_len = self._find_any_last(search_ids, self._answer_open_alts)
            if o_pos_rel == -1:
                continue
            o_pos = pos_assist + o_pos_rel
            label_tok_pos = o_pos + o_len
            logit_pos = label_tok_pos - 1
            logit_pos = max(0, min(logit_pos, int(attn_mask[i].sum().item()) - 1))
            answer_logit_pos[i] = logit_pos

        enc["labels"] = labels
        enc["binary_labels"] = binary_labels
        enc["teacher_p_yes"] = teacher_p_yes
        enc["answer_logit_pos"] = answer_logit_pos
        return enc


class PairwiseRerankerTrainer(SFTTrainer):
    def __init__(self, *args, yes_token_id=None, no_token_id=None,
                 train_batch_sampler=None, eval_batch_sampler=None,
                 tau_rank=10.0, tau_distill=5.0, tau_point=1.0,
                 lambda_distill=5.0, lambda_point=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_token_id = int(yes_token_id)
        self.no_token_id = int(no_token_id)
        self._train_batch_sampler = train_batch_sampler
        self._eval_batch_sampler = eval_batch_sampler
        self.tau_rank = float(tau_rank)
        self.tau_distill = float(tau_distill)
        self.tau_point = float(tau_point)
        self.lambda_distill = float(lambda_distill)
        self.lambda_point = float(lambda_point)

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_sampler=self._eval_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        y = inputs.pop("binary_labels")
        teacher_p_yes = inputs.pop("teacher_p_yes")
        answer_pos = inputs.pop("answer_logit_pos")
        inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        device = logits.device
        y = y.to(device=device, dtype=torch.float32)
        teacher_p_yes = teacher_p_yes.to(device=device, dtype=torch.float32)
        answer_pos = answer_pos.to(device=device, dtype=torch.long)

        B = logits.size(0)
        bidx = torch.arange(B, device=device)

        s = logits[bidx, answer_pos, self.yes_token_id] - logits[bidx, answer_pos, self.no_token_id]

        pos_mask = y > 0.5
        neg_mask = ~pos_mask
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            loss = F.binary_cross_entropy_with_logits(s, y)
            return (loss, outputs) if return_outputs else loss

        log_probs = F.log_softmax(s / self.tau_rank, dim=0)
        loss_rank = -log_probs[pos_mask].mean()

        valid_tp = teacher_p_yes >= 0.0
        if valid_tp.any():
            tp = teacher_p_yes.clamp(0.0, 1.0)
            loss_distill = F.binary_cross_entropy_with_logits(s / self.tau_distill, tp, reduction="none")
            loss_distill = loss_distill[valid_tp].mean()
        else:
            loss_distill = torch.zeros((), device=device)

        t_soft = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, 0.10))
        w = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, 0.5))
        loss_point = F.binary_cross_entropy_with_logits(s / self.tau_point, t_soft, weight=w)

        loss = loss_rank + self.lambda_distill * loss_distill + self.lambda_point * loss_point

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    if args.wandb:
        import wandb
        if config.WANDB_API_KEY:
            wandb.login(key=config.WANDB_API_KEY, relogin=True)

    run_ts = timestamp_for_name()
    model_save_dir = os.path.join(args.output_dir, f"reranker-{run_ts}")
    os.makedirs(model_save_dir, exist_ok=True)

    set_seed(args.seed)
    os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)
    os.environ.setdefault("WANDB_MODE", "online" if args.wandb else "disabled")

    processor = AutoProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    raw_train = load_dataset("json", data_files=args.train_data, split="train")
    raw_eval = load_dataset("json", data_files=args.eval_data, split="train")

    KEEP_COLS = {"messages", "query_id", "doc_id", "teacher_label", "true_label", "teacher_p_yes"}

    mapped_train = raw_train.map(
        lambda ex: build_messages(ex, args.fps, args.max_frames),
        remove_columns=[c for c in raw_train.column_names if c not in KEEP_COLS],
    )
    mapped_eval = raw_eval.map(
        lambda ex: build_messages(ex, args.fps, args.max_frames),
        remove_columns=[c for c in raw_eval.column_names if c not in KEEP_COLS],
    )

    train_dataset = mapped_train.filter(valid_example)
    eval_dataset = mapped_eval.filter(valid_example)

    train_qids, train_qpos, train_qneg = build_query_index(train_dataset)
    eval_qids, eval_qpos, eval_qneg = build_query_index(eval_dataset)

    logging.info("[groups] train queries usable=%d", len(train_qids))
    logging.info("[groups] eval queries usable=%d", len(eval_qids))

    train_batch_sampler = QueryGroupedBatchSampler(
        train_qids, train_qpos, train_qneg,
        num_negs=args.num_negs, seed=args.seed, shuffle=True, drop_last=True
    )
    eval_batch_sampler = QueryGroupedBatchSampler(
        eval_qids, eval_qpos, eval_qneg,
        num_negs=args.num_negs, seed=args.seed, shuffle=False, drop_last=False
    )

    assistant_prefix = "<|im_start|>assistant\n"
    assistant_prefix_ids = processor.tokenizer.encode(assistant_prefix, add_special_tokens=False)

    data_collator = RerankerCollator(
        processor, assistant_prefix_ids, args.fps, config.NFRAMES
    )

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
        run_name=f"rankvideo-reranker-{run_ts}",
        report_to=["wandb"] if args.wandb else [],
        logging_steps=1,
        save_steps=200,
        save_total_limit=4,
        eval_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
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
        dataloader_persistent_workers=True,
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
        "loss": {
            "tau_rank": args.tau_rank,
            "tau_distill": args.tau_distill,
            "tau_point": args.tau_point,
            "lambda_distill": args.lambda_distill,
            "lambda_point": args.lambda_point,
        },
        "stage": "reranker",
    }

    with open(os.path.join(model_save_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    trainer = PairwiseRerankerTrainer(
        model=args.model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        yes_token_id=data_collator.yes_token_id,
        no_token_id=data_collator.no_token_id,
        train_batch_sampler=train_batch_sampler,
        eval_batch_sampler=eval_batch_sampler,
        tau_rank=args.tau_rank,
        tau_distill=args.tau_distill,
        tau_point=args.tau_point,
        lambda_distill=args.lambda_distill,
        lambda_point=args.lambda_point,
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
