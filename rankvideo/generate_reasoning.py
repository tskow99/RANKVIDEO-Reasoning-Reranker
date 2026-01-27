#!/usr/bin/env python3
"""
Generate teacher reasoning labels for RANKVIDEO training.

Uses a large reasoning model to produce:
- Reasoning traces (rationales)
- Relevance labels
- Soft probability scores for distillation
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    import toml
except ImportError:
    toml = None

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def format_time(seconds: float) -> str:
    if seconds is None or seconds <= 0 or not (seconds == seconds):
        return "0s"
    total = int(max(1, round(seconds)))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def load_queries(path: str) -> Dict[str, str]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, text = line.rstrip("\n").split("\t", maxsplit=1)
            out[qid] = text
    return out


def load_candidates(path: str) -> Dict[str, List[Tuple[str, int, float]]]:
    """Load TREC run file: qid Q0 docid rank score tag"""
    runs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _q0, docid, rank, score, _runid = parts[:6]
            try:
                runs.setdefault(qid, []).append((docid, int(rank), float(score)))
            except Exception:
                continue
    for qid in runs:
        runs[qid].sort(key=lambda x: x[1])
    return runs


def load_jsonl_texts(path: Optional[str]) -> Dict[str, str]:
    """Load JSONL: {doc_id, text}"""
    texts = {}
    if not path:
        return texts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            doc_id = obj.get("doc_id")
            text = obj.get("text", "")
            if isinstance(doc_id, str):
                texts[doc_id] = text if isinstance(text, str) else str(text)
    return texts


class TokenCounter:
    def __init__(self, tokenizer_name: Optional[str] = None):
        self._hf_tok = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self._hf_tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            except Exception:
                pass

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        if self._hf_tok is not None:
            ids = self._hf_tok.encode(text, add_special_tokens=False)[:max_tokens]
            try:
                return self._hf_tok.decode(ids, skip_special_tokens=True)
            except Exception:
                return " ".join(text.split()[:max_tokens])
        return " ".join(text.split()[:max_tokens])


def build_evidence_text(caption: str, asr: str) -> str:
    parts = []
    if caption:
        parts.append(f"[Caption] {caption}")
    if asr:
        parts.append(f"[ASR] {asr}")
    return "\n".join(parts)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def find_last_subseq(hay: List[int], needle: List[int]) -> int:
    if not needle or not hay:
        return -1
    Lh, Ln = len(hay), len(needle)
    if Ln > Lh:
        return -1
    for i in range(Lh - Ln, -1, -1):
        if hay[i:i+Ln] == needle:
            return i
    return -1


SYSTEM_PROMPT = (
    "You are reviewing information that represents the visual content of a video. "
    "Your task is to judge whether the video is relevant to the query."
)

USER_PROMPT_TEMPLATE = (
    "Query: {query}\n\n"
    "Video Content:\n{passage}\n\n"
    "Provide a brief, visually-grounded rationale explaining the evidence "
    "for your decision. Write this reasoning between <think> and </think>.\n\n"
    "Then output exactly one line - 'yes' or 'no' - between <answer> and </answer>."
)


class ReasoningTeacher:
    """Generate reasoning traces and labels using a large LLM."""
    
    def __init__(
        self,
        model_path: str,
        num_gpus: int = 1,
        context_size: int = 8192,
        reasoning_maxlen: int = 512,
        gpu_memory_utilization: float = 0.9,
        cache_dir: Optional[str] = None,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM required. Install: pip install vllm")
        
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True
        )
        
        yes_seq = self._tokenizer.encode(" yes", add_special_tokens=False)
        no_seq = self._tokenizer.encode(" no", add_special_tokens=False)
        if not yes_seq or not no_seq:
            yes_seq = self._tokenizer.encode("yes", add_special_tokens=False)
            no_seq = self._tokenizer.encode("no", add_special_tokens=False)
        
        self._yes_token_id = int(yes_seq[-1])
        self._no_token_id = int(no_seq[-1])
        self._reasoning_maxlen = reasoning_maxlen
        
        self._answer_open_alts = [
            self._tokenizer.encode(p, add_special_tokens=False)
            for p in ["<answer>", " <answer>", "\n<answer>"]
        ]
        
        self._engine = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            trust_remote_code=True,
            max_model_len=context_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def _build_prompt(self, query: str, passage: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(query=query, passage=passage)},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt if isinstance(prompt, str) else str(prompt)

    def _extract_label(self, text: str) -> int:
        import re
        if not text:
            return 0
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if m:
            ans = m.group(1).strip().lower()
            if "no" in ans:
                return 0
            if "yes" in ans:
                return 1
        tail = text[-200:].lower()
        if "yes" in tail and "no" not in tail:
            return 1
        return 0

    def _find_any_last(self, hay: List[int], patterns: List[List[int]]) -> Tuple[int, int]:
        best_pos, best_len = -1, 0
        for pat in patterns:
            if not pat:
                continue
            j = find_last_subseq(hay, pat)
            if j > best_pos:
                best_pos, best_len = j, len(pat)
        return best_pos, best_len

    def _extract_logprob_value(self, v) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (float, int)):
            return float(v)
        for attr in ("logprob", "log_prob"):
            if hasattr(v, attr):
                try:
                    return float(getattr(v, attr))
                except Exception:
                    pass
        if isinstance(v, dict):
            for k in ("logprob", "log_prob"):
                if k in v:
                    try:
                        return float(v[k])
                    except Exception:
                        pass
        return None

    def _delta_from_logprobs(self, token_ids: List[int], logprobs) -> Optional[float]:
        try:
            o_pos, o_len = self._find_any_last(token_ids, self._answer_open_alts)
            if o_pos < 0:
                return None
            label_pos = o_pos + o_len
            if label_pos < 0 or not isinstance(logprobs, list) or label_pos >= len(logprobs):
                return None
            step = logprobs[label_pos]
            if step is None:
                return None
            lp_yes = self._extract_logprob_value(step.get(self._yes_token_id))
            lp_no = self._extract_logprob_value(step.get(self._no_token_id))
            if lp_yes is None or lp_no is None:
                return None
            return float(lp_yes) - float(lp_no)
        except Exception:
            return None

    def generate_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Generate reasoning for (query, passage) pairs."""
        if not pairs:
            return []
        
        prompts = [self._build_prompt(q, p) for q, p in pairs]
        params = SamplingParams(
            max_tokens=self._reasoning_maxlen,
            temperature=0.0,
            top_p=1.0,
            n=1,
            logprobs=10,
            stop=["</answer>"],
        )
        
        outs = self._engine.generate(prompts, params)
        results = []
        
        for req in outs:
            text = ""
            token_ids = None
            logprobs = None
            if req.outputs:
                out0 = req.outputs[0]
                text = getattr(out0, "text", "") or ""
                token_ids = getattr(out0, "token_ids", None)
                logprobs = getattr(out0, "logprobs", None)
            
            label = self._extract_label(text)
            delta = None
            if isinstance(token_ids, list) and logprobs is not None:
                delta = self._delta_from_logprobs([int(t) for t in token_ids], logprobs)
            
            p_yes = sigmoid(delta) if delta is not None else None
            results.append({
                "reasoning": text.strip(),
                "label": label,
                "p_yes": p_yes,
                "logit_delta": delta,
            })
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(description="Generate teacher reasoning labels")
    parser.add_argument("--queries", required=True, help="queries.tsv")
    parser.add_argument("--candidates", required=True, help="candidates.trec")
    parser.add_argument("--captions-jsonl", help="JSONL with captions: {doc_id, text}")
    parser.add_argument("--asr-jsonl", help="JSONL with ASR: {doc_id, text}")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--caption-tokens", type=int, default=512)
    parser.add_argument("--asr-tokens", type=int, default=0)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--context-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reasoning-maxlen", type=int, default=512)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    parser.add_argument("--cache-dir", default=config.CACHE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(args.seed)

    queries = load_queries(args.queries)
    candidates = load_candidates(args.candidates)
    captions = load_jsonl_texts(args.captions_jsonl)
    asr_texts = load_jsonl_texts(args.asr_jsonl)

    tok = TokenCounter(tokenizer_name=args.tokenizer)

    teacher = ReasoningTeacher(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        context_size=args.context_size,
        reasoning_maxlen=args.reasoning_maxlen,
        gpu_memory_utilization=args.gpu_memory,
        cache_dir=args.cache_dir,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_pairs = 0
    written = 0
    start = time.perf_counter()

    with open(args.output, "w", encoding="utf-8") as out:
        for qid, ranked in candidates.items():
            if qid not in queries:
                continue
            qtext = queries[qid]
            focus = ranked[:args.topk] if args.topk > 0 else ranked
            
            batch_items = []
            batch_pairs = []
            
            for docid, rank, score in focus:
                caption = captions.get(docid, "")
                asr = asr_texts.get(docid, "")
                cap_trunc = tok.truncate(caption, args.caption_tokens)
                asr_trunc = tok.truncate(asr, args.asr_tokens)
                passage = build_evidence_text(cap_trunc, asr_trunc)
                
                batch_items.append((qid, docid, qtext, rank, score, cap_trunc, asr_trunc))
                batch_pairs.append((qtext, passage))
                
                if len(batch_pairs) >= args.batch_size:
                    results = teacher.generate_batch(batch_pairs)
                    for item, res in zip(batch_items, results):
                        qid_i, docid_i, qtext_i, rank_i, score_i, cap_i, asr_i = item
                        record = {
                            "query_id": qid_i,
                            "doc_id": docid_i,
                            "query": qtext_i,
                            "evidence": {"caption": cap_i, "asr": asr_i},
                            "teacher_reasoning": res["reasoning"],
                            "teacher_label": res["label"],
                            "teacher_p_yes": res["p_yes"],
                            "teacher_logit_delta": res["logit_delta"],
                            "meta": {"rank": rank_i, "score": score_i},
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                    total_pairs += len(batch_pairs)
                    batch_items = []
                    batch_pairs = []
            
            if batch_pairs:
                results = teacher.generate_batch(batch_pairs)
                for item, res in zip(batch_items, results):
                    qid_i, docid_i, qtext_i, rank_i, score_i, cap_i, asr_i = item
                    record = {
                        "query_id": qid_i,
                        "doc_id": docid_i,
                        "query": qtext_i,
                        "evidence": {"caption": cap_i, "asr": asr_i},
                        "teacher_reasoning": res["reasoning"],
                        "teacher_label": res["label"],
                        "teacher_p_yes": res["p_yes"],
                        "teacher_logit_delta": res["logit_delta"],
                        "meta": {"rank": rank_i, "score": score_i},
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                total_pairs += len(batch_pairs)

    elapsed = time.perf_counter() - start
    logging.info("Done: written=%d pairs=%d elapsed=%s", written, total_pairs, format_time(elapsed))


if __name__ == "__main__":
    main()
