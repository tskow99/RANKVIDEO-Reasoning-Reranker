import os
import re
import math
import random
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_for_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def safe_serialize(obj):
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def maybe_to_dict(cfg):
    try:
        return cfg.to_dict()
    except Exception:
        try:
            return {k: safe_serialize(v) for k, v in cfg.__dict__.items()}
        except Exception:
            return safe_serialize(cfg)


def valid_example(ex: Dict[str, Any]) -> bool:
    if not isinstance(ex.get("messages"), list):
        return False
    roles = [m.get("role") for m in ex["messages"]]
    if "user" not in roles or "assistant" not in roles:
        return False
    u = [m for m in ex["messages"] if m["role"] == "user"][-1]
    for p in u.get("content", []):
        if p.get("type") == "video":
            v = p.get("video")
            ok = isinstance(v, str) and os.path.isfile(v)
            if not ok:
                logging.warning(
                    "[video-check] filter drop: invalid video. qid=%s did=%s v=%s",
                    ex.get('query_id'), ex.get('doc_id'), v,
                )
            return ok
    return False


def find_last_subsequence(haystack: List[int], needle: List[int]) -> int:
    last = -1
    N, M = len(haystack), len(needle)
    for i in range(0, N - M + 1):
        if haystack[i:i + M] == needle:
            last = i
    return last


def sanitize_multimodal_conversation(ex: Dict[str, Any], conv: List[Dict]) -> tuple:
    bad = False
    new_conv = []

    for msg in conv:
        content = msg.get("content")
        if isinstance(content, list):
            new_parts = []
            for ele in content:
                if not isinstance(ele, dict):
                    new_parts.append(ele)
                    continue

                ele = dict(ele)
                etype = ele.get("type")

                if etype == "video":
                    v = ele.get("video")
                    if isinstance(v, os.PathLike):
                        v = str(v)
                        ele["video"] = v
                    if v is None or not isinstance(v, (str, list, tuple)):
                        bad = True
                        continue
                else:
                    if "video" in ele and ele["video"] is None:
                        ele.pop("video", None)

                if etype == "audio":
                    a = ele.get("audio")
                    if isinstance(a, os.PathLike):
                        a = str(a)
                        ele["audio"] = a
                    if a is None or not isinstance(a, (str, list, tuple)):
                        bad = True
                        continue
                else:
                    if "audio" in ele and ele["audio"] is None:
                        ele.pop("audio", None)

                if "fps" in ele and ele["fps"] is None:
                    ele.pop("fps", None)
                if "max_frames" in ele and ele["max_frames"] is None:
                    ele.pop("max_frames", None)

                new_parts.append(ele)

            nm = dict(msg)
            nm["content"] = new_parts
            new_conv.append(nm)
        else:
            new_conv.append(msg)

    if bad:
        logging.warning(
            "[mm-check] dropped malformed media element qid=%s did=%s",
            ex.get("query_id"), ex.get("doc_id")
        )
    return new_conv, bad


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logsumexp(xs: List[float]):
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))
