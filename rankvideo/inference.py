"""
RANKVIDEO inference engines for video reranking.

Supports both HuggingFace Transformers and vLLM backends.
"""

import math
from typing import List, Dict, Any, Optional

from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False



SYSTEM_PROMPT = (
    "You are a helpful assistant specialized in video and text understanding. "
    + "Given a text query and a video, your task is to determine if the video is relevant to the query. "
    + "Respond with <answer>yes</answer> if the video is relevant, or <answer>no</answer> if it is not. "
)
 
USER_PROMPT = (
    "Query: {query}\n"
    + "Is the video relevant to the query? "
    + "Respond with <answer>yes</answer> or <answer>no</answer>."
)


DEFAULT_FPS = 2.0
DEFAULT_MAX_FRAMES = 32
MIN_TOKENS_PER_FRAME = 128
MAX_TOKENS_PER_FRAME = 128


class PromptBuilder:
    @staticmethod
    def build_messages(query: str, video_path: str, fps: float = DEFAULT_FPS, 
                       max_frames: int = DEFAULT_MAX_FRAMES) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(query=query)},
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": fps,
                        "max_frames": max_frames,
                        "min_pixels": MIN_TOKENS_PER_FRAME * 32 * 32,
                        "max_pixels": MAX_TOKENS_PER_FRAME * 32 * 32,
                        "total_pixels": max_frames * MAX_TOKENS_PER_FRAME * 32 * 32,
                    },
                ],
            }
        ]


def _logsumexp(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class VLMReranker:
    """
    Video-Language Model reranker using vLLM for efficient batched inference.
    
    Computes relevance scores as log-probability difference: log P(yes) - log P(no)
    """
    
    def __init__(
        self,
        model_path: str,
        cache_dir: Optional[str] = None,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for VLMReranker. Install with: pip install vllm")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        vocab_size = self.tokenizer.vocab_size
        
        self.llm = LLM(
            model=model_path,
            limit_mm_per_prompt={"video": 1},
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            enable_prefix_caching=True,
            max_logprobs=vocab_size
        )
        
        self.yes_ids = []
        self.no_ids = []
        for tid in range(self.tokenizer.vocab_size):
            s = self.tokenizer.decode([tid])
            norm = s.strip().lower()
            if norm == "yes":
                self.yes_ids.append(tid)
            elif norm == "no":
                self.no_ids.append(tid)

    def score_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        fps: float = DEFAULT_FPS,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> List[Dict[str, float]]:
        """
        Score query-video pairs for relevance.
        
        Returns list of dicts with:
            - logprob_yes: log probability of "yes"
            - logprob_no: log probability of "no"  
            - logit_delta: log P(yes) - log P(no), main ranking signal
            - p_yes: probability of relevance
        """
        assert len(queries) == len(video_paths)
        
        llm_inputs = []
        for query, video_path in zip(queries, video_paths):
            messages = PromptBuilder.build_messages(query, video_path, fps, max_frames)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            llm_inputs.append({
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            })

        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            logprobs=512,
            allowed_token_ids=self.yes_ids + self.no_ids,
        )

        outputs = self.llm.generate(llm_inputs, sampling_params=sampling_params)
        
        scores = []
        for out in outputs:
            comp = out.outputs[0]
            pos0 = comp.logprobs[0]
            
            yes_logs = [pos0[tid].logprob for tid in self.yes_ids if tid in pos0]
            no_logs = [pos0[tid].logprob for tid in self.no_ids if tid in pos0]
            
            logp_yes = _logsumexp(yes_logs)
            logp_no = _logsumexp(no_logs)

            if logp_yes is None or logp_no is None:
                scores.append({
                    "logprob_yes": 0.0,
                    "logprob_no": 0.0,
                    "logit_delta": 0.0,
                    "p_yes": 0.0,
                })
                continue
            
            delta = float(logp_yes - logp_no)
            p_yes = 1.0 / (1.0 + math.exp(-delta))
            
            scores.append({
                "logprob_yes": float(logp_yes),
                "logprob_no": float(logp_no),
                "logit_delta": delta,
                "p_yes": float(p_yes),
            })

        return scores

    def generate_batch(
        self,
        queries: List[str],
        video_paths: List[str],
        max_new_tokens: int = 512,
        fps: float = DEFAULT_FPS,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> List[str]:
        """Generate full responses (with reasoning) for query-video pairs."""
        llm_inputs = []
        for query, video_path in zip(queries, video_paths):
            messages = PromptBuilder.build_messages(query, video_path, fps, max_frames)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            llm_inputs.append({
                "prompt": text,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            })

        outputs = self.llm.generate(
            llm_inputs, 
            sampling_params=SamplingParams(max_tokens=max_new_tokens)
        )
        return [o.outputs[0].text for o in outputs]
