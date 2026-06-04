"""
prepare.py — Read-only evaluation, data loading, and oracle annotator.

Autoresearch equivalent of Karpathy's prepare.py.
DO NOT MODIFY THIS FILE — the experiment loop must not touch it.

Provides:
  - OracleAnnotator: uses ground truth labels (upper bound)
  - LocalLLMAnnotator: uses a local HuggingFace model (e.g., Qwen2.5-7B)
  - evaluate_experiment(): standardized evaluation matching LoCLE Table 2
  - LOCLE_BASELINES: target numbers to beat
"""

import json
import logging
import os
import os.path as osp
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Ensure json is available for LocalLLMCacheAnnotator
assert json  # suppress unused-import lint

# ══════════════════════════════════════════════════════════════
# LoCLE Baselines (from Table 2 — our targets to beat)
# ══════════════════════════════════════════════════════════════

LOCLE_BASELINES = {
    "cora": {
        "gcn": {"acc": 81.80, "f1": 79.64, "nmi": 64.22, "ari": 64.30},
        "gat": {"acc": 76.06, "f1": 73.24, "nmi": 56.75, "ari": 55.78},
    },
    "citeseer": {
        "gcn": {"acc": 76.16, "f1": 65.28, "nmi": 50.87, "ari": 54.14},
        "gat": {"acc": 73.97, "f1": 66.26, "nmi": 47.38, "ari": 50.27},
    },
    "pubmed": {
        "gcn": {"acc": 83.70, "f1": 83.26, "nmi": 48.86, "ari": 56.56},
        "gat": {"acc": 82.48, "f1": 81.94, "nmi": 46.08, "ari": 53.94},
    },
    "wikics": {
        "gcn": {"acc": 75.23, "f1": 69.54, "nmi": 55.18, "ari": 58.98},
        "gat": {"acc": 73.23, "f1": 67.80, "nmi": 54.22, "ari": 54.83},
    },
}


def get_locle_baseline(dataset: str, backbone: str) -> dict:
    """Get LoCLE baseline for a dataset/backbone pair."""
    ds = dataset.lower()
    bb = backbone.lower()
    return LOCLE_BASELINES.get(ds, {}).get(bb, {})


def compute_improvement(result: dict, dataset: str, backbone: str) -> dict:
    """Compute improvement over LoCLE baseline."""
    baseline = get_locle_baseline(dataset, backbone)
    if not baseline:
        return {}
    return {
        "acc_delta": result.get("accuracy", 0) * 100 - baseline.get("acc", 0),
        "f1_delta": result.get("macro_f1", 0) * 100 - baseline.get("f1", 0),
        "nmi_delta": result.get("nmi", 0) * 100 - baseline.get("nmi", 0),
        "ari_delta": result.get("ari", 0) * 100 - baseline.get("ari", 0),
    }


# ══════════════════════════════════════════════════════════════
# Oracle Annotator — uses ground truth labels
# ══════════════════════════════════════════════════════════════

class OracleAnnotator:
    """
    Annotator that uses ground truth labels with configurable noise.

    For oracle experiments (noise_rate=0.0), this gives an upper bound
    on what the pipeline can achieve with perfect annotations.
    For noisy oracle experiments, simulates realistic LLM error patterns.
    """

    def __init__(
        self,
        ground_truth_y: torch.Tensor,
        class_names: List[str],
        noise_rate: float = 0.0,
        noise_matrix: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.ground_truth_y = ground_truth_y
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.noise_rate = noise_rate
        self.noise_matrix = noise_matrix
        self.model = f"oracle-noise{noise_rate:.2f}"
        self.total_api_calls = 0
        self._rng = np.random.RandomState(seed)
        self._cache: Dict[int, Tuple[int, float]] = {}

    def annotate_nodes(self, node_ids, texts, class_names=None, force=False):
        results = {}
        for nid, text in zip(node_ids, texts):
            if not force and nid in self._cache:
                results[nid] = self._cache[nid]
                self.total_api_calls += 1
                continue

            true_label = int(self.ground_truth_y[nid].item())

            if self.noise_rate > 0:
                if self.noise_matrix is not None:
                    # Class-conditional noise from provided matrix
                    pred = self._rng.choice(
                        self.num_classes,
                        p=self.noise_matrix[true_label],
                    )
                else:
                    # Uniform random noise
                    if self._rng.random() < self.noise_rate:
                        pred = self._rng.randint(0, self.num_classes)
                    else:
                        pred = true_label
            else:
                pred = true_label

            conf = 95.0 if pred == true_label else float(self._rng.uniform(50, 80))
            results[nid] = (int(pred), conf)
            self._cache[nid] = (int(pred), conf)
            self.total_api_calls += 1

        return results

    def get_cached(self, node_ids):
        return {nid: self._cache[nid] for nid in node_ids if nid in self._cache}

    def get_all_cached(self):
        return dict(self._cache)


# ══════════════════════════════════════════════════════════════
# Local LLM Annotator — uses HuggingFace model for zero-shot classification
# ══════════════════════════════════════════════════════════════

class LocalLLMCacheAnnotator:
    """
    Annotator that reads from pre-generated local LLM annotation caches.

    Use annotate_all.py to generate caches first, then this annotator
    serves annotations instantly from disk — no model loading needed.

    Cache format: {str(node_id): [label_int, confidence_float], ...}
    File pattern: data/annotations/{dataset}_local_{model_name}.json
    """

    def __init__(
        self,
        dataset_name: str,
        class_names: List[str],
        model_name: str,
        cache_dir: str = "data/annotations/",
        seed: int = 42,
    ):
        self.dataset_name = dataset_name.lower()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = f"local-cache-{model_name}"
        self.total_api_calls = 0
        self._rng = np.random.RandomState(seed)
        self._cache: Dict[int, Tuple[int, float]] = {}

        # Load the pre-generated cache
        safe_model = model_name.replace("/", "_").replace("\\", "_")
        cache_path = osp.join(cache_dir, f"{self.dataset_name}_local_{safe_model}.json")

        if osp.exists(cache_path):
            with open(cache_path, "r") as f:
                raw = json.load(f)
            for k, (label, conf) in raw.items():
                self._cache[int(k)] = (int(label), float(conf))
            logger.info("LocalLLMCacheAnnotator: loaded %d annotations from %s",
                       len(self._cache), cache_path)
        else:
            raise FileNotFoundError(
                f"Annotation cache not found: {cache_path}\n"
                f"Run annotate_all.py first to generate annotations:\n"
                f"  python autoresearch/annotate_all.py --model models/{model_name} "
                f"--datasets {dataset_name}"
            )

    def annotate_nodes(self, node_ids, texts, class_names=None, force=False):
        results = {}
        missing = 0
        for nid, text in zip(node_ids, texts):
            if nid in self._cache:
                results[nid] = self._cache[nid]
            else:
                # Fallback for nodes not in cache (shouldn't happen if annotate_all ran fully)
                missing += 1
                pred = self._rng.randint(0, self.num_classes)
                conf = 10.0  # Low confidence for random fallback
                results[nid] = (int(pred), conf)
                self._cache[nid] = (int(pred), conf)
            self.total_api_calls += 1

        if missing > 0:
            logger.warning("LocalLLMCacheAnnotator: %d/%d nodes not in cache (random fallback)",
                         missing, len(node_ids))
        return results

    def get_cached(self, node_ids):
        return {nid: self._cache[nid] for nid in node_ids if nid in self._cache}

    def get_all_cached(self):
        return dict(self._cache)


class LocalLLMAnnotator:
    """
    Annotator using a local HuggingFace model for zero-shot node classification.

    Designed for models that fit on a single A100 40GB:
      - Qwen2.5-7B-Instruct (~14GB fp16)
      - Llama-3.1-8B-Instruct (~16GB fp16)
      - Qwen2.5-14B-Instruct-AWQ (~8GB 4-bit)
      - Qwen2.5-32B-Instruct-AWQ (~18GB 4-bit)

    Pre-download on login node (has internet):
      huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /path/to/models/qwen2.5-7b
    """

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        dataset_name: str = "cora",
        device: str = "cuda",
        max_new_tokens: int = 64,
        batch_size: int = 8,
        quantize_4bit: bool = False,
        cache_dir: str = "data/annotations/",
        seed: int = 42,
    ):
        self.model_path = model_path
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.dataset_name = dataset_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.quantize_4bit = quantize_4bit
        self.cache_dir = cache_dir
        self.model_name = osp.basename(model_path)
        self.model = f"local-{self.model_name}"
        self.total_api_calls = 0
        self._rng = np.random.RandomState(seed)
        self._cache: Dict[int, Tuple[int, float]] = {}
        self._model = None
        self._tokenizer = None

        # Load disk cache if exists
        self._cache_path = osp.join(
            cache_dir,
            f"{dataset_name}_{self.model_name.replace('/', '_')}.json",
        )
        self._load_disk_cache()

    def _load_disk_cache(self):
        if osp.exists(self._cache_path):
            with open(self._cache_path, "r") as f:
                raw = json.load(f)
            for k, (label, conf) in raw.items():
                self._cache[int(k)] = (int(label), float(conf))
            logger.info("Loaded %d cached annotations from %s", len(self._cache), self._cache_path)

    def _save_disk_cache(self):
        os.makedirs(osp.dirname(self._cache_path), exist_ok=True)
        serializable = {str(k): [v[0], v[1]] for k, v in self._cache.items()}
        with open(self._cache_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local LLM: %s (4bit=%s)", self.model_path, self.quantize_4bit)
        load_start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if self.quantize_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        logger.info("Model loaded in %.1fs", time.time() - load_start)

    def _build_prompt(self, text: str) -> str:
        class_list = ", ".join(self.class_names)
        return (
            f"Classify the following text into exactly one category.\n"
            f"Categories: {class_list}\n\n"
            f"Text: {text}\n\n"
            f'Reply with ONLY a JSON object: {{"label": "<category>", "confidence": <0-100>}}'
        )

    def _parse_response(self, response: str) -> Tuple[int, float]:
        """Parse model response to (label_index, confidence)."""
        import re

        # Try JSON parsing
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            try:
                obj = json.loads(json_match.group())
                label_str = obj.get("label", "")
                confidence = float(obj.get("confidence", 50.0))
                names_lower = [c.lower() for c in self.class_names]
                if label_str.lower() in names_lower:
                    return names_lower.index(label_str.lower()), min(max(confidence, 0), 100)
                # Fuzzy match
                from difflib import get_close_matches
                matches = get_close_matches(label_str.lower(), names_lower, n=1, cutoff=0.4)
                if matches:
                    return names_lower.index(matches[0]), min(max(confidence / 2, 0), 100)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: find class name in text
        names_lower = [c.lower() for c in self.class_names]
        response_lower = response.lower()
        for i, name in enumerate(names_lower):
            if name in response_lower or name.replace("_", " ") in response_lower:
                return i, 50.0

        return -1, 0.0

    def annotate_nodes(self, node_ids, texts, class_names=None, force=False):
        results = {}
        to_query = []

        for nid, text in zip(node_ids, texts):
            if not force and nid in self._cache:
                results[nid] = self._cache[nid]
                self.total_api_calls += 1
            else:
                to_query.append((nid, text))

        if not to_query:
            return results

        self._ensure_model()

        # Process in batches
        for batch_start in range(0, len(to_query), self.batch_size):
            batch = to_query[batch_start:batch_start + self.batch_size]

            for nid, text in batch:
                prompt = self._build_prompt(text[:2000])  # truncate long texts

                messages = [
                    {"role": "system", "content": "You are a precise text classifier. Always respond with valid JSON."},
                    {"role": "user", "content": prompt},
                ]

                # Use chat template if available
                if hasattr(self._tokenizer, "apply_chat_template"):
                    input_text = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                else:
                    input_text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

                inputs = self._tokenizer(
                    input_text, return_tensors="pt", truncation=True, max_length=2048,
                ).to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )

                label_idx, confidence = self._parse_response(response)
                if label_idx == -1:
                    label_idx = self._rng.randint(0, self.num_classes)
                    confidence = 0.0

                results[nid] = (int(label_idx), confidence)
                self._cache[nid] = (int(label_idx), confidence)
                self.total_api_calls += 1

        self._save_disk_cache()
        return results

    def get_cached(self, node_ids):
        return {nid: self._cache[nid] for nid in node_ids if nid in self._cache}

    def get_all_cached(self):
        return dict(self._cache)


# ══════════════════════════════════════════════════════════════
# Evaluation (matches LoCLE Table 2 protocol)
# ══════════════════════════════════════════════════════════════

def evaluate_experiment(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    labeled_mask: Optional[torch.Tensor] = None,
) -> dict:
    """
    Evaluate predictions matching LoCLE Table 2 metrics.

    Returns dict with accuracy, macro_f1, nmi, ari (all as fractions 0-1).
    """
    from sklearn.metrics import (
        accuracy_score, f1_score,
        normalized_mutual_info_score, adjusted_rand_score,
    )

    if predictions.dim() == 2:
        preds = predictions.argmax(dim=1)
    else:
        preds = predictions

    preds_np = preds.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()

    if labeled_mask is not None:
        eval_mask = ~labeled_mask.cpu()
        preds_np = preds_np[eval_mask.numpy()]
        gt_np = gt_np[eval_mask.numpy()]

    if len(preds_np) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "nmi": 0.0, "ari": 0.0}

    return {
        "accuracy": accuracy_score(gt_np, preds_np),
        "macro_f1": f1_score(gt_np, preds_np, average="macro", zero_division=0),
        "nmi": normalized_mutual_info_score(gt_np, preds_np),
        "ari": adjusted_rand_score(gt_np, preds_np),
        "num_evaluated": len(preds_np),
    }
