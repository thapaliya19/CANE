"""
Entry point for the NoisePref pipeline.

Supports Hydra-style key=value CLI arguments:
    python -m src.training.run experiment=full_pipeline dataset=cora seed=0 gnn=gcn max_rounds=3

Also supports standard argparse flags:
    python -m src.training.run --dataset cora --seed 0 --gnn gcn --max_rounds 3
"""

import argparse
import logging
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import yaml

# Ensure repo root is on path
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data_loading.tag_dataset import load_dataset
from src.evaluation.evaluator import evaluate_labels, evaluate_predictions
from src.llm_annotation.annotator import LLMAnnotator
from src.training.noisepref_pipeline import NoisePrefPipeline


logger = logging.getLogger(__name__)


# ── Default config values ─────────────────────────────────────

DEFAULTS = {
    "config": "",  # path to YAML config file (e.g. config/experiment/full_pipeline.yaml)
    "experiment": "full_pipeline",
    "dataset": "cora",
    "seed": 0,
    "gnn": "gcn",
    "max_rounds": 5,
    "budget_per_class": 20,
    "probing_fraction": 0.1,
    "initial_fraction": 0.5,
    "gnn_hidden_dim": 64,
    "gnn_epochs": 200,
    "gnn_lr": 0.01,
    "convergence_threshold": 0.01,
    "use_local_llm": False,
    "use_orpo": True,
    "local_llm_model": "meta-llama/Llama-3.2-1B",
    "lora_rank": 16,
    "warmup_rounds": 2,
    "use_noise_weighting": True,
    "use_rewiring": False,
    "rewiring_ratio": 0.05,
    "use_locle_cache": True,
    "data_path": "",
    "cache_dir": "",
    "device": "",
    "use_active_learning": False,
    "active_rounds": 5,
    "active_budget_per_round": 50,
    "log_level": "INFO",
    # Post-verification additions
    "use_bootstrap": False,
    "bootstrap_beta": 0.8,
    "use_elr": False,
    "elr_beta": 0.7,
    "elr_lambda": 3.0,
    "use_coteaching": False,
    "coteaching_frac": 0.3,
    "use_loss_pruning": False,
    "use_ilc": False,
    "use_ensemble": False,
    # NAAP toggles
    "use_naap": False,
    "naap_alpha_base": 0.3,
    "naap_beta": 0.4,
    "naap_gamma": 0.5,
    "naap_k_max": 30,
    # Self-label multiplier
    "self_label_multiplier": 3,
    # LoCLE verbatim rewire (Rewire_GNN + EstimateAdj)
    "use_locle_rewire": False,
}

# Type casting map
_TYPES = {
    "seed": int,
    "max_rounds": int,
    "budget_per_class": int,
    "probing_fraction": float,
    "initial_fraction": float,
    "gnn_hidden_dim": int,
    "gnn_epochs": int,
    "gnn_lr": float,
    "convergence_threshold": float,
    "use_local_llm": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "use_orpo": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "lora_rank": int,
    "warmup_rounds": int,
    "use_noise_weighting": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "use_rewiring": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "rewiring_ratio": float,
    "use_locle_cache": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "use_active_learning": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "active_rounds": int,
    "active_budget_per_round": int,
    "use_bootstrap": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "bootstrap_beta": float,
    "use_elr": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "elr_beta": float,
    "elr_lambda": float,
    "use_coteaching": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "coteaching_frac": float,
    "use_loss_pruning": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "use_ilc": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "use_ensemble": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "use_naap": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
    "naap_alpha_base": float,
    "naap_beta": float,
    "naap_gamma": float,
    "naap_k_max": int,
    "self_label_multiplier": int,
    "use_locle_rewire": lambda x: x.lower() in ("true", "1", "yes") if isinstance(x, str) else bool(x),
}


def parse_args():
    """Parse CLI arguments — supports both key=value and --key value styles."""
    parser = argparse.ArgumentParser(
        description="NoisePref pipeline runner",
        # Allow unrecognized args (Hydra-style overrides)
        allow_abbrev=False,
    )

    # Standard argparse flags
    for key, default in DEFAULTS.items():
        flag = f"--{key}"
        if isinstance(default, bool):
            parser.add_argument(flag, type=str, default=None)
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=None)
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=None)
        else:
            parser.add_argument(flag, type=str, default=None)

    args, remaining = parser.parse_known_args()

    # Start with defaults
    config = dict(DEFAULTS)

    # Collect CLI overrides (both argparse and key=value) BEFORE merging YAML,
    # so CLI always wins: defaults < default.yaml < experiment.yaml < CLI
    cli_overrides = {}

    # Argparse flags
    for key in DEFAULTS:
        val = getattr(args, key, None)
        if val is not None:
            cli_overrides[key] = val

    # Hydra-style key=value overrides
    for arg in remaining:
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-")
            if key in _TYPES:
                cli_overrides[key] = _TYPES[key](val)
            elif key in DEFAULTS:
                cli_overrides[key] = val
            else:
                cli_overrides[key] = val
        else:
            logger.warning("Ignoring unrecognized argument: %s", arg)

    # ── Layer 1: Load default.yaml (if it exists) ──
    default_yaml = osp.join(_REPO_ROOT, "config", "default.yaml")
    if osp.exists(default_yaml):
        with open(default_yaml, "r") as f:
            yaml_defaults = yaml.safe_load(f) or {}
        for key, val in yaml_defaults.items():
            if key in DEFAULTS:
                config[key] = val

    # ── Layer 2: Load experiment config YAML (if specified) ──
    # Check CLI first, then current config value
    config_path = cli_overrides.get("config", config.get("config", ""))
    if config_path:
        # Support both absolute and relative paths
        if not osp.isabs(config_path):
            config_path = osp.join(_REPO_ROOT, config_path)
        if osp.exists(config_path):
            with open(config_path, "r") as f:
                experiment_cfg = yaml.safe_load(f) or {}
            for key, val in experiment_cfg.items():
                if key in DEFAULTS or key in _TYPES:
                    config[key] = val
        else:
            logger.warning("Config file not found: %s", config_path)

    # ── Layer 3: CLI overrides always win ──
    config.update(cli_overrides)

    # Apply type casting for any string values
    for key, caster in _TYPES.items():
        if key in config and isinstance(config[key], str):
            try:
                config[key] = caster(config[key])
            except (ValueError, TypeError):
                pass

    return config


def resolve_paths(config):
    """Resolve data_path and cache_dir relative to repo root."""
    data_path = config.get("data_path", "")
    if not data_path:
        data_path = osp.join(_REPO_ROOT, "data")
        if not osp.exists(osp.join(data_path, "cora_random_sbert.pt")):
            alt = osp.join(_REPO_ROOT, "baselines", "locle", "data")
            if osp.exists(alt):
                data_path = alt
    config["data_path"] = data_path

    cache_dir = config.get("cache_dir", "")
    if not cache_dir:
        cache_dir = osp.join(data_path, "annotations")
    config["cache_dir"] = cache_dir

    return config


class CachedAnnotator:
    """
    Annotator that uses pre-cached LoCLE annotations instead of live API calls.

    For testing and reproducibility: loads the LoCLE annotation cache and
    serves annotations from it. Nodes not in cache get a fallback random
    annotation (simulating a noisy LLM).
    """

    def __init__(self, dataset_name, class_names, cache_dir, seed=42,
                 cache_filename=None):
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = "locle-cache"
        self.total_api_calls = 0
        self._rng = np.random.RandomState(seed)
        self._cache = {}
        # Optional override of the cached-annotation filename (e.g. to swap the
        # LLM annotator for the robustness study); defaults to the LoCLE file.
        self.cache_filename = cache_filename

        # Load LoCLE cache
        self._load_locle_cache(cache_dir)
        logger.info(
            "CachedAnnotator: %d entries loaded for %s",
            len(self._cache), dataset_name,
        )

    def _load_locle_cache(self, cache_dir):
        """Load LoCLE's pre-cached annotations."""
        import json

        dataset_key = self.dataset_name.lower()
        if dataset_key == "citeseer":
            dataset_key = "citeseer2"

        filename = (self.cache_filename
                    or f"{dataset_key}_temperature_0.7_n_3_output_seed_42_save_zero.json")
        path = osp.join(cache_dir, filename)

        if osp.exists(path):
            with open(path, "r") as f:
                raw = json.load(f)
            for node_str, (label, conf) in raw.items():
                self._cache[int(node_str)] = (int(label), float(conf))
            logger.info("Loaded %d LoCLE annotations from %s", len(self._cache), path)
        else:
            logger.warning("LoCLE cache not found: %s", path)

    def annotate_nodes(self, node_ids, texts, class_names=None, force=False):
        """Return cached annotations, falling back to random for uncached nodes."""
        results = {}
        for nid, text in zip(node_ids, texts):
            if nid in self._cache and not force:
                results[nid] = self._cache[nid]
            else:
                # Fallback: keyword matching + noise (like pipeline self-test)
                # Sort by name length descending so longer/more-specific names
                # match first (e.g. "probabilistic_methods" before "theory")
                text_lower = text.lower()
                pred = self._rng.randint(0, self.num_classes)
                sorted_classes = sorted(
                    enumerate(self.class_names), key=lambda x: len(x[1]), reverse=True
                )
                for idx, name in sorted_classes:
                    name_lower = name.lower()
                    # Match all variants: "case based", "case_based", "case-based"
                    if (name_lower.replace("_", " ") in text_lower
                            or name_lower in text_lower
                            or name_lower.replace("_", "-") in text_lower):
                        pred = idx
                        break
                # Add noise
                if self._rng.random() < 0.15:
                    pred = self._rng.randint(0, self.num_classes)
                conf = float(self._rng.uniform(60, 95))
                results[nid] = (int(pred), conf)
                self._cache[nid] = (int(pred), conf)
            self.total_api_calls += 1
        return results

    def get_cached(self, node_ids):
        """Return cached annotations for given node_ids."""
        return {nid: self._cache[nid] for nid in node_ids if nid in self._cache}

    def get_all_cached(self):
        return dict(self._cache)


def main():
    config = parse_args()
    config = resolve_paths(config)

    # Setup logging
    log_level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Set seeds
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("=" * 70)
    logger.info("NoisePref Pipeline — %s", config["experiment"])
    logger.info("=" * 70)
    logger.info("Config: %s", {k: v for k, v in config.items() if k not in ("data_path", "cache_dir")})
    logger.info("Data path: %s", config["data_path"])
    logger.info("Cache dir: %s", config["cache_dir"])

    # ── Load dataset ──
    dataset_name = config["dataset"]
    data = load_dataset(dataset_name, data_path=config["data_path"])
    num_nodes = data.x.size(0)
    num_classes = data.num_classes
    logger.info(
        "Dataset: %s | nodes=%d | features=%d | edges=%d | classes=%d",
        dataset_name, num_nodes, data.x.size(1),
        data.edge_index.size(1), num_classes,
    )

    # ── Budget ──
    total_budget = config["budget_per_class"] * num_classes
    logger.info("Total LLM budget: %d (%d per class × %d classes)",
                total_budget, config["budget_per_class"], num_classes)

    # ── Create annotator ──
    if config["use_locle_cache"]:
        # Use cached annotator (LoCLE cache) instead of live API calls
        annotator = CachedAnnotator(
            dataset_name=dataset_name,
            class_names=data.class_names,
            cache_dir=config["cache_dir"],
            seed=seed,
        )
    else:
        # Use live OpenAI API for annotation
        # Clear any stale cache so all nodes get fresh API calls
        stale_cache = osp.join(
            config["cache_dir"],
            "{}_{}.json".format(dataset_name.lower(),
                                config.get("llm_model", "gpt-4o-mini").replace("/", "_")),
        )
        if osp.exists(stale_cache):
            os.rename(stale_cache, stale_cache + ".bak")
            logger.info("Moved stale cache to %s.bak", stale_cache)
        annotator = LLMAnnotator(
            dataset_name=dataset_name,
            class_names=data.class_names,
            cache_dir=config["cache_dir"],
            model=config.get("llm_model", "gpt-4o-mini"),
            budget=total_budget,
        )
        logger.info("Using live LLM annotator: %s (fresh, no cache)", annotator.model)

    # ── Device ──
    device = config.get("device", "")
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ── Create pipeline ──
    pipeline = NoisePrefPipeline(
        annotator=annotator,
        dataset_name=dataset_name,
        class_names=data.class_names,
        backbone=config.get("gnn", "gcn"),
        total_budget=total_budget,
        probing_fraction=config["probing_fraction"],
        initial_fraction=config["initial_fraction"],
        total_rounds=config["max_rounds"],
        n_probes_per_class=max(2, total_budget // (num_classes * 5)),
        cache_dir=config["cache_dir"],
        seed=seed,
        gnn_hidden_dim=config["gnn_hidden_dim"],
        gnn_epochs=config["gnn_epochs"],
        gnn_lr=config["gnn_lr"],
        use_local_llm=config["use_local_llm"],
        local_llm_model=config["local_llm_model"],
        warmup_rounds=config["warmup_rounds"],
        use_noise_weighting=config["use_noise_weighting"],
        use_rewiring=config["use_rewiring"],
        rewiring_ratio=config["rewiring_ratio"],
        use_orpo=config["use_orpo"],
        lora_rank=config["lora_rank"],
        device=device,
        use_active_learning=config.get("use_active_learning", False),
        active_rounds=config.get("active_rounds", 5),
        active_budget_per_round=config.get("active_budget_per_round", 50),
        use_bootstrap=config.get("use_bootstrap", False),
        bootstrap_beta=config.get("bootstrap_beta", 0.8),
        use_elr=config.get("use_elr", False),
        elr_beta=config.get("elr_beta", 0.7),
        elr_lambda=config.get("elr_lambda", 3.0),
        use_coteaching=config.get("use_coteaching", False),
        coteaching_frac=config.get("coteaching_frac", 0.3),
        use_loss_pruning=config.get("use_loss_pruning", False),
        use_ilc=config.get("use_ilc", False),
        use_ensemble=config.get("use_ensemble", False),
        use_naap=config.get("use_naap", False),
        naap_alpha_base=config.get("naap_alpha_base", 0.3),
        naap_beta=config.get("naap_beta", 0.4),
        naap_gamma=config.get("naap_gamma", 0.5),
        naap_k_max=config.get("naap_k_max", 30),
        self_label_multiplier=config.get("self_label_multiplier", 3),
        use_locle_rewire=config.get("use_locle_rewire", False),
        select_downweight_floor=config.get("select_downweight_floor", 0.0),
        select_min_class_frac=config.get("select_min_class_frac", 0.0),
    )

    # ── Run pipeline ──
    start_time = time.time()
    labels, noise_matrix, metrics = pipeline.run(
        data, convergence_threshold=config["convergence_threshold"],
        ground_truth_y=data.y,  # for per-round accuracy logging only
    )
    elapsed = time.time() - start_time

    # ── Evaluate ──
    logger.info("=" * 70)
    logger.info("EVALUATION (vs ground truth — for debugging/comparison only)")
    logger.info("=" * 70)

    # Evaluate label quality on labeled nodes
    label_metrics = evaluate_labels(
        labels, data.y, num_nodes, class_names=data.class_names,
    )
    logger.info(
        "Label quality (on %d labeled nodes): accuracy=%.4f, macro_f1=%.4f",
        label_metrics["num_evaluated"],
        label_metrics["accuracy"],
        label_metrics["macro_f1"],
    )

    # Evaluate final GNN on ALL nodes (the real metric)
    if pipeline.gnn is not None:
        pipeline.gnn.eval()
        data_dev = data.to(pipeline.device)
        with torch.no_grad():
            logits = pipeline.gnn(data_dev)

        # Build labeled mask for evaluation
        labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for nid in labels:
            labeled_mask[nid] = True

        # Evaluate on unlabeled nodes only (fair evaluation)
        unlabeled_metrics = evaluate_predictions(
            logits, data.y, labeled_mask=labeled_mask,
            class_names=data.class_names,
        )
        logger.info(
            "GNN on unlabeled nodes (%d): accuracy=%.4f, macro_f1=%.4f, nmi=%.4f, ari=%.4f",
            unlabeled_metrics["num_evaluated"],
            unlabeled_metrics["accuracy"],
            unlabeled_metrics["macro_f1"],
            unlabeled_metrics["nmi"],
            unlabeled_metrics["ari"],
        )

        # Evaluate on ALL nodes
        all_metrics = evaluate_predictions(
            logits, data.y, labeled_mask=None,
            class_names=data.class_names,
        )
        logger.info(
            "GNN on all nodes (%d): accuracy=%.4f, macro_f1=%.4f, nmi=%.4f, ari=%.4f",
            all_metrics["num_evaluated"],
            all_metrics["accuracy"],
            all_metrics["macro_f1"],
            all_metrics["nmi"],
            all_metrics["ari"],
        )

        # Per-class F1
        if data.class_names and "per_class_f1" in all_metrics:
            logger.info("Per-class F1 (all nodes):")
            for i, name in enumerate(data.class_names):
                if i < len(all_metrics["per_class_f1"]):
                    logger.info("  %s: %.4f", name, all_metrics["per_class_f1"][i])
    else:
        all_metrics = label_metrics

    # ── Print summary ──
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Dataset: %s | GNN: %s | Seed: %d", dataset_name, config["gnn"], seed)
    logger.info("Total budget: %d | Probing: %.0f%% | Initial: %.0f%%",
                total_budget, config["probing_fraction"] * 100, config["initial_fraction"] * 100)
    logger.info("Rounds completed: %d", len(metrics["rounds"]))
    logger.info("Final labeled nodes: %d / %d (%.1f%%)",
                len(labels), num_nodes, 100 * len(labels) / num_nodes)
    logger.info("Elapsed: %.1f seconds", elapsed)
    logger.info("Budget: %s", metrics["budget_summary"])

    # Per-round summary
    logger.info("")
    logger.info("Per-round log:")
    for r in metrics["rounds"]:
        gnn_acc = r.get("gnn_accuracy")
        label_acc = r.get("label_accuracy")
        parts = []
        if gnn_acc is not None:
            parts.append(f"gnn_acc={gnn_acc:.1%}")
        if label_acc is not None:
            parts.append(f"label_acc={label_acc:.1%}")
        acc_str = (", " + ", ".join(parts)) if parts else ""
        logger.info(
            "Round %d: labeled=%d, flip_rate=%.4f, budget_remaining=%d%s",
            r["round"], r["num_labeled"],
            r.get("flip_rate", 0.0), r.get("budget_remaining", 0), acc_str,
        )

    logger.info("")
    logger.info("Noise matrix diagonal (per-class LLM accuracy):")
    logger.info("  %s", np.diag(noise_matrix).round(3))

    # ── Always-printed JSON summary (survives any log level) ──
    import json as _json
    summary_dict = {
        "dataset": dataset_name,
        "gnn": config.get("gnn", ""),
        "seed": seed,
        "experiment": config.get("experiment", ""),
        "budget": total_budget,
        "accuracy_all": all_metrics.get("accuracy"),
        "accuracy_unlabeled": unlabeled_metrics.get("accuracy") if pipeline.gnn is not None else None,
        "macro_f1_all": all_metrics.get("macro_f1"),
        "nmi_all": all_metrics.get("nmi"),
        "ari_all": all_metrics.get("ari"),
        "num_labeled": len(labels),
        "elapsed_sec": elapsed,
    }
    print("RESULT_JSON::" + _json.dumps(summary_dict))

    logger.info("")
    logger.info("Pipeline complete.")

    return labels, noise_matrix, metrics, all_metrics


if __name__ == "__main__":
    main()
