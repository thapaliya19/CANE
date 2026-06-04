#!/usr/bin/env python3
"""
experiment_runner.py — Runs a single NoisePref experiment given a config JSON.

Called by SLURM jobs. Read-only — do not modify.

Usage:
    python experiment_runner.py --config configs/exp001_cora_gcn_s0.json
    python experiment_runner.py --config configs/exp001_cora_gcn_s0.json --dry-run
"""

import argparse
import json
import logging
import os
import os.path as osp
import sys
import time
import traceback

import numpy as np
import torch

# Add repo root to path
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT = osp.abspath(osp.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from autoresearch.prepare import (
    OracleAnnotator,
    LocalLLMAnnotator,
    LocalLLMCacheAnnotator,
    evaluate_experiment,
    compute_improvement,
)

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None):
    """Configure logging to both console and file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(osp.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def load_dataset(dataset_name: str, data_path: str):
    """Load dataset, trying multiple paths."""
    from src.data_loading.tag_dataset import load_dataset as _load
    
    # Try paths in order
    candidates = [
        data_path,
        osp.join(REPO_ROOT, "data"),
        osp.join(REPO_ROOT, "baselines", "locle", "data"),
    ]
    
    for path in candidates:
        try:
            data = _load(dataset_name, data_path=path)
            logger.info("Loaded %s from %s: %d nodes, %d classes",
                       dataset_name, path, data.x.size(0), data.num_classes)
            return data
        except (FileNotFoundError, ValueError):
            continue
    
    raise FileNotFoundError(
        f"Dataset {dataset_name} not found in any of: {candidates}"
    )


def create_annotator(config: dict, data):
    """Create the appropriate annotator based on config."""
    mode = config.get("annotation_mode", "oracle")
    seed = config.get("seed", 0)
    
    if mode == "oracle":
        return OracleAnnotator(
            ground_truth_y=data.y,
            class_names=data.class_names,
            noise_rate=config.get("oracle_noise_rate", 0.0),
            seed=seed,
        )
    
    elif mode == "oracle_noisy":
        # Use a synthetic noise matrix that mimics LLM confusion patterns
        num_classes = data.num_classes
        rng = np.random.RandomState(seed)
        noise_matrix = np.eye(num_classes) * 0.7
        off_diag = rng.dirichlet(np.ones(num_classes), size=num_classes) * 0.3
        np.fill_diagonal(off_diag, 0.0)
        noise_matrix += off_diag
        noise_matrix /= noise_matrix.sum(axis=1, keepdims=True)
        
        return OracleAnnotator(
            ground_truth_y=data.y,
            class_names=data.class_names,
            noise_rate=config.get("oracle_noise_rate", 0.15),
            noise_matrix=noise_matrix,
            seed=seed,
        )
    
    elif mode == "local_llm":
        return LocalLLMAnnotator(
            model_path=config["local_llm_model_path"],
            class_names=data.class_names,
            dataset_name=config["dataset"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            batch_size=config.get("local_llm_batch_size", 8),
            quantize_4bit=config.get("local_llm_quantize_4bit", False),
            cache_dir=config.get("cache_dir", "data/annotations/"),
            seed=seed,
        )
    
    elif mode == "local_llm_cache":
        # Use pre-generated annotation cache (from annotate_all.py)
        model_name = config.get("local_llm_cache_model", "Qwen2.5-7B-Instruct")
        return LocalLLMCacheAnnotator(
            dataset_name=config["dataset"],
            class_names=data.class_names,
            model_name=model_name,
            cache_dir=config.get("cache_dir", "data/annotations/"),
            seed=seed,
        )
    
    elif mode == "locle_cache":
        # Use the CachedAnnotator from run.py
        from src.training.run import CachedAnnotator
        return CachedAnnotator(
            dataset_name=config["dataset"],
            class_names=data.class_names,
            cache_dir=config.get("cache_dir", "data/annotations/"),
            seed=seed,
            cache_filename=config.get("locle_cache_filename"),
        )

    elif mode == "gpt_live":
        # Live GPT annotation with fresh API calls for uncached nodes.
        # For arxiv: pre-loads 54,520 titles from arxiv_titles.json as raw_texts
        # and (by default) restricts available_nodes to that pool.
        # Budget = budget_per_class * num_classes.
        #
        # Opt-in config key `arxiv_unrestricted_pool: bool` (default False):
        #   * False (default): restrict active selection to the 54,520 titled
        #     nodes — preserves v4 / paper config behavior exactly.
        #   * True: keep loading titles into raw_texts, but do NOT restrict the
        #     pool; all 169,343 arxiv nodes become selectable, matching
        #     LLM-GNN's unrestricted behaviour. Nodes without titles carry a
        #     "node_{i}" placeholder (from tag_dataset.py) — the LLM annotator
        #     detects and skips these rather than burning API calls on them.
        from src.llm_annotation.annotator import LLMAnnotator

        model_name = config.get("gpt_cache_model", "gpt-3.5-turbo")
        num_classes = data.num_classes
        total_budget = config.get("budget_per_class", 20) * num_classes

        # Load API key
        api_key = None
        key_path = os.path.expanduser("~/.secrets/openai_api_key")
        if osp.exists(key_path):
            with open(key_path, "r") as f:
                api_key = f.read().strip()

        # Load arxiv titles into data.raw_texts and (optionally) define available pool
        available_pool = None
        unrestricted = bool(config.get("arxiv_unrestricted_pool", False))
        if config["dataset"].lower() == "arxiv":
            titles_filename = config.get("arxiv_titles_filename", "arxiv_titles.json")
            titles_path = osp.join(
                config.get("cache_dir", "data/annotations/"), titles_filename
            )
            if osp.exists(titles_path):
                import json as _json
                with open(titles_path, "r") as f:
                    titles = _json.load(f)
                raw_texts = list(data.raw_texts) if hasattr(data, 'raw_texts') else ["" for _ in range(data.x.size(0))]
                pool = set()
                for k, v in titles.items():
                    idx = int(k)
                    if 0 <= idx < len(raw_texts):
                        raw_texts[idx] = str(v)
                        pool.add(idx)
                data.raw_texts = raw_texts
                if unrestricted:
                    # Unrestricted: do NOT constrain the selector. Nodes without
                    # titles keep their "node_{i}" placeholder; the annotator
                    # will detect and skip those during actual LLM querying.
                    available_pool = None
                    logger.info(
                        "gpt_live: loaded %d arxiv titles; arxiv_unrestricted_pool=True, "
                        "active selection runs over all %d nodes (LLM-GNN parity)",
                        len(titles), data.x.size(0),
                    )
                else:
                    available_pool = pool
                    logger.info(
                        "gpt_live: loaded %d arxiv titles; available node pool = %d "
                        "(set arxiv_unrestricted_pool=true to use all %d nodes)",
                        len(titles), len(pool), data.x.size(0),
                    )

        if available_pool is not None:
            # Restricted mode (default, preserves v4 behaviour): expose
            # get_available_nodes() so the pipeline constrains selection to
            # the titled-node pool.
            class LiveGPTAnnotator(LLMAnnotator):
                """Wraps LLMAnnotator; exposes get_available_nodes() (restricted pool)."""
                def __init__(self, *args, available_nodes=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._available_nodes = available_nodes

                def get_available_nodes(self):
                    if self._available_nodes is not None:
                        return self._available_nodes
                    # Fall back to cached nodes (never constrains fresh calls
                    # since the caller only hits this path when the attribute
                    # exists).
                    return set(self._cache.keys())
        else:
            # Unrestricted mode (opt-in via arxiv_unrestricted_pool=True):
            # deliberately do NOT define get_available_nodes so the pipeline's
            # hasattr(annotator, 'get_available_nodes') check is False, which
            # makes _psfeatprop_w_select (and NoiseAwareSelector) run over all
            # nodes — matching LLM-GNN's behaviour.
            class LiveGPTAnnotator(LLMAnnotator):  # noqa: F811
                """Wraps LLMAnnotator; no pool restriction (unrestricted mode)."""
                def __init__(self, *args, available_nodes=None, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._available_nodes = None
                # Deliberately no get_available_nodes() — hasattr() must be False.

        # LLM-GNN-style consistency voting knobs. Defaults preserve the
        # original n=1 / temperature=0.0 behaviour exactly.
        annotator_n_samples = int(config.get("annotator_n_samples", 1))
        annotator_sampling_temperature = config.get(
            "annotator_sampling_temperature", None
        )
        annotator_budget_per_node = bool(
            config.get("annotator_budget_per_node", False)
        )
        # Prompt-template style. "ours" (default) is bit-identical to previous
        # behaviour. "llmgnn" switches the arxiv prompt to a bit-exact port of
        # LLM-GNN's top-k consistency prompt (which achieved 72.92% annotation
        # accuracy on ogbn-arxiv in their reference impl); non-arxiv datasets
        # fall through to "ours" regardless.
        annotator_prompt_style = str(config.get("arxiv_prompt_style", "ours"))

        # Force-fresh + cache-suffix knobs. Defaults (False, "") preserve the
        # pre-existing behaviour bit-identically. When force_fresh=True the
        # annotator ignores the cache for deciding what to query, so callers
        # should ALSO set cache_suffix to avoid stomping the baseline cache.
        annotator_force_fresh = bool(config.get("annotator_force_fresh", False))
        annotator_cache_suffix = str(config.get("annotator_cache_suffix", ""))

        return LiveGPTAnnotator(
            dataset_name=config["dataset"],
            class_names=data.class_names,
            cache_dir=config.get("cache_dir", "data/annotations/"),
            model=model_name,
            budget=total_budget,
            api_key=api_key,
            available_nodes=available_pool,
            n_samples=annotator_n_samples,
            sampling_temperature=annotator_sampling_temperature,
            budget_per_node=annotator_budget_per_node,
            prompt_style=annotator_prompt_style,
            force_fresh=annotator_force_fresh,
            cache_suffix=annotator_cache_suffix,
        )

    elif mode == "gpt_cache":
        # Use pre-generated GPT annotation cache (from annotate_arxiv_gpt.py)
        from src.training.run import CachedAnnotator

        class GPTCacheAnnotator(CachedAnnotator):
            """Annotator using pre-cached GPT-4o-mini annotations.

            Also exposes available_node_ids so the pipeline can constrain
            active selection to nodes that have cached annotations.
            """
            def _load_locle_cache(self, cache_dir):
                import json
                model_name = config.get("gpt_cache_model", "gpt-4o-mini")
                ds = config["dataset"]
                # LoCLE/cache convention: citeseer files are stored as "citeseer2".
                if str(ds).lower() == "citeseer":
                    ds = "citeseer2"
                filename = f"{ds}_{model_name.replace('/', '_')}.json"
                path = osp.join(cache_dir, filename)
                if osp.exists(path):
                    with open(path, "r") as f:
                        raw = json.load(f)
                    for node_str, val in raw.items():
                        if isinstance(val, list):
                            self._cache[int(node_str)] = (int(val[0]), float(val[1]))
                        else:
                            self._cache[int(node_str)] = (int(val), 80.0)
                    logger.info("Loaded %d GPT annotations from %s", len(self._cache), path)
                else:
                    logger.warning("GPT cache not found: %s", path)

            def get_available_nodes(self):
                """Return set of node IDs that have cached annotations."""
                return set(self._cache.keys())

        return GPTCacheAnnotator(
            dataset_name=config["dataset"],
            class_names=data.class_names,
            cache_dir=config.get("cache_dir", "data/annotations/"),
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown annotation mode: {mode}")


def run_experiment(config: dict) -> dict:
    """
    Run a single NoisePref experiment.
    
    Args:
        config: Experiment configuration dict.
    
    Returns:
        Results dict with metrics, timing, and improvement over LoCLE.
    """
    dataset_name = config["dataset"]
    backbone = config["backbone"]
    seed = config["seed"]
    
    logger.info("=" * 70)
    logger.info("EXPERIMENT: %s | %s | %s | seed=%d",
               config.get("experiment_tag", "unknown"), dataset_name, backbone, seed)
    logger.info("=" * 70)
    logger.info("Config: %s", json.dumps({
        k: v for k, v in config.items() 
        if k not in ("hypothesis", "datasets", "backbones", "seeds")
    }, indent=2))
    
    # Set seeds
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Optional full determinism (config: deterministic=true). Forces deterministic
    # CUDA kernels so re-running an identical config reproduces bit-for-bit. NOTE:
    # this only fixes WITHIN-config reproducibility — it does NOT remove the need
    # to multi-seed for ranking. Requires CUBLAS_WORKSPACE_CONFIG set in the env
    # (we set it in the sbatch wrap). Some ops (e.g. GAT scatter_softmax) may lack
    # a deterministic CUDA impl; warn_only=False makes those raise loudly so we
    # know rather than silently staying non-deterministic.
    if config.get("deterministic", False):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warn_only = bool(config.get("deterministic_warn_only", False))
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        logger.info("Determinism ON (use_deterministic_algorithms, warn_only=%s, "
                    "CUBLAS_WORKSPACE_CONFIG=%s)", warn_only,
                    os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

    # Load dataset
    data_path = config.get("data_path", osp.join(REPO_ROOT, "data"))
    data = load_dataset(dataset_name, data_path)
    
    # Create annotator
    annotator = create_annotator(config, data)

    # ── Synthetic-noise injection (Task 2 noise sweep) ─────────────────────
    # Config keys: synthetic_noise_tau (float), synthetic_noise_structure
    # ('sym' or 'asym'). When set, we REPLACE the annotator's cached labels
    # with ground-truth labels corrupted according to the chosen noise model,
    # and later inject the induced noise matrix directly into pipeline.noise_matrix
    # so Stage-0 probing is short-circuited.
    _synthetic_noise_tau = config.get("synthetic_noise_tau", None)
    _synthetic_noise_structure = config.get("synthetic_noise_structure", None)
    _synthetic_noise_matrix = None
    if _synthetic_noise_tau is not None and _synthetic_noise_structure is not None:
        tau = float(_synthetic_noise_tau)
        structure = str(_synthetic_noise_structure).lower()
        C = int(data.num_classes)
        num_nodes = int(data.x.size(0))
        rng = np.random.RandomState(seed)
        y_np = data.y.cpu().numpy().astype(np.int64)

        # Build the true (population) noise matrix N[i,j] = P(label'=j | y=i).
        N = np.zeros((C, C), dtype=np.float32)
        if structure == "sym":
            for i in range(C):
                N[i, i] = 1.0 - tau
                if C > 1:
                    for j in range(C):
                        if j != i:
                            N[i, j] = tau / (C - 1)
        elif structure == "asym":
            for i in range(C):
                j = (i + 1) % C
                N[i, i] = 1.0 - tau
                N[i, j] = tau
        else:
            raise ValueError(f"Unknown synthetic_noise_structure: {structure}")

        # Realise a per-node noisy label draw and replace annotator cache.
        # confidence is a fixed proxy (80.0) since there is no real LLM.
        synth_cache = {}
        for nid in range(num_nodes):
            true_c = int(y_np[nid])
            probs = N[true_c]
            noisy = int(rng.choice(C, p=probs))
            synth_cache[nid] = (noisy, 80.0)

        # Override annotator cache wholesale.
        annotator._cache = synth_cache
        _synthetic_noise_matrix = N
        logger.info(
            "Synthetic noise injected: tau=%.3f structure=%s — replaced %d annotator cache entries",
            tau, structure, len(synth_cache),
        )

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s (%.1f GB)", 
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    
    # Budget
    num_classes = data.num_classes
    total_budget = config.get("budget_per_class", 20) * num_classes
    # may18 v6: AQP runs in ISOBUDGET mode by default — Stage I gives up
    # aqp_n_probes queries to AQP so total LLM cost is unchanged from baseline.
    # The pipeline's _run_initial_annotation handles the carve-out.
    # To run "extra-budget" mode (AQP on top of Stage I), set aqp_extra_budget=True
    # in the config.
    if config.get("use_aqp", False) and config.get("aqp_extra_budget", False):
        total_budget += int(config.get("aqp_n_probes", 25))
    
    # Import and create pipeline
    from src.training.noisepref_pipeline import NoisePrefPipeline

    pipeline = NoisePrefPipeline(
        annotator=annotator,
        dataset_name=dataset_name,
        class_names=data.class_names,
        total_budget=total_budget,
        probing_fraction=config.get("probing_fraction", 0.10),
        initial_fraction=config.get("initial_fraction", 0.50),
        total_rounds=config.get("max_rounds", 5),
        n_probes_per_class=max(2, total_budget // (num_classes * 5)),
        cache_dir=config.get("cache_dir", "data/annotations/"),
        seed=seed,
        gnn_hidden_dim=config.get("gnn_hidden_dim", 64),
        gnn_epochs=config.get("gnn_epochs", 200),
        gnn_lr=config.get("gnn_lr", 0.01),
        gnn_num_layers=config.get("gnn_num_layers", 2),
        gnn_weight_decay=config.get("gnn_weight_decay", 5e-4),
        use_local_llm=config.get("use_local_llm_training", False),
        local_llm_model=config.get("local_llm_model_path", ""),
        warmup_rounds=config.get("warmup_rounds", 2),
        use_noise_weighting=config.get("use_noise_weighting", True),
        use_rewiring=config.get("use_rewiring", False),
        rewiring_ratio=config.get("rewiring_ratio", 0.05),
        use_orpo=config.get("use_orpo", True),
        lora_rank=config.get("lora_rank", 16),
        backbone=config.get("backbone", "gcn"),
        gnn_dropout=config.get("gnn_dropout", 0.5),
        lp_alpha=config.get("lp_alpha", 0.5),
        lp_iters=config.get("lp_iters", 50),
        label_smoothing=config.get("label_smoothing", 0.0),
        use_gce=config.get("use_gce", False),
        gce_q=config.get("gce_q", 0.7),
        use_noise_ada=config.get("use_noise_ada", False),
        device=device,
        use_full_graph_denoising=config.get("use_full_graph_denoising", False),
        denoising_lp_alpha=config.get("denoising_lp_alpha", 0.3),
        denoising_lp_iters=config.get("denoising_lp_iters", 20),
        use_neighborhood_correction=config.get("use_neighborhood_correction", False),
        neighborhood_correction_threshold=config.get("neighborhood_correction_threshold", 0.5),
        self_training_rounds=config.get("self_training_rounds", 3),
        dropedge_rate=config.get("dropedge_rate", 0.0),
        use_sce=config.get("use_sce", False),
        sce_beta=config.get("sce_beta", 0.5),
        use_active_learning=config.get("use_active_learning", False),
        active_rounds=config.get("active_rounds", 5),
        active_budget_per_round=config.get("active_budget_per_round", 50),
        entropy_reg=config.get("entropy_reg", 0.0),
        use_forward_correction=config.get("use_forward_correction", False),
        use_cluster_T_c_for_forward_correction=config.get(
            "use_cluster_T_c_for_forward_correction", False),
        use_xmod_weight=config.get("use_xmod_weight", False),
        xmod_alpha=config.get("xmod_alpha", 2.0),
        use_dual_xmod=config.get("use_dual_xmod", False),
        dual_xmod_k=config.get("dual_xmod_k", 10),
        dual_xmod_fallback=config.get("dual_xmod_fallback", False),
        use_xmod_relabel=config.get("use_xmod_relabel", False),
        xmod_relabel_min_graph=config.get("xmod_relabel_min_graph", 2),
        use_reliability_selection=config.get("use_reliability_selection", False),
        reliability_select_floor=config.get("reliability_select_floor", 0.3),
        reliability_select_beta=config.get("reliability_select_beta", 0.5),
        use_feat_prop=config.get("use_feat_prop", False),
        use_edge_prune=config.get("use_edge_prune", False),
        use_naap=config.get("use_naap", False),
        # Phase 17: cluster-conditional T_c for NAAP (unified NoisePref + RAACA)
        use_cluster_T_c=config.get("use_cluster_T_c", False),
        cluster_T_c_path=config.get("cluster_T_c_path", None),
        cluster_id_path=config.get("cluster_id_path", None),
        tc_within_budget=config.get("tc_within_budget", False),
        use_crossmodal_tc=config.get("use_crossmodal_tc", False),
        tc_global_average=config.get("tc_global_average", False),
        use_minibatch=config.get("use_minibatch", False),
        minibatch_size=config.get("minibatch_size", 1024),
        minibatch_fanout=config.get("minibatch_fanout", [15, 10]),
        # DUAL-LP: cross-view (graph + feature-kNN) label-propagation
        # disagreement filter. Per-node confidence modulates NAAP/ILC/GNN-loss.
        use_dual_lp=config.get("use_dual_lp", False),
        dual_lp_k=config.get("dual_lp_k", 10),
        dual_lp_alpha=config.get("dual_lp_alpha", 0.85),
        dual_lp_iters=config.get("dual_lp_iters", 20),
        dual_lp_floor=config.get("dual_lp_floor", 0.10),
        dual_lp_blend_agreement=config.get("dual_lp_blend_agreement", 0.7),
        dual_lp_blend_margin=config.get("dual_lp_blend_margin", 0.3),
        dual_lp_apply_naap=config.get("dual_lp_apply_naap", True),
        dual_lp_apply_ilc=config.get("dual_lp_apply_ilc", True),
        dual_lp_apply_loss=config.get("dual_lp_apply_loss", True),
        dual_lp_ilc_scale=config.get("dual_lp_ilc_scale", 0.2),
        naap_alpha_base=config.get("naap_alpha_base", 0.3),
        naap_beta=config.get("naap_beta", 0.8),
        naap_gamma=config.get("naap_gamma", 0.6),
        naap_k_max=config.get("naap_k_max", 30),
        naap_patience=config.get("naap_patience", 3),
        use_sgr=config.get("use_sgr", False),
        sgr_prune_ratio=config.get("sgr_prune_ratio", 0.1),
        sgr_augment_k=config.get("sgr_augment_k", 5),
        use_ncl=config.get("use_ncl", False),
        use_dmcr=config.get("use_dmcr", False),
        use_soft_labels=config.get("use_soft_labels", False),
        use_tc_soft_labels=config.get("use_tc_soft_labels", False),
        tc_soft_mode=config.get("tc_soft_mode", "column"),
        use_bayes_rectify=config.get("use_bayes_rectify", False),
        use_naew=config.get("use_naew", False),
        use_ilc=config.get("use_ilc", False),
        ilc_conf_threshold=config.get("ilc_conf_threshold", 0.9),
        ilc_max_rounds=config.get("ilc_max_rounds", 5),
        ilc_neighbor_threshold=config.get("ilc_neighbor_threshold", 0.5),
        ilc_use_cluster_T_c=config.get("ilc_use_cluster_T_c", False),
        ilc_tc_scale=config.get("ilc_tc_scale", 0.5),
        # may18 v5: per-round T_c update inside ILC
        ilc_update_T_c_per_round=config.get("ilc_update_T_c_per_round", False),
        ilc_tc_update_method=config.get("ilc_tc_update_method", "v3"),
        # may18 v6: NAAP clustering on GraphMAE2 features
        naap_use_graphmae_features=config.get("naap_use_graphmae_features", False),
        graphmae_features_path=config.get("graphmae_features_path", None),
        # may18 v6: Active Quality Probing (AQP)
        use_aqp=config.get("use_aqp", False),
        aqp_n_probes=config.get("aqp_n_probes", 25),
        aqp_drop_threshold=config.get("aqp_drop_threshold", 0.5),
        aqp_action=config.get("aqp_action", "hard_drop"),
        aqp_weight_floor=config.get("aqp_weight_floor", 0.1),
        # may18 v7: Consistency regularization on non-selected nodes
        use_consistency_reg=config.get("use_consistency_reg", False),
        consistency_lambda=config.get("consistency_lambda", 1.0),
        consistency_aug_dropedge=config.get("consistency_aug_dropedge", 0.5),
        consistency_aug_feat_mask=config.get("consistency_aug_feat_mask", 0.1),
        consistency_warmup_epochs=config.get("consistency_warmup_epochs", 50),
        consistency_conf_threshold=config.get("consistency_conf_threshold", 0.0),
        # may18 v8: T_c-aware self-training expansion gate
        use_tc_expansion_gate=config.get("use_tc_expansion_gate", False),
        tc_expansion_gate_alpha=config.get("tc_expansion_gate_alpha", 0.3),
        # may18 TGER: Static kNN + Interpolated Soft TGER rewiring
        # (src/structure_learning/tger.py). GCN-only; GAT bypasses.
        use_tger=config.get("use_tger", False),
        tger_beta=config.get("tger_beta", 0.6),
        tger_knn_k=config.get("tger_knn_k", 5),
        tger_embedding_path=config.get("tger_embedding_path", None),
        # may19 SSAF: Semantic-Structural Adaptive Fusion dual-channel GNN.
        # When use_ssaf=true, the backbone is swapped for SSAF_GNN; the
        # cctger_off loss machinery (T_c forward correction, ILC, ELR,
        # NAAP) keeps running unchanged because SSAF subclasses GCN.
        use_ssaf=config.get("use_ssaf", False),
        ssaf_knn_k=config.get("ssaf_knn_k", 5),
        ssaf_embedding_path=config.get("ssaf_embedding_path", None),
        # may19 RTAA: Robust Topology-Aware Augmentation (data-level).
        # Stock GCN / GAT backbones; weights edges by T_c[k_u, ŷ_u, ŷ_v]
        # and periodically adds synthetic same-class high-confidence
        # homophily edges during training.
        use_rtaa=config.get("use_rtaa", False),
        rtaa_conf_threshold=config.get("rtaa_conf_threshold", 0.90),
        rtaa_refresh_every=config.get("rtaa_refresh_every", 50),
        rtaa_k_per_class=config.get("rtaa_k_per_class", 20),
        rtaa_max_added=config.get("rtaa_max_added", 5000),
        rtaa_default_weight=config.get("rtaa_default_weight", 1.0),
        # may19 Tuning Phase — two cctger_off refinement knobs.
        forward_correction_temperature=config.get(
            "forward_correction_temperature", 1.0
        ),
        use_low_conf_drop=config.get("use_low_conf_drop", False),
        low_conf_threshold=config.get("low_conf_threshold", 0.85),
        # may19 DIAGNOSTIC — gold-label injection into LoCLE rewire path.
        # NOT a fix. Audit-only flag to measure LoCLE leak attribution.
        diag_inject_locle_gold_backup_y=config.get(
            "diag_inject_locle_gold_backup_y", False
        ),
        # may19 Cora-rescue ablation suite.
        gnn_variant=config.get("gnn_variant", "gcn"),
        gcnii_alpha=config.get("gcnii_alpha", 0.2),
        gcnii_beta=config.get("gcnii_beta", 0.2),
        dualpath_fusion=config.get("dualpath_fusion", "static"),
        dualpath_lambda=config.get("dualpath_lambda", 0.7),
        jk_mode=config.get("jk_mode", "concat"),
        gated_use_degree=config.get("gated_use_degree", True),
        appnp_K=config.get("appnp_K", 10),
        appnp_alpha=config.get("appnp_alpha", 0.1),
        dropedge_degree_aware=config.get("dropedge_degree_aware", False),
        use_ensemble=config.get("use_ensemble", False),
        use_bootstrap=config.get("use_bootstrap", False),
        bootstrap_beta=config.get("bootstrap_beta", 0.8),
        use_post_estimate_adj=config.get("use_post_estimate_adj", False),
        use_loss_pruning=config.get("use_loss_pruning", False),
        loss_prune_ratio=config.get("loss_prune_ratio", 0.10),
        final_dropedge_zero=config.get("final_dropedge_zero", False),
        use_locle_rewire=config.get("use_locle_rewire", False),
        locle_rewire_placement=config.get("locle_rewire_placement", "pre_st"),
        locle_rewire_overrides=config.get("locle_rewire_overrides", None),
        use_na_ea=config.get("use_na_ea", False),
        use_na_ea_v2=config.get("use_na_ea_v2", False),
        stage1_splits=config.get("stage1_splits", None),
        stage1_all_subspace=config.get("stage1_all_subspace", False),
        select_downweight_floor=config.get("select_downweight_floor", 0.0),
        select_min_class_frac=config.get("select_min_class_frac", 0.0),
        # Path 1 (apr13): LoCLE iterative refinement flag + knobs.
        use_locle_iterative=config.get("use_locle_iterative", False),
        # may17: LoCLE-faithful GCN backbone for _train_gnn.
        use_locle_gnn_backbone=config.get("use_locle_gnn_backbone", False),
        locle_gnn_epochs=config.get("locle_gnn_epochs", 30),
        locle_gnn_lr=config.get("locle_gnn_lr", 0.01),
        locle_gnn_wd=config.get("locle_gnn_wd", 5e-4),
        # may17 tricks pack
        use_mc_dropout=config.get("use_mc_dropout", False),
        mc_dropout_passes=config.get("mc_dropout_passes", 10),
        use_self_distill_final=config.get("use_self_distill_final", False),
        self_distill_n=config.get("self_distill_n", 200),
        self_distill_conf=config.get("self_distill_conf", 0.85),
        use_clean_subset_retrain=config.get("use_clean_subset_retrain", False),
        clean_subset_drop_frac=config.get("clean_subset_drop_frac", 0.30),
        clean_subset_iterations=config.get("clean_subset_iterations", 1),
        clean_subset_iter_drop_frac=config.get("clean_subset_iter_drop_frac", 0.30),
        clean_subset_class_balanced=config.get("clean_subset_class_balanced", False),
        # may18 v3: filter-strength ensemble + ELR at end-training only
        clean_subset_drop_fracs=config.get("clean_subset_drop_fracs", None),
        use_elr_final=config.get("use_elr_final", False),
        # may18 v3: calibration diagnostic dump path (if set, pipeline dumps
        # labels + ensemble softmax to this .npz file after the first ensemble).
        diag_dump_path=config.get("diag_dump_path", None),
        # may18 v3: noise-detection signal for clean-subset retraining.
        # Options: 'ensemble' (default), 'entropy', 'per_sample_loss', 'llm_conf'.
        clean_subset_signal=config.get("clean_subset_signal", "ensemble"),
        # may18 v4: integrated verify-and-drop inside ILC
        use_verify_and_drop_ilc=config.get("use_verify_and_drop_ilc", False),
        verify_drop_threshold=config.get("verify_drop_threshold", 0.3),
        # may18 v4: cap on pseudo-labels added per self-training round
        self_train_cap_per_class=config.get("self_train_cap_per_class", 0),
        # may18 v5: cluster-conditional T_c weighting of self-training scores
        use_tc_self_training=config.get("use_tc_self_training", False),
        # may18 v5: drive pseudo-labeling from T_c-aware LP only
        use_lp_pseudo_labels=config.get("use_lp_pseudo_labels", False),
        locle_iterative_stages=config.get("locle_iterative_stages", 3),
        locle_iterative_subspace_frac=config.get("locle_iterative_subspace_frac", 0.0),
        locle_iterative_confident_frac=config.get("locle_iterative_confident_frac", 0.4),
        locle_iterative_inconfident_frac=config.get("locle_iterative_inconfident_frac", 0.2),
        locle_iter_conf_thresh=config.get("locle_iter_conf_thresh", 0.80),
        locle_iter_max_flip=config.get("locle_iter_max_flip", 0.20),
        wp_best_metric=config.get("wp_best_metric", "latest"),
        # PS-FeatProp-W (LLM-GNN pagerank2 replication) + confidence filter.
        use_psfeatprop_selection=config.get("use_psfeatprop_selection", False),
        psfeatprop_alpha=config.get("psfeatprop_alpha", 0.33),
        # When True, density = k-means on RAW X (exact LLM-GNN pg2_query).
        # When False (default), density = k-means on AAX (paper's v4 config).
        use_pagerank2_selection=config.get("use_pagerank2_selection", False),
        # Stratified per-pseudo-class selection augmentation for PS-FeatProp-W.
        # When True, allocate a per-cluster quota over feature-space k-means
        # pseudo-classes to guarantee coverage (fixes arxiv 40-class collapse).
        stratified_per_class=config.get("stratified_per_class", False),
        conf_filter_quantile=config.get("conf_filter_quantile", 0.0),
        # apr14 alternative Stage-I selectors. Mutually exclusive with
        # use_psfeatprop_selection; if both kcenter+age_query True, kcenter wins.
        use_kcenter_selection=config.get("use_kcenter_selection", False),
        use_age_query_selection=config.get("use_age_query_selection", False),
        age_query_alpha=config.get("age_query_alpha", 0.2),
        age_query_beta=config.get("age_query_beta", 0.3),
        # apr15 v92: fixed-selection mode (plug in pre-computed node IDs).
        # Overrides all Stage-I selectors when True. Default off, bit-identical.
        use_fixed_selection=config.get("use_fixed_selection", False),
        fixed_selection_path=config.get("fixed_selection_path", ""),
        # apr15 gap-closers: held-out val early stopping + final ensemble.
        # All default-off → bit-identical to pre-apr15 behavior.
        val_holdout_frac=config.get("val_holdout_frac", 0.0),
        early_stop_patience=config.get("early_stop_patience", 20),
        val_check_every=config.get("val_check_every", 1),
        val_add_back_final=config.get("val_add_back_final", False),
        final_ensemble_n=config.get("final_ensemble_n", 1),
        # apr15 v72: voting-confidence-weighted training.
        # Default False → bit-identical to legacy (uniform CE loss).
        use_vote_confidence_weights=config.get("use_vote_confidence_weights", False),
        # Cluster-conditional CCR (CCCR): mean-renormalised per-node weight
        # = CCR_weight * T_c[cluster, label, label] ** cluster_conf_pow.
        use_cluster_conf_weights=config.get("use_cluster_conf_weights", False),
        cluster_conf_pow=config.get("cluster_conf_pow", 1.0),
        # CRBA — Cluster-Risk-Based Budget Allocation for Stage-I R2-3 selection
        crba_lambda=config.get("crba_lambda", 0.0),
        crba_quota=config.get("crba_quota", 0.0),
        use_iterative_reannotation=config.get("use_iterative_reannotation", False),
        iter_reannot_stages=config.get("iter_reannot_stages", 5),
        iter_reannot_total_budget=config.get("iter_reannot_total_budget", 0),
        iter_reannot_score=config.get("iter_reannot_score", "entropy"),
        use_rank_correct=config.get("use_rank_correct", False),
        rank_correct_llm_thresh=config.get("rank_correct_llm_thresh", 0.80),
        rank_correct_mode=config.get("rank_correct_mode", "rank"),
        use_tblc=config.get("use_tblc", False),
        tblc_rounds=config.get("tblc_rounds", 1),
        tblc_ratio=config.get("tblc_ratio", 1.5),
        probing_mode=config.get("probing_mode", "template"),
        iter_self_label_frac=config.get("iter_self_label_frac", 0.0),
        iter_self_label_conf=config.get("iter_self_label_conf", 0.90),
        vote_conf_schedule=config.get("vote_conf_schedule", "linear"),
        # apr15 v89: OGB gold-val early stopping (LLM-GNN arxiv protocol).
        # Default False → bit-identical. When True AND dataset=arxiv, we
        # attach `data.ogb_val_idx` below so the pipeline uses those indices
        # with GOLD labels as an early-stop signal (NOT for training loss).
        use_ogb_val_early_stop=config.get("use_ogb_val_early_stop", False),
        # C2 — two-pass cluster-aware selection (apr28).
        use_two_pass_selector=config.get("use_two_pass_selector", False),
        two_pass_alpha=config.get("two_pass_alpha", 1.5),
        two_pass_p1_fraction=config.get("two_pass_p1_fraction", 0.5),
        two_pass_min_per_cluster=config.get("two_pass_min_per_cluster", 1),
        two_pass_entropy_eps=config.get("two_pass_entropy_eps", 0.10),
        use_noiseselect_for_round1=config.get("use_noiseselect_for_round1", False),
        noiseselect_cluster_alpha=config.get("noiseselect_cluster_alpha", 1.5),
        # Robust-learning toggles (existed in pipeline; wired here)
        use_coteaching=config.get("use_coteaching", False),
        use_elr=config.get("use_elr", False),
        elr_beta=config.get("elr_beta", 0.7),
        elr_lambda=config.get("elr_lambda", 3.0),
    )

    # apr15 v89: load OGB validation indices and attach to `data` when
    # OGB gold-val early stopping is requested AND dataset is arxiv.
    # The gold labels (already in data.y for arxiv) are used ONLY for
    # early-stop monitoring inside GCN._fit_supervised (never in loss).
    if (config.get("use_ogb_val_early_stop", False)
            and dataset_name.lower() == "arxiv"):
        try:
            import warnings as _warn
            _warn.filterwarnings("ignore")
            from ogb.nodeproppred import PygNodePropPredDataset
            _ogb_ds = PygNodePropPredDataset(
                "ogbn-arxiv",
                root=osp.join(REPO_ROOT, "data", "ogb"),
            )
            _split = _ogb_ds.get_idx_split()
            data.ogb_val_idx = _split["valid"].long().view(-1)
            logger.info(
                "Attached OGB val idx to data (n=%d) for gold-val early stop",
                int(data.ogb_val_idx.numel()),
            )
        except Exception as _e:
            logger.warning(
                "use_ogb_val_early_stop=True but could not load OGB split: %s", _e,
            )

    # If synthetic noise was injected, pre-populate pipeline.noise_matrix so
    # Stage-0 probing is skipped. Configs should set probing_fraction=0.
    if _synthetic_noise_matrix is not None:
        pipeline.noise_matrix = _synthetic_noise_matrix
        pipeline.per_class_accuracy = np.diag(_synthetic_noise_matrix).astype(np.float32)
        pipeline.confused_pairs = []
        logger.info(
            "Injected synthetic noise_matrix into pipeline: mean diag=%.3f",
            float(np.diag(_synthetic_noise_matrix).mean()),
        )

    # Run pipeline
    start_time = time.time()
    labels, noise_matrix, metrics = pipeline.run(
        data,
        convergence_threshold=config.get("convergence_threshold", 0.01),
        ground_truth_y=data.y,
    )
    elapsed = time.time() - start_time
    
    # Evaluate final GNN
    results = {
        "dataset": dataset_name,
        "backbone": backbone,
        "seed": seed,
        "experiment_tag": config.get("experiment_tag", "unknown"),
        "annotation_mode": config.get("annotation_mode", "oracle"),
        "elapsed_seconds": elapsed,
        "num_labeled": len(labels),
        "rounds_completed": len(metrics.get("rounds", [])),
        "total_api_calls": annotator.total_api_calls,
        # Pipeline config (for TSV tracking)
        "budget_per_class": config.get("budget_per_class", ""),
        "max_rounds": config.get("max_rounds", ""),
        "gnn_hidden_dim": config.get("gnn_hidden_dim", ""),
        "gnn_epochs": config.get("gnn_epochs", ""),
        "gnn_lr": config.get("gnn_lr", ""),
        "probing_fraction": config.get("probing_fraction", ""),
        "initial_fraction": config.get("initial_fraction", ""),
        "use_noise_weighting": config.get("use_noise_weighting", ""),
        "use_rewiring": config.get("use_rewiring", ""),
        "use_active_learning": config.get("use_active_learning", ""),
        "active_rounds": config.get("active_rounds", ""),
        "active_budget_per_round": config.get("active_budget_per_round", ""),
        "use_local_llm_training": config.get("use_local_llm_training", ""),
        "local_llm_model_path": config.get("local_llm_model_path", ""),
        "local_llm_annotations": metrics.get("local_llm_annotations", 0),
    }
    
    if pipeline.gnn is not None:
        pipeline.gnn.eval()
        data_dev = data.to(device)
        with torch.no_grad():
            logits = pipeline.gnn(data_dev)

        # ── apr15 final ensemble (seed-perturbed GCNs, softmax averaged) ──
        # final_ensemble_n == 1 → skip (bit-identical to legacy).
        # may18 v3: runner_final_ensemble_n overrides for the runner-side only.
        # When pipeline already produced a strong ensemble (e.g. clean_subset_drop_fracs)
        # the runner's extra 4 fresh-on-full-labels GCNs dilute the pipeline's logits
        # to 1/N final weight. Set runner_final_ensemble_n=1 to skip the runner's
        # add-on entirely and let pipeline.gnn.forward()'s logits be the answer.
        _ens_n_cfg = config.get("runner_final_ensemble_n", None)
        ens_n = int(_ens_n_cfg) if _ens_n_cfg is not None else int(config.get("final_ensemble_n", 1))
        if ens_n > 1:
            try:
                from src.models.gcn import GCN as _GCN
                # Optionally add held-out val labels back to the train set for
                # the final ensemble — they are legitimate LLM-annotated anchors.
                final_labels_dict = dict(labels)
                # Build train mask for final ensemble: use all pipeline-labeled
                # nodes (no holdout at final stage — we already used holdout
                # for early stopping during iterative training).
                num_nodes_e = data.x.size(0)
                labels_tensor_e = torch.full((num_nodes_e,), -1, dtype=torch.long)
                for nid, lbl in final_labels_dict.items():
                    if 0 <= int(nid) < num_nodes_e:
                        labels_tensor_e[int(nid)] = int(lbl)
                train_mask_e = torch.zeros(num_nodes_e, dtype=torch.bool)
                for nid in final_labels_dict:
                    if 0 <= int(nid) < num_nodes_e:
                        train_mask_e[int(nid)] = True
                labels_tensor_e = labels_tensor_e.to(device)
                train_mask_e = train_mask_e.to(device)

                # Accumulate softmax probs. Start with the pipeline's already-
                # trained model (counts as member 0).
                probs_sum = torch.softmax(logits, dim=1).detach()
                members_used = 1

                input_dim_e = data.x.size(1)
                hidden_dim_e = int(config.get("gnn_hidden_dim", 64))
                output_dim_e = int(data.num_classes)
                dropout_e = float(config.get("gnn_dropout", 0.5))
                num_layers_e = int(config.get("gnn_num_layers", 2))
                epochs_e = int(config.get("gnn_epochs", 200))
                lr_e = float(config.get("gnn_lr", 0.01))
                wd_e = float(config.get("gnn_weight_decay", 5e-4))

                for i in range(1, ens_n):
                    member_seed = seed * 100 + i
                    torch.manual_seed(member_seed)
                    np.random.seed(member_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(member_seed)

                    # TGER attaches asymmetric edge weights on data_dev.
                    # GCNConv(normalize=True) would silently symmetrise them
                    # via gcn_norm, producing nonsense aggregation. Match the
                    # pipeline's TGER mode so the ensemble respects the same
                    # weighted-mean semantics.
                    _tger_on = bool(config.get("use_tger", False))
                    gnn_i = _GCN(
                        input_dim=input_dim_e,
                        hidden_dim=hidden_dim_e,
                        output_dim=output_dim_e,
                        num_layers=num_layers_e,
                        dropout=dropout_e,
                        normalize_conv=(not _tger_on),
                    ).to(device)
                    _fit_kwargs_e = dict(
                        labels=labels_tensor_e,
                        train_mask=train_mask_e,
                        epochs=epochs_e,
                        lr=lr_e,
                        weight_decay=wd_e,
                    )
                    # apr15 v89: propagate OGB gold-val early stopping into
                    # each ensemble member (gold labels used ONLY for early
                    # stop, never in loss). Matches LLM-GNN's protocol.
                    if (config.get("use_ogb_val_early_stop", False)
                            and dataset_name.lower() == "arxiv"
                            and hasattr(data, "ogb_val_idx")
                            and data.ogb_val_idx is not None):
                        _ogb_idx_e = data.ogb_val_idx
                        if not isinstance(_ogb_idx_e, torch.Tensor):
                            _ogb_idx_e = torch.as_tensor(_ogb_idx_e, dtype=torch.long)
                        _ogb_idx_e = _ogb_idx_e.long().view(-1)
                        _vm_e = torch.zeros(num_nodes_e, dtype=torch.bool)
                        _vm_e[_ogb_idx_e] = True
                        _fit_kwargs_e["val_mask"] = _vm_e.to(device)
                        _fit_kwargs_e["val_labels"] = data.y.view(-1).long().to(device)
                        _fit_kwargs_e["early_stop_patience"] = int(
                            config.get("early_stop_patience", 20))
                        _fit_kwargs_e["val_check_every"] = int(
                            config.get("val_check_every", 1))
                    gnn_i.fit(data_dev, **_fit_kwargs_e)
                    gnn_i.eval()
                    with torch.no_grad():
                        logits_i = gnn_i(data_dev)
                    probs_sum = probs_sum + torch.softmax(logits_i, dim=1).detach()
                    members_used += 1

                probs_avg = probs_sum / float(members_used)
                # Convert back to logits-like tensor (log-probs) so downstream
                # argmax / softmax behave identically.
                logits = torch.log(probs_avg.clamp(min=1e-8))
                logger.info(
                    "Final ensemble: averaged softmax over %d GCNs (seeds %d..%d)",
                    members_used, seed, seed * 100 + ens_n - 1,
                )
                # Restore the main rng state to the pipeline seed for downstream.
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception as _ens_err:
                logger.warning("Final ensemble failed (%s) — falling back to single model", _ens_err)

        # ── Self-distillation post-processing ────────────────────────────
        # Take the (already-ensembled) softmax predictions, pick top-K
        # high-confidence unlabeled nodes as pseudo-labels, add them to
        # the training set with reduced weight, and retrain the GCN.
        # Default use_self_distillation=False → bit-identical to legacy.
        if config.get("use_self_distillation", False):
            try:
                sd_conf_thr = float(config.get("self_distill_conf_threshold", 0.9))
                sd_budget = int(config.get("self_distill_budget", 2400))
                sd_weight = float(config.get("self_distill_weight", 0.5))

                with torch.no_grad():
                    sd_probs = torch.softmax(logits, dim=1).detach().cpu()
                max_probs, argmax_preds = sd_probs.max(dim=1)
                max_probs_np = max_probs.numpy()
                argmax_np = argmax_preds.numpy()

                num_nodes_sd = data.x.size(0)
                labeled_set = {int(nid) for nid in labels.keys() if 0 <= int(nid) < num_nodes_sd}

                # Candidate mask: unlabeled AND confidence > threshold
                conf_ok = max_probs_np > sd_conf_thr
                candidate_ids = [
                    i for i in range(num_nodes_sd)
                    if conf_ok[i] and i not in labeled_set
                ]
                # Sort by confidence descending and take top-K
                candidate_ids.sort(key=lambda i: -float(max_probs_np[i]))
                sd_selected = candidate_ids[:sd_budget]
                n_pseudo = len(sd_selected)
                logger.info(
                    "Self-distillation: %d pseudo-labels selected "
                    "(conf_thr=%.2f, budget=%d, available=%d, weight=%.2f)",
                    n_pseudo, sd_conf_thr, sd_budget, len(candidate_ids), sd_weight,
                )

                if n_pseudo > 0:
                    # Build expanded labels dict (real labels override pseudo-labels
                    # by construction; candidate_ids excluded labeled nodes).
                    expanded_labels = dict(labels)
                    for nid in sd_selected:
                        expanded_labels[int(nid)] = int(argmax_np[nid])

                    # Set per-node weights: 1.0 for real LLM-annotated nodes,
                    # sd_weight for pseudo-labels. _train_gnn reads
                    # data._node_weights if present.
                    node_weights = torch.ones(num_nodes_sd, dtype=torch.float32)
                    for nid in sd_selected:
                        node_weights[int(nid)] = sd_weight
                    prev_node_weights = getattr(data, "_node_weights", None)
                    data._node_weights = node_weights.to(device)

                    try:
                        pipeline.gnn = pipeline._train_gnn(data, expanded_labels)
                        pipeline.gnn.eval()
                        with torch.no_grad():
                            logits = pipeline.gnn(data.to(device))
                        logger.info(
                            "Self-distillation: retrained GCN on %d labels (%d real + %d pseudo)",
                            len(expanded_labels), len(labels), n_pseudo,
                        )
                    finally:
                        # Restore prior node weights state (for downstream passes).
                        if prev_node_weights is None:
                            try:
                                del data._node_weights
                            except AttributeError:
                                pass
                        else:
                            data._node_weights = prev_node_weights
                else:
                    logger.info("Self-distillation: no pseudo-labels above threshold; skipping retrain.")
            except Exception as _sd_err:
                logger.warning("Self-distillation failed (%s) — continuing with existing logits", _sd_err)

        # Adaptive post-hoc LP smoothing: strength based on graph homophily
        # High homophily (PubMed) → stronger smoothing (GCN over-smooths noise)
        # Low homophily (Cora) → weaker smoothing (GCN message passing helps)
        if config.get("use_posthoc_lp", False):
            import scipy.sparse as sp_sparse
            num_n = data.x.size(0)
            src = data.edge_index[0].cpu().numpy()
            dst = data.edge_index[1].cpu().numpy()
            A = sp_sparse.csr_matrix(
                (np.ones(len(src)), (src, dst)), shape=(num_n, num_n))
            deg = np.array(A.sum(axis=1)).flatten()
            deg[deg == 0] = 1.0
            D_inv = sp_sparse.diags(1.0 / deg)
            A_norm = D_inv @ A
            probs = torch.softmax(logits.cpu(), dim=1).numpy()

            alpha_lp = 0.8  # trust GNN predictions
            n_iters = 5

            orig = probs.copy()
            for _ in range(n_iters):
                probs = alpha_lp * orig + (1 - alpha_lp) * (A_norm @ probs)
            logits = torch.from_numpy(probs).float().to(device)
            logger.info("Post-hoc LP: alpha=%.2f, %d iters", alpha_lp, n_iters)

        # Correct & Smooth (Huang et al., ICLR 2021) post-processing.
        # Two LP stages on top of the GNN predictions, anchored to
        # actively-selected annotations only. Large wins on arxiv-scale
        # (few labels, many classes) where GNN alone can't exploit graph
        # topology enough. Uses symmetric-normalized adjacency A_hat =
        # D^(-1/2)(A+I)D^(-1/2) and two tunable alphas.
        if config.get("use_cs_postprocess", False) and len(labels) > 0:
            import scipy.sparse as sp_sparse
            num_n = data.x.size(0)
            num_c = data.num_classes
            src = data.edge_index[0].cpu().numpy()
            dst = data.edge_index[1].cpu().numpy()
            A_sp = sp_sparse.csr_matrix(
                (np.ones(len(src), dtype=np.float32), (src, dst)),
                shape=(num_n, num_n),
            )
            A_sp = A_sp + A_sp.T
            A_sp.data = np.clip(A_sp.data, 0.0, 1.0)
            A_sp = A_sp + sp_sparse.eye(num_n, dtype=np.float32)
            deg = np.array(A_sp.sum(axis=1)).flatten()
            d_inv_sqrt = np.power(deg + 1e-8, -0.5).astype(np.float32)
            D_inv_sqrt = sp_sparse.diags(d_inv_sqrt)
            A_hat = (D_inv_sqrt @ A_sp @ D_inv_sqrt).astype(np.float32).tocsr()

            soft = torch.softmax(logits.detach().cpu(), dim=1).numpy().astype(np.float32)

            # Build ground-truth indicator only for actively-selected nodes.
            # cs_seeds_only: anchor C&S on the ORIGINAL LLM-annotated seeds
            # (~budget) instead of the self-trained 94% pool. C&S needs a small-
            # labelled / large-unlabelled regime to smooth INTO (LoCLE's setup);
            # our self-training fills the pool, leaving C&S nothing to propagate to.
            _cs_anchor = labels
            if config.get("cs_seeds_only", False):
                _seed_ids = set(int(k) for k in (getattr(pipeline, '_annotation_conf', None) or {}).keys())
                if _seed_ids:
                    _cs_anchor = {nid: lbl for nid, lbl in labels.items() if int(nid) in _seed_ids}
                    logger.info("cs_seeds_only: C&S anchored on %d original LLM seeds "
                                "(vs %d pool labels)", len(_cs_anchor), len(labels))
            Y_known = np.zeros_like(soft)
            known_mask = np.zeros(num_n, dtype=bool)
            for nid, lbl in _cs_anchor.items():
                if 0 <= int(nid) < num_n and 0 <= int(lbl) < num_c:
                    Y_known[int(nid), int(lbl)] = 1.0
                    known_mask[int(nid)] = True

            # Reliability-gated C&S: plain C&S pins ALL labelled nodes to their
            # (noisy) label and propagates it → on a noisy LLM pool this spreads
            # noise (cora −10pp). Gate the anchor set to cross-modality-RELIABLE
            # nodes: a node anchors C&S only if a fraction >= thresh of its
            # labelled graph-neighbours share its label. Noisy-labelled nodes are
            # demoted to "unlabelled" so the Smooth step pulls them toward their
            # neighbourhood instead of pinning the wrong label. Label-free; this
            # is the cross-modality estimator gating the propagation.
            if config.get("use_cs_reliability_gate", False):
                _thr = float(config.get("cs_reliability_thresh", 0.5))
                _ei = data.edge_index.cpu().numpy()
                _lab = -np.ones(num_n, dtype=np.int64)
                for nid, lbl in labels.items():
                    if 0 <= int(nid) < num_n:
                        _lab[int(nid)] = int(lbl)
                _gs = np.zeros(num_n); _gc = np.zeros(num_n)
                for s, t in zip(_ei[0], _ei[1]):
                    si, ti = int(s), int(t)
                    if _lab[si] >= 0 and _lab[ti] >= 0:
                        _gc[si] += 1.0
                        _gs[si] += 1.0 if _lab[ti] == _lab[si] else 0.0
                _agree = np.where(_gc > 0, _gs / np.maximum(_gc, 1), 1.0)
                # keep reliable anchors (high neighbour-agreement) OR nodes with
                # no labelled neighbours (no evidence against them).
                _reliable = (_agree >= _thr) | (_gc == 0)
                _before = int(known_mask.sum())
                drop = known_mask & (~_reliable)
                known_mask = known_mask & _reliable
                Y_known[drop] = 0.0
                logger.info("Reliability-gated C&S: kept %d/%d anchors (thresh=%.2f, "
                            "dropped %d noisy-label nodes)", int(known_mask.sum()),
                            _before, _thr, int(drop.sum()))

            # ── Correct step ───────────────────────────────────────────
            # Propagate residual errors on labeled nodes.
            corr_alpha = float(config.get("cs_corr_alpha", 0.8))
            corr_iters = int(config.get("cs_corr_iters", 50))
            E = np.zeros_like(soft)
            E[known_mask] = Y_known[known_mask] - soft[known_mask]
            E_init = E.copy()
            for _ in range(corr_iters):
                E = (1 - corr_alpha) * E_init + corr_alpha * (A_hat @ E)
            # Autoscale (Huang et al. §3.2): sigma so that sum |E_hat[L]|
            # matches sum |E_init[L]|.
            num_sigma = float(np.abs(E_init[known_mask]).sum())
            denom_sigma = float(np.abs(E[known_mask]).sum()) + 1e-8
            sigma = num_sigma / denom_sigma
            E = sigma * E
            corrected = soft + E
            corrected = np.clip(corrected, 0.0, 1.0)
            row_sum = corrected.sum(axis=1, keepdims=True)
            corrected = corrected / (row_sum + 1e-8)

            # ── Smooth step ────────────────────────────────────────────
            # Pin labeled nodes to one-hot; propagate refined soft labels
            # to unlabeled nodes via graph structure.
            smooth_alpha = float(config.get("cs_smooth_alpha", 0.8))
            smooth_iters = int(config.get("cs_smooth_iters", 50))
            # cs_use_gcn_prior: if True (default), unlabeled nodes seed from
            # corrected GCN softmax; if False, they seed from zero (pure LP
            # from labeled anchors only — useful when the GCN is weak, e.g.
            # at high class count like ogbn-arxiv C=40).
            use_gcn_prior = bool(config.get("cs_use_gcn_prior", True))
            if use_gcn_prior:
                Z = corrected.copy()
            else:
                Z = np.zeros_like(corrected)
            Z[known_mask] = Y_known[known_mask]
            Z_init = Z.copy()
            for _ in range(smooth_iters):
                Z = (1 - smooth_alpha) * Z_init + smooth_alpha * (A_hat @ Z)
            logits = torch.from_numpy(Z).float().to(device)
            logger.info(
                "C&S: corr(alpha=%.2f, iters=%d, sigma=%.3f) + smooth(alpha=%.2f, iters=%d, "
                "gcn_prior=%s); labeled=%d",
                corr_alpha, corr_iters, sigma, smooth_alpha, smooth_iters,
                use_gcn_prior, int(known_mask.sum()),
            )

        # Build labeled mask
        num_nodes = data.x.size(0)
        labeled_mask = torch.zeros(num_nodes, dtype=torch.bool)
        for nid in labels:
            labeled_mask[nid] = True

        # DIAG (env-guarded): dump the final pool labels (node_id -> assigned
        # label) for per-cluster purity analysis. Off by default.
        if os.environ.get("CANE_DUMP_POOL"):
            import json as _jp
            _jp.dump({int(n): int(l) for n, l in labels.items()},
                     open(os.environ["CANE_DUMP_POOL"], "w"))
            logger.info("CANE_DUMP_POOL: wrote %d pool labels", len(labels))

        # Evaluate on ALL nodes (LoCLE protocol)
        all_metrics = evaluate_experiment(logits, data.y, labeled_mask=None)
        results.update({
            "accuracy": all_metrics["accuracy"],
            "macro_f1": all_metrics["macro_f1"],
            "nmi": all_metrics["nmi"],
            "ari": all_metrics["ari"],
        })

        # === ENSEMBLE SUPPORT: dump per-node softmax + train_mask ===
        if config.get("dump_predictions", False):
            import numpy as _np
            pred_dir = config.get("pred_dir", osp.join(REPO_ROOT, "results", "np_predictions"))
            os.makedirs(pred_dir, exist_ok=True)
            tag = f"{dataset_name}_{backbone}_s{seed}"
            sm_np = torch.softmax(logits.detach().cpu(), dim=1).numpy().astype(_np.float32)
            train_np = labeled_mask.detach().cpu().numpy().astype(bool)
            _np.savez(osp.join(pred_dir, f"np_{tag}.npz"), softmax=sm_np, train_mask=train_np)
            logger.info("Dumped predictions: %s", osp.join(pred_dir, f"np_{tag}.npz"))

        # apr15 v89: secondary eval on NON-SELECTED nodes (~|V| - budget).
        # LLM-GNN reports their "test acc" as argmax accuracy on all nodes
        # not in the selected/train pool. For arxiv (|V|≈169K, budget=800)
        # this is ~168.5K nodes. Directly comparable to LLM-GNN's 66.32.
        try:
            non_selected_mask = ~labeled_mask
            n_non_sel = int(non_selected_mask.sum().item())
            if n_non_sel > 0:
                from sklearn.metrics import f1_score as _f1_ns
                preds_ns = logits.detach().cpu().argmax(dim=1).numpy()
                y_ns = data.y.cpu().numpy().reshape(-1)
                mask_np = non_selected_mask.cpu().numpy()
                acc_ns = float((preds_ns[mask_np] == y_ns[mask_np]).mean())
                f1_ns = float(_f1_ns(
                    y_ns[mask_np], preds_ns[mask_np],
                    average="macro", zero_division=0,
                ))
                results["accuracy_non_selected"] = acc_ns
                results["macro_f1_non_selected"] = f1_ns
                results["n_non_selected"] = n_non_sel
                logger.info(
                    "RESULTS (non_selected, n=%d): acc=%.4f (%.2f%%), f1=%.4f (%.2f%%)",
                    n_non_sel, acc_ns, acc_ns * 100, f1_ns, f1_ns * 100,
                )
        except Exception as _e:
            logger.warning("Non-selected eval skipped: %s", _e)

        # Secondary eval for OGB-native datasets (arxiv/products): evaluate on
        # the official test split so numbers are directly comparable to
        # LLM-GNN / OGB leaderboard. Primary metric stays all-nodes.
        if dataset_name.lower() == "arxiv":
            try:
                import warnings as _warn
                _warn.filterwarnings("ignore")
                from ogb.nodeproppred import PygNodePropPredDataset
                ogb_ds = PygNodePropPredDataset(
                    "ogbn-arxiv",
                    root=osp.join(REPO_ROOT, "data", "ogb"),
                )
                split = ogb_ds.get_idx_split()
                from sklearn.metrics import f1_score as _f1
                preds_np = logits.detach().cpu().argmax(dim=1).numpy()
                y_np = data.y.cpu().numpy().reshape(-1)
                for split_name, idx_tensor in (
                    ("ogb_test", split["test"]),
                    ("ogb_valid", split["valid"]),
                ):
                    idx = idx_tensor.numpy().reshape(-1)
                    acc = float((preds_np[idx] == y_np[idx]).mean())
                    f1 = float(_f1(y_np[idx], preds_np[idx], average="macro", zero_division=0))
                    results[f"accuracy_{split_name}"] = acc
                    results[f"macro_f1_{split_name}"] = f1
                    logger.info(
                        "RESULTS (%s, n=%d): acc=%.4f (%.2f%%), f1=%.4f (%.2f%%)",
                        split_name, len(idx), acc, acc * 100, f1, f1 * 100,
                    )
            except Exception as _e:
                logger.warning("OGB split eval skipped: %s", _e)
        
        # Compute improvement over LoCLE
        improvement = compute_improvement(all_metrics, dataset_name, backbone)
        results.update(improvement)
        
        logger.info("RESULTS: acc=%.4f (%.2f%%), f1=%.4f (%.2f%%), nmi=%.4f, ari=%.4f",
                    all_metrics["accuracy"], all_metrics["accuracy"] * 100,
                    all_metrics["macro_f1"], all_metrics["macro_f1"] * 100,
                    all_metrics["nmi"], all_metrics["ari"])
        
        if improvement:
            logger.info("vs LoCLE: acc %+.2f%%, f1 %+.2f%%, nmi %+.2f%%, ari %+.2f%%",
                       improvement.get("acc_delta", 0),
                       improvement.get("f1_delta", 0),
                       improvement.get("nmi_delta", 0),
                       improvement.get("ari_delta", 0))
    else:
        results.update({"accuracy": 0.0, "macro_f1": 0.0, "nmi": 0.0, "ari": 0.0})
    
    # Add noise matrix info
    if noise_matrix is not None:
        results["noise_matrix_diag_mean"] = float(np.diag(noise_matrix).mean())
        results["noise_matrix_diag_min"] = float(np.diag(noise_matrix).min())
    
    # Per-round info
    results["round_details"] = metrics.get("rounds", [])
    
    logger.info("Experiment complete in %.1fs", elapsed)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run a single NoisePref experiment")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--output", default=None, help="Output results JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Setup logging
    log_file = args.output.replace(".json", ".log") if args.output else None
    setup_logging(log_file)
    
    if args.dry_run:
        logger.info("DRY RUN — config loaded:")
        logger.info(json.dumps(config, indent=2))
        return
    
    # Run experiment
    try:
        results = run_experiment(config)
    except Exception as e:
        logger.error("EXPERIMENT FAILED: %s", e)
        logger.error(traceback.format_exc())
        results = {
            "dataset": config.get("dataset", "unknown"),
            "backbone": config.get("backbone", "unknown"),
            "seed": config.get("seed", -1),
            "experiment_tag": config.get("experiment_tag", "unknown"),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "nmi": 0.0,
            "ari": 0.0,
        }
    
    # Save results
    output_path = args.output
    if not output_path:
        tag = config.get("experiment_tag", "exp")
        ds = config.get("dataset", "unk")
        bb = config.get("backbone", "unk")
        s = config.get("seed", 0)
        output_path = osp.join(SCRIPT_DIR, "results", f"{tag}_{ds}_{bb}_s{s}.json")
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Results saved to %s", output_path)
    
    # Print key metrics to stdout for easy parsing
    print(f"RESULT_ACC:{results.get('accuracy', 0.0):.6f}")
    print(f"RESULT_F1:{results.get('macro_f1', 0.0):.6f}")
    print(f"RESULT_NMI:{results.get('nmi', 0.0):.6f}")
    print(f"RESULT_ARI:{results.get('ari', 0.0):.6f}")
    print(f"RESULT_ACC_DELTA:{results.get('acc_delta', 0.0):.4f}")


if __name__ == "__main__":
    main()
