"""
NoisePref iterative pipeline.

Chains:  probe → select → annotate → train GNN → detect critical →
         re-annotate → partition → update labels → (optional: preference-train
         local LLM) → (optional: rewire graph) → repeat.

Budget flow (managed by NoiseBudgetManager):
    Stage 0  — Noise probing      (probing_fraction of total, default 10%)
    Stage I  — Initial annotation  (initial_fraction of annotation budget, default 50%)
    Stage II — Iterative rounds    (remainder split evenly across rounds)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from src.models.gcn import GCN
from src.models.gat import GAT
from src.noise_handling.budget_manager import NoiseBudgetManager
from src.noise_handling.noise_profiler import LLMNoiseProfiler
from src.noise_handling.pseudo_samples import PseudoSampleConstructor
from src.node_selection.noise_aware_selector import NoiseAwareSelector

logger = logging.getLogger(__name__)






class NoisePrefPipeline:
    """
    Full NoisePref pipeline orchestrator.

    Args:
        annotator:          LLM annotator instance.
        dataset_name:       Name of the dataset (e.g. "cora").
        class_names:        List of class name strings.
        total_budget:       Total LLM query budget (e.g. 20 * num_classes).
        probing_fraction:   Fraction of budget for Stage 0 noise probing.
        initial_fraction:   Fraction of annotation budget for Stage I.
        total_rounds:       Total rounds (1 initial + N refinement).
        n_probes_per_class: Desired probes per class for noise estimation.
        cache_dir:          Directory for caching annotations/noise matrices.
        seed:               Random seed for reproducibility.
        gnn_hidden_dim:     GNN hidden layer dimension.
        gnn_epochs:         GNN training epochs per round.
        gnn_lr:             GNN learning rate.
        use_local_llm:      Whether to use local LLM preference training.
        warmup_rounds:      Rounds before starting preference training.
        use_noise_weighting: Whether to use noise matrix for weighting (False = uniform).
        use_rewiring:       Whether to apply Dirichlet graph rewiring.
        rewiring_ratio:     Max fraction of edges to prune per round (default 0.05).
        use_orpo:           Whether to use ORPO preference loss (False = SFT-only).
        lora_rank:          LoRA rank for preference trainer.
        device:             Torch device ('cuda', 'cpu', or None for auto).
    """

    def __init__(
        self,
        annotator,
        dataset_name: str,
        class_names: List[str],
        total_budget: int,
        probing_fraction: float = 0.1,
        initial_fraction: float = 0.5,
        total_rounds: int = 5,
        n_probes_per_class: int = 20,
        cache_dir: str = "data/annotations/",
        seed: int = 0,
        gnn_hidden_dim: int = 64,
        gnn_epochs: int = 200,
        gnn_lr: float = 0.01,
        use_local_llm: bool = False,
        local_llm_model: str = "Qwen/Qwen2.5-3B",
        warmup_rounds: int = 2,
        use_noise_weighting: bool = True,
        use_rewiring: bool = False,
        rewiring_ratio: float = 0.05,
        use_orpo: bool = True,
        lora_rank: int = 16,
        backbone: str = "gcn",
        gnn_dropout: float = 0.5,
        lp_alpha: float = 0.5,
        lp_iters: int = 50,
        label_smoothing: float = 0.0,
        use_gce: bool = False,
        gce_q: float = 0.7,
        use_noise_ada: bool = False,
        device: Optional[str] = None,
        **kwargs,
    ):
        self.annotator = annotator
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.backbone = backbone.lower()
        self.gnn_dropout = gnn_dropout
        self.total_rounds = total_rounds
        self.initial_fraction = initial_fraction
        self.n_probes_per_class = n_probes_per_class
        self.seed = seed
        self.gnn_hidden_dim = gnn_hidden_dim
        self.gnn_epochs = gnn_epochs
        self.gnn_lr = gnn_lr
        self.gnn_num_layers = int(kwargs.get('gnn_num_layers', 2))
        self.gnn_weight_decay = float(kwargs.get('gnn_weight_decay', 5e-4))
        self.lp_alpha = lp_alpha
        self.lp_iters = lp_iters
        self.label_smoothing = label_smoothing

        # Entropy regularization (NegEntropy from LLM-GNN)
        self.entropy_reg = kwargs.get('entropy_reg', 0.0)
        # Forward loss correction (Patrini et al., CVPR 2017)
        self.use_forward_correction = kwargs.get('use_forward_correction', False)
        # Feature propagation (SGC/SIGN-style pre-processing)
        # Prediction-based edge reweighting
        # NAAP: Noise-Aware Anisotropic Propagation
        self.use_naap = kwargs.get('use_naap', False)
        # may18 v6: optionally pass pre-trained GraphMAE2 embeddings into NAAP's
        # subspace clustering instead of the raw SBERT (data.x) features.
        # GraphMAE2 features are graph-aware (encoder pre-trained on the full
        # unlabeled graph via masked-feature reconstruction). Expected to yield
        # higher cluster purity for NAAP's selection step.
        self.naap_use_graphmae_features = bool(kwargs.get('naap_use_graphmae_features', False))
        self.graphmae_features_path = kwargs.get('graphmae_features_path', None)
        # may18 v6: Active Quality Probing (AQP) — verify a small number of
        # pseudo-labels via LLM after self-training, compute per-cluster
        # agreement rate, drop pool labels in low-agreement clusters before ILC.
        # may18 v6: AQP action — 'hard_drop' (delete pool labels in low-q clusters)
        # or 'soft_weight' (multiply per-node GNN loss weight by q[cluster]).
        # soft_weight preserves pool size, avoids pool-inflation loss.
        # may18 v7: Consistency Regularization on non-selected nodes (UDA/FixMatch).
        # Adds KL(softmax(GNN_aug) || softmax(GNN_orig).detach()) for non-train
        # nodes. Avoids the pool-inflation trade-off — no pool labels are
        # touched; gain comes from new training signal on unlabeled nodes.
        self.use_consistency_reg = bool(kwargs.get('use_consistency_reg', False))
        self.consistency_lambda = float(kwargs.get('consistency_lambda', 1.0))
        self.consistency_aug_dropedge = float(kwargs.get('consistency_aug_dropedge', 0.5))
        self.consistency_aug_feat_mask = float(kwargs.get('consistency_aug_feat_mask', 0.1))
        self.consistency_warmup_epochs = int(kwargs.get('consistency_warmup_epochs', 50))
        # may18 v7b: FixMatch-style confidence threshold — only enforce
        # consistency where teacher's max softmax >= conf_threshold. Avoids
        # the echo-chamber on low-confidence teacher predictions.
        self.consistency_conf_threshold = float(kwargs.get('consistency_conf_threshold', 0.0))
        # may18 v8: T_c-aware self-training expansion gate. Modifies the GNN
        # confidence THRESHOLD (not the score) per-cluster: noisy clusters
        # need higher confidence to be expanded into. Targets pool quality.
        self.use_tc_expansion_gate = bool(kwargs.get('use_tc_expansion_gate', False))
        self.tc_expansion_gate_alpha = float(kwargs.get('tc_expansion_gate_alpha', 0.3))
        # may18 v6: self-training pool expansion cap (re-use existing knob).
        # When set, restrict per-class pseudo-label additions per round.
        # Smaller cap → smaller pool, but each label is higher confidence.
        self.naap_alpha_base = kwargs.get('naap_alpha_base', 0.3)
        self.naap_beta = kwargs.get('naap_beta', 0.8)
        self.naap_gamma = kwargs.get('naap_gamma', 0.6)
        self.naap_k_max = kwargs.get('naap_k_max', 30)
        self.naap_patience = kwargs.get('naap_patience', 3)

        # SGR: Simplified Graph Refinement
        # NCL: Noise-Aware Confident Learning

        # DMCR: Dual-Model Co-Refinement (novel contribution)
        # Soft Label Distillation: train GNN on NAAP/LP soft labels (KL divergence)
        # NAEW: Noise-Aware Edge Weighting for GNN training
        # ILC: Iterative Label Correction
        self.use_ilc = kwargs.get('use_ilc', False)
        self.ilc_conf_threshold = kwargs.get('ilc_conf_threshold', 0.9)
        self.ilc_max_rounds = kwargs.get('ilc_max_rounds', 5)
        # may18 v4: integrated verify-and-drop inside ILC. When enabled, each
        # ILC round produces a three-way decision per label: KEEP (graph
        # endorses), CORRECT (LoCLE-style, graph suggests another class), or
        # DROP (graph cannot defend the current label at any reasonable
        # confidence). Drops are removed from the label pool and become
        # permanent — the iteration converges when both correct-count and
        # drop-count are zero.
        self.use_verify_and_drop_ilc = bool(kwargs.get('use_verify_and_drop_ilc', False))
        # Graph endorses the current label if its softmax probability AT THAT
        # label is at least this value. Below this, the label is eligible
        # for either CORRECT (if graph confidently suggests another) or DROP.
        self.verify_drop_threshold = float(kwargs.get('verify_drop_threshold', 0.3))
        # Phase 24: cluster-conditional ILC threshold — must be persisted on
        # self so it survives across method calls (was previously only read
        # via getattr-default, silently ignoring config).
        self.ilc_use_cluster_T_c = bool(kwargs.get('ilc_use_cluster_T_c', False))
        self.ilc_tc_scale = float(kwargs.get('ilc_tc_scale', 0.5))
        # may22: cross-modality reliability weighting (label-free noise estimate).
        # Per-node CE weight w_i = 1 - g*(1-agree_i), agree_i = frac of labeled
        # graph-neighbours whose label == node's label; g = clip(alpha*(1-mean_agree),
        # 0, 1) is a global label-free noise gate (clean datasets -> g~0 -> no-op).
        self.use_xmod_weight = bool(kwargs.get('use_xmod_weight', False))
        self.xmod_alpha = float(kwargs.get('xmod_alpha', 2.0))
        # may23: dual-view density-adaptive cross-modality. Adds a feature-kNN
        # view to the graph-neighbour view, blended per-node by evidence count
        # (#labeled neighbours in each view) -> sparse-degree nodes lean on
        # features, dense nodes lean on the graph. Label-free; k fixed a priori.
        self.use_dual_xmod = bool(kwargs.get('use_dual_xmod', False))
        self.dual_xmod_k = int(kwargs.get("dual_xmod_k", 10))
        self.dual_xmod_fallback = bool(kwargs.get("dual_xmod_fallback", False))
        self._feat_knn = None  # cached feature-kNN indices [N, k]
        # may23 TVCR: two-view confident relabeling. When a pool node's graph-neighbour
        # majority AND its feature-kNN majority agree on a class != its LLM label, relabel
        # it to that consensus. Two-view intersection recovers GT 66-92% (estimator_quality.py)
        # — a corrector, not just a down-weight; label-free, coverage-preserving.
        self.use_xmod_relabel = bool(kwargs.get('use_xmod_relabel', False))
        self.xmod_relabel_min_graph = int(kwargs.get('xmod_relabel_min_graph', 2))
        # may23 RGS: reliability-guided selection. After round 1, compute per-cluster
        # cross-modality reliability from seed labels (label-free); in Stage-I rounds
        # 2-3 bias candidate scoring toward reliable clusters (clean LLM supervision).
        # Cluster-conditional-noise thesis applied to SELECTION. floor keeps every
        # cluster eligible; beta blends reliability into the entropy score.
        self.use_reliability_selection = bool(kwargs.get('use_reliability_selection', False))
        self.reliability_select_floor = float(kwargs.get('reliability_select_floor', 0.3))
        self.reliability_select_beta = float(kwargs.get('reliability_select_beta', 0.5))
        # may18 v5: per-ILC-round T_c update from current labels (label-free).
        self.ilc_update_T_c_per_round = bool(kwargs.get('ilc_update_T_c_per_round', False))
        self.ilc_tc_update_method = str(kwargs.get('ilc_tc_update_method', 'v3'))
        self.ilc_neighbor_threshold = kwargs.get('ilc_neighbor_threshold', 0.5)
        # Multi-seed GNN ensemble for final prediction
        self.use_ensemble = kwargs.get('use_ensemble', False)
        # Bootstrap loss (Reed et al., ICLR 2015)
        # ELR loss (Liu et al., NeurIPS 2020). Strictly better than Bootstrap
        # on 4/5 datasets in our benchmarks (see experiments_verification/).
        self.use_elr = kwargs.get('use_elr', False)
        self.elr_beta = kwargs.get('elr_beta', 0.7)
        self.elr_lambda = kwargs.get('elr_lambda', 3.0)
        # Co-teaching denoising (Han et al., NeurIPS 2018). Best or tied-best
        # in our benchmarks on 4/5 datasets, replaces GMM pruning.
        # Self-label multiplier: confident self-labeling per round = multiplier × round budget.
        # LoCLE self-labels ~3× budget → 120 per round on Cora. Default 3 matches LoCLE.
        self.self_label_multiplier = kwargs.get('self_label_multiplier', 3)
        # Graph Contrastive Regularization (GCA, WWW 2021)
        self.contrastive_weight = kwargs.get('contrastive_weight', 0.0)
        # Post-self-training EstimateAdj
        # Loss-based label pruning (DivideMix-inspired)
        self.use_loss_pruning = kwargs.get('use_loss_pruning', False)
        # Zero DropEdge for final GNN training

        # DropEdge: random edge dropout during GNN training (ICLR 2020)
        self.dropedge_rate = kwargs.get('dropedge_rate', 0.0)

        # Full-graph denoising parameters
        self.use_neighborhood_correction = kwargs.get('use_neighborhood_correction', False)
        self.neighborhood_correction_threshold = kwargs.get('neighborhood_correction_threshold', 0.5)
        self.self_training_rounds = kwargs.get('self_training_rounds', 3)
        # may18 v4: cap on pseudo-labels added per round, per class. When > 0,
        # limits each self-training round to TOP-K most-confident new
        # pseudo-labels per class. Keeps the training pool small and high
        # quality; intended to compose well with noise-robust losses (ELR).
        self.self_train_cap_per_class = int(kwargs.get('self_train_cap_per_class', 0))
        # may18 v5: cluster-conditional T_c weighting of self-training scores
        self.use_tc_self_training = bool(kwargs.get('use_tc_self_training', False))
        # may18 v5: drive pseudo-labeling from T_c-aware LP instead of GNN+LP consensus
        self.use_lp_pseudo_labels = bool(kwargs.get('use_lp_pseudo_labels', False))

        # Active learning with local LLM (NEW — exp180+)
        self.use_active_learning = kwargs.get('use_active_learning', False)
        # Path 1 (apr13): opt-in LoCLE iterative refinement (wholesale port).
        # Replaces our self-training loop with LoCLE's multi-stage refinement.
        # See src/training/locle_iterative_path.py.
        # may17: opt-in LoCLE GCN/GAT backbone for _train_gnn (pre-conv dropout,
        # Adam lr=0.01 wd=5e-4, 30 epochs, label-free). See
        # src/models/locle_gcn_backbone.py.
        self.use_locle_gnn_backbone = bool(kwargs.get('use_locle_gnn_backbone', False))
        self.locle_gnn_epochs = int(kwargs.get('locle_gnn_epochs', 30))
        # LoCLE verbatim rewire (Rewire_GNN + EstimateAdj)
        # Placement: 'pre_st' = before self-training (labels are fresh LLM seeds, matches LoCLE)
        #            'end'    = after self-training + ILC (original placement; rewire_gcn fits noisy self-labels)
        self.use_locle_rewire = kwargs.get('use_locle_rewire', False)
        self.locle_rewire_placement = kwargs.get('locle_rewire_placement', 'pre_st')
        # Phase 21: per-dataset rewire hyperparameter overrides (mirrors LoCLE's per-dataset configs)
        self.locle_rewire_overrides = kwargs.get('locle_rewire_overrides', None)
        # Stage-I budget splits. Default 40% subspace + 60% uncertainty; [1.0] = LoCLE-style.
        self.stage1_splits = kwargs.get('stage1_splits', None)
        # stage1_all_subspace: every Stage-I round selects via subspace clustering on
        # the current GNN embeddings (LoCLE's per-stage refinement), not entropy.
        self.stage1_all_subspace = bool(kwargs.get('stage1_all_subspace', False))
        # Noise-Aware EstimateAdj (NA-EA): noise-weighted variant of LoCLE rewire.
        # Mutually exclusive with use_locle_rewire (NA-EA wins if both set).
        self.use_na_ea = kwargs.get('use_na_ea', False)
        # NA-EA v2: three-fix variant (plain Dirichlet, 0.30 floor, no conf).
        # Mutually exclusive with use_na_ea and use_locle_rewire (v2 wins).
        self.use_na_ea_v2 = kwargs.get('use_na_ea_v2', False)
        self.active_rounds = kwargs.get('active_rounds', 5)

        # NoiseAwareSelector knobs (apr13 opt12 sweep)
        self.select_downweight_floor = float(kwargs.get('select_downweight_floor', 0.0))
        self.select_min_class_frac = float(kwargs.get('select_min_class_frac', 0.0))

        # PS-FeatProp-W (LLM-GNN pagerank2): single-round selection via
        # alpha*density(AAX) + (1-alpha)*pagerank. Overrides NoiseAwareSelector
        # in Stage-I when True. Built for arxiv-scale graphs.
        # When use_psfeatprop_selection is True, choose between:
        #   False (default): density is k-means on AAX (2-hop propagated X) —
        #     matches our paper's v4 arxiv config (paper's "PS-FeatProp-W").
        #   True: density is k-means on RAW X (no propagation) — exactly
        #     matches LLM-GNN's `pg2_query` / `density+pagerank` strategy,
        #     cache key `density_x_{N}_{B}.pt`. Use this to reproduce
        #     LLM-GNN's 66.32 arxiv baseline.
        # Stratified per-pseudo-class selection: after building per-node scores,
        # partition nodes via feature-space k-means (k=num_classes), then select
        # ceil(budget/num_classes) top-scoring nodes per pseudo-cluster. This
        # guarantees coverage of all pseudo-classes (fixes arxiv 40-class
        # collapse where vanilla top-K leaves 17/40 classes with zero anchors).
        # Default False → bit-identical to current top-K behaviour.
        # Drop bottom-quantile-confidence annotations after LLM labelling
        # (LLM-GNN "-W" filter). 0.0 = disabled.
        # Alternative Stage-I selection strategies (apr14). Both opt-in; kcenter
        # wins if both True. Mutually exclusive with use_psfeatprop_selection.
        # A) K-center greedy coreset selection on (L2-normalised) data.x.
        # B) age_query2 (LLM-GNN's 3-term: alpha*AAX-density + beta*pagerank +
        #    (1-alpha-beta)*raw-X-density). Defaults: alpha=0.2, beta=0.3.

        # apr15 v92: fixed-selection mode. When True, load a pre-computed list
        # of node IDs from `fixed_selection_path` (JSON list of ints) and use
        # them as the Stage-I active selection (bypass pagerank2 / k-center /
        # subspace-clustering / age_query entirely). Overrides all other
        # Stage-I selectors when True. Default False → bit-identical behavior.

        # C2 — Two-pass cluster-aware selection. Over-clusters into k = α·C
        # clusters, spends ~p1_fraction of the Stage-I budget on Pass-1
        # cluster representatives, then re-allocates Pass-2 budget across
        # clusters proportional to Pass-1 label entropy. Mutually exclusive
        # with the other alt selectors above; falls through the multi-round
        # subspace flow when False. Defaults preserve legacy behaviour.

        # NoiseSelect coreset injection for legacy Stage I round 1.
        # When True, replaces the existing NoiseAwareSelector.select_initial_nodes
        # call with NoiseSelectCoreset.select. Rounds 2-3 still run the legacy
        # GNN-uncertainty selection. Falls through to legacy behaviour when False.

        # ── apr15 gap-closer knobs (default: bit-identical to pre-apr15) ──
        # Held-out val split from the annotated labels for early stopping.
        # 0.0 → disabled (legacy behavior). Typical value: 0.1 (10% holdout).
        self.val_holdout_frac = float(kwargs.get('val_holdout_frac', 0.0))
        # Patience (in epochs) for early stopping on val_holdout. 0 → disabled.
        self.early_stop_patience = int(kwargs.get('early_stop_patience', 20))
        # How often to evaluate val accuracy during training.
        self.val_check_every = int(kwargs.get('val_check_every', 1))
        # Add held-out val nodes back to the trained labels for final eval.
        # Default False = conservative (drop them); True = reuse as LLM anchors.
        # Number of final ensemble GCNs. 1 = legacy behavior (single model).
        # When >1, experiment_runner trains additional GCNs with offset seeds
        # on the pipeline's final labels and averages softmax.
        self.final_ensemble_n = int(kwargs.get('final_ensemble_n', 1))
        # may17 tricks: inference-time tricks (independent of training).
        self.use_mc_dropout = bool(kwargs.get('use_mc_dropout', False))
        self.mc_dropout_passes = int(kwargs.get('mc_dropout_passes', 10))
        self.use_self_distill_final = bool(kwargs.get('use_self_distill_final', False))
        self.self_distill_n = int(kwargs.get('self_distill_n', 200))
        self.self_distill_conf = float(kwargs.get('self_distill_conf', 0.85))
        # may17 trick 5: clean-subset retraining (drop noisiest labels, retrain).
        self.clean_subset_drop_frac = float(kwargs.get('clean_subset_drop_frac', 0.30))
        # may18 v2 knobs: iterated clean-subset + class-balanced drop
        self.clean_subset_iterations = int(kwargs.get('clean_subset_iterations', 1))
        self.clean_subset_iter_drop_frac = float(kwargs.get('clean_subset_iter_drop_frac', 0.30))
        self.clean_subset_class_balanced = bool(kwargs.get('clean_subset_class_balanced', False))
        # may18 v3: filter-strength ensemble. If a list of drop fractions is
        # provided, train one ensemble per drop frac and average their logits.
        # Overrides clean_subset_drop_frac when non-empty.
        _df_list = kwargs.get('clean_subset_drop_fracs', None)
        self.clean_subset_drop_fracs = (
            [float(x) for x in _df_list] if _df_list else None)
        # may18 v3: ELR applied ONLY during the clean-subset retrain step.
        # Different from use_elr (which applies to the full pipeline).
        # may18 v3: calibration diagnostic dump path.
        self.diag_dump_path = kwargs.get('diag_dump_path', None)
        # may18 v3: noise-detection signal switch for ablation
        self.clean_subset_signal = str(kwargs.get('clean_subset_signal', 'ensemble'))
        # apr15: voting-confidence-weighted training.
        # When True, use per-node vote confidence (fraction of majority votes for
        # n_samples>1, verbalized conf for n_samples=1) as per-node CE loss weights.
        # Default False → bit-identical to legacy (uniform CE loss).
        self.use_vote_confidence_weights = bool(
            kwargs.get('use_vote_confidence_weights', False))
        # Cluster-conditional CCR: additionally multiply per-node weight by
        # T_c[cluster, label, label] ** cluster_conf_pow (cluster-side analogue
        # of CCR). Mean-renormalised so total loss magnitude is preserved.
        # Requires use_cluster_T_c=True with T_c + cluster_id loaded.
        self.use_cluster_conf_weights = bool(
            kwargs.get('use_cluster_conf_weights', False))
        self.cluster_conf_pow = float(kwargs.get('cluster_conf_pow', 1.0))
        # CRBA — Cluster-Risk-Based Budget Allocation for Stage-I round 2-3
        # selection. score = (1-λ)·entropy + λ·posterior_risk_under_T_c.
        # 0 disables. Active only when use_cluster_T_c is True.
        self.crba_lambda = float(kwargs.get('crba_lambda', 0.0))
        # Per-cluster quota cap (proportional to cluster size). 0 disables.
        # 1.0 = strict proportional; >1.0 = looser; <1.0 = tighter. LoCLE used
        # 1.0 in its working dblp +0.49pp result.
        self.crba_quota = float(kwargs.get('crba_quota', 0.0))
        # LoCLE-style iterative re-annotation (post-Stage-I, live LLM).
        # LoCLE rank-and-correct: after re-annotation, replace LLM label
        # with GNN argmax for nodes where GNN ranks higher than LLM in
        # confidence within the batch, OR LLM confidence < threshold.
        # TBLC: T_c Bayesian Label Correction (cluster-cond, no-extra-budget)
        # Probing mode: "template" (default Stage 0 with templates) or
        # "post_stageI" (defer to after Stage I, probe with real anchors).
        self.probing_mode = str(kwargs.get('probing_mode', 'template'))
        # Iterative self-labelling: each stage adds the top-frac most-confident
        # unlabelled nodes as PSEUDO-labels (no API cost), mirroring LoCLE's
        # confident-set behaviour. 0 disables.
        # Schedule for mapping confidence → weight.
        #   "linear": w = conf (conf already in [0,1]; 0.33 → 0.33, 1.0 → 1.0)
        #   "sharp":  w ∈ {0.3, 0.7, 1.0} bucketed on 33/67/100 (or 80/90/100)
        self.vote_conf_schedule = str(kwargs.get('vote_conf_schedule', 'linear'))

        # apr15 v89: OGB validation-set early stopping for ogbn-arxiv.
        # When True AND data has `ogb_val_idx` attribute, _train_gnn uses
        # those indices (~29.8K OGB val nodes) with their GOLD labels (data.y)
        # as the early-stopping signal. The gold labels are ONLY used to
        # monitor val accuracy — they are NEVER added to the training loss.
        # This matches LLM-GNN's arxiv training protocol (OGB val early-stop
        # during GCN training, patience=20). Default False → bit-identical.

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Budget manager tracks all LLM query consumption
        self.budget = NoiseBudgetManager(
            total_budget=total_budget,
            num_classes=self.num_classes,
            probing_fraction=probing_fraction,
        )

        # Noise profiler for Stage 0
        self.profiler = LLMNoiseProfiler(
            annotator=annotator,
            cache_dir=cache_dir,
        )

        # Will be populated by run()
        self.noise_matrix: Optional[np.ndarray] = None
        self.per_class_accuracy: Optional[np.ndarray] = None
        self.confused_pairs: Optional[list] = None

        # === Phase 17: cluster-conditional T_c (loaded if available) ===
        self.T_c: Optional[np.ndarray] = None              # [K, C, C]
        self.cluster_id: Optional[np.ndarray] = None       # [num_nodes]
        # Within-budget T_c: re-estimate the cluster-conditional matrix from ONLY
        # the LLM annotations obtained within the query budget (the selected seeds),
        # never the full annotation cache. Closes the all-node estimation leak.
        self._llm_annotated: Dict[int, int] = {}   # accumulates raw within-budget LLM seed labels
        # Cross-modality T_c: estimate per-cluster reliability from agreement between
        # a node's LLM label and its graph/feature-kNN neighbours (calibrated to true
        # accuracy even on impure clusters, unlike mode-concentration). Built from the
        # PROBE round only (round 0), so it can guide the remaining selection + the
        # expansion/correction gates without using the rest of the seed labels.
        self.use_crossmodal_tc = bool(kwargs.get('use_crossmodal_tc', False))
        # Ablation: collapse the cluster-conditional T_c to its cluster-average
        # (a class-conditional / global matrix) to test whether cluster-
        # conditioning matters.
        self.tc_global_average = bool(kwargs.get('tc_global_average', False))
        self._tc_from_probe_done = False
        # Opt-in scalable training for very large graphs (products/arxiv) that
        # OOM full-batch. Routes final training+inference through NeighborLoader.
        self.use_cluster_T_c = kwargs.get('use_cluster_T_c', False)
        self.cluster_T_c_path = kwargs.get('cluster_T_c_path', None)
        self.cluster_id_path = kwargs.get('cluster_id_path', None)
        if self.use_cluster_T_c and self.cluster_T_c_path and self.cluster_id_path:
            try:
                self.T_c = np.load(self.cluster_T_c_path).astype(np.float32)
                self.cluster_id = np.load(self.cluster_id_path).astype(np.int64)
                logger.info("Loaded cluster-conditional T_c: shape=%s, K=%d clusters from %s",
                             self.T_c.shape, self.T_c.shape[0], self.cluster_T_c_path)
            except Exception as _e_tc:
                logger.warning("Failed to load T_c (%s); falling back to class-conditional T", _e_tc)
                self.T_c = None
                self.cluster_id = None
        # RGS needs cluster ids even when T_c is off (use_cluster_T_c=False).
        if (self.cluster_id is None and self.cluster_id_path
                and getattr(self, 'use_reliability_selection', False)):
            try:
                self.cluster_id = np.load(self.cluster_id_path).astype(np.int64)
                logger.info("Loaded cluster ids for RGS (no T_c): K=%d from %s",
                            int(self.cluster_id.max()) + 1, self.cluster_id_path)
            except Exception as _e_cid:
                logger.warning("RGS: failed to load cluster ids (%s)", _e_cid)
                self.cluster_id = None

        # === may19 §6.A: derive global noise_matrix from cluster T_c ===
        # cctger_off declares use_forward_correction=true and use_naap=true, but
        # both gate on `self.noise_matrix is not None`. Without LLM probing
        # (probing_fraction=0.0), noise_matrix is never set and both stages
        # silently fall back to no-op behaviour. When the cluster-conditional
        # T_c is available, we can derive a global C×C transition matrix as the
        # uniform-weighted mean over clusters at zero LLM budget cost. Row-
        # normalise so each row is a valid conditional distribution.
        # Opt-in (default False) so the cctger_off baseline is unchanged unless
        # configs explicitly enable it (e.g. cctger_flcA_*.json).
        self.use_cluster_T_c_for_forward_correction = bool(
            kwargs.get('use_cluster_T_c_for_forward_correction', False)
        )

        # === T_c-calibrated soft-label loss (Experiment 2 / 2b, may19) ===
        # When True, ``_train_gnn`` overrides hard CE with KL-divergence
        # against a T_c-derived soft target. Two modes:
        #
        #   * "column" (default, Experiment 2b — Bayes-correct):
        #       target ∝ T_c[k_u, :, ŷ_u]  — posterior P(true=t | LLM=ŷ, k)
        #       with uniform prior. The mathematically correct soft label
        #       for "the LLM said ŷ; what is the true class actually likely
        #       to be?". Used with KL-div (gradient-equivalent to CE).
        #
        #   * "row" (Experiment 2 — empirically catastrophic):
        #       target = T_c[k_u, ŷ_u, :] — P(LLM_pred | true=ŷ, k). Trains
        #       the GNN to MIMIC the LLM's noise pattern; included only
        #       for reproducibility of the negative result.
        self.use_tc_soft_labels = bool(kwargs.get('use_tc_soft_labels', False))
        self.tc_soft_mode = str(kwargs.get('tc_soft_mode', 'column'))
        if self.tc_soft_mode not in ('row', 'column'):
            raise ValueError(
                f"tc_soft_mode must be 'row' or 'column', got {self.tc_soft_mode!r}"
            )

        # === Bayes Rectifier (Experiment 4, may19) ===
        # When True, every label returned by self.annotator passes through
        # a hard-flip rectifier: y' = argmax_t T_c[k_u, t, ŷ_u]. ILC- and
        # self-training-generated pseudo-labels are NOT touched — only
        # outputs of ``_annotate_nodes`` (i.e., raw LLM annotations).
        # Cluster-conditional T_c (use_cluster_T_c=True) is required.
        self.use_bayes_rectify = bool(kwargs.get('use_bayes_rectify', False))
        # Per-cell rectification counter for diagnostics.
        self._bayes_rectify_stats = {"total": 0, "flipped": 0}

        # Current GNN model (rebuilt each round)
        self.gnn: Optional[GCN] = None
        self._prev_gnn_state: Optional[dict] = None  # for warm-starting

        # === Static kNN + Interpolated Soft TGER edge rewiring ===
        # See src/structure_learning/tger.py. Mirrors LoCLE rewiring but
        # label-free: Phase 1 kNN-augments topology from GraphMAE2 embeddings
        # (cached), Phase 2 attaches asymmetric soft edge weights derived from
        # T_c + cluster_id + current pseudo-labels. GCN-only; GAT bypasses.
        self.use_tger = bool(kwargs.get('use_tger', False))
        self.tger_beta = float(kwargs.get('tger_beta', 0.6))
        self.tger_knn_k = int(kwargs.get('tger_knn_k', 5))
        self.tger_embedding_path = kwargs.get('tger_embedding_path', None)
        self._tger_emb: Optional[torch.Tensor] = None
        self._tger_aug_cache: Optional[torch.Tensor] = None

        # === SSAF: Semantic-Structural Adaptive Fusion (2026-05-19) ===
        # Dual-channel GNN architecture. Channel 1 message-passes on raw
        # `data.edge_index` (topological view). Channel 2 message-passes on
        # a k-NN graph built once from GraphMAE2 embeddings (semantic view).
        # A learnable per-node gate (sigmoid scalar) fuses the two hidden
        # representations before a shared classifier head. Loss path stays
        # identical to cctger_off (T_c forward correction, ILC, ELR, NAAP).
        self._ssaf_sem_edge_index: Optional[torch.Tensor] = None

        # === DIAGNOSTIC-only knobs (2026-05-19) — used to bracket the
        # LoCLE gold-leak; never enable in paper-grade runs. ===
        # Diag-1: inject true gold into Rewire_GNN's backup_y so that
        # LoCLE's leaky model-selection path activates inside our pipeline.
        self.diag_inject_locle_gold_backup_y = bool(
            kwargs.get('diag_inject_locle_gold_backup_y', False)
        )

        # === Cora-rescue ablation suite (2026-05-19) ===
        # `gnn_variant` swaps in a GCN-style backbone variant (gcnii_lite,
        # gcn_dualpath, gcn_jk, gated_gcn, appnp_head). When "gcn" or
        # "gat" (default), cctger_off behaviour is preserved bit-for-bit.
        self.gnn_variant = str(kwargs.get('gnn_variant', 'gcn')).lower()
        self._gnn_variant_extras = {
            'gcnii_alpha': float(kwargs.get('gcnii_alpha', 0.2)),
            'gcnii_beta': float(kwargs.get('gcnii_beta', 0.2)),
            'dualpath_fusion': str(kwargs.get('dualpath_fusion', 'static')),
            'dualpath_lambda': float(kwargs.get('dualpath_lambda', 0.7)),
            'jk_mode': str(kwargs.get('jk_mode', 'concat')),
            'gated_use_degree': bool(kwargs.get('gated_use_degree', True)),
            'appnp_K': int(kwargs.get('appnp_K', 10)),
            'appnp_alpha': float(kwargs.get('appnp_alpha', 0.1)),
        }
        # Degree-aware DropEdge: high-degree nodes drop at the configured
        # rate, low-degree nodes drop less. Disabled by default (uniform).
        self.dropedge_degree_aware = bool(kwargs.get('dropedge_degree_aware', False))

        # === Tuning Phase (2026-05-19) — cctger_off refinement knobs ===
        # (1) ``forward_correction_temperature`` — softmax-temperature scale
        #     applied to the noise transition matrix before forward
        #     correction. tau > 1 flattens T (less over-confident), tau < 1
        #     sharpens. Default 1.0 = bit-identical legacy behaviour.
        self.forward_correction_temperature = float(
            kwargs.get('forward_correction_temperature', 1.0)
        )
        # (2) ``use_low_conf_drop`` — at the start of each _train_gnn call,
        #     mask out nodes whose previous-round GNN max-prob is below
        #     ``low_conf_threshold`` (both as training-mask and as
        #     edge-source — their outgoing edges are zero-weighted).
        self.use_low_conf_drop = bool(kwargs.get('use_low_conf_drop', False))
        self.low_conf_threshold = float(kwargs.get('low_conf_threshold', 0.85))
        self._prev_gnn_probs: Optional[torch.Tensor] = None  # captured per round

        # === RTAA: Robust Topology-Aware Augmentation (2026-05-19) ===
        # Backbone-agnostic data-level mitigation. Two operations applied
        # before each _train_gnn round:
        #   (1) edge weights:   w_uv = T_c[k_u, ŷ_u, ŷ_v]
        #   (2) synthetic edges between high-confidence same-class nodes
        # Per-epoch synthetic refresh handled inside gcn._fit_supervised
        # via data._use_rtaa_dynamic flag (GCN backbone only); GAT runs the
        # per-round refresh only.
        self.use_rtaa = bool(kwargs.get('use_rtaa', False))
        self.rtaa_conf_threshold = float(kwargs.get('rtaa_conf_threshold', 0.90))
        self.rtaa_refresh_every = int(kwargs.get('rtaa_refresh_every', 50))
        self.rtaa_k_per_class = int(kwargs.get('rtaa_k_per_class', 20))
        self.rtaa_max_added = int(kwargs.get('rtaa_max_added', 5000))
        self.rtaa_default_weight = float(kwargs.get('rtaa_default_weight', 1.0))
        self._rtaa_base_edge_index: Optional[torch.Tensor] = None
        self._rtaa_base_edge_weight: Optional[torch.Tensor] = None

        # Track GNN predictions across rounds for flip rate
        self._prev_gnn_preds: Optional[torch.Tensor] = None

        # Best-round checkpointing: track the best label set by proxy quality
        self._best_labels: Optional[Dict[int, int]] = None
        self._best_round: int = 0
        self._best_quality: float = -1.0
        self._best_gnn_state: Optional[dict] = None

        # Ensemble voting across rounds (LoCLE-style)
        self.ensemble_alpha = 0.4  # LoCLE default: weight later rounds more
        self._ensemble_logits: list = []

        # Convergence patience: require N consecutive low-flip rounds
        self._consecutive_low_flip: int = 0

        # Preference trainer (lazy init)

        # Accumulate preference pairs across rounds so ORPO has enough data
        self._accumulated_sft_data: list = []
        self._accumulated_pref_data: list = []

        # DUAL-LP: cross-view (graph + feature-kNN) label-propagation
        # disagreement filter. Produces a per-node confidence in [floor, 1]
        # used to modulate NAAP seed weights, ILC neighbor threshold, and
        # the GNN per-sample loss weight. See src/data_side/dual_lp.py.
        self.use_dual_lp = bool(kwargs.get('use_dual_lp', False))
        self.dual_lp_k = int(kwargs.get('dual_lp_k', 10))
        self.dual_lp_alpha = float(kwargs.get('dual_lp_alpha', 0.85))
        self.dual_lp_iters = int(kwargs.get('dual_lp_iters', 20))
        self.dual_lp_floor = float(kwargs.get('dual_lp_floor', 0.10))
        self.dual_lp_blend_agreement = float(kwargs.get('dual_lp_blend_agreement', 0.7))
        self.dual_lp_blend_margin = float(kwargs.get('dual_lp_blend_margin', 0.3))
        self.dual_lp_apply_ilc = bool(kwargs.get('dual_lp_apply_ilc', True))
        # How aggressively low confidence raises the ILC neighbor threshold.
        # required_threshold += dual_lp_ilc_scale * (1 - w_i).
        self.dual_lp_ilc_scale = float(kwargs.get('dual_lp_ilc_scale', 0.2))
        self._dual_lp_w: Optional[np.ndarray] = None
        self._dual_lp_consensus: Optional[np.ndarray] = None
        self._dual_lp_feat_csr = None  # cached feature kNN sparse mat

    @property
    def _effective_noise_matrix(self) -> Optional[np.ndarray]:
        """Return noise matrix for downstream stages, or None if weighting is disabled."""
        return self.noise_matrix

    def _maybe_recompute_dual_lp(self, data, labels: Dict[int, int], stage_tag: str = "") -> None:
        """Recompute DUAL-LP confidence vector if enabled. Caches the result
        on self._dual_lp_w (np.ndarray [N]) and self._dual_lp_consensus.

        Reuses the feature-kNN graph across calls via self._dual_lp_feat_csr.
        Skips silently if disabled, no labels, or DUAL-LP module import fails.
        """
        if not self.use_dual_lp:
            return
        if labels is None or len(labels) == 0:
            return

    # ── Helper: build labels tensor from dict ────────────────

    def _labels_dict_to_tensor(self, labels: Dict[int, int], num_nodes: int) -> torch.Tensor:
        """Convert {node_id: label} dict to a tensor with -1 for unlabeled."""
        t = torch.full((num_nodes,), -1, dtype=torch.long)
        for nid, lbl in labels.items():
            if 0 <= nid < num_nodes:
                t[nid] = lbl
        return t

    def _build_train_mask(self, labels: Dict[int, int], num_nodes: int) -> torch.Tensor:
        """Boolean mask: True for labeled nodes."""
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        for nid in labels:
            if 0 <= nid < num_nodes:
                mask[nid] = True
        return mask

    def _compute_xmod_weights(self, data, labels: Dict[int, int],
                              num_nodes: int) -> torch.Tensor:
        """Label-free cross-modality reliability weight per node.

        agree_i = fraction of node i's *labeled* graph-neighbours whose (current
        pool) label equals i's label. The LLM-derived label is checked against
        the graph-topology consensus — a second, LLM-independent view. Nodes whose
        label disagrees with their neighbourhood are likely-noisy and down-weighted.

        Global gate g = clip(xmod_alpha * (1 - mean_agree), 0, 1): when the pool's
        labels are highly neighbour-consistent (clean), g~0 and weighting is a
        near no-op; when noisy, g~1 and disagreeing nodes are strongly suppressed.
        Returns w_i = 1 - g*(1 - agree_i) in [1-g, 1]; unlabeled / no-labeled-
        neighbour nodes get 1.0. Uses ONLY pool labels + graph (label-free).
        """
        lab = -np.ones(num_nodes, dtype=np.int64)
        for nid, c in labels.items():
            if 0 <= nid < num_nodes:
                lab[nid] = int(c)
        ei = data.edge_index.cpu().numpy()
        # --- graph view: per-node agreement sum + labeled-neighbour count ---
        g_sum = np.zeros(num_nodes, dtype=np.float64)
        g_cnt = np.zeros(num_nodes, dtype=np.float64)
        src, dst = ei[0], ei[1]
        for s, t in zip(src, dst):
            si = int(s)
            if lab[si] >= 0 and lab[int(t)] >= 0:
                g_cnt[si] += 1.0
                g_sum[si] += 1.0 if lab[int(t)] == lab[si] else 0.0

        # --- feature-kNN view (dual): per-node agreement among k feature-NN ---
        f_sum = np.zeros(num_nodes, dtype=np.float64)
        f_cnt = np.zeros(num_nodes, dtype=np.float64)
        if getattr(self, 'use_dual_xmod', False):
            knn = self._get_feat_knn(data, num_nodes)  # [N, k] (self excluded), cached
            # Degree-fallback: when on, the feature view is used ONLY for nodes whose
            # graph-evidence is starved (< k labeled graph-neighbours); dense nodes keep
            # the pure (stronger) graph view. Protects dense cells (wikics/dblp) from
            # feature dilution while rescuing sparse cells. Threshold fixed = k (label-free).
            fb = bool(getattr(self, 'dual_xmod_fallback', False))
            thr = int(self.dual_xmod_k)
            if knn is not None:
                for i in range(num_nodes):
                    if lab[i] < 0:
                        continue
                    if fb and g_cnt[i] >= thr:
                        continue
                    for j in knn[i]:
                        jj = int(j)
                        if 0 <= jj < num_nodes and lab[jj] >= 0:
                            f_cnt[i] += 1.0
                            f_sum[i] += 1.0 if lab[jj] == lab[i] else 0.0

        # --- evidence-weighted blend (degree-adaptive, label-free) ---
        tot = g_cnt + f_cnt
        agree = np.full(num_nodes, np.nan, dtype=np.float64)
        nz = tot > 0
        agree[nz] = (g_sum[nz] + f_sum[nz]) / tot[nz]
        if np.isnan(agree).all():
            return torch.ones(num_nodes, dtype=torch.float32)
        mean_agree = float(np.nanmean(agree))
        g = float(np.clip(self.xmod_alpha * (1.0 - mean_agree), 0.0, 1.0))
        a = np.where(np.isnan(agree), 1.0, agree)
        w = 1.0 - g * (1.0 - a)
        if getattr(self, 'use_dual_xmod', False):
            lab_mask = lab >= 0
            logger.info("Dual cross-modality weights: mean_agree=%.3f, gate g=%.3f, "
                        "mean graph-evidence=%.1f feat-evidence=%.1f, w range [%.2f, %.2f]",
                        mean_agree, g, float(g_cnt[lab_mask].mean()),
                        float(f_cnt[lab_mask].mean()), float(w.min()), float(w.max()))
        else:
            logger.info("Cross-modality weights: mean_agree=%.3f, gate g=%.3f, "
                        "w range [%.2f, %.2f]", mean_agree, g, float(w.min()), float(w.max()))
        return torch.from_numpy(w.astype(np.float32))

    def _xmod_relabel(self, data, labels: Dict[int, int], num_nodes: int) -> Dict[int, int]:
        """TVCR — Two-View Confident Relabel (label-free pool cleaning).

        For each labelled pool node, compute its graph-neighbour majority label
        (over labelled neighbours) and its feature-kNN majority label (over
        labelled feature-NN). When BOTH views agree on a class c that differs from
        the node's current label — and the graph view has >= min_graph evidence —
        relabel the node to c. The two-view intersection recovers the true label
        66-92% of the time (estimator_quality.py), far above either single view.
        Coverage-preserving (relabels, never drops). Returns the modified dict.
        """
        from collections import Counter as _Counter, defaultdict as _defaultdict
        lab = -np.ones(num_nodes, dtype=np.int64)
        for nid, c in labels.items():
            if 0 <= nid < num_nodes:
                lab[nid] = int(c)
        ei = data.edge_index.cpu().numpy()
        # graph-neighbour labelled-label lists
        gnb = _defaultdict(list)
        for s, t in zip(ei[0], ei[1]):
            si, ti = int(s), int(t)
            if lab[si] >= 0 and lab[ti] >= 0:
                gnb[si].append(lab[ti])
        knn = self._get_feat_knn(data, num_nodes)
        min_g = int(self.xmod_relabel_min_graph)
        flips = 0
        new_labels = dict(labels)
        for i in range(num_nodes):
            if lab[i] < 0:
                continue
            gl = gnb.get(i, [])
            if len(gl) < min_g:
                continue
            gcons = _Counter(gl).most_common(1)[0][0]
            if gcons == lab[i]:
                continue  # already consistent with graph view
            # feature-kNN consensus over labelled feature neighbours
            if knn is None:
                continue
            fl = [int(lab[int(j)]) for j in knn[i] if 0 <= int(j) < num_nodes and lab[int(j)] >= 0]
            if not fl:
                continue
            fcons = _Counter(fl).most_common(1)[0][0]
            if fcons == gcons and gcons != lab[i]:
                new_labels[i] = int(gcons)
                flips += 1
        logger.info("TVCR relabel: flipped %d/%d pool labels to two-view consensus "
                    "(min_graph=%d)", flips, int((lab >= 0).sum()), min_g)
        return new_labels

    def _cluster_reliability(self, data, labels: Dict[int, int], num_nodes: int):
        """Per-cluster cross-modality reliability R_k in [0,1] (label-free).

        For each labelled node, agreement = fraction of its labelled graph- AND
        feature-kNN neighbours whose label matches it (the cross-modality signal).
        R_k = mean agreement of labelled nodes in GraphMAE cluster k. Clusters with
        no labelled nodes get the global mean (neutral). Returns array [K] and the
        cluster-id vector [N], or (None, None) if cluster ids unavailable.
        """
        cid = getattr(self, 'cluster_id', None)
        if cid is None:
            return None, None
        cid_np = (cid.detach().cpu().numpy() if isinstance(cid, torch.Tensor)
                  else np.asarray(cid)).astype(np.int64)
        K = int(cid_np.max()) + 1
        lab = -np.ones(num_nodes, dtype=np.int64)
        for nid, c in labels.items():
            if 0 <= nid < num_nodes:
                lab[nid] = int(c)
        ei = data.edge_index.cpu().numpy()
        gnb = {}
        for s, t in zip(ei[0], ei[1]):
            si, ti = int(s), int(t)
            if lab[si] >= 0 and lab[ti] >= 0:
                gnb.setdefault(si, []).append(lab[ti])
        knn = self._get_feat_knn(data, num_nodes)
        csum = np.zeros(K, dtype=np.float64); ccnt = np.zeros(K, dtype=np.float64)
        for i in range(num_nodes):
            if lab[i] < 0:
                continue
            votes = list(gnb.get(i, []))
            if knn is not None:
                votes += [int(lab[int(j)]) for j in knn[i]
                          if 0 <= int(j) < num_nodes and lab[int(j)] >= 0]
            if not votes:
                continue
            ag = float(np.mean([v == lab[i] for v in votes]))
            k = int(cid_np[i])
            csum[k] += ag; ccnt[k] += 1.0
        Rk = np.full(K, np.nan, dtype=np.float64)
        nz = ccnt > 0
        Rk[nz] = csum[nz] / ccnt[nz]
        gmean = float(np.nanmean(Rk)) if nz.any() else 0.5
        Rk[~nz] = gmean
        # normalise to [0,1] across clusters for use as a selection multiplier
        lo, hi = float(np.nanmin(Rk)), float(np.nanmax(Rk))
        Rk_norm = (Rk - lo) / max(hi - lo, 1e-8)
        return Rk_norm, cid_np

    def _get_feat_knn(self, data, num_nodes: int):
        """Cached feature-space kNN indices [N, k] (self excluded). Features are
        static, so this is computed once. Uses data.x (the GNN's input features)."""
        if self._feat_knn is not None:
            return self._feat_knn
        try:
            from sklearn.neighbors import NearestNeighbors
            X = data.x.cpu().numpy()
            k = int(self.dual_xmod_k)
            nn = NearestNeighbors(n_neighbors=min(k + 1, num_nodes)).fit(X)
            _, idx = nn.kneighbors(X)
            self._feat_knn = idx[:, 1:]  # drop self
            logger.info("Feature-kNN cached: N=%d, k=%d", num_nodes, self._feat_knn.shape[1])
        except Exception as e:
            logger.warning("Feature-kNN failed (%s); dual-xmod falls back to graph-only", e)
            self._feat_knn = None
        return self._feat_knn

    def _build_train_val_split(
        self, labels: Dict[int, int], num_nodes: int, val_frac: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Deterministically split labeled nodes into a train/val mask.

        val_frac is the fraction of labeled nodes held out as val. Uses
        self.seed for reproducibility. Returns (train_mask, val_mask, val_ids).
        val_mask is derived ONLY from labeled (LLM-annotated) nodes — no
        ground-truth leakage. If <10 labeled nodes, returns full train + empty
        val (nothing to hold out).
        """
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        labeled_ids = sorted(int(nid) for nid in labels.keys() if 0 <= int(nid) < num_nodes)
        if val_frac <= 0.0 or len(labeled_ids) < 10:
            for nid in labeled_ids:
                train_mask[nid] = True
            return train_mask, val_mask, []

        rng = np.random.RandomState(self.seed + 31337)
        perm = rng.permutation(len(labeled_ids))
        n_val = max(1, int(round(len(labeled_ids) * val_frac)))
        val_idx_set = set(perm[:n_val].tolist())
        val_ids: list = []
        for i, nid in enumerate(labeled_ids):
            if i in val_idx_set:
                val_mask[nid] = True
                val_ids.append(nid)
            else:
                train_mask[nid] = True
        return train_mask, val_mask, val_ids

    # ── Helper: sparse LP ──────────────────────────────────

    def _sparse_lp(self, labels_dict: Dict[int, int], weights: torch.Tensor,
                   edge_index: torch.Tensor, num_nodes: int,
                   alpha: float = None, iters: int = None,
                   edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Sparse label propagation using scipy sparse matrix multiply.

        Y_{t+1} = alpha * Y_0 + (1 - alpha) * D^{-1} A Y_t

        Much faster than dense Python loops for large graphs.
        Returns soft label matrix [num_nodes, num_classes].
        """
        if alpha is None:
            alpha = self.lp_alpha
        if iters is None:
            iters = self.lp_iters

        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()

        # Build sparse adjacency with optional edge weights
        if edge_weight is not None:
            vals = edge_weight.cpu().numpy().astype(np.float32)
        else:
            vals = np.ones(len(src), dtype=np.float32)
        A = sp.csr_matrix((vals, (src, dst)), shape=(num_nodes, num_nodes))
        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        D_inv = sp.diags(1.0 / deg)
        A_norm = D_inv @ A

        # Initialize with weighted seed labels
        orig = np.zeros((num_nodes, self.num_classes), dtype=np.float32)
        for nid, lbl in labels_dict.items():
            orig[nid, lbl] = float(weights[nid])

        # Iterate
        soft = orig.copy()
        for _ in range(iters):
            soft = alpha * orig + (1 - alpha) * (A_norm @ soft)

        return torch.from_numpy(soft).float()

    # ── Helper: class-balanced weights ──────────────────────

    def _compute_class_balanced_weights(
        self, labels: Dict[int, int], num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute per-node weights that give each class equal total influence.

        Uses inverse class frequency: w_c = N_labeled / (C * n_c).
        This corrects for LLM annotation class bias (e.g., Theory 2.3x
        over-predicted on Cora, DB 1.8x on CiteSeer).

        Returns tensor of weights for all nodes (default 1.0 for unlabeled).
        """
        weights = torch.ones(num_nodes, dtype=torch.float32)

        if len(labels) < 2:
            return weights

        from collections import Counter
        class_counts = Counter(labels.values())
        n_labeled = len(labels)
        n_classes = len(class_counts)

        if n_classes < 2:
            return weights

        for nid, lbl in labels.items():
            count = class_counts[lbl]
            # Inverse class frequency: N / (C * n_c)
            weights[nid] = n_labeled / (n_classes * count)

        # Log class balance info
        max_w = max(weights[nid].item() for nid in labels)
        min_w = min(weights[nid].item() for nid in labels)
        logger.info("  Class-balanced weights: min=%.3f, max=%.3f (ratio=%.1f)",
                    min_w, max_w, max_w / (min_w + 1e-8))

        return weights

    # ── Helper: compute consistency weights ─────────────────

    def _compute_consistency_weights(
        self, data, labels: Dict[int, int],
    ) -> torch.Tensor:
        """
        Compute per-node consistency weights for training.

        Weight = fraction of labeled neighbors that agree with the node's label.
        Nodes with consistent neighborhoods get higher weight (likely correct).
        Nodes with inconsistent neighborhoods get lower weight (likely wrong).

        Returns tensor of weights for all nodes (default 1.0 for unlabeled).
        """
        num_nodes = data.x.size(0)
        weights = torch.ones(num_nodes, dtype=torch.float32)

        if len(labels) < 10:
            return weights

        edge_index = data.edge_index.cpu()
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()

        from collections import defaultdict
        adj = defaultdict(list)
        for s, d in zip(src, dst):
            adj[int(s)].append(int(d))

        n_lowweight = 0
        for nid, lbl in labels.items():
            neighbors = adj.get(nid, [])
            if not neighbors:
                continue

            labeled_neighbors = [n for n in neighbors if n in labels]
            if not labeled_neighbors:
                continue

            same_label = sum(1 for n in labeled_neighbors if labels[n] == lbl)
            consistency = same_label / len(labeled_neighbors)

            # Map consistency to weight: [0, 1] → [0.1, 1.0]
            weights[nid] = max(0.1, consistency)
            if consistency < 0.3:
                n_lowweight += 1

        if n_lowweight > 0:
            logger.info("  Consistency weights: %d/%d nodes with low weight (<0.3)",
                       n_lowweight, len(labels))

        return weights

    # ── Helper: train GNN on current labels ──────────────────

    def _apply_low_conf_drop(
        self,
        data,
        labels: Dict[int, int],
        train_mask: torch.Tensor,
    ) -> Tuple[Dict[int, int], torch.Tensor]:
        """Drop low-confidence nodes from training and zero their edges.

        Uses ``self._prev_gnn_probs`` (captured at end of previous round's
        ``_train_gnn``). Nodes with max-prob < ``self.low_conf_threshold``:
          (a) are removed from the training mask (label set to -1 effectively
              since the loss only looks at train_mask),
          (b) have all incident edges' edge_weight zeroed so they cannot
              propagate or be propagated to. If ``data.edge_weight`` is
              absent, it is initialised to all-ones first.

        Returns the (possibly modified) labels dict and the new train mask.
        """
        probs = getattr(self, '_prev_gnn_probs', None)
        if probs is None:
            return labels, train_mask
        thr = float(self.low_conf_threshold)
        max_prob = probs.max(dim=1).values
        low_mask = (max_prob < thr)
        n_low = int(low_mask.sum().item())
        if n_low == 0:
            return labels, train_mask

        # (a) mask out of training.
        device = train_mask.device
        low_mask_dev = low_mask.to(device)
        new_train_mask = train_mask & (~low_mask_dev)

        # (b) zero outgoing/incoming edges for these nodes.
        ei = data.edge_index
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            ew = data.edge_weight.clone()
        else:
            ew = torch.ones(ei.size(1), dtype=torch.float32, device=ei.device)
        low_on_dev = low_mask.to(ei.device)
        src_low = low_on_dev[ei[0]]
        dst_low = low_on_dev[ei[1]]
        zero_mask = src_low | dst_low
        if int(zero_mask.sum()) > 0:
            ew = ew.clone()
            ew[zero_mask] = 0.0
            data.edge_weight = ew
        logger.info(
            "LCD: dropped %d/%d low-conf nodes (thr=%.2f); zeroed %d/%d incident edges",
            n_low, low_mask.numel(), thr,
            int(zero_mask.sum()), ei.size(1),
        )
        return labels, new_train_mask

    def _capture_gnn_probs(self, gnn, data) -> None:
        """Run a no-grad forward and stash softmax probs as ``self._prev_gnn_probs``."""
        gnn.eval()
        with torch.no_grad():
            logits = gnn(data)
            self._prev_gnn_probs = F.softmax(logits, dim=1).cpu()

    def _train_gnn(self, data, labels: Dict[int, int], warm_start: bool = False):
        """Train a GNN (GCN or GAT) on the current label set.

        Args:
            data: PyG Data object.
            labels: {node_id: label} dict.
            warm_start: If True and previous GNN state exists, initialize from
                        previous weights and train for fewer epochs. Reduces
                        prediction instability between rounds.
        """
        num_nodes = data.x.size(0)
        input_dim = data.x.size(1)

        gnn_num_layers = int(getattr(self, 'gnn_num_layers', 2))
        _tger_will_apply = False  # GAT bypasses; LoCLE backbone owns its own norm.
        if self.backbone == "gat":
            # When RTAA is on, build GATConv layers with edge_dim=1 so they
            # can consume data.edge_weight as a 1-D attention feature.
            gat_edge_dim = 1 if getattr(self, 'use_rtaa', False) else None
            try:
                gnn = GAT(
                    input_dim=input_dim,
                    hidden_dim=self.gnn_hidden_dim,
                    output_dim=self.num_classes,
                    dropout=self.gnn_dropout,
                    num_layers=gnn_num_layers,
                    edge_dim=gat_edge_dim,
                ).to(self.device)
            except TypeError:
                gnn = GAT(
                    input_dim=input_dim,
                    hidden_dim=self.gnn_hidden_dim,
                    output_dim=self.num_classes,
                    dropout=self.gnn_dropout,
                ).to(self.device)
        else:
            # GCNConv uses symmetric D^{-1/2} A D^{-1/2} normalisation.
            _tger_will_apply = False
            gnn = GCN(
                input_dim=input_dim,
                hidden_dim=self.gnn_hidden_dim,
                output_dim=self.num_classes,
                dropout=self.gnn_dropout,
                num_layers=gnn_num_layers,
                normalize_conv=(not _tger_will_apply),
            ).to(self.device)

        epochs = (self.locle_gnn_epochs
                  if getattr(self, 'use_locle_gnn_backbone', False)
                  else self.gnn_epochs)

        if len(labels) == 0:
            gnn.fit(data.to(self.device), labels=None, epochs=epochs, lr=self.gnn_lr)
        else:
            labels_tensor = self._labels_dict_to_tensor(labels, num_nodes).to(self.device)
            # apr15: optional held-out val split from labeled (LLM) nodes only.
            # Default val_holdout_frac=0 → train_mask = all labeled, val_mask empty
            # (bit-identical to legacy path).
            if self.val_holdout_frac > 0.0:
                tm_cpu, vm_cpu, val_ids = self._build_train_val_split(
                    labels, num_nodes, self.val_holdout_frac,
                )
                train_mask = tm_cpu.to(self.device)
                val_mask = vm_cpu.to(self.device)
                if len(val_ids) > 0:
                    logger.info(
                        "Val holdout: %d/%d labeled nodes (frac=%.2f) used for early stopping",
                        len(val_ids), len(labels), self.val_holdout_frac,
                    )
            else:
                train_mask = self._build_train_mask(labels, num_nodes).to(self.device)
                val_mask = None

            # Combine any existing weights (e.g., LP confidence) with consistency
            existing_weights = getattr(data, '_node_weights', None)
            if existing_weights is not None:
                # Use LP confidence weights × consistency
                consistency_weights = self._compute_consistency_weights(data, labels).cpu()
                weights = existing_weights.cpu() * consistency_weights
                data_dev = data.to(self.device)
                data_dev._node_weights = weights.to(self.device)
            else:
                # No existing weights — train with plain CE loss (no weighting)
                data_dev = data.to(self.device)

            # ── Cross-modality reliability weighting (label-free noise estimate) ──
            # Folds w_i into _node_weights; gated so clean datasets are ~no-ops.
            if getattr(self, 'use_xmod_weight', False):
                xw = self._compute_xmod_weights(data, labels, num_nodes).to(self.device)
                cur = getattr(data_dev, '_node_weights', None)
                data_dev._node_weights = xw if cur is None else (cur * xw)


            # Degree-aware DropEdge: high-degree nodes drop at the configured
            # rate; low-degree nodes keep more edges. Read by gcn._fit_supervised.
            if getattr(self, 'dropedge_degree_aware', False):
                data_dev._dropedge_degree_aware = True

            # ── LCD: low-confidence drop. Uses last round's GNN probs to
            #    mask uncertain nodes from training and zero their incident
            #    edges so they neither propagate nor receive gradients.
            if getattr(self, 'use_low_conf_drop', False):
                labels, train_mask = self._apply_low_conf_drop(
                    data_dev, labels, train_mask,
                )

            # ── T_c-calibrated soft-label override (Experiment 2) ──
            # When ``use_tc_soft_labels`` is set, replace hard CE on
            # ``labels_tensor`` with KL-divergence against
            # ``T_c[k_u, ŷ_u, :]`` per labeled node. The pseudo-label
            # ŷ_u is taken from ``labels_tensor``; the cluster id k_u
            # from ``self.cluster_id``. The downstream GCN/GAT pickup
            # is automatic: both check ``data._soft_targets``.
            if (self.use_tc_soft_labels
                    and self.T_c is not None
                    and self.cluster_id is not None
                    and self.T_c.shape[1] == self.num_classes):
                T_c_t = torch.as_tensor(
                    self.T_c, dtype=torch.float32, device=self.device,
                )
                K_tc = int(T_c_t.shape[0])
                cid = torch.as_tensor(
                    self.cluster_id, dtype=torch.long, device=self.device,
                ).clamp(0, K_tc - 1)
                if cid.size(0) >= num_nodes:
                    cid = cid[:num_nodes]
                else:
                    cid = torch.cat(
                        [cid, torch.zeros(num_nodes - cid.size(0),
                                          dtype=torch.long, device=self.device)],
                        dim=0,
                    )
                soft = torch.zeros(
                    num_nodes, self.num_classes,
                    dtype=torch.float32, device=self.device,
                )
                has_label = (labels_tensor >= 0)
                labeled_idx = torch.where(has_label)[0]
                if labeled_idx.numel() > 0:
                    ks = cid[labeled_idx]
                    ys = labels_tensor[labeled_idx].clamp(0, self.num_classes - 1)
                    if self.tc_soft_mode == 'column':
                        # Bayes posterior: P(true=t | LLM=ŷ, k) ∝ T_c[k, t, ŷ]
                        # under uniform prior. Vectorised column gather.
                        soft[labeled_idx] = T_c_t[ks, :, ys]
                    else:
                        # Legacy "row" mode — LLM-mimic target (catastrophic).
                        soft[labeled_idx] = T_c_t[ks, ys, :]
                soft = soft.clamp(min=1e-8)
                soft = soft / soft.sum(dim=1, keepdim=True)
                data_dev._soft_targets = soft
                logger.info(
                    "T_c soft labels (mode=%s): built [%d, %d] target tensor for "
                    "%d labeled nodes (KL-div loss; gradient-equivalent to CE).",
                    self.tc_soft_mode, soft.size(0), soft.size(1),
                    int(labeled_idx.numel()),
                )

            # DUAL-LP: multiply per-sample loss weight by cross-view confidence.
            # Down-weights training nodes the graph and feature views disagree on.

            # Pass label smoothing to the GNN
            if self.label_smoothing > 0:
                data_dev._label_smoothing = self.label_smoothing

            # Pass GCE loss flag

            # Pass NoiseAda flag

            # Pass DropEdge rate
            if self.dropedge_rate > 0:
                data_dev._dropedge_rate = self.dropedge_rate

            # Pass SCE loss flag

            # Entropy regularization
            data_dev._entropy_reg = getattr(self, 'entropy_reg', 0.0)

            # Graph Contrastive Regularization (GCA, WWW 2021)
            contrastive_w = getattr(self, 'contrastive_weight', 0.0)
            if contrastive_w > 0:
                data_dev._contrastive_weight = contrastive_w

            # Bootstrap loss for noise-robust training

            # ELR loss (Liu et al., NeurIPS 2020) — benchmarked as superior to
            # Bootstrap/GCE/SCE on 4/5 datasets (see experiments_verification/).
            if getattr(self, 'use_elr', False):
                data_dev._use_elr = True
                data_dev._elr_beta = getattr(self, 'elr_beta', 0.7)
                data_dev._elr_lambda = getattr(self, 'elr_lambda', 3.0)

            # may18 v7: Consistency Regularization on non-selected nodes (FixMatch/UDA-style).
            # Aux loss: KL(softmax(GNN_aug(x_n)) || softmax(GNN_orig(x_n)).detach())
            # for non-labeled nodes. Pool stays the same; gain comes from explicit
            # generalization objective on unlabeled nodes.
            if getattr(self, 'use_consistency_reg', False):
                data_dev._use_consistency_reg = True
                data_dev._consistency_lambda = getattr(self, 'consistency_lambda', 1.0)
                data_dev._consistency_aug_dropedge = getattr(self, 'consistency_aug_dropedge', 0.5)
                data_dev._consistency_aug_feat_mask = getattr(self, 'consistency_aug_feat_mask', 0.1)
                data_dev._consistency_warmup_epochs = getattr(self, 'consistency_warmup_epochs', 50)
                data_dev._consistency_conf_threshold = getattr(self, 'consistency_conf_threshold', 0.0)

            # Forward loss correction: use noise transition matrix to correct loss
            # (Patrini et al., CVPR 2017: "Making DNNs Robust to Label Noise")
            if self.noise_matrix is not None and getattr(self, 'use_forward_correction', False):
                T = torch.from_numpy(self.noise_matrix).float().to(self.device)
                tau = float(getattr(self, 'forward_correction_temperature', 1.0))
                if tau != 1.0:
                    # Softmax-temperature scaling: log-T then softmax / tau.
                    # tau > 1 flattens cluster confidence (less over-confident),
                    # tau < 1 sharpens. Row-normalised by construction.
                    log_T = torch.log(T.clamp_min(1e-8))
                    T = F.softmax(log_T / tau, dim=1)
                    logger.info(
                        "Forward correction T tempered with tau=%.2f (mean diag=%.3f, "
                        "max off-diag=%.3f)",
                        tau, T.diag().mean().item(),
                        (T - torch.diag(T.diag())).max().item(),
                    )
                else:
                    # Normalize rows to be valid transition probabilities
                    T = T / (T.sum(dim=1, keepdim=True) + 1e-8)
                data_dev._noise_transition = T

            # Only GCN supports val-based early stopping plumbing; GAT falls
            # back to fixed-epoch training (legacy behavior).
            fit_kwargs = dict(
                labels=labels_tensor,
                train_mask=train_mask,
                epochs=epochs,
                lr=self.gnn_lr,
                weight_decay=self.gnn_weight_decay,
            )
            if (self.val_holdout_frac > 0.0 and val_mask is not None
                    and bool(val_mask.any().item())
                    and self.early_stop_patience > 0
                    and self.backbone == "gcn"):
                fit_kwargs["val_mask"] = val_mask
                fit_kwargs["early_stop_patience"] = self.early_stop_patience
                fit_kwargs["val_check_every"] = self.val_check_every

            # apr15 v89: OGB gold-val early stopping (LLM-GNN arxiv protocol).
            # Takes precedence over the LLM-holdout early stop above when both
            # are configured (rare). Uses `data.ogb_val_idx` (set by
            # experiment_runner when use_ogb_val_early_stop=True AND dataset
            # is arxiv). Gold labels (data.y) are passed as `val_labels` —
            # they feed ONLY the val-accuracy check in _fit_supervised; they
            # are never in the training loss.

            gnn.fit(data_dev, **fit_kwargs)

        self._prev_gnn_state = gnn.get_state()
        self.gnn = gnn
        # Capture softmax probabilities once for use by the LCD drop logic in
        # the next round. Cheap (one forward pass) and independent of other
        # diagnostic paths that consume self._prev_gnn_preds.
        if getattr(self, 'use_low_conf_drop', False):
            try:
                self._capture_gnn_probs(gnn, data_dev)
            except Exception as e:  # noqa: BLE001
                logger.warning("LCD: failed to capture GNN probs (%s)", e)
        return gnn

    # ── Helper: load self-supervised embeddings for TGER kNN augmentation ──

    def _load_tger_embeddings(self, data) -> torch.Tensor:
        """Return [N, D] embeddings for the kNN graph in Phase 1 of TGER.

        Resolution order:
          1. ``self.tger_embedding_path`` — explicit ``.pt`` of a saved
             [N, D] tensor.
          2. Standard GraphMAE2 checkpoint at
             ``raaca/phase2_raaca/results/graphmae2/{dataset}_gc1_h64.pt``
             (single GCNConv encoder, same format NAAP already uses).
          3. ``data.x`` (warning logged; falls back to raw features).
        """
        # (1) explicit tensor path
        if self.tger_embedding_path:
            try:
                obj = torch.load(
                    self.tger_embedding_path, map_location="cpu", weights_only=False,
                )
                if isinstance(obj, torch.Tensor):
                    logger.info(
                        "TGER embeddings loaded from %s: shape=%s",
                        self.tger_embedding_path, tuple(obj.shape),
                    )
                    return obj
                logger.warning(
                    "TGER: %s did not contain a Tensor (got %s); trying GraphMAE2",
                    self.tger_embedding_path, type(obj).__name__,
                )
            except Exception as e:
                logger.warning(
                    "TGER: failed to load %s (%s); trying GraphMAE2",
                    self.tger_embedding_path, e,
                )

        # (2) GraphMAE2 checkpoint (reuses NAAP's loading pattern)
        try:
            from torch_geometric.nn import GCNConv as _GCNConv
            gpath = (
                f"/gpfs/scratchfs1/jhf24001/ldn24004/new_noise/"
                f"raaca/phase2_raaca/results/graphmae2/"
                f"{self.dataset_name}_gc1_h64.pt"
            )
            ckpt = torch.load(gpath, map_location="cpu", weights_only=False)
            hid = int(ckpt.get("hid", 64))
            enc = _GCNConv(int(data.x.size(1)), hid, add_self_loops=True)
            enc.load_state_dict(ckpt["gcn_state"])
            enc.eval()
            with torch.no_grad():
                h = F.relu(enc(data.x.cpu(), data.edge_index.cpu()))
            logger.info(
                "TGER GraphMAE2 embeddings: shape=%s from %s", tuple(h.shape), gpath,
            )
            return h.detach()
        except Exception as e:
            logger.warning("TGER: GraphMAE2 load failed (%s); using data.x", e)

        # (3) fallback
        return data.x.detach().cpu()

    # ── Helper: annotate nodes via LLM ───────────────────────

    def _annotate_nodes(self, node_ids: List[int], raw_texts: list) -> Dict[int, int]:
        """
        Annotate nodes via LLM, consuming budget.

        Returns {node_id: label_index} for successfully annotated nodes.
        """
        if not node_ids:
            return {}

        # Cap to remaining budget
        available = self.budget.remaining
        if available <= 0:
            logger.warning("Budget exhausted, cannot annotate")
            return {}

        node_ids = node_ids[:available]
        texts = [raw_texts[nid] for nid in node_ids]

        annotations = self.annotator.annotate_nodes(node_ids, texts)
        # annotations: {node_id: (label_index, confidence)}

        # Track budget consumption
        actual_consumed = len(annotations)
        self.budget.consume(actual_consumed, stage="annotation")

        # Extract label indices and store confidences
        result = {}
        if not hasattr(self, '_annotation_conf'):
            self._annotation_conf = {}
        for nid, (label, conf) in annotations.items():
            if 0 <= label < self.num_classes:
                result[nid] = label
                self._annotation_conf[nid] = conf / 100.0  # normalize to [0, 1]

        # Accumulate the RAW within-budget LLM labels (before any rectification),
        # so a within-budget T_c can be built from exactly the queried seeds.
        self._llm_annotated.update(result)

        # ── Bayes Rectifier gateway (Experiment 4) ──
        # Applies hard-flip: y' = argmax_t T_c[k_u, t, ŷ_u] for every
        # LLM-annotated node, using the cluster-conditional T_c.
        # ILC pseudo-labels and self-training labels never touch this
        # function, so they remain unchanged per spec.
        if self.use_bayes_rectify and result \
                and self.T_c is not None and self.cluster_id is not None \
                and self.T_c.shape[1] == self.num_classes:
            n_total = 0
            n_flipped = 0
            K_tc = int(self.T_c.shape[0])
            for nid, llm_label in list(result.items()):
                if nid >= len(self.cluster_id):
                    continue
                k = int(self.cluster_id[nid])
                if k < 0 or k >= K_tc:
                    continue
                col = self.T_c[k, :, llm_label]  # P(LLM=ŷ | true=t, k) ∝ posterior
                col_sum = float(col.sum())
                if col_sum <= 0:
                    continue
                rectified = int(col.argmax())
                n_total += 1
                if rectified != llm_label:
                    n_flipped += 1
                    result[nid] = rectified
            self._bayes_rectify_stats["total"] += n_total
            self._bayes_rectify_stats["flipped"] += n_flipped
            logger.info(
                "Bayes rectifier: %d/%d LLM labels hard-flipped (cumulative "
                "%d/%d = %.1f%%).",
                n_flipped, n_total,
                self._bayes_rectify_stats["flipped"],
                self._bayes_rectify_stats["total"],
                100.0 * self._bayes_rectify_stats["flipped"] /
                max(1, self._bayes_rectify_stats["total"]),
            )

        return result

    # ── Stage 0: Noise Probing ───────────────────────────────

    def _run_noise_probing(self, data) -> np.ndarray:
        """
        Stage 0: Estimate LLM noise via pseudo-sample probing.

        Uses probing_fraction of total budget.  The noise matrix informs
        all downstream stages (selection weighting, agreement partitioning,
        preference training).
        """
        logger.info(
            "=== Stage 0: Noise Probing (budget: %d) ===",
            self.budget.get_probing_budget(),
        )

        constructor = PseudoSampleConstructor(
            dataset_name=self.dataset_name,
            class_names=self.class_names,
            seed=self.seed,
        )
        pseudo_samples = constructor.construct(data, n_per_class=self.n_probes_per_class)

        noise_matrix, per_class_acc, confused_pairs = self.profiler.estimate_noise_matrix(
            dataset_name=self.dataset_name,
            class_names=self.class_names,
            pseudo_samples=pseudo_samples,
            n_probes_per_class=self.n_probes_per_class,
            budget_manager=self.budget,
            use_cache=False,
        )

        self.noise_matrix = noise_matrix
        self.per_class_accuracy = per_class_acc
        self.confused_pairs = confused_pairs

        logger.info(
            "Noise probing complete. Mean accuracy: %.3f, Probing budget used: %d/%d",
            per_class_acc.mean(),
            self.budget._consumed_probing,
            self.budget._probing_budget,
        )

        return noise_matrix

    # ── PS-FeatProp-W selection (LLM-GNN pagerank2) ──────────


    # ── K-center greedy selection (apr14) ────────────────────


    # ── age_query (LLM-GNN's 3-term) selection (apr14) ───────


    # ── Stage I: Initial Annotation ──────────────────────────

    def _maybe_apply_vote_confidence_weights(self, data, labels):
        """Set `data._node_weights` from per-node annotation confidences.

        Only runs when `use_vote_confidence_weights=True`. Confidences are
        pulled from `self._annotation_conf` (already normalized to [0, 1]).
        Unlabeled nodes receive weight 0. Labeled nodes are mapped via one of:
          - "linear": weight = confidence (continuous in [0, 1]).
          - "sharp":  weight ∈ {0.3, 0.7, 1.0} bucketed.
                      For n_samples=3 voting: 33% → 0.3, 67% → 0.7, 100% → 1.0.
                      For n_samples=1 verbalized (≈80/90/100): same buckets.

        When `use_cluster_conf_weights=True` AND a cluster T_c is loaded
        (T_c + cluster_id), the per-node weight is additionally multiplied by
        T_c[cluster_id[i], label_i, label_i] ** cluster_conf_pow — the
        per-cluster LLM accuracy on this node's declared class. After
        multiplication, weights are renormalised so the labeled-node mean
        equals 1.0; this preserves overall loss magnitude and turns the
        weight into a pure REDISTRIBUTION across the labeled set.

        The resulting tensor is assigned to `data._node_weights`; `_train_gnn`
        picks it up via `getattr(data, '_node_weights', None)` and multiplies
        it by a consistency term (see `_train_gnn` line ~595).
        Default (both flags False) → no-op, fully bit-identical to legacy.
        """
        use_ccr = bool(getattr(self, 'use_vote_confidence_weights', False))
        use_cccr = bool(getattr(self, 'use_cluster_conf_weights', False))
        if not (use_ccr or use_cccr):
            return
        if not labels:
            return
        ann_conf = getattr(self, '_annotation_conf', None) or {}
        num_nodes = data.x.size(0)
        w = torch.zeros(num_nodes, dtype=torch.float32)
        sched = getattr(self, 'vote_conf_schedule', 'linear')
        n_sharp_low = n_sharp_mid = n_sharp_high = 0
        for nid in labels:
            if not (0 <= nid < num_nodes):
                continue
            c = float(ann_conf.get(nid, 1.0))  # missing conf → assume full trust
            c = max(0.0, min(1.0, c))
            if not use_ccr:
                # Cluster-only: CCR contribution disabled, start from 1.0
                wv = 1.0
            elif sched == 'sharp':
                # Bucket boundaries cover both voting (33/67/100) and
                # verbalized (80/90/100) regimes.
                if c < 0.5:
                    wv = 0.3
                    n_sharp_low += 1
                elif c < 0.85:
                    wv = 0.7
                    n_sharp_mid += 1
                else:
                    wv = 1.0
                    n_sharp_high += 1
            else:  # linear
                wv = c
            w[nid] = wv

        # Cluster-conditional multiplier: w_i *= T_c[k, label, label] ** pow
        if use_cccr and self.T_c is not None and self.cluster_id is not None:
            pow_c = float(getattr(self, 'cluster_conf_pow', 1.0))
            K = int(self.T_c.shape[0])
            C_tc = int(self.T_c.shape[1])
            for nid, lbl in labels.items():
                if not (0 <= nid < num_nodes) or not (0 <= int(lbl) < C_tc):
                    continue
                k = int(self.cluster_id[nid]) if nid < len(self.cluster_id) else 0
                k = max(0, min(K - 1, k))
                tc_diag = float(self.T_c[k, int(lbl), int(lbl)])
                tc_diag = max(0.0, min(1.0, tc_diag))
                w[nid] = float(w[nid]) * (tc_diag ** pow_c)
            # Renormalise: mean over labeled = 1.0  (redistribution, not attenuation)
            labeled_nids = [nid for nid in labels if 0 <= nid < num_nodes]
            if labeled_nids:
                lw = w[labeled_nids]
                m = float(lw.mean().item())
                if m > 0:
                    w[labeled_nids] = lw / m
                logger.info(
                    "CCCR weights: pow=%.2f, mean_pre=%.3f, post-normalise mean=1.0, "
                    "min=%.3f max=%.3f over %d labeled",
                    pow_c, m,
                    float(w[labeled_nids].min().item()),
                    float(w[labeled_nids].max().item()),
                    len(labeled_nids),
                )
        data._node_weights = w
        if sched == 'sharp':
            logger.info(
                "Vote-conf weights (sharp): %d low(0.3) / %d mid(0.7) / %d high(1.0), "
                "mean_w=%.3f over %d labeled",
                n_sharp_low, n_sharp_mid, n_sharp_high,
                float(w[list(labels.keys())].mean().item()) if labels else 0.0,
                len(labels),
            )
        else:
            labeled_nids = [nid for nid in labels if 0 <= nid < num_nodes]
            if labeled_nids:
                lw = w[labeled_nids]
                logger.info(
                    "Vote-conf weights (linear): mean=%.3f min=%.3f max=%.3f over %d labeled",
                    float(lw.mean().item()), float(lw.min().item()),
                    float(lw.max().item()), len(labeled_nids),
                )

    def _run_initial_annotation(self, data) -> Dict[int, int]:
        """
        Stage I: 3-round active selection + annotation.

        Round 1: Subspace clustering (40% budget)
        Round 2-3: GNN uncertainty selection (remaining budget)
        """
        round_budget = self.budget.get_round_budget(
            round_num=0,
            total_rounds=self.total_rounds,
            initial_fraction=self.initial_fraction,
        )
        # may18 v6: reserve AQP budget out of Stage I so AQP has queries left.
        # The runner added aqp_n_probes to total_budget, but with initial_fraction=1.0
        # Stage I would otherwise consume everything.
        logger.info(
            "=== Stage I: Initial Annotation (budget: %d) ===",
            round_budget,
        )

        num_nodes = data.x.size(0)
        # Budget splits for Stage-I rounds. Default [0.4, 0.3, 0.3]: 40% subspace-
        # clustering, 60% GNN-uncertainty. LoCLE uses [1.0] (all via subspace).
        # Override via `stage1_splits` config (e.g. [1.0], [0.8, 0.2], [0.5, 0.3, 0.2]).
        splits = getattr(self, 'stage1_splits', None) or [0.4, 0.3, 0.3]
        labels = {}

        # Check if annotator has a restricted set of available nodes (cache mode)
        available_nodes = None
        if hasattr(self.annotator, 'get_available_nodes'):
            available_nodes = self.annotator.get_available_nodes()
            logger.info("Active selection restricted to %d cached nodes", len(available_nodes))

        for i, frac in enumerate(splits):
            r_budget = int(round_budget * frac)
            if i == len(splits) - 1:
                r_budget = round_budget - len(labels)
            if r_budget <= 0 or self.budget.is_exhausted():
                break

            # stage1_all_subspace: replicate LoCLE's per-stage refinement — every
            # round selects via subspace clustering on the CURRENT GNN embeddings
            # (representative, cleaner-label nodes), not entropy (boundary, noisy
            # on cora). LoCLE ablation: this iteration is the +4.4pp cora-gcn lever.
            _all_subspace = bool(getattr(self, 'stage1_all_subspace', False))
            if i == 0 or _all_subspace:
                # Optional: replace Round-1 NoiseAwareSelector with NoiseSelect's
                # noise-weighted hierarchical coreset. Isolates the selector
                # contribution while keeping legacy's iterative refinement.
                selector = NoiseAwareSelector(
                    noise_matrix=self._effective_noise_matrix,
                    num_classes=self.num_classes,
                    seed=self.seed,
                    select_downweight_floor=self.select_downweight_floor,
                    select_min_class_frac=self.select_min_class_frac,
                )
                # may18 v6: optionally use GraphMAE2 features for NAAP
                # subspace clustering (graph-aware → higher cluster purity).
                naap_emb = data.x
                # stage1_all_subspace rounds 2+: subspace on CURRENT GNN embeddings
                # (the improving model guides representative selection, à la LoCLE).
                if _all_subspace and i > 0 and labels:
                    try:
                        _g = self._train_gnn(data, labels)
                        _g.eval()
                        with torch.no_grad():
                            _emb = _g.encode(data.to(self.device))
                        # select_initial_nodes is CPU (SVD/kmeans/degree): force
                        # the whole call onto CPU like the round-1 path to avoid
                        # device mismatches (_train_gnn may leave data on CUDA).
                        naap_emb = _emb.detach().cpu()
                        data = data.cpu()
                        logger.info("stage1_all_subspace R%d: subspace on GNN "
                                    "embeddings %s", i + 1, tuple(naap_emb.shape))
                    except Exception as _e_emb:
                        logger.warning("stage1_all_subspace: GNN-embed failed (%s); "
                                       "using data.x", _e_emb)
                        naap_emb = data.x
                if (not (_all_subspace and i > 0)) and getattr(self, 'naap_use_graphmae_features', False):
                    try:
                        gpath = self.graphmae_features_path or (
                            f'/gpfs/scratchfs1/jhf24001/ldn24004/new_noise/'
                            f'raaca/phase2_raaca/results/graphmae2/'
                            f'{self.dataset_name}_gc1_h64.pt'
                        )
                        import torch as _torch
                        import torch.nn.functional as _F
                        from torch_geometric.nn import GCNConv as _GCNConv
                        _ckpt = _torch.load(gpath, map_location='cpu', weights_only=False)
                        _hid = int(_ckpt.get('hid', 64))
                        _enc = _GCNConv(int(data.x.size(1)), _hid, add_self_loops=True)
                        _enc.load_state_dict(_ckpt['gcn_state'])
                        _enc.eval()
                        with _torch.no_grad():
                            _h = _F.relu(_enc(data.x.cpu(), data.edge_index.cpu()))
                        naap_emb = _h.detach()
                        logger.info("NAAP using GraphMAE2 features: shape=%s (vs data.x=%s)",
                                    tuple(naap_emb.shape), tuple(data.x.shape))
                    except Exception as _e_gmae:
                        logger.warning("Failed to load GraphMAE2 features (%s); "
                                       "falling back to data.x", _e_gmae)
                        naap_emb = data.x
                # Under stage1_all_subspace the selector re-ranks the SAME
                # representative nodes each round; request budget+|labelled| so
                # enough NEW nodes survive the exclusion below.
                _sel_budget = r_budget + (len(labels) if (_all_subspace and i > 0) else 0)
                annotation_pool, correction_pool = selector.select_initial_nodes(
                    data, naap_emb, budget=_sel_budget,
                    candidate_multiplier=4.0,
                )
                # Exclude already-labelled nodes (matters for rounds 2+ under
                # stage1_all_subspace; round 1 has none labelled yet).
                selected = [n for n in (annotation_pool + correction_pool)
                            if n not in labels][:r_budget]
                # Filter to available nodes if cache-restricted
                if available_nodes is not None:
                    selected = [n for n in selected if n in available_nodes]
                    if len(selected) < r_budget:
                        # Fill remaining from available nodes not yet selected
                        remaining = [n for n in available_nodes if n not in set(selected) and n not in labels]
                        import random
                        rng = random.Random(self.seed)
                        rng.shuffle(remaining)
                        selected.extend(remaining[:r_budget - len(selected)])
            else:
                gnn = self._train_gnn(data, labels)
                gnn.eval()
                data_dev = data.to(self.device)
                with torch.no_grad():
                    logits = gnn(data_dev)
                    probs = F.softmax(logits, dim=1).cpu()
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

                labeled_set = set(labels.keys())
                # CRBA — Cluster-Risk-Based Budget Allocation. Augment the
                # entropy score by posterior cluster-risk under T_c, biasing
                # round 2-3 selection toward unlabelled nodes in high-LLM-noise
                # clusters. λ=0 (default) → bit-identical entropy ranking.
                crba_lam = float(getattr(self, 'crba_lambda', 0.0))
                crba_quota = float(getattr(self, 'crba_quota', 0.0))
                use_crba = (crba_lam > 0 and self.T_c is not None
                            and self.cluster_id is not None
                            and self.T_c.shape[1] == self.num_classes)
                if use_crba:
                    K_c = int(self.T_c.shape[0])
                    cid_np = np.clip(np.asarray(self.cluster_id, dtype=np.int64), 0, K_c - 1)
                    T_diag = np.diagonal(self.T_c, axis1=1, axis2=2).astype(np.float32)
                    p_np = probs.numpy().astype(np.float32)
                    posterior_risk = 1.0 - (p_np * T_diag[cid_np]).sum(axis=1)
                    ent_np = entropy.numpy().astype(np.float32)
                    def _mm(x):
                        lo, hi = float(x.min()), float(x.max())
                        return (x - lo) / max(hi - lo, 1e-8)
                    score_np = (1.0 - crba_lam) * _mm(ent_np) + crba_lam * _mm(posterior_risk)
                    logger.info(
                        "CRBA stage-I R%d: λ=%.2f quota=%.2f, posterior_risk mean=%.3f, "
                        "entropy mean=%.3f", i + 1, crba_lam, crba_quota,
                        float(posterior_risk.mean()), float(ent_np.mean()),
                    )
                else:
                    score_np = entropy.numpy().astype(np.float32)
                # RGS — reliability-guided selection: bias round 2-3 toward
                # clusters whose round-1 seeds are cross-modality-reliable (clean
                # LLM labels). score *= floor + (1-floor)*R_{cluster}, blended by beta.
                if getattr(self, 'use_reliability_selection', False):
                    Rk, cid_rgs = self._cluster_reliability(data, labels, num_nodes)
                    if Rk is not None:
                        fl = float(self.reliability_select_floor)
                        beta = float(self.reliability_select_beta)
                        cid_clip = np.clip(cid_rgs, 0, len(Rk) - 1)
                        rel_factor = fl + (1.0 - fl) * Rk[cid_clip].astype(np.float32)
                        # min-max the base score so the multiplicative blend is balanced
                        s = score_np.astype(np.float32)
                        s = (s - s.min()) / max(float(s.max() - s.min()), 1e-8)
                        score_np = ((1.0 - beta) * s + beta * s * rel_factor).astype(np.float32)
                        logger.info("RGS R%d: reliability-weighted selection "
                                    "(floor=%.2f beta=%.2f, R_k range [%.2f,%.2f])",
                                    i + 1, fl, beta, float(Rk.min()), float(Rk.max()))
                # Filter candidates to available nodes if cache-restricted
                candidate_pool = available_nodes if available_nodes is not None else range(num_nodes)
                candidates = [(nid, float(score_np[nid]))
                             for nid in candidate_pool if nid not in labeled_set]
                candidates.sort(key=lambda x: -x[1])

                # Per-cluster quota cap (LoCLE's CRBA component): cap selection
                # per cluster at `ceil(quota * r_budget * cluster_size / N) + 1`
                # so no single cluster can monopolise a round's budget. Only
                # active when use_crba AND crba_quota > 0. Required ingredient
                # for CRBA per memory project_may13_phase4_consolidated.md
                # (LoCLE-side gave dblp +0.49pp with quota; without it the
                # extreme-skew posterior_risk over-picks one or two clusters).
                if use_crba and crba_quota > 0.0 and r_budget > 0:
                    K_c = int(self.T_c.shape[0])
                    cid_full = np.clip(np.asarray(self.cluster_id, dtype=np.int64), 0, K_c - 1)
                    cluster_sizes = np.bincount(cid_full, minlength=K_c).astype(np.float32)
                    per_cluster_cap = np.ceil(
                        crba_quota * r_budget * cluster_sizes / num_nodes
                    ).astype(np.int64) + 1
                    cluster_picked = np.zeros(K_c, dtype=np.int64)
                    selected = []
                    selected_set = set()
                    for nid, _ in candidates:
                        if len(selected) >= r_budget:
                            break
                        if nid in selected_set:
                            continue
                        k_v = int(cid_full[nid])
                        if cluster_picked[k_v] < per_cluster_cap[k_v]:
                            selected.append(int(nid))
                            selected_set.add(nid)
                            cluster_picked[k_v] += 1
                    # If quota too tight to fill r_budget, top up from remaining
                    if len(selected) < r_budget:
                        for nid, _ in candidates:
                            if len(selected) >= r_budget:
                                break
                            if nid not in selected_set:
                                selected.append(int(nid))
                                selected_set.add(nid)
                    n_clusters_used = int((cluster_picked > 0).sum())
                    logger.info(
                        "CRBA quota R%d: %d nodes selected across %d/%d clusters "
                        "(quota=%.2f, max-picked=%d)",
                        i + 1, len(selected), n_clusters_used, K_c, crba_quota,
                        int(cluster_picked.max()),
                    )
                else:
                    selected = [nid for nid, _ in candidates[:r_budget]]

            # DIAG (env-guarded): override selection with a fixed node pool from
            # disk, drawn in order per round (respects budget). For selection
            # ablations only; off by default.
            import os as _os2
            _fx = _os2.environ.get("CANE_FIXED_SEL")
            if _fx:
                import json as _j2
                if not hasattr(self, "_fixed_pool"):
                    self._fixed_pool = [int(z) for z in _j2.load(open(_fx))]
                selected = [n for n in self._fixed_pool if n not in labels][:r_budget]

            r_labels = self._annotate_nodes(selected, data.raw_texts)
            labels.update(r_labels)
            logger.info("Stage I Round %d: %d annotations, total=%d",
                       i + 1, len(r_labels), len(labels))

            # Probe-based cross-modality T_c: after the FIRST round (the probe),
            # estimate T_c from those probe labels only, so it can guide the
            # remaining selection rounds and the downstream gates.
            if self.use_crossmodal_tc and i == 0 and labels:
                self._set_crossmodal_Tc_from_probe(labels, data, num_nodes)

        if labels:
            self._maybe_apply_vote_confidence_weights(data, labels)
            gnn = self._train_gnn(data, labels)
            gnn.eval()
            data_dev = data.to(self.device)
            with torch.no_grad():
                logits = gnn(data_dev)
                self._prev_gnn_preds = logits.argmax(dim=1).cpu()

        logger.info("Stage I: %d annotated, budget remaining: %d",
                    len(labels), self.budget.remaining)
        return labels

    # ── Stage II: Iterative Refinement ───────────────────────

    def _run_refinement_round(
        self,
        data,
        labels: Dict[int, int],
        round_num: int,
    ) -> Tuple[Dict[int, int], float]:
        """
        LoCLE-style refinement round (rewritten from scratch).

        1. Train GNN with NoiseAda on current labels
        2. Dual-score selection: confident (self-label) + inconfident (query LLM)
        3. Self-label confident nodes (FREE — no budget cost)
        4. Query LLM only on inconfident nodes (costs budget)
        5. Add LLM labels to training set
        6. Track ensemble predictions across rounds

        Returns updated labels and label flip rate.
        """
        round_budget = self.budget.get_round_budget(
            round_num=round_num,
            total_rounds=self.total_rounds,
            initial_fraction=self.initial_fraction,
        )
        logger.info(
            "=== Stage II Round %d (budget: %d, remaining: %d) ===",
            round_num, round_budget, self.budget.remaining,
        )

        if self.budget.is_exhausted() or round_budget == 0:
            logger.warning("Budget exhausted, skipping round %d", round_num)
            return labels, 0.0

        num_nodes = data.x.size(0)
        prev_labels = dict(labels)

        # ── Step 1: Train GNN with NoiseAda on current labels ──
        gnn = self._train_gnn(data, labels)
        data_dev = data.to(self.device)

        gnn.eval()
        with torch.no_grad():
            gnn_logits = gnn(data_dev)
            gnn_probs = F.softmax(gnn_logits, dim=1)
            gnn_preds = gnn_logits.argmax(dim=1)

        # ── Step 2: Ensemble voting (LoCLE-style) ──
        # Accumulate this round's logits and compute weighted ensemble
        self._ensemble_logits.append(gnn_logits.detach().cpu())

        R = len(self._ensemble_logits)
        alpha = self.ensemble_alpha
        ensemble = torch.zeros_like(self._ensemble_logits[0])
        for i, logits_i in enumerate(self._ensemble_logits):
            w = (1 - alpha) * (alpha ** (R - 1 - i)) / (1 - alpha ** R)
            ensemble += w * logits_i
        # L1 normalize (matching LoCLE) — use directly as confidence
        ensemble = F.normalize(ensemble, dim=-1, p=1)
        ensemble_conf, ensemble_preds = ensemble.max(dim=1)

        # ── Step 3: Dual-score selection using ENSEMBLE (not single GNN) ──
        labeled_mask = self._build_train_mask(labels, num_nodes).to(self.device)
        selector = NoiseAwareSelector(
            noise_matrix=self._effective_noise_matrix,
            num_classes=self.num_classes,
            seed=self.seed,
            select_downweight_floor=self.select_downweight_floor,
            select_min_class_frac=self.select_min_class_frac,
        )

        # Self-label aggressively — LoCLE does ~120 per round on Cora.
        # Multiply by `self_label_multiplier` (default 3 → ~120/round on Cora).
        slm = getattr(self, 'self_label_multiplier', 3)
        n_confident = max(10, int(round_budget * slm))
        n_inconfident = min(round_budget, self.budget.remaining)

        confident_ids, inconfident_ids = selector.measure_confidence(
            ensemble.to(self.device), data.edge_index.to(self.device),
            labeled_mask, n_confident=n_confident, n_inconfident=n_inconfident,
        )

        # ── Step 4: Self-label using ENSEMBLE predictions (from round 1+) ──
        # Self-label the most confident unlabeled nodes from the ensemble
        n_self_labeled = 0
        for nid in confident_ids:
            if nid not in labels:
                labels[nid] = int(ensemble_preds[nid].item())
                n_self_labeled += 1
        if n_self_labeled > 0:
            logger.info("  Self-labeled %d confident nodes via ensemble (round %d)", n_self_labeled, round_num)

        # ── Step 5: Query LLM on inconfident nodes ──
        new_annotations = self._annotate_nodes(inconfident_ids, data.raw_texts)
        for nid, lbl in new_annotations.items():
            labels[nid] = lbl
        logger.info("  LLM-annotated %d inconfident nodes (budget cost)", len(new_annotations))

        # ── Step 6: Optional graph rewiring ──

        # ── Step 7: Compute flip rate ──
        if self._prev_gnn_preds is not None:
            flip_count = int((gnn_preds.cpu() != self._prev_gnn_preds.cpu()).sum().item())
            flip_rate = flip_count / num_nodes
        else:
            flip_count = num_nodes
            flip_rate = 1.0

        self._prev_gnn_preds = gnn_preds.cpu().clone()

        new_labeled = len(labels) - len(prev_labels)
        label_flips = sum(
            1 for nid in labels if nid in prev_labels and labels[nid] != prev_labels[nid]
        )
        logger.info(
            "Round %d: %d labeled (+%d new, %d relabeled), "
            "GNN flip_rate=%.4f (%d/%d), self-labeled=%d, LLM-queried=%d",
            round_num, len(labels), new_labeled, label_flips,
            flip_rate, flip_count, num_nodes,
            n_self_labeled, len(new_annotations),
        )

        return labels, flip_rate

    # ── Optional: Preference training ────────────────────────


    # ── Optional: Local LLM voting ─────────────────────────


    # ── Optional: Graph rewiring ─────────────────────────────


    # ── Full-graph LP denoising ──────────────────────────────


    def _correct_seeds_with_neighborhood(self, data, labels, all_cached, denoised_labels=None):
        """
        Correct seed labels using neighborhood LLM annotations and/or denoised labels.

        For each seed node:
        1. Check what its neighbors' cached LLM annotations say (majority vote)
        2. If denoised labels available, also check denoised prediction
        3. If seed label disagrees with BOTH neighborhood majority AND denoised label → correct

        Returns:
            corrected labels dict, number corrected
        """
        from collections import defaultdict, Counter

        num_nodes = data.x.size(0)
        edge_index = data.edge_index.cpu()
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()

        adj = defaultdict(list)
        for s, d in zip(src, dst):
            adj[int(s)].append(int(d))

        n_corrected = 0
        threshold = self.neighborhood_correction_threshold

        for nid, lbl in list(labels.items()):
            neighbors = adj.get(nid, [])
            if len(neighbors) < 2:
                continue

            # Get neighbor LLM annotations from cache
            nbr_labels = []
            for n in neighbors:
                if n in all_cached:
                    nbr_labels.append(all_cached[n][0])

            if len(nbr_labels) < 2:
                continue

            # Compute neighborhood majority
            counts = Counter(nbr_labels)
            majority_label, majority_count = counts.most_common(1)[0]
            agreement_rate = majority_count / len(nbr_labels)

            # Correction: neighborhood consensus + denoised agreement
            if agreement_rate >= threshold and majority_label != lbl:
                if denoised_labels is not None:
                    if denoised_labels[nid] == majority_label:
                        labels[nid] = majority_label
                        n_corrected += 1
                elif agreement_rate >= 0.65:
                    labels[nid] = majority_label
                    n_corrected += 1

        logger.info(
            "Neighborhood correction: %d/%d seed labels corrected (threshold=%.2f)",
            n_corrected, len(labels), threshold,
        )
        return labels, n_corrected

    # ── NAEW: Noise-Aware Edge Weighting for GNN training ──


    # ── Warmup Loss Denoising (DivideMix-inspired, ICLR 2020) ──


    # ── DMCR: Dual-Model Co-Refinement (novel contribution) ──




    # ── Active Learning Methods (NEW — exp180+) ─────────────



    def _run_self_training_loop(self, data, labels, ground_truth_y=None):
        """
        GNN self-training with dual confidence metric (entropy + harmonicity).

        When DMCR is enabled, also trains a feature-only MLP and uses
        MLP-GNN agreement for seed quality scoring, seed correction,
        and pseudo-label filtering (quad consensus: GNN+LP+MLP+harmonicity).

        Returns (labels, round_metrics).
        """
        num_nodes = data.x.size(0)
        round_metrics = []
        if not hasattr(self, '_pseudo_conf'):
            self._pseudo_conf = {}

        # DMCR: Multi-signal seed denoising BEFORE self-training
        dmcr_quality = None
        mlp_preds_cache = None
        mlp_probs_cache = None

        for round_num in range(1, self.active_rounds + 1):
            logger.info("=== Self-Training Round %d/%d ===", round_num, self.active_rounds)
            prev_labels = dict(labels)

            # Train GNN with confidence × class-balanced × DMCR quality weights
            nw = torch.ones(num_nodes, dtype=torch.float32)
            ann_conf = getattr(self, '_annotation_conf', {})
            for nid in labels:
                if nid in ann_conf:
                    nw[nid] = max(0.1, ann_conf[nid])
                elif nid in self._pseudo_conf:
                    nw[nid] = max(0.1, self._pseudo_conf[nid])

            # DMCR quality: multiply by instance-level quality score
            if dmcr_quality is not None:
                for nid in labels:
                    nw[nid] *= float(dmcr_quality[nid])

            # Class-balanced weights: correct for class bias in GPT labels
            from collections import Counter
            class_counts = Counter(labels.values())
            n_labeled = len(labels)
            n_classes_seen = max(len(class_counts), 1)
            for nid, lbl in labels.items():
                count = class_counts.get(lbl, 1)
                class_balance = n_labeled / (n_classes_seen * count)
                nw[nid] *= class_balance

            # Normalize to [0, 1]
            labeled_nids = list(labels.keys())
            if labeled_nids:
                w_vals = nw[labeled_nids]
                w_min, w_max = w_vals.min(), w_vals.max()
                if w_max > w_min:
                    for nid in labeled_nids:
                        nw[nid] = (nw[nid] - w_min) / (w_max - w_min)
            data._node_weights = nw

            gnn = self._train_gnn(data, labels)

            # SGR: one-shot graph refinement after first GNN training

            # NCL: noise-aware label correction after first GNN training

            gnn.eval()
            data_dev = data.to(self.device)
            with torch.no_grad():
                logits = gnn(data_dev)
                gnn_probs = F.softmax(logits, dim=1).cpu()
                gnn_preds = gnn_probs.argmax(dim=1)
                gnn_conf = gnn_probs.max(dim=1).values

            # NAEW: Update edge weights from GNN predictions for next round

            # Get MoDis disagreement from training dynamics
            modis = getattr(gnn, '_modis_disagreement', None)

            # Compute dual confidence: entropy + harmonicity
            entropy = -(gnn_probs * torch.log(gnn_probs + 1e-8)).sum(dim=1)
            # Harmonicity: how much does this node disagree with neighbors?
            from src.node_selection.noise_aware_selector import _compute_harmonicity
            harmonicity = _compute_harmonicity(gnn_probs, data.edge_index)

            # Seed refinement (inspired by R²LP, KDD 2024):
            # Quick LP pass → take high-confidence GNN+LP consensus → expand seeds
            if self.use_naap and round_num == 1:
                quick_w = self._compute_class_balanced_weights(labels, num_nodes)
                quick_lp = self._sparse_lp(
                    labels, quick_w, data.edge_index.cpu(), num_nodes,
                    alpha=0.5, iters=10,
                )
                quick_preds = quick_lp.argmax(dim=1)
                quick_conf = quick_lp.max(dim=1).values
                n_refined = 0
                for nid in range(num_nodes):
                    if nid not in labels:
                        if (int(gnn_preds[nid]) == int(quick_preds[nid]) and
                            float(gnn_conf[nid]) >= 0.9 and float(quick_conf[nid]) >= 0.5):
                            labels[nid] = int(gnn_preds[nid])
                            if not hasattr(self, '_pseudo_conf'):
                                self._pseudo_conf = {}
                            self._pseudo_conf[nid] = float(gnn_conf[nid])
                            n_refined += 1
                if n_refined > 0:
                    logger.info("Seed refinement (R²LP): added %d high-conf seeds", n_refined)

            # Label propagation: NAAP or standard LP
            # Recompute DUAL-LP confidence from the freshest label set so NAAP
            # sees the current view-disagreement signal.
            self._maybe_recompute_dual_lp(data, labels, stage_tag="refinement")
            weights = self._compute_class_balanced_weights(labels, num_nodes)
            if self.use_naap and self.noise_matrix is not None:
                from src.noise_handling.naap import naap_propagate
                edge_np = data.edge_index.cpu().numpy()
                # Phase 17: pass cluster-conditional T_c if loaded
                lp_np = naap_propagate(
                    labels, weights.numpy(), edge_np, num_nodes, self.num_classes,
                    self.noise_matrix,
                    alpha_base=self.naap_alpha_base, beta=self.naap_beta,
                    gamma=self.naap_gamma, k_max=self.naap_k_max,
                    patience=self.naap_patience,
                    gnn_entropy=entropy.numpy(),
                    T_c=self.T_c if self.use_cluster_T_c else None,
                    cluster_id=self.cluster_id if self.use_cluster_T_c else None,
                )
                lp_soft = torch.from_numpy(lp_np).float()
            else:
                lp_soft = self._sparse_lp(
                    labels, weights, data.edge_index.cpu(), num_nodes,
                    alpha=self.lp_alpha, iters=self.lp_iters,
                )
            lp_preds = lp_soft.argmax(dim=1)
            lp_conf = lp_soft.max(dim=1).values

            # Soft Label Distillation: store NAAP/LP soft predictions as training
            # targets for the next round's GNN. The GNN trains with KL divergence
            # on these soft labels instead of hard CE on noisy labels.
            # This preserves uncertainty and is inherently noise-robust.

            # Dual confidence pseudo-labeling with class-conditional thresholds:
            # Combine entropy + harmonicity scoring (exp220) with per-class
            # noise-aware thresholds (exp216)
            from collections import defaultdict

            # Class-conditional thresholds from noise matrix
            class_thresholds = {}
            if self.per_class_accuracy is not None:
                for c in range(self.num_classes):
                    acc_c = self.per_class_accuracy[c]
                    class_thresholds[c] = max(0.3, min(0.7, 1.0 - acc_c * 0.7))

            # DMCR: retrain MLP each round on expanded label set for fresh predictions

            class_candidates = defaultdict(list)
            for nid in range(num_nodes):
                if nid in labels:
                    continue
                gc = float(gnn_conf[nid])
                gp = int(gnn_preds[nid])
                lp_pred = int(lp_preds[nid])
                ent = float(entropy[nid])
                harm = float(harmonicity[nid])

                # Class-conditional GNN confidence threshold
                ct = class_thresholds.get(gp, 0.5)
                # may18 v8: cluster-conditional T_c GATE on self-training expansion.
                # Distinct from the existing use_tc_self_training (which reweights
                # SCORE post-consensus). Here we modify the THRESHOLD pre-consensus:
                # high-noise clusters (low T_c[k, gp, gp]) require higher GNN
                # confidence to add a pseudo-label; clean clusters can be expanded
                # with the base threshold. Targets pool QUALITY without uniformly
                # shrinking the pool.
                if (getattr(self, 'use_tc_expansion_gate', False)
                        and self.T_c is not None
                        and self.cluster_id is not None
                        and 0 <= int(nid) < len(self.cluster_id)):
                    k = int(self.cluster_id[nid])
                    tc_diag = float(self.T_c[k, gp, gp])
                    alpha = float(getattr(self, 'tc_expansion_gate_alpha', 0.3))
                    # Add a per-cluster penalty proportional to (1 - tc_diag).
                    # Clean cluster (tc_diag near 1): ct unchanged.
                    # Noisy cluster (tc_diag near 0): ct rises by alpha.
                    ct = min(0.99, ct + alpha * (1.0 - tc_diag))
                # may18 v5: when use_lp_pseudo_labels=True, drive pseudo-labeling
                # from the (T_c-aware NAAP) LP signal directly, ignoring the GNN
                # consensus check. LP doesn't overfit noisy labels the way the
                # GNN does. We still gate by LP confidence and use the LP's
                # predicted class as the pseudo-label.
                use_lp_pl = bool(getattr(self, 'use_lp_pseudo_labels', False))
                if use_lp_pl:
                    lc = float(lp_conf[nid])
                    if lc < ct:
                        continue
                    # Replace GNN-derived prediction with LP-derived one
                    gp = int(lp_pred)
                    gc = lc
                    pred_is_consensus = True
                else:
                    pred_is_consensus = (gp == lp_pred) and (gc >= ct)
                # DMCR quad consensus: GNN + LP + MLP + harmonicity
                # Standard: GNN confident + LP agrees (or LP-only when use_lp_pseudo_labels)
                if pred_is_consensus:
                    # Score: confidence - entropy - harmonicity - MoDis disagreement
                    score = gc - 0.2 * ent - 0.1 * harm
                    if modis is not None:
                        score -= 0.1 * float(modis[nid])

                    # DMCR bonus/penalty: MLP agreement boosts score
                    if mlp_preds_cache is not None:
                        mlp_pred = int(mlp_preds_cache[nid])
                        if mlp_pred == gp:
                            # All three models agree → strong signal, boost score
                            score += 0.15
                        else:
                            # MLP disagrees → uncertain, penalize score
                            score -= 0.15

                    # may18 v5: cluster-conditional T_c weighting of self-training
                    # score. Boosts pseudo-labels that fall in clusters where the
                    # LLM is reliable for the assigned class; suppresses them in
                    # unreliable clusters. Most training labels come from this
                    # self-training step, so T_c here should have larger effect
                    # than in ILC/loss alone.
                    if (getattr(self, 'use_tc_self_training', False)
                            and self.T_c is not None
                            and self.cluster_id is not None
                            and 0 <= int(nid) < len(self.cluster_id)):
                        k = int(self.cluster_id[nid])
                        tc_diag = float(self.T_c[k, int(gp), int(gp)])
                        # Multiply by (epsilon + tc_diag); epsilon avoids zero.
                        score = score * (0.1 + tc_diag)

                    class_candidates[gp].append((nid, score))

            # Class-balanced pseudo-labeling
            n_unlabeled = num_nodes - len(labels)
            # may18 v4: optional cap per class. When self_train_cap_per_class>0,
            # take only the top-K most-confident pseudo-labels per class per
            # round, instead of (effectively) all candidates. Keeps the pool
            # small and high quality.
            cap = int(getattr(self, 'self_train_cap_per_class', 0))
            if cap > 0:
                max_per_class = cap
            else:
                max_per_class = max(10, n_unlabeled // self.num_classes)
            n_added = 0
            for cls in range(self.num_classes):
                cands = class_candidates.get(cls, [])
                cands.sort(key=lambda x: -x[1])
                for nid, score in cands[:max_per_class]:
                    labels[nid] = cls
                    self._pseudo_conf[nid] = max(0.1, score)
                    n_added += 1

            flips = sum(1 for nid in prev_labels if nid in labels and labels[nid] != prev_labels[nid])
            flip_rate = flips / max(len(prev_labels), 1)

            round_accuracy = None
            if ground_truth_y is not None:
                gt = ground_truth_y.cpu()
                round_accuracy = float((gnn_preds == gt).float().mean())

            logger.info(
                "Self-training round %d: %d labeled (+%d new), flip_rate=%.4f%s",
                round_num, len(labels), n_added, flip_rate,
                f", gnn_acc={round_accuracy:.4f}" if round_accuracy is not None else "",
            )

            round_metrics.append({
                "round": round_num,
                "num_labeled": len(labels),
                "n_added": n_added,
                "flip_rate": flip_rate,
                "gnn_accuracy": round_accuracy,
            })

            if n_added == 0:
                logger.info("No new pseudo-labels added, stopping self-training")
                break
            if flip_rate < 0.005 and round_num > 1:
                logger.info("Self-training converged (flip_rate=%.4f)", flip_rate)
                break

        # Coverage expansion: fill remaining unlabeled nodes with LP predictions.
        # On Cora, ~15% of nodes (418/2708) fail the consensus filter and
        # remain unlabeled. The final GNN has no training signal for these.
        # Even weak LP predictions are better than no labels.
        n_unlabeled = num_nodes - len(labels)
        if n_unlabeled > 0 and n_unlabeled < num_nodes * 0.3:
            # Run final LP to get predictions for remaining nodes
            self._maybe_recompute_dual_lp(data, labels, stage_tag="coverage")
            final_weights = self._compute_class_balanced_weights(labels, num_nodes)
            if self.use_naap and self.noise_matrix is not None:
                from src.noise_handling.naap import naap_propagate
                edge_np = data.edge_index.cpu().numpy()
                final_lp = naap_propagate(
                    labels, final_weights.numpy(), edge_np, num_nodes, self.num_classes,
                    self.noise_matrix,
                    alpha_base=self.naap_alpha_base, beta=self.naap_beta,
                    gamma=self.naap_gamma, k_max=self.naap_k_max,
                    patience=self.naap_patience,
                    T_c=self.T_c if self.use_cluster_T_c else None,
                    cluster_id=self.cluster_id if self.use_cluster_T_c else None,
                )
                final_lp = torch.from_numpy(final_lp).float()
            else:
                final_lp = self._sparse_lp(
                    labels, final_weights, data.edge_index.cpu(), num_nodes,
                    alpha=self.lp_alpha, iters=self.lp_iters,
                )

            # Add LP predictions for unlabeled nodes with low weight
            n_expanded = 0
            for nid in range(num_nodes):
                if nid not in labels:
                    lp_pred = int(final_lp[nid].argmax())
                    lp_conf = float(final_lp[nid].max())
                    if lp_conf > 0.01:  # minimal filter
                        labels[nid] = lp_pred
                        self._pseudo_conf[nid] = 0.1  # very low weight
                        n_expanded += 1

            if n_expanded > 0:
                logger.info("Coverage expansion: filled %d remaining unlabeled nodes "
                           "(total coverage: %d/%d = %.1f%%)",
                           n_expanded, len(labels), num_nodes,
                           100 * len(labels) / num_nodes)

        return labels, round_metrics

    def _run_active_learning_path(self, data, labels, noise_matrix, ground_truth_y=None):
        """
        Active learning pipeline path (exp180+).

        Two modes:
        - use_local_llm_training=True: Fine-tune local LLM, use for annotation
        - use_local_llm_training=False: GNN self-training + LP expansion (zero cost)

        Returns (labels, noise_matrix, metrics) matching run()'s return format.
        """
        metrics = {"rounds": []}
        num_nodes = data.x.size(0)

        # DIAG (env-guarded, off by default): dump the active-selected Stage-I
        # seed labels (node_id -> LLM label + conf) for offline analysis. Uses
        # only the selected set (no cache leak). No effect on the pipeline.
        import os as _os
        if _os.environ.get("CANE_DUMP_SEEDS"):
            import json as _json
            _ac = getattr(self, '_annotation_conf', {})
            _dump = {int(n): [int(l), float(_ac.get(n, 0.5))] for n, l in labels.items()}
            _p = _os.environ["CANE_DUMP_SEEDS"]
            _json.dump(_dump, open(_p, "w"))
            logger.info("CANE_DUMP_SEEDS: wrote %d Stage-I seeds to %s", len(_dump), _p)

        # Self-training path (no local LLM)
        logger.info("=" * 70)
        logger.info("SELF-TRAINING: %d rounds, GNN+LP consensus (zero cost)",
                    self.active_rounds)
        logger.info("=" * 70)

        # ── Path 1: opt-in LoCLE iterative refinement (replaces self-training) ──
        labels, st_metrics = self._run_self_training_loop(
            data, labels, ground_truth_y,
        )
        metrics["rounds"] = st_metrics
        metrics["local_llm_annotations"] = 0

        # Final GNN training on all accumulated labels
        logger.info("=" * 70)
        logger.info("FINAL GNN: %d labels (%.1f%% coverage)",
                    len(labels), 100 * len(labels) / num_nodes)
        logger.info("=" * 70)

        # Set per-node confidence weights for final training
        nw = torch.ones(num_nodes, dtype=torch.float32)
        ann_conf = getattr(self, '_annotation_conf', {})
        pseudo_conf = getattr(self, '_pseudo_conf', {})
        for nid in labels:
            if nid in ann_conf:
                nw[nid] = max(0.1, ann_conf[nid])
            elif nid in pseudo_conf:
                nw[nid] = max(0.1, pseudo_conf[nid])
        labeled_nids = list(labels.keys())
        if labeled_nids:
            w_vals = nw[labeled_nids]
            w_min, w_max = w_vals.min(), w_vals.max()
            if w_max > w_min:
                for nid in labeled_nids:
                    nw[nid] = (nw[nid] - w_min) / (w_max - w_min)
        data._node_weights = nw

        # Loss-based label pruning: remove high-loss (likely wrong) pseudo-labels
        # Uses DivideMix insight (Li et al., ICLR 2020): after warmup, fit GMM
        # on per-node losses. If clearly bimodal → prune noisy component.
        # If unimodal → keep all labels (dense graphs don't need pruning).
        if getattr(self, 'use_loss_pruning', False) and len(labels) > 100:
            # Pre-check: skip warmup for large/dense graphs (avoid GPU noise + data loss)
            # GMM pruning helps small sparse graphs (Cora, CiteSeer) but hurts larger ones
            num_nodes_pre = data.x.size(0)
            num_edges_pre = data.edge_index.size(1)
            pre_avg_degree = num_edges_pre / max(num_nodes_pre, 1)
            if num_nodes_pre >= 5000 or pre_avg_degree >= 10:
                logger.info("GMM pruning: skipped (nodes=%d, avg_degree=%.1f)",
                           num_nodes_pre, pre_avg_degree)
            else:
                logger.info("=== GMM-Adaptive Loss Pruning (DivideMix) ===")

                # Save random state before warmup training (don't pollute main RNG)
                rng_state = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                np_rng = np.random.get_state()

                # Train GNN with warmup loss capture
                data._warmup_loss_epoch = 50
                warmup_gnn = self._train_gnn(data, labels)
                data._warmup_loss_epoch = 0

                # Restore random state (warmup was just for loss analysis)
                torch.set_rng_state(rng_state)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state(cuda_rng)
                np.random.set_state(np_rng)

                warmup_loss = getattr(warmup_gnn, '_warmup_per_node_loss', None)
                if warmup_loss is not None:
                    from sklearn.mixture import GaussianMixture

                    # Collect per-node losses for pseudo-labels only
                    ann_conf = getattr(self, '_annotation_conf', {})
                    pseudo_losses = []
                    pseudo_nids = []
                    for nid in labels:
                        if nid not in ann_conf:
                            pseudo_losses.append(float(warmup_loss[nid]))
                            pseudo_nids.append(nid)

                    if len(pseudo_losses) > 50:
                        losses_arr = np.array(pseudo_losses).reshape(-1, 1)
                        gmm = GaussianMixture(n_components=2, random_state=self.seed)
                        gmm.fit(losses_arr)
                        means = gmm.means_.flatten()
                        clean_comp = int(np.argmin(means))
                        noisy_comp = 1 - clean_comp
                        mean_diff = abs(means[0] - means[1])
                        std_avg = np.sqrt(gmm.covariances_.flatten().mean())
                        separation = mean_diff / (std_avg + 1e-8)
                        logger.info("GMM: clean_mean=%.4f, noisy_mean=%.4f, separation=%.2f",
                                   means[clean_comp], means[noisy_comp], separation)
                        max_prune_frac = 0.35
                        if separation > 1.3:
                            probs = gmm.predict_proba(losses_arr)
                            noisy_prob = probs[:, noisy_comp]
                            noisy_ranked = sorted(enumerate(noisy_prob), key=lambda x: -x[1])
                            max_prune = int(len(pseudo_nids) * max_prune_frac)
                            n_pruned = 0
                            for rank_idx, (i, prob) in enumerate(noisy_ranked):
                                if prob <= 0.5 or n_pruned >= max_prune:
                                    break
                                nid = pseudo_nids[i]
                                del labels[nid]
                                n_pruned += 1
                            if ground_truth_y is not None:
                                gt = ground_truth_y.cpu()
                                correct = sum(1 for nid, lbl in labels.items() if lbl == int(gt[nid]))
                                logger.info("GMM pruning: removed %d labels (sep=%.2f), acc=%.1f%%",
                                           n_pruned, separation, 100 * correct / len(labels))
                            else:
                                logger.info("GMM pruning: removed %d labels (sep=%.2f), %d remaining",
                                           n_pruned, separation, len(labels))
                        else:
                            logger.info("GMM: separation %.2f < 1.3, skipping pruning", separation)

        # Co-teaching intersection pruning (Han et al., NeurIPS 2018).
        # Benchmarked to beat GMM on 4/5 datasets (see experiments_verification/).
        # Train two GCNs with different seeds; intersect their small-loss sets;
        # keep labels that are small-loss in BOTH.

        # may18 v6: Active Quality Probing (AQP). Verify a small number of
        # pseudo-labels via LLM, then filter the pool by per-cluster agreement
        # rate before ILC. Addresses the pool-inflation failure mode: ILC has
        # no way to know which clusters' pseudo-labels are unreliable.

        # Iterative Label Correction (ILC): fix labels where GNN + neighbors
        # agree on a different class. This is a noise-correction step AFTER
        # self-training, when labels are ~72% accurate.
        if getattr(self, 'use_ilc', False):
            labels = self._iterative_label_correction(data, labels, ground_truth_y)

        # LoCLE verbatim graph-structure learning (Rewire_GNN + EstimateAdj)
        # Ported module: src/structure_learning/locle_rewire.py
        # 'end' placement: fires here, after self-training + ILC. Rewire_GCN fits ~2k noisy labels.
        # 'pre_st' placement: fires earlier, before self-training — see _run_active_learning_path.

        # Post-self-training graph refinement via EstimateAdj
        # Key insight: labels are ~72% accurate after self-training, MUCH better
        # than initial 63%. EstimateAdj works better with cleaner labels.

        # For final GNN training: disable DropEdge (use full graph for max accuracy)
        # Self-training uses DropEdge for noise regularization; final training
        # benefits from all edges since labels are already cleaned by ILC.
        saved_dropedge = self.dropedge_rate

        # Final GNN training: multi-seed ensemble for robustness
        if getattr(self, 'use_ensemble', False):
            n_models = int(getattr(self, 'final_ensemble_n', 5))
            self._train_gnn_ensemble(data, labels, n_models=n_models)

            # may18 v3 calibration diagnostic: dump (labels, ensemble_softmax)
            # AFTER the first ensemble but BEFORE any subsequent cleaning steps.
            # This lets us compare LLM-conf vs GNN-conf as noise detectors
            # post-hoc, on the pool the pipeline actually trained on.
            diag_path = getattr(self, 'diag_dump_path', None)
            if diag_path:
                try:
                    import numpy as _np
                    smax = F.softmax(self._ensemble_final_logits, dim=1).detach().cpu().numpy()
                    n_nodes = smax.shape[0]
                    label_arr = _np.full(n_nodes, -1, dtype=_np.int64)
                    for nid, lbl in labels.items():
                        if 0 <= int(nid) < n_nodes:
                            label_arr[int(nid)] = int(lbl)
                    gold = (ground_truth_y.cpu().numpy() if ground_truth_y is not None
                            else _np.full(n_nodes, -1, dtype=_np.int64))
                    _np.savez(diag_path,
                              softmax=smax,
                              assigned_label=label_arr,
                              gold=gold)
                    logger.info("Calibration diagnostic dumped: %s "
                                "(%d labeled / %d nodes)",
                                diag_path,
                                int((label_arr >= 0).sum()), n_nodes)
                except Exception as _e:
                    logger.warning("diag_dump failed: %s", _e)

            # may17 trick 4: self-distillation final round
            if getattr(self, 'use_self_distill_final', False):
                self._self_distill_final(data, labels, ground_truth_y=ground_truth_y)

        self.dropedge_rate = saved_dropedge

        metrics["budget_summary"] = self.budget.summary()
        logger.info("Pipeline complete. %s", self.budget)

        return labels, noise_matrix, metrics




    def _estimate_crossmodal_Tc(self, labels: Dict[int, int], data, num_nodes: int):
        """Cross-modality per-(cluster,class) reliability T_c from ONLY the given
        labels (the probe). Diagonal[k,c] = mean agreement between a class-c
        labelled node's LLM label and its graph + feature-kNN neighbours' labels;
        off-diagonal spread uniformly. Unlike mode-concentration, this measures LLM
        *accuracy* rather than cluster purity, so it stays calibrated on impure
        clusters (e.g. PubMed). Returns [K,C,C] or None."""
        cid = getattr(self, 'cluster_id', None)
        if cid is None:
            return None
        cid_np = (cid.detach().cpu().numpy() if isinstance(cid, torch.Tensor)
                  else np.asarray(cid)).astype(np.int64)
        K = int(cid_np.max()) + 1
        C = int(self.num_classes)
        lab = -np.ones(num_nodes, dtype=np.int64)
        for nid, c in labels.items():
            if 0 <= nid < num_nodes and 0 <= c < C:
                lab[nid] = int(c)
        ei = data.edge_index.cpu().numpy()
        gnb: Dict[int, list] = {}
        for s, t in zip(ei[0], ei[1]):
            si, ti = int(s), int(t)
            if lab[si] >= 0 and lab[ti] >= 0:
                gnb.setdefault(si, []).append(lab[ti])
        knn = self._get_feat_knn(data, num_nodes)
        agree = np.full(num_nodes, np.nan)
        for i in range(num_nodes):
            if lab[i] < 0:
                continue
            votes = list(gnb.get(i, []))
            if knn is not None:
                votes += [int(lab[int(j)]) for j in knn[i]
                          if 0 <= int(j) < num_nodes and lab[int(j)] >= 0]
            if votes:
                agree[i] = float(np.mean([v == lab[i] for v in votes]))
        finite = np.isfinite(agree)
        glob = float(np.nanmean(agree)) if finite.any() else 0.5
        Rk = np.array([
            float(np.nanmean(agree[(cid_np == k) & finite]))
            if ((cid_np == k) & finite).any() else glob
            for k in range(K)])
        Rk = np.nan_to_num(Rk, nan=glob)
        min_sup = 3
        Tc = np.zeros((K, C, C), dtype=np.float32)
        for k in range(K):
            for c in range(C):
                sel = (cid_np == k) & (lab == c) & finite
                r = float(np.nanmean(agree[sel])) if int(sel.sum()) >= min_sup else Rk[k]
                diag = float(np.clip(r, 0.02, 0.98))
                row = np.full(C, (1.0 - diag) / max(C - 1, 1), dtype=np.float32)
                row[c] = diag
                Tc[k, c] = row
        logger.info("cross-modality T_c from %d probe labels: diag_mean=%.3f "
                    "(global agreement=%.3f, K=%d)",
                    int((lab >= 0).sum()), float(Tc.diagonal(axis1=1, axis2=2).mean()),
                    glob, K)
        # DIAG (env-guarded): dump per-probe-node (agreement, llm_label, gt) to
        # measure whether correlated neighbour mislabeling inflates agreement on
        # mislabeled nodes. GT used for MEASUREMENT ONLY; not used by the method.
        import os as _osa
        if _osa.environ.get("CANE_DUMP_AGREE") and getattr(data, "y", None) is not None:
            import json as _ja
            _gt = data.y.detach().cpu().numpy()
            _d = {int(i): [float(agree[i]), int(lab[i]), int(_gt[i])]
                  for i in range(num_nodes) if lab[i] >= 0 and np.isfinite(agree[i])}
            _ja.dump(_d, open(_osa.environ["CANE_DUMP_AGREE"], "w"))
            logger.info("CANE_DUMP_AGREE: wrote %d probe nodes", len(_d))
        return Tc

    def _set_crossmodal_Tc_from_probe(self, labels: Dict[int, int], data, num_nodes: int) -> None:
        """Build the cross-modality T_c from the probe labels and install it as
        self.T_c (+ re-derive the global noise_matrix). Called once, right after
        the probe round, so downstream selection/expansion/correction use it."""
        if self._tc_from_probe_done:
            return
        Tc = self._estimate_crossmodal_Tc(labels, data, num_nodes)
        if Tc is None:
            return
        if self.tc_global_average:
            # Ablation: broadcast the cluster-average -> class-conditional (global) T.
            Tc = np.broadcast_to(Tc.mean(axis=0, keepdims=True), Tc.shape).astype(np.float32).copy()
            logger.info("tc_global_average: collapsed T_c to its cluster-average (global)")
        self.T_c = Tc
        self._tc_from_probe_done = True


    def _iterative_label_correction(self, data, labels, ground_truth_y=None):
        """
        Iterative Label Correction (ILC): Fix noisy pseudo-labels by
        comparing GNN predictions with current labels.

        For each labeled node where the GNN CONFIDENTLY predicts a different
        class AND the majority of labeled neighbors agree with the GNN:
        → correct the label.

        Repeats until convergence (no more corrections).

        This directly reduces the ~28% error rate in pseudo-labels,
        improving final GNN training quality.
        """
        from collections import defaultdict
        num_nodes = data.x.size(0)

        # Build adjacency list
        edge_index = data.edge_index.cpu()
        src_np, dst_np = edge_index[0].numpy(), edge_index[1].numpy()
        adj = defaultdict(list)
        for s, d in zip(src_np, dst_np):
            adj[int(s)].append(int(d))

        ilc_conf_threshold = getattr(self, 'ilc_conf_threshold', 0.9)
        ilc_max_rounds = getattr(self, 'ilc_max_rounds', 5)
        ilc_neighbor_threshold = getattr(self, 'ilc_neighbor_threshold', 0.5)
        # === Phase 24: cluster-conditional ILC threshold ===
        ilc_use_tc = bool(getattr(self, 'ilc_use_cluster_T_c', False)) and \
                     self.T_c is not None and self.cluster_id is not None
        ilc_tc_scale = float(getattr(self, 'ilc_tc_scale', 0.5))

        # DUAL-LP confidence freshly recomputed for this ILC pass. Low-w nodes
        # need *fewer* neighbor agreements to flip; high-w nodes are anchored.
        self._maybe_recompute_dual_lp(data, labels, stage_tag="ilc")
        dual_lp_on = (self.use_dual_lp and self.dual_lp_apply_ilc
                      and self._dual_lp_w is not None
                      and len(self._dual_lp_w) == data.num_nodes)

        # may18 v4: verify-and-drop knobs
        use_drop = bool(getattr(self, 'use_verify_and_drop_ilc', False))
        drop_thresh = float(getattr(self, 'verify_drop_threshold', 0.3))

        # may18 v5: per-round T_c update — re-estimate cluster-conditional T_c
        # from the current label dict at the END of each ILC round, so the next
        # round's threshold uses an updated estimate.
        update_tc = bool(getattr(self, 'ilc_update_T_c_per_round', False)) \
            and ilc_use_tc and self.cluster_id is not None
        tc_update_method = str(getattr(self, 'ilc_tc_update_method', 'v3'))
        logger.info("ILC startup: update_tc=%s (flag=%s, ilc_use_tc=%s, cid=%s) method=%s",
                    update_tc,
                    bool(getattr(self, 'ilc_update_T_c_per_round', False)),
                    ilc_use_tc,
                    self.cluster_id is not None,
                    tc_update_method)
        if update_tc:
            import sys as _sys
            _scripts_root = '/gpfs/scratchfs1/jhf24001/ldn24004/new_noise'
            if _scripts_root not in _sys.path:
                _sys.path.insert(0, _scripts_root)
            from scripts.compute_labelfree_T_c import (
                compute_labelfree_T_c as _lf_v3,
                compute_labelfree_T_c_bayes as _lf_bayes,
            )

        total_corrected = 0
        total_dropped = 0
        for ilc_round in range(ilc_max_rounds):
            # Train GNN on current labels
            gnn = self._train_gnn(data, labels)
            gnn.eval()
            data_dev = data.to(self.device)
            with torch.no_grad():
                logits = gnn(data_dev)
                probs = F.softmax(logits, dim=1).cpu()
                preds = probs.argmax(dim=1)
                conf = probs.max(dim=1).values

            n_corrected = 0
            n_dropped = 0
            for nid, lbl in list(labels.items()):
                gp = int(preds[nid])
                gc = float(conf[nid])
                # may18 v4: graph endorsement of the assigned label
                # (this is the same signal Stage-3 CSR uses, applied per-iteration)
                p_at_assigned = float(probs[nid, int(lbl)])

                # ---- KEEP branch: graph endorses current label OR
                # standard ILC's no-action condition (GNN agrees or low conf)
                if gp == lbl or gc < ilc_conf_threshold:
                    if use_drop and p_at_assigned < drop_thresh:
                        # graph cannot defend this label at any reasonable
                        # confidence (very low softmax at assigned class);
                        # remove it from the pool
                        del labels[nid]
                        n_dropped += 1
                    continue

                # GNN suggests a different class with high confidence
                # ---- CORRECT branch (LoCLE-style) ----
                labeled_neighbors = [n for n in adj.get(nid, []) if n in labels]
                if len(labeled_neighbors) < 2:
                    # Not enough neighbors to verify — fall through to
                    # the optional DROP branch instead of touching this label.
                    if use_drop and p_at_assigned < drop_thresh:
                        del labels[nid]
                        n_dropped += 1
                    continue

                agree_with_gnn = sum(1 for n in labeled_neighbors if labels[n] == gp)
                if ilc_use_tc:
                    k = int(self.cluster_id[nid]) if nid < len(self.cluster_id) else 0
                    tc_diag = float(self.T_c[k, int(lbl), int(lbl)])
                    required = ilc_neighbor_threshold + ilc_tc_scale * tc_diag
                else:
                    required = ilc_neighbor_threshold
                if dual_lp_on:
                    w_i = float(self._dual_lp_w[nid])
                    required = required + self.dual_lp_ilc_scale * (w_i - 0.5)
                required = max(0.0, min(1.0, required))
                if agree_with_gnn / len(labeled_neighbors) >= required:
                    labels[nid] = gp
                    n_corrected += 1
                elif use_drop and p_at_assigned < drop_thresh:
                    # GNN suggests a class but neighbors don't corroborate, AND
                    # graph won't defend the current label → drop it
                    del labels[nid]
                    n_dropped += 1

            total_corrected += n_corrected
            total_dropped += n_dropped
            if use_drop:
                logger.info("ILC round %d: corrected %d, dropped %d (pool size %d)",
                            ilc_round + 1, n_corrected, n_dropped, len(labels))
            else:
                logger.info("ILC round %d: corrected %d labels", ilc_round + 1, n_corrected)

            # may18 v5: per-round T_c update (BEFORE convergence check so even
            # a no-op round on labels still gets a fresh T_c estimate? No —
            # if labels didn't change, T_c won't change either; skip update
            # in the converged branch).
            if update_tc and (n_corrected > 0 or n_dropped > 0):
                N = int(data.num_nodes)
                C = int(self.T_c.shape[1])
                lf_preds = np.full(N, -1, dtype=np.int64)
                for _nid, _lbl in labels.items():
                    if 0 <= _nid < N and 0 <= _lbl < C:
                        lf_preds[_nid] = int(_lbl)
                cid = self.cluster_id.astype(np.int64)
                if len(cid) != N:
                    # Cluster array size mismatch — skip this round's update.
                    pass
                else:
                    try:
                        if tc_update_method == 'bayes':
                            self.T_c = _lf_bayes(lf_preds, cid, C,
                                                  smoothing=0.5,
                                                  prior_strength=1.0,
                                                  n_em_iters=1).astype(np.float32)
                        else:
                            self.T_c = _lf_v3(lf_preds, cid, C, smoothing=0.5).astype(np.float32)
                        diag_mean = float(self.T_c.diagonal(axis1=1, axis2=2).mean())
                        logger.info("ILC round %d: T_c updated label-free (method=%s, "
                                    "mean_diag=%.3f, pool=%d)",
                                    ilc_round + 1, tc_update_method, diag_mean, len(labels))
                    except Exception as _e_tc_up:
                        logger.warning("ILC round %d: T_c update failed (%s); keeping prior T_c",
                                       ilc_round + 1, _e_tc_up)

            if n_corrected == 0 and n_dropped == 0:
                logger.info("ILC converged after %d rounds, total corrected: %d, total dropped: %d",
                           ilc_round + 1, total_corrected, total_dropped)
                break

            # Debug: check label accuracy improvement
            if ground_truth_y is not None:
                gt = ground_truth_y.cpu()
                correct = sum(1 for nid, lbl in labels.items() if lbl == int(gt[nid]))
                logger.info("ILC: label accuracy after round %d: %.1f%% (%d/%d)",
                           ilc_round + 1, 100 * correct / len(labels), correct, len(labels))

        return labels

    def _train_gnn_ensemble(self, data, labels, n_models=3):
        """
        Train multiple GNNs with different seeds and store an ensemble.

        The ensemble averages logits from multiple models, reducing
        variance and improving prediction quality (+1-2% typically).

        Tricks:
          - use_mc_dropout=True + mc_dropout_passes=N: do N stochastic
            forward passes per model with dropout ON (model.train() mode),
            average logits across passes AND models.

        The 'self.gnn' is set to the first model for API compatibility,
        but the ensemble logits are stored for evaluation.
        """
        num_nodes = data.x.size(0)
        input_dim = data.x.size(1)
        all_logits = []
        use_mc = bool(getattr(self, 'use_mc_dropout', False))
        n_mc = int(getattr(self, 'mc_dropout_passes', 10))

        for i in range(n_models):
            model_seed = self.seed * 100 + i * 7  # deterministic but varied seeds
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)

            self._train_gnn(data, labels)
            data_dev = data.to(self.device)
            if use_mc:
                self.gnn.train()  # dropout ON
                with torch.no_grad():
                    mc_logits = []
                    for _ in range(n_mc):
                        mc_logits.append(self.gnn(data_dev).cpu())
                    all_logits.append(torch.stack(mc_logits).mean(dim=0))
            else:
                self.gnn.eval()
                with torch.no_grad():
                    logits = self.gnn(data_dev)
                    all_logits.append(logits.cpu())

            if i == 0:
                first_gnn_state = self.gnn.get_state()

            logger.info("Ensemble model %d/%d trained (seed=%d, mc=%s)",
                        i + 1, n_models, model_seed,
                        f"{n_mc}x" if use_mc else "off")

        # Average logits and store as a "virtual" forward pass result
        ensemble_logits = torch.stack(all_logits).mean(dim=0)
        self._ensemble_final_logits = ensemble_logits.to(self.device)

        # Restore first model for API compatibility
        self.gnn.load_state(first_gnn_state)

        # Override the GNN's forward method to return ensemble logits
        original_forward = self.gnn.forward
        def ensemble_forward(data_in):
            return self._ensemble_final_logits
        self.gnn.forward = ensemble_forward

        logger.info("Final ensemble: %d models averaged", n_models)

    def _self_distill_final(self, data, labels, ground_truth_y=None):
        """Trick 4: self-distillation final round.

        After the ensemble produces predictions, take top-K most-confident
        UNLABELED nodes (predicted), add them to the label set as pseudo-
        labels, then retrain ONE final GNN ensemble with the expanded set.
        Typically gives +0.5-2pp because the final model sees more diverse
        supervision.
        """
        if not hasattr(self, '_ensemble_final_logits'):
            return
        n_distill = int(getattr(self, 'self_distill_n', 200))
        conf_thresh = float(getattr(self, 'self_distill_conf', 0.85))
        n_models = int(getattr(self, 'final_ensemble_n', 5))

        probs = F.softmax(self._ensemble_final_logits, dim=1).cpu()
        confs, preds = probs.max(dim=1)
        num_nodes = data.x.size(0)
        labeled = set(labels.keys())
        candidates = []
        for nid in range(num_nodes):
            if nid in labeled:
                continue
            c = float(confs[nid].item())
            if c >= conf_thresh:
                candidates.append((nid, int(preds[nid].item()), c))
        candidates.sort(key=lambda t: -t[2])
        candidates = candidates[:n_distill]
        if len(candidates) == 0:
            logger.info("Self-distill: no candidates above conf=%.2f, skipping", conf_thresh)
            return

        new_labels = dict(labels)
        for nid, pred, _ in candidates:
            new_labels[nid] = pred

        acc_msg = ""
        if ground_truth_y is not None:
            gt = ground_truth_y.cpu()
            correct = sum(1 for nid, lbl, _ in candidates if int(gt[nid]) == lbl)
            acc_msg = f", pseudo_acc={100*correct/len(candidates):.1f}%"
        logger.info(
            "Self-distill: added %d pseudo-labels (conf>=%.2f, total=%d)%s",
            len(candidates), conf_thresh, len(new_labels), acc_msg,
        )

        # Retrain ensemble with expanded label set; overwrites _ensemble_final_logits
        self._train_gnn_ensemble(data, new_labels, n_models=n_models)

    def _clean_subset_retrain(self, data, labels, ground_truth_y=None):
        """Trick 5: clean-subset retraining (with optional iteration + class balance).

        Score each labeled node by GNN softmax prob at its assigned label.
        Drop the bottom `clean_subset_drop_frac` lowest-confidence labels,
        retrain the final ensemble on the cleaner subset.

        v2 knobs:
          - clean_subset_iterations > 1: iterate drop+retrain (each round
            uses NEW ensemble's confidence to re-score remaining labels).
            Iteration k>1 drops `clean_subset_iter_drop_frac` of REMAINING.
          - clean_subset_class_balanced=True: drop bottom drop_frac% PER
            CLASS instead of bottom drop_frac% overall. Preserves class
            distribution; avoids wiping out under-represented classes.

        v3 knobs (may18):
          - clean_subset_drop_fracs=[f1, f2, ...]: filter-strength ensemble.
            Train one ensemble per drop frac (each with class_balanced /
            iterations as configured), average the resulting logits.
            Overrides clean_subset_drop_frac.
          - use_elr_final=True: turn on ELR loss ONLY for the clean-subset
            retrain (save/restore self.use_elr around the call).
        """
        if not hasattr(self, '_ensemble_final_logits'):
            return

        # v3: filter-strength ensemble path
        drop_fracs_list = getattr(self, 'clean_subset_drop_fracs', None)
        if drop_fracs_list:
            self._clean_subset_retrain_multi(
                data, labels, drop_fracs_list, ground_truth_y=ground_truth_y)
            return

        # v3: ELR ON only for the final retrain
        _saved_use_elr = getattr(self, 'use_elr', False)

        drop_frac = float(getattr(self, 'clean_subset_drop_frac', 0.30))
        if drop_frac <= 0 or drop_frac >= 1:
            self.use_elr = _saved_use_elr
            return
        n_models = int(getattr(self, 'final_ensemble_n', 5))
        n_iters = int(getattr(self, 'clean_subset_iterations', 1))
        iter_frac = float(getattr(self, 'clean_subset_iter_drop_frac', 0.30))
        class_balanced = bool(getattr(self, 'clean_subset_class_balanced', False))

        # may18 v3: noise-detection signal can be switched for ablation.
        # Options: 'ensemble' (default — softmax-at-label), 'entropy' (ensemble
        # entropy), 'per_sample_loss' (CE loss per labeled node — classical
        # noisy-label-learning signal), 'llm_conf' (LLM verbalised confidence
        # from annotation cache).
        signal = str(getattr(self, 'clean_subset_signal', 'ensemble')).lower()

        cur_labels = dict(labels)
        for it in range(n_iters):
            fr = drop_frac if it == 0 else iter_frac
            probs = F.softmax(self._ensemble_final_logits, dim=1).cpu()

            # Pre-compute alternative signals once if needed
            llm_conf_arr = None
            if signal == 'llm_conf':
                llm_conf_arr = getattr(self, '_llm_conf_arr', None)
                if llm_conf_arr is None:
                    # Try to load from annotator's cache
                    try:
                        ann = getattr(self.annotator, 'annotations', {})
                        llm_conf_arr = {}
                        for k, v in ann.items():
                            if isinstance(v, (list, tuple)) and len(v) >= 2:
                                llm_conf_arr[int(k)] = float(v[1])
                        self._llm_conf_arr = llm_conf_arr
                    except Exception:
                        logger.warning(
                            "llm_conf signal requested but annotator cache unavailable; falling back to ensemble")
                        signal = 'ensemble'
                        llm_conf_arr = None

            # Cross-modality drop signal: trust = fraction of a node's labeled
            # graph-neighbours whose (current pool) label equals the node's label.
            # Computed on the EXPANDED pool (dense neighbour evidence); a hard-drop
            # use of the validated cross-modality reliability estimate.
            _xmod_nbr = None
            if signal == 'xmod':
                from collections import defaultdict as _dd
                _ein = data.edge_index.cpu().numpy()
                _lab = {int(k): int(v) for k, v in cur_labels.items()}
                _xmod_nbr = _dd(list)
                for _s, _t in zip(_ein[0], _ein[1]):
                    _ti = int(_t)
                    if _ti in _lab:
                        _xmod_nbr[int(_s)].append(_lab[_ti])

            scored = []  # (nid, lbl, score) — HIGHER means MORE confident / MORE trusted
            for nid, lbl in cur_labels.items():
                if not (0 <= int(nid) < probs.shape[0] and 0 <= int(lbl) < probs.shape[1]):
                    continue
                p_row = probs[int(nid)]
                if signal == 'entropy':
                    # high entropy = less trusted → use NEGATIVE entropy as score
                    eps = 1e-12
                    H = -float((p_row * (p_row + eps).log()).sum().item())
                    c = -H
                elif signal == 'per_sample_loss':
                    # high loss = less trusted → use NEGATIVE log-prob as inverse trust
                    p_at = float(p_row[int(lbl)].item())
                    c = -(-1.0 * (p_at + 1e-12)) if False else p_at  # fall back to p_at — see below
                    # actually: per-sample CE loss = -log(p_at). Higher loss → lower trust.
                    # We want HIGHER score = MORE trusted, so use p_at (which is equivalent to -loss).
                    # This is the SAME as 'ensemble' for partition order! So we use a different formulation:
                    # take the max-prob (argmax confidence), not the at-label confidence:
                    c = float(p_row.max().item())  # confidence in the model's own prediction
                elif signal == 'llm_conf' and llm_conf_arr is not None:
                    c = llm_conf_arr.get(int(nid), float('nan'))
                    if not np.isfinite(c):
                        # nodes without LLM-conf entry default to mid value
                        c = 0.5
                elif signal == 'xmod' and _xmod_nbr is not None:
                    nb = _xmod_nbr.get(int(nid), [])
                    c = float(np.mean([1.0 if int(x) == int(lbl) else 0.0 for x in nb])) if nb else 0.5
                else:  # 'ensemble' (default)
                    c = float(p_row[int(lbl)].item())
                scored.append((int(nid), int(lbl), c))
            if not scored:
                return

            if class_balanced:
                # Group by class; drop bottom fr% per class
                by_class = {}
                for nid, lbl, c in scored:
                    by_class.setdefault(lbl, []).append((nid, lbl, c))
                kept_all = []
                dropped_all = []
                for cls, items in by_class.items():
                    items.sort(key=lambda t: t[2])  # asc by conf
                    nd = int(len(items) * fr)
                    dropped_all.extend(items[:nd])
                    kept_all.extend(items[nd:])
                kept = kept_all
                dropped = dropped_all
            else:
                scored.sort(key=lambda t: t[2])
                nd = int(len(scored) * fr)
                dropped = scored[:nd]
                kept = scored[nd:]

            diag = ""
            if ground_truth_y is not None and dropped and kept:
                gt = ground_truth_y.cpu()
                n_dw = sum(1 for nid, lbl, _ in dropped if int(gt[nid]) != lbl)
                n_kw = sum(1 for nid, lbl, _ in kept if int(gt[nid]) != lbl)
                diag = (f", dropped_wrong={n_dw}/{len(dropped)} "
                        f"({100*n_dw/max(1,len(dropped)):.1f}%), "
                        f"kept_wrong={n_kw}/{len(kept)} "
                        f"({100*n_kw/max(1,len(kept)):.1f}%)")

            cleaner_labels = {nid: lbl for nid, lbl, _ in kept}
            logger.info(
                "Clean-subset iter %d/%d (cb=%s): dropping %d/%d (frac=%.2f), retraining%s",
                it + 1, n_iters, class_balanced, len(dropped), len(cur_labels), fr, diag,
            )
            self._train_gnn_ensemble(data, cleaner_labels, n_models=n_models)
            cur_labels = cleaner_labels

        # Restore use_elr to its pipeline-default state
        self.use_elr = _saved_use_elr

    def _clean_subset_retrain_multi(self, data, labels, drop_fracs, ground_truth_y=None):
        """v3: Filter-strength ensemble.

        For each drop_frac in `drop_fracs`, run the standard single-frac
        clean-subset retrain (using current class_balanced / iterations /
        use_elr_final settings) and stash the resulting ensemble logits.
        Average all stashed logits to form the final `_ensemble_final_logits`.

        Diversity source: filter aggressiveness. Same architecture, same
        seed schedule per ensemble member, but the model sees a different
        cleaned subset for each drop_frac → predictions decorrelate.
        """
        saved_drop_frac = self.clean_subset_drop_frac
        saved_drop_fracs = self.clean_subset_drop_fracs
        # Disable the list-path while iterating singletons to avoid recursion
        self.clean_subset_drop_fracs = None

        base_logits = self._ensemble_final_logits.detach().clone()
        all_logits = []
        for df in drop_fracs:
            self.clean_subset_drop_frac = float(df)
            # Reset to the pre-clean-subset ensemble so every drop_frac
            # scores labels against the SAME starting model (independent
            # variants), not against the previous drop_frac's output.
            self._ensemble_final_logits = base_logits.clone()
            logger.info("Filter-strength ensemble: variant drop_frac=%.2f", df)
            self._clean_subset_retrain(
                data, labels, ground_truth_y=ground_truth_y)
            all_logits.append(self._ensemble_final_logits.detach().cpu().clone())

        # Restore knobs
        self.clean_subset_drop_frac = saved_drop_frac
        self.clean_subset_drop_fracs = saved_drop_fracs

        # Average across drop_fracs
        merged = torch.stack(all_logits).mean(dim=0).to(self.device)
        self._ensemble_final_logits = merged

        # Re-bind forward override (the inner calls already did this, but
        # the closure captures the last single-frac logits — refresh it)
        original_forward_attr = getattr(self.gnn, '_orig_forward', None)
        def ensemble_forward(data_in):
            return self._ensemble_final_logits
        self.gnn.forward = ensemble_forward

        logger.info(
            "Filter-strength ensemble: averaged %d drop_fracs %s",
            len(drop_fracs), drop_fracs)

    # ── Main entry point ─────────────────────────────────────

    def run(self, data, convergence_threshold: float = 0.01, ground_truth_y: Optional[torch.Tensor] = None):
        """
        Execute the full NoisePref pipeline.

        Args:
            data:                   PyG Data object with x, edge_index, y, raw_texts.
            convergence_threshold:  Stop if label flip rate drops below this.
            ground_truth_y:         Optional ground-truth labels for per-round accuracy
                                    logging (debugging only, NOT used by the method).

        Returns:
            labels:       Final node label assignments {node_id: label}.
            noise_matrix: Estimated C×C confusion matrix from Stage 0.
            metrics:      Dict with per-round metrics and budget summary.
        """
        logger.info(
            "NoisePref pipeline starting: %s, %d classes, budget=%d, rounds=%d",
            self.dataset_name,
            self.num_classes,
            self.budget.total_budget,
            self.total_rounds,
        )

        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Feature normalization (L2, matching LoCLE)
        data.x = F.normalize(data.x.float(), p=2, dim=-1)
        logger.info("Applied L2 feature normalization (matching LoCLE)")

        # Feature propagation: smooth features through graph structure (SGC/SIGN)
        # x' = (1-α)(D^{-1}A)^K x + α*x  — gives neighborhood context

        # Compute graph density for adaptive decisions
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)
        avg_degree = num_edges / max(num_nodes, 1)
        logger.info("Graph: %d nodes, %d edges, avg_degree=%.1f, LP alpha=%.2f",
                    num_nodes, num_edges, avg_degree, self.lp_alpha)

        # Feature-based k-NN graph augmentation (sparse graphs only)
        # k-NN helps connectivity on sparse graphs (Cora, CiteSeer) but hurts
        # dense/high-accuracy graphs (PubMed GAT dropped 4%, WikiCS already dense).
        knn_k = 5
        knn_degree_threshold = 4  # Cora=4.0 skipped (k-NN hurt in exp254)
        if avg_degree >= knn_degree_threshold:
            logger.info("Skipping k-NN augmentation (avg_degree=%.1f >= %.1f)",
                        avg_degree, knn_degree_threshold)
        elif num_nodes <= 10000:
            x_np = data.x.cpu().numpy()
            sim = x_np @ x_np.T
            np.fill_diagonal(sim, -1)
            knn_indices = np.argsort(-sim, axis=1)[:, :knn_k]
            new_src, new_dst = [], []
            for i in range(num_nodes):
                for j in knn_indices[i]:
                    new_src.append(i)
                    new_dst.append(j)
            knn_edges = torch.tensor([new_src, new_dst], dtype=torch.long)
            data.edge_index = torch.cat([data.edge_index.cpu(), knn_edges], dim=1)
            logger.info("k-NN augmentation: added %d edges (k=%d)", knn_edges.size(1), knn_k)
        else:
            x_np = data.x.cpu().numpy()
            batch_size = 1000
            new_src, new_dst = [], []
            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)
                batch_sim = x_np[start:end] @ x_np.T
                for local_i in range(end - start):
                    global_i = start + local_i
                    batch_sim[local_i, global_i] = -1
                    top_k = np.argpartition(-batch_sim[local_i], knn_k)[:knn_k]
                    for j in top_k:
                        new_src.append(global_i)
                        new_dst.append(int(j))
            knn_edges = torch.tensor([new_src, new_dst], dtype=torch.long)
            data.edge_index = torch.cat([data.edge_index.cpu(), knn_edges], dim=1)
            logger.info("k-NN augmentation: added %d edges (k=%d, batched)", knn_edges.size(1), knn_k)

        # ── Stage 0: Noise Probing (uses probing budget) ──
        # `probing_mode`:
        #   "template" (default): template-based pseudo-samples BEFORE Stage I.
        #                         Quality is biased (templates too clean).
        #   "post_stageI": skip Stage 0; probe AFTER Stage I using REAL
        #                  high-confidence annotated nodes as pseudo-samples.
        #                  More realistic noise estimate.
        probing_mode = str(getattr(self, 'probing_mode', 'template'))
        if probing_mode == "template" and self.budget.get_probing_budget() > 0:
            noise_matrix = self._run_noise_probing(data)
        else:
            if probing_mode == "post_stageI":
                logger.info("Probing deferred to post-Stage-I (mode=post_stageI)")
            else:
                logger.info("Skipping noise probing (probing_fraction=0)")
            noise_matrix = None
            self.noise_matrix = None
            self.per_class_accuracy = None
            self.confused_pairs = None

            # may19 §6.A: even without LLM probing, derive a global noise matrix
            # from the cluster-conditional T_c so the gates at gcn.py:551
            # (forward correction) and pipeline.py:3896/4091 (NAAP) can fire.
            # See __init__ for the same logic; this re-derivation is needed
            # because the reset above clobbers it.
            if (self.T_c is not None
                    and getattr(self, 'use_cluster_T_c_for_forward_correction', True)):
                T_mean = self.T_c.mean(axis=0).astype(np.float32)
                T_mean = T_mean / (T_mean.sum(axis=1, keepdims=True) + 1e-8)
                noise_matrix = T_mean
                self.noise_matrix = T_mean
                diag_mean = float(np.diag(T_mean).mean())
                off_max = float((T_mean - np.diag(np.diag(T_mean))).max())
                logger.info(
                    "Re-derived global noise_matrix from cluster T_c after probing-reset "
                    "(K=%d clusters); diag_mean=%.3f, max off-diag=%.3f",
                    self.T_c.shape[0], diag_mean, off_max,
                )

        # ── Stage I: Initial Annotation ──
        labels = self._run_initial_annotation(data)

        # ── Within-budget T_c: rebuild from ONLY the seeds queried so far ──
        # With initial_fraction=1.0 the whole B=50C budget is spent above, so the
        # accumulated LLM annotations are the complete within-budget set. Every
        # downstream T_c consumer (ILC, forward correction) now uses this estimate
        # instead of the all-node precomputed matrix (leak fix).

        # ── TVCR: two-view confident relabel of the initial pool (label-free) ──
        if getattr(self, 'use_xmod_relabel', False):
            labels = self._xmod_relabel(data, labels, data.x.size(0))

        # ── Optional: post-Stage-I noise probing with REAL anchors ──
        if probing_mode == "post_stageI" and self.budget.get_probing_budget() > 0:
            logger.info("=== Post-Stage-I Probing with real anchors (budget=%d) ===",
                        self.budget.get_probing_budget())
            ann_conf = getattr(self, '_annotation_conf', None) or {}
            # Build initial_annotations dict in expected format: {nid: (label, conf*100)}
            init_ann = {}
            for nid, lbl in labels.items():
                conf = float(ann_conf.get(nid, 0.5)) * 100.0
                init_ann[int(nid)] = (int(lbl), conf)
            try:
                constructor = PseudoSampleConstructor(
                    dataset_name=self.dataset_name,
                    class_names=self.class_names,
                    seed=self.seed,
                )
                pseudo_samples = constructor.construct(
                    data,
                    initial_annotations=init_ann,
                    n_per_class=self.n_probes_per_class,
                )
                noise_matrix, per_class_acc, confused_pairs = \
                    self.profiler.estimate_noise_matrix(
                        dataset_name=self.dataset_name,
                        class_names=self.class_names,
                        pseudo_samples=pseudo_samples,
                        n_probes_per_class=self.n_probes_per_class,
                        budget_manager=self.budget,
                        use_cache=False,
                    )
                self.noise_matrix = noise_matrix
                self.per_class_accuracy = per_class_acc
                self.confused_pairs = confused_pairs
                logger.info(
                    "Post-Stage-I probing complete: %d total samples, mean acc=%.3f, budget used=%d/%d",
                    sum(len(v) for v in pseudo_samples.values()),
                    per_class_acc.mean() if per_class_acc is not None else float("nan"),
                    self.budget._consumed_probing, self.budget._probing_budget,
                )
            except Exception as e:
                logger.exception("Post-Stage-I probing failed: %s — continuing without noise matrix", e)
                self.noise_matrix = None
                self.per_class_accuracy = None
                self.confused_pairs = None

        # ── Adaptive DropEdge: set rate based on Stage I label homophily ──
        # Uses ONLY the labels obtained from active selection (annotation visibility).
        # High homophily → low DropEdge (preserves useful signal).
        # Low homophily → high DropEdge (prevents noise propagation).
        if self.dropedge_rate > 0 and len(labels) > 20:
            label_arr = np.full(num_nodes, -1, dtype=np.int64)
            for nid, lbl in labels.items():
                if 0 <= nid < num_nodes and 0 <= lbl < self.num_classes:
                    label_arr[nid] = lbl
            s_arr = data.edge_index[0].cpu().numpy()
            d_arr = data.edge_index[1].cpu().numpy()
            both = (label_arr[s_arr] >= 0) & (label_arr[d_arr] >= 0)
            if both.sum() > 0:
                raw_homo = (label_arr[s_arr[both]] == label_arr[d_arr[both]]).mean()
                if raw_homo > 0.73:
                    adaptive_rate = self.dropedge_rate * 0.1
                    adaptive_dropout = 0.5
                else:
                    adaptive_rate = self.dropedge_rate
                    adaptive_dropout = self.gnn_dropout
                logger.info("Adaptive DropEdge: raw_homophily=%.3f, rate=%.3f (base=%.3f), "
                           "dropout=%.2f (base=%.2f)",
                           raw_homo, adaptive_rate, self.dropedge_rate,
                           adaptive_dropout, self.gnn_dropout)
                self.dropedge_rate = adaptive_rate
                self.gnn_dropout = adaptive_dropout
                self._raw_homophily = raw_homo
            else:
                logger.info("Adaptive DropEdge: too few labeled edges, using default rate=%.3f",
                           self.dropedge_rate)

        # ── Checkpoint initial labels as baseline best ──
        self._best_labels = dict(labels)
        self._best_round = 0
        self._best_quality = 0.0  # start at 0 so refinement rounds can improve
        if self.gnn is not None:
            self._best_gnn_state = self.gnn.get_state()

        # DUAL-LP: initial compute over Stage-I labels so the first refinement /
        # active round already has cross-view confidence available.
        self._maybe_recompute_dual_lp(data, labels, stage_tag="post-stageI")

        # ── LoCLE-style iterative re-annotation (live LLM, post-Stage-I) ──
        # Each stage: train GNN → find inconfident UNLABELLED nodes → LLM live-
        # query on those → update labels. Budget for this comes from the
        # remaining budget after Stage I. Default off (set by config).

        # ── TBLC: T_c Bayesian Label Correction (no extra budget) ──
        # Apply cluster-conditional Bayes to correct seed labels in-place
        # using GNN softmax × T_c[k, c, c']. Does NOT cost budget.


        # ── Active Learning Path (replaces refinement + LP-denoising) ──
        if self.use_active_learning:
            return self._run_active_learning_path(data, labels, noise_matrix, ground_truth_y)

        # ── Stage II: Iterative Refinement (legacy path) ──
        metrics = {"rounds": []}
        for round_num in range(1, self.total_rounds):
            labels, flip_rate = self._run_refinement_round(data, labels, round_num)

            # Per-round GNN accuracy (debugging only)
            round_accuracy = None
            round_label_accuracy = None
            if ground_truth_y is not None and self.gnn is not None:
                self.gnn.eval()
                data_dev = data.to(self.device)
                with torch.no_grad():
                    logits = self.gnn(data_dev)
                    preds = logits.argmax(dim=1).cpu()
                gt = ground_truth_y.cpu()
                round_accuracy = float((preds == gt).float().mean().item())
                # Also compute label quality
                correct = sum(1 for nid, lbl in labels.items() if lbl == int(gt[nid]))
                round_label_accuracy = correct / len(labels) if labels else 0.0

            round_info = {
                "round": round_num,
                "num_labeled": len(labels),
                "flip_rate": flip_rate,
                "budget_remaining": self.budget.remaining,
                "gnn_accuracy": round_accuracy,
                "label_accuracy": round_label_accuracy,
            }
            metrics["rounds"].append(round_info)

            logger.info(
                "Round %d: %d labeled, flip_rate=%.4f, budget_remaining=%d%s",
                round_num,
                len(labels),
                flip_rate,
                self.budget.remaining,
                f", gnn_acc={round_accuracy:.4f}, label_acc={round_label_accuracy:.4f}" if round_accuracy is not None else "",
            )

            # Convergence: use patience-based approach.
            # Require 2 consecutive rounds with flip_rate below threshold,
            # or use a more relaxed threshold (5%) for single-round convergence.
            if flip_rate < convergence_threshold and round_num > 1:
                self._consecutive_low_flip += 1
                if self._consecutive_low_flip >= 2:
                    logger.info(
                        "Converged at round %d (flip_rate=%.4f < %.4f for 2 consecutive rounds)",
                        round_num, flip_rate, convergence_threshold,
                    )
                    break
            elif flip_rate < convergence_threshold * 5 and round_num > 2:
                # Relaxed convergence: flip rate below 5% after round 2
                self._consecutive_low_flip += 1
                if self._consecutive_low_flip >= 2:
                    logger.info(
                        "Converged at round %d (flip_rate=%.4f, relaxed threshold)",
                        round_num, flip_rate,
                    )
                    break
            else:
                self._consecutive_low_flip = 0

            if self.budget.is_exhausted():
                logger.info("Budget exhausted after round %d", round_num)
                break

        # ── Final: LP-based label cleaning and expansion ──
        if labels:
            num_nodes = data.x.size(0)
            label_coverage = len(labels) / num_nodes
            all_cached = self.annotator.get_all_cached()

            edge_index = data.edge_index.cpu()
            src, dst = edge_index[0].numpy(), edge_index[1].numpy()

            from collections import defaultdict, Counter
            adj = defaultdict(list)
            for s, d in zip(src, dst):
                adj[int(s)].append(int(d))

            # ── NEW: Full-graph LP denoising ──
            denoised_labels = None
            denoised_conf = None
            denoised_soft = None
            denoised_trust = 0.5  # how much to trust denoised over raw LLM [0-1]

            # ── Edge weighting by denoised label agreement ──
            # Downweight edges between nodes with different denoised labels.
            # Same-class edges propagate useful signal; cross-class edges propagate noise.
            if denoised_labels is not None:
                src_e = data.edge_index[0].cpu().numpy()
                dst_e = data.edge_index[1].cpu().numpy()
                same_class = denoised_labels[src_e] == denoised_labels[dst_e]
                src_conf = denoised_conf[src_e]
                dst_conf = denoised_conf[dst_e]
                min_conf = np.minimum(src_conf, dst_conf)
                # Same-class: full weight. Cross-class: scaled by min endpoint confidence.
                edge_weights = np.where(same_class, 1.0, min_conf * 0.3 + 0.5)
                data.edge_weight = torch.from_numpy(edge_weights.astype(np.float32))
                n_downweighted = int((~same_class).sum())
                logger.info("Edge weighting: %d/%d cross-class edges downweighted (denoised-label based)",
                           n_downweighted, len(src_e))

            # ── NEW: Neighborhood-based seed correction ──
            if self.use_neighborhood_correction:
                labels, n_nbr_corrected = self._correct_seeds_with_neighborhood(
                    data, labels, all_cached, denoised_labels,
                )

            # ── Branch: high coverage → simple LP; low coverage → full pipeline ──
            if label_coverage > 0.5:
                # High coverage: most nodes already labeled. Use sparse LP cleaning.
                logger.info("High label coverage (%.1f%%) — using sparse LP cleaning",
                           label_coverage * 100)

                # Use class-balanced weights even for high coverage
                cb_weights = self._compute_class_balanced_weights(labels, num_nodes)
                soft_labels = self._sparse_lp(labels, cb_weights, data.edge_index,
                                              num_nodes, alpha=0.8, iters=20)

                lp_preds = soft_labels.argmax(dim=1)
                lp_conf = soft_labels.max(dim=1).values

                n_changed = 0
                for nid in list(labels.keys()):
                    lp_pred = int(lp_preds[nid].item())
                    if labels[nid] != lp_pred and lp_conf[nid] > 0.4:
                        labels[nid] = lp_pred
                        n_changed += 1

                logger.info("Sparse LP: changed %d/%d labels", n_changed, len(labels))
                torch.manual_seed(self.seed)
                self._train_gnn(data, labels)

                metrics["budget_summary"] = self.budget.summary()
                logger.info("Pipeline complete. %s", self.budget)
                return labels, noise_matrix, metrics

            # ── Low coverage: full denoising + self-training pipeline ──
            logger.info("Low label coverage (%.1f%%) — using full denoising pipeline",
                       label_coverage * 100)

            # ── Step 1: GNN-based seed label correction ──
            # Train a GNN on the noisy seed labels, then use its predictions
            # to correct obviously wrong annotations.
            # This is similar to LoCLE's rank_and_correct_v2 but applied to ALL seeds.
            if self.gnn is not None:
                self.gnn.eval()
                data_dev = data.to(self.device)
                with torch.no_grad():
                    gnn_logits = self.gnn(data_dev)
                    gnn_probs = F.softmax(gnn_logits, dim=1).cpu()
                    gnn_preds = gnn_logits.argmax(dim=1).cpu()
                    gnn_conf = gnn_probs.max(dim=1).values

                n_corrected_gnn = 0
                for nid in list(labels.keys()):
                    gnn_pred = int(gnn_preds[nid].item())
                    gnn_c = float(gnn_conf[nid].item())
                    if labels[nid] != gnn_pred and gnn_c > 0.8:
                        # GNN strongly disagrees — check neighbor agreement
                        neighbors = adj.get(nid, [])
                        labeled_neighbors = [n for n in neighbors if n in labels]
                        if len(labeled_neighbors) >= 2:
                            gnn_agree = sum(1 for n in labeled_neighbors
                                          if labels[n] == gnn_pred)
                            if gnn_agree / len(labeled_neighbors) > 0.5:
                                labels[nid] = gnn_pred
                                n_corrected_gnn += 1
                logger.info("GNN correction: flipped %d/%d seed labels (conf>0.8, neighbor agree>50%%)",
                           n_corrected_gnn, len(labels))

            # Compute neighbor-agreement weights for seed labels
            seed_weights = torch.ones(num_nodes, dtype=torch.float32)
            n_downweighted = 0
            for nid, lbl in labels.items():
                neighbors = adj.get(nid, [])
                labeled_neighbors = [n for n in neighbors if n in labels]
                if len(labeled_neighbors) >= 2:
                    agree = sum(1 for n in labeled_neighbors if labels[n] == lbl)
                    agreement_rate = agree / len(labeled_neighbors)
                    seed_weights[nid] = max(0.1, agreement_rate)
                    if agreement_rate < 0.3:
                        n_downweighted += 1

            # Aggressive confidence gating (inspired by LoCLE's conf threshold=70)
            n_low_conf = 0
            for nid in labels:
                if nid in all_cached:
                    _, conf = all_cached[nid]
                    # Sharp confidence gating: low conf → near-zero weight
                    if conf < 70:
                        seed_weights[nid] *= 0.05  # nearly ignore
                        n_low_conf += 1
                    elif conf < 80:
                        seed_weights[nid] *= 0.3
                    elif conf < 90:
                        seed_weights[nid] *= 0.7
                    # conf >= 90 → keep full weight
            logger.info("Confidence gating: %d/%d annotations with conf<70 (weight=0.05)",
                       n_low_conf, len(labels))

            logger.info("Neighbor denoising: %d/%d seed labels downweighted (<0.3 agreement)",
                       n_downweighted, len(labels))

            # ── Class-balanced weighting ─��
            # LLM annotations have severe class bias (e.g., Theory 2.3x over-predicted
            # on Cora, DB 1.8x on CiteSeer). Apply inverse class frequency to LP seeds
            # so each class has equal total influence during propagation.
            class_balance_weights = self._compute_class_balanced_weights(labels, num_nodes)
            seed_weights = seed_weights * class_balance_weights

            # ── Step 2+3: Sparse LP expansion + self-training loop ──
            current_labels = dict(labels)
            current_weights = seed_weights.clone()

            n_st_rounds = self.self_training_rounds
            for st_round in range(n_st_rounds):
                # ── Mid-pipeline re-denoising: after round 2, LP-denoise GNN predictions ──
                # The GNN after 2 rounds produces predictions better than raw LLM labels.
                # Re-denoising these with LP creates a cleaner label base for remaining rounds.
                if st_round in (1, 3) and self.gnn is not None:
                    self.gnn.eval()
                    data_dev_re = data.to(self.device)
                    with torch.no_grad():
                        re_logits = self.gnn(data_dev_re)
                        re_probs = F.softmax(re_logits, dim=1).cpu()
                        re_preds = re_logits.argmax(dim=1).cpu()
                        re_conf = re_probs.max(dim=1).values
                    # Build re-denoising input: GNN predictions for all nodes
                    re_dict = {nid: int(re_preds[nid].item()) for nid in range(num_nodes)}
                    re_weights = re_conf.clone()
                    # Anchor seed labels (strongest signal)
                    for nid in labels:
                        re_dict[nid] = labels[nid]
                        re_weights[nid] = max(re_weights[nid].item(), 0.9)
                    # LP denoise with moderate alpha (trust GNN but smooth errors)
                    ew_re = getattr(data, 'edge_weight', None)
                    # Adaptive re-denoising strength: noisy data needs more graph trust
                    re_homo = getattr(self, '_raw_homophily', 0.6)
                    re_alpha = 0.7 if re_homo > 0.73 else 0.4
                    re_iters = 10 if re_homo > 0.73 else 20
                    re_smooth = self._sparse_lp(re_dict, re_weights,
                                                data.edge_index, num_nodes,
                                                alpha=re_alpha, iters=re_iters,
                                                edge_weight=ew_re)
                    re_smooth_preds = re_smooth.argmax(dim=1)
                    re_smooth_conf = re_smooth.max(dim=1).values
                    # Update current labels with re-denoised predictions
                    n_re_changed = 0
                    for nid in range(num_nodes):
                        if nid not in labels:
                            new_label = int(re_smooth_preds[nid].item())
                            if current_labels.get(nid) != new_label:
                                n_re_changed += 1
                            current_labels[nid] = new_label
                            current_weights[nid] = float(re_smooth_conf[nid].item())
                    logger.info("Mid-pipeline re-denoising: %d labels updated from GNN+LP", n_re_changed)

                ew = getattr(data, 'edge_weight', None)
                soft_labels = self._sparse_lp(current_labels, current_weights,
                                              data.edge_index, num_nodes,
                                              edge_weight=ew)
                lp_preds = soft_labels.argmax(dim=1)
                lp_conf = soft_labels.max(dim=1).values

                unlabeled_confs = [float(lp_conf[n].item()) for n in range(num_nodes) if n not in labels]
                conf_median = sorted(unlabeled_confs)[len(unlabeled_confs) // 2] if unlabeled_confs else 0.0

                # Build full training set with multi-signal consensus
                # When denoised labels available: use them (graph-smoothed, class-balanced)
                # Otherwise: fallback to raw LLM labels (exp122 behavior)
                full_labels = {}
                full_weights = torch.ones(num_nodes, dtype=torch.float32)
                n_consensus = 0
                n_denoised_used = 0

                for nid in range(num_nodes):
                    if nid in labels:
                        full_labels[nid] = current_labels[nid]
                        full_weights[nid] = float(current_weights[nid])
                    elif float(lp_conf[nid].item()) >= conf_median:
                        lp_pred = int(lp_preds[nid].item())
                        lp_c = float(lp_conf[nid].item())

                        # Triple consensus: LP + raw LLM + denoised
                        llm_label = llm_c = None
                        d_label = d_conf_val = None
                        if nid in all_cached:
                            llm_label, llm_conf_raw = all_cached[nid]
                            llm_c = llm_conf_raw / 100.0
                        if denoised_labels is not None:
                            d_label = int(denoised_labels[nid])
                            d_conf_val = float(denoised_conf[nid])

                        llm_agrees = (llm_label is not None and llm_label == lp_pred)
                        den_agrees = (d_label is not None and d_label == lp_pred)

                        if llm_agrees and den_agrees:
                            # All three agree → highest confidence
                            full_labels[nid] = lp_pred
                            full_weights[nid] = min(1.0, lp_c * 1.5)
                            n_consensus += 1
                        elif llm_agrees or den_agrees:
                            # Two of three agree → medium-high confidence
                            full_labels[nid] = lp_pred
                            full_weights[nid] = min(1.0, lp_c * 1.3)
                            n_consensus += 1
                        elif llm_c is not None and llm_c > 0.85 and lp_c < 0.4:
                            full_labels[nid] = int(llm_label)
                            full_weights[nid] = llm_c * 0.5
                        elif d_conf_val is not None and d_conf_val > 0.5 and lp_c < 0.3:
                            full_labels[nid] = d_label
                            full_weights[nid] = d_conf_val * 0.4
                            n_denoised_used += 1
                        else:
                            full_labels[nid] = lp_pred
                            full_weights[nid] = lp_c
                    else:
                        # Below LP threshold: weak label from best source
                        best_label = None
                        best_weight = 0.0
                        if nid in all_cached:
                            llm_label, llm_conf_raw = all_cached[nid]
                            llm_c = llm_conf_raw / 100.0
                            if llm_c > 0.8:
                                best_label = int(llm_label)
                                best_weight = 0.15
                        if denoised_labels is not None:
                            d_label = int(denoised_labels[nid])
                            d_conf_val = float(denoised_conf[nid])
                            if best_label is not None and d_label == best_label:
                                # Both agree → boost weight
                                best_weight = 0.25
                                n_denoised_used += 1
                            elif d_conf_val > 0.3 and best_label is None:
                                best_label = d_label
                                best_weight = 0.15
                                n_denoised_used += 1
                        if best_label is not None:
                            full_labels[nid] = best_label
                            full_weights[nid] = best_weight

                if st_round == 0:
                    logger.info("  Consensus: %d nodes agree (LP+denoised/LLM), "
                               "%d denoised-only, %d total training nodes",
                               n_consensus, n_denoised_used, len(full_labels))

                # Class-balanced weighting on full training set
                cb_weights = self._compute_class_balanced_weights(full_labels, num_nodes)
                balanced_weights = full_weights * cb_weights

                # Train fresh GNN
                torch.manual_seed(self.seed + st_round)
                data_train = data.clone()
                data_train._node_weights = balanced_weights
                gnn = self._train_gnn(data_train, full_labels)

                # Get GNN predictions for next round
                gnn.eval()
                data_dev = data.to(self.device)
                with torch.no_grad():
                    logits = gnn(data_dev)
                    gnn_probs = F.softmax(logits, dim=1).cpu()
                    gnn_preds = logits.argmax(dim=1).cpu()
                    gnn_conf = gnn_probs.max(dim=1).values

                # Iterative label correction: where GNN + neighborhood agree
                # against current label, correct it for next round
                n_corrected_iter = 0
                for nid in range(num_nodes):
                    if nid in labels:
                        continue
                    gnn_pred = int(gnn_preds[nid].item())
                    gnn_c = float(gnn_conf[nid].item())
                    cur_label = current_labels.get(nid)
                    if cur_label is not None and cur_label != gnn_pred and gnn_c > 0.7:
                        neighbors = adj.get(nid, [])
                        if len(neighbors) >= 2:
                            nbr_agree_gnn = sum(1 for n in neighbors
                                              if current_labels.get(n) == gnn_pred)
                            if nbr_agree_gnn / len(neighbors) > 0.4:
                                current_labels[nid] = gnn_pred
                                current_weights[nid] = gnn_c
                                n_corrected_iter += 1

                # Update pseudo-labels from GNN predictions
                # Apply LP denoising to GNN predictions to create cleaner labels
                gnn_pred_dict = {nid: int(gnn_preds[nid].item()) for nid in range(num_nodes)}
                gnn_conf_weights = gnn_conf.clone()
                # Seeds keep their corrected labels
                for nid in labels:
                    gnn_pred_dict[nid] = labels[nid]
                    gnn_conf_weights[nid] = max(gnn_conf_weights[nid].item(), 0.8)

                # LP smooth GNN predictions (denoise them)
                smoothed = self._sparse_lp(
                    gnn_pred_dict, gnn_conf_weights,
                    data.edge_index, num_nodes,
                    alpha=0.7, iters=5,
                )
                smoothed_preds = smoothed.argmax(dim=1)

                n_updated = 0
                for nid in range(num_nodes):
                    if nid not in labels:
                        new_pred = int(smoothed_preds[nid].item())
                        if current_labels.get(nid) != new_pred:
                            n_updated += 1
                        current_labels[nid] = new_pred
                        current_weights[nid] = float(smoothed[nid].max().item())

                logger.info("Self-training round %d: %d updated, %d corrected, "
                           "GNN on %d nodes", st_round, n_updated,
                           n_corrected_iter, len(full_labels))

                if n_updated < 10:
                    logger.info("Self-training converged (<%d updates)", 10)
                    break

        # ── Post-processing: adaptive ensemble + LP smoothing ──
        # Use ensemble only when estimated annotation quality is LOW (noisy datasets).
        # For clean datasets, ensemble hurts because it disrupts well-learned patterns.
        # Smooth the final GNN's predictions using graph LP.
        # This is a lightweight post-processing that leverages graph structure
        # to correct remaining prediction errors at class boundaries.
        if self.gnn is not None and labels and denoised_labels is not None:
            num_nodes = data.x.size(0)

            # Estimate whether ensemble would help: compare GNN predictions
            # with raw LLM annotations. Low agreement = noisy → ensemble helps.
            self.gnn.eval()
            data_dev = data.to(self.device)
            with torch.no_grad():
                final_logits = self.gnn(data_dev)
                final_preds = final_logits.argmax(dim=1).cpu()

            # Compare GNN preds with raw LLM annotations
            n_agree = 0
            n_total = 0
            for nid, (llm_l, _) in all_cached.items():
                if 0 <= nid < num_nodes:
                    n_total += 1
                    if final_preds[nid].item() == llm_l:
                        n_agree += 1
            gnn_llm_agreement = n_agree / max(n_total, 1)

            # If agreement is LOW → noisy dataset → ensemble helps
            # If agreement is HIGH → clean dataset → skip ensemble
            if gnn_llm_agreement < 0.65:
                logger.info("Low GNN-LLM agreement (%.3f) — running adaptive ensemble", gnn_llm_agreement)
                # Train 3 ensemble models and average
                ensemble_logits = []
                final_labels_dict = dict(labels)
                final_w = torch.ones(num_nodes)
                for nid in final_labels_dict:
                    if nid in all_cached:
                        final_w[nid] = all_cached[nid][1] / 100.0

                cb_ens = self._compute_class_balanced_weights(final_labels_dict, num_nodes)
                for ens_i in range(3):
                    torch.manual_seed(self.seed * 10 + ens_i + 100)
                    data_ens = data.clone()
                    data_ens._node_weights = final_w * cb_ens
                    gnn_ens = self._train_gnn(data_ens, final_labels_dict)
                    gnn_ens.eval()
                    with torch.no_grad():
                        logits_ens = gnn_ens(data_dev).cpu()
                    ensemble_logits.append(logits_ens)

                avg_logits = sum(ensemble_logits) / len(ensemble_logits)
                ens_preds = avg_logits.argmax(dim=1)
                for nid in range(num_nodes):
                    labels[nid] = int(ens_preds[nid].item())
                logger.info("Ensemble: 3 models averaged")
            else:
                logger.info("High GNN-LLM agreement (%.3f) — skipping ensemble", gnn_llm_agreement)

        metrics["budget_summary"] = self.budget.summary()
        logger.info("Pipeline complete. %s", self.budget)

        return labels, noise_matrix, metrics


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)

    from src.data_loading.tag_dataset import load_dataset

    # Load Cora
    data_dir = os.path.join(repo_root, "data")
    locle_dir = os.path.join(repo_root, "baselines", "locle", "data")
    data_path = (
        data_dir
        if os.path.exists(os.path.join(data_dir, "cora_random_sbert.pt"))
        else locle_dir
    )
    data = load_dataset("cora", data_path=data_path)
    print(f"Loaded Cora: {data.x.size(0)} nodes, {data.num_classes} classes\n")

    # Mock annotator (no real API calls)
    class MockAnnotator:
        model = "mock-pipeline"
        total_api_calls = 0

        def __init__(self, class_names, seed=42):
            self.class_names = class_names
            self.num_classes = len(class_names)
            self._rng = np.random.RandomState(seed)
            self._cache = {}

        def annotate_nodes(self, node_ids, texts, force=False):
            results = {}
            for nid, text in zip(node_ids, texts):
                text_lower = text.lower()
                pred = 0
                for idx, name in enumerate(self.class_names):
                    if name.lower().replace("_", " ") in text_lower:
                        pred = idx
                        break
                if self._rng.random() < 0.15:
                    pred = self._rng.randint(0, self.num_classes)
                conf = float(self._rng.uniform(60, 95))
                results[nid] = (int(pred), conf)
                self._cache[nid] = (int(pred), conf)
                self.total_api_calls += 1
            return results

        def get_cached(self, node_ids):
            return {nid: self._cache[nid] for nid in node_ids if nid in self._cache}

    mock = MockAnnotator(data.class_names, seed=42)

    pipeline = NoisePrefPipeline(
        annotator=mock,
        dataset_name="cora",
        class_names=data.class_names,
        total_budget=140,  # 20 * 7 classes
        probing_fraction=0.1,
        initial_fraction=0.5,
        total_rounds=5,
        n_probes_per_class=20,
        cache_dir=tempfile.mkdtemp(),
        seed=0,
        gnn_epochs=50,  # fewer epochs for testing
        device="cpu",
    )

    labels, noise_matrix, metrics = pipeline.run(data)

    print(f"\n{'=' * 60}")
    print("Pipeline self-test results:")
    print(f"  Noise matrix shape: {noise_matrix.shape}")
    print(f"  Mean per-class accuracy: {np.diag(noise_matrix).mean():.3f}")
    print(f"  Final labeled nodes: {len(labels)}")
    print(f"  Budget summary: {metrics['budget_summary']}")
    print(f"  Rounds completed: {len(metrics['rounds'])}")
    print(f"  Mock API calls: {mock.total_api_calls}")

    # Verify budget was respected
    bm = pipeline.budget
    assert bm._consumed_probing <= bm._probing_budget, "Probing exceeded budget"
    assert bm._consumed <= bm.total_budget, "Total exceeded budget"

    # Verify we actually labeled some nodes
    assert len(labels) > 0, "Should have labeled some nodes"

    # Verify labels are valid
    for nid, lbl in labels.items():
        assert 0 <= lbl < data.num_classes, f"Invalid label {lbl} for node {nid}"

    # Verify GNN was trained
    assert pipeline.gnn is not None, "GNN should be trained"

    # Check round metrics
    for ri in metrics["rounds"]:
        assert "round" in ri
        assert "num_labeled" in ri
        assert "flip_rate" in ri
        assert "budget_remaining" in ri

    print("\nPipeline self-test PASS.")
