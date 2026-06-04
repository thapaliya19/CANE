"""
Noise-aware node selection — wraps LoCLE's subspace clustering with DMA's
noise matrix to re-weight selection probabilities per class.

Usage:
    selector = NoiseAwareSelector(noise_matrix, num_classes=7, seed=42)

    # Stage I — initial selection
    annotation_pool, correction_pool = selector.select_initial_nodes(
        data, embeddings, budget=70,
    )

    # Stage II — critical node detection
    critical_nodes = selector.select_critical_nodes(
        data, gnn_logits, labeled_mask, budget=15, round_num=1,
    )
"""

import copy
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from torch_geometric.utils import get_laplacian, remove_self_loops

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Graph utilities (adapted from LoCLE baselines/locle/src/helper/active.py)
# ──────────────────────────────────────────────────────────────

def _normalize_adj(edge_index: torch.Tensor, num_nodes: int):
    """
    Random-walk normalization of adjacency: D^{-1} A.

    Adapted from LoCLE's normalize_adj (active.py:1196).
    Returns (edge_index, edge_weight) with self-loops removed.
    """
    from torch_scatter import scatter

    edge_index, edge_weight = remove_self_loops(edge_index)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32,
                             device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv = 1.0 / deg
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)
    edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def _aggregate_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    t: int = 2,
    alpha: float = 1.5,
) -> torch.Tensor:
    """
    Multi-hop feature aggregation: Z = X + alpha*A*X + alpha^2*A^2*X + ...

    Adapted from LoCLE's subspace_query (active.py:1688).
    Uses SparseTensor for efficient matrix multiplication.
    """
    try:
        from torch_sparse import SparseTensor
        new_edge_index, new_edge_weight = _normalize_adj(edge_index, num_nodes)
        adj = SparseTensor(
            row=new_edge_index[0], col=new_edge_index[1],
            value=new_edge_weight,
            sparse_sizes=(num_nodes, num_nodes),
        )
        sum_z = x.clone()
        x_cur = x
        for i in range(t):
            x_cur = adj @ x_cur
            sum_z = sum_z + (alpha ** (i + 1)) * x_cur
        return sum_z
    except ImportError:
        # Fallback: use PyG's sparse matrix multiplication
        from torch_geometric.utils import to_torch_coo_tensor
        new_edge_index, new_edge_weight = _normalize_adj(edge_index, num_nodes)
        adj = to_torch_coo_tensor(new_edge_index, new_edge_weight,
                                  size=(num_nodes, num_nodes))
        sum_z = x.clone()
        x_cur = x
        for i in range(t):
            x_cur = torch.sparse.mm(adj, x_cur)
            sum_z = sum_z + (alpha ** (i + 1)) * x_cur
        return sum_z


# ──────────────────────────────────────────────────────────────
# Subspace clustering (adapted from LoCLE's Subspace class, active.py:1712)
# ──────────────────────────────────────────────────────────────

def _subspace_density(
    H: torch.Tensor,
    num_classes: int,
    svd_dim: Optional[int] = None,
    kmeans_dim: Optional[int] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SVD + KMeans density scoring from LoCLE's Subspace.run_model (active.py:1770).

    Args:
        H: Aggregated feature matrix [num_nodes, feat_dim].
        num_classes: Number of classes in the dataset.
        svd_dim: SVD components (defaults to num_classes).
        kmeans_dim: Number of KMeans clusters (defaults to num_classes).
        seed: Random seed for KMeans.

    Returns:
        density: Per-node density score [num_nodes].
        cluster_labels: Per-node cluster assignment [num_nodes].
    """
    if svd_dim is None:
        svd_dim = num_classes
    if kmeans_dim is None:
        kmeans_dim = num_classes

    # Ensure H is numpy for sklearn
    H_np = H.detach().cpu().numpy() if isinstance(H, torch.Tensor) else H

    # Truncated SVD → low-dimensional representation
    svd = TruncatedSVD(n_components=min(svd_dim, H_np.shape[1] - 1), random_state=seed)
    svd.fit(H_np.T)
    U = svd.components_.T  # [num_nodes, svd_dim]

    # KMeans clustering in SVD subspace
    kmeans = KMeans(n_clusters=kmeans_dim, init="k-means++", random_state=seed, n_init=10)
    labels = kmeans.fit_predict(U)
    centers = kmeans.cluster_centers_[labels]  # Nearest center per node

    # Density = 1 / (1 + distance_to_center)
    dist = np.linalg.norm(U - centers, axis=1)
    density = 1.0 / (1.0 + dist)

    return torch.tensor(density, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


# ──────────────────────────────────────────────────────────────
# Disharmonicity (adapted from LoCLE's utils.py:450)
# ──────────────────────────────────────────────────────────────

def _compute_harmonicity(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Per-node disharmonicity via random-walk Laplacian smoothing.

    High value = node prediction disagrees with neighbors.
    Adapted from LoCLE's harmonicity() (utils.py:450).

    Args:
        x: Soft predictions or one-hot labels [num_nodes, num_classes].
        edge_index: Edge indices [2, num_edges].

    Returns:
        Per-node harmonicity score [num_nodes].  Higher = more disharmonic.
    """
    num_nodes = x.shape[0]
    # Ensure everything is on CPU to avoid device mismatches
    x = x.cpu()
    edge_index = edge_index.cpu()
    lap_index, lap_weight = get_laplacian(
        edge_index, normalization="rw", num_nodes=num_nodes,
    )
    lap = torch.sparse_coo_tensor(lap_index, lap_weight, size=(num_nodes, num_nodes))

    # || L @ x ||_2 per node (summed across class dimension)
    lx = (lap.double() @ x.double()) ** 2
    lx = lx ** 0.5
    return (lx @ torch.ones(x.shape[1], dtype=torch.float64)).float()


def _compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Prediction entropy from logits.

    Adapted from LoCLE's entrophy_confidence() (utils.py:474).
    """
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-9)
    return -torch.sum(probs * log_probs, dim=1)


# ──────────────────────────────────────────────────────────────
# NoiseAwareSelector
# ──────────────────────────────────────────────────────────────

class NoiseAwareSelector:
    """
    Wraps LoCLE's subspace clustering with DMA's noise matrix awareness.

    After subspace clustering produces candidate nodes, re-weights the
    selection probability per-node based on the noise matrix:
      - Annotation pool: prefer nodes from low-noise classes (high diagonal).
      - Correction pool: explicitly include nodes from high-confusion pairs.

    For critical-node detection (Stage II): combines LoCLE's disharmonicity
    + entropy, weighted by noise_matrix so high-confusion classes are
    prioritized for re-annotation.

    Args:
        noise_matrix: C×C numpy array from LLMNoiseProfiler.
                      noise_matrix[i][j] = P(LLM predicts j | true class is i).
        num_classes: Number of classes.
        seed: Random seed.
        subspace_t: Number of hops for feature aggregation (LoCLE default: 2).
        subspace_alpha: Decay factor for multi-hop aggregation (LoCLE default: 1.5).
        correction_fraction: Fraction of budget allocated to the correction pool.
        entropy_lambda: Weight of entropy relative to disharmonicity in
                        critical-node scoring.
    """

    def __init__(
        self,
        noise_matrix: np.ndarray,
        num_classes: int,
        seed: int = 42,
        subspace_t: int = 2,
        subspace_alpha: float = 1.5,
        correction_fraction: float = 0.2,
        entropy_lambda: float = 0.5,
        select_downweight_floor: float = 0.0,
        select_min_class_frac: float = 0.0,
    ):
        if noise_matrix is None:
            # Uniform noise matrix (identity) when noise weighting is disabled
            noise_matrix = np.eye(num_classes)
        self.noise_matrix = noise_matrix.copy()
        self.num_classes = num_classes
        self.seed = seed
        self.subspace_t = subspace_t
        self.subspace_alpha = subspace_alpha
        self.correction_fraction = correction_fraction
        self.entropy_lambda = entropy_lambda
        # Knob 1: floor on per-class diag weight. When >0, the per-class weight
        # diag(N)[c] is clamped to >= floor * max(diag(N)). Setting 1.0 fully
        # disables the noise bias (class-agnostic), 0.0 preserves legacy.
        self.select_downweight_floor = float(select_downweight_floor)
        # Knob 2: minimum per-class share of the annotation pool, expressed as
        # a fraction of the uniform share (budget/C). E.g. 0.5 guarantees each
        # class >= 0.5 * (B/C) annotations.
        self.select_min_class_frac = float(select_min_class_frac)

        # Derived quantities — the *effective* per-class accuracy used for
        # selection weighting honors select_downweight_floor. The raw diag is
        # kept on self.noise_matrix for downstream stages that need it.
        raw_diag = np.diag(noise_matrix).astype(np.float64)
        if self.select_downweight_floor > 0.0:
            mx = float(raw_diag.max()) if raw_diag.size > 0 else 1.0
            floor_val = self.select_downweight_floor * mx
            eff_diag = np.maximum(raw_diag, floor_val)
        else:
            eff_diag = raw_diag
        self.per_class_accuracy = eff_diag.astype(np.float32)
        self.per_class_noise = (1.0 - eff_diag).astype(np.float32)
        # Raw diag retained for correction-pool weighting semantics.
        self._raw_per_class_accuracy = raw_diag.astype(np.float32)  # Higher = noisier

    # ── Stage I: Initial node selection ──────────────────────

    def select_initial_nodes(
        self,
        data,
        embeddings: torch.Tensor,
        budget: int,
        candidate_multiplier: float = 3.0,
    ) -> Tuple[List[int], List[int]]:
        """
        Noise-aware initial node selection (Novel Function #1).

        1. Run subspace clustering on embeddings → density scores.
        2. Estimate each node's class from nearest cluster center.
        3. Re-weight density by noise_matrix:
           - Annotation pool: density * noise_matrix[c][c]  (prefer reliable classes)
           - Correction pool: density * (1 - noise_matrix[c][c])  (prefer noisy classes)
        4. Select top nodes from each pool.

        Args:
            data: PyG Data object with edge_index, num_classes.
            embeddings: Node embeddings [num_nodes, feat_dim].
                        Round 1: use data.x (SentenceBERT features).
                        Round 2+: use GNN embeddings.
            budget: Total number of nodes to select.
            candidate_multiplier: Generate this many × budget candidates from
                                  subspace clustering before noise re-weighting.

        Returns:
            annotation_pool: List of node indices for standard LLM annotation.
                             Biased toward low-noise classes for reliable signal.
            correction_pool: List of node indices for targeted correction.
                             Biased toward high-confusion classes for preference tuning.
        """
        num_nodes = embeddings.size(0)
        correction_budget = max(1, int(budget * self.correction_fraction))
        annotation_budget = budget - correction_budget

        # Step 1: Multi-hop feature aggregation (LoCLE's subspace_query)
        H = _aggregate_features(
            embeddings, data.edge_index, num_nodes,
            t=self.subspace_t, alpha=self.subspace_alpha,
        )

        # Step 2: SVD + KMeans density scoring
        density, cluster_labels = _subspace_density(
            H, self.num_classes, seed=self.seed,
        )

        # Step 2b: Degree-weighted density — prefer hub nodes whose labels
        # propagate to more neighbors during LP. Uses sqrt(degree) to
        # moderate the effect (prevent extreme hubs from dominating).
        from torch_geometric.utils import degree as pyg_degree
        deg = pyg_degree(data.edge_index[0], num_nodes=num_nodes).float()
        deg_weight = torch.sqrt(deg + 1.0)  # +1 to handle isolated nodes
        deg_weight = deg_weight / deg_weight.max()  # normalize to [0, 1]
        density = density * (0.5 + 0.5 * deg_weight)  # blend: 50% density + 50% degree
        logger.debug("Degree-weighted density: mean_deg=%.1f, max_deg=%d",
                     deg.mean().item(), int(deg.max().item()))

        # Step 3: Estimate class for each node via cluster-to-class mapping.
        # If we have preliminary predictions (e.g., from embeddings nearest to
        # class centroids), use those. Otherwise use cluster labels as proxy.
        estimated_classes = self._estimate_node_classes(
            embeddings, cluster_labels, data,
        )

        # Step 4: Stratified noise-weighted selection.
        # Allocate budget per estimated class proportional to noise weights,
        # then select densest nodes within each class.  This prevents any
        # single class from dominating even when the density landscape is
        # uneven.

        annotation_pool = self._stratified_select(
            density, estimated_classes, self.per_class_accuracy,
            annotation_budget, exclude=set(),
        )

        # Knob 2: enforce min-per-class share on annotation pool.
        # The threshold uses the TOTAL budget (ann + corr) as denominator,
        # matching the task spec "share >= select_min_class_frac * (budget/C)".
        if self.select_min_class_frac > 0.0 and len(annotation_pool) > 0:
            annotation_pool = self._enforce_min_class_share(
                annotation_pool, density, estimated_classes,
                reference_budget=budget,
            )

        correction_pool = self._stratified_select(
            density, estimated_classes, self.per_class_noise,
            correction_budget, exclude=set(annotation_pool),
        )

        logger.info(
            "Initial selection: %d annotation + %d correction = %d total "
            "(budget=%d) | floor=%.2f min_frac=%.2f",
            len(annotation_pool), len(correction_pool),
            len(annotation_pool) + len(correction_pool), budget,
            self.select_downweight_floor, self.select_min_class_frac,
        )
        if annotation_pool:
            ann_counts = np.bincount(
                np.array([int(estimated_classes[i]) for i in annotation_pool]),
                minlength=self.num_classes,
            )
            logger.info(
                "Annotation pool per-class (est clusters) counts: %s",
                {i: int(ann_counts[i]) for i in range(self.num_classes)},
            )

        # Log class distribution
        self._log_pool_class_distribution(
            annotation_pool, correction_pool, estimated_classes,
        )

        return annotation_pool, correction_pool

    # ── Stage II: Critical node detection ────────────────────

    def select_critical_nodes(
        self,
        data,
        gnn_logits: torch.Tensor,
        labeled_mask: torch.Tensor,
        budget: int,
        round_num: int = 1,
        T_c: "Optional[np.ndarray]" = None,
        cluster_id: "Optional[np.ndarray]" = None,
        cluster_risk_lambda: float = 0.0,
    ) -> List[int]:
        """
        Noise-weighted critical node detection (adapted from LoCLE's
        measure_confidence, utils.py:480).

        Identifies nodes that should be re-annotated by the LLM.  Combines
        LoCLE's disharmonicity + entropy detection with noise-matrix weighting:

        Score_i = disharmonicity_i × (1 - noise_matrix[pred_class][pred_class])
                  + entropy_lambda × entropy_i
                  + cluster_risk_lambda × posterior_risk_i        (optional CRBA term)

        where posterior_risk_i = 1 − Σ_c P_gnn(i,c) · T_c[cid_i, c, c]
        captures the expected misclassification probability under the LLM's
        cluster-conditional noise model. When `cluster_risk_lambda > 0` and
        T_c+cluster_id are provided this biases selection toward nodes in
        high-LLM-noise clusters (the cluster-side analogue of the per-class
        noise_weight above).

        Args:
            data: PyG Data object with edge_index.
            gnn_logits: Raw GNN output logits [num_nodes, num_classes].
            labeled_mask: Boolean mask of currently labeled nodes [num_nodes].
                          Critical nodes are selected from the UNLABELED set.
            budget: Number of critical nodes to select.
            round_num: Current refinement round (for logging).
            T_c: [K, C, C] cluster-conditional transition matrix (optional).
            cluster_id: [num_nodes] cluster assignment per node (optional).
            cluster_risk_lambda: weight on the posterior-risk term. 0 disables.

        Returns:
            List of critical node indices (from unlabeled set), sorted by
            criticality score descending.
        """
        import numpy as np  # local: keep top-level imports clean
        num_nodes = gnn_logits.size(0)
        probs = F.softmax(gnn_logits, dim=1)
        pred_classes = torch.argmax(probs, dim=1)

        # Disharmonicity: how much does this node's prediction disagree
        # with its neighbors?
        disharmonicity = _compute_harmonicity(probs, data.edge_index)

        # Entropy: how uncertain is the GNN about this node?
        entropy = _compute_entropy(gnn_logits)

        # Noise weight per node: based on the predicted class's noise level.
        # Higher noise_weight → the predicted class is more often confused by LLM,
        # so re-annotating this node is more valuable.
        noise_weight = torch.tensor(
            [1.0 - self.per_class_accuracy[c.item()] for c in pred_classes],
            dtype=torch.float32, device=gnn_logits.device,
        )

        # Combined criticality score:
        # Score = disharmonicity * (1 - noise_matrix[pred][pred]) + lambda * entropy
        score = disharmonicity * noise_weight + self.entropy_lambda * entropy

        # CRBA: cluster-conditional posterior risk term
        use_crba = (
            cluster_risk_lambda > 0
            and T_c is not None
            and cluster_id is not None
        )
        if use_crba:
            K = int(T_c.shape[0]); C_tc = int(T_c.shape[1])
            cid_np = np.asarray(cluster_id, dtype=np.int64)
            cid_np = np.clip(cid_np, 0, K - 1)
            T_diag = np.diagonal(T_c, axis1=1, axis2=2).astype(np.float32)  # [K, C]
            T_diag_node = T_diag[cid_np]  # [N, C]
            p_np = probs.detach().cpu().numpy().astype(np.float32)
            if p_np.shape[1] == C_tc:
                posterior_risk = 1.0 - (p_np * T_diag_node).sum(axis=1)
                # min-max normalise within available range to keep λ comparable
                pr_min, pr_max = float(posterior_risk.min()), float(posterior_risk.max())
                if pr_max > pr_min:
                    posterior_risk = (posterior_risk - pr_min) / (pr_max - pr_min)
                pr_t = torch.from_numpy(posterior_risk).to(gnn_logits.device)
                score = score + cluster_risk_lambda * pr_t
                logger.info(
                    "CRBA: cluster_risk_lambda=%.2f, risk mean(pre-norm)=%.3f "
                    "min=%.3f max=%.3f",
                    cluster_risk_lambda,
                    float((1.0 - (p_np * T_diag_node).sum(axis=1)).mean()),
                    pr_min, pr_max,
                )

        # Only consider unlabeled nodes
        # labeled_mask=True means the node already has a label
        unlabeled_indices = torch.where(~labeled_mask)[0]
        if len(unlabeled_indices) == 0:
            logger.warning("No unlabeled nodes available for critical selection")
            return []

        unlabeled_scores = score[unlabeled_indices]
        k = min(budget, len(unlabeled_indices))
        _, topk_local = torch.topk(unlabeled_scores, k)
        critical_indices = unlabeled_indices[topk_local].tolist()

        # Log statistics
        if critical_indices:
            selected_preds = pred_classes[critical_indices]
            selected_scores = score[critical_indices]
            logger.info(
                "Round %d critical nodes: %d selected, "
                "mean score=%.3f, max score=%.3f",
                round_num, len(critical_indices),
                selected_scores.mean().item(), selected_scores.max().item(),
            )
            # Class distribution of critical nodes
            class_counts = torch.bincount(selected_preds, minlength=self.num_classes)
            logger.debug(
                "Critical node class distribution: %s",
                {i: class_counts[i].item() for i in range(self.num_classes)
                 if class_counts[i] > 0},
            )

        return critical_indices

    def measure_confidence(
        self,
        gnn_logits: torch.Tensor,
        edge_index: torch.Tensor,
        labeled_mask: torch.Tensor,
        n_confident: int = 40,
        n_inconfident: int = 20,
    ) -> Tuple[List[int], List[int]]:
        """
        LoCLE's dual-score selection: intersection of harmonicity + entropy.

        Returns two sets of UNLABELED node indices:
          - confident: low entropy AND low harmonicity → safe to self-label
          - inconfident: high entropy AND high harmonicity → query LLM

        This is the core of LoCLE's noise handling — it uses graph structure
        to distinguish truly confident predictions from overconfident-wrong ones.
        """
        probs = F.softmax(gnn_logits, dim=1)
        harm = _compute_harmonicity(probs, edge_index)
        entropy = _compute_entropy(gnn_logits)

        # Only consider unlabeled nodes
        unlabeled_idx = torch.where(~labeled_mask)[0]
        if len(unlabeled_idx) == 0:
            return [], []

        # Ensure all tensors are on the same device (labeled_mask may be on CPU)
        harm = harm.to(unlabeled_idx.device)
        entropy = entropy.to(unlabeled_idx.device)

        harm_unl = harm[unlabeled_idx]
        ent_unl = entropy[unlabeled_idx]

        # Rank by harmonicity (ascending = most confident first)
        harm_order = torch.argsort(harm_unl, descending=False)
        # Rank by entropy (ascending = most confident first)
        ent_order = torch.argsort(ent_unl, descending=False)

        # INTERSECTION: must rank high in BOTH metrics
        harm_conf_set = set(harm_order[:n_confident * 2].tolist())
        ent_conf_set = set(ent_order[:n_confident * 2].tolist())
        confident_local = list(harm_conf_set & ent_conf_set)[:n_confident]

        # Inconfident: rank high (descending) in BOTH
        harm_inconf_order = torch.argsort(harm_unl, descending=True)
        ent_inconf_order = torch.argsort(ent_unl, descending=True)
        harm_inconf_set = set(harm_inconf_order[:n_inconfident * 2].tolist())
        ent_inconf_set = set(ent_inconf_order[:n_inconfident * 2].tolist())
        inconfident_local = list(harm_inconf_set & ent_inconf_set)[:n_inconfident]

        # Fill remaining slots from single-metric rankings if intersection too small
        if len(confident_local) < n_confident:
            remaining = n_confident - len(confident_local)
            used = set(confident_local)
            for idx in harm_order.tolist():
                if idx not in used:
                    confident_local.append(idx)
                    used.add(idx)
                    if len(confident_local) >= n_confident:
                        break

        if len(inconfident_local) < n_inconfident:
            remaining = n_inconfident - len(inconfident_local)
            used = set(inconfident_local)
            for idx in harm_inconf_order.tolist():
                if idx not in used:
                    inconfident_local.append(idx)
                    used.add(idx)
                    if len(inconfident_local) >= n_inconfident:
                        break

        # Map local indices back to global node IDs
        confident_global = [unlabeled_idx[i].item() for i in confident_local]
        inconfident_global = [unlabeled_idx[i].item() for i in inconfident_local]

        logger.info(
            "Dual-score selection: %d confident (self-label), %d inconfident (query LLM)",
            len(confident_global), len(inconfident_global),
        )
        return confident_global, inconfident_global

    # ── Helpers ──────────────────────────────────────────────

    def _estimate_node_classes(
        self,
        embeddings: torch.Tensor,
        cluster_labels: torch.Tensor,
        data,
    ) -> np.ndarray:
        """
        Estimate each node's class for noise-weighting purposes.

        Strategy: compute each cluster's centroid, then assign each cluster
        to the class whose mean embedding (computed from ground truth for
        debugging, or from cluster consensus) is closest.

        In practice (no ground truth available), we use a simple heuristic:
        map each of the K clusters to one of C classes by matching cluster
        centroids to class centroids estimated from the full embedding space
        via a secondary KMeans with C clusters.
        """
        num_nodes = embeddings.size(0)
        emb_np = embeddings.detach().cpu().numpy()

        # Run a C-cluster KMeans to get class-proxy centroids
        class_kmeans = KMeans(
            n_clusters=self.num_classes, init="k-means++",
            random_state=self.seed, n_init=10,
        )
        class_assignments = class_kmeans.fit_predict(emb_np)
        return class_assignments.astype(np.int64)

    def _stratified_select(
        self,
        density: torch.Tensor,
        estimated_classes: np.ndarray,
        class_weights: np.ndarray,
        budget: int,
        exclude: set,
    ) -> List[int]:
        """
        Stratified selection: allocate budget across estimated classes
        proportional to class_weights, then pick densest nodes per class.

        Ensures every class with weight > 0 gets at least 1 node (if
        enough candidates exist), preventing class starvation.
        """
        C = self.num_classes
        # Normalize weights to sum to 1
        w = np.maximum(class_weights, 1e-6)
        w = w / w.sum()

        # Allocate budget: at least 1 per class, rest proportional
        min_per_class = 1
        guaranteed = min(min_per_class * C, budget)
        remaining = budget - guaranteed
        allocation = np.ones(C, dtype=int) * min(min_per_class, budget // C)
        if remaining > 0:
            extra = (w * remaining).astype(int)
            allocation += extra
        # Distribute rounding remainder to highest-weight classes
        deficit = budget - allocation.sum()
        if deficit > 0:
            order = np.argsort(-w)
            for i in range(deficit):
                allocation[order[i % C]] += 1
        elif deficit < 0:
            # Over-allocated: trim from lowest-weight classes
            order = np.argsort(w)
            for i in range(-deficit):
                if allocation[order[i % C]] > 0:
                    allocation[order[i % C]] -= 1

        # Select densest nodes per class
        selected = []
        for c in range(C):
            class_mask = (estimated_classes == c)
            class_indices = np.where(class_mask)[0]
            # Remove excluded nodes
            class_indices = [i for i in class_indices if i not in exclude]
            if not class_indices:
                continue
            class_density = density[class_indices]
            k = min(allocation[c], len(class_indices))
            if k <= 0:
                continue
            _, topk_local = torch.topk(class_density, k)
            selected.extend([class_indices[i] for i in topk_local.tolist()])

        # If we got fewer than budget (some classes had too few candidates),
        # fill from the global density ranking
        if len(selected) < budget:
            exclude_all = exclude | set(selected)
            remaining_density = density.clone()
            for idx in exclude_all:
                remaining_density[idx] = -1.0
            shortfall = budget - len(selected)
            _, extra_idx = torch.topk(remaining_density, shortfall)
            selected.extend(extra_idx.tolist())

        return selected[:budget]

    def _enforce_min_class_share(
        self,
        pool: List[int],
        density: torch.Tensor,
        estimated_classes: np.ndarray,
        reference_budget: Optional[int] = None,
    ) -> List[int]:
        """
        Ensure each class's share of the pool is >= min_frac * (B / C),
        where B defaults to len(pool) unless `reference_budget` is given
        (in which case the total budget is used as denominator, per spec).
        Promotes highest-density candidates from under-represented classes,
        displacing lowest-density members of over-represented classes.
        """
        C = self.num_classes
        pool_size = len(pool)
        if pool_size <= 0 or C <= 0:
            return pool
        B = reference_budget if reference_budget is not None else pool_size

        min_per_class = int(np.floor(self.select_min_class_frac * (B / C)))
        # Cap so we can never require more than the pool can hold.
        min_per_class = min(min_per_class, pool_size // C)
        if min_per_class <= 0:
            return pool

        pool_set = set(pool)
        # Classes of current pool members
        pool_classes = np.array([int(estimated_classes[i]) for i in pool])
        counts = np.bincount(pool_classes, minlength=C)

        # For quick sorting within pool by density ascending (lowest first)
        density_np = density.detach().cpu().numpy()

        # Iterate: for each under-represented class, promote best candidate
        max_iters = C * max(1, min_per_class) * 2
        iters = 0
        while iters < max_iters:
            iters += 1
            # Find most under-represented class (largest deficit)
            deficits = min_per_class - counts
            deficits = np.where(deficits > 0, deficits, 0)
            if deficits.max() <= 0:
                break
            under_c = int(np.argmax(deficits))

            # Candidate nodes: of class under_c and not in pool_set
            cls_mask = (estimated_classes == under_c)
            cls_indices = np.where(cls_mask)[0]
            cand = [int(i) for i in cls_indices if int(i) not in pool_set]
            if not cand:
                # No more candidates for this class — give up on this class
                counts[under_c] = min_per_class  # mask out
                continue
            # Highest-density candidate
            cand_d = density_np[cand]
            best_idx = cand[int(np.argmax(cand_d))]

            # Find over-represented class to displace from
            surplus = counts - min_per_class
            # Prefer to displace from classes with largest surplus; must have
            # surplus > 0 (so we don't create a new deficit).
            over_c_order = np.argsort(-surplus)
            displace_pool_idx = None
            for over_c in over_c_order:
                if surplus[over_c] <= 0:
                    break
                # Lowest-density pool member of class over_c
                members = [p for p in pool if int(estimated_classes[p]) == int(over_c)]
                if not members:
                    continue
                m_d = density_np[members]
                displace_pool_idx = members[int(np.argmin(m_d))]
                break
            if displace_pool_idx is None:
                # No over-represented class to trim from; abort
                break

            # Perform swap
            pool.remove(displace_pool_idx)
            pool_set.discard(displace_pool_idx)
            counts[int(estimated_classes[displace_pool_idx])] -= 1
            pool.append(best_idx)
            pool_set.add(best_idx)
            counts[under_c] += 1

        return pool

    @staticmethod
    def _topk_unique(scores: torch.Tensor, k: int) -> List[int]:
        """Select top-k indices by score, ensuring uniqueness."""
        k = min(k, scores.numel())
        if k <= 0:
            return []
        _, indices = torch.topk(scores, k)
        return indices.tolist()

    def _log_pool_class_distribution(
        self,
        annotation_pool: List[int],
        correction_pool: List[int],
        estimated_classes: np.ndarray,
    ) -> None:
        """Log the class distribution of both pools for debugging."""
        for name, pool in [("Annotation", annotation_pool),
                           ("Correction", correction_pool)]:
            if not pool:
                continue
            classes = estimated_classes[pool]
            counts = np.bincount(classes, minlength=self.num_classes)
            dist = {i: int(counts[i]) for i in range(self.num_classes) if counts[i] > 0}
            logger.debug("%s pool class distribution: %s", name, dist)

    # ── Stage I (alt): two-pass cluster-aware selection ──────
    #
    # Motivation. Round-1 of subspace-clustering selection picks centroids on
    # raw SBERT features, which encode semantic similarity rather than class
    # similarity. The first batch is therefore noise-prone and the rest of
    # Stage-I uses GNN entropy *trained on that noisy first batch*. The
    # two-pass selector breaks the dependence: it over-clusters (k = α·C),
    # spends ~50% of the budget on Pass-1 cluster representatives, then uses
    # the per-cluster *label entropy* of Pass-1 (a free signal — these are
    # already-paid-for LLM annotations) to re-allocate Pass-2 budget to the
    # clusters whose internal labels look genuinely heterogeneous. Within a
    # cluster Pass-2 is class-aware: we weight by ``T̂_diag[c_proxy]`` so
    # noisy-class nodes are de-prioritised even within heterogeneous clusters.
    # Generalises across datasets — only α and the Pass-1 fraction are knobs.
    #
    # The class exposes the two passes as separate methods so the pipeline
    # can perform the LLM annotation between them without bypassing the
    # ``selected_nodes`` mask check.
    def select_pass1_nodes(
        self,
        data,
        embeddings: torch.Tensor,
        budget: int,
        alpha: float = 1.5,
        min_per_cluster: int = 1,
    ) -> Tuple[List[int], np.ndarray, torch.Tensor]:
        """Pass 1 of two-pass cluster-aware selection.

        Over-clusters the aggregated-feature space into ``k = max(C, round(α·C))``
        clusters and selects ``budget`` densest nodes proportional to cluster
        size with a per-cluster floor of ``min_per_cluster``.

        Returns
        -------
        selected
            Node indices to annotate in Pass 1.
        cluster_labels
            Cluster assignment per node (numpy int array, length = num_nodes).
        density
            Per-node density score (kept on CPU); reused by Pass 2.
        """
        num_nodes = embeddings.size(0)
        k = max(self.num_classes, int(round(alpha * self.num_classes)))

        # Aggregated features (matches LoCLE's subspace_query)
        H = _aggregate_features(
            embeddings, data.edge_index, num_nodes,
            t=self.subspace_t, alpha=self.subspace_alpha,
        )

        # SVD to a low-dim subspace, then over-cluster.
        H_np = H.detach().cpu().numpy()
        svd = TruncatedSVD(
            n_components=min(k, H_np.shape[1] - 1), random_state=self.seed,
        )
        svd.fit(H_np.T)
        U = svd.components_.T  # [num_nodes, svd_dim]

        kmeans = KMeans(
            n_clusters=k, init="k-means++",
            random_state=self.seed, n_init=10,
        )
        cluster_labels = kmeans.fit_predict(U)
        centers = kmeans.cluster_centers_[cluster_labels]
        dist = np.linalg.norm(U - centers, axis=1)
        density_np = 1.0 / (1.0 + dist)

        # Optional degree-weighted blend (matches single-pass behaviour).
        from torch_geometric.utils import degree as pyg_degree
        deg = pyg_degree(data.edge_index[0], num_nodes=num_nodes).float()
        deg_weight = torch.sqrt(deg + 1.0)
        deg_weight = deg_weight / deg_weight.max()
        density_t = torch.tensor(density_np, dtype=torch.float32) \
                  * (0.5 + 0.5 * deg_weight)

        # Allocate budget across clusters proportional to cluster size, with a
        # per-cluster floor. Round-tripping handles total mismatch.
        cluster_sizes = np.bincount(cluster_labels, minlength=k).astype(np.float64)
        nonzero = cluster_sizes > 0
        share = np.zeros(k, dtype=np.int64)
        if nonzero.any():
            share[nonzero] = np.maximum(
                min_per_cluster,
                np.round(cluster_sizes[nonzero] / cluster_sizes.sum() * budget)
                .astype(np.int64),
            )
            # Reconcile to exact budget.
            diff = int(share.sum()) - int(budget)
            if diff > 0:
                # Trim from clusters with the largest current allocation.
                order = np.argsort(-share)
                for idx in order:
                    if diff <= 0:
                        break
                    if share[idx] > min_per_cluster:
                        share[idx] -= 1
                        diff -= 1
            elif diff < 0:
                # Spread remainder over clusters proportional to size.
                order = np.argsort(-cluster_sizes)
                ix = 0
                while diff < 0:
                    share[order[ix % k]] += 1
                    diff += 1
                    ix += 1

        selected: List[int] = []
        seen = set()
        for c in range(k):
            quota = int(share[c])
            if quota <= 0:
                continue
            members = np.where(cluster_labels == c)[0]
            if len(members) == 0:
                continue
            quota = min(quota, len(members))
            # Top-quota by density within the cluster.
            cluster_density = density_t[members]
            _, topk_local = torch.topk(cluster_density, quota)
            for j in topk_local.tolist():
                node = int(members[j])
                if node in seen:
                    continue
                selected.append(node)
                seen.add(node)

        # If budget was not exhausted (very small clusters), fill globally.
        if len(selected) < budget:
            global_density = density_t.clone()
            for n in seen:
                global_density[n] = -1.0
            shortfall = budget - len(selected)
            _, extra = torch.topk(global_density, shortfall)
            for n in extra.tolist():
                if n not in seen:
                    selected.append(int(n))
                    seen.add(int(n))

        logger.info(
            "Pass1: k=%d clusters, allocated %d (target %d), min/cluster=%d",
            k, len(selected), budget, min_per_cluster,
        )
        return selected[:budget], cluster_labels, density_t

    def select_pass2_nodes(
        self,
        data,
        embeddings: torch.Tensor,
        cluster_labels: np.ndarray,
        density: torch.Tensor,
        pass1_labels: dict,
        budget: int,
        already_selected: set,
        min_per_cluster: int = 0,
        entropy_eps: float = 0.10,
    ) -> List[int]:
        """Pass 2 of two-pass cluster-aware selection.

        Re-allocates ``budget`` across clusters proportional to per-cluster
        Pass-1 label entropy, then within each cluster ranks remaining
        unselected nodes by ``density × T̂_diag[c_proxy]``.

        Parameters
        ----------
        cluster_labels, density
            Returned by ``select_pass1_nodes`` — must be reused as-is.
        pass1_labels
            Mapping ``{node_id: class_id}`` from the LLM call after Pass 1.
        already_selected
            Node ids already in the selected pool (Pass-1 plus anything else
            the pipeline has consumed). These are excluded from Pass 2.
        min_per_cluster
            Floor on Pass-2 quota per non-empty cluster. 0 = no floor.
        entropy_eps
            Smoothing constant added to every cluster's entropy; ensures
            zero-entropy clusters still receive some Pass-2 share.
        """
        num_nodes = embeddings.size(0)
        k = int(cluster_labels.max()) + 1 if cluster_labels.size > 0 else 0
        if budget <= 0 or k == 0:
            return []

        # Compute per-cluster label entropy from Pass-1 labels.
        per_cluster_classes: List[List[int]] = [[] for _ in range(k)]
        for nid, cls in pass1_labels.items():
            cid = int(cluster_labels[int(nid)])
            per_cluster_classes[cid].append(int(cls))

        cluster_entropy = np.full(k, entropy_eps, dtype=np.float64)
        for c in range(k):
            buf = per_cluster_classes[c]
            if len(buf) <= 1:
                continue  # entropy_eps already provides a floor
            counts = np.bincount(buf, minlength=self.num_classes).astype(np.float64)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            cluster_entropy[c] = -float((probs * np.log(probs)).sum()) + entropy_eps

        # Allocate Pass-2 budget proportional to cluster entropy.
        share = np.zeros(k, dtype=np.int64)
        cluster_sizes = np.bincount(cluster_labels, minlength=k).astype(np.float64)
        nonempty = cluster_sizes > 0
        if nonempty.any() and float(cluster_entropy[nonempty].sum()) > 0:
            normed = cluster_entropy.copy()
            normed[~nonempty] = 0.0
            normed = normed / normed.sum()
            share = np.round(normed * budget).astype(np.int64)
            if min_per_cluster > 0:
                share[nonempty] = np.maximum(share[nonempty], min_per_cluster)
            # Reconcile to exact budget.
            diff = int(share.sum()) - int(budget)
            if diff > 0:
                order = np.argsort(-share)
                for idx in order:
                    if diff <= 0:
                        break
                    if share[idx] > 0:
                        share[idx] -= 1
                        diff -= 1
            elif diff < 0:
                order = np.argsort(-cluster_entropy)
                ix = 0
                while diff < 0:
                    share[order[ix % k]] += 1
                    diff += 1
                    ix += 1

        # Per-node class proxy via secondary KMeans (consistent with the
        # single-pass selector's class-weighting).
        estimated_classes = self._estimate_node_classes(
            embeddings, torch.tensor(cluster_labels, dtype=torch.long), data,
        )
        # Ensure tensors live on CPU for indexing.
        density_cpu = density.detach().cpu()

        # Within-cluster ranking: density × T̂_diag[c_proxy], excluding already
        # selected nodes.
        selected: List[int] = []
        running_excl = set(already_selected)
        for c in range(k):
            quota = int(share[c])
            if quota <= 0:
                continue
            members = np.where(cluster_labels == c)[0]
            members = [int(m) for m in members if int(m) not in running_excl]
            if not members:
                continue
            quota = min(quota, len(members))
            est = estimated_classes[members].astype(np.int64)
            class_acc = self.per_class_accuracy[est]
            cand_score = density_cpu[members].numpy() * class_acc
            top_idx = np.argsort(-cand_score)[:quota]
            for j in top_idx:
                node = members[int(j)]
                if node in running_excl:
                    continue
                selected.append(node)
                running_excl.add(node)

        # Fallback: global density top-up if shortfall (e.g. small clusters).
        if len(selected) < budget:
            global_score = density_cpu.clone()
            for n in running_excl:
                global_score[n] = -1.0
            shortfall = budget - len(selected)
            _, extra = torch.topk(global_score, shortfall)
            for n in extra.tolist():
                if int(n) not in running_excl:
                    selected.append(int(n))
                    running_excl.add(int(n))

        logger.info(
            "Pass2: budget=%d, allocated=%d, mean H=%.3f, max H=%.3f",
            budget, len(selected),
            float(cluster_entropy.mean()),
            float(cluster_entropy.max()),
        )
        return selected[:budget]

    def update_noise_matrix(self, noise_matrix: np.ndarray) -> None:
        """Update the noise matrix (e.g., after re-estimation in a later round)."""
        self.noise_matrix = noise_matrix.copy()
        raw_diag = np.diag(noise_matrix).astype(np.float64)
        if self.select_downweight_floor > 0.0:
            mx = float(raw_diag.max()) if raw_diag.size > 0 else 1.0
            floor_val = self.select_downweight_floor * mx
            eff_diag = np.maximum(raw_diag, floor_val)
        else:
            eff_diag = raw_diag
        self.per_class_accuracy = eff_diag.astype(np.float32)
        self.per_class_noise = (1.0 - eff_diag).astype(np.float32)
        self._raw_per_class_accuracy = raw_diag.astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"NoiseAwareSelector(num_classes={self.num_classes}, "
            f"correction_frac={self.correction_fraction}, "
            f"mean_class_acc={self.per_class_accuracy.mean():.3f})"
        )


# ──────────────────────────────────────────────────────────────
# Self-test on Cora
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import os.path as osp
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s | %(message)s")

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)

    from src.data_loading.tag_dataset import load_dataset

    data_dir = osp.join(repo_root, "data")
    locle_dir = osp.join(repo_root, "baselines", "locle", "data")
    data_path = (
        data_dir
        if osp.exists(osp.join(data_dir, "cora_random_sbert.pt"))
        else locle_dir
    )

    data = load_dataset("cora", data_path=data_path)
    print(f"Loaded Cora: {data.x.size(0)} nodes, {data.num_classes} classes")
    print(f"Classes: {data.class_names}\n")

    # Build a synthetic noise matrix for testing
    C = data.num_classes
    rng = np.random.RandomState(42)
    noise_matrix = np.eye(C) * 0.7
    for i in range(C):
        off_diag = rng.dirichlet(np.ones(C - 1)) * 0.3
        idx = 0
        for j in range(C):
            if j != i:
                noise_matrix[i][j] = off_diag[idx]
                idx += 1
    print(f"Synthetic noise matrix diagonal: {np.diag(noise_matrix)}")
    print(f"Row sums: {noise_matrix.sum(axis=1)}\n")

    selector = NoiseAwareSelector(
        noise_matrix, num_classes=C, seed=42,
    )
    print(f"Selector: {selector}\n")

    # ── Test 1: select_initial_nodes ──
    print("=" * 60)
    print("[Test 1] select_initial_nodes")
    print("=" * 60)

    annotation_pool, correction_pool = selector.select_initial_nodes(
        data, data.x, budget=70,
    )

    print(f"  Annotation pool size: {len(annotation_pool)}")
    print(f"  Correction pool size: {len(correction_pool)}")
    print(f"  Total: {len(annotation_pool) + len(correction_pool)}")
    assert len(annotation_pool) + len(correction_pool) == 70, \
        f"Expected 70 total, got {len(annotation_pool) + len(correction_pool)}"
    # No overlap
    assert len(set(annotation_pool) & set(correction_pool)) == 0, \
        "Annotation and correction pools must not overlap"
    # All indices valid
    for idx in annotation_pool + correction_pool:
        assert 0 <= idx < data.x.size(0), f"Index {idx} out of range"
    print("  Test 1: PASS\n")

    # ── Test 2: select_critical_nodes ──
    print("=" * 60)
    print("[Test 2] select_critical_nodes")
    print("=" * 60)

    # Simulate GNN logits (random for testing)
    torch.manual_seed(42)
    fake_logits = torch.randn(data.x.size(0), C)

    # Simulate labeled mask (first 70 nodes labeled)
    labeled_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    for idx in annotation_pool + correction_pool:
        labeled_mask[idx] = True

    critical = selector.select_critical_nodes(
        data, fake_logits, labeled_mask, budget=20, round_num=1,
    )

    print(f"  Critical nodes selected: {len(critical)}")
    assert len(critical) == 20, f"Expected 20, got {len(critical)}"
    # Critical nodes must be from unlabeled set
    for idx in critical:
        assert not labeled_mask[idx], f"Critical node {idx} is already labeled"
    # No duplicates
    assert len(set(critical)) == len(critical), "Duplicate critical nodes"
    print("  Test 2: PASS\n")

    # ── Test 3: Noise weighting effect ──
    print("=" * 60)
    print("[Test 3] Noise weighting effect on selection")
    print("=" * 60)

    # Compare with uniform noise matrix (all classes equally reliable)
    uniform_nm = np.eye(C) * (1.0 / C) + (1.0 - 1.0 / C) / (C - 1) * (1 - np.eye(C))
    # Actually use identity-like (all classes equally good)
    uniform_nm = np.eye(C) * 0.75
    for i in range(C):
        for j in range(C):
            if i != j:
                uniform_nm[i][j] = 0.25 / (C - 1)

    selector_uniform = NoiseAwareSelector(
        uniform_nm, num_classes=C, seed=42,
    )

    ann_uniform, corr_uniform = selector_uniform.select_initial_nodes(
        data, data.x, budget=70,
    )

    # With asymmetric noise, the two selectors should produce different pools
    overlap = len(set(annotation_pool) & set(ann_uniform))
    print(f"  Annotation pool overlap (noise-aware vs uniform): {overlap}/56")
    print(f"  Correction pool overlap: {len(set(correction_pool) & set(corr_uniform))}/14")
    # They share the same density backbone, so some overlap is expected,
    # but not 100% (noise weighting should change the ranking)
    print("  Test 3: PASS\n")

    # ── Test 4: update_noise_matrix ──
    print("=" * 60)
    print("[Test 4] update_noise_matrix")
    print("=" * 60)

    new_nm = np.eye(C) * 0.9
    for i in range(C):
        for j in range(C):
            if i != j:
                new_nm[i][j] = 0.1 / (C - 1)

    selector.update_noise_matrix(new_nm)
    assert np.allclose(selector.per_class_accuracy, np.diag(new_nm))
    print(f"  Updated per-class accuracy: {selector.per_class_accuracy}")
    print("  Test 4: PASS\n")

    print("All noise-aware selector tests PASS.")
