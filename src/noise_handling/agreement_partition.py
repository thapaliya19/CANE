"""
Noise-aware agreement/disagreement partition for label-free node classification.

Adapted from GNN-as-Judge (baselines/gnn-as-judge/node_selection.py):
  - find_agreed_and_disagreed_nodes(): partitions by GNN vs LLM prediction match
  - filter_disagreed_by_preference(): filters by P(gnn_class) - P(llm_class)

Key difference:
GNN-as-Judge assumes GNN is trained on seed labels (ground truth), so agreement
between GNN and LLM is a strong correctness signal.  In our label-free setting
the GNN is trained on LLM annotations — both can share systematic errors.
We compensate by weighting agreement/disagreement with the noise_matrix from
Stage 0 probing: high diagonal = trustworthy agreement, high off-diagonal =
known LLM confusion pair where GNN's disagreement is more informative.

Pipeline position: Stage II — called each refinement round after GNN training.

Usage:
    partitioner = AgreementPartitioner(noise_matrix)
    agreement, preference, uncertain = partitioner.partition(
        gnn_predictions=logits,        # raw logits [N, C]
        gnn_probabilities=None,        # computed from logits if None
        llm_annotations=llm_labels,    # {node_id: int}
        llm_confidences=llm_conf,      # {node_id: float} or None
        node_ids=list(llm_labels.keys()),
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Result containers ────────────────────────────────────────

@dataclass
class AgreedNode:
    """Node where GNN and LLM agree on the label."""
    node_id: int
    label: int
    weight: float       # noise_matrix[label][label] * gnn_prob — trustworthiness
    gnn_prob: float     # GNN confidence for this label
    llm_conf: float     # LLM confidence (if available)


@dataclass
class PreferenceNode:
    """Disagreement node included in preference training set."""
    node_id: int
    preferred: int      # GNN prediction (chosen label)
    rejected: int       # LLM prediction (rejected label)
    weight: float       # noise-adjusted preference weight for loss scaling
    pref_score: float   # raw preference score before thresholding
    gnn_prob_preferred: float
    gnn_prob_rejected: float


@dataclass
class UncertainNode:
    """Disagreement node below preference threshold — candidate for re-annotation."""
    node_id: int
    gnn_pred: int
    llm_pred: int
    gnn_prob_gnn: float
    gnn_prob_llm: float
    adjusted_score: float


@dataclass
class PartitionResult:
    """Full partition output for one refinement round."""
    agreement_set: List[AgreedNode] = field(default_factory=list)
    preference_set: List[PreferenceNode] = field(default_factory=list)
    uncertain_set: List[UncertainNode] = field(default_factory=list)

    @property
    def n_agreed(self) -> int:
        return len(self.agreement_set)

    @property
    def n_preference(self) -> int:
        return len(self.preference_set)

    @property
    def n_uncertain(self) -> int:
        return len(self.uncertain_set)

    def summary(self) -> str:
        total = self.n_agreed + self.n_preference + self.n_uncertain
        return (
            f"Partition: {total} nodes — "
            f"agreed={self.n_agreed}, preference={self.n_preference}, "
            f"uncertain={self.n_uncertain}"
        )


# ── Main partitioner ─────────────────────────────────────────

class AgreementPartitioner:
    """
    Noise-weighted agreement/disagreement partition.

    For each node with both GNN and LLM predictions:
      - Agreement (gnn_pred == llm_label):
          weight = noise_matrix[pred][pred] * gnn_prob[pred]
          High weight = low-noise class AND GNN is confident.

      - Disagreement (gnn_pred != llm_label):
          S_pref = gnn_prob[gnn_pred] - gnn_prob[llm_label]
          noise_weight = noise_matrix[llm_label][gnn_pred]
          adjusted_score = S_pref * noise_weight

          If adjusted_score > tau → preference set (for ORPO training)
          Else → uncertain set (candidate for teacher re-annotation)

    Args:
        noise_matrix: C×C numpy array from LLMNoiseProfiler.
            noise_matrix[i][j] = P(LLM predicts j | true class is i).
        preference_threshold: Minimum adjusted_score for inclusion in
            the preference set.  Default 0.0 (include all disagreements
            where GNN is more confident than LLM on its own prediction).
        min_preference_set: If |preference_set| < this, fall back to
            including top disagreements by adjusted_score.  Prevents
            empty preference sets that make ORPO training degenerate.
    """

    def __init__(
        self,
        noise_matrix: Optional[np.ndarray] = None,
        preference_threshold: float = 0.0,
        min_preference_set: int = 10,
        num_classes: Optional[int] = None,
    ):
        if noise_matrix is not None:
            self.noise_matrix = np.asarray(noise_matrix, dtype=np.float64)
            self.num_classes = self.noise_matrix.shape[0]
            self._diag = np.diag(self.noise_matrix)  # per-class LLM accuracy
            self._uniform = False
        else:
            # Uniform (unweighted) mode — replicates GNN-as-Judge baseline.
            # All agreement weights become gnn_prob alone; preference scores
            # are raw S_pref without noise scaling.
            if num_classes is None:
                raise ValueError(
                    "Must provide num_classes when noise_matrix is None"
                )
            self.num_classes = num_classes
            self.noise_matrix = np.ones((num_classes, num_classes), dtype=np.float64)
            self._diag = np.ones(num_classes, dtype=np.float64)
            self._uniform = True

        self.preference_threshold = preference_threshold
        self.min_preference_set = min_preference_set

    def partition(
        self,
        gnn_predictions: torch.Tensor,
        gnn_probabilities: Optional[torch.Tensor],
        llm_annotations: Dict[int, int],
        llm_confidences: Optional[Dict[int, float]],
        node_ids: Optional[List[int]] = None,
    ) -> PartitionResult:
        """
        Partition nodes into agreement, preference, and uncertain sets.

        Args:
            gnn_predictions: Raw GNN logits, shape [N, C].
            gnn_probabilities: Softmax probabilities [N, C].  Computed from
                gnn_predictions if None.
            llm_annotations: {node_id: predicted_class_index}.
            llm_confidences: {node_id: confidence_score} or None.
                Used only for logging/diagnostics; does NOT affect partition
                (LLM confidence is unreliable for high-confusion classes, so
                it is never used alone as a noise filter).
            node_ids: Subset of node IDs to partition.  If None, uses all
                keys from llm_annotations.

        Returns:
            PartitionResult with agreement_set, preference_set, uncertain_set.
        """
        # Compute probabilities from logits if not provided
        if gnn_probabilities is None:
            gnn_probabilities = F.softmax(gnn_predictions, dim=1)

        gnn_preds = gnn_predictions.argmax(dim=1)

        # Detach to numpy for fast indexing
        gnn_preds_np = gnn_preds.cpu().numpy()
        gnn_probs_np = gnn_probabilities.cpu().detach().numpy()

        num_gnn_nodes = gnn_predictions.size(0)

        if node_ids is None:
            node_ids = list(llm_annotations.keys())

        if llm_confidences is None:
            llm_confidences = {}

        agreement_set: List[AgreedNode] = []
        preference_candidates: List[Tuple[float, PreferenceNode]] = []
        uncertain_set: List[UncertainNode] = []

        for nid in node_ids:
            if nid not in llm_annotations:
                continue
            if not (0 <= nid < num_gnn_nodes):
                continue

            llm_label = llm_annotations[nid]
            if not (0 <= llm_label < self.num_classes):
                continue

            gnn_pred = int(gnn_preds_np[nid])
            gnn_prob_pred = float(gnn_probs_np[nid, gnn_pred])
            llm_conf = llm_confidences.get(nid, 0.0)

            if gnn_pred == llm_label:
                # ── Agreement ──
                # Weight = class reliability * GNN confidence
                weight = float(self._diag[gnn_pred]) * gnn_prob_pred
                agreement_set.append(AgreedNode(
                    node_id=nid,
                    label=gnn_pred,
                    weight=weight,
                    gnn_prob=gnn_prob_pred,
                    llm_conf=llm_conf,
                ))
            else:
                # ── Disagreement ──
                gnn_prob_llm = float(gnn_probs_np[nid, llm_label])

                # GNN-as-Judge baseline: S_pref = P(gnn_class) - P(llm_class)
                s_pref = gnn_prob_pred - gnn_prob_llm

                # Our addition: weight by how often LLM confuses
                # llm_label → gnn_pred (off-diagonal of noise_matrix)
                noise_weight = float(self.noise_matrix[llm_label, gnn_pred])

                # Adjusted score: high when GNN disagrees strongly AND
                # this is a known LLM confusion pair
                adjusted_score = s_pref * noise_weight

                # Preference weight for ORPO loss scaling:
                # Use the confusion rate directly — pairs the LLM confuses
                # more often get higher training weight
                pref_weight = noise_weight

                pnode = PreferenceNode(
                    node_id=nid,
                    preferred=gnn_pred,
                    rejected=llm_label,
                    weight=pref_weight,
                    pref_score=adjusted_score,
                    gnn_prob_preferred=gnn_prob_pred,
                    gnn_prob_rejected=gnn_prob_llm,
                )
                # Store with score for sorting/thresholding
                preference_candidates.append((adjusted_score, pnode))

        # ── Normalize preference scores to [0, 1] range ──
        # Raw scores (s_pref * noise_weight) are often near zero because
        # off-diagonal noise_matrix entries are small. Rank-based
        # normalization makes the threshold meaningful.
        if preference_candidates:
            raw_scores = [s for s, _ in preference_candidates]
            score_min = min(raw_scores)
            score_max = max(raw_scores)
            score_range = score_max - score_min
            if score_range > 1e-8:
                preference_candidates = [
                    ((s - score_min) / score_range, pnode)
                    for s, pnode in preference_candidates
                ]
                # Update pref_score in nodes too
                for norm_score, pnode in preference_candidates:
                    pnode.pref_score = norm_score

        # Sort candidates descending by normalized score
        preference_candidates.sort(key=lambda x: -x[0])

        preference_set: List[PreferenceNode] = []
        for score, pnode in preference_candidates:
            if score > self.preference_threshold:
                preference_set.append(pnode)
            else:
                uncertain_set.append(UncertainNode(
                    node_id=pnode.node_id,
                    gnn_pred=pnode.preferred,
                    llm_pred=pnode.rejected,
                    gnn_prob_gnn=pnode.gnn_prob_preferred,
                    gnn_prob_llm=pnode.gnn_prob_rejected,
                    adjusted_score=score,
                ))

        # ORPO needs balanced preference pairs: if the disagreement set is tiny
        # (e.g., 5 nodes), preference training won't learn anything.
        # If we have too few preference nodes but have uncertain candidates,
        # promote top uncertain nodes to preference set.
        if len(preference_set) < self.min_preference_set and uncertain_set:
            deficit = self.min_preference_set - len(preference_set)
            # uncertain_set was built from low-scoring candidates;
            # they're already in descending order from preference_candidates
            promote = uncertain_set[:deficit]
            for unode in promote:
                preference_set.append(PreferenceNode(
                    node_id=unode.node_id,
                    preferred=unode.gnn_pred,
                    rejected=unode.llm_pred,
                    weight=float(self.noise_matrix[unode.llm_pred, unode.gnn_pred]),
                    pref_score=unode.adjusted_score,
                    gnn_prob_preferred=unode.gnn_prob_gnn,
                    gnn_prob_rejected=unode.gnn_prob_llm,
                ))
            uncertain_set = uncertain_set[deficit:]

        result = PartitionResult(
            agreement_set=agreement_set,
            preference_set=preference_set,
            uncertain_set=uncertain_set,
        )

        logger.info(result.summary())
        if agreement_set:
            mean_agree_weight = np.mean([a.weight for a in agreement_set])
            logger.info("  Agreement mean weight: %.4f", mean_agree_weight)
        if preference_set:
            mean_pref_score = np.mean([p.pref_score for p in preference_set])
            logger.info("  Preference mean score: %.4f", mean_pref_score)

        return result

    def __repr__(self) -> str:
        mode = "uniform" if self._uniform else "noise-weighted"
        return (
            f"AgreementPartitioner(num_classes={self.num_classes}, "
            f"mode={mode}, tau={self.preference_threshold}, "
            f"min_pref={self.min_preference_set})"
        )


# ──────────────────────────────────────────────────────────────
# Self-test on Cora
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os.path as osp

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)

    from src.data_loading.tag_dataset import load_dataset

    data_dir = osp.join(repo_root, "data")
    locle_data_dir = osp.join(repo_root, "baselines", "locle", "data")
    data_path = (
        data_dir
        if osp.exists(osp.join(data_dir, "cora_random_sbert.pt"))
        else locle_data_dir
    )

    data = load_dataset("cora", data_path=data_path)
    num_nodes = data.x.size(0)
    num_classes = data.num_classes
    print(f"Loaded Cora: {num_nodes} nodes, {num_classes} classes")
    print(f"Classes: {data.class_names}\n")

    # ── Build synthetic noise matrix ──
    rng = np.random.RandomState(42)
    # Realistic noise matrix: ~80% diagonal, rest spread
    noise_matrix = np.eye(num_classes) * 0.8
    off_diag = rng.dirichlet(np.ones(num_classes), size=num_classes) * 0.2
    np.fill_diagonal(off_diag, 0.0)
    noise_matrix += off_diag
    # Re-normalize rows
    noise_matrix /= noise_matrix.sum(axis=1, keepdims=True)
    print("Synthetic noise matrix (diagonal = per-class LLM accuracy):")
    print(f"  {np.diag(noise_matrix).round(3)}\n")

    # ── Simulate GNN logits and LLM annotations ──
    # Use ground-truth labels to create semi-realistic scenario
    gt_labels = data.y.numpy()

    # GNN logits: true class gets high logit + noise
    gnn_logits = torch.randn(num_nodes, num_classes) * 0.5
    for i in range(num_nodes):
        gnn_logits[i, gt_labels[i]] += 3.0  # GNN mostly correct

    # LLM annotations: apply noise per noise_matrix
    # For each node, sample LLM prediction from noise_matrix[true_class]
    test_node_ids = list(range(0, min(200, num_nodes)))
    llm_annotations = {}
    llm_confidences = {}
    for nid in test_node_ids:
        true_class = gt_labels[nid]
        llm_pred = rng.choice(num_classes, p=noise_matrix[true_class])
        llm_annotations[nid] = int(llm_pred)
        llm_confidences[nid] = float(rng.uniform(60, 95))

    print(f"Simulated {len(llm_annotations)} LLM annotations")
    n_llm_correct = sum(1 for nid, p in llm_annotations.items() if p == gt_labels[nid])
    print(f"LLM accuracy (vs GT): {n_llm_correct / len(llm_annotations):.3f}")

    gnn_preds = gnn_logits.argmax(dim=1).numpy()
    n_gnn_correct = sum(1 for nid in test_node_ids if gnn_preds[nid] == gt_labels[nid])
    print(f"GNN accuracy (vs GT): {n_gnn_correct / len(test_node_ids):.3f}\n")

    # ── Test 1: Basic partition ──
    print("=" * 60)
    print("[Test 1] Basic partition")
    print("=" * 60)

    partitioner = AgreementPartitioner(
        noise_matrix=noise_matrix,
        preference_threshold=0.0,
        min_preference_set=5,
    )

    result = partitioner.partition(
        gnn_predictions=gnn_logits,
        gnn_probabilities=None,
        llm_annotations=llm_annotations,
        llm_confidences=llm_confidences,
        node_ids=test_node_ids,
    )

    print(f"  {result.summary()}")
    total = result.n_agreed + result.n_preference + result.n_uncertain
    assert total == len(test_node_ids), (
        f"Partition should cover all nodes: {total} != {len(test_node_ids)}"
    )
    print("  All nodes accounted for: PASS")

    # ── Test 2: Agreement weights ──
    print("\n" + "=" * 60)
    print("[Test 2] Agreement weight properties")
    print("=" * 60)

    if result.agreement_set:
        weights = [a.weight for a in result.agreement_set]
        print(f"  Agreement weights: min={min(weights):.4f}, "
              f"max={max(weights):.4f}, mean={np.mean(weights):.4f}")
        assert all(0 <= w <= 1.0 for w in weights), "Weights should be in [0, 1]"
        print("  Weight bounds: PASS")

        # Check that weight = diag * gnn_prob
        for a in result.agreement_set[:5]:
            expected = float(noise_matrix[a.label, a.label]) * a.gnn_prob
            assert abs(a.weight - expected) < 1e-6, (
                f"Weight mismatch: {a.weight} != {expected}"
            )
        print("  Weight formula verified: PASS")
    else:
        print("  No agreement nodes (unexpected with these parameters)")

    # ── Test 3: Preference score properties ──
    print("\n" + "=" * 60)
    print("[Test 3] Preference score properties")
    print("=" * 60)

    if result.preference_set:
        scores = [p.pref_score for p in result.preference_set]
        print(f"  Preference scores: min={min(scores):.4f}, "
              f"max={max(scores):.4f}, mean={np.mean(scores):.4f}")

        # Verify score computation: S_pref * noise_matrix[llm][gnn]
        probs = F.softmax(gnn_logits, dim=1).numpy()
        for p in result.preference_set[:5]:
            s_pref = probs[p.node_id, p.preferred] - probs[p.node_id, p.rejected]
            nw = noise_matrix[p.rejected, p.preferred]
            expected = s_pref * nw
            assert abs(p.pref_score - expected) < 1e-5, (
                f"Score mismatch: {p.pref_score} != {expected}"
            )
        print("  Score formula verified: PASS")

        # preferred != rejected
        for p in result.preference_set:
            assert p.preferred != p.rejected
        print("  Preferred != rejected: PASS")
    else:
        print("  No preference nodes (check threshold)")

    # ── Test 4: High threshold → more uncertain ──
    print("\n" + "=" * 60)
    print("[Test 4] High threshold pushes nodes to uncertain")
    print("=" * 60)

    strict_partitioner = AgreementPartitioner(
        noise_matrix=noise_matrix,
        preference_threshold=0.5,
        min_preference_set=0,  # disable promotion
    )
    strict_result = strict_partitioner.partition(
        gnn_predictions=gnn_logits,
        gnn_probabilities=None,
        llm_annotations=llm_annotations,
        llm_confidences=None,
        node_ids=test_node_ids,
    )

    print(f"  Strict (tau=0.5): {strict_result.summary()}")
    # Higher threshold should yield fewer preference, more uncertain
    assert strict_result.n_preference <= result.n_preference, (
        "Higher threshold should give fewer preference nodes"
    )
    assert strict_result.n_uncertain >= result.n_uncertain, (
        "Higher threshold should give more uncertain nodes"
    )
    print("  Threshold effect verified: PASS")

    # ── Test 5: min_preference_set promotion ──
    print("\n" + "=" * 60)
    print("[Test 5] min_preference_set promotion from uncertain")
    print("=" * 60)

    promo_partitioner = AgreementPartitioner(
        noise_matrix=noise_matrix,
        preference_threshold=999.0,  # nothing passes threshold
        min_preference_set=10,
    )
    promo_result = promo_partitioner.partition(
        gnn_predictions=gnn_logits,
        gnn_probabilities=None,
        llm_annotations=llm_annotations,
        llm_confidences=None,
        node_ids=test_node_ids,
    )

    n_disagreed = promo_result.n_preference + promo_result.n_uncertain
    if n_disagreed >= 10:
        assert promo_result.n_preference == 10, (
            f"Should promote exactly 10, got {promo_result.n_preference}"
        )
        print(f"  Promoted 10 nodes to preference set: PASS")
    else:
        assert promo_result.n_preference == n_disagreed
        print(f"  All {n_disagreed} disagreed nodes promoted (< 10 total): PASS")

    # ── Test 6: Edge case — empty input ──
    print("\n" + "=" * 60)
    print("[Test 6] Edge cases")
    print("=" * 60)

    empty_result = partitioner.partition(
        gnn_predictions=gnn_logits,
        gnn_probabilities=None,
        llm_annotations={},
        llm_confidences=None,
        node_ids=[],
    )
    assert empty_result.n_agreed == 0
    assert empty_result.n_preference == 0
    assert empty_result.n_uncertain == 0
    print("  Empty input handled: PASS")

    print(f"\nrepr: {partitioner}")
    print("\nAll agreement partition tests complete.")
