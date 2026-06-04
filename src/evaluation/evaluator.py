"""
Evaluation metrics for label-free node classification.

Computes: Accuracy, Macro-F1, NMI, ARI, per-class F1.
Matches LoCLE's evaluation protocol:
  - Evaluate on ALL remaining nodes vs ground truth.
  - No fixed train/val/test split.

Additional (our contribution):
  - Per-class accuracy
  - Top-5 confused class pairs
  - Confusion matrix (C×C)
  - Noise correction quality (predicted vs actual confusion)
  - Convergence tracking (entropy, flip rate, agreement rate per round)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)

logger = logging.getLogger(__name__)


def evaluate_predictions(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    labeled_mask: Optional[torch.Tensor] = None,
    class_names: Optional[list] = None,
) -> Dict:
    """
    Evaluate GNN predictions against ground truth.

    Args:
        predictions: Predicted labels or logits [num_nodes] or [num_nodes, C].
        ground_truth: Ground-truth labels [num_nodes].
        labeled_mask: If provided, evaluate only on UNLABELED nodes
                      (labeled_mask=True means node was used for training).
        class_names: Optional class names for per-class reporting.

    Returns:
        Dict with accuracy, macro_f1, nmi, ari, per_class_f1.
    """
    if predictions.dim() == 2:
        preds = predictions.argmax(dim=1)
    else:
        preds = predictions

    preds_np = preds.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()

    if labeled_mask is not None:
        eval_mask = ~labeled_mask.cpu()
        preds_np = preds_np[eval_mask]
        gt_np = gt_np[eval_mask]

    if len(preds_np) == 0:
        logger.warning("No nodes to evaluate")
        return {"accuracy": 0.0, "macro_f1": 0.0, "nmi": 0.0, "ari": 0.0}

    acc = accuracy_score(gt_np, preds_np)
    macro_f1 = f1_score(gt_np, preds_np, average="macro", zero_division=0)
    nmi = normalized_mutual_info_score(gt_np, preds_np)
    ari = adjusted_rand_score(gt_np, preds_np)

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "nmi": nmi,
        "ari": ari,
        "num_evaluated": len(preds_np),
    }

    # Per-class F1
    per_class = f1_score(gt_np, preds_np, average=None, zero_division=0)
    results["per_class_f1"] = per_class.tolist()

    if class_names:
        for i, name in enumerate(class_names):
            if i < len(per_class):
                results[f"f1_{name}"] = float(per_class[i])

    return results


def evaluate_labels(
    labels: Dict[int, int],
    ground_truth: torch.Tensor,
    num_nodes: int,
    class_names: Optional[list] = None,
) -> Dict:
    """
    Evaluate a label dict against ground truth.

    Evaluates on ALL nodes that have a label assignment.
    """
    if not labels:
        return {"accuracy": 0.0, "macro_f1": 0.0, "nmi": 0.0, "ari": 0.0}

    node_ids = sorted(labels.keys())
    preds = np.array([labels[nid] for nid in node_ids])
    gt = ground_truth.cpu().numpy()[node_ids]

    acc = accuracy_score(gt, preds)
    macro_f1 = f1_score(gt, preds, average="macro", zero_division=0)
    nmi = normalized_mutual_info_score(gt, preds)
    ari = adjusted_rand_score(gt, preds)

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "nmi": nmi,
        "ari": ari,
        "num_evaluated": len(node_ids),
    }

    per_class = f1_score(gt, preds, average=None, zero_division=0)
    results["per_class_f1"] = per_class.tolist()

    if class_names:
        for i, name in enumerate(class_names):
            if i < len(per_class):
                results[f"f1_{name}"] = float(per_class[i])

    return results


# ---------------------------------------------------------------------------
# New evaluation functions (our contribution)
# ---------------------------------------------------------------------------


def evaluate_full(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    class_names: Optional[list] = None,
    labeled_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Full evaluation matching LoCLE Table 2 plus our additional metrics.

    Metrics (matching LoCLE Table 2):
      - Accuracy (overall)
      - Macro-F1
      - NMI (Normalized Mutual Information)
      - ARI (Adjusted Rand Index)

    ADDITIONAL (our contribution — no prior paper reports these):
      - Per-class F1 (C values, one per class)
      - Per-class accuracy
      - Top-5 confused class pairs (pred_class, true_class, count)
      - Confusion matrix (C×C)

    Args:
        predictions: Predicted labels or logits [num_nodes] or [num_nodes, C].
        ground_truth: Ground-truth labels [num_nodes].
        class_names: Optional class names for reporting.
        labeled_mask: If provided, evaluate only on UNLABELED nodes.

    Returns:
        Dict with all metrics.
    """
    if predictions.dim() == 2:
        preds = predictions.argmax(dim=1)
    else:
        preds = predictions

    preds_np = preds.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()

    if labeled_mask is not None:
        eval_mask = ~labeled_mask.cpu()
        preds_np = preds_np[eval_mask]
        gt_np = gt_np[eval_mask]

    if len(preds_np) == 0:
        logger.warning("No nodes to evaluate")
        return {"accuracy": 0.0, "macro_f1": 0.0, "nmi": 0.0, "ari": 0.0}

    # --- Standard metrics (LoCLE Table 2) ---
    acc = accuracy_score(gt_np, preds_np)
    macro_f1 = f1_score(gt_np, preds_np, average="macro", zero_division=0)
    nmi = normalized_mutual_info_score(gt_np, preds_np)
    ari = adjusted_rand_score(gt_np, preds_np)

    per_class_f1 = f1_score(gt_np, preds_np, average=None, zero_division=0)

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "nmi": nmi,
        "ari": ari,
        "num_evaluated": len(preds_np),
        "per_class_f1": per_class_f1.tolist(),
    }

    # --- Per-class accuracy ---
    all_classes = sorted(set(gt_np.tolist()) | set(preds_np.tolist()))
    num_classes = max(all_classes) + 1 if all_classes else 0
    per_class_acc = []
    for c in range(num_classes):
        mask_c = gt_np == c
        if mask_c.sum() > 0:
            per_class_acc.append(float((preds_np[mask_c] == c).mean()))
        else:
            per_class_acc.append(0.0)
    results["per_class_accuracy"] = per_class_acc

    # --- Confusion matrix (C×C) ---
    labels_range = list(range(num_classes))
    cm = confusion_matrix(gt_np, preds_np, labels=labels_range)
    results["confusion_matrix"] = cm.tolist()

    # --- Top-5 confused class pairs ---
    confused_pairs = _top_confused_pairs(cm, n=5, class_names=class_names)
    results["top_confused_pairs"] = confused_pairs

    # --- Named per-class metrics ---
    if class_names:
        for i, name in enumerate(class_names):
            if i < len(per_class_f1):
                results[f"f1_{name}"] = float(per_class_f1[i])
            if i < len(per_class_acc):
                results[f"acc_{name}"] = per_class_acc[i]

    return results


def _top_confused_pairs(
    cm: np.ndarray,
    n: int = 5,
    class_names: Optional[list] = None,
) -> List[Dict]:
    """Extract the top-n off-diagonal (true_class, pred_class, count) triples."""
    num_c = cm.shape[0]
    pairs = []
    for i in range(num_c):
        for j in range(num_c):
            if i != j and cm[i, j] > 0:
                true_name = class_names[i] if class_names and i < len(class_names) else i
                pred_name = class_names[j] if class_names and j < len(class_names) else j
                pairs.append({
                    "true_class": true_name,
                    "pred_class": pred_name,
                    "count": int(cm[i, j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:n]


def evaluate_noise_correction(
    noise_matrix_predicted: np.ndarray,
    actual_confusion_matrix: np.ndarray,
) -> Dict:
    """
    How well did DMA-style probing predict actual LLM errors?

    Compares the predicted noise matrix (from probing) against the actual
    confusion matrix (LLM labels vs ground truth). Both are C×C matrices.

    Args:
        noise_matrix_predicted: C×C matrix from noise probing.
            noise_matrix_predicted[i][j] = P(LLM predicts j | true class i).
        actual_confusion_matrix: C×C matrix of actual LLM annotation errors.
            actual_confusion_matrix[i][j] = count of (true=i, LLM_label=j).

    Returns:
        Dict with correlation metrics and per-class accuracy comparison.
    """
    predicted = np.asarray(noise_matrix_predicted, dtype=np.float64)
    actual = np.asarray(actual_confusion_matrix, dtype=np.float64)

    assert predicted.shape == actual.shape, (
        f"Shape mismatch: predicted {predicted.shape} vs actual {actual.shape}"
    )
    num_classes = predicted.shape[0]

    # Normalise actual to row-stochastic (probabilities) for comparison
    row_sums = actual.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    actual_norm = actual / row_sums

    # --- Frobenius distance ---
    frob_dist = float(np.linalg.norm(predicted - actual_norm, "fro"))

    # --- Element-wise correlation (flatten both matrices) ---
    pred_flat = predicted.flatten()
    actual_flat = actual_norm.flatten()
    if np.std(pred_flat) > 0 and np.std(actual_flat) > 0:
        correlation = float(np.corrcoef(pred_flat, actual_flat)[0, 1])
    else:
        correlation = 0.0

    # --- Diagonal accuracy comparison (per-class accuracy) ---
    pred_diag = np.diag(predicted)
    actual_diag = np.diag(actual_norm)
    diag_mae = float(np.mean(np.abs(pred_diag - actual_diag)))

    # --- Off-diagonal correlation (error pattern similarity) ---
    mask_off = ~np.eye(num_classes, dtype=bool)
    pred_off = predicted[mask_off]
    actual_off = actual_norm[mask_off]
    if np.std(pred_off) > 0 and np.std(actual_off) > 0:
        off_diag_corr = float(np.corrcoef(pred_off, actual_off)[0, 1])
    else:
        off_diag_corr = 0.0

    # --- Top confused pair agreement ---
    pred_top = np.unravel_index(
        np.argmax(predicted * mask_off), predicted.shape
    )
    actual_top = np.unravel_index(
        np.argmax(actual_norm * mask_off), actual_norm.shape
    )
    top_pair_match = bool(pred_top == actual_top)

    return {
        "frobenius_distance": frob_dist,
        "element_correlation": correlation,
        "diagonal_mae": diag_mae,
        "off_diagonal_correlation": off_diag_corr,
        "top_confused_pair_match": top_pair_match,
        "predicted_top_pair": (int(pred_top[0]), int(pred_top[1])),
        "actual_top_pair": (int(actual_top[0]), int(actual_top[1])),
        "per_class_predicted_acc": pred_diag.tolist(),
        "per_class_actual_acc": actual_diag.tolist(),
    }


def evaluate_convergence(
    label_history_per_round: List[Dict[int, int]],
) -> Dict:
    """
    Track convergence across pipeline rounds.

    Args:
        label_history_per_round: List of label dicts, one per round.
            Each dict maps node_id -> predicted_class.

    Returns:
        Dict with per-round: label_entropy, flip_rate, agreement_rate,
        and overall convergence_round (first round where flip_rate < 0.01).
    """
    num_rounds = len(label_history_per_round)
    if num_rounds == 0:
        return {"rounds": [], "converged_at_round": None}

    round_metrics = []

    for r in range(num_rounds):
        labels_r = label_history_per_round[r]
        node_ids = sorted(labels_r.keys())
        label_arr = np.array([labels_r[nid] for nid in node_ids])

        # --- Label entropy: H(label distribution) ---
        if len(label_arr) > 0:
            _, counts = np.unique(label_arr, return_counts=True)
            probs = counts / counts.sum()
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        else:
            entropy = 0.0

        # --- Flip rate: fraction of nodes whose label changed from prev round ---
        if r == 0:
            flip_rate = 1.0  # All labels are "new"
            agreement_rate = 0.0
        else:
            labels_prev = label_history_per_round[r - 1]
            common_ids = sorted(set(node_ids) & set(labels_prev.keys()))
            if len(common_ids) > 0:
                flips = sum(
                    1 for nid in common_ids
                    if labels_r[nid] != labels_prev[nid]
                )
                flip_rate = flips / len(common_ids)
                agreement_rate = 1.0 - flip_rate
            else:
                flip_rate = 1.0
                agreement_rate = 0.0

        round_metrics.append({
            "round": r,
            "num_labeled": len(node_ids),
            "label_entropy": entropy,
            "flip_rate": flip_rate,
            "agreement_rate": agreement_rate,
        })

    # --- Convergence round: first round where flip_rate < 1% ---
    converged_at = None
    for m in round_metrics:
        if m["round"] > 0 and m["flip_rate"] < 0.01:
            converged_at = m["round"]
            break

    return {
        "rounds": round_metrics,
        "converged_at_round": converged_at,
        "final_flip_rate": round_metrics[-1]["flip_rate"] if round_metrics else None,
        "final_entropy": round_metrics[-1]["label_entropy"] if round_metrics else None,
    }


if __name__ == "__main__":
    # Minimal test on synthetic data
    print("=== evaluator.py self-test ===")

    torch.manual_seed(0)
    num_nodes, num_classes = 200, 5
    gt = torch.randint(0, num_classes, (num_nodes,))
    # Predictions: mostly correct with some noise
    preds = gt.clone()
    preds[:40] = torch.randint(0, num_classes, (40,))

    # --- evaluate_full ---
    res = evaluate_full(preds, gt, class_names=[f"C{i}" for i in range(num_classes)])
    print(f"evaluate_full: acc={res['accuracy']:.3f}, "
          f"macro_f1={res['macro_f1']:.3f}, nmi={res['nmi']:.3f}")
    print(f"  per_class_acc: {[f'{v:.2f}' for v in res['per_class_accuracy']]}")
    print(f"  top confused: {res['top_confused_pairs'][:3]}")
    assert "confusion_matrix" in res
    assert len(res["per_class_accuracy"]) == num_classes

    # --- evaluate_noise_correction ---
    predicted_noise = np.eye(num_classes) * 0.8 + 0.04
    actual_cm = np.zeros((num_classes, num_classes))
    for i in range(num_nodes):
        actual_cm[gt[i].item(), preds[i].item()] += 1
    nc = evaluate_noise_correction(predicted_noise, actual_cm)
    print(f"\nevaluate_noise_correction: corr={nc['element_correlation']:.3f}, "
          f"diag_mae={nc['diagonal_mae']:.3f}, "
          f"top_match={nc['top_confused_pair_match']}")

    # --- evaluate_convergence ---
    history = []
    labels = {i: gt[i].item() for i in range(100)}
    history.append(dict(labels))
    # Round 1: flip 10 labels
    labels_r1 = dict(labels)
    for i in range(10):
        labels_r1[i] = (labels_r1[i] + 1) % num_classes
    history.append(labels_r1)
    # Round 2: flip 0 labels (converged)
    history.append(dict(labels_r1))

    conv = evaluate_convergence(history)
    print(f"\nevaluate_convergence:")
    for rm in conv["rounds"]:
        print(f"  round {rm['round']}: flip_rate={rm['flip_rate']:.3f}, "
              f"entropy={rm['label_entropy']:.3f}")
    print(f"  converged_at_round: {conv['converged_at_round']}")
    assert conv["converged_at_round"] == 2  # Round 2 has 0 flips

    print("\nAll tests passed.")
