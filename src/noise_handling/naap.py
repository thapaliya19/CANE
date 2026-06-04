"""
Noise-Aware Anisotropic Propagation (NAAP).

A novel label propagation algorithm that uses the noise transition matrix
to parameterize per-node propagation rates and edge weights.

Standard LP:  Y_{t+1} = alpha * Y_0 + (1-alpha) * D^{-1}A * Y_t
NAAP:         Y_{t+1} = Phi * Y_0 + (I-Phi) * D_w^{-1} W * Y_t

where Phi = diag(alpha_1, ..., alpha_n) is a per-node teleport matrix
parameterized by the noise transition matrix T.

Key idea: noisy-class seeds should propagate LESS (high alpha_i),
clean-class seeds should propagate MORE (low alpha_i).

"""

import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def compute_local_homophily(
    node_id: int,
    labels_dict: Dict[int, int],
    adj: Dict[int, list],
) -> float:
    """
    Compute local homophily for a labeled node.

    h_i = fraction of labeled neighbors with the same label as node i.
    Returns 0.5 if no labeled neighbors (neutral).
    """
    if node_id not in labels_dict:
        return 0.5
    my_label = labels_dict[node_id]
    neighbors = adj.get(node_id, [])
    labeled_neighbors = [n for n in neighbors if n in labels_dict]
    if len(labeled_neighbors) == 0:
        return 0.5
    agree = sum(1 for n in labeled_neighbors if labels_dict[n] == my_label)
    return agree / len(labeled_neighbors)


def compute_per_node_alpha(
    labels_dict: Dict[int, int],
    noise_matrix: np.ndarray,
    edge_index: np.ndarray,
    num_nodes: int,
    alpha_base: float = 0.3,
    gamma: float = 0.6,
    gnn_entropy: Optional[np.ndarray] = None,
    T_c: Optional[np.ndarray] = None,
    cluster_id: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute per-node adaptive teleport rates from noise matrix.

    For labeled node i with class c_i:
        phi_i = (1 - T[c_i][c_i]) * gamma + (1 - h_i) * (1 - gamma)
        alpha_i = alpha_base + (1 - alpha_base) * phi_i * gate_i

    The confidence gate (gate_i) prevents over-restricting correctly-labeled
    nodes from noisy classes. When the GNN is confident about a node (low
    entropy), gate_i ≈ 0 → normal propagation. When uncertain (high entropy),
    gate_i ≈ 1 → apply noise restriction.

    Inspired by ProCon (IJCAI 2025) and SAUC (2024).

    Args:
        gnn_entropy: [num_nodes] GNN prediction entropy per node (optional).
                     If provided, gates the noise-aware α by uncertainty.
        T_c: [K, C, C] cluster-conditional transition matrix (optional). When
             provided together with cluster_id, the per-node noise factor is
             1 - T_c[cluster_id[i], label_i, label_i] instead of the global
             noise_matrix diagonal.
        cluster_id: [num_nodes] cluster assignment per node (optional).

    Returns:
        alpha_vec: [num_nodes] array of per-node teleport rates
    """
    # Build adjacency list
    adj = defaultdict(list)
    for s, d in zip(edge_index[0], edge_index[1]):
        adj[int(s)].append(int(d))

    use_cluster_cond = T_c is not None and cluster_id is not None
    if use_cluster_cond:
        K, C_tc, _ = T_c.shape
        per_class_acc_global = np.diag(noise_matrix) if noise_matrix is not None else None
    else:
        per_class_acc = np.diag(noise_matrix)
    alpha_vec = np.zeros(num_nodes, dtype=np.float32)

    # Compute entropy-based confidence gate
    if gnn_entropy is not None:
        labeled_nids = list(labels_dict.keys())
        if labeled_nids:
            labeled_entropy = gnn_entropy[labeled_nids]
            ent_median = np.median(labeled_entropy)
            ent_std = max(np.std(labeled_entropy), 1e-6)
        else:
            ent_median, ent_std = 0.5, 0.5

    for nid, label in labels_dict.items():
        # Per-node diagonal lookup
        if use_cluster_cond:
            k = int(cluster_id[nid])
            if 0 <= label < C_tc and 0 <= k < K:
                node_acc = float(T_c[k, label, label])
                in_range = True
            elif per_class_acc_global is not None and 0 <= label < len(per_class_acc_global):
                node_acc = float(per_class_acc_global[label])
                in_range = True
            else:
                in_range = False
        else:
            in_range = 0 <= label < len(per_class_acc)
            if in_range:
                node_acc = float(per_class_acc[label])

        if in_range:
            noise_factor = 1.0 - node_acc
            # Structural homophily factor
            h_i = compute_local_homophily(nid, labels_dict, adj)
            struct_factor = 1.0 - h_i
            # Combined noise score
            phi_i = noise_factor * gamma + struct_factor * (1 - gamma)

            # Confidence gate: only apply strong α if GNN is uncertain
            if gnn_entropy is not None:
                # Sigmoid gate: high entropy → gate=1 (restrict), low → gate=0 (propagate)
                z = (gnn_entropy[nid] - ent_median) / ent_std
                gate_i = 1.0 / (1.0 + np.exp(-2.0 * z))  # steeper sigmoid
            else:
                gate_i = 1.0  # no gating, full noise-aware α

            # Per-node teleport rate
            alpha_vec[nid] = alpha_base + (1 - alpha_base) * phi_i * gate_i
        else:
            alpha_vec[nid] = alpha_base

    return alpha_vec


def compute_noise_edge_weights(
    edge_index: np.ndarray,
    soft_labels: np.ndarray,
    noise_matrix: np.ndarray,
    beta: float = 0.8,
    min_weight: float = 0.2,
    T_c: Optional[np.ndarray] = None,
    cluster_id: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute noise-informed edge weights from the confusion structure.

    Class-conditional (default):
        confusion(c_i, c_j) = (T[c_i][c_j] + T[c_j][c_i]) / 2  (symmetric)
        w_ij = 1 - beta * confusion(c_i, c_j)

    Cluster-conditional (T_c + cluster_id provided):
        confusion(i, j) = T_c[cluster_id[src], c_i, c_j]       (asymmetric,
                                                                source-cluster)
        w_ij = 1 - beta * confusion(i, j)

    Edges between frequently confused class pairs are down-weighted. In
    cluster-conditional mode the lookup is per-edge and asymmetric — w_ij
    can differ from w_ji because T_c may differ across clusters.

    Args:
        edge_index: [2, E] numpy array
        soft_labels: [N, C] soft label matrix (argmax used for class)
        noise_matrix: C x C transition matrix (used when T_c is None)
        beta: down-weighting strength
        min_weight: floor to prevent edge deletion
        T_c: [K, C, C] cluster-conditional transition matrix (optional).
        cluster_id: [N] cluster assignment per node (optional).

    Returns:
        edge_weights: [E] array
    """
    use_cluster_cond = T_c is not None and cluster_id is not None
    num_classes = T_c.shape[1] if use_cluster_cond else noise_matrix.shape[0]

    # Get predicted class per node
    preds = soft_labels.argmax(axis=1)

    src_classes = preds[edge_index[0]]
    dst_classes = preds[edge_index[1]]
    src_classes = np.clip(src_classes, 0, num_classes - 1)
    dst_classes = np.clip(dst_classes, 0, num_classes - 1)

    if use_cluster_cond:
        K = T_c.shape[0]
        src_clusters = np.clip(cluster_id[edge_index[0]], 0, K - 1).astype(np.int64)
        # Asymmetric: source-cluster determined
        conf_values = T_c[src_clusters, src_classes, dst_classes]
    else:
        # Symmetric confusion (legacy behaviour)
        confusion = (noise_matrix + noise_matrix.T) / 2.0
        conf_values = confusion[src_classes, dst_classes]

    edge_weights = 1.0 - beta * conf_values
    edge_weights = np.clip(edge_weights, min_weight, 1.0).astype(np.float32)

    return edge_weights


def naap_propagate(
    labels_dict: Dict[int, int],
    seed_weights: np.ndarray,
    edge_index: np.ndarray,
    num_nodes: int,
    num_classes: int,
    noise_matrix: np.ndarray,
    alpha_base: float = 0.3,
    beta: float = 0.8,
    gamma: float = 0.6,
    k_max: int = 30,
    patience: int = 3,
    use_edge_weights: bool = True,
    use_progressive: bool = True,
    gnn_entropy: Optional[np.ndarray] = None,
    T_c: Optional[np.ndarray] = None,
    cluster_id: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Noise-Aware Anisotropic Propagation.

    Y_{t+1} = Phi * Y_0 + (I - Phi) * D_w^{-1} W * Y_t

    where Phi is a per-node diagonal teleport matrix parameterized by
    the noise transition matrix T.

    Args:
        labels_dict: {node_id: class_label} seed labels
        seed_weights: [num_nodes] per-node confidence weights
        edge_index: [2, E] numpy array of edges (src, dst)
        num_nodes: total nodes
        num_classes: number of classes
        noise_matrix: C x C noise transition matrix (used when T_c is None)
        alpha_base: minimum teleport rate
        beta: edge down-weighting strength
        gamma: noise vs structure weight
        k_max: maximum propagation iterations
        patience: convergence patience (stable argmax for this many iters)
        use_edge_weights: enable Component 2 (noise-informed edges)
        use_progressive: enable Component 3 (progressive convergence)
        T_c: [K, C, C] cluster-conditional transition matrix (optional). When
             provided together with cluster_id, NAAP uses per-node and
             per-edge T_c lookups instead of the global noise_matrix.
        cluster_id: [num_nodes] cluster assignment per node (optional).

    Returns:
        soft_labels: [num_nodes, num_classes] numpy array
    """
    src = edge_index[0]
    dst = edge_index[1]
    use_cluster_cond = T_c is not None and cluster_id is not None

    # Component 1: Per-node adaptive teleport rates (confidence-gated)
    alpha_vec = compute_per_node_alpha(
        labels_dict, noise_matrix, edge_index, num_nodes,
        alpha_base=alpha_base, gamma=gamma,
        gnn_entropy=gnn_entropy,
        T_c=T_c, cluster_id=cluster_id,
    )

    # Initialize seed label matrix Y_0
    Y_0 = np.zeros((num_nodes, num_classes), dtype=np.float32)
    for nid, label in labels_dict.items():
        if 0 <= label < num_classes:
            Y_0[nid, label] = float(seed_weights[nid])

    # Build adjacency matrix with ADAPTIVE noise-informed edge weights
    # Scale beta by noise matrix reliability: reliable matrix → strong weighting
    have_noise_signal = use_cluster_cond or noise_matrix is not None
    if use_edge_weights and have_noise_signal and beta > 0:
        if use_cluster_cond:
            # Average diag across clusters (per-edge scaling stays via T_c lookup)
            noise_confidence = float(np.mean(np.diagonal(T_c, axis1=1, axis2=2)))
        else:
            noise_confidence = float(np.mean(np.diag(noise_matrix)))
        adaptive_beta = beta * noise_confidence
        logger.info("NAAP edge weights: noise_conf=%.3f, adaptive_beta=%.3f, mode=%s",
                   noise_confidence, adaptive_beta,
                   "cluster_cond" if use_cluster_cond else "class_cond")
        edge_w = compute_noise_edge_weights(
            edge_index, Y_0, noise_matrix, beta=adaptive_beta,
            T_c=T_c, cluster_id=cluster_id,
        )
    else:
        edge_w = np.ones(len(src), dtype=np.float32)

    A = sp.csr_matrix((edge_w, (src, dst)), shape=(num_nodes, num_nodes))
    deg = np.array(A.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    D_inv = sp.diags(1.0 / deg)
    A_norm = D_inv @ A

    # Propagation
    Y = Y_0.copy()
    converged = np.zeros(num_nodes, dtype=bool)
    pred_history = []

    n_active_log = []
    for t in range(k_max):
        # Y_new = Phi * Y_0 + (I - Phi) * A_norm * Y
        propagated = A_norm @ Y
        alpha_col = alpha_vec[:, np.newaxis]  # [N, 1]
        Y_new = alpha_col * Y_0 + (1.0 - alpha_col) * propagated

        if use_progressive:
            # Only update non-converged nodes
            Y[~converged] = Y_new[~converged]
        else:
            Y = Y_new

        # Component 3: Progressive convergence monitoring
        if use_progressive:
            curr_preds = Y.argmax(axis=1)
            pred_history.append(curr_preds.copy())

            if len(pred_history) >= patience:
                # Check if predictions have been stable for `patience` iterations
                stable = np.ones(num_nodes, dtype=bool)
                for k in range(1, patience + 1):
                    stable &= (pred_history[-k] == curr_preds)
                newly_converged = stable & ~converged
                converged |= newly_converged

            n_active = int((~converged).sum())
            n_active_log.append(n_active)

            if converged.all():
                logger.info("NAAP converged at iteration %d (all nodes stable)", t + 1)
                break

        # Update edge weights based on current soft labels (every 5 iterations)
        if use_edge_weights and have_noise_signal and t > 0 and t % 5 == 0:
            edge_w = compute_noise_edge_weights(
                edge_index, Y, noise_matrix, beta=beta,
                T_c=T_c, cluster_id=cluster_id,
            )
            A = sp.csr_matrix((edge_w, (src, dst)), shape=(num_nodes, num_nodes))
            deg = np.array(A.sum(axis=1)).flatten()
            deg[deg == 0] = 1.0
            D_inv = sp.diags(1.0 / deg)
            A_norm = D_inv @ A

    # Log convergence stats
    if use_progressive and n_active_log:
        logger.info(
            "NAAP: %d iters, active nodes: %d->%d (%.0f%% converged early)",
            len(n_active_log), n_active_log[0], n_active_log[-1],
            100 * (1 - n_active_log[-1] / max(n_active_log[0], 1)),
        )
    else:
        logger.info("NAAP: %d iterations completed", k_max)

    return Y.astype(np.float32)
