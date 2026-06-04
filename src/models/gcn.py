"""
GCN model for node classification on text-attributed graphs.

Adapted from LoCLE/LLM-GNN.
Adds convenience methods `fit()` and `encode()` for the NoisePref pipeline.

When trained with labels (supervised), uses CrossEntropyLoss on labeled nodes.
When trained without labels (labels=None), uses a self-supervised DGI-style
objective: maximizes agreement between a node's embedding and a global graph
summary, which produces useful embeddings for downstream subspace clustering.

Usage:
    gnn = GCN(input_dim=768, hidden_dim=64, output_dim=7)
    gnn.fit(data, labels=llm_labels, train_mask=mask, epochs=200)
    embeddings = gnn.encode(data)   # [num_nodes, hidden_dim]
    logits = gnn(data)              # [num_nodes, output_dim]
"""

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv


def _build_uniform_P(num_classes, noise_rate=0.1):
    """Build a uniform noise transition matrix for NoiseAda init."""
    P = np.eye(num_classes) * (1.0 - noise_rate)
    P += noise_rate / num_classes
    return P


class NoiseAda(nn.Module):
    """Learnable noise transition matrix (from LoCLE's noisy.py).

    Models P(noisy_label | true_class) as a C×C stochastic matrix.
    During training: output = softmax(logits) @ P.
    At inference: use logits directly (skip P).

    The GNN learns clean representations, P absorbs annotation noise.
    """

    def __init__(self, num_classes, noise_rate=0.1):
        super().__init__()
        P = torch.FloatTensor(_build_uniform_P(num_classes, noise_rate))
        self.B = nn.Parameter(torch.log(P + 1e-8))

    def forward(self, pred):
        P = F.softmax(self.B, dim=1)
        return pred @ P

logger = logging.getLogger(__name__)


class GCN(nn.Module):
    """
    Multi-layer GCN following LoCLE's architecture.

    Architecture: GCNConv → BatchNorm → ReLU → Dropout (repeated) → GCNConv.
    The last layer maps to output_dim (num_classes) for classification.
    Intermediate layers use hidden_dim.

    Args:
        input_dim:  Feature dimension (e.g. 768 for SentenceBERT).
        hidden_dim: Hidden layer dimension.
        output_dim: Number of classes (output logits dimension).
        num_layers: Number of GCN layers (default 2).
        dropout:    Dropout rate (applied before each conv, matching LoCLE).
        norm:       Whether to use BatchNorm on intermediate layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 7,
        num_layers: int = 2,
        dropout: float = 0.5,
        norm: bool = True,
        normalize_conv: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # normalize_conv=False disables GCNConv's symmetric D^{-1/2} A D^{-1/2}
        # so callers can supply pre-normalised, ASYMMETRIC edge_weight (e.g.
        # row-stochastic TGER weights from src/structure_learning/tger.py).
        self.normalize_conv = normalize_conv

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, output_dim, cached=False, normalize=normalize_conv))
        else:
            # First layer: input_dim → hidden_dim
            self.convs.append(GCNConv(input_dim, hidden_dim, cached=False, normalize=normalize_conv))
            self.norms.append(nn.BatchNorm1d(hidden_dim) if norm else nn.Identity())

            # Middle layers: hidden_dim → hidden_dim
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=False, normalize=normalize_conv))
                self.norms.append(nn.BatchNorm1d(hidden_dim) if norm else nn.Identity())

            # Last layer: hidden_dim → output_dim
            self.convs.append(GCNConv(hidden_dim, output_dim, cached=False, normalize=normalize_conv))

    def forward(self, data) -> torch.Tensor:
        """
        Full forward pass returning logits [num_nodes, output_dim].

        Matches LoCLE's GCN.forward() (baselines/llmgnn/src/models/nn.py:186).
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
        return x

    def encode(self, data) -> torch.Tensor:
        """
        Extract intermediate embeddings [num_nodes, hidden_dim].

        Runs the forward pass but stops before the last GCN layer,
        returning the hidden representation used for subspace clustering
        and node selection.

        For a 1-layer GCN, returns the logits directly (no hidden layer).
        """
        self.eval()
        with torch.no_grad():
            x = data.x
            edge_index = data.edge_index
            edge_weight = getattr(data, "edge_weight", None)

            if self.num_layers == 1:
                # No hidden layer — return logits as embeddings
                if edge_weight is not None:
                    return self.convs[0](x, edge_index, edge_weight)
                return self.convs[0](x, edge_index)

            # Process through all layers except the last
            for i in range(self.num_layers - 1):
                x = F.dropout(x, p=0.0, training=False)  # No dropout at inference
                if edge_weight is not None:
                    x = self.convs[i](x, edge_index, edge_weight)
                else:
                    x = self.convs[i](x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)
            return x

    def forward_with_hidden(self, data):
        """Like forward() but ALSO returns the penultimate hidden representation
        (the last layer before the classification head), WITH gradients — for an
        auxiliary projection/reconstruction head. Returns (logits, hidden).
        For a 1-layer GCN there is no hidden layer, so hidden = input features.
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        hidden = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                hidden = x
        return x, hidden

    def get_state(self) -> dict:
        """Return a copy of the model state dict for checkpointing."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def load_state(self, state: dict):
        """Load a previously saved state dict (for warm-starting)."""
        self.load_state_dict(state)

    def fit(
        self,
        data,
        labels=None,
        train_mask=None,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        val_mask=None,
        val_labels=None,
        early_stop_patience: int = 0,
        val_check_every: int = 1,
    ) -> dict:
        """
        Train the GCN.

        Supervised mode (labels provided):
            Standard CrossEntropyLoss on labeled nodes.

        Unsupervised mode (labels=None):
            Self-supervised via neighbor prediction — trains the GCN to
            produce embeddings where connected nodes are closer than random
            pairs.  This gives meaningful embeddings for subspace clustering
            even without any labels (Round 1 of the pipeline).

        Args:
            data:         PyG Data object.
            labels:       Node labels tensor [num_nodes]. If None, self-supervised.
            train_mask:   Boolean mask for training nodes. If None with labels,
                          uses all nodes with valid labels.
            epochs:       Number of training epochs.
            lr:           Learning rate.
            weight_decay: L2 regularization.
            val_mask:     Optional boolean mask for a held-out validation set
                          (subset of labeled nodes). If provided along with
                          `early_stop_patience > 0`, training tracks val
                          accuracy every `val_check_every` epochs and stops
                          early after `early_stop_patience` epochs with no
                          improvement. Best-val weights are restored.
            val_labels:   Optional labels tensor [num_nodes] to use ONLY for
                          validation-accuracy computation. When provided,
                          val accuracy is `(argmax(logits)[val_mask] == val_labels[val_mask]).mean()`
                          — i.e. the validation target is DIFFERENT from the
                          training labels. This enables LLM-GNN's arxiv
                          protocol: early-stop on OGB gold val labels while
                          training on LLM-annotated labels (the gold labels
                          are NOT used for gradients, only for monitoring).
                          When None (default), falls back to comparing against
                          the training `labels` tensor (legacy behavior).
            early_stop_patience: Patience in epochs for early stopping. 0
                          disables early stopping (default; bit-identical
                          to legacy behavior).
            val_check_every: Check val accuracy every N epochs.

        Returns:
            dict with 'train_loss' history and 'mode' ('supervised'/'self_supervised').
        """
        device = next(self.parameters()).device
        data = data.to(device)

        if labels is not None:
            return self._fit_supervised(
                data, labels, train_mask, epochs, lr, weight_decay,
                val_mask=val_mask,
                val_labels=val_labels,
                early_stop_patience=early_stop_patience,
                val_check_every=val_check_every,
            )
        else:
            return self._fit_self_supervised(data, epochs, lr, weight_decay)

    @staticmethod
    def _bootstrap_loss(logits, labels, beta=0.8):
        """Bootstrap loss (Reed et al., ICLR 2015).

        L = β * CE(y, ŷ) + (1-β) * CE(ŷ_hard, ŷ)

        where y is the noisy label and ŷ_hard = argmax(ŷ) is the model's
        own prediction. This self-corrects toward the model's belief,
        which is closer to truth after warmup. Works well with noisy labels
        because the model learns clean patterns early and bootstrap
        gradually shifts training from noisy labels toward model predictions.
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        with torch.no_grad():
            model_preds = logits.argmax(dim=1)
        model_ce = F.cross_entropy(logits, model_preds, reduction='none')
        return beta * ce_loss + (1 - beta) * model_ce

    @staticmethod
    def _gce_loss(logits, labels, q=0.7):
        """Generalized Cross Entropy loss (Zhang & Sabuncu, NeurIPS 2018).

        GCE = (1 - p_y^q) / q  where p_y = softmax(logit)[true_class].
        Bounded and noise-tolerant: q→0 gives CE, q→1 gives MAE.
        """
        probs = F.softmax(logits, dim=1)
        p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_y = p_y.clamp(min=1e-7)  # numerical stability
        loss = (1.0 - p_y ** q) / q
        return loss  # per-node loss

    @staticmethod
    def _sce_loss(logits, labels, num_classes, beta=0.5):
        """Symmetric Cross-Entropy loss (Wang et al., ICCV 2019).

        SCE = CE + beta * RCE. RCE is bounded → noisy samples can't dominate.
        RCE = -sum_c p_model(c) * log(q_label(c)) where q is smoothed label dist.
        """
        ce = F.cross_entropy(logits, labels, reduction='none')
        probs = F.softmax(logits, dim=1)
        # Build smoothed label distribution: 1-eps for true class, eps/(C-1) for others
        eps = 1.0 / (num_classes + 1)
        q = torch.full_like(probs, eps / (num_classes - 1))
        q.scatter_(1, labels.unsqueeze(1), 1.0 - eps)
        rce = -(probs * torch.log(q + 1e-8)).sum(dim=1)
        return ce + beta * rce  # per-node loss

    @staticmethod
    def _elr_loss(logits, labels, target_running, lam=3.0):
        """Early-Learning Regularization loss (Liu et al., NeurIPS 2020).

        L = CE(y, ŷ) + λ · log(1 − ⟨p, t⟩)

        where p = softmax(logits) (per train node), and t is a running EMA of
        p maintained OUTSIDE this function (target_running is the [N_train, C]
        tensor corresponding to train nodes at this iteration).

        This penalizes the model for pulling away from the early-learned
        target (a "ratchet" that prevents overfitting to noisy labels).
        """
        ce = F.cross_entropy(logits, labels, reduction="none")
        probs = F.softmax(logits, dim=1)
        # (1 - <p, t>) ∈ [0, 1]; log(...) is negative; minimize -> push <p,t> up
        dot = (probs * target_running).sum(dim=1).clamp(min=1e-7, max=1 - 1e-7)
        reg = torch.log(1.0 - dot)
        return ce + lam * reg

    def _fit_supervised(self, data, labels, train_mask, epochs, lr, weight_decay,
                        val_mask=None, val_labels=None, early_stop_patience: int = 0,
                        val_check_every: int = 1):
        """Supervised training with optional NoiseAda noise transition layer.

        If val_mask is provided and early_stop_patience > 0, tracks val
        accuracy every val_check_every epochs, keeps best-val weights, and
        stops early after `early_stop_patience` epochs of no improvement.

        When `val_labels` is provided, val accuracy is computed against
        val_labels[val_mask] instead of labels[val_mask]. This supports the
        LLM-GNN arxiv protocol of early-stopping on OGB gold labels while
        training on LLM-annotated labels (gold labels are NEVER used for
        gradients — only for the early-stop signal).
        """
        labels = labels.to(data.x.device)
        if train_mask is None:
            train_mask = labels >= 0
        train_mask = train_mask.to(data.x.device)
        device = data.x.device

        # Early-stopping bookkeeping
        use_early_stop = (val_mask is not None) and (early_stop_patience > 0)
        if use_early_stop:
            val_mask = val_mask.to(device)
            # val_labels: when provided, validation targets are DECOUPLED from
            # training labels. This is the LLM-GNN arxiv protocol: train on
            # noisy LLM labels, early-stop on OGB gold val. These gold labels
            # are read-only — they feed ONLY the accuracy check below, never
            # the loss.
            if val_labels is not None:
                val_labels = val_labels.to(device)
            best_val_acc = -1.0
            best_state = None
            best_epoch = -1
            epochs_no_improve = 0
            val_acc_history: list = []

        # NoiseAda: learnable noise transition matrix (from LoCLE)
        use_noise_ada = getattr(data, '_use_noise_ada', False)
        noise_ada = None
        if use_noise_ada:
            noise_ada = NoiseAda(self.output_dim, noise_rate=0.1).to(device)

        # Build optimizer — include NoiseAda parameters if used
        params = list(self.parameters())
        if noise_ada is not None:
            params += list(noise_ada.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # Soft targets: LP probability distributions (KL divergence loss)
        soft_targets = getattr(data, '_soft_targets', None)
        weights = getattr(data, '_node_weights', None)
        use_gce = getattr(data, '_use_gce', False)
        gce_q = getattr(data, '_gce_q', 0.7)
        use_sce = getattr(data, '_use_sce', False)
        sce_beta = getattr(data, '_sce_beta', 0.5)
        use_bootstrap = getattr(data, '_use_bootstrap', False)
        bootstrap_beta = getattr(data, '_bootstrap_beta', 0.8)
        use_elr = getattr(data, '_use_elr', False)
        elr_beta = getattr(data, '_elr_beta', 0.7)
        elr_lambda = getattr(data, '_elr_lambda', 3.0)

        # ELR running target: one vector per train node, EMA of softmax probs.
        # Initialized to uniform; updated per epoch.
        elr_target = None
        if use_elr:
            n_train = int(train_mask.sum().item())
            elr_target = torch.full(
                (n_train, self.output_dim), 1.0 / self.output_dim, device=device,
            )

        if soft_targets is not None:
            soft_targets = soft_targets.to(device)
            use_soft = True
        else:
            use_soft = False
            label_smoothing = getattr(data, '_label_smoothing', 0.0)
            if not use_gce and not use_sce and not use_bootstrap and not use_elr:
                if weights is not None:
                    weights = weights.to(device)
                    loss_fn = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
                else:
                    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if weights is not None:
            weights = weights.to(device)

        # DropEdge: randomly drop edges each epoch to reduce noise propagation
        dropedge_rate = getattr(data, '_dropedge_rate', 0.0)
        orig_edge_index = data.edge_index.clone() if dropedge_rate > 0 else None
        orig_edge_weight = None
        if dropedge_rate > 0 and hasattr(data, 'edge_weight') and data.edge_weight is not None:
            orig_edge_weight = data.edge_weight.clone()

        # Degree-aware DropEdge: per-edge keep prob scaled so that
        # high-degree source nodes drop at the configured rate while
        # low-degree sources keep most of their edges. ``edge_keep_p`` is
        # a fixed [E] tensor used by the dropedge mask below.
        dropedge_degree_aware = bool(getattr(data, '_dropedge_degree_aware', False))
        edge_keep_p = None
        if dropedge_rate > 0 and dropedge_degree_aware:
            from torch_geometric.utils import degree as _deg
            src = (orig_edge_index if orig_edge_index is not None else data.edge_index)[0]
            deg = _deg(src, num_nodes=data.x.size(0)).float()
            # Reference scale: median non-zero degree.
            ref = float(deg[deg > 0].median().item()) if (deg > 0).any() else 1.0
            deg_src = deg[src]
            # drop_p_per_edge = rate * min(1, deg_src / ref)
            drop_p = dropedge_rate * (deg_src / max(ref, 1e-8)).clamp_(0.0, 1.0)
            edge_keep_p = (1.0 - drop_p).clamp_(0.0, 1.0).to(device)

        # ── RTAA dynamic edge refinement (data-level, backbone-agnostic) ──
        # When enabled, every ``rtaa_refresh_every`` epochs we recompute
        # synthetic edges between high-confidence (prob >= τ) nodes that
        # share a predicted class, append them with weight 1.0, and use
        # the augmented graph for the rest of training.  The base T_c
        # weights are preserved on the original edges via ``data.edge_weight``.
        use_rtaa_dynamic = bool(getattr(data, '_use_rtaa_dynamic', False))
        rtaa_threshold = float(getattr(data, '_rtaa_conf_threshold', 0.9))
        rtaa_refresh_every = int(getattr(data, '_rtaa_refresh_every', 50))
        rtaa_k_per_class = int(getattr(data, '_rtaa_k_per_class', 20))
        rtaa_max_added = int(getattr(data, '_rtaa_max_added', 5000))
        rtaa_base_ei = None
        rtaa_base_ew = None
        rtaa_synth_ei = None
        rtaa_synth_ew = None
        if use_rtaa_dynamic:
            rtaa_base_ei = getattr(data, '_rtaa_base_edge_index', None)
            rtaa_base_ew = getattr(data, '_rtaa_base_edge_weight', None)
            if rtaa_base_ei is None:
                rtaa_base_ei = data.edge_index.clone()
            if rtaa_base_ew is None and hasattr(data, 'edge_weight') and data.edge_weight is not None:
                rtaa_base_ew = data.edge_weight.clone()
            rtaa_base_ei = rtaa_base_ei.to(device)
            if rtaa_base_ew is not None:
                rtaa_base_ew = rtaa_base_ew.to(device)

        # may18 v7: Consistency regularization (UDA/FixMatch-style) on unlabeled nodes.
        use_consistency = bool(getattr(data, '_use_consistency_reg', False))
        if use_consistency:
            cr_lambda = float(getattr(data, '_consistency_lambda', 1.0))
            cr_aug_dropedge = float(getattr(data, '_consistency_aug_dropedge', 0.5))
            cr_aug_feat_mask = float(getattr(data, '_consistency_aug_feat_mask', 0.1))
            cr_warmup = int(getattr(data, '_consistency_warmup_epochs', 50))
            cr_conf_thresh = float(getattr(data, '_consistency_conf_threshold', 0.0))
            # Need full edge index for augmentation; if dropedge wasn't on, save once now.
            if orig_edge_index is None:
                orig_edge_index = data.edge_index.clone()
                if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                    orig_edge_weight = data.edge_weight.clone()
            cr_non_train_mask = ~train_mask
            cr_n_unlabeled = int(cr_non_train_mask.sum().item())

        # Warmup loss capture: record per-node loss at an early epoch
        # for DivideMix-style clean/noisy separation (Li et al., ICLR 2020).
        # The memorization effect means low-loss nodes at warmup are more likely
        # to have correct labels.
        warmup_epoch = getattr(data, '_warmup_loss_epoch', 0)
        self._warmup_per_node_loss = None  # will be set at warmup_epoch

        self.train()
        loss_history = []
        # MoDis (WWW 2024): track per-node prediction history for disagreement
        pred_snapshots = []
        snapshot_interval = max(1, epochs // 5)  # 5 snapshots during training

        for epoch in range(epochs):
            # Compose effective base graph for this epoch (RTAA + synthetic).
            if use_rtaa_dynamic:
                eff_ei = rtaa_base_ei
                eff_ew = rtaa_base_ew
                if rtaa_synth_ei is not None and rtaa_synth_ei.size(1) > 0:
                    eff_ei = torch.cat([eff_ei, rtaa_synth_ei], dim=1)
                    if eff_ew is not None:
                        eff_ew = torch.cat([eff_ew, rtaa_synth_ew], dim=0)
                base_for_epoch_ei = eff_ei
                base_for_epoch_ew = eff_ew
            else:
                base_for_epoch_ei = orig_edge_index
                base_for_epoch_ew = orig_edge_weight

            # DropEdge / RTAA edge assignment for this epoch.
            if dropedge_rate > 0:
                rand_E = torch.rand(base_for_epoch_ei.size(1), device=device)
                if edge_keep_p is not None and base_for_epoch_ei.size(1) == edge_keep_p.numel():
                    # Per-edge keep probability — high for low-degree edges.
                    edge_mask = rand_E < edge_keep_p
                else:
                    edge_mask = rand_E > dropedge_rate
                data.edge_index = base_for_epoch_ei[:, edge_mask]
                if base_for_epoch_ew is not None:
                    data.edge_weight = base_for_epoch_ew[edge_mask]
            elif use_rtaa_dynamic:
                data.edge_index = base_for_epoch_ei
                if base_for_epoch_ew is not None:
                    data.edge_weight = base_for_epoch_ew

            optimizer.zero_grad()
            logits = self.forward(data)

            # Forward loss correction: map predictions through noise transition
            # (Patrini et al., CVPR 2017) — corrects for known label noise patterns
            noise_T = getattr(data, '_noise_transition', None)
            if noise_T is not None:
                probs = F.softmax(logits, dim=1)
                corrected_probs = probs @ noise_T  # P(noisy|x) = P(clean|x) @ T
                train_logits = torch.log(corrected_probs + 1e-8)
            elif noise_ada is not None:
                probs = F.softmax(logits, dim=1)
                noisy_probs = noise_ada(probs)
                train_logits = torch.log(noisy_probs + 1e-8)
            else:
                train_logits = logits

            if use_soft:
                log_probs = F.log_softmax(train_logits[train_mask], dim=1)
                target_probs = soft_targets[train_mask]
                target_probs = target_probs / (target_probs.sum(dim=1, keepdim=True) + 1e-8)
                loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
            elif use_sce:
                per_node_loss = self._sce_loss(train_logits[train_mask], labels[train_mask],
                                               self.output_dim, beta=sce_beta)
                if weights is not None:
                    w = weights[train_mask]
                    w = w / (w.sum() + 1e-8) * w.numel()
                    loss = (per_node_loss * w).mean()
                else:
                    loss = per_node_loss.mean()
            elif use_gce:
                per_node_loss = self._gce_loss(train_logits[train_mask], labels[train_mask], q=gce_q)
                if weights is not None:
                    w = weights[train_mask]
                    w = w / (w.sum() + 1e-8) * w.numel()
                    loss = (per_node_loss * w).mean()
                else:
                    loss = per_node_loss.mean()
            elif use_bootstrap:
                per_node_loss = self._bootstrap_loss(
                    train_logits[train_mask], labels[train_mask], beta=bootstrap_beta,
                )
                if weights is not None:
                    w = weights[train_mask]
                    w = w / (w.sum() + 1e-8) * w.numel()
                    loss = (per_node_loss * w).mean()
                else:
                    loss = per_node_loss.mean()
            elif use_elr:
                # Per-train-node softmax (detached for EMA update)
                with torch.no_grad():
                    train_probs_now = F.softmax(train_logits[train_mask], dim=1)
                # EMA update of running target
                elr_target = elr_beta * elr_target + (1 - elr_beta) * train_probs_now
                per_node_loss = self._elr_loss(
                    train_logits[train_mask], labels[train_mask],
                    target_running=elr_target, lam=elr_lambda,
                )
                if weights is not None:
                    w = weights[train_mask]
                    w = w / (w.sum() + 1e-8) * w.numel()
                    loss = (per_node_loss * w).mean()
                else:
                    loss = per_node_loss.mean()
            elif weights is not None:
                per_node_loss = loss_fn(train_logits[train_mask], labels[train_mask])
                w = weights[train_mask]
                w = w / (w.sum() + 1e-8) * w.numel()
                loss = (per_node_loss * w).mean()
            else:
                loss = loss_fn(train_logits[train_mask], labels[train_mask])

            # Entropy regularization: prevent overconfident predictions on noisy labels
            entropy_reg = getattr(data, '_entropy_reg', 0.0)
            if entropy_reg > 0:
                all_probs = F.softmax(logits, dim=1)
                neg_entropy = (all_probs * torch.log(all_probs + 1e-8)).sum(dim=1).mean()
                loss = loss + entropy_reg * neg_entropy

            # may18 v7: Consistency Regularization on non-selected nodes.
            # The "main" logits computed above are on a (possibly DropEdged) graph
            # — that's our weak view (teacher). Build a STRONG view: more aggressive
            # edge drop + feature masking. KL(student | teacher.detach()) on
            # non-train nodes drives the GNN to be invariant under augmentation,
            # which improves generalization to held-out nodes.
            if use_consistency and cr_n_unlabeled > 0:
                # Linear warmup of consistency weight to avoid early collapse.
                cur_lambda = cr_lambda * min(1.0, (epoch + 1) / max(cr_warmup, 1))

                # Snapshot current data state
                saved_x = data.x
                saved_ei = data.edge_index
                saved_ew = getattr(data, 'edge_weight', None)

                # Strong view: edge drop
                aug_em = torch.rand(orig_edge_index.size(1), device=device) > cr_aug_dropedge
                data.edge_index = orig_edge_index[:, aug_em]
                if orig_edge_weight is not None:
                    data.edge_weight = orig_edge_weight[aug_em]
                else:
                    if hasattr(data, 'edge_weight'):
                        data.edge_weight = None
                # Strong view: feature mask (set masked rows to zero)
                feat_keep = (torch.rand(data.x.size(0), 1, device=device) > cr_aug_feat_mask).float()
                data.x = data.x * feat_keep

                logits_aug = self.forward(data)

                # Restore original data state
                data.x = saved_x
                data.edge_index = saved_ei
                if saved_ew is not None:
                    data.edge_weight = saved_ew

                teacher_full = F.softmax(logits[cr_non_train_mask], dim=1).detach()
                student_full = F.log_softmax(logits_aug[cr_non_train_mask], dim=1)
                # FixMatch-style confidence threshold: only enforce consistency
                # where teacher's max softmax >= cr_conf_thresh. Skips uncertain
                # predictions to avoid echo-chamber on low-confidence teachers.
                if cr_conf_thresh > 0.0:
                    max_p, _ = teacher_full.max(dim=1)
                    keep = max_p >= cr_conf_thresh
                    n_kept = int(keep.sum().item())
                    if n_kept > 0:
                        teacher = teacher_full[keep]
                        student = student_full[keep]
                        consistency_loss = F.kl_div(student, teacher, reduction='batchmean')
                        loss = loss + cur_lambda * consistency_loss
                else:
                    consistency_loss = F.kl_div(student_full, teacher_full, reduction='batchmean')
                    loss = loss + cur_lambda * consistency_loss

            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            # Capture per-node loss at warmup epoch for DivideMix-style denoising
            if warmup_epoch > 0 and epoch == warmup_epoch - 1:
                self.eval()
                with torch.no_grad():
                    warmup_logits = self.forward(data)
                    warmup_ce = F.cross_entropy(
                        warmup_logits[train_mask], labels[train_mask], reduction='none'
                    )
                    full_loss = torch.zeros(warmup_logits.size(0), device=device)
                    full_loss[train_mask] = warmup_ce
                    self._warmup_per_node_loss = full_loss.cpu()
                self.train()
                logger.info("Captured warmup per-node loss at epoch %d", epoch + 1)

            # MoDis: snapshot predictions periodically
            if (epoch + 1) % snapshot_interval == 0 or epoch == epochs - 1:
                self.eval()
                with torch.no_grad():
                    snap_logits = self.forward(data)
                    pred_snapshots.append(snap_logits.argmax(dim=1).cpu())
                self.train()

            # Held-out val tracking + early stopping
            if use_early_stop and ((epoch + 1) % val_check_every == 0 or epoch == epochs - 1):
                self.eval()
                with torch.no_grad():
                    val_logits = self.forward(data)
                    val_preds = val_logits.argmax(dim=1)
                    # Compare against val_labels (gold) when provided; else
                    # fall back to training labels (legacy path).
                    _val_target = val_labels if val_labels is not None else labels
                    val_correct = (val_preds[val_mask] == _val_target[val_mask]).float()
                    val_acc = float(val_correct.mean().item()) if val_correct.numel() > 0 else 0.0
                self.train()
                val_acc_history.append((epoch + 1, val_acc))
                if val_acc > best_val_acc + 1e-6:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += val_check_every
                if epochs_no_improve >= early_stop_patience:
                    logger.info(
                        "Early stopping at epoch %d (best val_acc=%.4f @ epoch %d, patience=%d)",
                        epoch + 1, best_val_acc, best_epoch, early_stop_patience,
                    )
                    break

        # Store prediction stability (MoDis memory disagreement)
        if len(pred_snapshots) >= 2:
            stacked = torch.stack(pred_snapshots)  # [num_snapshots, num_nodes]
            # Disagreement = fraction of snapshot pairs that disagree
            n_snap = stacked.size(0)
            disagree = torch.zeros(stacked.size(1))
            for i in range(n_snap):
                for j in range(i + 1, n_snap):
                    disagree += (stacked[i] != stacked[j]).float()
            disagree /= (n_snap * (n_snap - 1) / 2)
            self._modis_disagreement = disagree  # [num_nodes], 0=stable, 1=unstable

        # Restore original edges after training so callers that inspect
        # data outside the loop (e.g., gnn.eval() + forward in the pipeline)
        # see a self-consistent (edge_index, edge_weight) pair.
        if use_rtaa_dynamic:
            data.edge_index = rtaa_base_ei
            if rtaa_base_ew is not None:
                data.edge_weight = rtaa_base_ew
            else:
                if hasattr(data, "edge_weight"):
                    data.edge_weight = None
        elif dropedge_rate > 0:
            data.edge_index = orig_edge_index
            if orig_edge_weight is not None:
                data.edge_weight = orig_edge_weight

        # Restore best-val weights if early stopping tracked one
        if use_early_stop and best_state is not None:
            self.load_state_dict(best_state)
            self._val_acc_history = val_acc_history
            self._best_val_acc = best_val_acc
            self._best_val_epoch = best_epoch

        mode_str = "NoiseAda+" if noise_ada else ""
        if use_soft:
            mode_str += "soft-target KL"
        elif use_elr:
            mode_str += f"ELR(beta={elr_beta:.2f}, lam={elr_lambda:.1f})"
        elif use_sce:
            mode_str += f"SCE(beta={sce_beta:.2f})"
        elif use_gce:
            mode_str += f"GCE(q={gce_q:.1f})"
        elif use_bootstrap:
            mode_str += f"Bootstrap(beta={bootstrap_beta:.2f})"
        elif weights is not None:
            mode_str += "weighted-CE"
        else:
            mode_str += "CE"
        if dropedge_rate > 0:
            mode_str += f", DropEdge={dropedge_rate:.1f}"
        if use_early_stop:
            mode_str += (
                f", EarlyStop(best_val={best_val_acc:.4f}@e{best_epoch}, "
                f"patience={early_stop_patience})"
            )
        actual_epochs = len(loss_history)
        logger.info(
            "Supervised training: %d epochs (requested=%d), final loss=%.4f (%s)",
            actual_epochs, epochs, loss_history[-1], mode_str,
        )
        result = {"train_loss": loss_history, "mode": "supervised"}
        if use_early_stop:
            result["val_acc_history"] = val_acc_history
            result["best_val_acc"] = best_val_acc
            result["best_val_epoch"] = best_epoch
        return result

    def _fit_self_supervised(self, data, epochs, lr, weight_decay):
        """
        Self-supervised training via contrastive neighbor prediction.

        Objective: for each edge (u, v), the embeddings of u and v should
        be more similar than embeddings of u and a random node.  This is a
        simplified DGI/GraphSAGE unsupervised objective.

        Uses the hidden layer output (before last conv) as the embedding
        space, with a bilinear discriminator.
        """
        device = data.x.device
        num_nodes = data.x.size(0)

        # Bilinear discriminator for positive/negative pair scoring
        disc = nn.Bilinear(self.hidden_dim, self.hidden_dim, 1).to(device)

        params = list(self.parameters()) + list(disc.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        self.train()
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Get hidden embeddings (before last layer)
            x = data.x
            edge_index = data.edge_index
            edge_weight = getattr(data, "edge_weight", None)

            for i in range(self.num_layers - 1):
                x = F.dropout(x, p=self.dropout, training=True)
                if edge_weight is not None:
                    x = self.convs[i](x, edge_index, edge_weight)
                else:
                    x = self.convs[i](x, edge_index)
                x = self.norms[i](x)
                x = F.relu(x)

            embeddings = x  # [num_nodes, hidden_dim]

            # Graph-level summary (mean pooling)
            summary = torch.sigmoid(embeddings.mean(dim=0, keepdim=True))  # [1, hidden_dim]
            summary = summary.expand(num_nodes, -1)  # [num_nodes, hidden_dim]

            # Positive scores: real embeddings vs summary
            pos_scores = disc(embeddings, summary).squeeze()

            # Negative scores: shuffled embeddings vs summary
            perm = torch.randperm(num_nodes, device=device)
            neg_scores = disc(embeddings[perm], summary).squeeze()

            # Binary cross-entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones(num_nodes, device=device),
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros(num_nodes, device=device),
            )
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        logger.info(
            "Self-supervised training: %d epochs, final loss=%.4f",
            epochs, loss_history[-1],
        )
        return {"train_loss": loss_history, "mode": "self_supervised"}

    def reset_parameters(self):
        """Re-initialize all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def __repr__(self) -> str:
        return (
            f"GCN(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, layers={self.num_layers}, "
            f"dropout={self.dropout})"
        )
