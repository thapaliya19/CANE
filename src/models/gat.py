"""
GAT model for node classification on text-attributed graphs.

Matches LoCLE's GAT2 architecture.
Uses multi-head attention with ELU activation, concatenation in intermediate
layers, and averaging in the output layer.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv

from src.models.gcn import NoiseAda, GCN

logger = logging.getLogger(__name__)


class GAT(nn.Module):
    """
    Multi-layer GAT matching LoCLE's architecture.

    Architecture: GATConv(concat) → BatchNorm → ELU → Dropout (repeated) → GATConv(avg).
    Intermediate layers concatenate attention heads, output layer averages them.

    Args:
        input_dim:  Feature dimension (e.g. 768 for SentenceBERT).
        hidden_dim: Per-head hidden dimension.
        output_dim: Number of classes.
        num_layers: Number of GAT layers (default 2).
        num_heads:  Number of attention heads (default 8).
        num_out_heads: Number of output heads (default 1).
        dropout:    Feature dropout rate.
        attn_drop:  Attention dropout rate.
        norm:       Whether to use BatchNorm.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 7,
        num_layers: int = 2,
        num_heads: int = 8,
        num_out_heads: int = 1,
        dropout: float = 0.5,
        attn_drop: float = 0.5,
        norm: bool = True,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        # When edge_dim is set, GATConv consumes `edge_attr` (built from
        # data.edge_weight) as a 1-D feature in its attention computation.
        # This is the standard PyG hook for edge-weight-aware attention —
        # no custom Module subclass needed.
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(input_dim, output_dim, heads=num_out_heads,
                        concat=False, dropout=attn_drop, edge_dim=edge_dim)
            )
        else:
            # First layer: input_dim → hidden_dim * num_heads (concat)
            self.convs.append(
                GATConv(input_dim, hidden_dim, heads=num_heads,
                        concat=True, dropout=attn_drop, edge_dim=edge_dim)
            )
            self.norms.append(
                nn.BatchNorm1d(hidden_dim * num_heads) if norm else nn.Identity()
            )

            # Middle layers: hidden_dim * num_heads → hidden_dim * num_heads
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                            concat=True, dropout=attn_drop, edge_dim=edge_dim)
                )
                self.norms.append(
                    nn.BatchNorm1d(hidden_dim * num_heads) if norm else nn.Identity()
                )

            # Last layer: hidden_dim * num_heads → output_dim (average heads)
            self.convs.append(
                GATConv(hidden_dim * num_heads, output_dim, heads=num_out_heads,
                        concat=False, dropout=attn_drop, edge_dim=edge_dim)
            )

    def _edge_attr(self, data) -> torch.Tensor | None:
        if self.edge_dim is None:
            return None
        ew = getattr(data, "edge_weight", None)
        if ew is None:
            return None
        if ew.dim() == 1:
            ew = ew.unsqueeze(-1)
        return ew.float()

    def forward(self, data) -> torch.Tensor:
        """Full forward pass returning logits [num_nodes, output_dim]."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = self._edge_attr(data)

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.elu(x)
        return x

    def encode(self, data) -> torch.Tensor:
        """Extract intermediate embeddings [num_nodes, hidden_dim * num_heads]."""
        self.eval()
        with torch.no_grad():
            x = data.x
            edge_index = data.edge_index
            edge_attr = self._edge_attr(data)

            def _conv(layer, h):
                if edge_attr is not None:
                    return layer(h, edge_index, edge_attr=edge_attr)
                return layer(h, edge_index)

            if self.num_layers == 1:
                return _conv(self.convs[0], x)

            for i in range(self.num_layers - 1):
                x = _conv(self.convs[i], x)
                x = self.norms[i](x)
                x = F.elu(x)
            return x

    def forward_with_hidden(self, data):
        """Like forward() but ALSO returns the penultimate hidden representation
        (before the classification head), WITH gradients. Returns (logits, hidden).
        For a 1-layer GAT, hidden = input features."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = self._edge_attr(data)
        hidden = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if edge_attr is not None:
                x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = self.norms[i](x)
                x = F.elu(x)
                hidden = x
        return x, hidden

    def get_state(self) -> dict:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def load_state(self, state: dict):
        self.load_state_dict(state)

    def fit(
        self,
        data,
        labels=None,
        train_mask=None,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
    ) -> dict:
        """Train the GAT (same interface as GCN.fit)."""
        device = next(self.parameters()).device
        data = data.to(device)

        if labels is not None:
            return self._fit_supervised(data, labels, train_mask, epochs, lr, weight_decay)
        else:
            return self._fit_self_supervised(data, epochs, lr, weight_decay)

    @staticmethod
    def _bootstrap_loss(logits, labels, beta=0.8):
        """Bootstrap loss (Reed et al., ICLR 2015)."""
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        with torch.no_grad():
            model_preds = logits.argmax(dim=1)
        model_ce = F.cross_entropy(logits, model_preds, reduction='none')
        return beta * ce_loss + (1 - beta) * model_ce

    @staticmethod
    def _gce_loss(logits, labels, q=0.7):
        """Generalized Cross Entropy loss (Zhang & Sabuncu, NeurIPS 2018)."""
        probs = F.softmax(logits, dim=1)
        p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_y = p_y.clamp(min=1e-7)
        loss = (1.0 - p_y ** q) / q
        return loss

    @staticmethod
    def _sce_loss(logits, labels, num_classes, beta=0.5):
        """Symmetric Cross-Entropy loss (Wang et al., ICCV 2019)."""
        ce = F.cross_entropy(logits, labels, reduction='none')
        probs = F.softmax(logits, dim=1)
        eps = 1.0 / (num_classes + 1)
        q = torch.full_like(probs, eps / (num_classes - 1))
        q.scatter_(1, labels.unsqueeze(1), 1.0 - eps)
        rce = -(probs * torch.log(q + 1e-8)).sum(dim=1)
        return ce + beta * rce

    def _fit_supervised(self, data, labels, train_mask, epochs, lr, weight_decay):
        """Supervised training with optional NoiseAda noise transition layer."""
        labels = labels.to(data.x.device)
        if train_mask is None:
            train_mask = labels >= 0
        train_mask = train_mask.to(data.x.device)
        device = data.x.device

        # NoiseAda: learnable noise transition matrix
        use_noise_ada = getattr(data, '_use_noise_ada', False)
        noise_ada = None
        if use_noise_ada:
            noise_ada = NoiseAda(self.output_dim, noise_rate=0.1).to(device)

        params = list(self.parameters())
        if noise_ada is not None:
            params += list(noise_ada.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

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

        # ── RTAA dynamic edge refinement (data-level, vanilla GATConv) ──
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

        warmup_epoch = getattr(data, '_warmup_loss_epoch', 0)
        self._warmup_per_node_loss = None

        self.train()
        loss_history = []
        pred_snapshots = []
        snapshot_interval = max(1, epochs // 5)

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
                edge_mask = torch.rand(base_for_epoch_ei.size(1), device=device) > dropedge_rate
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
            noise_T = getattr(data, '_noise_transition', None)
            if noise_T is not None:
                probs = F.softmax(logits, dim=1)
                corrected_probs = probs @ noise_T
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
                with torch.no_grad():
                    train_probs_now = F.softmax(train_logits[train_mask], dim=1)
                elr_target = elr_beta * elr_target + (1 - elr_beta) * train_probs_now
                per_node_loss = GCN._elr_loss(
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

            if (epoch + 1) % snapshot_interval == 0 or epoch == epochs - 1:
                self.eval()
                with torch.no_grad():
                    snap_logits = self.forward(data)
                    pred_snapshots.append(snap_logits.argmax(dim=1).cpu())
                self.train()

        if len(pred_snapshots) >= 2:
            stacked = torch.stack(pred_snapshots)
            n_snap = stacked.size(0)
            disagree = torch.zeros(stacked.size(1))
            for i in range(n_snap):
                for j in range(i + 1, n_snap):
                    disagree += (stacked[i] != stacked[j]).float()
            disagree /= (n_snap * (n_snap - 1) / 2)
            self._modis_disagreement = disagree

        # Restore original edges after training so post-fit forward passes
        # (e.g., for entropy ranking in selection) see a self-consistent
        # (edge_index, edge_weight) pair.
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

        mode_str = "NoiseAda+" if noise_ada else ""
        mode_str += "soft-target KL" if use_soft else ("SCE(beta={:.1f})".format(sce_beta) if use_sce else ("GCE(q={:.1f})".format(gce_q) if use_gce else ("weighted-CE" if weights is not None else "CE")))
        if dropedge_rate > 0:
            mode_str += f", DropEdge={dropedge_rate:.1f}"
        logger.info("GAT supervised training: %d epochs, final loss=%.4f (%s)", epochs, loss_history[-1], mode_str)
        return {"train_loss": loss_history, "mode": "supervised"}

    def _fit_self_supervised(self, data, epochs, lr, weight_decay):
        device = data.x.device
        num_nodes = data.x.size(0)
        # Use hidden_dim * num_heads for the embedding dimension
        embed_dim = self.hidden_dim * self.num_heads if self.num_layers > 1 else self.output_dim

        disc = nn.Bilinear(embed_dim, embed_dim, 1).to(device)
        params = list(self.parameters()) + list(disc.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        self.train()
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            x = data.x
            edge_index = data.edge_index

            for i in range(self.num_layers - 1):
                x = F.dropout(x, p=self.dropout, training=True)
                x = self.convs[i](x, edge_index)
                x = self.norms[i](x)
                x = F.elu(x)

            embeddings = x
            summary = torch.sigmoid(embeddings.mean(dim=0, keepdim=True))
            summary = summary.expand(num_nodes, -1)

            pos_scores = disc(embeddings, summary).squeeze()
            perm = torch.randperm(num_nodes, device=device)
            neg_scores = disc(embeddings[perm], summary).squeeze()

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones(num_nodes, device=device))
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros(num_nodes, device=device))
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        logger.info("GAT self-supervised: %d epochs, final loss=%.4f", epochs, loss_history[-1])
        return {"train_loss": loss_history, "mode": "self_supervised"}

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def __repr__(self) -> str:
        return (
            f"GAT(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, layers={self.num_layers}, "
            f"heads={self.num_heads}, dropout={self.dropout})"
        )
