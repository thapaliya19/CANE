# CANE: Cluster-Aware Noise Estimation

Official code for the paper **“Where LLM Annotators Fail: Label-Free Learning on
Graphs with LLMs.”**

CANE is a **label-free** node-classification framework for text-attributed
graphs. An LLM annotates a small budget of nodes zero-shot; CANE then estimates
*where* those annotations are reliable and uses that estimate to decide which
pseudo-labels to trust and which labels to correct — **without any ground-truth
labels at any stage**.

> **TL;DR** — LLM annotation errors on graphs are not just class-dependent but
> **region-dependent**: within a single true class, reliability varies sharply
> across feature-space clusters. CANE estimates this *cluster-conditional*
> reliability label-free and applies it throughout the pipeline, improving over
> the strongest label-free baselines — with the largest gains exactly where
> cluster-conditional noise is strongest.

---

## Method

Node attributes (paper abstracts, web pages, product descriptions) let an LLM
read a node and emit a candidate label. These labels are noisy, and prior
label-free methods model the noise as *global* or *class-conditional*. CANE
instead estimates a **cluster-conditional transition matrix** `T_c` over
`K = 2C` feature-space clusters and applies it wherever LLM-noisy labels enter
the pipeline.

**Pipeline (2 stages, `max_rounds = 1`):**

1. **Coverage selection + annotation.** Nodes are embedded with a self-supervised
   GraphMAE2 encoder and grouped into `K = 2C` clusters. The densest, most
   central node in each cluster is selected as a seed and annotated by the LLM.
   The first `ρ = 0.4` of the budget doubles as a **probe** set (no extra cost).
2. **Cross-modal `T_c` estimation (within budget, probe nodes only).** For each
   probe node, reliability = agreement between its LLM label and its graph- and
   feature-kNN neighbours, aggregated per `(cluster, class)` into `T_c`. This
   measures *accuracy*, not cluster purity, and tracks an oracle per-cluster
   accuracy (Pearson `r ≈ 0.81–0.96`). `T_c` then drives:
   - **`T_c`-gated pseudo-label expansion** (α = 0.3),
   - **`T_c`-modulated iterative label correction (ILC) self-training** (scale 0.5),
   - the final GNN trained on the refined labels.

   Forward correction is **off** by default — the estimated matrix is too diffuse
   to invert safely.

**Evaluation protocol.** Active selection labels `B` nodes; the GNN is trained on
those labels and evaluated against ground truth on **all remaining nodes** (no
fixed train/val/test split). Metrics: accuracy, macro-F1, NMI, ARI.

---

## Repository structure

```
cane/
├── README.md
├── run.sbatch                 # run ONE config on SLURM (CFG=<config name>)
├── run_all.sh                 # submit the full main table
├── src/
│   ├── data_loading/tag_dataset.py          # dataset loader (5 benchmarks)
│   ├── evaluation/evaluator.py              # accuracy / macro-F1 / NMI / ARI
│   ├── llm_annotation/annotator.py          # LLM query + annotation-cache interface
│   ├── models/{gcn,gat}.py                  # GCN / GAT backbones (+ ELR, loss pruning)
│   ├── noise_handling/
│   │   ├── agreement_partition.py           # GNN/LLM agree–disagree split
│   │   ├── budget_manager.py                # LLM-query budget accounting
│   │   ├── noise_profiler.py                # confusion-matrix probing
│   │   ├── pseudo_samples.py                # class-representative probe text
│   │   └── naap.py                          # noise-aware label propagation
│   ├── node_selection/noise_aware_selector.py   # coverage selection + harmonicity
│   └── training/
│       ├── noisepref_pipeline.py            # the CANE pipeline
│       └── run.py                           # cached-annotator entry helper
├── autoresearch/
│   ├── prepare.py                           # annotators + evaluation helpers
│   └── experiment_runner.py                 # entry point: config JSON → run
├── configs/                   # one example config (derive others — see below)
└── data/                      # bundled: node features, annotation caches, T_c
```

---

## Installation

Python 3.10 with PyTorch and PyTorch Geometric. Core dependencies:

```bash
pip install torch torchvision                     # PyTorch >= 2.3
pip install torch_geometric                        # PyG >= 2.5
pip install numpy scipy scikit-learn pyyaml
pip install openai                                 # only for live LLM annotation
```

LLM annotation uses cached annotations by default (`annotation_mode=locle_cache`),
so an OpenAI API key is **not** required to reproduce the paper.

---

## Data

All data needed to run and reproduce the paper is **bundled in `data/`** — the
repo runs out of the box, no downloads required:

```
data/
├── cora_random_sbert.pt          # SBERT features + Data(x, edge_index, y, raw_texts)
├── citeseer2_random_sbert.pt
├── pubmed_random_sbert.pt
├── wikics_fixed_sbert.pt
├── dblp/                         # geometric_data_processed.pt + raw_texts.pkl
├── annotations/                  # LLM annotation caches (gpt-3.5-turbo, gpt-4o-mini)
└── tc/<dataset>/                 # T_c + GraphMAE2 cluster assignments (K = C, 1.5C, 2C, 3C)
```

- **Node features and annotation caches** follow the LoCLE format
  ([HKBU-LAGAS/Locle](https://github.com/HKBU-LAGAS/Locle)); each `.pt` is a
  `torch_geometric` `Data` object with `x`, `edge_index`, `y`, and `raw_texts`.
- **`tc/<dataset>/`** holds the precomputed label-free transition matrix and the
  GraphMAE2 cluster assignments.

Config paths are **repo-relative**, so commands must be run **from the repository
root** (`run.sbatch` already does this). A quick run needs only `cora`; the full
main table uses all five datasets.

---

## Running experiments

One example config is included — `configs/wbXmod_cora_gcn_s0.json` (Cora, GCN,
seed 0, the main-table setting). Every experiment is one config JSON; **run from
the repository root** so the relative data paths resolve:

```bash
# quick run — one cell (~30s on a GPU)
python autoresearch/experiment_runner.py --config configs/wbXmod_cora_gcn_s0.json
```

Results → `autoresearch/results/<tag>_<dataset>_<backbone>_s<seed>.json`
(accuracy, macro-F1, NMI, ARI, per-round details).

### Deriving the other experiments

Copy the example config and change a few fields. **All data is already bundled
under `data/`.**

| To run… | Edit |
|---|---|
| **Other backbone** | `backbone`: `"gcn"` → `"gat"` |
| **Other seed** | `seed` and `seeds` (e.g. `"seed": 3`, `"seeds": [3]`) |
| **Other dataset** | `dataset` + `cluster_T_c_path` + `cluster_id_path` (table below) |
| **gpt-4o-mini annotator** (robustness ablation) | add `"locle_cache_filename": "<dataset>_gpt-4o-mini.json"` |
| **Cluster count `K`** (sensitivity ablation) | `cluster_id_path`: `cluster_assignments_graphmae2.npy` (2C) → `cluster_K{1.0,1.5,3.0}.npy` |

Per-dataset `T_c` / cluster paths:

| Dataset | `cluster_T_c_path` | `cluster_id_path` |
|---|---|---|
| `cora` | `data/tc/cora/T_c_lfv3_gmae.npy` | `data/tc/cora/cluster_assignments_graphmae2.npy` |
| `citeseer` | `data/tc/citeseer/T_c_lfv3_gmae_g35.npy` | `data/tc/citeseer/cluster_assignments_graphmae2.npy` |
| `pubmed` | `data/tc/pubmed/T_c_lfv3_gmae_g35.npy` | `data/tc/pubmed/cluster_assignments_graphmae2.npy` |
| `wikics` | `data/tc/wikics/T_c_lfv3_gmae_g35.npy` | `data/tc/wikics/cluster_assignments_graphmae2.npy` |
| `dblp` | `data/tc/dblp/T_c_lfv3_gmae.npy` | `data/tc/dblp/cluster_assignments_graphmae2.npy` |

The **main table** is 5 datasets × 2 backbones × 5 seeds (budget `B = 50C` via
`budget_per_class = 50`); report the 5-seed mean. The `run.sbatch`
(`CFG=<config name>`) and `run_all.sh` (sweep over `configs/wbXmod_*.json`) SLURM
helpers assume one config file per cell exists locally.

> **Reproducibility.** GNN training uses non-deterministic CUDA kernels, so
> single-run numbers vary slightly across hardware. Set `"deterministic": true`
> for bit-reproducible runs on fixed hardware; report the 5-seed mean.
