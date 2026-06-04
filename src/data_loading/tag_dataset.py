"""
Unified dataset loader for text-attributed graphs (TAGs).

Loads datasets in LoCLE's format (.pt files with SentenceBERT features).
Supported: Cora, CiteSeer, PubMed, WikiCS, DBLP.
"""

import os
import os.path as osp
import pickle

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# ──────────────────────────────────────────────────────────────
# Class name registry — human-readable labels per dataset
# ──────────────────────────────────────────────────────────────

CLASS_NAMES = {
    "cora": [
        "Rule_Learning",
        "Neural_Networks",
        "Case_Based",
        "Genetic_Algorithms",
        "Theory",
        "Reinforcement_Learning",
        "Probabilistic_Methods",
    ],
    "citeseer": [
        "Agents",
        "ML",
        "IR",
        "DB",
        "HCI",
        "AI",
    ],
    "pubmed": [
        "Diabetes Mellitus, Experimental",
        "Diabetes Mellitus Type 1",
        "Diabetes Mellitus Type 2",
    ],
    "wikics": [
        "Computational linguistics",
        "Databases",
        "Operating systems",
        "Computer architecture",
        "Computer security",
        "Internet protocols",
        "Computer file systems",
        "Distributed computing architecture",
        "Web technology",
        "Programming language topics",
    ],
    "dblp": [
        "Database",
        "Data Mining",
        "AI",
        "Information Retrieval",
    ],
    "arxiv": [  # OGB label order (NOT alphabetical!)
        "cs.NA", "cs.MM", "cs.LO", "cs.CY", "cs.CR", "cs.DC", "cs.HC", "cs.CE",
        "cs.NI", "cs.CC", "cs.AI", "cs.MA", "cs.GL", "cs.NE", "cs.SC", "cs.AR",
        "cs.CV", "cs.GR", "cs.ET", "cs.SY", "cs.CG", "cs.OH", "cs.PL", "cs.SE",
        "cs.LG", "cs.SD", "cs.SI", "cs.RO", "cs.IT", "cs.PF", "cs.CL", "cs.IR",
        "cs.MS", "cs.FL", "cs.DS", "cs.OS", "cs.GT", "cs.DB", "cs.DL", "cs.DM",
    ],
}


def get_class_names(dataset_name: str) -> list[str]:
    """Return human-readable class labels for a dataset."""
    key = dataset_name.lower()
    if key == "citeseer2":
        key = "citeseer"
    if key in CLASS_NAMES:
        return CLASS_NAMES[key]
    # For datasets not in the registry, try loading from OGB mapping
    if key == "products":
        return _load_products_class_names()
    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        f"Available: {list(CLASS_NAMES.keys())}"
    )


def _load_products_class_names():
    """Load product category names from OGB mapping file."""
    import gzip, csv
    mapping_path = osp.join(
        osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))),
        "data", "ogb", "ogbn_products", "mapping", "labelidx2productcategory.csv.gz"
    )
    if osp.exists(mapping_path):
        cats = []
        with gzip.open(mapping_path, 'rt') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                cats.append(row[1].strip())
        return cats
    # Fallback: generic category names
    return [f"Category_{i}" for i in range(47)]


# ──────────────────────────────────────────────────────────────
# Dataset file name mapping (matches LoCLE's naming convention)
# ──────────────────────────────────────────────────────────────

# LoCLE stores citeseer as "citeseer2", wikics as fixed split
_PT_FILE_MAP = {
    "cora": "cora_random_sbert.pt",
    "citeseer": "citeseer2_random_sbert.pt",
    "pubmed": "pubmed_random_sbert.pt",
    "wikics": "wikics_fixed_sbert.pt",
    "arxiv": "arxiv_fixed_sbert.pt",
    # products_fixed_sbert.pt carries raw_texts + label_names (47 real Amazon
    # categories); products_sbert.pt is embeddings-only. Prefer the _fixed file.
    "products": "products_fixed_sbert.pt",
}


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows (same as LoCLE's default preprocessing)."""
    norms = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return x / norms


def _load_standard_dataset(dataset_name: str, data_path: str) -> Data:
    """Load Cora / CiteSeer / PubMed / WikiCS from LoCLE .pt files."""
    key = dataset_name.lower()
    if key == "citeseer2":
        key = "citeseer"

    filename = _PT_FILE_MAP.get(key)
    if filename is None:
        raise ValueError(
            f"No .pt mapping for '{dataset_name}'. "
            f"Available: {list(_PT_FILE_MAP.keys())}"
        )

    pt_path = osp.join(data_path, filename)
    if not osp.exists(pt_path):
        raise FileNotFoundError(f"Data file not found: {pt_path}")

    raw_data = torch.load(pt_path, map_location="cpu", weights_only=False)

    class_names = (
        raw_data.label_names
        if hasattr(raw_data, "label_names")
        else get_class_names(key)
    )

    y = raw_data.y
    if y.dim() != 1:
        y = y.view(-1)

    # Make edge_index undirected if needed
    edge_index = raw_data.edge_index
    if hasattr(raw_data, 'is_undirected') and not raw_data.is_undirected():
        edge_index = to_undirected(edge_index)

    data = Data(
        x=_normalize(raw_data.x),
        edge_index=edge_index,
        y=y,
    )
    # raw_texts: use actual texts if available, else dummy for oracle mode
    if hasattr(raw_data, 'raw_texts') and raw_data.raw_texts is not None:
        data.raw_texts = raw_data.raw_texts
    else:
        # For datasets without text (e.g., ogbn-arxiv), generate placeholders
        data.raw_texts = [f"node_{i}" for i in range(data.x.size(0))]
    data.num_classes = int(y.max().item()) + 1
    data.class_names = class_names
    data.dataset_name = key

    return data


def _load_dblp(data_path: str) -> Data:
    """Load DBLP from TSGFM-format processed files (matches LoCLE's load_one_tag_dataset)."""
    dblp_base = osp.join(data_path, "dblp")

    # Geometric data
    geo_path = osp.join(dblp_base, "processed", "geometric_data_processed.pt")
    if not osp.exists(geo_path):
        raise FileNotFoundError(f"DBLP geometric data not found: {geo_path}")
    raw_data = torch.load(geo_path, map_location="cpu", weights_only=False)[0]

    # Raw texts
    texts_path = osp.join(dblp_base, "dblp", "raw_texts.pkl")
    if not osp.exists(texts_path):
        raise FileNotFoundError(f"DBLP raw texts not found: {texts_path}")
    with open(texts_path, "rb") as f:
        raw_texts = pickle.load(f)

    y = raw_data.y.view(-1)
    x = raw_data.node_text_feat if hasattr(raw_data, "node_text_feat") else raw_data.x
    edge_index = to_undirected(raw_data.edge_index)

    class_names = get_class_names("dblp")

    data = Data(
        x=_normalize(x),
        edge_index=edge_index,
        y=y,
    )
    data.raw_texts = raw_texts
    data.num_classes = int(y.max().item()) + 1
    data.class_names = class_names
    data.dataset_name = "dblp"

    return data


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def load_dataset(
    dataset_name: str,
    data_path: str = "data/",
) -> Data:
    """
    Load a text-attributed graph dataset in LoCLE format.

    Args:
        dataset_name: One of 'cora', 'citeseer', 'pubmed', 'wikics', 'dblp'.
        data_path:    Root data directory (contains .pt files and dblp/ subfolder).

    Returns:
        PyG Data object with attributes:
            x           — (num_nodes, feat_dim) SentenceBERT features, L2-normalized
            edge_index  — (2, num_edges) sparse edge list
            y           — (num_nodes,) ground-truth labels
            raw_texts   — list[str] of length num_nodes
            num_classes  — int
            class_names  — list[str] of length num_classes
            dataset_name — str
    """
    key = dataset_name.lower()
    if key == "citeseer2":
        key = "citeseer"

    if key == "dblp":
        return _load_dblp(data_path)

    return _load_standard_dataset(key, data_path)


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Resolve data path relative to repo root
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    # Try our data/ first, fall back to baselines/locle/data/
    data_dir = osp.join(repo_root, "data")
    locle_data_dir = osp.join(repo_root, "baselines", "locle", "data")

    for name in ["cora", "citeseer", "pubmed", "wikics", "dblp"]:
        # Pick whichever data directory has the files
        if name == "dblp":
            path = data_dir if osp.exists(osp.join(data_dir, "dblp")) else locle_data_dir
        else:
            pt_file = _PT_FILE_MAP[name]
            path = data_dir if osp.exists(osp.join(data_dir, pt_file)) else locle_data_dir

        try:
            d = load_dataset(name, data_path=path)
            num_edges = d.edge_index.size(1)
            print(
                f"{name:10s} | nodes={d.x.size(0):6d} | "
                f"feat_dim={d.x.size(1):4d} | edges={num_edges:7d} | "
                f"classes={d.num_classes} | class_names={d.class_names}"
            )
            print(f"{'':10s} | raw_texts[0][:80] = {d.raw_texts[0][:80]}")
            print()
        except FileNotFoundError as e:
            print(f"{name:10s} | SKIPPED — {e}")
            print()
