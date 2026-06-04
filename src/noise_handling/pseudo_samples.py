"""
Pseudo-sample construction for noise probing (Stage 0).

Adapted from DMA (baselines/dma/data/gen_pseudo_sample_*.py).

DMA's approach: prompt an LLM to generate synthetic class-representative text,
then query the *same* LLM on that text to build a confusion matrix.  Our
adaptation provides two modes:

  Option A (no annotations yet):
    Build template-based pseudo-samples from class names + domain descriptions.
    Cheaper than DMA's generative approach — no extra LLM calls needed.

  Option B (after initial round):
    Sample real node texts from high-confidence LLM annotations.
    More realistic probes because they come from the actual data distribution.
"""

import logging
import random
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Domain descriptions per dataset — adapted from DMA's Cora script
# ──────────────────────────────────────────────────────────────

_DOMAIN_CONTEXT = {
    "cora": "arXiv computer science subcategories",
    "citeseer": "computer science research themes",
    "pubmed": "diabetes research subcategories",
    "wikics": "Wikipedia Computer Science entity subcategories",
    "dblp": "computer science research areas",
}

# Per-class descriptions modeled after DMA's category_descriptions.
# Used in Option A to generate template pseudo-samples without LLM calls.
_CLASS_DESCRIPTIONS = {
    "cora": {
        "Rule_Learning": (
            "Rule learning refers to automatically extracting useful rules or "
            "patterns from data. Rule learning algorithms discover interpretable "
            "rules capturing relationships between features. Applications include "
            "classification, association mining, and knowledge discovery."
        ),
        "Neural_Networks": (
            "Neural networks consist of interconnected nodes organized in layers. "
            "Research covers architectures such as CNNs, RNNs, transformers, and "
            "graph neural networks applied to vision, NLP, and other domains."
        ),
        "Case_Based": (
            "Case-based reasoning enables agents to solve new problems by "
            "retrieving and reusing solutions from similar past cases. Techniques "
            "include similarity assessment, case adaptation, and knowledge "
            "representation for domains like diagnosis and recommendation."
        ),
        "Genetic_Algorithms": (
            "Genetic algorithms are optimization techniques inspired by natural "
            "selection. A population of candidate solutions evolves through "
            "selection, crossover, and mutation over successive generations to "
            "find high-quality solutions to complex optimization problems."
        ),
        "Theory": (
            "Theory of computation covers mathematical frameworks, algorithm "
            "analysis, complexity theory, cryptography, and formal methods. "
            "Research focuses on proving theorems and exploring the limits of "
            "efficient computation."
        ),
        "Reinforcement_Learning": (
            "Reinforcement learning teaches agents to make sequential decisions "
            "by interacting with environments to maximize cumulative rewards. "
            "Applications span robotics, game playing, control systems, and "
            "autonomous decision-making."
        ),
        "Probabilistic_Methods": (
            "Probabilistic methods leverage probability theory to model "
            "uncertainty. Techniques include Bayesian inference, probabilistic "
            "graphical models, and probabilistic programming for classification, "
            "regression, and decision-making under uncertainty."
        ),
    },
    "citeseer": {
        "Agents": (
            "Research on intelligent software agents, multi-agent systems, "
            "agent communication, cooperation, negotiation, and distributed "
            "problem solving in autonomous environments."
        ),
        "ML": (
            "Machine learning research covering supervised, unsupervised, and "
            "semi-supervised learning algorithms, model selection, feature "
            "engineering, and statistical learning theory."
        ),
        "IR": (
            "Information retrieval research on document indexing, search "
            "engines, query processing, relevance ranking, text mining, "
            "and retrieval evaluation methodologies."
        ),
        "DB": (
            "Database research including query optimization, data modeling, "
            "transaction management, distributed databases, data warehousing, "
            "and database management systems."
        ),
        "HCI": (
            "Human-computer interaction research on user interface design, "
            "usability evaluation, interaction techniques, visualization, "
            "and user experience studies."
        ),
        "AI": (
            "Artificial intelligence research on knowledge representation, "
            "reasoning, planning, natural language understanding, expert "
            "systems, and cognitive modeling."
        ),
    },
    "pubmed": {
        "Diabetes Mellitus, Experimental": (
            "Experimental studies on diabetes mellitus using animal models, "
            "cell cultures, and laboratory techniques to investigate disease "
            "mechanisms, drug effects, and potential therapeutic interventions."
        ),
        "Diabetes Mellitus Type 1": (
            "Research on type 1 diabetes mellitus, an autoimmune condition "
            "where the immune system destroys insulin-producing beta cells. "
            "Studies cover pathogenesis, insulin therapy, immunological "
            "mechanisms, and clinical management."
        ),
        "Diabetes Mellitus Type 2": (
            "Research on type 2 diabetes mellitus characterized by insulin "
            "resistance and relative insulin deficiency. Studies cover risk "
            "factors, metabolic syndrome, oral medications, lifestyle "
            "interventions, and epidemiology."
        ),
    },
    "wikics": {
        "Computational linguistics": (
            "Computational linguistics applies computational methods to "
            "natural language processing, including parsing, machine "
            "translation, speech recognition, and text analysis."
        ),
        "Databases": (
            "Database systems for storing, managing, and querying structured "
            "data. Topics include relational models, SQL, NoSQL, indexing, "
            "query optimization, and data warehousing."
        ),
        "Operating systems": (
            "Operating systems manage hardware resources and provide services "
            "for applications. Topics include process scheduling, memory "
            "management, file systems, and kernel design."
        ),
        "Computer architecture": (
            "Computer architecture covers the design of processors, memory "
            "hierarchies, instruction sets, pipelining, parallelism, and "
            "hardware-software interfaces."
        ),
        "Computer security": (
            "Computer security encompasses cryptography, access control, "
            "network security, intrusion detection, malware analysis, and "
            "secure software development."
        ),
        "Internet protocols": (
            "Internet protocols define rules for data transmission including "
            "TCP/IP, HTTP, DNS, routing protocols, and network layer "
            "communication standards."
        ),
        "Computer file systems": (
            "File systems organize and manage data storage on disk. Topics "
            "include file allocation, journaling, distributed file systems, "
            "and storage management."
        ),
        "Distributed computing architecture": (
            "Distributed computing architectures coordinate multiple machines "
            "for parallel processing. Topics include cloud computing, "
            "MapReduce, consensus protocols, and fault tolerance."
        ),
        "Web technology": (
            "Web technologies for building internet applications including "
            "HTML, CSS, JavaScript, web frameworks, REST APIs, and content "
            "management systems."
        ),
        "Programming language topics": (
            "Programming language research covers language design, type "
            "systems, compilers, interpreters, formal semantics, and "
            "programming paradigms."
        ),
    },
    "dblp": {
        "Database": (
            "Database research on data management, query processing, "
            "indexing, transaction processing, and database systems."
        ),
        "Data Mining": (
            "Data mining research on pattern discovery, clustering, "
            "classification, association rules, anomaly detection, and "
            "knowledge extraction from large datasets."
        ),
        "AI": (
            "Artificial intelligence research on machine learning, neural "
            "networks, reasoning, planning, computer vision, and natural "
            "language processing."
        ),
        "Information Retrieval": (
            "Information retrieval research on search, ranking, text "
            "indexing, relevance feedback, and evaluation of retrieval "
            "systems."
        ),
    },
}


# ──────────────────────────────────────────────────────────────
# Template builder (Option A)
# ──────────────────────────────────────────────────────────────

_TEMPLATE_PAPER = (
    "This paper presents research in a specific area of computer science. "
    "{description} "
    "We propose novel approaches and evaluate them on standard benchmarks, "
    "demonstrating improvements over existing methods in this field."
)

_TEMPLATE_ENTITY = (
    "This is a topic in computer science. "
    "{description}"
)


def _build_template_samples(
    class_names: List[str],
    dataset_name: str,
    n_per_class: int,
) -> Dict[int, List[str]]:
    """
    Build pseudo-samples from templates + class descriptions.

    DMA generates these via LLM; we use deterministic templates instead
    to avoid spending probing budget on generation.  The descriptions are
    adapted from DMA's category_descriptions (Cora) and extended to all
    5 datasets.

    Each class gets `n_per_class` slightly varied samples built by
    combining the class description with different template phrasings.
    """
    dataset_key = dataset_name.lower()
    descriptions = _CLASS_DESCRIPTIONS.get(dataset_key, {})
    is_entity = dataset_key == "wikics"

    # Variation templates — NO class names, only descriptions.
    # Forces the LLM to infer the class from semantic content.
    _VARIATIONS = [
        "This paper presents the following research. {description}",
        "We study the following topic in computer science. {description}",
        "{description} This area remains an active research direction.",
        "In this study we address the following problem. {description} "
        "Our results contribute to the growing body of work in this domain.",
        "{description} We propose novel approaches and evaluate on standard benchmarks.",
    ]

    if is_entity:
        _VARIATIONS = [
            "This is a subfield of computer science. {description}",
            "This article discusses a topic in computing. {description}",
            "{description}",
            "An overview of a computing topic. {description} "
            "It is an important area of computer science.",
            "This entity relates to a computing discipline. {description}",
        ]

    samples = {}  # type: Dict[int, List[str]]

    for class_id, class_name in enumerate(class_names):
        desc = descriptions.get(class_name, "")
        if not desc:
            # Fallback: use the human-readable form of the class name as
            # a topic description, but avoid literally naming the category.
            readable = class_name.lower().replace("_", " ")
            desc = (
                f"This research area in computer science studies problems "
                f"and methods related to {readable}."
            )

        class_samples = []
        base = _TEMPLATE_ENTITY if is_entity else _TEMPLATE_PAPER
        class_samples.append(base.format(description=desc))

        for i in range(n_per_class - 1):
            tmpl = _VARIATIONS[i % len(_VARIATIONS)]
            class_samples.append(tmpl.format(description=desc))

        samples[class_id] = class_samples

    return samples


# ──────────────────────────────────────────────────────────────
# Adversarial cross-class probes
# ──────────────────────────────────────────────────────────────


def _build_adversarial_samples(
    class_names: List[str],
    dataset_name: str,
    n_per_class: int,
    seed: int = 42,
) -> Dict[int, List[str]]:
    """
    Build adversarial probes that blend descriptions from two classes.

    For each class i, constructs probes by mixing 70% of class i's
    description with 30% of a confuser class j's description.  The
    true label is i (the majority class).  If the LLM predicts j,
    that reveals a genuine confusion boundary.

    These probes are harder than pure description-only templates and
    specifically target the off-diagonal entries of the noise matrix.
    """
    rng = random.Random(seed)
    dataset_key = dataset_name.lower()
    descriptions = _CLASS_DESCRIPTIONS.get(dataset_key, {})

    num_classes = len(class_names)
    samples: Dict[int, List[str]] = {}

    for class_id, class_name in enumerate(class_names):
        desc_i = descriptions.get(class_name, "")
        if not desc_i:
            readable = class_name.lower().replace("_", " ")
            desc_i = f"This area studies problems and methods related to {readable}."

        # Split description into sentences for mixing
        sentences_i = [s.strip() for s in desc_i.split(". ") if s.strip()]

        class_samples = []
        # Pick confuser classes — all other classes, shuffled
        other_classes = [c for c in range(num_classes) if c != class_id]
        rng.shuffle(other_classes)

        for idx in range(min(n_per_class, len(other_classes))):
            confuser_id = other_classes[idx]
            confuser_name = class_names[confuser_id]
            desc_j = descriptions.get(confuser_name, "")
            if not desc_j:
                readable_j = confuser_name.lower().replace("_", " ")
                desc_j = f"This area studies problems and methods related to {readable_j}."

            sentences_j = [s.strip() for s in desc_j.split(". ") if s.strip()]

            # Take ~70% of sentences from true class, ~30% from confuser
            n_from_i = max(1, int(len(sentences_i) * 0.7))
            n_from_j = max(1, min(len(sentences_j), len(sentences_i) - n_from_i + 1))

            picked_i = sentences_i[:n_from_i]
            picked_j = rng.sample(sentences_j, min(n_from_j, len(sentences_j)))

            # Interleave: start with true class content, insert confuser
            mixed = list(picked_i)
            insert_pos = max(1, len(mixed) // 2)
            for s in picked_j:
                mixed.insert(insert_pos, s)
                insert_pos += 2

            blended_text = (
                "This paper presents research in computer science. "
                + ". ".join(mixed) + "."
            )
            class_samples.append(blended_text)

        samples[class_id] = class_samples

    return samples


# ──────────────────────────────────────────────────────────────
# Real-text sampler (Option B)
# ──────────────────────────────────────────────────────────────

def _build_annotation_samples(
    class_names: List[str],
    annotations: Dict[int, Tuple[int, float]],
    raw_texts: List[str],
    n_per_class: int,
    confidence_threshold: float = 80.0,
    seed: int = 42,
) -> Dict[int, List[str]]:
    """
    Build pseudo-samples from real node texts of high-confidence annotations.

    After an initial annotation round, we have LLM labels with confidence
    scores.  We pick nodes where the LLM was highly confident (>threshold)
    as pseudo ground-truth for noise probing.  This gives more realistic
    probes than templates because the text comes from the actual data
    distribution.

    If a class has fewer than n_per_class high-confidence nodes, we relax
    the threshold progressively and finally fall back to templates.
    """
    rng = random.Random(seed)

    # Group nodes by their annotated class
    class_nodes = {c: [] for c in range(len(class_names))}  # type: Dict[int, List[Tuple[int, float]]]
    for node_id, (label, conf) in annotations.items():
        if 0 <= label < len(class_names):
            class_nodes[label].append((node_id, conf))

    samples = {}  # type: Dict[int, List[str]]

    for class_id in range(len(class_names)):
        nodes = class_nodes[class_id]

        # Sort by confidence descending
        nodes.sort(key=lambda x: -x[1])

        # Filter by threshold, relaxing if needed
        selected_texts = []
        for threshold in [confidence_threshold, confidence_threshold * 0.75, 0.0]:
            candidates = [(nid, conf) for nid, conf in nodes if conf >= threshold]
            if len(candidates) >= n_per_class:
                # Sample diverse subset: pick spread across the list
                if len(candidates) > n_per_class:
                    chosen = rng.sample(candidates, n_per_class)
                else:
                    chosen = candidates
                selected_texts = [raw_texts[nid] for nid, _ in chosen]
                break
            elif candidates:
                selected_texts = [raw_texts[nid] for nid, _ in candidates]
                break

        if not selected_texts:
            # No annotations for this class at all — fall back to template
            logger.warning(
                "Class %d (%s): no annotated nodes found, using template fallback",
                class_id,
                class_names[class_id],
            )
            template_samples = _build_template_samples(
                class_names, "cora", n_per_class
            )
            selected_texts = template_samples.get(class_id, ["No text available."])

        samples[class_id] = selected_texts

    return samples


# ──────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────

class PseudoSampleConstructor:
    """
    Constructs pseudo-samples for noise probing (Stage 0 of NoisePref).

    These samples have *known* (pseudo) ground-truth class labels.  By
    querying the LLM on them, we can estimate the C×C confusion matrix
    (noise_matrix) without needing any real ground-truth labels.

    Adapted from DMA (baselines/dma/data/gen_pseudo_sample_*.py), which
    uses an LLM to generate class-representative abstracts.  We provide
    two cheaper options:
      - Option A: template-based (deterministic, no LLM cost)
      - Option B: sample from high-confidence annotations (after round 1)

    Usage:
        constructor = PseudoSampleConstructor(dataset_name="cora",
                                              class_names=data.class_names)
        # Option A — before any annotations
        samples = constructor.construct(data)
        # Option B — after initial annotation round
        samples = constructor.construct(data, initial_annotations=annotations)
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name.lower() if dataset_name else None
        self.class_names = class_names
        self.num_classes = len(class_names) if class_names else None
        self.seed = seed

    def construct(
        self,
        dataset,  # PyG Data object with .raw_texts
        initial_annotations: Optional[Dict[int, Tuple[int, float]]] = None,
        n_per_class: int = 50,
        confidence_threshold: float = 80.0,
    ) -> Dict[int, List[str]]:
        """
        Construct pseudo-samples for each class.

        For each class c in the dataset:
          Option A (no annotations yet): Use class name + generic description.
            Template-based construction adapted from DMA's category descriptions.
          Option B (after initial round): Sample node texts from nodes annotated
            as class c by the LLM with high confidence (>threshold).  Pick a
            diverse subset.

        Args:
            dataset:              PyG Data object (needs .raw_texts attribute).
            initial_annotations:  Optional dict {node_id: (label, confidence)}
                                  from a prior annotation round.  If provided
                                  and sufficient high-confidence nodes exist,
                                  Option B is used.
            n_per_class:          Number of pseudo-samples to produce per class.
            confidence_threshold: Minimum LLM confidence (0-100) for Option B.

        Returns:
            Dict mapping class_id -> list of pseudo text samples.
            Each list has up to n_per_class entries.
        """
        # Infer dataset_name / class_names from the data object if not set
        dataset_name = self.dataset_name
        class_names = self.class_names
        num_classes = self.num_classes

        if dataset_name is None:
            dataset_name = getattr(dataset, "dataset_name", "cora").lower()
            self.dataset_name = dataset_name
        if class_names is None:
            class_names = getattr(dataset, "class_names", None)
            if class_names is None:
                raise ValueError(
                    "class_names must be provided either at init or via dataset.class_names"
                )
            self.class_names = class_names
            num_classes = len(class_names)
            self.num_classes = num_classes

        use_annotations = False

        if initial_annotations is not None:
            # Check if we have enough annotations to use Option B
            class_counts = [0] * self.num_classes
            for _, (label, conf) in initial_annotations.items():
                if 0 <= label < self.num_classes and conf >= confidence_threshold:
                    class_counts[label] += 1

            min_needed = max(1, n_per_class // 5)  # need at least 20% coverage
            covered = sum(1 for c in class_counts if c >= min_needed)

            if covered >= self.num_classes // 2:
                use_annotations = True
                logger.info(
                    "Option B: using annotation-based pseudo-samples "
                    "(%d/%d classes have >= %d high-confidence nodes)",
                    covered,
                    self.num_classes,
                    min_needed,
                )
            else:
                logger.info(
                    "Option B insufficient coverage (%d/%d classes). "
                    "Falling back to Option A (templates).",
                    covered,
                    self.num_classes,
                )

        if use_annotations:
            samples = _build_annotation_samples(
                class_names=self.class_names,
                annotations=initial_annotations,
                raw_texts=dataset.raw_texts,
                n_per_class=n_per_class,
                confidence_threshold=confidence_threshold,
                seed=self.seed,
            )
        else:
            # Split budget: half description-only, half adversarial cross-class
            n_desc = max(1, n_per_class // 2)
            n_adv = max(1, n_per_class - n_desc)

            samples = _build_template_samples(
                class_names=self.class_names,
                dataset_name=self.dataset_name,
                n_per_class=n_desc,
            )

            # Add adversarial cross-class probes
            adv_samples = _build_adversarial_samples(
                class_names=self.class_names,
                dataset_name=self.dataset_name,
                n_per_class=n_adv,
                seed=self.seed,
            )
            for cid in samples:
                if cid in adv_samples:
                    samples[cid].extend(adv_samples[cid])

        total = sum(len(v) for v in samples.values())
        logger.info(
            "Constructed %d pseudo-samples across %d classes (%s)",
            total,
            len(samples),
            "annotation-based" if use_annotations else "template-based",
        )

        return samples

    def get_probing_prompt(self, text: str) -> str:
        """
        Build the probing prompt for a single pseudo-sample.

        This is the prompt sent to the LLM to classify a pseudo-sample
        whose true class we already know.  The response lets us populate
        one entry of the confusion matrix.

        Uses the same prompt format as the annotator to ensure the
        confusion matrix reflects real annotation behavior.
        """
        domain = _DOMAIN_CONTEXT.get(self.dataset_name, "research categories")
        class_list = ", ".join(self.class_names)
        return (
            f"Text: {text}\n\n"
            f"Which of the following {domain} does this text belong to? "
            f"Here are the {self.num_classes} categories: {class_list}. "
            f"Reply with a JSON object: {{\"label\": \"<category>\", \"confidence\": <0-100>}}. "
            f"Only use category names from the list above."
        )

    def __repr__(self) -> str:
        return (
            f"PseudoSampleConstructor(dataset={self.dataset_name!r}, "
            f"num_classes={self.num_classes})"
        )


# ──────────────────────────────────────────────────────────────
# Self-test on Cora
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import os.path as osp
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)

    from src.data_loading.tag_dataset import load_dataset

    data_dir = osp.join(repo_root, "data")
    locle_data_dir = osp.join(repo_root, "baselines", "locle", "data")
    data_path = data_dir if osp.exists(osp.join(data_dir, "cora_random_sbert.pt")) else locle_data_dir

    data = load_dataset("cora", data_path=data_path)
    print(f"Loaded Cora: {data.x.size(0)} nodes, {data.num_classes} classes")
    print(f"Classes: {data.class_names}\n")

    constructor = PseudoSampleConstructor(
        dataset_name="cora",
        class_names=data.class_names,
        seed=42,
    )

    # --- Test 1: Option A (template-based) ---
    print("=" * 60)
    print("[Test 1] Option A — Template-based pseudo-samples")
    print("=" * 60)
    samples_a = constructor.construct(data, n_per_class=5)
    assert len(samples_a) == data.num_classes
    for cid, texts in samples_a.items():
        print(f"\n  Class {cid} ({data.class_names[cid]}): {len(texts)} samples")
        print(f"    [0]: {texts[0][:100]}...")
        assert len(texts) == 5
    print("\n  Option A: PASS")

    # --- Test 2: Option B (annotation-based) ---
    print("\n" + "=" * 60)
    print("[Test 2] Option B — Annotation-based pseudo-samples")
    print("=" * 60)

    # Simulate annotations using ground truth (for testing only)
    fake_annotations = {}
    for i in range(min(500, data.x.size(0))):
        gt_label = data.y[i].item()
        # Simulate high-confidence correct annotations
        fake_annotations[i] = (gt_label, 90.0)

    samples_b = constructor.construct(
        data,
        initial_annotations=fake_annotations,
        n_per_class=10,
        confidence_threshold=80.0,
    )
    assert len(samples_b) == data.num_classes
    for cid, texts in samples_b.items():
        print(f"\n  Class {cid} ({data.class_names[cid]}): {len(texts)} samples")
        print(f"    [0]: {texts[0][:100]}...")
    print("\n  Option B: PASS")

    # --- Test 3: Option B fallback when annotations are sparse ---
    print("\n" + "=" * 60)
    print("[Test 3] Option B fallback — sparse annotations")
    print("=" * 60)
    sparse_annotations = {0: (0, 95.0), 1: (0, 85.0)}
    samples_sparse = constructor.construct(
        data,
        initial_annotations=sparse_annotations,
        n_per_class=5,
    )
    assert len(samples_sparse) == data.num_classes
    print("  Sparse annotations fall back to templates: PASS")

    # --- Test 4: Probing prompt format ---
    print("\n" + "=" * 60)
    print("[Test 4] Probing prompt")
    print("=" * 60)
    prompt = constructor.get_probing_prompt("This paper studies neural networks.")
    print(f"  {prompt[:200]}...")
    assert "categories" in prompt
    assert "JSON" in prompt
    print("  Probing prompt: PASS")

    # --- Test 5: All datasets ---
    print("\n" + "=" * 60)
    print("[Test 5] All datasets (template mode)")
    print("=" * 60)
    for ds_name in ["cora", "citeseer", "pubmed", "wikics", "dblp"]:
        from src.data_loading.tag_dataset import get_class_names
        cnames = get_class_names(ds_name)
        c = PseudoSampleConstructor(dataset_name=ds_name, class_names=cnames)
        # Use a minimal mock dataset with raw_texts
        class MockData:
            raw_texts = ["placeholder"] * 100
        s = c.construct(MockData(), n_per_class=3)
        assert len(s) == len(cnames), f"{ds_name}: expected {len(cnames)} classes, got {len(s)}"
        total = sum(len(v) for v in s.values())
        print(f"  {ds_name:10s}: {len(cnames)} classes, {total} samples")
    print("  All datasets: PASS")

    print(f"\nrepr: {constructor}")
    print("\nAll pseudo-sample tests complete.")
