"""
LLM noise profiler — estimates the C×C confusion matrix via pseudo-sample probing.

Adapted from DMA (baselines/dma/data/gen_llm_sim.py).  DMA's original approach
embeds class-representative texts with a local LLM and computes cosine similarity
between embeddings.  Our adaptation directly queries the LLM annotator on
pseudo-samples whose true class is known, giving an empirical confusion matrix
that reflects actual annotation behavior (not just embedding proximity).

Pipeline position: Stage 0 — Noise Probing (~10% of LLM budget).

Usage:
    profiler = LLMNoiseProfiler(annotator)
    noise_matrix, per_class_acc, confused_pairs = profiler.estimate_noise_matrix(
        dataset_name="cora",
        class_names=data.class_names,
        pseudo_samples=samples,        # from PseudoSampleConstructor
        n_probes_per_class=20,
    )
    weights, pair_weights = profiler.get_noise_weights(noise_matrix)
"""

import json
import logging
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.llm_annotation.annotator import LLMAnnotator

logger = logging.getLogger(__name__)


class LLMNoiseProfiler:
    """
    Estimates LLM annotation noise by probing with pseudo-samples of known class.

    For each class c, sends pseudo_samples[c] to the LLM for classification.
    Since the true class is known, we can directly measure:
        noise_matrix[i][j] = P(LLM predicts j | true class is i)

    The diagonal gives per-class accuracy; off-diagonal entries reveal which
    class pairs the LLM systematically confuses.

    Args:
        annotator: LLMAnnotator instance (handles API calls + response parsing).
        cache_dir: Directory for caching noise matrices to disk.
    """

    def __init__(
        self,
        annotator: LLMAnnotator,
        cache_dir: str = "data/annotations/",
    ):
        self.annotator = annotator
        self.cache_dir = cache_dir

    # ── Main estimation ──────────────────────────────────────

    def estimate_noise_matrix(
        self,
        dataset_name: str,
        class_names: List[str],
        pseudo_samples: Dict[int, List[str]],
        n_probes_per_class: int = 50,
        budget_manager=None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, float]]]:
        """
        Estimate the C×C noise (confusion) matrix by probing the LLM.

        For each class c, sends up to n_probes_per_class pseudo-samples
        (whose true label is c) to the LLM.  Records the LLM's predicted
        class for each probe to build an empirical confusion matrix.

        Args:
            dataset_name:       Dataset identifier (for caching).
            class_names:        List of class name strings.
            pseudo_samples:     {class_id: [text, ...]} from PseudoSampleConstructor.
            n_probes_per_class: Max probes per class (actual count may be less
                                if pseudo_samples[c] has fewer entries or budget
                                is exhausted).
            budget_manager:     Optional NoiseBudgetManager for budget tracking.
            use_cache:          If True, load cached matrix when available.

        Returns:
            noise_matrix:      C×C numpy array.  noise_matrix[i][j] = P(predict j | true i).
                               Rows sum to 1 (or 0 if no probes were sent for that class).
            per_class_accuracy: 1D array of length C.  Diagonal of noise_matrix.
            confused_pairs:    List of (i, j, rate) for i != j, sorted descending by rate.

        Budget cost: up to n_probes_per_class * num_classes LLM queries.
        """
        num_classes = len(class_names)

        # Try loading from cache
        cache_path = self._cache_path(dataset_name)
        if use_cache and osp.exists(cache_path):
            logger.info("Loading cached noise matrix from %s", cache_path)
            noise_matrix = self._load_cache(cache_path)
            if noise_matrix.shape == (num_classes, num_classes):
                per_class_accuracy = np.diag(noise_matrix)
                confused_pairs = self._extract_confused_pairs(noise_matrix)
                return noise_matrix, per_class_accuracy, confused_pairs
            else:
                logger.warning(
                    "Cached matrix shape %s != expected (%d, %d), re-estimating",
                    noise_matrix.shape, num_classes, num_classes,
                )

        # Build raw count matrix
        counts = np.zeros((num_classes, num_classes), dtype=np.float64)
        total_probes = 0
        total_failed = 0

        # Pre-allocate probes per class to guarantee every class is probed.
        # Without this, sequential iteration exhausts the budget on early classes
        # and leaves later classes with zero probes (defaulting to ~1/C accuracy).
        if budget_manager is not None:
            total_probing_budget = budget_manager.probing_remaining
            probes_per_class = max(2, total_probing_budget // num_classes)
        else:
            probes_per_class = n_probes_per_class

        for true_class in range(num_classes):
            texts = pseudo_samples.get(true_class, [])
            if not texts:
                logger.warning(
                    "No pseudo-samples for class %d (%s), skipping",
                    true_class, class_names[true_class],
                )
                continue

            n_probes = min(probes_per_class, n_probes_per_class, len(texts))

            # Budget check
            if budget_manager is not None:
                available = budget_manager.probing_remaining
                if available <= 0:
                    logger.warning(
                        "Probing budget exhausted at class %d/%d",
                        true_class, num_classes,
                    )
                    break
                n_probes = min(n_probes, available)

            probing_texts = texts[:n_probes]

            # Use annotator to classify each pseudo-sample.
            # We use negative node IDs to avoid colliding with real node caches.
            probe_node_ids = [
                -(true_class * n_probes_per_class + i + 1)
                for i in range(len(probing_texts))
            ]

            results = self.annotator.annotate_nodes(
                node_ids=probe_node_ids,
                texts=probing_texts,
                force=True,  # always re-probe, don't use cached node annotations
            )

            # Record budget consumption
            if budget_manager is not None:
                budget_manager.consume(len(probing_texts), stage="probing")

            # Tally predictions
            class_probes = 0
            class_failed = 0
            for nid in probe_node_ids:
                if nid in results:
                    predicted_label, _conf = results[nid]
                    if 0 <= predicted_label < num_classes:
                        counts[true_class][predicted_label] += 1
                        class_probes += 1
                    else:
                        class_failed += 1
                else:
                    class_failed += 1

            total_probes += class_probes
            total_failed += class_failed

            if class_probes > 0:
                diag_rate = counts[true_class][true_class] / class_probes
                logger.debug(
                    "Class %d (%s): %d probes, accuracy=%.3f",
                    true_class, class_names[true_class], class_probes, diag_rate,
                )

        # Normalize rows to probabilities
        noise_matrix = self._normalize_rows(counts)

        # Apply Laplace smoothing for classes with very few probes
        # This prevents zero rows from causing downstream division issues
        noise_matrix = self._smooth(noise_matrix, counts, alpha=0.01)

        # Cache to disk
        self._save_cache(cache_path, noise_matrix)

        per_class_accuracy = np.diag(noise_matrix)
        confused_pairs = self._extract_confused_pairs(noise_matrix)

        logger.info(
            "Noise matrix estimated: %d total probes (%d failed), "
            "mean accuracy=%.3f, worst class=%.3f (%s)",
            total_probes,
            total_failed,
            per_class_accuracy.mean(),
            per_class_accuracy.min(),
            class_names[int(per_class_accuracy.argmin())] if num_classes > 0 else "N/A",
        )

        if confused_pairs:
            top = confused_pairs[0]
            logger.info(
                "Top confusion: %s -> %s (rate=%.3f)",
                class_names[top[0]], class_names[top[1]], top[2],
            )

        return noise_matrix, per_class_accuracy, confused_pairs

    # ── Noise weights ────────────────────────────────────────

    @staticmethod
    def get_noise_weights(
        noise_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], float]]:
        """
        Convert noise matrix into per-class reliability weights and
        pairwise confusion weights.

        Args:
            noise_matrix: C×C array from estimate_noise_matrix().

        Returns:
            per_class_weights:   1D array of length C.
                weight[c] = noise_matrix[c][c] (diagonal = LLM accuracy for class c).
                Higher = more reliable.  Range [0, 1].

            confused_pair_weights: Dict {(i, j): weight} for i != j.
                weight = noise_matrix[i][j] = P(LLM predicts j | true is i).
                Higher = LLM more likely to confuse i -> j.
                Used downstream in agreement_partition and preference_trainer
                to weight preference pairs.
        """
        num_classes = noise_matrix.shape[0]
        per_class_weights = np.diag(noise_matrix).copy()

        confused_pair_weights = {}
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    confused_pair_weights[(i, j)] = float(noise_matrix[i][j])

        return per_class_weights, confused_pair_weights

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _normalize_rows(counts: np.ndarray) -> np.ndarray:
        """Normalize each row of counts to sum to 1 (probability distribution)."""
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with no probes
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return counts / row_sums

    @staticmethod
    def _smooth(
        noise_matrix: np.ndarray,
        counts: np.ndarray,
        alpha: float = 0.01,
    ) -> np.ndarray:
        """
        Apply Laplace smoothing to classes with fewer than 5 probes.

        Prevents zero entries from causing downstream issues while
        preserving the empirical distribution for well-sampled classes.
        """
        num_classes = noise_matrix.shape[0]
        row_totals = counts.sum(axis=1)
        smoothed = noise_matrix.copy()

        for i in range(num_classes):
            if row_totals[i] < 5:
                # Laplace: (count + alpha) / (total + alpha * C)
                total = row_totals[i] + alpha * num_classes
                if total > 0:
                    smoothed[i] = (counts[i] + alpha) / total

        return smoothed

    @staticmethod
    def _extract_confused_pairs(
        noise_matrix: np.ndarray,
    ) -> List[Tuple[int, int, float]]:
        """Extract off-diagonal (i, j, rate) entries sorted by confusion rate."""
        num_classes = noise_matrix.shape[0]
        pairs = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and noise_matrix[i][j] > 0:
                    pairs.append((i, j, float(noise_matrix[i][j])))
        pairs.sort(key=lambda x: -x[2])
        return pairs

    def _cache_path(self, dataset_name: str) -> str:
        model_name = self.annotator.model.replace("/", "_")
        return osp.join(
            self.cache_dir,
            f"{dataset_name}_noise_matrix_{model_name}.npy",
        )

    @staticmethod
    def _save_cache(path: str, noise_matrix: np.ndarray) -> None:
        os.makedirs(osp.dirname(path), exist_ok=True)
        np.save(path, noise_matrix)
        logger.info("Saved noise matrix to %s", path)

    @staticmethod
    def _load_cache(path: str) -> np.ndarray:
        return np.load(path)

    def __repr__(self) -> str:
        return (
            f"LLMNoiseProfiler(annotator={self.annotator!r}, "
            f"cache_dir={self.cache_dir!r})"
        )


# ──────────────────────────────────────────────────────────────
# Self-test on Cora
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)

    from src.data_loading.tag_dataset import load_dataset
    from src.noise_handling.pseudo_samples import PseudoSampleConstructor
    from src.noise_handling.budget_manager import NoiseBudgetManager

    data_dir = osp.join(repo_root, "data")
    locle_data_dir = osp.join(repo_root, "baselines", "locle", "data")
    data_path = (
        data_dir
        if osp.exists(osp.join(data_dir, "cora_random_sbert.pt"))
        else locle_data_dir
    )

    data = load_dataset("cora", data_path=data_path)
    print(f"Loaded Cora: {data.x.size(0)} nodes, {data.num_classes} classes")
    print(f"Classes: {data.class_names}\n")

    # --- Test 1: Synthetic probing (no API calls) ---
    print("=" * 60)
    print("[Test 1] Synthetic noise matrix estimation (simulated)")
    print("=" * 60)

    # Build pseudo-samples
    constructor = PseudoSampleConstructor(
        dataset_name="cora", class_names=data.class_names, seed=42
    )
    samples = constructor.construct(data, n_per_class=10)

    # Simulate: mock annotator that returns noisy predictions
    # We test the profiler logic without real API calls
    class MockAnnotator:
        model = "mock-model"
        total_api_calls = 0

        def __init__(self, class_names, noise_rate=0.2, seed=42):
            self.class_names = class_names
            self.num_classes = len(class_names)
            self.noise_rate = noise_rate
            self._rng = np.random.RandomState(seed)

        def annotate_nodes(self, node_ids, texts, force=False):
            results = {}
            for nid, text in zip(node_ids, texts):
                # Infer true class from the negative node ID encoding
                true_class = (-nid - 1) // 10  # n_probes_per_class=10
                if self._rng.random() < self.noise_rate:
                    # Random wrong class
                    pred = self._rng.randint(0, self.num_classes)
                else:
                    pred = true_class
                pred = max(0, min(pred, self.num_classes - 1))
                results[nid] = (pred, 80.0)
                self.total_api_calls += 1
            return results

    mock_annotator = MockAnnotator(data.class_names, noise_rate=0.15, seed=42)
    profiler = LLMNoiseProfiler(
        annotator=mock_annotator,
        cache_dir=tempfile.mkdtemp(),
    )

    noise_matrix, per_class_acc, confused_pairs = profiler.estimate_noise_matrix(
        dataset_name="cora",
        class_names=data.class_names,
        pseudo_samples=samples,
        n_probes_per_class=10,
        use_cache=False,
    )

    print(f"\n  Noise matrix shape: {noise_matrix.shape}")
    print(f"  Per-class accuracy: {per_class_acc}")
    print(f"  Mean accuracy: {per_class_acc.mean():.3f}")
    print(f"  Row sums (should be ~1.0): {noise_matrix.sum(axis=1)}")
    assert noise_matrix.shape == (7, 7), f"Expected (7,7), got {noise_matrix.shape}"
    assert np.allclose(noise_matrix.sum(axis=1), 1.0, atol=0.01), "Rows must sum to ~1"
    assert per_class_acc.mean() > 0.5, "Mock noise rate 0.15 should give >50% accuracy"
    print("  Test 1: PASS")

    # --- Test 2: get_noise_weights ---
    print("\n" + "=" * 60)
    print("[Test 2] Noise weights extraction")
    print("=" * 60)

    per_class_w, pair_w = LLMNoiseProfiler.get_noise_weights(noise_matrix)
    print(f"  Per-class weights: {per_class_w}")
    print(f"  Number of pair weights: {len(pair_w)}")
    assert len(per_class_w) == 7
    assert len(pair_w) == 7 * 6  # C*(C-1) pairs (some may be 0)
    assert np.allclose(per_class_w, per_class_acc)
    print("  Test 2: PASS")

    # --- Test 3: Confused pairs ---
    print("\n" + "=" * 60)
    print("[Test 3] Confused pairs (sorted)")
    print("=" * 60)
    print(f"  Top 5 confused pairs:")
    for i, j, rate in confused_pairs[:5]:
        print(
            f"    {data.class_names[i]:25s} -> {data.class_names[j]:25s}  rate={rate:.3f}"
        )
    if confused_pairs:
        assert confused_pairs[0][2] >= confused_pairs[-1][2], "Should be sorted descending"
    print("  Test 3: PASS")

    # --- Test 4: Cache round-trip ---
    print("\n" + "=" * 60)
    print("[Test 4] Cache save/load round-trip")
    print("=" * 60)
    cache_path = profiler._cache_path("cora")
    assert osp.exists(cache_path), f"Cache file should exist at {cache_path}"
    loaded = np.load(cache_path)
    assert np.allclose(loaded, noise_matrix), "Loaded matrix should match"
    print(f"  Cache path: {cache_path}")
    print("  Test 4: PASS")

    # --- Test 5: Budget integration ---
    print("\n" + "=" * 60)
    print("[Test 5] Budget manager integration")
    print("=" * 60)

    bm = NoiseBudgetManager(total_budget=140, num_classes=7, probing_fraction=0.1)
    print(f"  Probing budget: {bm.get_probing_budget()}")  # 14

    mock_annotator2 = MockAnnotator(data.class_names, noise_rate=0.1, seed=99)
    profiler2 = LLMNoiseProfiler(
        annotator=mock_annotator2,
        cache_dir=tempfile.mkdtemp(),
    )

    nm2, acc2, _ = profiler2.estimate_noise_matrix(
        dataset_name="cora",
        class_names=data.class_names,
        pseudo_samples=samples,
        n_probes_per_class=10,
        budget_manager=bm,
        use_cache=False,
    )

    print(f"  Probing consumed: {bm._consumed_probing}/{bm._probing_budget}")
    print(f"  Probing remaining: {bm.probing_remaining}")
    # Budget is 14 for probing; 7 classes × 10 probes = 70 needed, but only 14 available
    # So probing should stop after budget is exhausted
    assert bm._consumed_probing <= bm._probing_budget
    print("  Test 5: PASS")

    # --- Test 6: Live API test (optional) ---
    if os.environ.get("OPENAI_API_KEY"):
        print("\n" + "=" * 60)
        print("[Test 6] Live API noise profiling (3 probes/class)")
        print("=" * 60)
        cache_dir = osp.join(repo_root, "data", "annotations")
        annotator = LLMAnnotator(
            dataset_name="cora",
            class_names=data.class_names,
            cache_dir=cache_dir,
            budget=500,
        )
        live_profiler = LLMNoiseProfiler(annotator=annotator, cache_dir=cache_dir)
        live_nm, live_acc, live_pairs = live_profiler.estimate_noise_matrix(
            dataset_name="cora",
            class_names=data.class_names,
            pseudo_samples=samples,
            n_probes_per_class=3,
            use_cache=False,
        )
        print(f"  Per-class accuracy: {live_acc}")
        print(f"  API calls used: {annotator.total_api_calls}")
        if live_pairs:
            print(f"  Top confusion: {data.class_names[live_pairs[0][0]]} -> "
                  f"{data.class_names[live_pairs[0][1]]} ({live_pairs[0][2]:.3f})")
        print("  Test 6: PASS")
    else:
        print("\n[Test 6] OPENAI_API_KEY not set — skipping live API test")

    print(f"\nrepr: {profiler}")
    print("\nAll noise profiler tests complete.")
