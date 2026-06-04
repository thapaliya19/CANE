"""
Budget manager for the NoisePref pipeline.

Tracks LLM query consumption across pipeline stages:
  - Stage 0 (Noise Probing):       probing_fraction of total budget
  - Stage I+II (Annotation + Refinement): remainder

All LLM calls must go through consume() so the pipeline can enforce
hard budget limits and stop when exhausted.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NoiseBudgetManager:
    """
    Tracks and enforces LLM query budget across pipeline stages.

    Args:
        total_budget:     Total number of LLM queries allowed (e.g. 20 * num_classes).
        num_classes:      Number of classes in the dataset.
        probing_fraction: Fraction of total budget reserved for noise probing (Stage 0).
    """

    def __init__(
        self,
        total_budget: int,
        num_classes: int,
        probing_fraction: float = 0.1,
    ):
        self.total_budget = total_budget
        self.num_classes = num_classes
        self.probing_fraction = probing_fraction

        self._probing_budget = int(total_budget * probing_fraction)
        self._annotation_budget = total_budget - self._probing_budget

        # Consumption tracking
        self._consumed = 0
        self._consumed_probing = 0
        self._consumed_annotation = 0

    # ── Budget queries ──────────────────────────────────────

    def get_probing_budget(self) -> int:
        """Budget reserved for Stage 0 noise probing."""
        return self._probing_budget

    def get_annotation_budget(self) -> int:
        """Budget for Stages I+II (initial annotation + iterative refinement)."""
        return self._annotation_budget

    @property
    def remaining(self) -> int:
        """Total remaining budget (annotation pool)."""
        return max(0, self._annotation_budget - self._consumed_annotation)

    @property
    def probing_remaining(self) -> int:
        """Remaining probing budget."""
        return max(0, self._probing_budget - self._consumed_probing)

    def is_exhausted(self) -> bool:
        """True if annotation budget is fully consumed."""
        return self._consumed_annotation >= self._annotation_budget

    def is_probing_exhausted(self) -> bool:
        """True if probing budget is fully consumed."""
        return self._consumed_probing >= self._probing_budget

    # ── Probing cost helpers ────────────────────────────────

    def recommend_probes_per_class(self, max_probes: int = 50) -> int:
        """
        Recommend how many probes per class fit within the probing budget.

        Returns at least 1 (even if budget is tiny) and at most max_probes.
        """
        if self.num_classes <= 0:
            return 0
        return max(1, min(self._probing_budget // self.num_classes, max_probes))

    def probing_cost_report(self, n_probes_per_class: int) -> dict:
        """
        Compute a cost report for a given probing configuration.

        Follows DMA convention: probing cost is documented separately so
        comparisons with baselines (LoCLE, LLM-GNN) that have no probing
        stage remain fair.

        Returns dict with:
            requested_queries:    n_probes_per_class * num_classes
            probing_budget:       budget allocated for probing
            actual_queries:       min(requested, probing_budget)
            actual_per_class:     actual_queries // num_classes
            fraction_of_total:    actual_queries / total_budget
            annotation_remaining: annotation budget (untouched by probing)
            fairness_ok:          True if probing ≤ 20% of total budget
        """
        requested = n_probes_per_class * self.num_classes
        actual = min(requested, self._probing_budget)
        actual_per_class = actual // self.num_classes if self.num_classes > 0 else 0
        fraction = actual / self.total_budget if self.total_budget > 0 else 0.0

        return {
            "requested_queries": requested,
            "probing_budget": self._probing_budget,
            "actual_queries": actual,
            "actual_per_class": actual_per_class,
            "fraction_of_total": fraction,
            "annotation_remaining": self._annotation_budget,
            "fairness_ok": fraction <= 0.20,
        }

    # ── Consumption ─────────────────────────────────────────

    def consume(self, n: int, stage: str = "annotation") -> int:
        """
        Record n LLM queries consumed.

        Args:
            n:     Number of queries to consume.
            stage: 'probing' or 'annotation'.

        Returns:
            Number of queries actually consumed (may be less than n
            if budget is nearly exhausted).
        """
        if stage == "probing":
            available = max(0, self._probing_budget - self._consumed_probing)
            actual = min(n, available)
            self._consumed_probing += actual
        else:
            available = max(0, self._annotation_budget - self._consumed_annotation)
            actual = min(n, available)
            self._consumed_annotation += actual

        self._consumed += actual

        if actual < n:
            logger.warning(
                "Budget trimmed: requested %d, consumed %d (%s stage, %d remaining)",
                n,
                actual,
                stage,
                available - actual,
            )

        return actual

    # ── Per-round allocation ────────────────────────────────

    def get_round_budget(
        self,
        round_num: int,
        total_rounds: int,
        initial_fraction: float = 0.5,
    ) -> int:
        """
        LoCLE-style per-round budget allocation.

        Stage I (round_num == 0) gets ``initial_fraction`` of the annotation
        budget.  Stage II rounds (round_num >= 1) split the remainder evenly
        across ``total_rounds - 1`` refinement rounds.

        Args:
            round_num:        0-based round index.
            total_rounds:     Total number of rounds (including the initial one).
            initial_fraction: Fraction of annotation budget for Stage I (default 0.5).

        Returns:
            Number of LLM queries allocated for this round, capped at remaining budget.
        """
        if total_rounds <= 0:
            return 0

        if round_num == 0:
            # Stage I — initial annotation
            alloc = int(self._annotation_budget * initial_fraction)
        else:
            # Stage II — refinement rounds split remainder equally
            stage2_budget = self._annotation_budget - int(
                self._annotation_budget * initial_fraction
            )
            refinement_rounds = max(total_rounds - 1, 1)
            alloc = stage2_budget // refinement_rounds

        return min(alloc, self.remaining)

    def allocate_round(
        self,
        num_rounds_remaining: int,
        min_per_round: Optional[int] = None,
    ) -> int:
        """
        Suggest how many queries to use this round.

        Divides remaining annotation budget evenly across remaining rounds,
        with an optional minimum per round.
        """
        if num_rounds_remaining <= 0:
            return 0
        per_round = self.remaining // num_rounds_remaining
        if min_per_round is not None:
            per_round = max(per_round, min_per_round)
        return min(per_round, self.remaining)

    # ── Summary ─────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "total_budget": self.total_budget,
            "probing_budget": self._probing_budget,
            "annotation_budget": self._annotation_budget,
            "consumed_probing": self._consumed_probing,
            "consumed_annotation": self._consumed_annotation,
            "consumed_total": self._consumed,
            "remaining_probing": self.probing_remaining,
            "remaining_annotation": self.remaining,
            "exhausted": self.is_exhausted(),
        }

    def __repr__(self) -> str:
        return (
            f"NoiseBudgetManager(total={self.total_budget}, "
            f"probing={self._consumed_probing}/{self._probing_budget}, "
            f"annotation={self._consumed_annotation}/{self._annotation_budget})"
        )


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    bm = NoiseBudgetManager(total_budget=140, num_classes=7, probing_fraction=0.1)
    print(f"Probing budget: {bm.get_probing_budget()}")    # 14
    print(f"Annotation budget: {bm.get_annotation_budget()}")  # 126
    assert bm.get_probing_budget() == 14
    assert bm.get_annotation_budget() == 126

    # Consume probing
    bm.consume(10, stage="probing")
    assert bm.probing_remaining == 4
    print(f"Probing remaining: {bm.probing_remaining}")

    # Consume annotation
    bm.consume(50)
    print(f"Remaining: {bm.remaining}")  # 76
    assert bm.remaining == 76
    assert not bm.is_exhausted()

    bm.consume(80)
    print(f"Remaining after over-consume: {bm.remaining}")  # 0
    assert bm.is_exhausted()

    # Round allocation (legacy)
    bm2 = NoiseBudgetManager(total_budget=140, num_classes=7, probing_fraction=0.1)
    alloc = bm2.allocate_round(num_rounds_remaining=3)
    print(f"Per-round allocation (3 rounds): {alloc}")  # 42
    assert alloc == 42

    # get_round_budget (LoCLE-style)
    bm3 = NoiseBudgetManager(total_budget=140, num_classes=7, probing_fraction=0.1)
    # annotation_budget = 126, initial_fraction=0.5 -> Stage I = 63
    r0 = bm3.get_round_budget(round_num=0, total_rounds=5, initial_fraction=0.5)
    print(f"Stage I budget: {r0}")  # 63
    assert r0 == 63
    # Stage II: (126 - 63) / 4 = 15 per refinement round
    r1 = bm3.get_round_budget(round_num=1, total_rounds=5, initial_fraction=0.5)
    print(f"Stage II round 1 budget: {r1}")  # 15
    assert r1 == 15
    r4 = bm3.get_round_budget(round_num=4, total_rounds=5, initial_fraction=0.5)
    assert r4 == 15
    # Respects remaining cap
    bm3.consume(120, stage="annotation")
    r_capped = bm3.get_round_budget(round_num=1, total_rounds=5, initial_fraction=0.5)
    print(f"Capped to remaining: {r_capped}")  # 6
    assert r_capped == 6

    print(f"\nSummary: {bm.summary()}")
    print(f"repr: {bm}")
    print("\nAll budget manager tests PASS.")
