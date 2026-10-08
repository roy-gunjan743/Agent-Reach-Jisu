from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent_reach.reliability.scorer import BackendStats


@dataclass
class BackendResult:
    backend: str
    result: str
    evaluation: dict
    score: float


class AdaptiveRouter:
    """
    Context-aware backend router.

    Executes candidate backends, evaluates their actual output
    using the semantic judge, and falls back when the result
    is not useful enough.

    Backend ordering is **deterministic**: when reliability scores are equal
    (e.g. fresh stats at 0), backends are tried in dict insertion order.
    Callers control priority by inserting Backend A before Backend B.
    """

    def __init__(
        self,
        backends: dict[str, Callable[[str], str]],
        stats: dict[str, BackendStats] | None = None,
        quality_threshold: float = 0.65,
        judge: Callable[[str, str], dict] | None = None,
    ):
        self.backends = backends
        self.stats = stats or {
            name: BackendStats(name)
            for name in backends
        }
        self.quality_threshold = quality_threshold
        self._judge = judge

    def _get_judge(self) -> Callable[[str, str], dict]:
        """Return the judge callable, lazily importing the default."""
        if self._judge is not None:
            return self._judge
        from agent_reach.reliability.judge import judge_result

        return judge_result

    def _quality_score(self, evaluation: dict) -> float:
        """
        Combine semantic quality dimensions into one score.
        """

        relevance = float(evaluation.get("relevance", 0))
        freshness = float(evaluation.get("freshness", 0))
        completeness = float(evaluation.get("completeness", 0))
        confidence = float(evaluation.get("confidence", 0))

        return (
            relevance * 0.40
            + freshness * 0.20
            + completeness * 0.25
            + confidence * 0.15
        )

    def route(self, query: str) -> BackendResult | None:
        """
        Try backends until a sufficiently good result is found.
        """

        # Deterministic order: when reliability scores tie (e.g. all 0),
        # Python's sorted() is stable, so dict insertion order is preserved.
        insertion_order = list(self.stats.keys())
        ranked = sorted(
            self.stats.values(),
            key=lambda b: (
                b.reliability_score,
                -insertion_order.index(b.backend),
            ),
            reverse=True,
        )

        attempts: list[BackendResult] = []

        judge_fn = self._get_judge()

        for backend_stat in ranked:
            backend_name = backend_stat.backend

            if backend_name not in self.backends:
                continue

            print(f"\n🔎 Trying backend: {backend_name}")

            try:
                result = self.backends[backend_name](query)

                if not result or not result.strip():
                    backend_stat.record_failure()

                    print("   ❌ Empty result")
                    continue

                print(f"   📄 Received {len(result):,} characters")

                evaluation = judge_fn(query, result)

                score = self._quality_score(evaluation)

                print(
                    f"   Relevance:    {evaluation.get('relevance', 0):.2f}"
                )
                print(
                    f"   Freshness:    {evaluation.get('freshness', 0):.2f}"
                )
                print(
                    f"   Completeness: {evaluation.get('completeness', 0):.2f}"
                )
                print(
                    f"   Confidence:   {evaluation.get('confidence', 0):.2f}"
                )
                print(f"   Quality:      {score:.2f}")
                print(
                    f"   Decision:     {evaluation.get('decision', 'unknown')}"
                )

                attempts.append(
                    BackendResult(
                        backend=backend_name,
                        result=result,
                        evaluation=evaluation,
                        score=score,
                    )
                )

                if (
                    evaluation.get("decision") == "accept"
                    and score >= self.quality_threshold
                ):
                    backend_stat.record_success(score)

                    print(f"   ✅ ACCEPTED: {backend_name}")

                    return attempts[-1]

                backend_stat.record_failure(quality=score)

                print(f"   ⚠️ REJECTED: {backend_name}")
                print("   ↳ Falling back to next backend...")

            except Exception as exc:
                backend_stat.record_failure()

                print(f"   ❌ Backend error: {exc}")

        # If nothing passed the threshold, return the best result
        # rather than returning nothing.
        if attempts:
            best = max(attempts, key=lambda item: item.score)

            print(
                f"\n⚠️ No backend passed the threshold."
                f" Returning best result: {best.backend}"
            )

            return best

        print("\n❌ No usable backend result found.")

        return None

    def stats_summary(self) -> dict[str, dict]:
        """
        Return per-backend statistics as a plain dict for inspection/logging.

        Example return value::

            {
                "jina-mcp-wiki": {
                    "successes": 1,
                    "failures": 0,
                    "quality_score": 0.82,
                    "success_rate": 1.0,
                    "reliability_score": 0.82,
                },
                ...
            }
        """
        summary: dict[str, dict] = {}
        for name, stat in self.stats.items():
            summary[name] = {
                "successes": stat.successes,
                "failures": stat.failures,
                "quality_score": stat.quality_score,
                "success_rate": stat.success_rate,
                "reliability_score": stat.reliability_score,
            }
        return summary

    def print_stats(self) -> None:
        """Pretty-print per-backend statistics."""
        print("\n" + "=" * 50)
        print("        BACKEND STATISTICS")
        print("=" * 50)
        for name, s in self.stats_summary().items():
            print(f"\n  📊 {name}")
            print(f"     Successes:        {s['successes']}")
            print(f"     Failures:         {s['failures']}")
            print(f"     Quality score:    {s['quality_score']:.2f}")
            print(f"     Success rate:     {s['success_rate']:.2f}")
            print(f"     Reliability:      {s['reliability_score']:.2f}")
        print()