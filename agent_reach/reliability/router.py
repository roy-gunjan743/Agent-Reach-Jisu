from dataclasses import dataclass
from typing import Callable

from agent_reach.reliability.judge import judge_result
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
    """

    def __init__(
        self,
        backends: dict[str, Callable[[str], str]],
        stats: dict[str, BackendStats] | None = None,
        quality_threshold: float = 0.65,
    ):
        self.backends = backends
        self.stats = stats or {
            name: BackendStats(name)
            for name in backends
        }
        self.quality_threshold = quality_threshold

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

        ranked = sorted(
            self.stats.values(),
            key=lambda backend: backend.reliability_score,
            reverse=True,
        )

        attempts = []

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

                evaluation = judge_result(query, result)

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

                backend_stat.record_failure()

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