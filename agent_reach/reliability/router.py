from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from agent_reach.reliability.scorer import BackendStats


def _clean_score(val: Any) -> float:
    """Coerce and clamp score value to [0.0, 1.0]. Values 1 < x <= 100 are treated as percentages."""
    if val is None:
        return 0.0
    try:
        fval = float(val)
    except (ValueError, TypeError):
        return 0.0
    if 1.0 < fval <= 100.0:
        fval /= 100.0
    return max(0.0, min(1.0, fval))


@dataclass
class BackendResult:
    backend: str
    result: str
    evaluation: dict
    score: float
    partial_judge_outage: bool = False


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
        normalizer: Callable[[str, str], Any] | None = None,
    ):
        self.backends = backends
        self.stats = stats or {
            name: BackendStats(name)
            for name in backends
        }
        self.quality_threshold = quality_threshold
        self._judge = judge
        self.normalizer = normalizer
        self.last_trace: list[dict[str, Any]] = []

    def _get_judge(self) -> Callable[[str, str], dict]:
        """Return the judge callable, lazily importing the default."""
        if self._judge is not None:
            return self._judge
        from agent_reach.reliability.judge import judge_result

        return judge_result

    def _quality_score(self, evaluation: dict) -> float:
        """Combine semantic quality dimensions into one score."""
        has_metrics = any(
            k in evaluation
            for k in ("relevance", "freshness", "completeness", "confidence")
        )
        if not has_metrics and "score" in evaluation:
            return _clean_score(evaluation["score"])

        relevance = _clean_score(evaluation.get("relevance", 0))
        freshness = _clean_score(evaluation.get("freshness", 0))
        completeness = _clean_score(evaluation.get("completeness", 0))
        confidence = _clean_score(evaluation.get("confidence", 0))

        return (
            relevance * 0.40
            + freshness * 0.20
            + completeness * 0.25
            + confidence * 0.15
        )

    def route(self, query: str) -> BackendResult | None:
        """Try backends until a sufficiently good result is found."""
        self.last_trace = []

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

        from agent_reach.reliability.judge import MODEL, judge_result
        is_default_gemma = (self._judge is None) or (judge_fn is judge_result) or (getattr(judge_fn, "__name__", "") == "judge_result")

        for backend_stat in ranked:
            backend_name = backend_stat.backend

            if backend_name not in self.backends:
                continue

            print(f"\n🔎 Trying backend: {backend_name}")

            # 1. Fetch
            try:
                raw_result = self.backends[backend_name](query)
            except Exception as exc:
                backend_stat.record_failure()
                err_msg = str(exc)
                print(f"   ❌ Backend error: {err_msg}")
                self.last_trace.append({
                    "backend": backend_name,
                    "outcome": "fetch_error",
                    "chars": 0,
                    "score": None,
                    "evaluation": None,
                    "error": err_msg,
                    "latency_ms": 0.0,
                })
                continue

            if not raw_result or not raw_result.strip():
                backend_stat.record_failure()
                print("   ❌ Empty result")
                self.last_trace.append({
                    "backend": backend_name,
                    "outcome": "empty",
                    "chars": 0,
                    "score": None,
                    "evaluation": None,
                    "error": "Empty result",
                    "latency_ms": 0.0,
                })
                continue

            # 2. Normalize
            if self.normalizer is not None:
                norm_out = self.normalizer(raw_result, query)
            else:
                from agent_reach.reliability.normalizer import normalize

                norm_out = normalize(raw_result, query=query)

            normalized_text = norm_out.text if hasattr(norm_out, "text") else str(norm_out)
            chars = len(normalized_text)
            print(f"   📄 Received {len(raw_result):,} characters -> normalized to {chars:,} characters")

            # 3. Judge evaluation
            t_judge_start = time.perf_counter()
            try:
                evaluation = judge_fn(query, normalized_text)
                latency_ms = (time.perf_counter() - t_judge_start) * 1000
            except Exception as exc:
                latency_ms = (time.perf_counter() - t_judge_start) * 1000
                err_msg = str(exc)
                print(f"   ⚠️ judge unavailable: {err_msg}")
                self.last_trace.append({
                    "backend": backend_name,
                    "outcome": "judge_error",
                    "chars": chars,
                    "score": None,
                    "evaluation": None,
                    "error": err_msg,
                    "latency_ms": latency_ms,
                })
                continue

            if not isinstance(evaluation, dict) or evaluation.get("judge_error"):
                reason = (
                    evaluation.get("reason", "Judge error")
                    if isinstance(evaluation, dict)
                    else f"Judge returned non-dict: {type(evaluation).__name__}"
                )
                print(f"   ⚠️ judge unavailable: {reason}")
                self.last_trace.append({
                    "backend": backend_name,
                    "outcome": "judge_error",
                    "chars": chars,
                    "score": None,
                    "evaluation": evaluation if isinstance(evaluation, dict) else None,
                    "error": reason,
                    "latency_ms": latency_ms,
                })
                continue

            # 4. Normalize decision & scores
            raw_decision = evaluation.get("decision")
            decision = str(raw_decision).strip().lower() if raw_decision is not None else "reject"
            evaluation["decision"] = decision

            score = self._quality_score(evaluation)
            evaluation["score"] = score

            rel = _clean_score(evaluation.get("relevance", 0))
            fresh = _clean_score(evaluation.get("freshness", 0))
            comp = _clean_score(evaluation.get("completeness", 0))
            conf = _clean_score(evaluation.get("confidence", 0))

            if is_default_gemma:
                print(f"   🧠 Gemma 4 evaluation ({MODEL}, {latency_ms:.0f} ms):")
            else:
                print(f"   🧠 Judge evaluation ({latency_ms:.0f} ms):")

            print(f"   Relevance:    {rel:.2f}")
            print(f"   Freshness:    {fresh:.2f}")
            print(f"   Completeness: {comp:.2f}")
            print(f"   Confidence:   {conf:.2f}")
            print(f"   Quality:      {score:.2f}")
            print(f"   Decision:     {decision.upper()}")
            print(f"   Reason:       {evaluation.get('reason', '')}")

            res_obj = BackendResult(
                backend=backend_name,
                result=raw_result,
                evaluation=evaluation,
                score=score,
            )

            if decision == "accept" and score >= self.quality_threshold:
                backend_stat.record_success(score)
                print(f"   ✅ ACCEPTED: {backend_name}")
                self.last_trace.append({
                    "backend": backend_name,
                    "outcome": "accepted",
                    "chars": chars,
                    "score": score,
                    "evaluation": evaluation,
                    "error": None,
                    "latency_ms": latency_ms,
                })
                return res_obj

            backend_stat.record_failure(quality=score)
            print(f"   ⚠️ REJECTED: {backend_name}")
            print("   ↳ Falling back to next backend...")
            attempts.append(res_obj)
            self.last_trace.append({
                "backend": backend_name,
                "outcome": "rejected",
                "chars": chars,
                "score": score,
                "evaluation": evaluation,
                "error": None,
                "latency_ms": latency_ms,
            })

        judge_error_backends = [
            t["backend"] for t in self.last_trace if t["outcome"] == "judge_error"
        ]

        if attempts:
            best = max(attempts, key=lambda item: item.score)
            if judge_error_backends:
                best.partial_judge_outage = True
                print(f"\n⚠️ judge unavailable for: {', '.join(judge_error_backends)}")
                print(f"⚠️ Returning best rejected result: {best.backend}")
            else:
                print(
                    f"\n⚠️ No backend passed; returning best rejected result: {best.backend}"
                )
            return best

        if judge_error_backends:
            print(f"\n⚠️ judge unavailable for: {', '.join(judge_error_backends)}")

        print("\n❌ Judge unavailable, no verdict.")
        return None

    def stats_summary(self) -> dict[str, dict]:
        """Return per-backend statistics as a plain dict for inspection/logging."""
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