from dataclasses import dataclass, field


@dataclass
class BackendStats:
    backend: str
    successes: int = 0
    failures: int = 0
    quality_score: float = 1.0
    _quality_sum: float = field(default=0.0, repr=False)
    _quality_count: int = field(default=0, repr=False)

    @property
    def total_requests(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        total = self.total_requests

        if total == 0:
            return 0.0

        return self.successes / total

    @property
    def reliability_score(self) -> float:
        return self.success_rate * self.quality_score

    def record_success(self, quality: float = 1.0) -> None:
        self.successes += 1
        self._quality_sum += quality
        self._quality_count += 1
        self.quality_score = self._quality_sum / self._quality_count

    def record_failure(self, quality: float | None = None) -> None:
        self.failures += 1
        if quality is not None:
            self._quality_sum += quality
            self._quality_count += 1
            self.quality_score = self._quality_sum / self._quality_count


def rank_backends(backends: list[BackendStats]) -> list[BackendStats]:
    return sorted(
        backends,
        key=lambda backend: backend.reliability_score,
        reverse=True,
    )