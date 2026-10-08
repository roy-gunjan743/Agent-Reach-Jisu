from dataclasses import dataclass


@dataclass
class BackendStats:
    backend: str
    successes: int = 0
    failures: int = 0
    quality_score: float = 1.0

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
        self.quality_score = quality

    def record_failure(self) -> None:
        self.failures += 1


def rank_backends(backends: list[BackendStats]) -> list[BackendStats]:
    return sorted(
        backends,
        key=lambda backend: backend.reliability_score,
        reverse=True,
    )