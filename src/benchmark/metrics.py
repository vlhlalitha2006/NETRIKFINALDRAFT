from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class RequestRecord:
    start_time: float
    end_time: float
    latency_ms: float
    status_code: int


def compute_metrics(
    records: list[RequestRecord],
    total_duration_s: float,
) -> dict[str, Any]:
    total_requests = len(records)
    successful_requests = sum(1 for record in records if 200 <= record.status_code < 300)
    failed_requests = total_requests - successful_requests

    latencies = [record.latency_ms for record in records]
    start_times = [record.start_time for record in records]
    end_times = [record.end_time for record in records]

    computed_total_duration = (
        float(max(end_times) - min(start_times)) if records else float(total_duration_s)
    )
    avg_latency = sum(latencies) / total_requests if total_requests else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

    metrics = {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "average_latency_ms": float(avg_latency),
        "min_latency_ms": float(min_latency),
        "max_latency_ms": float(max_latency),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "requests_per_second": float(successful_requests / computed_total_duration)
        if computed_total_duration > 0
        else 0.0,
        "total_duration_s": computed_total_duration,
    }
    return metrics
