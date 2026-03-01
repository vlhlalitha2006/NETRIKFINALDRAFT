from __future__ import annotations

from typing import Any


def print_summary(
    endpoint: str,
    concurrency_level: int,
    metrics: dict[str, Any],
) -> None:
    rows = [
        ("Endpoint", endpoint),
        ("Concurrency", str(concurrency_level)),
        ("Total Duration (s)", f"{metrics['total_duration_s']:.3f}"),
        ("RPS", f"{metrics['requests_per_second']:.2f}"),
        ("Total Requests", str(metrics["total_requests"])),
        ("Successful", str(metrics["successful_requests"])),
        ("Failed", str(metrics["failed_requests"])),
        ("Avg Latency (ms)", f"{metrics['average_latency_ms']:.2f}"),
        ("P50 Latency (ms)", f"{metrics['p50_latency_ms']:.2f}"),
        ("P95 Latency (ms)", f"{metrics['p95_latency_ms']:.2f}"),
        ("Min Latency (ms)", f"{metrics['min_latency_ms']:.2f}"),
        ("Max Latency (ms)", f"{metrics['max_latency_ms']:.2f}"),
    ]
    key_width = max(len(key) for key, _ in rows)
    value_width = max(len(value) for _, value in rows)

    border = f"+-{'-' * key_width}-+-{'-' * value_width}-+"
    print(border)
    print(f"| {'Metric'.ljust(key_width)} | {'Value'.ljust(value_width)} |")
    print(border)
    for key, value in rows:
        print(f"| {key.ljust(key_width)} | {value.ljust(value_width)} |")
    print(border)
