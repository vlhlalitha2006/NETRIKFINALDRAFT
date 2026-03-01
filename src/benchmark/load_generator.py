from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path

import httpx
import pandas as pd

from src.benchmark.metrics import RequestRecord, compute_metrics
from src.benchmark.report import print_summary


def _load_loan_ids(data_csv_path: Path) -> list[str]:
    if not data_csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_csv_path}")
    dataframe = pd.read_csv(data_csv_path)
    if "Loan_ID" not in dataframe.columns:
        raise ValueError("Loan_ID column is required in dataset.")
    return dataframe["Loan_ID"].astype(str).tolist()


async def _send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    loan_id: str,
    semaphore: asyncio.Semaphore,
) -> RequestRecord:
    async with semaphore:
        start_ts = time.perf_counter()
        start_perf = time.perf_counter()
        status_code = 0
        try:
            response = await client.post(endpoint, json={"loan_id": loan_id})
            status_code = response.status_code
        except Exception:
            status_code = 0
        end_perf = time.perf_counter()
        end_ts = time.perf_counter()
        return RequestRecord(
            start_time=start_ts,
            end_time=end_ts,
            latency_ms=(end_perf - start_perf) * 1000.0,
            status_code=status_code,
        )


async def run_benchmark(
    base_url: str,
    endpoint_name: str,
    total_requests: int,
    concurrency_level: int,
    loan_ids: list[str],
    timeout_s: float = 30.0,
) -> tuple[list[RequestRecord], float]:
    endpoint_path = "/score" if endpoint_name == "score" else "/explain"
    semaphore = asyncio.Semaphore(concurrency_level)
    benchmark_start = time.perf_counter()

    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        tasks: list[asyncio.Task[RequestRecord]] = []
        for _ in range(total_requests):
            sampled_loan_id = random.choice(loan_ids)
            tasks.append(
                asyncio.create_task(
                    _send_request(
                        client=client,
                        endpoint=endpoint_path,
                        loan_id=sampled_loan_id,
                        semaphore=semaphore,
                    )
                )
            )
        records = await asyncio.gather(*tasks)

    benchmark_duration = time.perf_counter() - benchmark_start
    return records, benchmark_duration


def _resolve_concurrency(mode: str, requested_concurrency: int) -> int:
    if mode == "baseline":
        return 1
    if mode == "concurrent":
        return requested_concurrency if requested_concurrency > 0 else 8
    return max(1, requested_concurrency)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async load generator for risk service.")
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["score", "explain"],
        default="score",
        help="Endpoint to benchmark.",
    )
    parser.add_argument("--requests", type=int, default=100, help="Total request count.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Concurrent request limit.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["custom", "baseline", "concurrent"],
        default="custom",
        help="Scenario mode: baseline=1, concurrent=8/custom.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="FastAPI server base URL.",
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=Path("data/raw/TRAIN.csv"),
        help="CSV file used to sample Loan_ID values.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds.")
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    concurrency = _resolve_concurrency(args.mode, args.concurrency)
    loan_ids = _load_loan_ids(args.data_csv)

    records, total_duration_s = await run_benchmark(
        base_url=args.base_url,
        endpoint_name=args.endpoint,
        total_requests=args.requests,
        concurrency_level=concurrency,
        loan_ids=loan_ids,
        timeout_s=args.timeout,
    )
    metrics = compute_metrics(records=records, total_duration_s=total_duration_s)
    print_summary(
        endpoint=args.endpoint,
        concurrency_level=concurrency,
        metrics=metrics,
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
