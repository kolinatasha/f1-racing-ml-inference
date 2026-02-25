"""Load test: simulate a live F1 race with 20 cars hitting the API.

This script focuses on latency and throughput characteristics.
"""

import asyncio
import os
import random
import statistics
import time
from typing import List

import httpx

API_BASE = os.environ.get(
    "F1_API_BASE", "https://u59hxhp9l5.execute-api.us-east-1.amazonaws.com/prod"
)


# Limit concurrent Lambda invocations to avoid cold-start cascade timeouts
MAX_CONCURRENCY = 5


async def call_laptime(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    driver: str,
    lap: int,
) -> float:
    payload = {
        "driver": driver,
        "track": "bahrain",  # seeded in DynamoDB
        "tire_compound": random.choice(["SOFT", "MEDIUM", "HARD"]),
        "tire_age_laps": max(1, lap - 1),
        "fuel_load_kg": max(5, 100 - lap),
        "track_temp": 38,
        "air_temp": 24,
    }
    async with semaphore:
        start = time.perf_counter()
        r = await client.post(f"{API_BASE}/predict/laptime", json=payload)
        r.raise_for_status()
        return (time.perf_counter() - start) * 1000.0


async def simulate_driver(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    name: str,
    total_laps: int,
) -> List[float]:
    latencies: List[float] = []
    errors = 0
    for lap in range(1, total_laps + 1):
        try:
            lat = await call_laptime(client, semaphore, name, lap)
            latencies.append(lat)
        except Exception:
            errors += 1
    return latencies


async def main_async():
    drivers = [f"DRV{i:02d}" for i in range(1, 21)]
    total_laps = int(os.environ.get("F1_TOTAL_LAPS", "10"))  # default 10 laps for quick run

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with httpx.AsyncClient(timeout=15.0) as client:
        tasks = [simulate_driver(client, semaphore, d, total_laps) for d in drivers]
        all_latencies = await asyncio.gather(*tasks)

    flat = [x for sub in all_latencies for x in sub]
    if not flat:
        print("No successful requests.")
        return
    flat.sort()

    p95 = flat[int(0.95 * len(flat))]
    p99 = flat[int(0.99 * len(flat))]
    mean = statistics.mean(flat)

    print(f"Requests:     {len(flat)} / {len(drivers) * total_laps} total")
    print(f"Mean latency: {mean:.0f} ms")
    print(f"p95 latency:  {p95:.0f} ms")
    print(f"p99 latency:  {p99:.0f} ms")


if __name__ == "__main__":
    asyncio.run(main_async())
