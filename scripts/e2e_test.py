"""
End-to-end test for the Month-End Close pipeline.

Triggers a close run, polls Redis until it completes, prints the final state.
Run via: docker compose exec api python scripts/e2e_test.py
"""

import json
import time
import uuid

import redis
import requests

from app.db.database import settings

API_URL = "http://api:8000/api/v1/trigger-close"
POLL_INTERVAL = 10          # seconds between polls
MAX_WAIT = 30 * 60          # 30 min hard cap


def main() -> None:
    r = redis.from_url(settings.redis_url, decode_responses=True)

    # ---- 1. Trigger the close ----------------------------------------
    print("[1/4] Triggering month-end close...")
    resp = requests.post(API_URL, timeout=10)
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    print(f"      run_id = {run_id}\n")

    # ---- 2. Poll until terminal state --------------------------------
    print("[2/4] Waiting for completion (polling every 10s)...")
    start = time.time()
    last_status = None
    while time.time() - start < MAX_WAIT:
        status = r.get(f"close:{run_id}:status") or "unknown"
        phase2_count = r.get(f"close:{run_id}:phase2_count") or "0"
        phase3 = r.get(f"close:{run_id}:phase3") or "-"

        if status != last_status:
            elapsed = int(time.time() - start)
            print(f"      [{elapsed:>4}s] status={status:<10} phase2_count={phase2_count}/8 phase3={phase3}")
            last_status = status

        if status in ("completed", "failed"):
            break

        time.sleep(POLL_INTERVAL)

    elapsed = int(time.time() - start)
    print(f"      finished in {elapsed}s\n")

    # ---- 3. Print Phase 1 + 2 summary --------------------------------
    print("[3/4] Phase summary:")
    keys = r.keys(f"close:{run_id}:phase1:*")
    phase1_done = len([k for k in keys if k.endswith(":done") is False and not k.startswith(f"close:{run_id}:phase1_failures")])
    phase1_failures = [k for k in keys if "failures" in k]

    print(f"      Phase 1: {phase1_done}/8 companies done")
    for k in phase1_failures:
        company = k.split(":")[-1]
        failed_agents = r.get(k)
        print(f"               {company}: {failed_agents}")

    phase2_done = len(r.keys(f"close:{run_id}:phase2:*")) - 1  # -1 for counter key
    print(f"      Phase 2: {r.get(f'close:{run_id}:phase2_count')}/8 companies done")

    phase3_status = r.get(f"close:{run_id}:phase3_status") or "not-run"
    print(f"      Phase 3: {phase3_status}\n")

    # ---- 4. Print Phase 3 elimination result -------------------------
    raw = r.get(f"close:{run_id}:phase3:result")
    if raw:
        result = json.loads(raw)
        print("[4/4] Phase 3 elimination result:")
        print(f"      status          : {result.get('status')}")
        print(f"      transactions    : {result.get('total_transactions')}")
        print(f"      unique pairs    : {result.get('unique_pairs')}")
        print(f"      mismatches      : {result.get('mismatch_count')}")
        print(f"      asymmetry (USD) : ${result.get('total_asymmetry_usd', 0):,.2f}")
        print(f"      summary         : {result.get('summary', '')[:200]}")
    else:
        print("[4/4] No Phase 3 result stored in Redis.")

    print(f"\nDone. Final status: {r.get(f'close:{run_id}:status')}")


if __name__ == "__main__":
    main()