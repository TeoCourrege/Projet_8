"""Structured JSON-lines logging of every prediction.

This is the (local) production-data storage referenced in the project's
monitoring deliverable: each line is a self-contained JSON record with the
input, the output and the latency, ready to be replayed by
``scripts/run_drift_analysis.py`` or the monitoring dashboard. In a real
deployment this would instead write to a managed store (e.g. Elasticsearch,
PostgreSQL, or an object store such as S3) — see README "Monitoring &
data drift".
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def log_prediction(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_record = {"logged_at": datetime.now(UTC).isoformat(), **record}
    line = json.dumps(full_record, default=str) + "\n"
    with _lock, log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
