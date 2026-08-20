from __future__ import annotations

import sys
from datetime import datetime, timezone


def log(level: str, stage: str, message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    stream = sys.stderr if level == "ERROR" else sys.stdout
    print(f"{ts} {level:5} [{stage}] {message}", file=stream, flush=True)
