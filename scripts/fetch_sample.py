#!/usr/bin/env python3
"""Fetch OpenCV's vtest.avi (pedestrians) so file-input demos work without a camera."""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi"


def main() -> int:
    dest = Path("samples") / "vtest.avi"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"already have {dest} ({dest.stat().st_size} bytes)")
        return 0
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
