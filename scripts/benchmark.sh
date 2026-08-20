#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
python benchmarks/benchmark.py --config configs/benchmark.yaml "$@"
