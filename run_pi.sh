#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PIPER_MODEL="${PIPER_MODEL:-models/en_US-lessac-medium.onnx}"
exec .venv/bin/python main.py --no-preview "$@"
