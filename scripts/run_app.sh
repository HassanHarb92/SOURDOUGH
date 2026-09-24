#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/hharb/Desktop/Projects/Yeast"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
exec streamlit run app/streamlit_app.py
