#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Creating SmartMelt environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
if ! python - <<'PY' >/dev/null 2>&1
import streamlit, pyarrow
parts = tuple(int(x) for x in streamlit.__version__.split('.')[:2])
assert parts >= (1, 60)
PY
then
  echo "Installing or upgrading SmartMelt dependencies..."
  python -m pip install --upgrade pip
  python -m pip install --upgrade -r requirements.txt
fi
exec python -m streamlit run streamlit_app.py
