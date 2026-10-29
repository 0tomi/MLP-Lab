#!/usr/bin/env bash
# Start the MLP lab. The first run creates .venv and installs requirements.txt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

supported() { "$1" -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 13))' 2>/dev/null; }

if [ ! -x .venv/bin/python ]; then
  PY=""
  for candidate in python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null && supported "$candidate"; then PY="$candidate"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "Se necesita Python 3.12 o 3.13." >&2
    exit 1
  fi
  "$PY" -m venv .venv
fi

stamp=.venv/.requirements.cksum
sum="$(cksum < requirements.txt)"
if [ "$(cat "$stamp" 2>/dev/null)" != "$sum" ]; then
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  echo "$sum" > "$stamp"
fi

exec .venv/bin/streamlit run app.py "$@"
