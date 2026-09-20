#!/usr/bin/env bash
# From the repository root: bash scripts/setup_encoder_env.sh [environment-path]
set -euo pipefail
encoder_env="${1:-.venv}"
python3.10 -m venv "$encoder_env"
"$encoder_env/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
"$encoder_env/bin/python" -m pip install --no-cache-dir -r requirements-bootstrap.txt
"$encoder_env/bin/python" -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
"$encoder_env/bin/python" -m pip install --no-cache-dir -r requirements-inference.txt
"$encoder_env/bin/python" -m pip install --no-cache-dir -r requirements-dev.txt
"$encoder_env/bin/python" -m pip check
"$encoder_env/bin/python" -m pytest tests -q
