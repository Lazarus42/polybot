#!/usr/bin/env bash
set -euo pipefail

# Prefer Homebrew Python if installed.
if [ -x "/usr/local/bin/python3" ]; then
  export PATH="/usr/local/bin:/usr/local/opt/python@3.14/libexec/bin:${PATH}"
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "Activated venv and installed requirements."
