#!/bin/bash
set -euo pipefail

echo "==> Installing Homebrew if needed"
if ! command -v brew >/dev/null 2>&1 && [ ! -x /opt/homebrew/bin/brew ]; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

if [ -x /opt/homebrew/bin/brew ]; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

echo "==> Installing Colima and Docker CLI"
brew install colima docker

echo "==> Starting Colima Linux VM"
colima start --cpu 4 --memory 8 --disk 60 --vm-type vz --mount-type virtiofs

echo
echo "==> Done. Docker is ready:"
docker version
echo
echo "You can close this terminal window now."
