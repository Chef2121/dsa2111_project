#!/usr/bin/env bash
# Deploy Radview (API + dashboard) on a RONIN / Ubuntu machine with the already-trained models.
#
# Run ON the RONIN machine, from the cloned repo:   bash scripts/deploy_ronin.sh
# Prerequisite: Radview-trained-models-yelp.zip copied to the machine (default: ~/Radview-trained-models-yelp.zip,
# override with BUNDLE=/path/to/zip). See docs/DEPLOY_RONIN.md for the full guide.
set -euo pipefail

BUNDLE="${BUNDLE:-$HOME/Radview-trained-models-yelp.zip}"
cd "$(dirname "$0")/.."

echo "== 1/5 Docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found - installing Docker Engine + compose plugin (official convenience script)"
  curl -fsSL https://get.docker.com | sudo sh
fi
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"   # current shell is not yet in the docker group
  sudo usermod -aG docker "$USER" || true
fi
$DOCKER compose version >/dev/null

echo "== 2/5 Trained models"
if [ ! -f data/work/yelp/outputs/risk_latest.parquet ]; then
  [ -f "$BUNDLE" ] || { echo "ERROR: $BUNDLE not found - copy the trained-models zip to the machine first"; exit 1; }
  command -v unzip >/dev/null || sudo apt-get install -y unzip
  unzip -q -o "$BUNDLE" -d .
fi
# The containers run as uid 1000; make sure they can write feedback and refreshed plans.
chmod -R a+rwX data

echo "== 3/5 Disk check"
df -h . | tail -1

echo "== 4/5 Build image (runs the test suite) and start API + dashboard"
$DOCKER compose up -d --build api dashboard

echo "== 5/5 Wait for the API"
for i in $(seq 1 60); do
  if curl -fs http://localhost:8000/health >/dev/null; then break; fi
  sleep 5
done
curl -fs http://localhost:8000/health && echo
curl -fs http://localhost:8000/meta | head -c 200 && echo
$DOCKER compose ps
echo
echo "Done. Dashboard: http://localhost:8501   API docs: http://localhost:8000/docs   (on the machine)"
echo "From your laptop use an SSH tunnel / RONIN LINK port forward - see docs/DEPLOY_RONIN.md"
