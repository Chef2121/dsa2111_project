#!/usr/bin/env bash
# Deploy Radview on the group's RONIN machine with k3s (single-node Kubernetes): the API and two dashboard
# replicas from the k8s/ manifests, serving the already-trained models. This is how the shared machine runs.
#
# Run ON the RONIN machine, from the cloned repo:   bash scripts/deploy_ronin_k3s.sh
# Prerequisite: Radview-trained-models-yelp.zip on the machine (default ~/Radview-trained-models-yelp.zip,
# override with BUNDLE=/path/to/zip) - only needed the first time. See docs/DEPLOY_RONIN.md.
#
# Safe to re-run (e.g. after `git pull`): it rebuilds the image, re-applies the manifests and restarts the pods.
# The trained models are copied onto the Kubernetes volume only if the volume does not have them yet, so
# feedback and refreshed plans written by the running app are kept.
set -euo pipefail

BUNDLE="${BUNDLE:-$HOME/Radview-trained-models-yelp.zip}"
cd "$(dirname "$0")/.."

echo "== 1/6 Docker"
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found - installing Docker Engine (official convenience script)"
  curl -fsSL https://get.docker.com | sudo sh
fi
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"   # current shell is not yet in the docker group
  sudo usermod -aG docker "$USER" || true
fi

echo "== 2/6 Trained models"
if [ ! -f data/work/yelp/outputs/risk_latest.parquet ]; then
  [ -f "$BUNDLE" ] || { echo "ERROR: $BUNDLE not found - copy the trained-models zip to the machine first"; exit 1; }
  command -v unzip >/dev/null || sudo apt-get install -y unzip
  unzip -q -o "$BUNDLE" -d .
fi

echo "== 3/6 Build image (runs the test suite)"
$DOCKER build -t radview:latest .

echo "== 4/6 k3s"
# Docker Compose and k3s must not both serve Radview on an 8 GB machine.
if $DOCKER ps --format '{{.Names}}' | grep -Eq -- '-(api|dashboard)-[0-9]+$'; then
  echo "Stopping the Docker Compose deployment (k3s replaces it)"
  $DOCKER compose stop api dashboard || true
fi
if ! command -v k3s >/dev/null 2>&1; then
  # --docker: k3s runs pods with the host's Docker, so it uses radview:latest directly (no second 3.6 GB copy).
  # --disable traefik: no ingress controller needed; the dashboard Service is exposed by k3s' built-in load balancer.
  curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--docker --disable traefik --write-kubeconfig-mode 644" sh -
fi
kubectl wait --for=condition=Ready node --all --timeout=180s

echo "== 5/6 Kubernetes objects and data"
kubectl apply -f k8s/00-namespace-config.yaml
kubectl -n radview patch configmap radview-config --type merge -p '{"data":{"RADVIEW_SOURCE":"yelp"}}'
kubectl apply -f k8s/20-services.yaml
for _ in $(seq 1 60); do
  [ "$(kubectl -n radview get pvc radview-data -o jsonpath='{.status.phase}')" = Bound ] && break
  sleep 3
done
PV=$(kubectl -n radview get pvc radview-data -o jsonpath='{.spec.volumeName}')
VOL=$(kubectl get pv "$PV" -o jsonpath='{.spec.hostPath.path}{.spec.local.path}')
if ! sudo test -f "$VOL/work/yelp/outputs/risk_latest.parquet"; then
  echo "Copying the trained models onto the volume ($VOL)"
  sudo cp -a data/. "$VOL"/
  sudo chown -R 1000:1000 "$VOL"   # the containers run as uid 1000
fi
kubectl -n radview rollout restart deploy/radview-api deploy/radview-dashboard
kubectl -n radview rollout status deploy/radview-api --timeout=300s
kubectl -n radview rollout status deploy/radview-dashboard --timeout=300s

echo "== 6/6 Check"
kubectl -n radview get pods
curl -fs -o /dev/null -w "dashboard health: %{http_code}\n" http://localhost/_stcore/health
kubectl -n radview exec deploy/radview-api -- python -c \
  "import urllib.request as u; print('api:', u.urlopen('http://localhost:8000/health').read().decode()); print('meta:', u.urlopen('http://localhost:8000/meta').read().decode()[:200])"
df -h / | tail -1
echo
echo "Done. The dashboard is on port 80 of the machine (Kubernetes LoadBalancer Service)."
echo "From your laptop: ssh -i <key.pem> -N -L 8501:localhost:80 ubuntu@<machine-address>  then open http://localhost:8501"
