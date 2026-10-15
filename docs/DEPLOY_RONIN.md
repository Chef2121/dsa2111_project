# Deploying Radview on the group's RONIN machine

RONIN (NUS, region ap-southeast-1) gives the group one cloud machine. Radview runs on it as two Docker
containers — the **REST API** and the **dashboard** — serving the **already-trained models**. Nothing is
retrained on RONIN.

```
your laptop ──(RONIN LINK / SSH tunnel)──►  RONIN machine
  http://localhost:8501  ─────────────────►   dashboard container :8501 ──► API container :8000
  http://localhost:8000/docs ─────────────►   API container :8000 ──► ./data (trained models, scores, plans)
```

## Before you start — budget

The project dashboard shows a **$100 budget** and the machine costs about **$0.11/hour**.
Running 24/7 for the rest of the semester (~43 days × 24 h × $0.11 ≈ $113) would **exceed the budget**,
and storage adds ~$4/month on top. So:

- **Stop the machine when you are not using it** (RONIN *Machines* page → stop), or set an auto-stop /
  schedule. The containers restart by themselves when the machine starts again.
- Start it ~5 minutes before a demo or the Week 12 presentation.
- Do **not** run the training pipeline or download the 6 GB raw Yelp files on RONIN — it is slow on a small
  CPU machine and wastes budget and disk (40 GB). Use the trained-models zip.

**Machine size:** at least 2 vCPU and **8 GB RAM** (the API loads DistilBERT); Ubuntu 22.04/24.04 assumed.
About 10 GB of the disk is used by Docker images and the data.

---

## 1. Connect to the machine

Use the connection details on the RONIN *Machines* page — either **RONIN LINK** (RONIN's desktop app, which
handles the SSH key and port tunnels for you) or plain SSH with the key file RONIN provides:

```bash
ssh -i ~/path/to/ronin-key.pem ubuntu@<machine-address>
```

(The user name may differ — use the one RONIN shows.) All commands in steps 2–4 run **on the RONIN machine**.

## 2. Get the code

The repo is private, so authenticate once. Easiest is the GitHub CLI or a personal access token:

```bash
sudo apt-get update && sudo apt-get install -y git unzip
git clone https://github.com/Chef2121/dsa2111_project.git
cd dsa2111_project
```

When `git clone` asks for a password, paste a GitHub **personal access token** (GitHub → Settings → Developer
settings → *Fine-grained tokens*, read-only access to this repo), not your GitHub password.

## 3. Copy the trained-models zip to the machine

From **your laptop** (not the RONIN shell), upload the zip to the machine's home folder:

```bash
scp -i ~/path/to/ronin-key.pem Radview-trained-models-yelp.zip ubuntu@<machine-address>:~/
```

(RONIN LINK can also transfer files.) It is ~320 MB and contains Yelp review text — keep it on the group's
machine and drive only.

## 4. Deploy — one command

Back on the RONIN machine, inside `dsa2111_project`:

```bash
bash scripts/deploy_ronin.sh
```

The script:

1. installs Docker Engine + the compose plugin if they are missing;
2. unzips the trained models into `data/` (and makes it writable for the containers);
3. builds the image — **the build runs the test suite**, so a broken version cannot be deployed;
4. starts the API and dashboard with `docker compose` (`restart: unless-stopped`);
5. waits for the API health check and prints the data source (should say `yelp`).

The first build takes ~10–15 minutes (it downloads PyTorch and DistilBERT); later builds are cached.
If Docker was just installed, log out and back in once so `docker` works without `sudo`.

## 5. Open it from your laptop

The services listen on the machine only. Forward the two ports through SSH (or set up the same two port
tunnels in RONIN LINK):

```bash
ssh -i ~/path/to/ronin-key.pem -N -L 8501:localhost:8501 -L 8000:localhost:8000 ubuntu@<machine-address>
```

Keep that window open, then browse:

| What | URL on your laptop |
|---|---|
| Dashboard | http://localhost:8501 |
| API + interactive docs | http://localhost:8000/docs |

**Why a tunnel and not a public port:** the dashboard has no login and the data includes Yelp review text.
Opening ports 8501/8000 to the internet would expose both. If the graders must reach it directly, ask the
module staff how RONIN should expose web apps for your project, and restrict access to NUS addresses.

---

## Everyday operations (on the machine, in `dsa2111_project`)

| Task | Command |
|---|---|
| Status | `docker compose ps` |
| Logs | `docker compose logs -f api dashboard` |
| Stop the services | `docker compose stop` |
| Start them again | `docker compose up -d api dashboard` |
| Update to the latest code | `git pull && docker compose up -d --build api dashboard` |
| Enable Claude-written plans | `export ANTHROPIC_API_KEY=sk-ant-...` then `docker compose up -d api dashboard` |
| Weekly re-score of new data | `docker compose --profile jobs run --rm pipeline python -m radview.pipeline --mode rescore` (needs raw data in `data/yelp/`) |
| Check disk space | `df -h` and `docker system df` (free space with `docker system prune`) |

Owner usefulness ratings submitted in the dashboard are saved to `data/work/yelp/outputs/feedback.csv` on the
machine — copy it back with `scp` if you want to report them.

## Optional: run it on Kubernetes on the same machine

The `k8s/` manifests also work on a single-node **k3s** cluster, if you want to demonstrate Kubernetes on RONIN
(Docker Compose is simpler and enough for the demo):

```bash
curl -sfL https://get.k3s.io | sh -                       # single-node Kubernetes
docker save radview:latest | sudo k3s ctr images import -  # make the image available to k3s
sudo k3s kubectl apply -f k8s/00-namespace-config.yaml
sudo k3s kubectl -n radview patch configmap radview-config --type merge -p '{"data":{"RADVIEW_SOURCE":"yelp"}}'
sudo k3s kubectl apply -f k8s/20-services.yaml
# copy the trained models onto the volume through the API pod, then restart it:
POD=$(sudo k3s kubectl -n radview get pod -l role=api -o name | head -1)
sudo k3s kubectl -n radview cp data/work "${POD#pod/}:/data/"
sudo k3s kubectl -n radview rollout restart deploy/radview-api
sudo k3s kubectl -n radview port-forward svc/radview-dashboard 8501:80
```

Don't run Docker Compose and k3s at the same time on an 8 GB machine.

## Troubleshooting

| Problem | Fix |
|---|---|
| `permission denied ... docker.sock` | Log out and back in (docker group), or prefix commands with `sudo`. |
| Dashboard shows `synthetic` data | `data/work/yelp/outputs/` is missing — re-run step 3 and the deploy script. |
| Build fails at the `test` stage | A test failed; run `python -m pytest -q` locally, fix, push, `git pull` on RONIN. |
| Container keeps restarting / killed | Out of memory — use an 8 GB+ machine and don't run k3s alongside Compose. |
| Tunnel: `bind: address already in use` | A local dashboard is already on 8501 — stop it, or use `-L 9501:localhost:8501` and open :9501. |
| "site cannot be reached" | The SSH tunnel window was closed, or the RONIN machine is stopped. |
