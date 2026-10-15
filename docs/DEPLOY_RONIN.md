# Deploying Radview on the group's RONIN machine

RONIN (NUS, region ap-southeast-1) gives the group one cloud machine, an AWS EC2 instance. Radview runs on it on
**k3s** — a single-node Kubernetes — using the manifests in [`k8s/`](../k8s/): one **REST API** pod and **two
dashboard** pods, serving the **already-trained models** from a Kubernetes volume. Nothing is retrained on RONIN.

```
your laptop ──(SSH tunnel)──►  RONIN machine (AWS EC2 t3.large, Ubuntu 22.04)
  http://localhost:8501 ─────►   k3s Service radview-dashboard :80 (LoadBalancer)
                                   └─► 2 × dashboard pod :8501 ──► Service radview-api :8000
                                                                    └─► API pod ──► PVC radview-data
                                                                                    (trained models, scores, plans)
```

> **Current state of the shared machine:** this setup is already deployed and starts by itself whenever the
> machine is started. You only need sections 1 and 5 to use it. Use **either** k3s **or** Docker Compose on the
> machine, never both — they would run two copies of the app on 8 GB of memory.

## Before you start — budget

The project dashboard shows a **$100 budget** and a t3.large costs about **$0.11/hour**.
Running 24/7 for the rest of the semester (~43 days × 24 h × $0.11 ≈ $113) would **exceed the budget**,
and storage adds ~$4/month on top. So:

- **Stop the machine when you are not using it** (RONIN *Machines* page → stop), or set an auto-stop /
  schedule. The pods come back by themselves when the machine starts again.
- Start it ~5 minutes before a demo or the Week 12 presentation.
- Do **not** run the training pipeline or download the 6 GB raw Yelp files on RONIN — it is slow on a small
  CPU machine and wastes budget and disk (40 GB, ~7 GB free). Use the trained-models zip.

**Machine size:** at least 2 vCPU and **8 GB RAM** (the API loads DistilBERT) — the group machine is a
**t3.large**. Only switch to a GPU type for the free-LLM experiment below, and switch back afterwards.

---

## 1. Connect to the machine

Use the connection details from the RONIN email/*Machines* page — plain SSH with the group's key file, or
**RONIN LINK** (RONIN's desktop app, which handles the key and port tunnels for you):

```bash
ssh -i ~/path/to/dsa2111-g10-key.pem ubuntu@dsa2111-group10.nus.cloud
```

Everyone logs in as `ubuntu`, so **don't delete other members' folders**, and leave `~/dsa2111_project` alone
unless you are updating the deployment.

## 2. Get the code (first time only)

The repo is private, so authenticate once. Easiest is a personal access token:

```bash
sudo apt-get update && sudo apt-get install -y git unzip
git clone https://github.com/Chef2121/dsa2111_project.git
cd dsa2111_project
```

When `git clone` asks for a password, paste a GitHub **personal access token** (GitHub → Settings → Developer
settings → *Fine-grained tokens*, read-only access to this repo), not your GitHub password.

## 3. Copy the trained-models zip to the machine (first time only)

From **your laptop** (not the RONIN shell):

```bash
scp -i ~/path/to/dsa2111-g10-key.pem Radview-trained-models-yelp.zip ubuntu@dsa2111-group10.nus.cloud:~/
```

It is ~320 MB and contains Yelp review text — keep it on the group's machine and drive only.

## 4. Deploy or update — one command

On the RONIN machine, inside `dsa2111_project`:

```bash
bash scripts/deploy_ronin_k3s.sh
```

The script:

1. installs Docker Engine if it is missing;
2. unzips the trained models into `data/` (first time only);
3. builds the image — **the build runs the test suite**, so a broken version cannot be deployed;
4. installs k3s if it is missing, with `--docker` (pods use the Docker image directly, no second copy) and
   without Traefik (not needed); if a Docker Compose deployment is running it stops it first;
5. applies `k8s/00-namespace-config.yaml` and `k8s/20-services.yaml`, sets `RADVIEW_SOURCE=yelp`, copies the
   trained models onto the `radview-data` volume if it does not have them yet, and restarts the pods;
6. checks the dashboard and the API health and prints the data source (should say `yelp`).

The first build takes ~10–15 minutes (PyTorch and DistilBERT downloads); later builds are cached.
To deploy a newer version later: `git pull && bash scripts/deploy_ronin_k3s.sh`.

The training **Job** and weekly re-scoring **CronJob** in `k8s/10-pipeline-jobs.yaml` are deliberately **not**
applied on RONIN: the Job would retrain everything for ~2 hours and overwrite the trained models, and the
CronJob needs the raw Yelp files.

## 5. Open it from your laptop

The dashboard is exposed on port 80 of the machine only — no public port. Tunnel it through SSH:

```bash
ssh -i ~/path/to/dsa2111-g10-key.pem -N -L 8501:localhost:80 ubuntu@dsa2111-group10.nus.cloud
```

Keep that window open and browse **http://localhost:8501**. For the API's interactive docs, first run
`kubectl -n radview port-forward svc/radview-api 8000:8000` on the machine, add `-L 8000:localhost:8000` to the
tunnel, and open http://localhost:8000/docs.

**Why a tunnel and not a public port:** the dashboard has no login and the data includes Yelp review text.
Opening the ports to the internet would expose both. If the graders must reach it directly, ask the module staff
how RONIN should expose web apps for your project, and restrict access to NUS addresses.

---

## Everyday operations (on the machine)

| Task | Command |
|---|---|
| Status | `kubectl -n radview get pods,svc` |
| Logs | `kubectl -n radview logs deploy/radview-api -f` (or `deploy/radview-dashboard`) |
| Restart the app | `kubectl -n radview rollout restart deploy/radview-api deploy/radview-dashboard` |
| Stop the app (keep the machine on) | `kubectl -n radview scale deploy --all --replicas=0` (undo: `bash scripts/deploy_ronin_k3s.sh`) |
| Update to the latest code | `cd ~/dsa2111_project && git pull && bash scripts/deploy_ronin_k3s.sh` |
| Where the data lives | `kubectl get pv` → the `radview-data` volume under `/var/lib/rancher/k3s/storage/` |
| Enable Claude-written plans | `kubectl -n radview create secret generic radview-secrets --from-literal=ANTHROPIC_API_KEY=sk-ant-... --dry-run=client -o yaml \| kubectl apply -f -` then restart the app |
| Check disk space | `df -h` and `docker system df` (free space with `docker image prune`) |

Re-running the deploy script re-applies `00-namespace-config.yaml`, which resets the `radview-secrets` Secret to
an empty key — set the key again afterwards if you use Claude.

Owner usefulness ratings submitted in the dashboard are saved on the volume under
`work/yelp/outputs/feedback.csv` — copy it back with `sudo cp` + `scp` if you want to report them.

## Free open-source LLM for the M5 agent (GPU, optional)

The agent can run a free open-weight model, **Qwen 2.5 7B**, with [Ollama](https://ollama.com) instead of Claude
(`RADVIEW_LLM_PROVIDER=open`). It needs a GPU to be practical, so the planner comparison in the report was run
like this — and the machine was switched back afterwards:

1. RONIN: stop the machine, change the type to **g4dn.xlarge** (1 × NVIDIA T4, ~$0.74/hour; if AWS reports
   *InsufficientInstanceCapacity*, try again later or another GPU type), start it. The NVIDIA driver is
   already installed.
2. The g4dn's extra 116 GB NVMe drive is temporary (wiped when the machine stops) — use it for the model:
   ```bash
   sudo mkfs.ext4 -q -F /dev/nvme1n1 && sudo mkdir -p /mnt/nvme && sudo mount /dev/nvme1n1 /mnt/nvme
   sudo mkdir -p /mnt/nvme/ollama && sudo chown -R ollama:ollama /mnt/nvme/ollama
   sudo systemctl restart ollama && ollama pull qwen2.5:7b
   ```
   Ollama itself is already installed, with a systemd override that keeps models on `/mnt/nvme/ollama` and sets
   a 16k context (`/etc/systemd/system/ollama.service.d/radview.conf`). On a fresh machine install it with
   `curl -fsSL https://ollama.com/install.sh | sh` and create that override (`OLLAMA_MODELS=/mnt/nvme/ollama`,
   `OLLAMA_CONTEXT_LENGTH=16384`).
3. Run the comparison in a one-off container next to the app (it writes `m5_planner_comparison.json` onto the
   volume, which the dashboard's *Model performance* page shows):
   ```bash
   PV=$(kubectl get pv $(kubectl -n radview get pvc radview-data -o jsonpath={.spec.volumeName}) -o jsonpath={.spec.hostPath.path}{.spec.local.path})
   docker run --rm --network host -v "$PV":/data -e RADVIEW_DATA_DIR=/data -e RADVIEW_SOURCE=yelp \
     -e RADVIEW_LLM_PROVIDER=open radview:latest python -m radview.m5_agent.cli --compare 20
   ```
   About 10 minutes for 20 restaurants on a T4.
4. **Stop the machine and switch it back to t3.large.**

## Alternative: Docker Compose (another machine, or your laptop)

Docker Compose is simpler and fine on a machine **without** k3s: `bash scripts/deploy_ronin.sh` builds the
image and starts the API (port 8000) and dashboard (port 8501) with `docker compose`, restarting them
automatically after a reboot. Tunnel with `-L 8501:localhost:8501 -L 8000:localhost:8000`. Day-to-day:
`docker compose ps`, `docker compose logs -f api dashboard`, `docker compose stop`. **Don't** use it on the group
machine while the k3s deployment is there.

## Troubleshooting

| Problem | Fix |
|---|---|
| `permission denied ... docker.sock` | Log out and back in (docker group), or prefix Docker commands with `sudo`. |
| A pod is `Pending` / `CrashLoopBackOff` | `kubectl -n radview describe pod <name>` and `kubectl -n radview logs <name>`; usually memory (use 8 GB+) or the data volume missing the trained models (re-run the deploy script). |
| Dashboard shows `synthetic` data | The ConfigMap lost `RADVIEW_SOURCE=yelp` or the volume has no `work/yelp/outputs/` — re-run the deploy script. |
| Build fails at the `test` stage | A test failed; run `python -m pytest -q` locally, fix, push, `git pull` on RONIN. |
| Tunnel: `bind: address already in use` | A local dashboard is already on 8501 — stop it, or use `-L 9501:localhost:80` and open :9501. |
| "site cannot be reached" | The SSH tunnel window was closed, or the RONIN machine is stopped. |
| Disk nearly full | `docker image prune -f`, and delete old build cache with `docker builder prune` — never other members' files. |
