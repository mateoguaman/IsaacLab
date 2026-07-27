# UW Cluster Deployment

This branch (`feature/uw-cluster`) adds University of Washington Hyak/Klone + Tillicum
cluster deployment on top of IsaacLab.

It is designed to be **squash-merged transiently** onto your working branch at
job-submission time, not permanently integrated — so your branch history stays
free of cluster infrastructure files, and upgrades to `feature/uw-cluster` propagate
automatically on the next submission.

> **Version requirements:** this branch targets **IsaacLab 3.0** (branched off
> `upstream/develop`) and assumes the Docker image is built on top of
> **Isaac Sim 6.0**. Older IsaacLab 2.x checkouts and older Isaac Sim images
> are not supported — `train.py` auto-resume, the `run_singularity.sh` bind
> mounts, and several other pieces depend on 3.0/6.0 structure.

## Quick install

Run these from the root of your `UW-Lab/IsaacLab` clone. The canonical home
of `feature/uw-cluster` is `UW-Lab/IsaacLab` itself — your `origin` after a
fresh clone — so the helpers' default `origin/feature/uw-cluster` ref just
works.

```bash
# 1. Make sure feature/uw-cluster is fetched (no-op right after a clone).
git fetch origin feature/uw-cluster

# 2. Install the cluster helpers (sourced from your shell config every shell).
mkdir -p ~/.local/share
git show origin/feature/uw-cluster:docker/cluster/cluster_helpers.sh \
    > ~/.local/share/isaaclab_cluster_helpers.sh
echo 'source ~/.local/share/isaaclab_cluster_helpers.sh' >> ~/.bashrc
source ~/.bashrc

# 3. Generate your user config (prompts for UW NetID + SLURM account + storage).
cluster_setup
```

### 3.5 (optional) Use a custom rsl_rl fork

If you want to use a fork of rsl_rl (e.g.,
[UW-Lab/rsl_rl](https://github.com/UW-Lab/rsl_rl)), clone it into the IsaacLab
repo root **before** running `cluster_build`. The Docker build detects the
local clone and installs it in editable mode, so any edits you make in
`rsl_rl/` propagate to the cluster on the next `cluster_submit`.

```bash
git clone -b feature/locomotion git@github.com:UW-Lab/rsl_rl.git
```

Skipping this step gives you stock `rsl-rl-lib` from PyPI in the container.

```bash
# 4. Build the Docker image locally and push it to the cluster.
cluster_build                        # local build; cluster-agnostic
cluster_push --cluster hyak          # or omit --cluster for tillicum (default)

# 5. Submit your first job.
cluster_submit --cluster hyak --task Isaac-Cartpole-v0 --headless \
    --num_envs 64 --max_iterations 20
```

Cloned a personal fork instead of `UW-Lab/IsaacLab`? Replace
`origin/feature/uw-cluster` with `<your-remote>/feature/uw-cluster` in step 2's
`git show`, and add a matching `CLUSTER_BUNDLE_REF` export so the helpers use
the same ref at call time:

```bash
echo 'export CLUSTER_BUNDLE_REF=<your-remote>/feature/uw-cluster' >> ~/.bashrc
```

The helpers resolve `CLUSTER_BUNDLE_REF` at call time, so the export line can
appear before or after the `source` in `.bashrc`.

Not using bash? Swap `~/.bashrc` for `~/.zshrc` — the helpers are bash-syntax
but work under zsh. Fish users should drop into a `bash` subshell when running
`cluster_*` commands.

## Helpers

After install, run `cluster_help` for a reminder of the available commands:

| Command | Purpose |
|---|---|
| `cluster_setup` | First-time config (generates `docker/cluster/.env.user`, gitignored so it persists) |
| `cluster_build` | Build the docker image locally (includes local build of rsl_rl if present) |
| `cluster_push` | Push the Singularity image to the cluster |
| `cluster_submit` | Submit a single training job |
| `cluster_sweep` | Submit a hyperparameter sweep |
| `cluster_collect` | Pull sweep logs back to your local machine |
| `handoff_build` | Package the current source + a frozen train command into a Docker image and push it — see [Docker handoff](#docker-handoff) |
| `cluster_update_helpers` | Re-pull the latest `cluster_helpers.sh` |

Each `cluster_*` and `handoff_*` command transparently applies + reverts the
`feature/uw-cluster` bundle, so your working branch stays unchanged after
every call.

## Docker handoff

`handoff_build` packages the current source tree plus a frozen `train.py`
invocation into a single Docker image, then pushes it. The receiving machine
runs the image with no flags, env vars, or overrides.

```bash
handoff_build --image-name <image> --tag <tag> -- <train.py args>
```

`isaac-lab-base` is rebuilt from the current working tree first, so every
image reflects that exact snapshot. `WANDB_API_KEY` and `WANDB_USERNAME` are
read from the calling shell and baked into the image as `ENV`. Add
`--no-push` for local builds.

On the receiving machine:

```bash
docker run --gpus all --network=host --rm <registry>/<image>:<tag>
```

### Inspecting

On a normal run, the entrypoint prints a one-line banner with metadata
(git SHA, branch, dirty state, build time, baked-in args) before launching
the simulator. To inspect *without* launching:

```bash
# Pre-flight: prints baked-in metadata + train command, then exits.
docker run --rm <registry>/<image>:<tag> --show

# Full label set (same metadata, plus the raw train_args string).
docker inspect <image> --format '{{json .Config.Labels}}' | jq .

# Full entrypoint script (everything that runs at startup).
docker run --rm --entrypoint cat <image> /handoff/entrypoint.sh
```

The `isaaclab.git_sha` label cross-references the source repo:
`git show <sha>` in your IsaacLab clone reveals the exact code inside the
image.

### Debugging

```bash
# Drop into a shell, skipping the baked train command.
docker run --rm -it --entrypoint bash <image>

# Append extra args; argparse takes the LAST occurrence, so this overrides
# the baked-in value (useful for shortening test runs):
docker run --rm <image> --max_iterations 5
```

## Re-adding a failed job

Preemption and time-limit hits auto-resume. For a crash that needs manual
re-submit, use `cluster_submit` with the original `--run_id`,
`--experiment_name`, and `--run_name`:

```bash
cluster_submit --cluster tillicum --run_id <orig_slurm_id> \
    --experiment_name <orig> --run_name <orig> \
    <original training args>
```

`cluster_sweep` rewrites `--experiment_name` and errors on `--run_id` to redirect you here.

## Full docs

- **[`docs/source/deployment/uw_cluster.rst`](docs/source/deployment/uw_cluster.rst)** —
  complete deployment guide: multi-cluster dispatch, auto-resume on preemption,
  hyperparameter sweeps, Weights & Biases propagation, architecture reference,
  and troubleshooting.
- **[`docs/source/deployment/cluster.rst`](docs/source/deployment/cluster.rst)** —
  local prerequisites (`apptainer` install, SSH setup) shared with the upstream
  IsaacLab cluster workflow.
