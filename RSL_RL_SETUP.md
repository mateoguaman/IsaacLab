# Setting up UW-Lab's rsl_rl fork

UW-Lab maintains a custom fork of `rsl_rl` at
[UW-Lab/rsl_rl](https://github.com/UW-Lab/rsl_rl). The
[`feature/locomotion`](https://github.com/UW-Lab/rsl_rl/tree/feature/locomotion)
branch carries our additions: wandb requeue resume, curriculum
save/load, per-tensor distributed broadcast, and a defensive fix to
`resolve_callable`. Vendored upstream is on
[`vendor/leggedrobotics`](https://github.com/UW-Lab/rsl_rl/tree/vendor/leggedrobotics).

This guide is for someone using the IsaacLab repo who wants those changes
active locally and on the cluster.

## One-time setup

```bash
cd /path/to/IsaacLab
git clone -b feature/locomotion git@github.com:UW-Lab/rsl_rl.git
```

That's it for the source. The directory is gitignored at the IsaacLab repo
root and ignored from the docker build context only as far as `.gitignore`
goes — `.dockerignore` lets it through.

## Local development

**Option A: pip directly (any branch)**:

```bash
source env_isaaclab/bin/activate
pip install -e ./rsl_rl --config-settings editable_mode=compat
```

The `editable_mode=compat` flag is required, since the default strict mode breaks rsl_rl installation.

**Option B: via the install hook (run from `feature/uw-cluster`)**:

```bash
git checkout feature/uw-cluster
./isaaclab.sh --install rsl_rl
git checkout -   # back to your working branch
```

Verify either way:

```bash
./isaaclab.sh -p -c "import rsl_rl.utils.wandb_utils as w; print(w.__file__)"
# should print: /path/to/IsaacLab/rsl_rl/rsl_rl/utils/wandb_utils.py
```

## Cluster builds (Hyak / Tillicum)

The Dockerfile has a conditional `RUN` that installs local rsl_rl, but this lives on
`feature/uw-cluster`. So image builds **must** run either from that branch or
with `feature/uw-cluster` squash-merged onto your working branch:

**Option A: use the cluster helper**:

If you haven't set up the UW cluster yet, follow [UW_CLUSTER.md](https://github.com/UW-Lab/IsaacLab/blob/feature/uw-cluster/UW_CLUSTER.md).

If you have already set this up, but want to update your docker build so that it now includes
your local version of `rsl_rl`, run from your development branch (e.g. `feature/locomotion`):

```bash
cluster_update_helpers
cluster_build                        # local build, cluster-agnostic
cluster_push --cluster tillicum      # or --cluster hyak
```

**Option B: Build manually**:

```bash
git checkout feature/uw-cluster
./docker/container.py start          # builds the image with local rsl_rl baked in
cluster_push --cluster tillicum      # or --cluster hyak
git checkout -                       # back to your working branch
```

Once the image is on the cluster, day-to-day submission goes through
`cluster_submit` from any working branch — the helpers handle the
`feature/uw-cluster` bundle transparently.

## Pulling updates to the rsl_rl fork

When `UW-Lab/rsl_rl/feature/locomotion` gets new commits, just pull inside
the local checkout:

```bash
cd rsl_rl
git pull origin feature/locomotion
```

The editable install picks up changes immediately for local and cluster runs. 

## Falling back to PyPI rsl-rl-lib

pip uninstalls the wheel when you `pip install -e ./rsl_rl`. So just removing the `rsl_rl/`
directory leaves you with no rsl_rl at all (imports fail).

To switch back to stock PyPI rsl-rl-lib, force-reinstall the wheel:

```bash
source env_isaaclab/bin/activate
pip install --force-reinstall --no-deps rsl-rl-lib
```

If you also want the cluster image to fall back to PyPI, delete `IsaacLab/rsl_rl/`, then run `cluster_build` followed by `cluster_push`.
