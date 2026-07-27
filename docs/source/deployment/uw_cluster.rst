.. _deployment-uw-cluster:


UW Cluster Deployment (Hyak & Tillicum)
=======================================

This guide covers deploying IsaacLab on the University of Washington's two SLURM clusters:
**Tillicum** (``/gpfs``) and **Hyak/Klone** (``/gscratch``). It builds on the generic
:doc:`cluster` guide but handles the UW-specific filesystem layout, multi-cluster switching,
auto-resume on preemption, and hyperparameter sweeps.

.. contents:: Table of Contents
   :local:
   :depth: 2

Quick Start (5 minutes)
-----------------------

**Prerequisites:**

- Docker installed locally (see :doc:`docker`)
- Apptainer installed locally (see :ref:`cluster-setup-instructions` in :doc:`cluster`)
- SSH key-based access to the cluster (see :ref:`uw-ssh-tips` below)

.. note::

   You do NOT need to install IsaacLab manually on the cluster — the Docker/Apptainer
   container you push carries everything needed.

This Quick Start uses the ``cluster_helpers.sh`` shell helpers described in
:ref:`one-time-setup`. Each ``cluster_*`` command transparently stages the
``feature/uw-cluster`` bundle on top of your current branch for the duration of the call,
then resets — so your working branch stays free of cluster infra files.

**Steps:**

1. **Install the cluster helpers** (one time). See :ref:`one-time-setup` below for
   the full install snippet (three lines of ``git show`` + ``source``). This adds
   ``cluster_setup``, ``cluster_push``, ``cluster_submit``, ``cluster_sweep``, and
   ``cluster_collect`` as shell commands.

2. **Create your user config** (one time):

   .. code:: bash

      cluster_setup

   Follow the prompts: UW NetID, Hyak SLURM account, and storage group (press
   ENTER to default the storage group to the SLURM account). This generates
   ``docker/cluster/.env.user`` as an untracked (gitignored) file that persists
   across branch switches and squash-merges. You do **NOT** need to manually
   create directories on the cluster; they will be created automatically the first
   time you run a ``push`` or ``job`` command.

   You can later customize SLURM defaults (partitions, accounts, time limits) or paths
   directly in ``docker/cluster/.env.user``, or re-run ``cluster_setup`` to regenerate it.

3. **Build the Docker image** (one time, or when dependencies change):

   .. code:: bash

      ./docker/container.py start

4. **Push to the cluster** (one time per cluster, or when image changes):

   .. code:: bash

      # Tillicum (default)
      cluster_push

      # Hyak
      cluster_push --cluster hyak

5. **Submit your first job:**

   .. code:: bash

      cluster_submit --task Isaac-Cartpole-v0 --headless

   The first run initializes Isaac Sim caches and may be slow. Subsequent runs are faster.


Submitting Jobs
---------------

All commands run from your **local machine**. The ``cluster_interface.sh`` script handles
code syncing, container management, and job submission over SSH.

The default format is:

   .. code:: bash

      <JOB_FLAGS> ./docker/cluster/cluster_interface.sh --cluster <cluster_name> job <flags>

The ``<JOB_FLAGS>`` placeholder corresponds to flags that determine how many GPUs and nodes to
use, which account, partition (Hyak only) to use, time for the jobs, etc. See :ref:`environment_variable_overrides`
for all options.

Everything after ``job`` corresponds to the same flags you would use when running ``rsl_rl/train.py``, such as
``--task Cartpole-v0`` or ``--headless``.

.. important::

   You can use the ``--cluster`` flag to specify which cluster to use.
   The options are ``hyak`` or ``tillicum``, and ``tillicum`` is the default.

Single GPU
~~~~~~~~~~

.. code:: bash

   ./docker/cluster/cluster_interface.sh job --task Isaac-Cartpole-v0 --headless

Multi-GPU (single node)
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   GPUS_PER_NODE=4 ./docker/cluster/cluster_interface.sh job --task Isaac-Velocity-Rough-Anymal-C-v0 --headless

Multi-node, multi-GPU
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   NODES=2 GPUS_PER_NODE=2 ./docker/cluster/cluster_interface.sh job --task Isaac-Velocity-Rough-Anymal-C-v0 --headless

Hyak-specific overrides
~~~~~~~~~~~~~~~~~~~~~~~

Hyak supports additional SLURM parameters via environment variables or by setting
defaults in ``docker/cluster/.env.user`` (e.g., ``HYAK_SLURM_ACCOUNT``,
``HYAK_STORAGE_GROUP``, ``HYAK_PARTITION``, ``HYAK_SLURM_LOGS_DIR``).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Scenario
     - Command prefix (or set in .env.user)
   * - Weirdlab checkpoint partition
     - ``ACCOUNT=weirdlab-ckpt PARTITION=ckpt-all``
   * - Robotics A40 partition
     - ``ACCOUNT=robotics PARTITION=gpu-a40``
   * - High-memory L40
     - ``PARTITION=gpu-l40 CONSTRAINT=l40 CPUS_PER_TASK=16 MEM_PER_GPU=180G``
   * - Short test (5 min, forces requeue)
     - ``TIME=00:05:00``

Example:

.. code:: bash

   ACCOUNT=weirdlab-ckpt PARTITION=ckpt-all \
       ./docker/cluster/cluster_interface.sh --cluster hyak job --task Isaac-Cartpole-v0 --headless

.. _environment_variable_overrides:

Environment variable overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All overrides are set as environment variable prefixes before the command:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Variable
     - Default
     - Description
   * - ``NODES``
     - 1
     - Number of nodes
   * - ``GPUS_PER_NODE``
     - 1
     - GPUs per node
   * - ``TIME``
     - 24:00:00
     - SLURM time limit
   * - ``ACCOUNT``
     - weirdlab (Hyak only)
     - SLURM account
   * - ``PARTITION``
     - gpu-a40 (Hyak only)
     - SLURM partition
   * - ``CPUS_PER_TASK``
     - 6 (Hyak only)
     - CPUs per task
   * - ``MEM_PER_GPU``
     - 60G (Hyak only)
     - Memory per GPU
   * - ``CONSTRAINT``
     - h200|l40|l40s|a40|a100 (Hyak)
     - GPU type constraint


Hyperparameter Sweeps
---------------------

The ``sweep.py`` utility submits multiple jobs by expanding comma-separated values into
a cartesian product. It automatically generates unique experiment names and run names.

Basic usage
~~~~~~~~~~~

.. code:: bash

   python scripts/reinforcement_learning/rsl_rl/sweep.py \
       --cluster hyak \
       --task Isaac-Cartpole-v0 \
       --num_envs 32,64 \
       --seed 1,2,3

This submits 6 jobs (2 num_envs x 3 seeds). Each job gets an auto-generated ``--run_name``
based on the parameters that vary (e.g., ``num_envs-32_seed-1``).

.. caution::

   Note that all flags, including ``--cluster`` and flags from :ref:`environment_variable_overrides` are
   passed AFTER the ``sweep.py`` for sweeps (unlike when using ``cluster_interface.sh``)

Config overrides
~~~~~~~~~~~~~~~~

Use dotted path syntax for Hydra config overrides:

.. code:: bash

   python scripts/reinforcement_learning/rsl_rl/sweep.py \
       --task Isaac-Velocity-Rough-Anymal-C-v0 \
       --num_envs 4096 \
       --headless \
       agent.max_iterations=500,1000,2000

Hydra presets
~~~~~~~~~~~~~

``presets=`` is the one Hydra key where comma is **not** a sweep separator —
``,`` keeps its native Hydra meaning of "list of presets applied in one combo",
so ``presets=physx,newton`` is a single value passed through to every job.

To sweep across preset alternatives, use ``/`` as the alternatives separator
(no shell quoting needed):

.. code:: bash

   # 2 preset combos x 2 seeds = 4 jobs
   python scripts/reinforcement_learning/rsl_rl/sweep.py \
       --task Isaac-Cartpole-v0 \
       --seed 1,2 \
       presets=physx/physx,newton

Dry run
~~~~~~~

Preview commands without submitting:

.. code:: bash

   python scripts/reinforcement_learning/rsl_rl/sweep.py --dry-run \
       --task Isaac-Cartpole-v0 --seed 1,2,3

Collecting results
~~~~~~~~~~~~~~~~~~

After jobs complete, pull logs back to your local machine:

.. code:: bash

   python scripts/reinforcement_learning/rsl_rl/sweep_collect.py --sweep <SWEEP_ID>

   # With tensorboard
   python scripts/reinforcement_learning/rsl_rl/sweep_collect.py --sweep <SWEEP_ID> --tensorboard

The ``<SWEEP_ID>`` is the timestamp printed when you ran ``sweep.py`` (e.g., ``20260212_111948``).
Sweep manifests are stored in ``logs/sweeps/``.


Docker handoff
--------------

``handoff_build`` packages the current source tree plus a frozen ``train.py``
invocation into a single Docker image, then pushes it. The receiving machine
runs the image with no flags, env vars, or overrides.

.. code:: bash

   handoff_build --image-name <image> --tag <tag> -- <train.py args>

``isaac-lab-base`` is rebuilt from the current working tree first, so every
image reflects that exact snapshot. ``WANDB_API_KEY`` and ``WANDB_USERNAME``
are read from the calling shell and baked into the image as ``ENV``. Add
``--no-push`` for local builds.

On the receiving machine:

.. code:: bash

   docker run --gpus all --network=host --rm <registry>/<image>:<tag>

Inspecting
~~~~~~~~~~

On a normal run, the entrypoint prints a one-line banner with metadata
(git SHA, branch, dirty state, build time, baked-in args) before launching
the simulator. To inspect *without* launching:

.. code:: bash

   # Pre-flight: prints baked-in metadata + train command, then exits.
   docker run --rm <registry>/<image>:<tag> --show

   # Full label set (same metadata, plus the raw train_args string).
   docker inspect <image> --format '{{json .Config.Labels}}' | jq .

   # Full entrypoint script (everything that runs at startup).
   docker run --rm --entrypoint cat <image> /handoff/entrypoint.sh

The ``isaaclab.git_sha`` label cross-references the source repo:
``git show <sha>`` in your IsaacLab clone reveals the exact code inside the
image.

Debugging
~~~~~~~~~

.. code:: bash

   # Drop into a shell, skipping the baked train command.
   docker run --rm -it --entrypoint bash <image>

   # Append extra args; argparse takes the LAST occurrence, so this overrides
   # the baked-in value (useful for shortening test runs):
   docker run --rm <image> --max_iterations 5


Logging & Monitoring
--------------------

Tensorboard
~~~~~~~~~~~

Logs are written to ``logs/rsl_rl/<experiment_name>/`` on the cluster. Use ``sweep_collect.py``
to rsync them locally, then launch tensorboard:

.. code:: bash

   tensorboard --logdir logs/rsl_rl/<experiment_name>

Weights & Biases
~~~~~~~~~~~~~~~~

1. Set your WANDB credentials:

   - **Locally**: export in your shell profile (e.g., ``~/.bashrc``).
     The ``cluster_interface.sh`` script automatically forwards all ``WANDB_*``
     environment variables to the cluster via SSH.

   .. code:: bash

      export WANDB_API_KEY=<your-key>
      export WANDB_USERNAME=<your-username>

2. Submit jobs with ``--logger wandb``:

   .. code:: bash

      ./docker/cluster/cluster_interface.sh job --task Isaac-Cartpole-v0 --headless --logger wandb --log_project_name my-project

3. The environment propagation chain ensures your key reaches the container:
   local ``WANDB_*`` vars -> SSH inline env -> ``sbatch --export=ALL`` -> ``srun`` -> Apptainer (via ``--env`` flags).

4. Verify from your local machine:

   .. code:: bash

      echo "Local: WANDB_API_KEY_len=${#WANDB_API_KEY}"
      ssh <cluster-login> 'echo "Cluster: WANDB_API_KEY_len=${#WANDB_API_KEY}"'

SLURM logs
~~~~~~~~~~

Job stdout/stderr are written to the directory specified by ``SLURM_LOGS_DIR`` in your
``.env.user`` (defaulting to the parent of your IsaacLab directory). Check them with:

.. code:: bash

   # On the cluster
   squeue -u $USER                    # list running jobs
   sacct -j <JOB_ID> --format=...     # detailed job accounting
   cat /path/to/slurm_logs/<name>.out # stdout


Preemption & Auto-Resume
------------------------

The system handles both **checkpoint partition preemption** and **time limit expiration** gracefully.

How it works
~~~~~~~~~~~~

1. **Deterministic directories:** The SLURM Job ID is passed as ``--run_id`` to ``train.py``.
   This creates a log directory like ``logs/rsl_rl/<experiment>/12345/`` instead of a timestamp.
   When the job restarts, it finds the same directory.

2. **Auto-resume:** ``train.py`` checks if the log directory contains ``.pt`` checkpoints.
   If so, it automatically resumes training from the latest one and calculates the remaining
   iterations.

3. **Signal handling:** The SLURM scripts install trap handlers for:

   - ``USR1``: Sent 30-120 seconds before time limit. Triggers ``scontrol requeue``.
   - ``TERM``: Sent during preemption. Also triggers requeue.

   The job continues saving checkpoints until SIGKILL forces termination.

Checkpoint partitions on Hyak
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkpoint partitions (``ckpt-all``) are free but preemptible. Jobs are automatically
requeued and resume from the last checkpoint:

.. code:: bash

   ACCOUNT=weirdlab-ckpt PARTITION=ckpt-all \
       ./docker/cluster/cluster_interface.sh --cluster hyak job --task Isaac-Cartpole-v0 --headless

Killing a job permanently
~~~~~~~~~~~~~~~~~~~~~~~~~

Because jobs auto-requeue on ``SIGTERM``, a normal ``scancel`` will just restart the job.
To permanently stop it:

.. code:: bash

   scancel --signal=KILL <JOB_ID>


Architecture
------------

Workflow diagram
~~~~~~~~~~~~~~~~

.. code-block:: text

   Local Machine                    Cluster Login Node              Compute Node
   ─────────────                    ──────────────────              ────────────
   cluster_interface.sh
     ├─ push: build .sif ────────► store .tar on shared FS
     └─ job:
        ├─ rsync code ───────────► timestamped copy
        └─ ssh submit_job_slurm_*.sh
                                    ├─ generate job.sh
                                    └─ sbatch ──────────────────► run_singularity.sh
                                                                    ├─ copy cache to $TMPDIR
                                                                    ├─ extract .sif
                                                                    ├─ singularity exec
                                                                    │   └─ torch.distributed.run
                                                                    │       └─ train.py
                                                                    └─ rsync cache back

File map
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``docker/cluster/setup.sh``
     - Interactive setup script to generate ``.env.user``.
   * - ``docker/cluster/.env.user.template``
     - User configuration template (tracked).
   * - ``docker/cluster/.env.user``
     - Your personal config (gitignored). Contains all SLURM defaults and paths.
   * - ``docker/cluster/.env.tillicum``
     - Shared Tillicum cluster settings (maps ``.env.user`` vars to generic ``CLUSTER_*`` vars).
   * - ``docker/cluster/.env.hyak``
     - Shared Hyak cluster settings (maps ``.env.user`` vars to generic ``CLUSTER_*`` vars).
   * - ``docker/cluster/cluster_interface.sh``
     - Main entry point. Sources ``.env.user`` + ``.env.<cluster>``, handles push/job commands.
   * - ``docker/cluster/submit_job_slurm_tillicum.sh``
     - Tillicum SLURM submission (generates ``job.sh``, fills placeholders, runs ``sbatch``).
   * - ``docker/cluster/submit_job_slurm_hyak.sh``
     - Hyak SLURM submission (adds account/partition/constraint handling).
   * - ``docker/cluster/run_singularity.sh``
     - Runs on compute node. Manages cache, extracts container, launches training in Apptainer.
   * - ``scripts/.../sweep.py``
     - Hyperparameter sweep launcher (cartesian product of comma-separated values).
   * - ``scripts/.../sweep_collect.py``
     - Collects sweep logs from cluster via rsync.

Code snapshotting
~~~~~~~~~~~~~~~~~

Each ``job`` command creates a timestamped copy of your local code on the cluster
(e.g., ``isaaclab_20260217_143000``). This ensures that:

- Jobs in the queue run the code version from submission time, not the latest.
- Multiple jobs can run different code versions simultaneously.
- Results are reproducible.

Local rsl_rl override
~~~~~~~~~~~~~~~~~~~~~

If you are developing a custom version of ``rsl_rl`` (e.g., a fork with new algorithms),
clone it into the IsaacLab root directory:

.. code:: bash

   cd /path/to/IsaacLab
   git clone https://github.com/<your-user>/rsl_rl.git

When ``isaaclab.sh --install`` runs (during Docker image build, or locally), it detects the
``rsl_rl/`` directory and installs it in editable mode, overriding the PyPI version of
``rsl-rl-lib``.

This works seamlessly with the cluster workflow:

- ``rsl_rl/`` is in ``.gitignore`` (not committed) but **not** in ``.dockerignore``, so both
  the Docker build context and ``cluster_interface.sh``'s rsync include it in the code snapshot.
- The editable install points to ``/workspace/isaaclab/rsl_rl`` inside the container. At job
  time ``/workspace/isaaclab`` is bind-mounted from the rsync'd code snapshot, so any local
  edits you make inside ``rsl_rl/`` are reflected on the next job submission without
  rebuilding the image.

If the ``rsl_rl/`` directory is absent, the standard PyPI ``rsl-rl-lib`` package is used with
no changes to behavior.

Cache management
~~~~~~~~~~~~~~~~

Isaac Sim requires writable cache directories. The ``run_singularity.sh`` script:

1. Pre-creates required cache directories on the shared filesystem (GPFS/gscratch).
2. Copies the cache to the compute node's local ``$TMPDIR`` (fast NVMe storage).
3. Each job gets an isolated directory (``$TMPDIR/job_$SLURM_JOB_ID/``) to prevent
   interference when multiple jobs share a node.
4. After the job, syncs cache changes back to the shared filesystem.


Configuration Reference
-----------------------

.env.user variables
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``CLUSTER_USER``
     - Your UW NetID (used on both clusters).
   * - ``HYAK_SLURM_ACCOUNT``
     - Hyak SLURM ``--account=`` — which group's GPU allocation you use
       (e.g., ``weirdlab``, ``robotics``).
   * - ``HYAK_STORAGE_GROUP``
     - Hyak ``/gscratch`` group for storage. Usually same as
       ``HYAK_SLURM_ACCOUNT``; override for cross-group setups.
   * - ``TILLICUM_LOGIN``
     - Derived: ``${CLUSTER_USER}@tillicum.hyak.uw.edu``
   * - ``TILLICUM_DIR``
     - Derived: ``/gpfs/scrubbed/${CLUSTER_USER}/isaaclab``
   * - ``HYAK_LOGIN``
     - Derived: ``${CLUSTER_USER}@klone1.hyak.uw.edu``
   * - ``HYAK_DIR``
     - Derived: ``/gscratch/${HYAK_STORAGE_GROUP}/${CLUSTER_USER}/isaaclab``
   * - ``HYAK_SLURM_LOGS_DIR``
     - Derived: ``/gscratch/${HYAK_STORAGE_GROUP}/${CLUSTER_USER}/slurm_logs``
   * - ``TILLICUM_SLURM_LOGS_DIR``
     - Derived: ``/gpfs/scrubbed/${CLUSTER_USER}/slurm_logs``

.env.tillicum / .env.hyak shared variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These files map ``.env.user`` paths to generic ``CLUSTER_*`` variables used by the scripts:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``CLUSTER_JOB_SCHEDULER``
     - Always ``SLURM`` for UW clusters.
   * - ``CLUSTER_LOGIN``
     - SSH login (from ``.env.user``).
   * - ``CLUSTER_ISAACLAB_DIR``
     - Base directory for IsaacLab code and logs.
   * - ``CLUSTER_ISAAC_SIM_CACHE_DIR``
     - Persistent Isaac Sim cache directory.
   * - ``CLUSTER_SIF_PATH``
     - Where ``.tar`` container images are stored.
   * - ``CLUSTER_MOUNT_DIR``
     - Filesystem root to bind into container (``/gpfs`` or ``/mmfs1``).
   * - ``REMOVE_CODE_COPY_AFTER_JOB``
     - Whether to delete the timestamped code copy after the job (default: ``false``).
   * - ``CLUSTER_PYTHON_EXECUTABLE``
     - Python script to execute (default: ``scripts/reinforcement_learning/rsl_rl/train.py``).


.. _uw-troubleshooting:

Troubleshooting
---------------

Cache directory issues
~~~~~~~~~~~~~~~~~~~~~~

**Nested ``docker-isaac-sim`` directories:** Older scripts could create recursive cache
directories. The script now warns about this. Fix with:

.. code:: bash

   rm -rf $CLUSTER_ISAAC_SIM_CACHE_DIR/docker-isaac-sim

**First-run initialization:** The first job on a fresh cache may be slow or crash while
Isaac Sim compiles shaders and Warp kernels. Run a simple task first:

.. code:: bash

   ./docker/cluster/cluster_interface.sh job --task Isaac-Cartpole-v0 --num_envs 64 --headless

Container extraction failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If extraction fails with I/O errors, the script retries up to 3 times. Persistent failures
usually indicate GPFS issues. Check with ``df -h`` on the cluster.

Vulkan / VK_LAYER_NV_optimus errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Server GPUs (H200, A100, etc.) can crash with ``VkResult: ERROR_DEVICE_LOST`` due to the
laptop-oriented Optimus Vulkan layer. The ``run_singularity.sh`` script automatically sets:

.. code:: bash

   export VK_LOADER_LAYERS_DISABLE=VK_LAYER_NV_optimus

TMPDIR space issues
~~~~~~~~~~~~~~~~~~~

Each job copies the container image, code, and cache to ``$TMPDIR``. If the node's local
storage is full, jobs will fail. Check with ``df -h $TMPDIR`` inside a job. Consider reducing
``GPUS_PER_NODE`` to run fewer jobs per node.

The container's own ``/tmp`` is also bind-mounted to a job-specific directory on disk
(``$JOB_TMPDIR/container_tmp``) so NVRTC and other in-container JIT compilers don't overflow
the default Apptainer tmpfs. If you see a SIGBUS crash deep inside ``warp.so`` after a Python
traceback, that bind mount has disappeared — check that ``run_singularity.sh`` still includes
the ``-B "$JOB_TMPDIR/container_tmp":/tmp:rw`` line.

Debug mode (CLUSTER_DEBUG=1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``CLUSTER_DEBUG=1`` when submitting to enable extra instrumentation for investigating
hard-to-diagnose crashes:

.. code:: bash

   CLUSTER_DEBUG=1 cluster_submit --cluster hyak --task Isaac-Cartpole-v0 --headless

This enables, on the compute node:

* A container-version probe that prints the baked-in ``warp``, ``torch``, and CUDA runtime
  versions before the exec (helps spot stale ``.sif`` mismatches).
* ``CUDA_LAUNCH_BLOCKING=1`` so CUDA kernels run synchronously and crashes point at the real
  failing launch instead of whichever kernel happened to be reaping queued work.
* ``PYTHONFAULTHANDLER=1`` so SIGBUS/SEGV landing on a Python-managed thread prints a native
  traceback.
* Core-dump rescue into ``$CLUSTER_ISAACLAB_DIR/logs/cores/`` from three possible sources: the
  job-local workspace (``core.*`` files), ``coredumpctl``, and
  ``/var/lib/systemd/coredump/`` (when the systemd-coredump journal is group-restricted).

Debug mode is opt-in; unsetting the variable restores the quieter, faster default.

Warp kernel compilation race conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple processes compile Warp kernels simultaneously on a shared filesystem, race
conditions can cause crashes. The cache-to-TMPDIR strategy prevents this by isolating each
job's cache on local disk. If you still see issues, ensure you ran a single-process warm-up
job first.

W&B authentication failures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If W&B logging fails:

1. Verify the key is visible on the login node:
   ``ssh <login> 'echo ${#WANDB_API_KEY}'`` (should print a number > 0).
2. Check the SLURM ``.out`` file for ``[ERROR] --logger wandb was requested but WANDB_API_KEY is empty``.
3. Check for ``Forwarding environment variables into container: WANDB_API_KEY`` in the output.
4. Ensure your ``~/.bashrc`` exports the key and is sourced for non-interactive SSH sessions.

.. _uw-ssh-tips:

SSH connection tips
~~~~~~~~~~~~~~~~~~~

The ``cluster_interface.sh`` script makes multiple SSH connections per invocation. To avoid
repeated password/2FA prompts, configure SSH connection sharing:

1. Add to ``~/.ssh/config``:

   .. code::

      Host tillicum tillicum.hyak.uw.edu
          HostName tillicum.hyak.uw.edu
          User <your-netid>
          ControlMaster auto
          ControlPath ~/.ssh/sockets/%r@%h-%p
          ControlPersist 8h

      Host klone1 klone1.hyak.uw.edu
          HostName klone1.hyak.uw.edu
          User <your-netid>
          ControlMaster auto
          ControlPath ~/.ssh/sockets/%r@%h-%p
          ControlPersist 8h

2. Create the sockets directory: ``mkdir -p ~/.ssh/sockets``

3. Authenticate once at the start of your day: ``ssh tillicum`` (then ``exit``).
   All subsequent commands reuse the connection for 8 hours.


Adding Cluster Support to your Branch
--------------------------------------

If you want to deploy a personal branch to the cluster without permanently
integrating cluster infrastructure into that branch's history, the recommended
pattern is a **squash-merge-and-reset workflow**:

1. Squash-merge ``feature/uw-cluster`` onto your working branch (stages the cluster
   files without touching history).
2. Commit the bundle as a temporary snapshot so we can unambiguously roll back.
3. Submit the job (``cluster_interface.sh`` rsyncs the working tree, which now
   contains both your code and the cluster infra, up to the cluster).
4. Reset the branch back to its exact pre-bundle commit — the temp commit is
   discarded and the working tree goes back to exactly what it was.

Net effect: your branch history stays free of cluster infra commits, and each
submission automatically picks up whatever is currently on ``feature/uw-cluster`` —
no re-merging, no maintenance.

The helpers below wrap this flow in five convenience commands: ``cluster_setup``,
``cluster_push``, ``cluster_submit``, ``cluster_sweep``, and ``cluster_collect``.

.. _one-time-setup:

One-time setup
~~~~~~~~~~~~~~

1. Make sure ``origin/feature/uw-cluster`` is reachable. If the branch lives on a
   different remote, add it and fetch:

   .. code:: bash

      # Example: add the remote hosting the cluster branch, then fetch it.
      git remote add <name> <url-of-fork>
      git fetch <name> feature/uw-cluster

2. Install the cluster helpers. This pulls ``cluster_helpers.sh`` from the
   ``feature/uw-cluster`` branch (without checking it out) and sources it from your
   shell config on every new shell:

   .. code:: bash

      mkdir -p ~/.local/share
      git show origin/feature/uw-cluster:docker/cluster/cluster_helpers.sh \
          > ~/.local/share/isaaclab_cluster_helpers.sh
      echo 'source ~/.local/share/isaaclab_cluster_helpers.sh' >> ~/.bashrc
      source ~/.bashrc

   .. note::

      The snippet above assumes ``bash``. If you use ``zsh``, replace
      ``~/.bashrc`` with ``~/.zshrc`` — ``cluster_helpers.sh`` uses
      bash-compatible syntax that works in zsh too. For ``fish`` users:
      the helpers are bash-syntax and won't source in fish. Drop into a
      ``bash`` subshell when you want to use them.

3. Verify the helpers are loaded:

   .. code:: bash

      cluster_help

   If the bundle lives on a different remote than ``origin``, override the ref
   (in your shell, or export it from ``~/.bashrc``):

   .. code:: bash

      export CLUSTER_BUNDLE_REF=myfork/feature/uw-cluster

4. Generate your user config (once per machine):

   .. code:: bash

      cluster_setup

   ``cluster_setup`` transiently applies the ``feature/uw-cluster`` bundle, runs
   ``docker/cluster/setup.sh`` to prompt for your UW NetID, SLURM account, and
   storage group, and
   resets. The generated ``docker/cluster/.env.user`` is gitignored and
   untracked, so it **persists** across the reset and across branch switches.
   Re-run ``cluster_setup`` at any time to regenerate or update it.

Day-to-day usage
~~~~~~~~~~~~~~~~

From inside your IsaacLab working tree (any branch you're developing on):

.. code:: bash

   # Submit a single job
   cluster_submit --task Isaac-Cartpole-v0 --headless

   # Target a specific cluster (default: tillicum)
   cluster_submit --cluster hyak --task Isaac-Cartpole-v0 --headless

   # Submit a sweep
   cluster_sweep --task Isaac-Cartpole-v0 --seed 1,2,3

   # Pull sweep logs back after jobs complete
   cluster_collect --sweep <SWEEP_ID>

Each call transparently squashes ``feature/uw-cluster`` onto a temporary commit, runs
the command, and resets. Your branch state afterwards is identical to before
the call.

.. important::

   When you have uncommitted changes, the helpers prompt before doing anything:

   - **(c) commit** — bail out so you can ``git commit`` first, then re-run.
     Use this when your edits are real changes you want to deploy.
   - **(s) stash** — auto-stash the changes, run the submission against your
     last committed state, then auto-pop the stash back into your working tree
     when the helper finishes. Use this when your edits are unrelated WIP that
     shouldn't go to the cluster.
   - **(a) abort** — exit without doing anything.

   Stashing is fully automatic — you never need to manually run
   ``git stash`` or ``git stash pop``. If the squash merge later fails (e.g.
   because of a conflict), the helper restores your working tree to its
   pre-submit state, including your stashed WIP, before bailing out.

Updating the helpers
~~~~~~~~~~~~~~~~~~~~

When ``cluster_helpers.sh`` on ``feature/uw-cluster`` changes (new safety checks, new
commands, bug fixes), re-pull it with:

.. code:: bash

   cluster_update_helpers

from inside any IsaacLab working tree. This rewrites
``~/.local/share/isaaclab_cluster_helpers.sh`` in place. Open a new shell, or
re-source the file, to pick up the changes.

Handling squash-merge conflicts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few files on ``feature/uw-cluster`` overlap with commonly-edited files on working
branches:

- ``scripts/reinforcement_learning/rsl_rl/train.py`` (``--run_id`` +
  auto-resume additions)
- ``source/isaaclab/isaaclab/cli/commands/install.py`` (local ``rsl_rl``
  override)
- ``.gitignore``

If your branch has independent edits to any of those files, the squash merge
will conflict. The helpers abort cleanly in that case: the half-applied merge
is reverted, any stashed WIP is popped back, and your working tree is exactly
what it was before you ran the helper.

The recommended fix is to move cluster-related changes onto ``feature/uw-cluster``
itself, so they live in the bundle for everyone. If you genuinely need a
one-off manual resolution, see ``cluster_helpers.sh`` for the underlying
``git merge --squash`` / ``git reset --hard`` sequence and adapt it inline.

How the workflow works internally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For reference, each helper runs this sequence:

.. code:: bash

   # If the working tree has uncommitted changes, prompt the user. On 's'
   # (stash), save them; on 'c' (commit), bail and ask the user to commit
   # first; on 'a' (abort), exit.
   git stash push -m "cluster_helpers/<timestamp>"  # only if user picked 's'

   prev_sha=$(git rev-parse HEAD)
   git merge --squash <bundle_ref>      # Stage cluster files without committing.
   git commit -m "temp: cluster bundle" # Snapshot the combined state.
   <actual command>                     # Submit, sweep, or collect — rsync sees
                                        # both your code and the cluster infra.
   git reset --hard "$prev_sha"         # Erase the temp commit; working tree
                                        # returns to its pre-bundle state.
   git stash pop                        # only if we stashed earlier

The temp commit lives only for the duration of the command; once
``reset --hard`` runs it becomes unreachable and is eventually garbage-collected
by Git. Nothing about your branch's published history ever changes.
