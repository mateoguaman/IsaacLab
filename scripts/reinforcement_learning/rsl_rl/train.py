# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

import warnings

warnings.warn(
    "scripts/reinforcement_learning/rsl_rl/train.py is deprecated. Use "
    "`./isaaclab.sh train --rl_library rsl_rl --task <TASK>` instead. "
    "Example: `./isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole`.",
    DeprecationWarning,
    stacklevel=1,
)

import argparse
import contextlib
import importlib.metadata as metadata
import logging
import os
import platform
import sys
import time
from datetime import datetime

os.environ["WANDB_API_KEY"] = "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"
os.environ["WANDB_USERNAME"] = "uw-lab"
os.environ["WANDB_ENTITY"] = "uw-lab"

# Convert NCCL hangs into timeouts so SLURM auto-requeue can fire on multi-GPU/-node jobs.
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

import gymnasium as gym
import torch
from packaging import version

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli, write_run_manifest
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

RSL_RL_VERSION = "5.4.1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
# Enable cuDNN autotune: on the first few calls cuDNN measures algorithm options for
# each conv shape and caches the fastest. PPO repeats the same conv shapes 20+ times
# per iteration, so the warmup cost is amortized immediately and convs typically run
# 20-40% faster after the first iteration. Profiling on the position-CNN configuration
# showed conv backward dominating Learning time at ~33% of total update; this flag is
# the highest-ROI conv-side speedup available.
torch.backends.cudnn.benchmark = True

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help=(
        "Unique run identifier (e.g., SLURM_JOB_ID). If provided, uses this as the log directory name "
        "instead of a timestamp, enabling automatic resumption when a cluster job is requeued."
    ),
)
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# Snapshot the original CLI args for the run manifest before Hydra clobbers sys.argv.
original_train_args = sys.argv[1:]

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function. Both sides of this
# intersection share the same token vocabulary (the callback reads the user's
# original sys.argv), so preset tokens like ``physics=NAME`` compare correctly.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# -- check RSL-RL version ----------------------------------------------------
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        # For distributed training, launch_simulation() already resolved the
        # correct per-rank device; only apply a CLI --device override for
        # non-distributed runs (the default "cuda:0" would clobber the
        # per-rank device otherwise).
        if not args_cli.distributed:
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        # check for invalid combination of CPU device with distributed training
        if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
            raise ValueError(
                "Distributed training is not supported when using CPU device. "
                "Please use GPU device (e.g., --device cuda) for distributed training."
            )

        # multi-gpu training configuration
        if args_cli.distributed:
            global_rank = int(os.getenv("RANK", "0"))
            # env_cfg.sim.device is resolved by launch_simulation() which
            # accounts for CUDA_VISIBLE_DEVICES restrictions.
            agent_cfg.device = env_cfg.sim.device

            # use global rank for seed diversity across all nodes
            seed = agent_cfg.seed + global_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # Gate expensive or duplicate work (config dumps, stdout prints) behind rank 0.
        is_main_process = not args_cli.distributed or int(os.getenv("RANK", "0")) == 0

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        if is_main_process:
            print(f"[INFO] Logging experiment in directory: {log_root_path}")

        # Build the per-run log directory. With --run_id (e.g., SLURM_JOB_ID) we use a
        # deterministic name so that a cluster requeue writes to the same directory and
        # picks up the previous attempt's checkpoints. Without --run_id we keep the
        # timestamp-based layout used for local/interactive runs.
        if args_cli.run_id:
            log_dir = args_cli.run_id
        else:
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
        # change it (see PR #2346, comment-2819298849)
        if is_main_process:
            print(f"Exact experiment name requested from command line: {log_dir}")

        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        if is_main_process:
            write_run_manifest(
                log_dir,
                args_cli.task,
                agent_cfg.experiment_name,
                train_args=original_train_args,
            )

        # Auto-resume: if --run_id points to a directory that already contains .pt
        # checkpoints (e.g., after SLURM preemption or a time-limit requeue), enable
        # resume so the next attempt picks up where the last one left off.
        if args_cli.run_id and os.path.exists(log_dir):
            checkpoint_files = [f for f in os.listdir(log_dir) if f.endswith(".pt")]
            if checkpoint_files:
                if is_main_process:
                    print(f"[INFO] Found existing run with {len(checkpoint_files)} checkpoint(s). Auto-resuming...")
                agent_cfg.resume = True
                agent_cfg.load_run = os.path.basename(log_dir)

        # set the IO descriptors export flag if requested
        if isinstance(env_cfg, ManagerBasedRLEnvCfg):
            env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        else:
            logger.warning(
                "IO descriptors are only supported for manager based RL environments."
                " No IO descriptors will be exported."
            )

        # set the log directory for the environment (works for all environment types)
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # save resume path before creating a new log_dir
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # create runner from rsl-rl
        runner_cfg = agent_cfg.to_dict()
        # Forward --run_id so the wandb writer can use it as a deterministic run id;
        # requeued cluster jobs then resume the same wandb run instead of creating a new one.
        # Guard against clobbering a run_id already set by --wandb_run_id (which routes
        # through update_rsl_rl_cfg and lands on agent_cfg.run_id before to_dict).
        if args_cli.run_id and not runner_cfg.get("run_id"):
            runner_cfg["run_id"] = args_cli.run_id
        runner = agent_cfg.class_type(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
        # write git state to logs
        runner.add_git_repo_to_log(__file__)
        # load the checkpoint
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)

        # Calculate the number of learning iterations to run. runner.learn() runs
        # num_learning_iterations starting from runner.current_learning_iteration,
        # so when resuming we must subtract what was already completed.
        iterations_to_run = agent_cfg.max_iterations
        if agent_cfg.resume and hasattr(runner, "current_learning_iteration"):
            iterations_to_run = agent_cfg.max_iterations - runner.current_learning_iteration
            if is_main_process:
                print(
                    f"[INFO] Resuming from iteration {runner.current_learning_iteration}."
                    f" Remaining iterations: {iterations_to_run}"
                )
            if iterations_to_run <= 0:
                if is_main_process:
                    print(
                        f"[INFO] Training already completed (Current:"
                        f" {runner.current_learning_iteration} >= Max: {agent_cfg.max_iterations})."
                        f" Exiting."
                    )
                exit(0)

        # dump the configuration into log-directory (rank 0 only to avoid all ranks
        # racing to write the same files in a distributed run)
        if is_main_process:
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
            # Forward IsaacLab artifacts to rsl_rl so wandb/neptune upload them when
            # the logging writer initializes inside runner.learn(). Gated on
            # add_file_to_log presence so train.py still runs against rsl-rl-lib
            # releases that predate this hook.
            if hasattr(runner, "add_file_to_log"):
                runner.add_file_to_log(os.path.join(log_dir, "manifest.json"))
                runner.add_file_to_log(os.path.join(log_dir, "params", "env.yaml"))
                runner.add_file_to_log(os.path.join(log_dir, "params", "agent.yaml"))

        # run training
        try:
            runner.learn(num_learning_iterations=iterations_to_run, init_at_random_ep_len=True)
            print(f"Training time: {round(time.time() - start_time, 2)} seconds")
            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
