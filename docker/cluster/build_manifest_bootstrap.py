# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the manifest bootstrap JSON to stdout.

Invoked from ``cluster_interface.sh`` after rsync'ing the code snapshot. The
output is piped via SSH into ``<snapshot_dir>/.manifest.bootstrap.json`` on
the cluster, where ``isaaclab_tasks.utils.write_run_manifest`` reads it on the
compute node to finalize ``manifest.json`` next to the run's artifacts.

Positional args (all required, in order):

  git_sha git_branch git_dirty submitted_at submitted_by
  cluster profile nodes gpus_per_node snapshot_dir persistent_dir
  -- [train_args ...]

``git_dirty`` is the literal string ``"true"`` or ``"false"``. ``nodes`` and
``gpus_per_node`` parse as integers. Everything after ``--`` is the user's
train command argv.
"""

from __future__ import annotations

import json
import sys

_FIXED_KEYS = [
    "git_sha",
    "git_branch",
    "git_dirty",
    "submitted_at",
    "submitted_by",
    "cluster",
    "profile",
    "nodes",
    "gpus_per_node",
    "snapshot_dir",
    "persistent_dir",
]


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print("error: missing -- separator before train_args", file=sys.stderr)
        return 2
    sep = argv.index("--")
    fixed = argv[:sep]
    train_args = argv[sep + 1 :]
    if len(fixed) != len(_FIXED_KEYS):
        print(
            f"error: expected {len(_FIXED_KEYS)} fixed args before --, got {len(fixed)}",
            file=sys.stderr,
        )
        return 2

    fields: dict = dict(zip(_FIXED_KEYS, fixed))
    fields["git_dirty"] = fields["git_dirty"] == "true"
    fields["nodes"] = int(fields["nodes"])
    fields["gpus_per_node"] = int(fields["gpus_per_node"])
    fields["train_args"] = train_args
    json.dump(fields, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
