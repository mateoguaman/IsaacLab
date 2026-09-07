# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch robot description bundles that are too large to track in git.

Downloads and extracts URDF + mesh bundles into
:data:`~isaaclab_assets.ISAACLAB_ASSETS_DATA_DIR`. Each bundle is skipped when
its marker file already exists, so re-running is cheap.

Run::

    ./isaaclab.sh -p scripts/fetch_robot_descriptions.py
    ./isaaclab.sh -p scripts/fetch_robot_descriptions.py --bundle unitree_description --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Bundle:
    """One downloadable robot-description archive.

    Args:
        name: Directory name created under the assets data dir.
        url: Source archive URL.
        marker: Path relative to :attr:`name` proving a complete extraction.
        note: Provenance shown to the user.
    """

    name: str
    url: str
    marker: str
    note: str


BUNDLES: tuple[Bundle, ...] = (
    Bundle(
        name="unitree_description",
        url="https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz",
        marker="urdf/g1/main.urdf",
        note=(
            "Unitree G1 (29 DoF) URDF + meshes, as distributed by the BeyondMimic / whole_body_tracking"
            " project (github.com/HybridRobotics/whole_body_tracking)."
        ),
    ),
)


def _assets_data_dir() -> Path:
    """Return the isaaclab_assets data directory."""
    from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

    return Path(ISAACLAB_ASSETS_DATA_DIR)


def fetch(bundle: Bundle, dest_root: Path, force: bool = False) -> Path:
    """Download and extract one bundle, returning its directory.

    Args:
        bundle: Archive to fetch.
        dest_root: Assets data directory the bundle is extracted into.
        force: Re-download even when the marker file is already present.

    Returns:
        Path to the extracted bundle directory.
    """
    dest = dest_root / bundle.name
    if (dest / bundle.marker).exists() and not force:
        print(f"[fetch] {bundle.name}: already present at {dest}")
        return dest

    if dest.exists():
        shutil.rmtree(dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] {bundle.name}: downloading {bundle.url}")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as handle:
        archive = Path(handle.name)
    try:
        urllib.request.urlretrieve(bundle.url, archive)
        print(f"[fetch] {bundle.name}: extracting into {dest_root}")
        with tarfile.open(archive, "r:gz") as tar:
            # ``filter="data"`` rejects absolute paths, parent-dir escapes, and
            # special files -- the archive is third-party, so extraction is
            # constrained to plain files under ``dest_root``.
            tar.extractall(dest_root, filter="data")
    finally:
        archive.unlink(missing_ok=True)

    if not (dest / bundle.marker).exists():
        raise RuntimeError(f"{bundle.name}: extraction finished but {bundle.marker} is missing under {dest}")
    print(f"[fetch] {bundle.name}: ready at {dest}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=str, default=None, help="Fetch only this bundle (default: all).")
    parser.add_argument("--force", action="store_true", help="Re-download even if already present.")
    args = parser.parse_args()

    selected = [b for b in BUNDLES if args.bundle in (None, b.name)]
    if not selected:
        print(f"Unknown bundle {args.bundle!r}. Available: {[b.name for b in BUNDLES]}", file=sys.stderr)
        return 1

    dest_root = _assets_data_dir()
    for bundle in selected:
        print(f"[fetch] {bundle.name}: {bundle.note}")
        fetch(bundle, dest_root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
