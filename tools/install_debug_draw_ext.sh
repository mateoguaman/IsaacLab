#!/usr/bin/env bash
# Install the isaacsim.util.debug_draw Kit extension into the local uv env.
#
# The pip isaacsim distribution omits extsInternal, so isaacsim.util.debug_draw is
# absent. isaaclab.python.rendering depends on it, which makes any --viz kit run
# (and therefore --video) abort with "Failed to resolve extension dependencies".
# Omniverse Hub would normally fetch it on demand, but it fails to launch here, so
# copy it out of the Docker image instead.
#
# Re-run after rebuilding env_isaaclab; the extension lives inside the env and is
# not tracked by git.
#
# Usage: ./tools/install_debug_draw_ext.sh [image]
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=${1:-isaac-lab-base:latest}
EXT=isaacsim.util.debug_draw
DEST="$PWD/env_isaaclab/lib/python3.12/site-packages/isaacsim/exts"

[ -d "$DEST" ] || { echo "not found: $DEST (is env_isaaclab built?)" >&2; exit 1; }

if [ -f "$DEST/$EXT/config/extension.toml" ]; then
    echo "already installed: $DEST/$EXT"
    exit 0
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "docker image not found: $IMAGE (build it with cluster_build)" >&2; exit 1; }

echo "copying $EXT from $IMAGE ..."
rm -rf "$DEST/$EXT"
# tar preserves the relative symlinks under PACKAGE-LICENSES that `docker cp` rejects.
docker run --rm --entrypoint bash "$IMAGE" \
    -c "cd /isaac-sim/extsInternal && tar -ch $EXT 2>/dev/null" | tar -x -C "$DEST"

if [ -f "$DEST/$EXT/config/extension.toml" ]; then
    echo "installed: $DEST/$EXT ($(du -sh "$DEST/$EXT" | cut -f1))"
else
    echo "copy failed: no config/extension.toml under $DEST/$EXT" >&2
    exit 1
fi
