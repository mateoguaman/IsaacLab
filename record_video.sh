#!/usr/bin/env bash
# Record policy rollout videos from local or Hyak training runs.
#
# Rendering always happens locally. Hyak checkpoints are rsynced into
# logs/rsl_rl/<experiment>/<run>/ first, so a cluster run can be sampled while it
# is still training.
#
# Runs play.py directly rather than via `isaaclab.sh play`: the latter dispatches to
# play_rsl_rl.py. Both guard the JIT export now, but play.py is the path this is
# tested against.
#
# Usage:
#   ./record_video.sh                          newest local run, newest checkpoint
#   ./record_video.sh --list                   list local and Hyak runs
#   ./record_video.sh --list <run>             list that run's checkpoints
#   ./record_video.sh <run>                    newest checkpoint of <run>
#   ./record_video.sh <run> model_400.pt       one checkpoint
#   ./record_video.sh <run> model_200.pt model_400.pt model_600.pt
#   ./record_video.sh <run> --every 400        every 400th checkpoint
#   ./record_video.sh <run> --last 3           the 3 newest checkpoints
#
# Hyak runs are named "<jobid>_<run_name>" and are fetched automatically when the
# run is not present locally.
#
# Env overrides:
#   LENGTH=1800   video length in policy steps (default 600 ~= 24 s)
#   ENVS=16       number of environments to simulate
#   GPU=0         local GPU index
#   EXPERIMENT=g1_position_command
set -euo pipefail

cd "$(dirname "$0")"

LENGTH=${LENGTH:-600}
ENVS=${ENVS:-16}
GPU=${GPU:-0}
EXPERIMENT=${EXPERIMENT:-g1_position_command}
HYAK_HOST=${HYAK_HOST:-klone1}
HYAK_LOGS=${HYAK_LOGS:-/gscratch/robotics/mateogc/isaaclab/logs/rsl_rl}

LOCAL_ROOT="logs/rsl_rl/$EXPERIMENT"
REMOTE_ROOT="$HYAK_LOGS/$EXPERIMENT"

die() { echo "error: $*" >&2; exit 1; }

remote_runs() {
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$HYAK_HOST" \
        "ls -t $REMOTE_ROOT 2>/dev/null" 2>/dev/null || true
}

remote_ckpts() {
    ssh -o BatchMode=yes -o ConnectTimeout=20 "$HYAK_HOST" \
        "ls -t $REMOTE_ROOT/$1/model_*.pt 2>/dev/null | xargs -r -n1 basename" 2>/dev/null || true
}

if [ "${1:-}" = "--list" ]; then
    if [ -n "${2:-}" ]; then
        echo "local checkpoints in $2:"
        ls -t "$LOCAL_ROOT/$2"/model_*.pt 2>/dev/null | xargs -r -n1 basename | sed 's/^/  /' || echo "  (none)"
        echo "Hyak checkpoints in $2:"
        remote_ckpts "$2" | sed 's/^/  /' || echo "  (none)"
    else
        echo "local runs:"; ls -t "$LOCAL_ROOT" 2>/dev/null | sed 's/^/  /' || echo "  (none)"
        echo "Hyak runs:";  remote_runs | sed 's/^/  /' || echo "  (unreachable)"
    fi
    exit 0
fi

RUN=${1:-}
shift || true
if [ -z "$RUN" ]; then
    RUN=$(ls -t "$LOCAL_ROOT" 2>/dev/null | head -1)
    [ -n "$RUN" ] || die "no local runs under $LOCAL_ROOT"
fi

# Resolve which checkpoints to render.
SELECT_EVERY=""
SELECT_LAST=""
CKPTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --every) SELECT_EVERY=${2:?--every needs a number}; shift 2 ;;
        --last)  SELECT_LAST=${2:?--last needs a number}; shift 2 ;;
        *)       CKPTS+=("$1"); shift ;;
    esac
done

# Always ask Hyak as well as looking locally. Deciding from the local directory alone would
# pin the run to whatever was fetched first, so a still-training job would never show its
# newer checkpoints. An unreachable cluster degrades to local-only rather than failing.
local_ckpts() {
    ls "$LOCAL_ROOT/$RUN"/model_*.pt 2>/dev/null | xargs -r -n1 basename
}

REMOTE_LIST=$(remote_ckpts "$RUN")
[ -n "$REMOTE_LIST" ] && HAS_REMOTE=1 || HAS_REMOTE=0

# Newest first, by iteration number: local and remote mtimes are not comparable.
available() {
    { local_ckpts; printf '%s\n' "$REMOTE_LIST"; } | grep -E '^model_[0-9]+\.pt$' |
        sort -t_ -k2 -n -r -u
}

if [ ${#CKPTS[@]} -eq 0 ]; then
    mapfile -t ALL < <(available)
    [ ${#ALL[@]} -gt 0 ] || die "no checkpoints found for run '$RUN' (try --list)"
    if [ -n "$SELECT_EVERY" ]; then
        mapfile -t CKPTS < <(printf '%s\n' "${ALL[@]}" | sed 's/model_\([0-9]*\)\.pt/\1/' | sort -n |
            awk -v n="$SELECT_EVERY" '$1 % n == 0 {print "model_"$1".pt"}')
        [ ${#CKPTS[@]} -gt 0 ] || die "no checkpoints divisible by $SELECT_EVERY"
    elif [ -n "$SELECT_LAST" ]; then
        mapfile -t CKPTS < <(printf '%s\n' "${ALL[@]}" | head -"$SELECT_LAST")
    else
        CKPTS=("${ALL[0]}")
    fi
fi

echo "run        : $RUN  (local: $(local_ckpts | wc -l) ckpts, Hyak: $(printf '%s\n' "$REMOTE_LIST" | grep -c . || true))"
echo "checkpoints: ${CKPTS[*]}"
echo "length     : $LENGTH policy steps   envs: $ENVS   gpu: $GPU"
echo

mkdir -p "$LOCAL_ROOT/$RUN"

unset VIRTUAL_ENV
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="$GPU"

for CKPT in "${CKPTS[@]}"; do
    CKPT_PATH="$PWD/$LOCAL_ROOT/$RUN/$CKPT"
    if [ ! -f "$CKPT_PATH" ]; then
        [ "$HAS_REMOTE" -eq 1 ] || die "checkpoint not found locally and $HYAK_HOST has none: $CKPT"
        echo "--- fetching $RUN/$CKPT from $HYAK_HOST"
        rsync -q --info=progress2 \
            "$HYAK_HOST:$REMOTE_ROOT/$RUN/$CKPT" "$LOCAL_ROOT/$RUN/" ||
            die "rsync failed for $RUN/$CKPT"
    fi

    echo "--- rendering $CKPT"
    # The reset_*_on_load flags skip checkpoint state sized per task: the task table is
    # not bit-reproducible across processes, so those buffers never match.
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task Isaac-Position-v0 presets=g1 \
        --checkpoint "$CKPT_PATH" \
        --num_envs "$ENVS" --video --video_length "$LENGTH" \
        --viz kit \
        agent.reset_curriculum_on_load=true \
        agent.reset_command_on_load=true \
        agent.reset_event_on_load=true || echo "  (render failed for $CKPT)"

    # play.py always writes videos/play/rl-video-step-0.mp4; keep one file per checkpoint.
    SRC="$LOCAL_ROOT/$RUN/videos/play/rl-video-step-0.mp4"
    if [ -f "$SRC" ]; then
        DEST="$LOCAL_ROOT/$RUN/videos/${RUN}_${CKPT%.pt}_${LENGTH}steps.mp4"
        mv "$SRC" "$DEST"
        echo "  -> $DEST ($(du -h "$DEST" | cut -f1))"
    else
        echo "  no mp4 produced for $CKPT"
    fi
done

echo
echo "=== videos in $LOCAL_ROOT/$RUN/videos ==="
ls -lht "$LOCAL_ROOT/$RUN/videos"/*.mp4 2>/dev/null | awk '{print "  "$9"  "$5}' | head -10 || echo "  (none)"
