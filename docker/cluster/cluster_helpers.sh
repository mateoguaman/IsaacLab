# IsaacLab UW cluster helpers.
#
# Each public command (cluster_submit / cluster_sweep / cluster_collect)
# temporarily applies the feature/uw-cluster bundle on top of your current branch,
# runs the requested cluster operation, then resets the branch back to its
# exact pre-bundle state. Your working branch stays free of cluster infra
# files; upgrades to feature/uw-cluster propagate automatically on the next call.
#
# Usage:
#   cluster_submit  [args]          Submit one training job
#   cluster_sweep   [args]          Submit a hyperparameter sweep
#   cluster_collect [args]          Pull sweep logs back from the cluster
#   handoff_build   [args]          Build (and push) a self-contained Docker
#                                   image that runs a baked-in train command
#   cluster_help                    Show a short usage reminder
#   cluster_update_helpers          Re-pull the latest version of this file
#
# Installation: see docs/source/deployment/uw_cluster.rst (one-time setup
# writes this file to ~/.local/share/isaaclab_cluster_helpers.sh and sources
# it from ~/.bashrc). Override the bundle ref for non-default forks with:
#   export CLUSTER_BUNDLE_REF=myfork/feature/uw-cluster

_with_cluster_bundle() {
    # Resolve the bundle ref at call time so $CLUSTER_BUNDLE_REF can be set or
    # changed after this file is sourced (e.g. via ~/.bashrc in any order).
    local bundle_ref="${CLUSTER_BUNDLE_REF:-origin/feature/uw-cluster}"

    # Must be inside a git repository.
    local repo_root
    repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
        echo "[cluster] not inside a git repo — cd into the IsaacLab working tree first" >&2
        return 1
    }

    # Split "remote/ref" into its components for the fetch.
    local remote="${bundle_ref%%/*}"
    local short_ref="${bundle_ref#*/}"

    # Fetch so the ref is up-to-date; ignore fetch errors (network, auth) and
    # rely on the verify check below to surface a helpful message if needed.
    git -C "$repo_root" fetch --quiet "$remote" "$short_ref" 2>/dev/null || true

    # Bundle ref must now be resolvable.
    if ! git -C "$repo_root" rev-parse --verify --quiet "$bundle_ref" >/dev/null; then
        echo "[cluster] ref '$bundle_ref' not found" >&2
        echo "[cluster] add the remote that hosts feature/uw-cluster, or set CLUSTER_BUNDLE_REF" >&2
        return 1
    fi

    # Self-heal: strip any leftover "temp: cluster bundle" commits from a
    # previous interrupted run (e.g., Ctrl-C during the wizard). Safe because
    # we only peel commits whose subject exactly matches our own marker.
    local _tmp_subj="temp: cluster bundle (auto-reset)"
    local _tries=0
    while [ "$(git -C "$repo_root" log -1 --format='%s' 2>/dev/null)" = "$_tmp_subj" ] && [ "$_tries" -lt 10 ]; do
        echo "[cluster] cleaning up leftover temp commit from prior run" >&2
        git -C "$repo_root" reset --hard --quiet HEAD~1 || break
        _tries=$((_tries + 1))
    done

    # Handle uncommitted changes. The final reset --hard would otherwise
    # destroy them. Prompt rather than auto-stash silently: silent stashing
    # would mean WIP the user wants to deploy is dropped from the rsync
    # without warning.
    local auto_stashed=0
    local stash_marker=""
    if ! git -C "$repo_root" diff --quiet || ! git -C "$repo_root" diff --cached --quiet; then
        if [ ! -t 0 ] || [ ! -t 1 ]; then
            echo "[cluster] working tree has uncommitted changes — commit or stash first" >&2
            echo "[cluster] (no TTY available to prompt)" >&2
            return 1
        fi
        echo "[cluster] you have uncommitted changes:" >&2
        git -C "$repo_root" status --short >&2
        echo "" >&2
        echo "[cluster] how should we handle them?" >&2
        echo "    (c) commit them yourself first, then re-run — your changes will reach the cluster" >&2
        echo "    (s) stash them for this submit — the last committed state runs, not your WIP" >&2
        echo "    (a) abort" >&2
        local _choice=""
        while true; do
            read -r -p "[cluster] choice [c/s/a]: " _choice </dev/tty
            case "$_choice" in
                c|C)
                    echo "[cluster] commit your changes (e.g., git add -A && git commit), then re-run." >&2
                    return 1
                    ;;
                s|S)
                    stash_marker="cluster_helpers/$(date +%s)"
                    if ! git -C "$repo_root" stash push --quiet -m "$stash_marker"; then
                        echo "[cluster] git stash push failed — aborting" >&2
                        return 1
                    fi
                    auto_stashed=1
                    echo "[cluster] stashed as '$stash_marker' — will be popped after submission" >&2
                    break
                    ;;
                a|A|"")
                    echo "[cluster] aborted." >&2
                    return 1
                    ;;
                *)
                    echo "[cluster] invalid choice '$_choice' — please answer c, s, or a" >&2
                    ;;
            esac
        done
    fi

    # Snapshot the current commit so we can reset unambiguously later, even if
    # something (a hook, a background process, etc.) commits between steps.
    local prev_sha
    prev_sha=$(git -C "$repo_root" rev-parse HEAD) || return 1

    # Step 1: stage the bundle on top of the current branch.
    if ! git -C "$repo_root" merge --squash "$bundle_ref"; then
        # Clear the half-applied merge from working tree and index, then
        # restore the user's stashed WIP if we created one. The user is left
        # in exactly their pre-submit state.
        git -C "$repo_root" reset --hard --quiet HEAD
        if [ "$auto_stashed" = "1" ]; then
            git -C "$repo_root" stash pop --quiet \
                || echo "[cluster] WARNING: stash pop failed — recover with: git stash list | grep $stash_marker" >&2
        fi
        echo "[cluster] squash merge of '$bundle_ref' failed (likely conflicts on shared files)." >&2
        echo "[cluster] your working tree has been restored to its pre-submit state." >&2
        return 1
    fi

    # Steps 2-4 run inside a subshell whose EXIT trap is guaranteed to fire on
    # any exit path — normal return, error, SIGINT (Ctrl-C), SIGTERM, terminal
    # close, etc. Prior to this guard, interrupting the user command between
    # the temp commit and the reset would leave the bundle commit at HEAD; any
    # subsequent `git commit` would then cement cluster-infra files onto the
    # working branch. The trap is scoped to the subshell so it does not leak
    # to the caller's shell. The reset is guarded by a HEAD-subject check so
    # that an early subshell failure (e.g., the commit itself fails) does not
    # spuriously rewind the user's branch.
    local exit_code
    (
        trap '
            if [ "$(git -C "$repo_root" log -1 --format=%s 2>/dev/null)" = "temp: cluster bundle (auto-reset)" ]; then
                git -C "$repo_root" reset --hard --quiet "$prev_sha" 2>/dev/null
            fi
        ' EXIT

        # Step 2: snapshot the combined state so the reset is a single command.
        git -C "$repo_root" commit -m "temp: cluster bundle (auto-reset)" --quiet || exit $?

        # Step 3: run the actual command from the repo root so relative paths
        # like ./docker/cluster/cluster_interface.sh resolve correctly
        # regardless of the user's CWD.
        cd "$repo_root" && "$@"
        # Step 4 (the reset to $prev_sha) runs via the EXIT trap above.
    )
    exit_code=$?

    # Step 5: restore the user's WIP from stash, if we created one. The reset
    # above leaves the working tree clean (matching $prev_sha), so pop should
    # always succeed.
    if [ "$auto_stashed" = "1" ]; then
        if ! git -C "$repo_root" stash pop --quiet; then
            echo "[cluster] WARNING: 'git stash pop' failed unexpectedly" >&2
            echo "[cluster] your WIP is preserved — recover with: git stash list | grep $stash_marker" >&2
        fi
    fi

    return $exit_code
}

cluster_setup()   { _with_cluster_bundle ./docker/cluster/setup.sh "$@"; }
cluster_build()   { _with_cluster_bundle ./docker/container.py start "$@"; }
cluster_push()    { _with_cluster_bundle ./docker/cluster/cluster_interface.sh push "$@"; }
cluster_submit()  { _with_cluster_bundle ./docker/cluster/cluster_interface.sh job "$@"; }
cluster_sweep()   { _with_cluster_bundle python3 scripts/reinforcement_learning/rsl_rl/sweep.py "$@"; }
cluster_collect() { _with_cluster_bundle python3 scripts/reinforcement_learning/rsl_rl/sweep_collect.py "$@"; }
handoff_build()   { _with_cluster_bundle ./docker/handoff/handoff_interface.sh "$@"; }

cluster_help() {
    cat <<'EOF'
IsaacLab UW cluster helpers:
  cluster_setup            First-time config: generates docker/cluster/.env.user
                           (persists across branch switches — gitignored).
  cluster_build   [args]   Build the local Docker image (with local rsl_rl baked in
                           if a clone exists at IsaacLab/rsl_rl/).
  cluster_push    [args]   Push the Docker -> Singularity image to the cluster.
  cluster_submit  [args]   Submit one training job.
  cluster_sweep   [args]   Submit a hyperparameter sweep.
  cluster_collect [args]   Pull sweep logs back from the cluster.
  handoff_build   [args]   Build (and push) a self-contained Docker image that
                           runs a baked-in train command; for handing off a
                           single training job to a collaborator's machine.
  cluster_update_helpers   Re-pull the latest cluster_helpers.sh.

All cluster_* and handoff_* commands transparently apply + revert the
feature/uw-cluster bundle so your working branch stays free of cluster infra
files. Run them from inside an IsaacLab working tree. Override the bundle ref
with:
    export CLUSTER_BUNDLE_REF=myfork/feature/uw-cluster
EOF
}

# Re-pull this file from the bundle ref and overwrite the sourced copy in place.
# The user must open a new shell (or re-source the file) to pick up the changes.
cluster_update_helpers() {
    local bundle_ref="${CLUSTER_BUNDLE_REF:-origin/feature/uw-cluster}"

    local target="${BASH_SOURCE[0]:-}"
    if [ -z "$target" ]; then
        echo "[cluster_helpers] could not locate the currently-sourced helpers file" >&2
        return 1
    fi

    local repo_root
    repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
        echo "[cluster_helpers] run from inside an IsaacLab working tree" >&2
        return 1
    }

    local remote="${bundle_ref%%/*}"
    local short_ref="${bundle_ref#*/}"
    if ! git -C "$repo_root" fetch --quiet "$remote" "$short_ref"; then
        echo "[cluster_helpers] fetch $remote $short_ref failed" >&2
        return 1
    fi

    if ! git -C "$repo_root" show "$bundle_ref:docker/cluster/cluster_helpers.sh" > "$target"; then
        echo "[cluster_helpers] could not read cluster_helpers.sh from $bundle_ref" >&2
        return 1
    fi

    echo "[cluster_helpers] updated $target"
    echo "[cluster_helpers] open a new shell or run: source $target"
}
