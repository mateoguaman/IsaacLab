# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner that pushes full rollouts to a PLR :class:`LevelSampler` curriculum.

Locates the curriculum term that exposes both ``update_with_rollouts(rollouts)`` and a
``level_sampler`` attribute (i.e. was configured with ``presets=plr``). Push timing piggybacks on
``alg.compute_returns`` so the storage's ``advantages`` / ``returns`` / ``values`` tensors are read
once they exist and before ``alg.update`` flattens them for mini-batches. The full rollout buffer
is pushed so any scoring function in the :class:`ScoringCfg` family can be selected without changing
this file.

Select via Hydra by adding ``plr`` to the ``presets`` list; the agent-side ``plr`` preset repoints
``class_type`` to this runner, and the env-side ``plr`` preset selects the
:func:`~...mdp.curriculums.level_sampler_curriculum` term.
"""

from __future__ import annotations

from rsl_rl.runners import OnPolicyRunner


class OnPolicyRunnerWithLevelSampler(OnPolicyRunner):
    """PPO runner that feeds full rollouts to a :class:`LevelSampler`-backed curriculum.

    Locates the curriculum term by duck-typing on the ``CurriculumManager`` term list: any term
    whose ``func`` exposes ``update_with_rollouts`` and ``level_sampler`` is accepted. Raises at
    construction if no such term is configured, since the runner would otherwise silently no-op.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._curriculum_term = self._find_curriculum_term()
        if self._curriculum_term is None:
            raise RuntimeError(
                "OnPolicyRunnerWithLevelSampler requires a curriculum term exposing"
                " ``update_with_rollouts`` and ``level_sampler``. Configure the curriculum with a"
                " LevelSamplerCfg (e.g. ``presets=plr``)."
            )

    def _find_curriculum_term(self):
        """Return the curriculum term hosting a level sampler, or ``None`` if absent."""
        env = getattr(self.env, "unwrapped", self.env)
        if not hasattr(env, "curriculum_manager"):
            return None
        for term_cfg in getattr(env.curriculum_manager, "_term_cfgs", []):
            func = getattr(term_cfg, "func", None)
            if func is not None and hasattr(func, "update_with_rollouts") and hasattr(func, "level_sampler"):
                return func
        return None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Patch ``alg.compute_returns`` to push rollouts once per iter, then run training.

        Wrapping ``compute_returns`` is the cleanest hook point: it fires once per iter right after
        advantages exist and before ``alg.update`` consumes them. The patch is restored in a
        ``finally`` block so the runner remains reusable.
        """
        original_compute_returns = self.alg.compute_returns

        def compute_returns_with_rollout_push(*args, **kwargs):
            result = original_compute_returns(*args, **kwargs)
            self._push_rollouts_to_curriculum()
            return result

        self.alg.compute_returns = compute_returns_with_rollout_push
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)
        finally:
            self.alg.compute_returns = original_compute_returns

    def _push_rollouts_to_curriculum(self) -> None:
        """Adapt rsl_rl storage into a :class:`RolloutData` and push it to the curriculum."""
        from isaaclab_tasks.core.multi_task.curriculum import RolloutData

        storage = self.alg.storage
        rollouts = RolloutData(
            advantages=getattr(storage, "advantages", None),
            returns=getattr(storage, "returns", None),
            values=getattr(storage, "values", None),
            rewards=getattr(storage, "rewards", None),
            gamma=getattr(self.alg, "gamma", None),
        )
        self._curriculum_term.update_with_rollouts(rollouts)
