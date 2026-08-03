# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-trip tests for curriculum ``SuccessMonitor`` checkpoint state.

These exercise the checkpoint-resume path added for the training campaign: the monitor's
sliding-window ring buffers must survive a ``state_dict`` / ``load_state_dict`` cycle, and the
restore must happen **in place** so the Warp backend's ``wp.from_torch`` views keep pointing at
valid storage. No sim app is required — the monitor is a standalone torch/warp component.
"""

from __future__ import annotations

import pytest
import torch

from isaaclab_tasks.core.multi_task.curriculum.success_monitor import SuccessMonitor
from isaaclab_tasks.core.multi_task.curriculum.success_monitor_cfg import SuccessMonitorCfg

# The torch backend runs anywhere (CPU included). The warp backend needs an initialized Warp
# runtime + CUDA; in real training the sim boot initializes Warp, so here we initialize it
# explicitly and only add the warp case when that succeeds.
_CASES = [(False, "cpu")]
if torch.cuda.is_available():
    _CASES.append((False, "cuda:0"))
    try:
        import warp as wp

        wp.init()
        _CASES.append((True, "cuda:0"))
    except Exception:
        pass


def _make_monitor(warp: bool, device: str) -> tuple[SuccessMonitor, torch.Tensor]:
    cfg = SuccessMonitorCfg(num_monitored_data=6, monitored_history_len=4, device=device, warp=warp)
    success_rate = torch.zeros(6, device=device, dtype=torch.float32)
    return SuccessMonitor(cfg, success_rate), success_rate


def _drive(monitor: SuccessMonitor, ids: list[int], success: list[bool]) -> None:
    device = monitor.success_buf.device
    monitor.success_update(
        torch.tensor(ids, dtype=torch.int64, device=device),
        torch.tensor(success, dtype=torch.bool, device=device),
    )


@pytest.mark.parametrize("warp,device", _CASES)
def test_success_monitor_state_roundtrip(warp: bool, device: str) -> None:
    """A drifted monitor restores its ring buffers exactly, in place, and stays functional."""
    monitor, success_rate = _make_monitor(warp, device)
    _drive(monitor, [0, 1, 2, 0, 3], [True, False, True, True, False])
    _drive(monitor, [1, 4, 5], [True, True, False])

    snapshot = monitor.state_dict()
    buf_ptr = monitor.success_buf.data_ptr()
    pointer_ptr = monitor.success_pointer.data_ptr()

    # Drift after the snapshot so the restore is a genuine change, not a no-op. Use True
    # appends so the bool ring buffer itself changes (appending False into already-False
    # cells would only advance the pointer), and the pointer advances regardless.
    _drive(monitor, [0, 1, 2, 3, 4, 5], [True, True, True, True, True, True])
    assert not torch.equal(monitor.success_buf, snapshot["success_buf"])
    assert not torch.equal(monitor.success_pointer, snapshot["success_pointer"])

    monitor.load_state_dict(snapshot)

    assert torch.equal(monitor.success_buf, snapshot["success_buf"])
    assert torch.equal(monitor.success_pointer, snapshot["success_pointer"])
    assert torch.equal(monitor.success_size, snapshot["success_size"])
    assert torch.equal(monitor.success_count, snapshot["success_count"])

    # In-place restore preserved the underlying storage — this is what keeps the Warp views valid.
    assert monitor.success_buf.data_ptr() == buf_ptr
    assert monitor.success_pointer.data_ptr() == pointer_ptr

    # The monitor is still functional after a load: a further update writes a finite rate.
    _drive(monitor, [0], [True])
    assert torch.isfinite(success_rate).all()


@pytest.mark.parametrize("warp,device", _CASES)
def test_success_monitor_snapshot_is_decoupled(warp: bool, device: str) -> None:
    """The snapshot is a clone: mutating the live monitor must not change a prior snapshot."""
    monitor, _ = _make_monitor(warp, device)
    _drive(monitor, [0, 1], [True, True])
    snapshot = monitor.state_dict()
    before = snapshot["success_count"].clone()
    _drive(monitor, [0, 1], [True, True])
    assert torch.equal(snapshot["success_count"], before)
