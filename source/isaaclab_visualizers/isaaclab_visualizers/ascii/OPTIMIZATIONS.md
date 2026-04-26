# ASCII Visualizer — Optimization Notes

A record of how the ASCII visualizer's producer pipeline was taken from a
single-threaded numpy software rasterizer (unusable beyond a few hundred
pixels) to a fused-kernel warp GPU rasterizer that keeps up with 4096
parallel Newton physics environments. Kept alongside the code so the
rationale for each design choice doesn't get lost.

Starting point: at 1264×784 with 4 envs the numpy rasterizer took 4.5 s
per frame and training never advanced past step 1. End point: at 4096 envs
(239 M triangles per frame) we render at ~35 fps while training runs at
143k environment-steps/sec. ~700k× speedup on the rasterize stage, ~3000×
on training throughput.

---

## Current status (1264×784, Anymal-C flat locomotion, Newton physics)

| num_envs | n_instances | n_triangles | fps | steps/s  | iteration |
|---------:|------------:|------------:|----:|---------:|----------:|
| 4        | 185         | 233k        | 45–65 | 222     | 0.43 s    |
| 1024     | 47k         | 60M         | ~26  | 26,867   | 0.91 s    |
| 4096     | 188k        | 239M        | ~35  | **143,034** | **0.69 s** |

At 4096 envs we are now **sim-bound**, not render-bound. Each PPO sim step
is ~28 ms, of which ~15 ms is render GPU work + sync and ~13 ms is Newton
itself. Further rasterizer optimization gives diminishing returns.

---

## Optimization journey

Each step below lists the change, what it targeted, and the measured win.
All numbers are at 1264×784 resolution.

### 0. Baseline — numpy software rasterizer

```
NumpyRasterizer.draw_triangles(...) iterates triangles in a Python
for-loop; each _raster_one() builds a meshgrid over the bbox and runs
vectorized numpy edge tests.
```

At 233k triangles: **4500 ms per frame, fps 0.2, training stalled at step 1**.

Python per-triangle dispatch cost is the killer (~19 µs/triangle). At
60M tris that extrapolates to 72 s/frame — which is exactly why step 1
never finished.

### 1. Warp GPU rasterizer — 3 kernels, per-triangle

Replaced `NumpyRasterizer` with a GPU rasterizer written in NVIDIA Warp:

- `_setup_kernel` — world → clip → NDC → screen, bbox, inv_area, flat
  Lambertian shading baked into per-triangle `lit_colors`, near-plane and
  degenerate-tri culling.
- `_raster_kernel` — per-triangle thread walks its screen-space bbox,
  runs edge functions, and does `atomic_min` on a `uint64` depth buffer
  packed as `(sortable_depth_u32 << 32) | tri_id`. Deterministic
  tiebreaking via the packed layout.
- `_resolve_kernel` — per-pixel; unpacks winning `tri_id`, writes the
  shaded color (or background) into an `uint8[H,W,4]` buffer.

Output is bit-exact with the numpy rasterizer on test scenes
(`max_diff == 0`).

At 240k tris, 1264×784: **6.82 ms vs 8,503 ms numpy — 1246×**. Sim
unblocks, training throughput at 4 envs: **93 → 222 steps/s**.

### 2. Warp expand kernel — eliminate the Python per-instance loop

`_world_triangles` was the remaining Python bottleneck (16 ms/frame at
185 instances). Replaced with `_expand_kernel`: one GPU thread per
output triangle, looks up `(inst, tri)`, applies scale → quat_rotate →
translate, writes world-space verts.

`log_mesh` now calls `rasterizer.upload_mesh(verts, indices)` once per
mesh; the returned `wp.array` handles are retained on-GPU and reused
every frame.

At 4 envs: fps 25–30 → **45–65**, training **93 → 222** steps/s.

### 3. Fused kernel — collapse expand + setup + raster

At 4096 envs the separate-kernel design allocates scratch proportional
to triangle count:
- `world_tris` (N,3) vec3 — 8.6 GB
- `screen` (N,3) vec3 — 8.6 GB
- `base_colors`, `lit_colors` (N,) vec3 — 2.9 GB each
- `bbox` (N,) vec4 — 3.8 GB
- `inv_area`, `valid` — 1 GB each

**Total ~28 GB at 239 M tris.** A 4090 has 23 GB, so the overflow fell
to CUDA unified memory, making every access PCIe-slow.

Fix: collapse `_expand_kernel` + `_setup_kernel` + `_raster_kernel` into
`_fused_expand_raster_kernel`. Each thread keeps screen verts, bbox,
inv_area, and lit color in registers through expand → MVP → NDC → raster
→ atomic_min. The only per-triangle global-memory output is
`lit_colors_buf` (needed by the resolve pass to colorize the winner).

Scratch drops from ~28 GB to ~3 GB. No more UVM paging.

At 4096 envs:
- stage 3 time: 52 ms → **8 ms** (6.5×)
- stage 4 time: 7.8 ms → **0.9 ms** (9×)
- fps: 11.4 → **27.1**
- training: 47k → **110k** steps/s

### 4. Branch-free winding — (b)

The old kernel had a divergent branch per triangle:

```python
if area < 0.0:
    swap v1 and v2
    area = -area
```

Replaced by **signed barycentrics** — `b = w * inv_a` is positive inside
the triangle for either winding as long as `inv_a` carries the area's
sign. No branch, no warp divergence:

```python
area = (sx1 - sx0) * (sy2 - sy0) - (sy1 - sy0) * (sx2 - sx0)
if wp.abs(area) < 1.0e-6: return
inv_a = 1.0 / area   # signed

# inside raster loop
b0 = w0 * inv_a
b1 = w1 * inv_a
b2 = w2 * inv_a
if b0 < 0.0 or b1 < 0.0 or b2 < 0.0: continue
```

### 5. Zero-copy from Newton — (c)

The previous `log_instances` did `_to_numpy(xforms)` which forces a
GPU→CPU copy, then `draw_batch` uploaded back to GPU via
`wp.array(np_array, ...)`. Per frame at 4096 envs: ~10 MB each direction
of pointless PCIe traffic plus multiple `wp.array` allocations per batch.

Fixed at two layers:

1. Kernel now takes `xforms: wp.array(dtype=wp.transform)` directly —
   Newton's native format. Inside: `xf = xforms[inst]; p =
   wp.transform_get_translation(xf); q = wp.transform_get_rotation(xf)`.
2. `log_instances` keeps warp arrays as-is; only numpy-like inputs are
   converted to numpy.
3. `draw_batch` dispatches on input type via helpers
   `_as_wp_transform` / `_as_wp_vec3` — zero-copy for `wp.array`, upload
   for `np.ndarray` (test compat).

The subtle but biggest win: Python no longer blocks on numpy↔GPU copies
mid-frame, so **sim and render overlap** on the GPU. Measured stage
totals go up (GPU sync-wait shifts into `rgba_pack`), but wall-clock
iteration time dropped 22%.

Combined with (b):
- stage 3: 8 ms → **0.58 ms** (pure launch overhead)
- fps: 27.1 → **~35**
- training: 110k → **143k** steps/s
- iteration time: 0.89 s → **0.69 s**

---

## Code layout

```
isaaclab_visualizers/ascii/
├── ascii_visualizer.py         — BaseVisualizer subclass. Filters env IDs
│                                  (framework max_worlds), drives
│                                  begin_frame/log_state/end_frame per step.
├── ascii_visualizer_cfg.py     — cfg dataclass (render size, camera,
│                                  transport mode, CLI overrides).
├── viewer_ascii.py             — Newton ViewerBase subclass.
│   ├─ ViewerAscii              — log_mesh (rasterizer.upload_mesh once per
│   │                              mesh), log_instances (keeps wp.array inputs
│   │                              as-is, no _to_numpy), end_frame (per-batch
│   │                              rasterizer.draw_batch), three transports.
│   ├─ _AsciiPerfMeter          — env-var-gated per-stage timer. Off by
│   │                              default; enable via
│   │                              `ISAACLAB_ASCII_PROFILE=1`. Period in
│   │                              frames via
│   │                              `ISAACLAB_ASCII_PROFILE_PERIOD=N`.
│   └─ _InlineTransport /        — RGBA -> ascii-cli consumer (child
│       _FifoTransport /          process, named pipe, or TCP listener).
│       _TcpTransport
└── warp_rasterizer.py          — all warp kernels + WarpRasterizer class.
    ├─ _clear_depth_kernel      — depth_buf <- UINT64_MAX each frame.
    ├─ _fused_expand_raster_kernel
    │                           — hot kernel. One thread per output tri:
    │                               expand (quat_rotate + scale + translate),
    │                               shade (Lambert+ambient), MVP → NDC →
    │                               screen, near-plane cull, signed-bary
    │                               raster with atomic_min on packed
    │                               (depth, tri_id) uint64.
    ├─ _resolve_kernel          — one thread per pixel: unpack winning
    │                              tri_id, write lit_colors[tri_id] to RGBA8
    │                              (or bg).
    └─ WarpRasterizer           — clear / upload_mesh / draw_batch / rgba.
                                   Only persistent scratch is lit_colors_buf
                                   (~3 GB at 4096 envs). draw_batch accepts
                                   wp.array (zero-copy) or np.ndarray inputs.
```

### Data flow per frame

```
Newton physics (warp arrays on GPU)
    │
    ▼  [state.body_q etc. — still warp, zero-copy in log_instances]
ViewerAscii.log_instances(name, mesh, xforms=wp.array(transform), scales, colors, ...)
    │
    ▼
ViewerAscii.end_frame()
    ├─ rasterizer.clear()
    │      └─ _clear_depth_kernel(H*W)   — depth_buf = UINT64_MAX
    ├─ for each batch:
    │      rasterizer.draw_batch(mesh_verts_wp, mesh_idx_wp, xforms_wp, ...)
    │         └─ _fused_expand_raster_kernel(dim=I*T)
    │                expand → shade → MVP → cull → bbox → raster
    │                writes lit_colors_buf, atomic_min on depth_buf
    └─ rasterizer.rgba()
           ├─ _resolve_kernel(H*W)       — depth_buf → rgba
           └─ rgba.numpy()                — single GPU→CPU copy (~4 MB)
    ▼
_write_frame(rgba_bytes)
    └─ transport.write(bytes)
           └─ TCP / FIFO / stdin → ascii-cli → terminal
```

---

## Key design decisions (why the code looks this way)

- **Packed `uint64(depth << 32 | tri_id)` in the depth buffer.** One
  `atomic_min` does both the depth test *and* deterministic tiebreaking
  (lowest tri_id wins on exact depth matches). The resolve pass recovers
  the winning `tri_id` from the low 32 bits and uses it to index
  `lit_colors_buf`.

- **Fuse everything into one kernel when possible.** Intermediate
  per-triangle arrays at 239 M triangles cost tens of GB. Keeping state
  in registers inside one kernel keeps scratch proportional to pixel
  count (depth_buf, rgba) + a single per-tri color buffer, not dozens of
  per-tri arrays.

- **Signed barycentrics > winding flip.** A branch per triangle causes
  warp divergence at ~50% rate (random winding). Using signed `inv_a`
  and testing `b = w * inv_a >= 0` is branchless and correct for both
  windings. Measurable hot-path cleanup.

- **Kernel takes `wp.array(dtype=wp.transform)`.** This is Newton's
  native layout for `body_q` and similar arrays. Accepting it directly
  avoids a GPU→CPU→GPU roundtrip every frame and lets Python advance
  without blocking so sim and render overlap.

- **`upload_mesh` once per mesh.** Static geometry (verts, indices) lives
  on GPU for the lifetime of the viewer. Only per-frame xforms get
  touched, and those are zero-copy from Newton.

- **Env-var profiler.** `ISAACLAB_ASCII_PROFILE=1` turns on per-stage
  timing with near-zero overhead when off. Important detail: measured
  `stage 4` (`rgba_pack`) absorbs GPU sync-wait from previous async
  kernel launches. Total per-frame render cost = stage 3 + stage 4, not
  stage 4 alone.

- **The framework handles env-subset rendering.** Don't reinvent
  `max_worlds` — it's in `BaseVisualizer._compute_visualized_env_ids`
  and the launcher's `--visualizer_max_worlds` already plumb through to
  `cfg.max_worlds`. Works for viser/rerun/newton/ascii alike.

---

## How to benchmark / reproduce

### Terminal A — Isaac training + producer

```bash
cd ~/projects/Isaac/IsaacLab
source .venv/bin/activate

# Optional profiler: per-stage timings every N frames
ISAACLAB_ASCII_PROFILE=1 ISAACLAB_ASCII_PROFILE_PERIOD=30 \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Anymal-C-v0 \
  --num_envs 4096 \
  --viz ascii \
  --ascii_output tcp:5582 \
  --ascii_resolution 1264x784 \
  --ascii_sample_res 8 \
  presets=newton
```

Note: `presets=newton` switches physics from PhysX to Newton (kitless
path). Without it, PhysX + Kit would be required.

### Terminal B — ASCII consumer

Resize the terminal to ~158 cols × 73 rows first (1264/8 × 784/10.67):

```bash
cd ~/projects/ascii_renderer
nc localhost 5582 | ./bin/ascii-cli.mjs \
  --raw-in 1264x784 \
  --sample-res 8 \
  --mode truecolor \
  --bg dark
```

To auto-size to the current terminal dims, before starting Terminal A:

```bash
COLS=$(tput cols); ROWS=$(tput lines); SR=8
W=$(( COLS * SR ))
H=$(awk "BEGIN{print int($ROWS * $SR * 1.3333 + 0.5)}")
echo "Use in Terminal A: --ascii_resolution ${W}x${H} --ascii_sample_res ${SR}"
echo "Then in Terminal B: nc localhost 5582 | ~/projects/ascii_renderer/bin/ascii-cli.mjs \\"
echo "  --raw-in ${W}x${H} --sample-res ${SR} --mode truecolor --bg dark"
```

The `1.3333` constant is the alphabet metadata aspect
(`height/width` from `ascii_renderer/src/core/alphabet.json`).

### Profiler output format

```
[ViewerAscii perf] frame=30 fps=35.6 1_clear=0.01ms 2_matrices=0.09ms \
  3_expand_and_draw=0.58ms 4_rgba_pack=14.70ms 5_transport_write=0.92ms \
  n_instances=188417.0 n_triangles=238927874.0
```

- `1_clear` / `2_matrices` — trivial, included for completeness.
- `3_expand_and_draw` — Python time to iterate batches and launch the
  fused kernel per batch. With (c) it's ~launch overhead (0.6 ms).
- `4_rgba_pack` — runs `_resolve_kernel` then `rgba.numpy()`. The
  `.numpy()` call synchronizes with all pending kernel launches, so
  this stage absorbs real GPU execution time from stage 3's async work.
  **Not a resolve-kernel cost; it's the GPU sync.**
- `5_transport_write` — `transport.write(rgba.tobytes())`. Backpressure
  would show up here; typically <1 ms.

---

## Things we chose not to do (and why)

- **Tile binning with per-pixel scan.** For our workload (very many
  sub-pixel triangles at 4096 envs), per-pixel scan is ~60× *worse* than
  per-triangle-iterate-bbox. Math: avg pixels/tri ≈ 4 vs avg tris/tile
  ≈ 62k. Per-pixel scan only wins when big triangles dominate.

- **Tile binning with per-(tri, tile) threads.** Would add back ~10–14
  GB of scratch (screen data per triangle) and give modest (1.2–1.5×)
  atomic-contention savings. Net: likely OOM at 4096 envs for marginal
  gain. Right move if you go to 8192+ envs with denser meshes.

- **Shared-memory depth buffer.** Would help atomic contention a lot
  (~2–4×), but Warp 1.12 doesn't expose easy shared-memory atomics for
  arbitrary types. Revisit if Warp exposes `wp.tile_atomic_min` on
  shared arrays.

- **Multi-GPU render split.** The box has 2×4090 and Newton only uses
  one. A render-on-GPU1 / train-on-GPU0 split would remove the render
  time entirely from the critical path. Biggest single-step win
  available but needs real plumbing work.

- **Mesh LOD.** Anymal ships with ~1268 tri/body meshes. A 128-tri LOD
  would give ~10× render speedup but is a user-side config choice, not
  an ASCII-visualizer concern.

---

## Future directions

1. **Dual-GPU split** (highest ROI). Render on cuda:1, train on cuda:0.
   Remove render from the sim critical path entirely.
2. **Tile binning with shared-mem depth buffer** — wait for Warp API
   support.
3. **Front-to-back depth pre-sort per mesh** — cheap early-out on tris
   occluded by closer geometry. Moderate gain for dense scenes.
4. **Pinned-memory readback** — shaves ~0.5 ms off the final
   `rgba.numpy()` copy.
5. **Integrate the profiler with training telemetry** (tensorboard) so
   render cost is tracked alongside loss curves.

---

## Key numbers at a glance

| stage                                       | 4 envs | 1024 envs | 4096 envs |
|--------------------------------------------:|-------:|----------:|----------:|
| n_instances                                 | 185    | 47,105    | 188,417   |
| n_triangles                                 | 233k   | 59.7 M    | 239 M     |
| numpy rasterizer frame time                 | 4500 ms| (would be hours) | — |
| separate-kernel warp frame                  | —      | 17 ms     | 60 ms     |
| fused-kernel warp frame                     | 4.3 ms | 14 ms     | 10 ms     |
| fused + (b) + (c) frame                     | —      | —         | **~15 ms (w/ sim overlap)** |
| training steps/s (original stall → current) | 0 → 222| — → 26,867| **0 → 143,034** |
