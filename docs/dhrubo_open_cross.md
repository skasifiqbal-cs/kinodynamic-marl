# Open cross scenario

Port K-ARC's Open Cross benchmark (arXiv:2501.01559, section V-B-1) into our env at 2, 4, 8, 16
and 32 robots.

Their description is "robots on the same row need to swap positions in an empty environment".
So: N/2 rows, each row holding one pair that swaps head-on across the workspace, mirrored about
the vertical centreline. Their second-order unicycle is `[x,y,theta,v,omega]` with controls
`[a,alpha]`, which is already `conf/robot/unicycle_db.yaml` — use it unchanged.

This is config generation, not env code. `MultiAgentNav` already handles any number of agents and
`FixedInitializer` (`src/init/initializer.py:29`) reads starts and goals straight out of the YAML.
Read `conf/env/swap2_unicycle2.yaml` first: it is the N=2 case and the house style for these files.

Write `scripts/gen_open_cross.py` emitting `conf/env/open_cross_{N}_unicycle2.yaml`. Don't
hand-write them, N=32 is 32 agent blocks.

Geometry:

```
rows   = N // 2
s_y    = 1.0                      # row spacing = turning radius v_max / omega_max
margin = 1.0
world  = max(5.0, (rows - 1) * s_y + 2 * margin)
y_k    = world/2 + (k - (rows - 1)/2) * s_y
x_left, x_right = world/2 - 1.5, world/2 + 1.5
```

The traverse stays 3.0 m at every N on purpose, so runtime against N measures congestion and not
distance. World comes out 5.0 up to N=8, 9.0 at 16, 17.0 at 32. Use `max_steps: 300` up to N=8
and 600 above. Copy `dt`, `goal_radius`, `stop_speed`, the flags and the whole `reward:` block
from swap2 unchanged, and keep its heading convention — left robot theta=0, right robot
theta=3.14159 at both start and goal. `obstacles: []`.

The check that matters: `--n 2` has to reproduce swap2_unicycle2's geometry exactly, world 5.0
with agents at `[1.0, 2.5, ...]` and `[4.0, 2.5, ...]`. Assert it in `__main__`. If that holds the
formula is right at every N, and if it doesn't nothing else you test means anything.

Things that will bite you:

The world is square. `world_size` is a single scalar and `src/obs/full_state.py:96-99` uses it for
both axes, so you cannot build the wide short box their Fig. 2(a) shows. Record it as a deviation
and move on — don't add rectangular world support.

Observation dimension is `11 + 2*(N-1)`, so a policy trained at one N won't load at another. Every
N is its own run.

Use `shaping=braking` or `shaping=euclidean`, not `dijkstra`. The world is empty so the grid buys
nothing, and it costs one solve per agent per env.

Collision checking is O(N^2). Time a single N=32 episode before queueing a long run.

The renderer is greyscale and numbers each robot and its goal (`src/viz/renderer.py`), so any N
renders without repeats. Above ~16 robots the bodies get small; raise `fig_px` rather than
changing the renderer.

Put a deviations header on every generated file the way swap2 does at lines 8-13. K-ARC publishes
no workspace dimensions, no d_min, no segment count, no timestep and no velocity limits, and runs
C++ on a 32-core i9-14900K. This is a scenario port, not a benchmark reproduction — our runtimes
are not comparable to their published ones and must not be presented as if they were.

Last thing. Every row is a symmetric head-on pair, which we already know is unsolved at N=2 (see
the note at `conf/env/swap2_unicycle2.yaml:15-17`). A shared-weights policy sees mirrored
observations and produces mirrored actions, so expect pairs to drive into each other. That is a
result, not a bug: log the per-row collision rate and report it. Don't offset the lanes to make
training easier.

Cluttered cross is not part of this.
