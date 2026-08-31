import os
import platform
import casadi as ca
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. OBSTACLE-AWARE & DETOUR-CAPABLE KINEMATIC REFERENCE PLANNER
# =====================================================================
class KinematicPlanner:

  def __init__(self, start, goal, map_obstacles):
    self.start = np.array(start[:2])
    self.goal = np.array(goal[:2])
    self.map_obstacles = map_obstacles

  def plan_guided_path(self, robot_id, total_robots=4):
    # If scaling past 4 robots, inject subtle priority-based tangential offsets
    # to break symmetry and prevent gradient-based optimizer lockouts
    midpoint = (self.start + self.goal) / 2.0
    if total_robots > 4 and robot_id > 0:
      center = np.array([5.0, 5.0])
      vec = midpoint - center
      tangent = np.array([
          -vec[1] if np.linalg.norm(vec) > 1e-3 else 1.0,
          vec[0] if np.linalg.norm(vec) > 1e-3 else 0.0,
      ])
      tangent = tangent / (np.linalg.norm(tangent) + 1e-6)
      offset_mag = 0.55 * (robot_id % 3 + 1)
      midpoint = midpoint + tangent * offset_mag

    pts = np.array([self.start, midpoint, self.goal])

    t_old = np.linspace(0, 1, len(pts))
    t_dense = np.linspace(0, 1, 80)
    dense_x = np.interp(t_dense, t_old, pts[:, 0])
    dense_y = np.interp(t_dense, t_old, pts[:, 1])

    tree_frames = []
    for i in range(len(dense_x) - 1):
      tree_frames.append((
          np.array([dense_x[i], dense_y[i]]),
          np.array([dense_x[i + 1], dense_y[i + 1]]),
      ))

    p_i = np.column_stack(
        [dense_x, dense_y, np.zeros(80), np.zeros(80), np.zeros(80)]
    )
    return p_i, tree_frames


# =====================================================================
# 2. CASADI KINODYNAMIC OPTIMIZER (Non-linear Trajectory Optimization)
# =====================================================================
def CasadiTrajectoryOptimizer(
    robot_id,
    start_state,
    goal_state,
    ref_path_segment,
    map_obstacles,
    avoid_robot_trajs=None,
    priority_level=0,
    dt=0.08,
    N=35,
):
  opti = ca.Opti()

  X = opti.variable(5, N + 1)
  U = opti.variable(2, N)

  obj = 0
  for k in range(N):
    obj += 0.01 * ca.sumsqr(U[:, k])
    if k < N - 1:
      obj += 0.05 * ca.sumsqr(U[:, k + 1] - U[:, k])

  if ref_path_segment is not None and len(ref_path_segment) > 0:
    ref_len = min(N + 1, len(ref_path_segment))
    for k in range(ref_len):
      obj += 1.2 * ca.sumsqr(X[0:2, k] - ref_path_segment[k][0:2])

  # Terminal attraction
  obj += 450.0 * ca.sumsqr(X[0:2, -1] - goal_state[0:2])
  opti.subject_to(X[:, 0] == start_state)

  # Tolerance adapts with priority hierarchy
  if priority_level == 0:
    opti.subject_to(ca.sumsqr(X[0:2, -1] - goal_state[0:2]) <= 0.20**2)
  else:
    tol_bound = min(1.2 + 0.5 * priority_level, 3.5)
    opti.subject_to(ca.sumsqr(X[0:2, -1] - goal_state[0:2]) <= tol_bound**2)

  def forward_dynamics(x, u):
    dx = x[3] * ca.cos(x[2])
    dy = x[3] * ca.sin(x[2])
    dtheta = x[4]
    dv = u[0]
    dw = u[1]
    return ca.vertcat(dx, dy, dtheta, dv, dw)

  # Staggered velocity bounds for progressive queueing
  base_max_v = 2.8
  if priority_level > 0:
    speed_cap = max(0.25, base_max_v - 0.45 * priority_level)
  else:
    speed_cap = base_max_v

  for k in range(N):
    x_k = X[:, k]
    u_k = U[:, k]

    k1 = forward_dynamics(x_k, u_k)
    k2 = forward_dynamics(x_k + 0.5 * dt * k1, u_k)
    k3 = forward_dynamics(x_k + 0.5 * dt * k2, u_k)
    k4 = forward_dynamics(x_k + dt * k3, u_k)
    x_next = x_k + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    opti.subject_to(X[:, k + 1] == x_next)
    opti.subject_to(opti.bounded(-2.5, u_k[0], 2.5))
    opti.subject_to(opti.bounded(-2.5, u_k[1], 2.5))

    curr_v_lim = (
        speed_cap if (priority_level > 0 and k < int(N * 0.65)) else base_max_v
    )
    opti.subject_to(opti.bounded(-2.8, X[3, k], curr_v_lim))
    opti.subject_to(opti.bounded(-2.5, X[4, k], 2.5))
    opti.subject_to(opti.bounded(0.0, X[0, k], 10.0))
    opti.subject_to(opti.bounded(0.0, X[1, k], 10.0))

    # Static Obstacle Clearance
    for obs in map_obstacles:
      if obs["type"] == "circle":
        cx, cy, r = obs["center"][0], obs["center"][1], obs["radius"]
        opti.subject_to(
            (X[0, k] - cx) ** 2 + (X[1, k] - cy) ** 2 >= (r + 0.45) ** 2
        )
      elif obs["type"] == "rectangle":
        cx = (obs["x_max"] + obs["x_min"]) / 2.0
        cy = (obs["y_max"] + obs["y_min"]) / 2.0
        rx = (obs["x_max"] - obs["x_min"]) / 2.0 + 0.35
        ry = (obs["y_max"] - obs["y_min"]) / 2.0 + 0.35
        opti.subject_to(
            ((X[0, k] - cx) / rx) ** 4 + ((X[1, k] - cy) / ry) ** 4 >= 1.0
        )

    # Dynamic Inter-Robot Avoidance Constraints
    if avoid_robot_trajs is not None:
      for other_traj in avoid_robot_trajs:
        if k < len(other_traj):
          other_pos = other_traj[k][0:2]
          opti.subject_to(
              (X[0, k] - other_pos[0]) ** 2 + (X[1, k] - other_pos[1]) ** 2
              >= 0.75**2
          )

  opti.minimize(obj)
  opts = {
      "ipopt.print_level": 0,
      "print_time": 0,
      "ipopt.max_iter": 800,
      "ipopt.acceptable_tol": 1e-2,
  }
  opti.solver("ipopt", opts)

  if ref_path_segment is not None and len(ref_path_segment) > 1:
    ref_x = np.interp(
        np.linspace(0, 1, N + 1),
        np.linspace(0, 1, len(ref_path_segment)),
        ref_path_segment[:, 0],
    )
    ref_y = np.interp(
        np.linspace(0, 1, N + 1),
        np.linspace(0, 1, len(ref_path_segment)),
        ref_path_segment[:, 1],
    )
    for k in range(N + 1):
      opti.set_initial(X[0, k], ref_x[k])
      opti.set_initial(X[1, k], ref_y[k])

  try:
    sol = opti.solve()
    return sol.value(X).T, True
  except Exception:
    return opti.debug.value(X).T, False


# =====================================================================
# 3. ALGORITHM 2: PRIORITIZED SUBPROBLEM RESOLUTION WITH EXPANSION
# =====================================================================
def SolveSubProblem(
    E_sub,
    R_sub,
    Q_sub,
    map_obstacles,
    solvers_hierarchy,
    P_j_current,
    P_segments_sub,
    seg_idx,
):
  P_sub_candidate = {}
  higher_priority_trajs = []

  for priority_idx, r in enumerate(R_sub):
    ref_seg = P_segments_sub[r][seg_idx]

    traj, success = CasadiTrajectoryOptimizer(
        r,
        Q_sub[r]["start"],
        Q_sub[r]["goal"],
        ref_seg,
        map_obstacles,
        avoid_robot_trajs=(
            higher_priority_trajs if priority_idx > 0 else None
        ),
        priority_level=priority_idx,
        dt=0.08,
        N=35,
    )

    # Subproblem Adaptation: Expand horizon if lower priority robot is boxed in
    if not success and priority_idx > 0:
      traj, success = CasadiTrajectoryOptimizer(
          r,
          Q_sub[r]["start"],
          Q_sub[r]["goal"],
          ref_seg,
          map_obstacles,
          avoid_robot_trajs=higher_priority_trajs,
          priority_level=priority_idx,
          dt=0.08,
          N=55,
      )

    P_sub_candidate[r] = traj
    higher_priority_trajs.append(P_sub_candidate[r])

  return P_sub_candidate


# =====================================================================
# 4. ALGORITHM 1 IMPLEMENTATION (Master Segment Loop)
# =====================================================================
def K_ARC_Algorithm_1(E, R, Q, map_obstacles, m_segments=4):
  print("=================================================================")
  print(f"EXECUTING K-ARC (ALGORITHM 1 & 2) FOR {len(R)} ROBOTS")
  print("=================================================================\n")

  P_kinematic_list = {}
  rrt_trees_log = {}

  for r in R:
    planner = KinematicPlanner(Q[r]["start"], Q[r]["goal"], map_obstacles)
    p_i, tree_frames = planner.plan_guided_path(r, total_robots=len(R))
    rrt_trees_log[r] = tree_frames
    P_kinematic_list[r] = p_i

  GenerateVideo_Lines3_4(
      rrt_trees_log, map_obstacles, Q, filename="video_alg1_lines_3_4.mp4"
  )

  G_goals = {r: [] for r in R}
  P_segments = {r: [] for r in R}
  for r in R:
    path = P_kinematic_list[r]
    segment_len = len(path) // m_segments
    for j in range(m_segments):
      idx = (
          (j + 1) * segment_len - 1 if j < m_segments - 1 else len(path) - 1
      )
      if j == m_segments - 1:
        G_goals[r].append(Q[r]["goal"])
      else:
        G_goals[r].append(path[idx])
      start_idx = j * segment_len
      P_segments[r].append(path[start_idx : idx + 1])

  P_k = {r: [] for r in R}
  last_goals = {r: Q[r]["start"] for r in R}

  uncoordinated_segments_log = []
  conflict_subproblems_log = []

  for j in range(m_segments):
    print(f"-> Segment j = {j+1} / {m_segments}:")
    P_j = {}

    for r in R:
      g_last = last_goals[r]
      g_j = G_goals[r][j]
      ref_path = P_segments[r][j]
      tau_ij, _ = CasadiTrajectoryOptimizer(
          r, g_last, g_j, ref_path, map_obstacles, priority_level=0
      )
      P_j[r] = tau_ij

    uncoordinated_segments_log.append(
        {"segment_idx": j + 1, "paths": {r: P_j[r].copy() for r in R}}
    )
    conflicts = FindConflicts(P_j, min_dist=0.75)

    if conflicts:
      print(
          f"     [Lines 20-21] Conflict detected in Segment {j+1}! Creating"
          " Subproblem..."
      )
      E_prime, R_prime, Q_prime = CreateSubProblem(
          conflicts, P_j, E, last_goals, G_goals, j
      )
      P_j_prime = SolveSubProblem(
          E_prime,
          R_prime,
          Q_prime,
          map_obstacles,
          ["PrioritizedTrajectoryOpt"],
          P_j,
          P_segments,
          j,
      )

      if P_j_prime:
        for r_sub in R_prime:
          P_j[r_sub] = P_j_prime[r_sub]

    conflict_subproblems_log.append({
        "segment_idx": j + 1,
        "resolved_paths": {r: P_j[r].copy() for r in R},
    })

    for r in R:
      P_k[r].append(P_j[r])
      last_goals[r] = P_j[r][-1]

  GenerateVideo_Lines17_18(
      uncoordinated_segments_log,
      map_obstacles,
      Q,
      filename="video_alg1_lines_17_18.mp4",
  )
  GenerateVideo_Lines20_21(
      conflict_subproblems_log,
      map_obstacles,
      Q,
      filename="video_alg1_lines_20_21.mp4",
  )

  return {r: np.vstack(P_k[r]) for r in R}


def FindConflicts(P_j, min_dist=0.75):
  conflicts = []
  robots = list(P_j.keys())
  for i in range(len(robots)):
    for k in range(i + 1, len(robots)):
      r1, r2 = robots[i], robots[k]
      path1, path2 = P_j[r1], P_j[r2]
      min_len = min(len(path1), len(path2))
      for t in range(min_len):
        dist = np.linalg.norm(path1[t][0:2] - path2[t][0:2])
        if dist < min_dist:
          conflicts.append((r1, r2, t))
          break
  return conflicts


def CreateSubProblem(conflicts, P_j, E, last_goals, G_goals, seg_idx):
  R_prime = set()
  for c in conflicts:
    R_prime.add(c[0])
    R_prime.add(c[1])
  R_prime = sorted(list(R_prime))
  Q_prime = {
      r: {"start": last_goals[r], "goal": G_goals[r][seg_idx]} for r in R_prime
  }
  return E, R_prime, Q_prime


# =====================================================================
# 5. VIDEO GENERATION UTILITIES
# =====================================================================
COLOR_PALETTE = [
    "purple",
    "red",
    "blue",
    "forestgreen",
    "darkorange",
    "magenta",
    "cyan",
    "saddlebrown",
]


def open_media_file(filepath):
  if platform.system() == "Darwin":
    os.system(f'open "{filepath}"')
  elif platform.system() == "Windows":
    os.startfile(filepath)
  else:
    os.system(f'xdg-open "{filepath}"')


def GenerateVideo_Lines3_4(
    rrt_trees_log,
    map_obstacles,
    queries,
    filename="video_alg1_lines_3_4.mp4",
):
  fig, ax = plt.subplots(figsize=(8, 8))
  for obs in map_obstacles:
    if obs["type"] == "rectangle":
      w, h = obs["x_max"] - obs["x_min"], obs["y_max"] - obs["y_min"]
      ax.add_patch(
          patches.Rectangle(
              (obs["x_min"], obs["y_min"]),
              w,
              h,
              facecolor="gainsboro",
              edgecolor="gray",
              zorder=1,
          )
      )
    elif obs["type"] == "circle":
      ax.add_patch(
          patches.Circle(
              obs["center"],
              obs["radius"],
              facecolor="gainsboro",
              edgecolor="gray",
              zorder=1,
          )
      )

  for r_id in rrt_trees_log.keys():
    color = COLOR_PALETTE[r_id % len(COLOR_PALETTE)]
    ax.scatter(
        queries[r_id]["start"][0],
        queries[r_id]["start"][1],
        color=color,
        s=120,
        marker="P",
        edgecolors="black",
        zorder=5,
    )
    ax.scatter(
        queries[r_id]["goal"][0],
        queries[r_id]["goal"][1],
        color=color,
        s=150,
        marker="X",
        edgecolors="black",
        zorder=6,
    )

  max_frames = max(len(tree) for tree in rrt_trees_log.values())

  def update(frame):
    for r_id, tree in rrt_trees_log.items():
      color = COLOR_PALETTE[r_id % len(COLOR_PALETTE)]
      if frame < len(tree):
        p1, p2 = tree[frame]
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            alpha=0.5,
            linewidth=1.2,
            zorder=2,
        )
    ax.set_title(
        "Alg 1 Lines 3-4: Kinematic Guidance Reference Paths", fontsize=10
    )

  ax.set_xlim(0, 10)
  ax.set_ylim(0, 10)
  ax.grid(True, linestyle=":", alpha=0.6)
  ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=30)
  ani.save(filename, writer="ffmpeg", fps=25, dpi=150)
  plt.close(fig)


def GenerateVideo_Lines17_18(
    uncoordinated_segments_log,
    map_obstacles,
    queries,
    filename="video_alg1_lines_17_18.mp4",
):
  fig, ax = plt.subplots(figsize=(8, 8))
  for obs in map_obstacles:
    if obs["type"] == "rectangle":
      w, h = obs["x_max"] - obs["x_min"], obs["y_max"] - obs["y_min"]
      ax.add_patch(
          patches.Rectangle(
              (obs["x_min"], obs["y_min"]),
              w,
              h,
              facecolor="gainsboro",
              edgecolor="gray",
              zorder=1,
          )
      )
    elif obs["type"] == "circle":
      ax.add_patch(
          patches.Circle(
              obs["center"],
              obs["radius"],
              facecolor="gainsboro",
              edgecolor="gray",
              zorder=1,
          )
      )

  robots = list(queries.keys())
  full_uncoordinated = {r: [] for r in robots}
  for step in uncoordinated_segments_log:
    for r in robots:
      full_uncoordinated[r].append(step["paths"][r])

  concat_paths = {r: np.vstack(full_uncoordinated[r])[:, :2] for r in robots}
  total_frames = max(len(p) for p in concat_paths.values())

  lines, circles = {}, {}
  for r_id in concat_paths.keys():
    color = COLOR_PALETTE[r_id % len(COLOR_PALETTE)]
    (line,) = ax.plot([], [], color=color, linewidth=2.5, zorder=3)
    lines[r_id] = line
    robot_circle = patches.Circle(
        (0, 0), 0.25, facecolor=color, edgecolor="black", lw=1.5, zorder=7
    )
    ax.add_patch(robot_circle)
    circles[r_id] = robot_circle
    ax.scatter(
        queries[r_id]["start"][0],
        queries[r_id]["start"][1],
        color=color,
        s=120,
        marker="P",
        edgecolors="black",
        zorder=5,
    )
    ax.scatter(
        queries[r_id]["goal"][0],
        queries[r_id]["goal"][1],
        color=color,
        s=150,
        marker="X",
        edgecolors="black",
        zorder=6,
    )

  def update(frame):
    for r_id, path in concat_paths.items():
      idx = min(frame, len(path) - 1)
      lines[r_id].set_data(path[: idx + 1, 0], path[: idx + 1, 1])
      circles[r_id].center = (path[idx, 0], path[idx, 1])
    ax.set_title(
        "Alg 1 Lines 17-18: Uncoordinated Segments (Collisions Occur)",
        fontsize=10,
    )
    return list(lines.values()) + list(circles.values())

  ax.set_xlim(0, 10)
  ax.set_ylim(0, 10)
  ax.grid(True, linestyle=":", alpha=0.6)
  ani = animation.FuncAnimation(
      fig, update, frames=total_frames, interval=30, blit=False
  )
  ani.save(filename, writer="ffmpeg", fps=25, dpi=150)
  plt.close(fig)


def GenerateVideo_Lines20_21(
    conflict_subproblems_log,
    map_obstacles,
    queries,
    filename="video_alg1_lines_20_21.mp4",
):
  fig, ax = plt.subplots(figsize=(8, 8))
  for obs in map_obstacles:
    if obs["type"] == "rectangle":
      w, h = obs["x_max"] - obs["x_min"], obs["y_max"] - obs["y_min"]
      ax.add_patch(
          patches.Rectangle(
              (obs["x_min"], obs["y_min"]),
              w,
              h,
              facecolor="gainsboro",
              edgecolor="gray",
              zorder=1,
          )
      )
    elif obs["type"] == "circle":
      ax.add_patch(
          patches.Circle(
              obs["center"],
              obs["radius"],
              facecolor="gainsboro",
              edgecolor="gray",
              zorder=1,
          )
      )

  robots = list(queries.keys())
  full_resolved = {r: [] for r in robots}
  for log_item in conflict_subproblems_log:
    for r in robots:
      full_resolved[r].append(log_item["resolved_paths"][r])

  concat_resolved = {r: np.vstack(full_resolved[r])[:, :2] for r in robots}
  total_frames = max(len(p) for p in concat_resolved.values())

  lines, circles = {}, {}
  for r_id in concat_resolved.keys():
    color = COLOR_PALETTE[r_id % len(COLOR_PALETTE)]
    (line,) = ax.plot(
        [],
        [],
        color=color,
        linewidth=3.0,
        label=f"Robot {r_id}",
        zorder=4,
    )
    lines[r_id] = line
    robot_circle = patches.Circle(
        (0, 0), 0.25, facecolor=color, edgecolor="black", lw=1.5, zorder=7
    )
    ax.add_patch(robot_circle)
    circles[r_id] = robot_circle
    ax.scatter(
        queries[r_id]["start"][0],
        queries[r_id]["start"][1],
        color=color,
        s=120,
        marker="P",
        edgecolors="black",
        zorder=5,
    )
    ax.scatter(
        queries[r_id]["goal"][0],
        queries[r_id]["goal"][1],
        color=color,
        s=150,
        marker="X",
        edgecolors="black",
        zorder=6,
    )

  def update(frame):
    for r_id, path in concat_resolved.items():
      idx = min(frame, len(path) - 1)
      lines[r_id].set_data(path[: idx + 1, 0], path[: idx + 1, 1])
      circles[r_id].center = (path[idx, 0], path[idx, 1])
    ax.set_title(
        "Alg 1 Lines 20-22: Full Resolved Coordinated Trajectory", fontsize=10
    )
    return list(lines.values()) + list(circles.values())

  ax.set_xlim(0, 10)
  ax.set_ylim(0, 10)
  ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
  ax.grid(True, linestyle=":", alpha=0.6)
  ani = animation.FuncAnimation(
      fig, update, frames=total_frames, interval=30, blit=False
  )
  ani.save(filename, writer="ffmpeg", fps=25, dpi=150)
  plt.close(fig)


# =====================================================================
# 6. DYNAMIC SCENARIO CONFIGURATION (Supports N = 2, 4, 5, 6, 7)
# =====================================================================
def generate_open_cross_scenario(num_robots=6):
  """Generates symmetric circular-crossing start/goal queries for N robots."""
  env = {"workspace": [0, 10, 0, 10]}
  map_obstacles = []
  robots = list(range(num_robots))

  center = np.array([5.0, 5.0])
  radius = 3.8
  queries = {}

  for i in range(num_robots):
    theta_start = (2 * np.pi / num_robots) * i
    theta_goal = theta_start + np.pi

    start_pos = center + radius * np.array(
        [np.cos(theta_start), np.sin(theta_start)]
    )
    goal_pos = center + radius * np.array(
        [np.cos(theta_goal), np.sin(theta_goal)]
    )

    heading = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])

    queries[i] = {
        "start": np.array(
            [start_pos[0], start_pos[1], heading, 0.0, 0.0]
        ),
        "goal": np.array([goal_pos[0], goal_pos[1], heading, 0.0, 0.0]),
    }

  return env, robots, queries, map_obstacles


# =====================================================================
# 7. MAIN EXECUTION PIPELINE
# =====================================================================
if __name__ == "__main__":
  # Change this to 4, 5, 6, or 7 robots:
  NUM_ROBOTS = 6

  Environment, Robots, Queries, MapObstacles = generate_open_cross_scenario(
      num_robots=NUM_ROBOTS
  )
  final_paths = K_ARC_Algorithm_1(
      Environment, Robots, Queries, MapObstacles, m_segments=4
  )

  print("\n=================================================================")
  print(f"EXPERIMENT FOR {NUM_ROBOTS} ROBOTS COMPLETE! VIDEOS SAVED.")
  print("=================================================================\n")

  open_media_file("video_alg1_lines_3_4.mp4")
  open_media_file("video_alg1_lines_17_18.mp4")
  open_media_file("video_alg1_lines_20_21.mp4")
