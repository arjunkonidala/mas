import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.optimize import linprog, minimize


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SEED = 3403
rng = np.random.default_rng(SEED)
NAME = "ARJUN"


def connected_erdos_renyi(n, p, rng):
    while True:
        a = rng.random((n, n)) < p
        a = np.triu(a, 1)
        a = a + a.T
        if is_connected(a):
            return a.astype(float)


def is_connected(a):
    n = a.shape[0]
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        for j in np.flatnonzero(a[i]):
            if j not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == n


def graph_laplacian(a):
    return np.diag(a.sum(axis=1)) - a


def metropolis_weights(a):
    n = a.shape[0]
    degrees = a.sum(axis=1)
    w = np.zeros_like(a, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and a[i, j] > 0:
                w[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
    np.fill_diagonal(w, 1.0 - w.sum(axis=1))
    return w


def plot_graph(a, filename, title):
    n = a.shape[0]
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.c_[np.cos(theta), np.sin(theta)]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] > 0:
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="#9aa0a6", lw=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=230, color="#2f80ed", edgecolor="white", zorder=3)
    for i, (x, y) in enumerate(pos):
        ax.text(x, y, str(i + 1), ha="center", va="center", color="white", fontsize=9)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def sample_strokes(strokes, n=20):
    """Return n evenly spaced points on a list of letter strokes."""
    dense = []
    for stroke in strokes:
        stroke = np.asarray(stroke, dtype=float)
        for a, b in zip(stroke[:-1], stroke[1:]):
            length = np.linalg.norm(b - a)
            count = max(3, int(30 * length))
            for s in np.linspace(0.0, 1.0, count, endpoint=False):
                dense.append((1.0 - s) * a + s * b)
    dense = np.asarray(dense)
    idx = np.linspace(0, len(dense) - 1, n).astype(int)
    pts = dense[idx]
    pts -= pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    return pts / max(scale, 1e-9) * 4.0


def letter_targets(letter, n=20):
    letter = letter.upper()
    strokes = {
        "A": [
            [(-1.0, -1.0), (0.0, 1.0), (1.0, -1.0)],
            [(-0.45, -0.05), (0.45, -0.05)],
        ],
        "R": [
            [(-0.8, -1.0), (-0.8, 1.0)],
            [(-0.8, 1.0), (0.35, 1.0), (0.65, 0.55), (0.35, 0.15), (-0.8, 0.15)],
            [(-0.2, 0.15), (0.85, -1.0)],
        ],
        "J": [
            [(-0.9, 1.0), (0.9, 1.0)],
            [(0.45, 1.0), (0.45, -0.55), (0.15, -0.95), (-0.45, -0.95), (-0.75, -0.55)],
        ],
        "U": [
            [(-0.8, 1.0), (-0.8, -0.45), (-0.55, -0.9), (0.0, -1.0),
             (0.55, -0.9), (0.8, -0.45), (0.8, 1.0)],
        ],
        "N": [
            [(-0.85, -1.0), (-0.85, 1.0)],
            [(-0.85, 1.0), (0.85, -1.0)],
            [(0.85, -1.0), (0.85, 1.0)],
        ],
    }
    if letter not in strokes:
        raise ValueError(f"No stroke template is defined for letter {letter!r}")
    return sample_strokes(strokes[letter], n)


def simulate_problem1():
    name = NAME
    n = 20
    a = connected_erdos_renyi(n, 0.28, rng)
    l = graph_laplacian(a)
    plot_graph(a, "p1_agent_graph.png", "Formation-control communication graph")
    dt = 0.025
    anchor_gain = 1.0
    formation_gain = 0.45
    x = rng.normal(size=(n, 2)) * 2.0

    snapshots = []
    frames = []
    errors = {}
    for letter in name:
        target = letter_targets(letter, n)
        for k in range(170):
            # Displacement formation law over the communication graph.
            error = x - target
            u = -anchor_gain * error - formation_gain * (l @ error)
            x = x + dt * u
            if k % 7 == 0:
                frames.append((letter, x.copy()))
        final_error = np.linalg.norm(x - target, axis=1)
        errors[letter] = {
            "mean": float(final_error.mean()),
            "max": float(final_error.max()),
        }
        snapshots.append((letter, x.copy(), target.copy()))

    fig, axes = plt.subplots(1, len(snapshots), figsize=(2.4 * len(snapshots), 2.6))
    for ax, (letter, pos, target) in zip(axes, snapshots):
        ax.scatter(pos[:, 0], pos[:, 1], s=30, color="#1f77b4")
        ax.scatter(target[:, 0], target[:, 1], s=12, color="#ef476f", alpha=0.65)
        ax.set_title(letter)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"Sequential 20-agent formation for name {name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p1_formation_sequence.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame):
        letter, pos = frame
        ax.clear()
        ax.scatter(pos[:, 0], pos[:, 1], s=45, color="#1f77b4")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title(f"Formation control: {letter}")
        ax.grid(True, alpha=0.2)

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=70)
    try:
        ani.save(FIG_DIR / "p1_formation_sequence.gif", writer="pillow", fps=14)
    except Exception:
        pass
    plt.close(fig)
    return a, errors


def simulate_problem2():
    n = 12
    tmax = 60
    a = connected_erdos_renyi(n, 0.35, rng)
    w = metropolis_weights(a)
    plot_graph(a, "p2_drone_graph.png", "Connected Erdos-Renyi communication graph")

    sensor_pos = rng.uniform(-8, 8, size=(n, 3))
    z = np.zeros((tmax + 1, 3))
    z[0] = rng.multivariate_normal([1.0, -2.0, 2.0], np.diag([1.0, 1.0, 0.6]))
    sigma_w = np.diag([0.08, 0.08, 0.05])
    sigmas_v = np.array([np.diag(rng.uniform(0.06, 0.25, size=3)) for _ in range(n)])
    estimates = np.zeros_like(z)
    centralized = np.zeros_like(z)

    alpha = 0.004
    consensus_steps = 450
    for t in range(tmax + 1):
        if t > 0:
            z[t] = z[t - 1] + rng.multivariate_normal(np.zeros(3), sigma_w)
        measurements = np.zeros((n, 3))
        for i in range(n):
            noise = rng.multivariate_normal(np.zeros(3), sigmas_v[i])
            y_i = sensor_pos[i] - z[t] + noise
            measurements[i] = sensor_pos[i] - y_i

        information = np.zeros((3, 3))
        rhs = np.zeros(3)
        for i in range(n):
            inv_r = np.linalg.inv(sigmas_v[i])
            information += inv_r
            rhs += inv_r @ measurements[i]
        centralized[t] = np.linalg.solve(information, rhs)

        local = measurements.copy()
        grad = np.zeros_like(local)
        for i in range(n):
            inv_r = np.linalg.inv(sigmas_v[i])
            grad[i] = inv_r @ (local[i] - measurements[i])
        tracker = grad.copy()
        for _ in range(consensus_steps):
            next_local = w @ local - alpha * tracker
            next_grad = np.zeros_like(local)
            for i in range(n):
                inv_r = np.linalg.inv(sigmas_v[i])
                next_grad[i] = inv_r @ (next_local[i] - measurements[i])
            tracker = w @ tracker + next_grad - grad
            local = next_local
            grad = next_grad
        estimates[t] = local.mean(axis=0)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
               marker="^", s=55, label="drones/sensors", color="#2f80ed")
    ax.plot(z[:, 0], z[:, 1], z[:, 2], label="true intruder", color="#111111", lw=2)
    ax.plot(estimates[:, 0], estimates[:, 1], estimates[:, 2],
            label="distributed estimate", color="#ef476f", lw=2, ls="--")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_tracking_trajectory.png", dpi=220)
    plt.close(fig)

    err = estimates - z
    centralized_err = centralized - z
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(np.linalg.norm(err, axis=1), color="#ef476f", lw=2)
    ax.plot(np.linalg.norm(centralized_err, axis=1), color="#2f80ed", lw=1.4, ls="--",
            label="centralized ML benchmark")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|\hat z(t)-z(t)\|_2$")
    ax.set_title("Distributed estimation error")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "p2_tracking_error.png", dpi=220)
    plt.close(fig)
    return {
        "graph_edges": int(a.sum() // 2),
        "weight_row_error": float(np.max(np.abs(w.sum(axis=1) - 1.0))),
        "weight_symmetry_error": float(np.max(np.abs(w - w.T))),
        "mean_error": float(np.mean(np.linalg.norm(err, axis=1))),
        "max_error": float(np.max(np.linalg.norm(err, axis=1))),
        "mean_centralized_error": float(np.mean(np.linalg.norm(centralized_err, axis=1))),
        "mean_gap_to_centralized": float(np.mean(np.linalg.norm(estimates - centralized, axis=1))),
    }


def capped_simplex_projection(v, total):
    lo = np.min(v) - 1.0
    hi = np.max(v)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        x = np.clip(v - mid, 0.0, 1.0)
        if x.sum() > total:
            lo = mid
        else:
            hi = mid
    return np.clip(v - hi, 0.0, 1.0)


def graph_edges(a):
    return [(i, j) for i in range(a.shape[0]) for j in range(i + 1, a.shape[1]) if a[i, j] > 0]


def local_assignment_update(agent, previous, edge_terms, cost_row, capacity, rho):
    n, m = previous.shape
    degree = len(edge_terms)
    linear = np.zeros((n, m))
    linear[agent] = cost_row
    center_sum = np.zeros((n, m))
    for z_edge, u_edge in edge_terms:
        center_sum += z_edge - u_edge

    def objective(flat):
        x = flat.reshape(n, m)
        diff = x[None, :, :] - np.array([z - u for z, u in edge_terms])
        return float(np.sum(linear * x) + 0.5 * rho * np.sum(diff * diff))

    def gradient(flat):
        x = flat.reshape(n, m)
        grad = linear + rho * (degree * x - center_sum)
        return grad.reshape(-1)

    constraints = []
    for task in range(m):
        constraints.append({
            "type": "eq",
            "fun": lambda flat, task=task: np.sum(flat.reshape(n, m)[:, task]) - 1.0,
            "jac": lambda flat, task=task: column_constraint_jac(n, m, task),
        })
    constraints.append({
        "type": "eq",
        "fun": lambda flat: np.sum(flat.reshape(n, m)[agent]) - capacity,
        "jac": lambda flat: row_constraint_jac(n, m, agent),
    })

    result = minimize(
        objective,
        previous.reshape(-1),
        jac=gradient,
        bounds=[(0.0, 1.0)] * (n * m),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 200, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Local ADMM update failed for agent {agent + 1}: {result.message}")
    return result.x.reshape(n, m)


def column_constraint_jac(n, m, task):
    jac = np.zeros(n * m)
    jac[task::m] = 1.0
    return jac


def row_constraint_jac(n, m, agent):
    jac = np.zeros(n * m)
    jac[agent * m:(agent + 1) * m] = 1.0
    return jac


def edge_consensus_admm_task_allocation(cost, capacities, adjacency, rho=1.5, iters=500):
    n, m = cost.shape
    edges = graph_edges(adjacency)
    base = np.repeat((capacities / m)[:, None], m, axis=1)
    local = [base.copy() for _ in range(n)]
    z = {edge: base.copy() for edge in edges}
    u = {}
    for i, j in edges:
        u[(i, j)] = np.zeros((n, m))
        u[(j, i)] = np.zeros((n, m))

    consensus_residuals = []
    feasibility_residuals = []
    objectives = []
    for _ in range(iters):
        new_local = []
        for i in range(n):
            terms = []
            for j in np.flatnonzero(adjacency[i]):
                edge = (min(i, j), max(i, j))
                terms.append((z[edge], u[(i, int(j))]))
            new_local.append(local_assignment_update(i, local[i], terms, cost[i], capacities[i], rho))
        local = new_local

        for i, j in edges:
            z[(i, j)] = 0.5 * (local[i] + u[(i, j)] + local[j] + u[(j, i)])
        for i, j in edges:
            u[(i, j)] += local[i] - z[(i, j)]
            u[(j, i)] += local[j] - z[(i, j)]

        consensus = np.sqrt(sum(np.linalg.norm(local[i] - local[j]) ** 2 for i, j in edges))
        average = sum(local) / n
        row_error = np.max(np.abs(average.sum(axis=1) - capacities))
        column_error = np.max(np.abs(average.sum(axis=0) - 1.0))
        feasibility = max(float(row_error), float(column_error))
        consensus_residuals.append(float(consensus))
        feasibility_residuals.append(float(feasibility))
        objectives.append(float(np.sum(cost * average)))
        if consensus < 1e-7 and feasibility < 1e-7:
            break
    return {
        "average": sum(local) / n,
        "locals": local,
        "consensus_residuals": np.array(consensus_residuals),
        "feasibility_residuals": np.array(feasibility_residuals),
        "objectives": np.array(objectives),
        "edges": edges,
    }


def exact_assignment(cost, capacities):
    n, m = cost.shape
    c = cost.reshape(-1)
    a_eq = []
    b_eq = []
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1
        a_eq.append(row)
        b_eq.append(capacities[i])
    for j in range(m):
        col = np.zeros(n * m)
        col[j::m] = 1
        a_eq.append(col)
        b_eq.append(1)
    res = linprog(c, A_eq=np.array(a_eq), b_eq=np.array(b_eq),
                  bounds=(0, 1), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return np.rint(res.x.reshape(n, m)), float(res.fun)


def plot_assignment(x, filename, title):
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.imshow(x, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] > 0.5:
                ax.text(j, i, "1", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlabel("task")
    ax.set_ylabel("agent")
    ax.set_xticks(range(x.shape[1]))
    ax.set_yticks(range(x.shape[0]))
    ax.set_xticklabels(range(1, x.shape[1] + 1))
    ax.set_yticklabels(range(1, x.shape[0] + 1))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=220)
    plt.close(fig)


def simulate_problem3():
    n, m = 6, 14
    capacities = np.array([3, 2, 3, 2, 2, 2])
    a = connected_erdos_renyi(n, 0.45, rng)
    plot_graph(a, "p3_agent_graph.png", "Task-allocation communication graph")
    summaries = []
    for case in range(1, 3):
        cost = rng.uniform(1, 12, size=(n, m))
        if case == 2:
            cost += np.linspace(0, 4, n)[:, None] * rng.uniform(0.3, 1.0, size=(1, m))
        admm = edge_consensus_admm_task_allocation(cost, capacities, a)
        x_relaxed = admm["average"]
        x_binary, lp_objective = exact_assignment(cost, capacities)
        plot_assignment(x_binary, f"p3_assignment_case{case}.png",
                        f"Binary task allocation, cost case {case}")
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.semilogy(admm["consensus_residuals"], color="#2f80ed", lw=2,
                    label="edge consensus")
        ax.semilogy(admm["feasibility_residuals"], color="#ef476f", lw=1.6,
                    label="assignment feasibility")
        ax.set_xlabel("ADMM iteration")
        ax.set_ylabel("residual")
        ax.set_title(f"Graph-distributed ADMM convergence, cost case {case}")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"p3_admm_residual_case{case}.png", dpi=220)
        plt.close(fig)
        summaries.append({
            "case": case,
            "objective": float(np.sum(cost * x_binary)),
            "lp_objective": lp_objective,
            "admm_iterations": int(len(admm["consensus_residuals"])),
            "final_consensus_residual": float(admm["consensus_residuals"][-1]),
            "final_feasibility_residual": float(admm["feasibility_residuals"][-1]),
            "admm_relaxed_objective": float(np.sum(cost * x_relaxed)),
            "admm_gap_to_lp": float(abs(np.sum(cost * x_relaxed) - lp_objective)),
            "relaxed_integrality_error": float(np.max(np.minimum(x_relaxed, 1.0 - x_relaxed))),
            "row_capacity_error": float(np.max(np.abs(x_binary.sum(axis=1) - capacities))),
            "column_assignment_error": float(np.max(np.abs(x_binary.sum(axis=0) - 1.0))),
        })
    return {
        "graph_edges": int(a.sum() // 2),
        "capacities": capacities.tolist(),
        "cases": summaries,
    }


if __name__ == "__main__":
    p1_graph, p1_errors = simulate_problem1()
    p2_stats = simulate_problem2()
    p3_summaries = simulate_problem3()

    lines = []
    lines.append("Assignment 3 validation summary")
    lines.append(f"Random seed: {SEED}")
    lines.append("")
    lines.append("Problem 1")
    lines.append(f"- name: {NAME}")
    lines.append(f"- graph connected: {is_connected(p1_graph)}")
    lines.append(f"- graph nodes: {p1_graph.shape[0]}")
    lines.append(f"- graph edges: {int(p1_graph.sum() // 2)}")
    for letter, stats in p1_errors.items():
        lines.append(
            f"- letter {letter}: mean formation error={stats['mean']:.6f}, "
            f"max formation error={stats['max']:.6f}"
        )
    lines.append("")
    lines.append("Problem 2")
    lines.append(f"- graph edges: {p2_stats['graph_edges']}")
    lines.append(f"- Metropolis row-sum error: {p2_stats['weight_row_error']:.3e}")
    lines.append(f"- Metropolis symmetry error: {p2_stats['weight_symmetry_error']:.3e}")
    lines.append(f"- distributed mean tracking error: {p2_stats['mean_error']:.6f}")
    lines.append(f"- distributed max tracking error: {p2_stats['max_error']:.6f}")
    lines.append(f"- centralized ML mean tracking error: {p2_stats['mean_centralized_error']:.6f}")
    lines.append(f"- mean distributed-to-centralized gap: {p2_stats['mean_gap_to_centralized']:.6f}")
    lines.append("")
    lines.append("Problem 3")
    lines.append(f"- graph edges: {p3_summaries['graph_edges']}")
    lines.append(f"- capacities: {p3_summaries['capacities']}")
    for case in p3_summaries["cases"]:
        lines.append(
            f"- case {case['case']}: objective={case['objective']:.6f}, "
            f"LP objective={case['lp_objective']:.6f}, "
            f"ADMM relaxed objective={case['admm_relaxed_objective']:.6f}, "
            f"iterations={case['admm_iterations']}, "
            f"final consensus residual={case['final_consensus_residual']:.3e}, "
            f"final feasibility residual={case['final_feasibility_residual']:.3e}, "
            f"ADMM-LP objective gap={case['admm_gap_to_lp']:.3e}, "
            f"relaxed integrality error={case['relaxed_integrality_error']:.3e}, "
            f"row capacity error={case['row_capacity_error']:.1e}, "
            f"column assignment error={case['column_assignment_error']:.1e}"
        )
    summary = "\n".join(lines)
    (ROOT / "validation_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
