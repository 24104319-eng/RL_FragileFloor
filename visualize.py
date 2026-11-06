"""Vẽ map, đường đi (trajectory) của policy và learning curve."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TILE_COLORS = {".": "#f7f7f7", "S": "#a6cee3", "G": "#b2df8a", "1": "#fb9a99",
               "2": "#fdbf6f", "H": "#222222", "T": "#ffe066", "#": "#9e9e9e"}


def draw_grid(ax, env):
    for r in range(env.H):
        for c in range(env.W):
            ch = env.grid[r][c]
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=TILE_COLORS.get(ch, "white"),
                                   edgecolor="black", linewidth=0.8))
            if ch not in ".#":
                ax.text(c, r + 0.33, ch, ha="center", va="center", fontsize=9,
                        color="white" if ch == "H" else "dimgray")
    ax.set_xlim(-0.5, env.W - 0.5)
    ax.set_ylim(env.H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_trajectory(ax, env, traj, title=""):
    """Mũi tên màu viridis theo thời gian (tím = đầu, vàng = cuối).
    R đỏ = ô được sửa bằng repair kit, sao = nhặt kho báu, X đỏ = rơi."""
    draw_grid(ax, env)
    pos, T = traj["pos"], max(1, len(traj["pos"]) - 1)
    cmap = plt.get_cmap("viridis")
    for t in range(len(pos) - 1):
        p, q = pos[t], pos[t + 1]
        if p == q:
            continue
        ax.annotate("", xy=(q[1], q[0]), xytext=(p[1], p[0]),
                    arrowprops=dict(arrowstyle="->", color=cmap(t / T), lw=1.8,
                                    connectionstyle="arc3,rad=0.25", shrinkA=7, shrinkB=7))
    for kind, t, p in traj["events"]:
        if kind == "repair":
            ax.text(p[1], p[0], "R", color="red", fontsize=15, fontweight="bold", ha="center", va="center")
        elif kind == "treasure":
            ax.text(p[1], p[0] - 0.3, "★", color="darkgreen", fontsize=14, ha="center", va="center")
        elif kind == "fall":
            ax.text(p[1], p[0], "X", color="red", fontsize=18, fontweight="bold", ha="center", va="center")
    r0, c0 = pos[0]
    ax.plot(c0, r0, "o", color="blue", ms=7)
    res = "tới goal" if traj["success"] else "THẤT BẠI"
    ax.set_title(f"{title}\nreturn={traj['return']:.0f}, {traj['steps']} bước, {res}", fontsize=9)


def moving_average(x, w):
    w = max(1, min(int(w), len(x)))
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[w:] - c[:-w]) / w


def plot_curve(ax, R, window, label=None, color=None):
    """R: mảng (số seed, số episode). Vẽ trung bình trượt ± độ lệch chuẩn giữa các seed."""
    R = np.asarray(R)
    w = max(1, min(int(window), R.shape[1]))
    m = np.array([moving_average(r, w) for r in R])
    x = np.arange(w, R.shape[1] + 1)
    mu, sd = m.mean(axis=0), m.std(axis=0)
    ax.plot(x, mu, label=label, color=color)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.2, color=color)
