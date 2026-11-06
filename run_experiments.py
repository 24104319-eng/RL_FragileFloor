"""
Chạy các thí nghiệm cho đề tài Fragile Floor. Kết quả (hình .png, bảng .csv) lưu vào thư mục results/.

  --exp curves    : Q-learning (nhiều seed) vs Policy Iteration trên 3 map, learning curve + policy cuối
  --exp variants  : so sánh các biến thể (trượt, có/không repair kit, kho báu bắt buộc/tùy chọn)
  --exp rewards   : thử nhiều cấu hình reward (Policy Iteration), xem policy thay đổi thế nào
  --exp failure   : phân tích thất bại / chậm hội tụ của Q-learning (map hard)
  --exp all       : chạy tất cả

Ví dụ:  python run_experiments.py --exp curves --seeds 5
        python run_experiments.py --exp all --seeds 3 --scale 0.2      # chạy nhanh để thử
Trong Colab nên dùng:  %run run_experiments.py --exp curves --seeds 5   (hình hiện ngay trong notebook)
"""
import argparse
import csv
import os
import time

import numpy as np
import matplotlib
try:
    get_ipython  # noqa: F821  (có khi chạy trong notebook)
    SHOW = True
except NameError:
    matplotlib.use("Agg")
    SHOW = False
import matplotlib.pyplot as plt

from common import (load_config, make_env, solve_pi, pi_policy_fn, q_policy_fn,
                    evaluate, rollout)
from q_learning import q_learning
from visualize import plot_trajectory, plot_curve, moving_average


# ---------------------------------------------------------------- tiện ích
def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"  -> đã lưu {path}")
    if SHOW:
        plt.show()
    plt.close(fig)


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> đã lưu {path}")


def print_table(rows):
    cols = list(rows[0].keys())
    w = {c: max(len(str(c)), *(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(str(c).ljust(w[c]) for c in cols))
    print("-+-".join("-" * w[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w[c]) for c in cols))


def qcfg(cfg, map_name, scale=1.0, extra=None):
    p = dict(cfg["q_learning"]["default"])
    p.update(cfg["q_learning"].get(map_name, {}))
    p.update(extra or {})
    mult = p.pop("episodes_mult", 1.0)
    p["episodes"] = max(300, int(p["episodes"] * scale * mult))
    return p


def run_q(factory, params, seeds, label=""):
    """Huấn luyện Q-learning với nhiều seed. Trả về (R, S, evals, Q của seed 0)."""
    R, S, evals, Q0 = [], [], [], None
    for seed in range(seeds):
        t0 = time.time()
        env = factory()
        res = q_learning(env, seed=seed, **params)
        ev = evaluate(env, q_policy_fn(res["Q"]), episodes=300, seed=10_000 + seed)
        R.append(res["returns"])
        S.append(res["success"])
        evals.append(ev)
        if seed == 0:
            Q0 = res["Q"]
        print(f"   [{label}] seed {seed}: greedy return={ev['mean_return']:.1f}, "
              f"thành công={ev['success_rate']:.0%}, {time.time() - t0:.0f}s", flush=True)
    return np.array(R), np.array(S), evals, Q0


def ms(evals, key, pct=False):
    v = np.array([e[key] for e in evals])
    return f"{v.mean():.0%} ± {v.std():.0%}" if pct else f"{v.mean():.1f} ± {v.std():.1f}"


def episodes_to_reach(R, ref, w, frac=0.9):
    """Episode đầu tiên mà return TB trượt (có epsilon) đạt >= 90% return tối ưu của PI."""
    m = moving_average(R.mean(axis=0), w)
    idx = np.flatnonzero(m >= ref - (1 - frac) * abs(ref))
    return str(int(idx[0]) + w) if len(idx) else "chưa đạt"


def window_for(episodes):
    return max(20, episodes // 50)


# ---------------------------------------------------------------- thí nghiệm 1
def exp_curves(cfg, args):
    maps = args.maps
    n = len(maps)
    fig_c, ax_c = plt.subplots(1, n, figsize=(5.5 * n, 4), squeeze=False)
    fig_p, ax_p = plt.subplots(2, n, figsize=(5 * n, 7.5), squeeze=False)
    rows = []
    for j, m in enumerate(maps):
        print(f"\n=== [curves] map {m} ===")
        env = make_env(m, cfg=cfg)
        model, V, pol, info, pid = solve_pi(env)
        pi_fn = pi_policy_fn(pid)
        pi_ev = evaluate(env, pi_fn, episodes=1)               # không trượt -> xác định
        params = qcfg(cfg, m, args.scale)
        R, S, evals, Q0 = run_q(lambda: make_env(m, cfg=cfg), params, args.seeds, m)
        w = window_for(params["episodes"])

        plot_curve(ax_c[0][j], R, w, label="Q-learning (TB ± std các seed)", color="tab:blue")
        ax_c[0][j].axhline(pi_ev["mean_return"], ls="--", color="k", label="Policy Iteration (tối ưu)")
        ax_c[0][j].set_title(f"map {m}: {params['episodes']} episode, {args.seeds} seed")
        ax_c[0][j].set_xlabel("episode")
        ax_c[0][j].set_ylabel(f"return (TB trượt {w})")
        ax_c[0][j].legend(fontsize=8)

        plot_trajectory(ax_p[0][j], env, rollout(env, pi_fn), f"Policy Iteration - {m}")
        plot_trajectory(ax_p[1][j], env, rollout(env, q_policy_fn(Q0)), f"Q-learning (seed 0) - {m}")

        rows.append({
            "map": m, "|S| đầy đủ": env.n_states, "|S| reachable": model.n, "|A|": env.n_actions,
            "PI vòng lặp": info["iterations"], "PI thời gian(s)": f"{info['time']:.2f}",
            "PI return": f"{pi_ev['mean_return']:.0f}", "PI số bước": f"{pi_ev['mean_steps']:.0f}",
            "Q episode": params["episodes"], "Q return": ms(evals, "mean_return"),
            "Q tỉ lệ tới goal": ms(evals, "success_rate", pct=True),
            "Q đạt 90% PI sau (episode)": episodes_to_reach(R, pi_ev["mean_return"], w),
        })
    finish(fig_c, os.path.join(args.out, "curves_learning.png"))
    finish(fig_p, os.path.join(args.out, "curves_policies.png"))
    print()
    print_table(rows)
    save_csv(os.path.join(args.out, "curves_table.csv"), rows)


# ---------------------------------------------------------------- thí nghiệm 2
def exp_variants(cfg, args):
    m = args.map or cfg["variants_map"]
    variants = cfg["variants"]
    n = len(variants)
    fig_c, ax_c = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig_p, ax_p = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    rows = []
    for j, (name, ov) in enumerate(variants.items()):
        print(f"\n=== [variants] map {m}: {name} ===")
        env = make_env(m, cfg=cfg, **ov)
        model, V, pol, info, pid = solve_pi(env)
        pi_fn = pi_policy_fn(pid)
        pi_ev = evaluate(env, pi_fn, episodes=300)
        params = qcfg(cfg, m, args.scale)
        R, S, evals, Q0 = run_q(lambda ov=ov: make_env(m, cfg=cfg, **ov), params, args.seeds, name)
        w = window_for(params["episodes"])

        plot_curve(ax_c[0][j], R, w, label="Q-learning", color="tab:blue")
        ax_c[0][j].axhline(pi_ev["mean_return"], ls="--", color="k", label="PI (tối ưu)")
        ax_c[0][j].set_title(name, fontsize=9)
        ax_c[0][j].set_xlabel("episode")
        ax_c[0][j].legend(fontsize=8)

        env_vis = make_env(m, cfg=cfg, **{**ov, "slip_prob": 0.0})     # vẽ đường đi không trượt
        plot_trajectory(ax_p[0][j], env_vis, rollout(env_vis, pi_fn), f"PI - {name}")
        plot_trajectory(ax_p[1][j], env_vis, rollout(env_vis, q_policy_fn(Q0)), f"Q-learning - {name}")

        rows.append({
            "biến thể": name, "|S| reachable": model.n,
            "PI return": f"{pi_ev['mean_return']:.1f}", "PI tới goal": f"{pi_ev['success_rate']:.0%}",
            "PI repair/ván": f"{pi_ev['repairs']:.2f}", "PI kho báu/ván": f"{pi_ev['treasures']:.2f}",
            "Q return": ms(evals, "mean_return"), "Q tới goal": ms(evals, "success_rate", pct=True),
            "Q repair/ván": f"{np.mean([e['repairs'] for e in evals]):.2f}",
            "Q kho báu/ván": f"{np.mean([e['treasures'] for e in evals]):.2f}",
        })
    finish(fig_c, os.path.join(args.out, "variants_learning.png"))
    finish(fig_p, os.path.join(args.out, "variants_policies.png"))
    print()
    print_table(rows)
    save_csv(os.path.join(args.out, "variants_table.csv"), rows)


# ---------------------------------------------------------------- thí nghiệm 3
def exp_rewards(cfg, args):
    exp = cfg["rewards_exp"]
    m = args.map or exp["map"]
    configs = exp["configs"]
    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    rows = []
    for j, (name, ov) in enumerate(configs.items()):
        kw = {**exp["base"], **ov}
        print(f"\n=== [rewards] map {m}: {name} ===")
        env = make_env(m, cfg=cfg, **kw)
        model, V, pol, info, pid = solve_pi(env)
        pi_fn = pi_policy_fn(pid)
        ev = evaluate(env, pi_fn, episodes=500)
        env_vis = make_env(m, cfg=cfg, **{**kw, "slip_prob": 0.0})
        plot_trajectory(axes[0][j], env_vis, rollout(env_vis, pi_fn), name)
        rows.append({
            "cấu hình": name, "V(start)": f"{V[model.start_idx].mean():.1f}",
            "return TB": f"{ev['mean_return']:.1f}", "tới goal": f"{ev['success_rate']:.0%}",
            "số bước TB": f"{ev['mean_steps']:.1f}", "repair/ván": f"{ev['repairs']:.2f}",
            "kho báu/ván": f"{ev['treasures']:.2f}",
        })
    finish(fig, os.path.join(args.out, "rewards_policies.png"))
    print("\n(Lưu ý: return của các cấu hình khác nhau KHÔNG so sánh trực tiếp được vì thang reward khác nhau;"
          " hãy so sánh hành vi: tới goal, số bước, repair, kho báu.)")
    print_table(rows)
    save_csv(os.path.join(args.out, "rewards_table.csv"), rows)


# ---------------------------------------------------------------- thí nghiệm 4
def exp_failure(cfg, args):
    exp = cfg["failure_exp"]
    m = args.map or exp["map"]
    seeds = min(args.seeds, 3)
    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(13, 4.5))
    rows = []
    for k, (name, sc) in enumerate(exp["scenarios"].items()):
        print(f"\n=== [failure] map {m}: {name} ===")
        ov = sc.get("env", {})
        params = qcfg(cfg, m, args.scale, sc.get("q"))
        env = make_env(m, cfg=cfg, **ov)
        model, V, pol, info, pid = solve_pi(env)
        pi_ev = evaluate(env, pi_policy_fn(pid), episodes=300)
        R, S, evals, _ = run_q(lambda ov=ov: make_env(m, cfg=cfg, **ov), params, seeds, name)
        w = window_for(params["episodes"])
        color = f"C{k}"
        plot_curve(ax_r, R, w, label=name, color=color)
        plot_curve(ax_s, S, w, label=name, color=color)
        rows.append({
            "kịch bản": name, "Q episode": params["episodes"], "alpha": params["alpha"],
            "eps_decay_frac": params["eps_decay_frac"],
            "PI return": f"{pi_ev['mean_return']:.1f}", "PI tới goal": f"{pi_ev['success_rate']:.0%}",
            "Q return": ms(evals, "mean_return"), "Q tới goal": ms(evals, "success_rate", pct=True),
            "Q đạt 90% PI sau": episodes_to_reach(R, pi_ev["mean_return"], w),
        })
    ax_r.set_title(f"Return khi huấn luyện (map {m})")
    ax_s.set_title("Tỉ lệ episode tới goal khi huấn luyện")
    for ax in (ax_r, ax_s):
        ax.set_xlabel("episode")
        ax.legend(fontsize=8)
    finish(fig, os.path.join(args.out, "failure_curves.png"))
    print()
    print_table(rows)
    save_csv(os.path.join(args.out, "failure_table.csv"), rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["curves"],
                    choices=["curves", "variants", "rewards", "failure", "all"])
    ap.add_argument("--seeds", type=int, default=5, help="số seed Q-learning")
    ap.add_argument("--scale", type=float, default=1.0, help="nhân số episode (0.1 = chạy nhanh để thử)")
    ap.add_argument("--maps", nargs="+", default=["easy", "medium", "hard"], help="map cho --exp curves")
    ap.add_argument("--map", default=None, help="ghi đè map cho variants/rewards/failure")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config()
    todo = ["curves", "variants", "rewards", "failure"] if "all" in args.exp else args.exp
    funcs = {"curves": exp_curves, "variants": exp_variants, "rewards": exp_rewards, "failure": exp_failure}
    for name in todo:
        funcs[name](cfg, args)


if __name__ == "__main__":
    main()
