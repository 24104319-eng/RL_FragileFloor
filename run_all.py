
import argparse
import os
import pickle
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fragile_floor_env import FragileFloorEnv
from dp_solvers import TabularModel, q_values
from common import load_config, rollout, evaluate
from dp_snap import vi_snap, pi_snap, tpi_snap
from learners import train, q_policy_dict
from visualize import draw_grid, plot_trajectory, moving_average

METHODS = ["Value Iteration", "Policy Iteration", "Truncated PI", "Monte Carlo", "Q-learning"]
LEARNERS = ["Monte Carlo", "Q-learning"]
ARROW = ["↑", "↓", "←", "→", "R↑", "R↓", "R←", "R→"]
CP_FRACS = [0, .01, .02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
VIEW_FRACS = {"init": 0.0, "c1": 0.1, "c2": 0.5, "final": 1.0}
COLORS = {"Value Iteration": "tab:blue", "Policy Iteration": "tab:orange", "Truncated PI": "tab:green",
          "Monte Carlo": "tab:red", "Q-learning": "tab:purple"}



# Môi trường theo kịch bản
def make_scn_env(cfg, map_name, scn, **ov):
    layout = list(cfg["maps"][map_name])
    tmode = scn.get("treasure", "none")
    if tmode == "none":                                   # không có kho báu: bỏ ký tự T
        layout = [row.replace("T", ".") for row in layout]
    kw = dict(cfg["configs"]["default"])
    kw.update(slip_prob=scn.get("slip_prob", 0.0), random_treasure=(tmode == "random"),
              require_all_treasures=scn.get("require", True))
    kw.update(ov)
    return FragileFloorEnv(layout=layout, **kw)


def canon_states(env, tcell):
    """State 'chuẩn' của mỗi ô: độ bền ban đầu, còn kit, chưa nhặt kho báu (random: kho báu ở tcell)."""
    kit = 1 if env.use_repair_kit else 0
    got = env.spawn_idx[tcell] if env.random_treasure else 0
    return {(r, c): env.encode((r, c, env.init_dur, kit, got))
            for r in range(env.H) for c in range(env.W) if env.grid[r][c] not in "#HG"}


# Tiện ích phân tích
def states_under_policy(model, pol):
    """Các state (không terminal) mà policy tối ưu có thể ghé thăm từ các state bắt đầu."""
    seen = np.zeros(model.n, dtype=bool)
    dq = deque(model.start_idx)
    for i in model.start_idx:
        seen[i] = True
    while dq:
        i = dq.popleft()
        a = pol[i]
        for k in range(model.K):
            if model.W[i, a, k] > 0:
                j = model.NS[i, a, k]
                if not seen[j]:
                    seen[j] = True
                    dq.append(j)
    term = model.W.sum(axis=(1, 2)) == 0
    return seen & ~term


def opt_action_rate_arr(opt_ok, reach, arr):
    idx = np.flatnonzero(reach)
    return float(opt_ok[idx, arr[idx]].mean()) if len(idx) else float("nan")


def opt_action_rate_Q(model, opt_ok, reach, Q):
    idx = np.flatnonzero(reach)
    good = 0
    for i in idx:
        code = int(model.ids[i])
        a = int(np.argmax(Q[code])) if code in Q else 0
        good += bool(opt_ok[i, a])
    return good / max(1, len(idx))


def opt_path_states(envv, fn, tcell):
    """State lúc agent lần đầu tới mỗi ô khi chạy policy tối ưu (không trượt) -> dùng làm state đại diện."""
    obs, _ = envv.reset(seed=0, options={"treasure": tcell} if tcell else None)
    out = {}
    for _ in range(envv.max_steps):
        cell = envv.state[:2]
        if cell not in out and envv.grid[cell[0]][cell[1]] not in "#HG":
            out[cell] = obs
        obs, _, te, tr, _ = envv.step(fn(obs))
        if te or tr:
            break
    return out


def grid_actions(canon, known, act):
    return {cell: (act(code) if known(code) else None) for cell, code in canon.items()}


def decision_map(envv, fn):
    """Random treasure: với từng ô spawn, policy có đi nhặt không (không trượt)."""
    out = {}
    for cell in envv.spawn_cells:
        tr = rollout(envv, fn, options={"treasure": cell})
        out[cell] = bool(tr["treasures"] > 0)
    return out


def pick_snap(keys, total, frac):
    ks = sorted(k for k in keys if isinstance(k, int) and k >= 1)
    if not ks:
        return None
    return min(ks, key=lambda k: abs(k - frac * total))


def small_traj(tr):
    return {k: tr[k] for k in ("pos", "events", "return", "steps", "success")}


def downsample_ma(x, n=400):
    w = max(1, len(x) // 50)
    ma = moving_average(x, w)
    xs = np.linspace(0, len(ma) - 1, min(n, len(ma))).astype(int)
    return (xs + w).tolist(), ma[xs].astype(float).tolist()


# Một công việc = (map, kịch bản): chạy cả 5 thuật toán
def run_job(task):
    m, sid, scn, cfg, o, cache_path = task
    t_job = time.time()
    env = make_scn_env(cfg, m, scn)
    envv = make_scn_env(cfg, m, scn, slip_prob=0.0)          # bản không trượt: để vẽ đường đi
    gamma = cfg["q_learning"]["default"]["gamma"]
    model = TabularModel(env)
    ar = np.arange(model.n)
    res = {"map": m, "sid": sid, "scn": scn, "opts": o, "n_states_full": env.n_states,
           "n_reach": model.n, "K": model.K, "env_info": dict(
               H=env.H, W=env.W, n_fragile=env.n_fragile, n_treasure=env.n_treasure,
               n_spawn=len(env.spawn_cells))}

    # Quy hoạch động
    dp = {"Value Iteration": vi_snap(model, gamma, o["theta"]),
          "Policy Iteration": pi_snap(model, gamma, o["theta"]),
          "Truncated PI": tpi_snap(model, gamma, o["tpi_k"], o["theta"])}
    V_vi, pol_vi, _, _ = dp["Value Iteration"]
    Qstar = q_values(model, V_vi, gamma)
    opt_ok = Qstar >= Qstar.max(axis=1, keepdims=True) - 1e-6
    reach = states_under_policy(model, pol_vi)
    res["n_reach_opt"] = int(reach.sum())

    def dp_fn(arr):
        return lambda s: int(arr[model.idx[s]]) if s in model.idx else 0

    # kho báu minh họa (random): ô spawn mà policy tối ưu đi nhặt xa nhất
    tcell = None
    if env.random_treasure:
        best_len = -1
        for cell in env.spawn_cells:
            tr = rollout(envv, dp_fn(pol_vi), options={"treasure": cell})
            if tr["treasures"] > 0 and tr["steps"] > best_len:
                tcell, best_len = cell, tr["steps"]
        tcell = tcell or env.spawn_cells[len(env.spawn_cells) // 2]
    res["demo_treasure"] = tcell
    canon = canon_states(env, tcell)
    canon.update(opt_path_states(envv, dp_fn(pol_vi), tcell))   # ưu tiên state dọc đường tối ưu
    known_dp = lambda code: code in model.idx
    opt_sets = {cell: set(np.flatnonzero(opt_ok[model.idx[code]]).tolist())
                for cell, code in canon.items() if code in model.idx}
    start_code = canon[env.start]
    res["opt_sets"] = opt_sets
    res["start_Qstar"] = Qstar[model.idx[start_code]].tolist() if start_code in model.idx else None
    res["start_Vstar"] = float(V_vi[model.idx[start_code]]) if start_code in model.idx else None

    opt_eval = evaluate(env, dp_fn(pol_vi), episodes=o["eval_eps"], seed=424242)
    opt_ret = opt_eval["mean_return"]
    res["opt_return"], res["opt_success"] = opt_ret, opt_eval["success_rate"]

    rows_final, rows_ckpt, rows_dp, views, trajs, decisions, curves = [], [], [], {}, {}, {}, {}
    base = {"map": m, "scenario": sid}

    for name, (V, pol, info, snaps) in dp.items():
        fn = dp_fn(pol)
        ev = evaluate(env, fn, episodes=o["eval_eps"], seed=424242)
        ev_opt_rate = opt_action_rate_arr(opt_ok, reach, pol.astype(np.int64))
        rows_final.append({**base, "method": name, "seed": -1, "return": ev["mean_return"],
                           "return_std_ep": ev["std_return"], "success": ev["success_rate"],
                           "steps": ev["mean_steps"], "repairs": ev["repairs"], "treasures": ev["treasures"],
                           "runtime_s": info["time"], "opt_action_rate": ev_opt_rate})
        rows_dp.append({**base, "method": name, "iterations": info["iterations"], "sweeps": info["sweeps"],
                        "theta": o["theta"],
                        "bellman_residual": float(np.abs(q_values(model, V, gamma).max(axis=1) - V).max()),
                        "V_start": float(V[model.start_idx].mean()), "runtime_s": round(info["time"], 3),
                        "policy_stable_from_iter": (max([i for i, c in enumerate(info["changes"]) if c > 0], default=-1) + 2),
                        "policy_changes_total": int(sum(c for c in info["changes"] if c > 0))})
        curves[name] = {k: info[k] for k in ("deltas", "changes", "vstart")}
        # snapshot -> bảng + view
        total = info["iterations"]
        labels = {"init": 0, "final": "final",
                  "c1": pick_snap(snaps.keys(), total, VIEW_FRACS["c1"]),
                  "c2": pick_snap(snaps.keys(), total, VIEW_FRACS["c2"])}
        for key, arr in snaps.items():
            evs = evaluate(env, dp_fn(arr), episodes=o["ckpt_eps"], seed=777)
            rows_ckpt.append({**base, "method": name, "seed": -1, "step": "final" if key == "final" else key,
                              "step_type": "iteration", "return": evs["mean_return"], "success": evs["success_rate"],
                              "opt_action_rate": opt_action_rate_arr(opt_ok, reach, arr.astype(np.int64))})
        v = []
        for slot in ("init", "c1", "c2", "final"):
            lab = labels[slot]
            if lab is None:
                lab = "final"
            arr = snaps[lab]
            fnl = dp_fn(arr)
            g = grid_actions(canon, known_dp, lambda code, arr=arr: int(arr[model.idx[code]]))
            tr = None
            if slot in ("init", "final"):
                tr = small_traj(rollout(envv, fnl, options={"treasure": tcell} if tcell else None))
            nm = "π0" if lab == 0 else ("cuối" if lab == "final" else f"iter {lab}")
            v.append({"slot": slot, "label": nm, "grid": g, "traj": tr})
        views[name] = v
        if env.random_treasure:
            decisions[name] = decision_map(envv, fn)
    # chính sách tối ưu để vẽ cột "tối ưu"
    res["opt_grid"] = grid_actions(canon, known_dp, lambda code: int(pol_vi[model.idx[code]]))

    #Học từ trải nghiệm
    qp = dict(cfg["q_learning"]["default"])
    E = max(300, int(o["episodes"][m] * o["scale"]))
    res["episodes"] = E
    cps = sorted(set(int(round(f * E)) for f in CP_FRACS))
    view_eps = {slot: int(round(f * E)) for slot, f in VIEW_FRACS.items()}
    start_rows = []
    for algo in LEARNERS:
        runs_curve = []
        for seed in range(o["seeds"]):
            t0 = time.time()
            env_t = make_scn_env(cfg, m, scn)
            env_e = make_scn_env(cfg, m, scn)
            env_v = make_scn_env(cfg, m, scn, slip_prob=0.0)
            seed_views = []

            def cb(ep, Q, seed=seed, algo=algo, env_e=env_e, env_v=env_v, seed_views=seed_views):
                fn = q_policy_dict(Q)
                n_eval = o["eval_eps"] if ep == E else o["ckpt_eps"]
                ev = evaluate(env_e, fn, episodes=n_eval, seed=9000 + seed)
                rate = opt_action_rate_Q(model, opt_ok, reach, Q)
                rows_ckpt.append({**base, "method": algo, "seed": seed, "step": ep, "step_type": "episode",
                                  "return": ev["mean_return"], "success": ev["success_rate"],
                                  "opt_action_rate": rate})
                if ep == E:
                    rows_final.append({**base, "method": algo, "seed": seed, "return": ev["mean_return"],
                                       "return_std_ep": ev["std_return"], "success": ev["success_rate"],
                                       "steps": ev["mean_steps"], "repairs": ev["repairs"],
                                       "treasures": ev["treasures"], "runtime_s": None,
                                       "opt_action_rate": rate})
                if seed == 0:
                    for slot, vep in view_eps.items():
                        if vep == ep:
                            known = (lambda code: code in Q) if ep > 0 else (lambda code: False)
                            g = grid_actions(canon, known, lambda code: int(np.argmax(Q[code])))
                            if ep == 0:
                                g = {cell: -1 for cell in canon}          # Q = 0: policy ngẫu nhiên
                            tr = None
                            if slot in ("init", "final"):
                                tr = small_traj(rollout(env_v, fn, options={"treasure": tcell} if tcell else None))
                            nm = "init (Q=0)" if ep == 0 else (f"{ep} ep" if slot != "final" else f"cuối ({ep} ep)")
                            seed_views.append({"slot": slot, "label": nm, "grid": g, "traj": tr})
                            Qs = Q.get(start_code)
                            start_rows.append({**base, "method": algo, "episodes": ep,
                                               **{f"Q_{ARROW[j]}": (float(Qs[j]) if Qs is not None else 0.0)
                                                  for j in range(8)}})
                    if ep == E and env.random_treasure:
                        decisions[algo] = decision_map(env_v, fn)

            out = train(env_t, algo, E, seed, alpha=qp["alpha"], gamma=gamma, eps_start=qp["eps_start"],
                        eps_end=qp["eps_end"], eps_decay_frac=qp["eps_decay_frac"],
                        mc_alpha=o.get("mc_alpha"), checkpoints=cps, on_checkpoint=cb)
            dt = time.time() - t0
            for r in rows_final:
                if r["method"] == algo and r["seed"] == seed:
                    r["runtime_s"] = round(dt, 2)
            runs_curve.append({"train": downsample_ma(out["returns"]),
                               "train_success": downsample_ma(out["success"])})
            if seed == 0:
                views[algo] = sorted(seed_views, key=lambda x: ["init", "c1", "c2", "final"].index(x["slot"]))
            print(f"    [{m}/{sid}] {algo} seed {seed}: {E} ep, {dt:.0f}s", flush=True)
        curves[algo] = runs_curve
    res.update(rows_final=rows_final, rows_ckpt=rows_ckpt, rows_dp=rows_dp, views=views,
               decisions=decisions, curves=curves, start_rows=start_rows, job_time=time.time() - t_job)
    with open(cache_path, "wb") as f:
        pickle.dump(res, f)
    print(f"  ✔ xong {m}/{sid} ({res['job_time']:.0f}s)", flush=True)
    return cache_path



# Vẽ hình
def draw_policy_panel(ax, env, grid, ok_sets=None, title="", tcell=None):
    draw_grid(ax, env)
    for (r, c), a in grid.items():
        if a is None:
            ax.text(c, r - 0.12, "·", ha="center", va="center", fontsize=12, color="gray")
            continue
        if a == -1:
            ax.text(c, r - 0.12, "?", ha="center", va="center", fontsize=12, color="gray")
            continue
        bad = ok_sets is not None and (r, c) in ok_sets and a not in ok_sets[(r, c)]
        ax.text(c, r - 0.12, ARROW[a], ha="center", va="center", fontsize=11 if a < 4 else 9,
                color="red" if bad else ("darkblue" if a >= 4 else "black"), fontweight="bold")
    if tcell:
        ax.text(tcell[1] - 0.3, tcell[0] - 0.3, "★", ha="center", va="center", fontsize=11, color="goldenrod")
    ax.set_title(title, fontsize=8)


def fig_policy_grids(path, env, res):
    views = res["views"]
    cols = ["init", "c1", "c2", "final", "opt"]
    fig, axes = plt.subplots(len(METHODS), 5, figsize=(5 * (env.W * 0.5 + 0.4), len(METHODS) * (env.H * 0.5 + 0.7)),
                             squeeze=False)
    for i, meth in enumerate(METHODS):
        for j, slot in enumerate(cols):
            ax = axes[i][j]
            if slot == "opt":
                draw_policy_panel(ax, env, res["opt_grid"], None, f"{meth}\nTỐI ƯU (VI)" if False else "TỐI ƯU (VI)",
                                  res["demo_treasure"])
            else:
                v = next(x for x in views[meth] if x["slot"] == slot)
                draw_policy_panel(ax, env, v["grid"], res["opt_sets"] if slot != "init" else None,
                                  f"{meth}\n{v['label']}" if slot == "init" else v["label"], res["demo_treasure"])
    fig.suptitle(f"Policy | map {res['map']} | {res['sid']}\n"
                 "Mỗi ô hiển thị action tại state mà policy TỐI ƯU (không trượt) tới ô đó lần đầu; ô ngoài đường tối ưu "
                 "dùng state chuẩn (độ bền ban đầu, còn kit, chưa nhặt kho báu)\n"
                 "đỏ = khác action tối ưu, · = state không tới được/chưa thăm, ? = chưa học (ngẫu nhiên)"
                 + (", ★ = vị trí kho báu minh họa" if res["demo_treasure"] else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.5)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_trajectories(path, env, res):
    fig, axes = plt.subplots(2, len(METHODS), figsize=(len(METHODS) * (env.W * 0.55 + 0.5),
                                                       2 * (env.H * 0.55 + 1.0)), squeeze=False)
    for j, meth in enumerate(METHODS):
        for i, slot in enumerate(("init", "final")):
            v = next(x for x in res["views"][meth] if x["slot"] == slot)
            plot_trajectory(axes[i][j], env, v["traj"], f"{meth} - {v['label']}")
    fig.suptitle(f"Đường đi của policy (không trượt) | map {res['map']} | {res['sid']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fig_decisions(path, env, res):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(len(METHODS) * (env.W * 0.5 + 0.4), env.H * 0.5 + 1.0),
                             squeeze=False)
    for j, meth in enumerate(METHODS):
        ax = axes[0][j]
        draw_grid(ax, env)
        d = res["decisions"].get(meth, {})
        for (r, c), got in d.items():
            ax.text(c, r - 0.12, "o" if got else "x", ha="center", va="center", fontsize=12,
                    color="darkgreen" if got else "firebrick", fontweight="bold")
        rate = np.mean(list(d.values())) if d else float("nan")
        ax.set_title(f"{meth}\nnhặt {rate:.0%} ô spawn", fontsize=8)
    fig.suptitle(f"Kho báu ngẫu nhiên: o = đi nhặt, x = bỏ | map {res['map']} | {res['sid']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fig_curves(out, m, results, scn_ids):
    for kind in ("train", "greedy"):
        fig, axes = plt.subplots(2, 5, figsize=(22, 8), squeeze=False)
        for k, sid in enumerate(scn_ids):
            ax = axes[k // 5][k % 5]
            res = results[(m, sid)]
            for algo in LEARNERS:
                if kind == "train":
                    mats = [np.array(r["train"][1]) for r in res["curves"][algo]]
                    xs = res["curves"][algo][0]["train"][0]
                    mu, sd = np.mean(mats, axis=0), np.std(mats, axis=0)
                    ax.plot(xs, mu, color=COLORS[algo], label=algo)
                    ax.fill_between(xs, mu - sd, mu + sd, color=COLORS[algo], alpha=0.2)
                else:
                    df = pd.DataFrame([r for r in res["rows_ckpt"] if r["method"] == algo])
                    g = df.groupby("step")["success"].agg(["mean", "std"]).fillna(0)
                    ax.plot(g.index, g["mean"], color=COLORS[algo], marker="o", ms=3, label=algo)
                    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[algo], alpha=0.2)
            if kind == "train":
                ax.axhline(res["opt_return"], ls="--", color="k", label="tối ưu (VI)")
                ax.set_ylabel("return khi huấn luyện (ε-greedy)")
            else:
                ax.axhline(0.9, ls=":", color="gray", label="ngưỡng 90%")
                ax.axhline(res["opt_success"], ls="--", color="k", label="tối ưu (VI)")
                ax.set_ylim(-0.05, 1.05)
                ax.set_ylabel("tỉ lệ tới goal của policy greedy")
            ax.set_title(sid, fontsize=9)
            ax.set_xlabel("episode")
            if k == 0:
                ax.legend(fontsize=7)
        fig.suptitle(f"map {m}: " + ("learning curve (trung bình trượt ± std các seed)" if kind == "train"
                                      else "tỉ lệ tới goal của policy greedy theo số episode (mean ± std các seed)"))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(out, f"curves_{kind}_{m}.png"), dpi=100)
        plt.close(fig)


def fig_dp(out, m, results, scn_ids):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    for sid in scn_ids:
        for name, c in results[(m, sid)]["curves"].items():
            if name not in ("Value Iteration", "Policy Iteration", "Truncated PI"):
                continue
            col = COLORS[name]
            axes[0].semilogy(np.maximum(c["deltas"], 1e-12), color=col, alpha=0.35, lw=1)
            ch = np.array(c["changes"], dtype=float)
            ch[ch < 0] = np.nan
            axes[1].plot(np.arange(1, len(ch) + 1), ch, color=col, alpha=0.35, lw=1)
            axes[2].plot(np.arange(1, len(c["vstart"]) + 1), c["vstart"], color=col, alpha=0.35, lw=1)
    th = next(iter(results.values()))["opts"]["theta"]
    axes[0].axhline(th, color="k", ls="--", label=f"θ = {th:g}")
    axes[0].set_title("Bellman residual (max|ΔV|) theo vòng lặp")
    axes[1].set_title("Số state đổi action theo vòng lặp")
    axes[1].set_xscale("log")
    axes[1].set_yscale("symlog")
    axes[2].set_title("V(start) theo vòng lặp")
    for ax in axes:
        ax.set_xlabel("vòng lặp (log)" if ax is axes[1] else "vòng lặp")
        ax.set_xscale("log")
    for name in ("Value Iteration", "Policy Iteration", "Truncated PI"):
        axes[0].plot([], [], color=COLORS[name], label=name)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Hội tụ của quy hoạch động - map {m} (mỗi đường mờ = 1 kịch bản)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, f"dp_convergence_{m}.png"), dpi=100)
    plt.close(fig)


# Bảng
def first_reach(df, thr_ratio, opt_ret, need_success=None):
    d = df.sort_values("step")
    target = opt_ret - (1 - thr_ratio) * abs(opt_ret)          # đạt thr_ratio của return tối ưu
    ok = d[(d["return"] >= target) & ((d["success"] >= need_success) if need_success else True)]
    return int(ok["step"].iloc[0]) if len(ok) else np.nan


def md_table(df, floatfmt=3):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and not pd.isna(v)
                                      else ("" if (isinstance(v, float) and pd.isna(v)) else str(v))
                                      for v in r) + " |")
    return "\n".join(out)


def build_tables(out, results, cfg, o):
    tdir = os.path.join(out, "tables")
    os.makedirs(tdir, exist_ok=True)
    space, dprows, final, ckpt, start = [], [], [], [], []
    for (m, sid), r in results.items():
        e = r["env_info"]
        space.append({"map": m, "scenario": sid, "kích thước": f"{e['H']}x{e['W']}", "fragile": e["n_fragile"],
                      "treasure": e["n_treasure"], "ô spawn": e["n_spawn"], "|A|": 8,
                      "|S| lý thuyết": r["n_states_full"], "|S| reachable": r["n_reach"],
                      "|S| dưới π*": r["n_reach_opt"], "V*(start)": r["start_Vstar"],
                      "return tối ưu (mô phỏng)": r["opt_return"], "episodes (MC/Q)": r["episodes"]})
        dprows += r["rows_dp"]
        for x in r["rows_final"]:
            x = dict(x)
            x["opt_return"] = r["opt_return"]
            final.append(x)
        for x in r["rows_ckpt"]:
            x = dict(x)
            x["opt_return"] = r["opt_return"]
            ckpt.append(x)
        for x in r["start_rows"]:
            start.append(x)
        if r["start_Qstar"]:
            start.append({"map": m, "scenario": sid, "method": "Q* (Value Iteration)", "episodes": -1,
                          **{f"Q_{ARROW[j]}": v for j, v in enumerate(r["start_Qstar"])}})
    sp = pd.DataFrame(space)
    sp.to_csv(os.path.join(tdir, "T1_state_action_space.csv"), index=False, encoding="utf-8-sig")
    dpdf = pd.DataFrame(dprows)
    dpdf.to_csv(os.path.join(tdir, "T2_dp_convergence.csv"), index=False, encoding="utf-8-sig")
    fin = pd.DataFrame(final)
    ck = pd.DataFrame(ckpt)
    ck.to_csv(os.path.join(tdir, "T5_policy_vs_steps_checkpoints.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(start).to_csv(os.path.join(tdir, "T6_start_state_Q_values.csv"), index=False, encoding="utf-8-sig")

    # T3: so sánh cuối cùng (mỗi kịch bản x thuật toán), trung bình qua seed
    g = fin.groupby(["map", "scenario", "method"])
    t3 = g.agg(return_mean=("return", "mean"), return_std_seeds=("return", "std"),
               return_std_episodes=("return_std_ep", "mean"), success=("success", "mean"),
               steps=("steps", "mean"), repairs=("repairs", "mean"), treasures=("treasures", "mean"),
               runtime_s=("runtime_s", "mean"), opt_action_rate=("opt_action_rate", "mean"),
               opt_return=("opt_return", "first"), n_seeds=("seed", "count")).reset_index()
    t3["return_std_seeds"] = t3["return_std_seeds"].fillna(0.0)
    t3["return_ratio"] = t3["return_mean"] / t3["opt_return"]
    coll = []
    for (m, sid), r in results.items():
        for meth, d in r["decisions"].items():
            coll.append({"map": m, "scenario": sid, "method": meth, "collect_rate_spawn": np.mean(list(d.values()))})
    if coll:
        t3 = t3.merge(pd.DataFrame(coll), how="left", on=["map", "scenario", "method"])
    t3["method"] = pd.Categorical(t3["method"], METHODS, ordered=True)
    t3 = t3.sort_values(["map", "scenario", "method"])
    t3.to_csv(os.path.join(tdir, "T3_final_comparison_by_scenario.csv"), index=False, encoding="utf-8-sig")

    # T3b: tổng hợp theo map x thuật toán
    t3b = t3.groupby(["map", "method"], observed=True).agg(
        return_ratio=("return_ratio", "mean"), success=("success", "mean"), steps=("steps", "mean"),
        opt_action_rate=("opt_action_rate", "mean"), runtime_s=("runtime_s", "mean")).reset_index()
    t3b.to_csv(os.path.join(tdir, "T3b_final_comparison_by_map.csv"), index=False, encoding="utf-8-sig")

    # T4: ngưỡng hội tụ của MC / Q-learning (số episode để đạt ngưỡng)
    rows = []
    lk = ck[ck["method"].isin(LEARNERS)]
    for (m, sid, meth), grp in lk.groupby(["map", "scenario", "method"]):
        opt = grp["opt_return"].iloc[0]
        per = {k: [] for k in ("ret90", "ret95", "succ90", "both")}
        for seed, gs in grp.groupby("seed"):
            per["ret90"].append(first_reach(gs, 0.90, opt))
            per["ret95"].append(first_reach(gs, 0.95, opt))
            per["succ90"].append(first_reach(gs.assign(**{"return": np.inf}), 0.0, 1.0, need_success=0.9))
            per["both"].append(first_reach(gs, 0.90, opt, need_success=0.9))
        n = len(per["ret90"])
        row = {"map": m, "scenario": sid, "method": meth, "episodes_total": int(grp["step"].max()), "n_seeds": n}
        for k, v in per.items():
            arr = np.array(v, dtype=float)
            row[f"ep_to_{k}_mean"] = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else np.nan
            row[f"seeds_reached_{k}"] = int(np.sum(~np.isnan(arr)))
        rows.append(row)
    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(tdir, "T4_convergence_thresholds_learners.csv"), index=False, encoding="utf-8-sig")

    # markdown cho báo cáo
    md = ["# Bảng số liệu (tự sinh bởi run_all.py)\n",
          f"Tham số: gamma={cfg['q_learning']['default']['gamma']}, θ={o['theta']}, Truncated PI k={o['tpi_k']}, "
          f"seeds={o['seeds']}, scale={o['scale']}, eval_eps={o['eval_eps']}\n",
          "## T1. Không gian state / action\n", md_table(sp, 2), "\n",
          "## T2. Hội tụ quy hoạch động (ngưỡng θ)\n", md_table(dpdf, 4), "\n",
          "## T3b. So sánh 5 thuật toán (trung bình 10 kịch bản)\n", md_table(t3b, 3), "\n",
          "## T4. Số episode để đạt ngưỡng (MC / Q-learning)\n", md_table(t4, 0), "\n",
          "## T3. So sánh chi tiết theo kịch bản\n",
          md_table(t3.drop(columns=["n_seeds"]), 3), "\n"]
    with open(os.path.join(out, "report_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return t3, t3b


def fig_compare(out, t3b):
    maps = list(dict.fromkeys(t3b["map"]))
    fig, axes = plt.subplots(2, len(maps), figsize=(5.5 * len(maps), 7), squeeze=False)
    for j, m in enumerate(maps):
        d = t3b[t3b["map"] == m].set_index("method").reindex(METHODS)
        for i, (col, ttl) in enumerate([("success", "Tỉ lệ tới goal"), ("return_ratio", "return / return tối ưu")]):
            axes[i][j].bar(range(len(METHODS)), d[col], color=[COLORS[x] for x in METHODS])
            axes[i][j].set_xticks(range(len(METHODS)))
            axes[i][j].set_xticklabels([x.replace(" ", "\n") for x in METHODS], fontsize=7)
            axes[i][j].set_title(f"{ttl} - map {m}", fontsize=10)
            axes[i][j].set_ylim(min(0, np.nanmin(d[col]) - 0.05), 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "compare_methods.png"), dpi=110)
    plt.close(fig)


# Thí nghiệm cấu hình reward (Value Iteration, nhanh)
def exp_rewards(cfg, out, maps):
    exp = cfg["rewards_exp"]
    gamma = cfg["q_learning"]["default"]["gamma"]
    rows = []
    for m in maps:
        names = list(exp["configs"])
        fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), squeeze=False)
        for j, name in enumerate(names):
            kw = {**exp["base"], **exp["configs"][name]}
            env = make_scn_env(cfg, m, {"treasure": "fixed"}, **kw)
            model = TabularModel(env)
            V, pol, info, _ = vi_snap(model, gamma, 1e-6)
            fn = lambda s, p=pol, mo=model: int(p[mo.idx[s]]) if s in mo.idx else 0
            ev = evaluate(env, fn, episodes=500, seed=5)
            envv = make_scn_env(cfg, m, {"treasure": "fixed"}, **{**kw, "slip_prob": 0.0})
            tr = rollout(envv, fn)
            plot_trajectory(axes[0][j], envv, tr, name)
            rows.append({"map": m, "cấu hình": name, "V(start)": float(V[model.start_idx].mean()),
                         "return TB": ev["mean_return"], "tới goal": ev["success_rate"],
                         "số bước TB": ev["mean_steps"], "repair/ván": ev["repairs"], "kho báu/ván": ev["treasures"],
                         "đường đi không trượt: bước": tr["steps"], "repair": tr["repairs"],
                         "kho báu": tr["treasures"], "tới goal (không trượt)": tr["success"]})
        fig.suptitle(f"Policy tối ưu (VI) theo cấu hình reward - map {m} (vẽ không trượt)")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(out, f"rewards_policies_{m}.png"), dpi=100)
        plt.close(fig)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    df.to_csv(os.path.join(out, "tables", "T7_reward_configs.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string())
    print("\n(Return của các cấu hình KHÔNG so sánh trực tiếp vì thang reward khác nhau; so hành vi: "
          "tới goal, số bước, repair, kho báu.)")


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["main"], choices=["main", "rewards"])
    ap.add_argument("--maps", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--scenarios", nargs="+", default=None, help="vd: S01 S05 (mặc định: cả 10)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scale", type=float, default=1.0, help="nhân số episode của MC/Q-learning")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 1))))
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="chạy thử nhanh (ít episode, 2 seed)")
    ap.add_argument("--force", action="store_true", help="bỏ cache, chạy lại")
    ap.add_argument("--plots-only", action="store_true", help="chỉ dựng lại bảng/hình từ cache")
    args = ap.parse_args()

    cfg = load_config()
    ra = cfg["run_all"]
    o = {"seeds": 2 if args.quick else args.seeds, "scale": 0.03 if args.quick else args.scale,
         "episodes": ra["episodes"], "theta": ra["theta"], "tpi_k": ra["tpi_k"],
         "eval_eps": 60 if args.quick else ra["eval_eps"], "ckpt_eps": 15 if args.quick else ra["ckpt_eps"],
         "mc_alpha": ra.get("mc_alpha")}
    out = args.out + ("_quick" if args.quick else "")
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "cache")
    os.makedirs(cache, exist_ok=True)
    scn_all = cfg["scenarios"]
    scn_ids = [s for s in scn_all if args.scenarios is None or s.split("_")[0] in args.scenarios]

    if "rewards" in args.exp:
        exp_rewards(cfg, out, args.maps)
    if "main" not in args.exp:
        return

    tasks = []
    for m in args.maps:
        for sid in scn_ids:
            path = os.path.join(cache, f"{m}__{sid}.pkl")
            if not args.force and os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        if pickle.load(f)["opts"] == o:
                            continue
                except Exception:
                    pass
            tasks.append((m, sid, scn_all[sid], cfg, o, path))
    order = {"easy": 0, "medium": 1, "hard": 2}
    tasks.sort(key=lambda t: -order.get(t[0], 1))            # việc nặng (hard) chạy trước
    if not args.plots_only and tasks:
        print(f"Chạy {len(tasks)} công việc (map, kịch bản) với {args.workers} tiến trình; "
              f"mỗi việc = 3 thuật toán DP + MC/Q-learning x {o['seeds']} seed", flush=True)
        t0 = time.time()
        if args.workers > 1:
            with Pool(args.workers) as p:
                for _ in p.imap_unordered(run_job, tasks):
                    pass
        else:
            for t in tasks:
                run_job(t)
        print(f"Xong phần chạy: {(time.time() - t0) / 60:.1f} phút", flush=True)

    results = {}
    for m in args.maps:
        for sid in scn_ids:
            path = os.path.join(cache, f"{m}__{sid}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    results[(m, sid)] = pickle.load(f)
    if not results:
        print("Chưa có kết quả nào trong cache.")
        return
    for d in ("policy_grids", "trajectories", "decision_maps"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    print("Đang dựng bảng và hình...", flush=True)
    t3, t3b = build_tables(out, results, cfg, o)
    fig_compare(out, t3b)
    for (m, sid), res in results.items():
        env = make_scn_env(cfg, m, res["scn"])
        fig_policy_grids(os.path.join(out, "policy_grids", f"{m}__{sid}.png"), env, res)
        fig_trajectories(os.path.join(out, "trajectories", f"{m}__{sid}.png"), env, res)
        if env.random_treasure:
            fig_decisions(os.path.join(out, "decision_maps", f"{m}__{sid}.png"), env, res)
    for m in args.maps:
        ids = [s for s in scn_ids if (m, s) in results]
        if len(ids) == 10:
            fig_curves(out, m, results, ids)
        if ids:
            fig_dp(out, m, results, ids)
    print(t3b.to_string())
    print(f"\nXONG. Mọi kết quả nằm trong: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
