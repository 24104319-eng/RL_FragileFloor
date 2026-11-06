"""Hàm dùng chung: đọc cấu hình, tạo env, giải Policy Iteration, rollout, đánh giá policy."""
import json
import os

import numpy as np

from fragile_floor_env import FragileFloorEnv, MOVES
from dp_solvers import TabularModel, policy_iteration

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps.json")


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_env(map_name, cfg=None, config="default", **overrides):
    """Tạo env từ maps.json. overrides ghi đè tham số env (slip_prob, fall_penalty, ...)."""
    cfg = cfg or load_config()
    kw = {**cfg["configs"]["default"], **cfg["configs"].get(config, {}), **overrides}
    return FragileFloorEnv(layout=cfg["maps"][map_name], **kw)


# ---------------- Policy Iteration ----------------
def solve_pi(env, gamma=0.99, theta=1e-6):
    """Trả về (model, V, policy, info, pid) với pid: {mã state -> action}."""
    model = TabularModel(env)
    V, pol, info = policy_iteration(model, gamma, theta)
    pid = {int(model.ids[i]): int(pol[i]) for i in range(model.n)}
    return model, V, pol, info, pid


def pi_policy_fn(pid):
    return lambda s: pid.get(int(s), 0)


def q_policy_fn(Q):
    return lambda s: int(np.argmax(Q[s]))


# ---------------- Rollout / đánh giá ----------------
def rollout(env, policy_fn, seed=None, options=None):
    """Chơi 1 episode theo policy_fn(obs) -> action. Ghi lại đường đi và sự kiện."""
    obs, _ = env.reset(seed=seed, options=options)
    pos = [env.state[:2]]
    actions, events = [], []
    total, t, n_rep, n_tre = 0.0, 0, 0, 0
    while True:
        a = int(policy_fn(obs))
        prev = env.state
        obs, r, term, trunc, _ = env.step(a)
        t += 1
        cur = env.state
        actions.append(a)
        total += r
        pos.append(cur[:2])
        if a >= 4 and cur[3] < prev[3]:                  # repair hợp lệ (kit giảm)
            dr, dc = MOVES[a - 4]
            events.append(("repair", t, (prev[0] + dr, prev[1] + dc)))
            n_rep += 1
        if cur[4] != prev[4]:                            # nhặt kho báu
            events.append(("treasure", t, cur[:2]))
            n_tre += 1
        if term or trunc:
            break
    success = bool(term and cur[:2] == env.goal)
    if term and not success:
        events.append(("fall", t, cur[:2]))
    return {"pos": pos, "actions": actions, "events": events, "return": total,
            "steps": t, "success": success, "repairs": n_rep, "treasures": n_tre}


def evaluate(env, policy_fn, episodes=300, seed=0):
    """Chạy nhiều episode, trả về thống kê trung bình."""
    env.reset(seed=seed)
    runs = [rollout(env, policy_fn) for _ in range(episodes)]
    rets = np.array([r["return"] for r in runs])
    return {"mean_return": float(rets.mean()), "std_return": float(rets.std()),
            "success_rate": float(np.mean([r["success"] for r in runs])),
            "mean_steps": float(np.mean([r["steps"] for r in runs])),
            "repairs": float(np.mean([r["repairs"] for r in runs])),
            "treasures": float(np.mean([r["treasures"] for r in runs]))}
