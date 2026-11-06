"""
Value Iteration / Policy Iteration / Truncated Policy Iteration có GHI LẠI:
  - policy ban đầu (pi_0) và policy tại một số vòng lặp (snapshot),
  - số state đổi action mỗi vòng, Bellman residual (delta), V(start) mỗi vòng.
Dùng lại TabularModel / q_values / greedy_policy / evaluate_policy của dp_solvers.py.

Định nghĩa policy ban đầu:
  - Value Iteration : policy tham lam theo V_0 = 0 (chỉ nhìn reward tức thời)
  - Policy Iteration / Truncated PI : pi_0 = "luôn đi LÊN" (action 0) ở mọi state
"""
import time
import numpy as np
from dp_solvers import q_values, greedy_policy, evaluate_policy

SNAP_ITERS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)


def _vstart(model, V):
    return float(V[model.start_idx].mean())


def vi_snap(model, gamma=0.99, theta=1e-6, max_iter=100000):
    t0 = time.time()
    V = np.zeros(model.n)
    snaps, changes, deltas, vstart = {}, [], [], []
    prev = None
    for k in range(1, max_iter + 1):
        Q = q_values(model, V, gamma)
        pol = Q.argmax(axis=1)                       # policy tham lam theo V_{k-1}
        if prev is None:
            snaps[0] = pol.astype(np.int8)           # policy ban đầu: tham lam theo V_0 = 0
        else:
            changes.append(int((pol != prev).sum()))
            if (k - 1) in SNAP_ITERS:
                snaps[k - 1] = pol.astype(np.int8)
        prev = pol
        Vn = Q.max(axis=1)
        delta = float(np.abs(Vn - V).max())
        V = Vn
        deltas.append(delta)
        vstart.append(_vstart(model, V))
        if delta < theta:
            break
    pol = greedy_policy(model, V, gamma)
    changes.append(int((pol != prev).sum()))
    snaps["final"] = pol.astype(np.int8)
    info = dict(iterations=k, sweeps=k, deltas=deltas, changes=changes, vstart=vstart,
                time=time.time() - t0, final_delta=deltas[-1])
    return V, pol, info, snaps


def pi_snap(model, gamma=0.99, theta=1e-6, max_iter=10000):
    t0 = time.time()
    V = np.zeros(model.n)
    policy = np.zeros(model.n, dtype=np.int64)
    snaps = {0: policy.astype(np.int8)}
    changes, deltas, vstart, sweeps_hist, total = [], [], [], [], 0
    for it in range(1, max_iter + 1):
        V_old = V
        V, sw = evaluate_policy(model, policy, V, gamma, theta)
        total += sw
        sweeps_hist.append(sw)
        deltas.append(float(np.abs(V - V_old).max()))
        vstart.append(_vstart(model, V))
        new = greedy_policy(model, V, gamma, old_policy=policy)
        changes.append(int((new != policy).sum()))
        done = changes[-1] == 0
        policy = new
        if it in SNAP_ITERS:
            snaps[it] = policy.astype(np.int8)
        if done:
            break
    snaps["final"] = policy.astype(np.int8)
    info = dict(iterations=it, sweeps=total, deltas=deltas, changes=changes, vstart=vstart,
                sweeps_hist=sweeps_hist, time=time.time() - t0, final_delta=deltas[-1])
    return V, policy, info, snaps


def tpi_snap(model, gamma=0.99, k=5, theta=1e-6, max_iter=100000):
    t0 = time.time()
    V = np.zeros(model.n)
    policy = np.zeros(model.n, dtype=np.int64)
    snaps = {0: policy.astype(np.int8)}
    changes, deltas, vstart = [], [], []
    for it in range(1, max_iter + 1):
        new = greedy_policy(model, V, gamma, old_policy=policy)
        changes.append(int((new != policy).sum()))
        policy = new
        V_old = V
        V, _ = evaluate_policy(model, policy, V, gamma, theta=0.0, max_sweeps=k)
        deltas.append(float(np.abs(V - V_old).max()))
        vstart.append(_vstart(model, V))
        if it in SNAP_ITERS:
            snaps[it] = policy.astype(np.int8)
        if deltas[-1] < theta:
            break
    policy = greedy_policy(model, V, gamma, old_policy=policy)
    snaps["final"] = policy.astype(np.int8)
    info = dict(iterations=it, sweeps=it * k, deltas=deltas, changes=changes, vstart=vstart,
                time=time.time() - t0, final_delta=deltas[-1])
    return V, policy, info, snaps
