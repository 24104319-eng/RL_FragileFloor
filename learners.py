"""
Hai thuật toán học KHÔNG cần mô hình (tabular, không DQN), dùng Q-table dạng dict {mã state: mảng 8 Q-value}
(tiết kiệm bộ nhớ trên map hard có hàng triệu state tiềm năng):
  - Q-learning (off-policy TD(0), cập nhật từng bước)
  - Monte Carlo control (on-policy, first-visit, epsilon-greedy, trung bình tăng dần 1/N hoặc alpha cố định)
Cả hai dùng cùng lịch epsilon tuyến tính. Có callback tại các checkpoint (số episode đã học) để lưu policy.
"""
import numpy as np


def greedy_tie_random(row, rng):
    best = np.flatnonzero(row >= row.max() - 1e-12)
    return int(rng.choice(best))


def train(env, algo, episodes, seed, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
          eps_decay_frac=0.6, mc_alpha=None, checkpoints=(), on_checkpoint=None):
    """algo in {'Q-learning', 'Monte Carlo'}. Trả về dict(Q, returns, success, lengths, visited)."""
    rng = np.random.default_rng(seed)
    nA = env.n_actions
    Q, N = {}, {}
    returns = np.zeros(episodes, dtype=np.float32)
    success = np.zeros(episodes, dtype=np.float32)
    decay = max(1, int(eps_decay_frac * episodes))
    cps = set(int(c) for c in checkpoints)
    env.reset(seed=seed)

    def row(s):
        r = Q.get(s)
        if r is None:
            r = Q[s] = np.zeros(nA)
        return r

    for ep in range(episodes):
        if ep in cps and on_checkpoint:
            on_checkpoint(ep, Q)
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay)
        s, _ = env.reset()
        total, traj = 0.0, []
        while True:
            qs = row(s)
            a = int(rng.integers(nA)) if rng.random() < eps else greedy_tie_random(qs, rng)
            s2, r, term, trunc, _ = env.step(a)
            total += r
            if algo == "Q-learning":
                nxt = 0.0 if term else gamma * (Q[s2].max() if s2 in Q else 0.0)
                qs[a] += alpha * (r + nxt - qs[a])
            else:
                traj.append((s, a, r))
            s = s2
            if term or trunc:
                break
        if algo == "Monte Carlo":
            first = {}
            for t, (si, ai, _) in enumerate(traj):
                first.setdefault((si, ai), t)
            G = 0.0
            for t in range(len(traj) - 1, -1, -1):
                si, ai, ri = traj[t]
                G = ri + gamma * G
                if first[(si, ai)] == t:
                    n = N.get((si, ai), 0) + 1
                    N[(si, ai)] = n
                    qr = row(si)
                    qr[ai] += (G - qr[ai]) * (mc_alpha if mc_alpha else 1.0 / n)
        returns[ep] = total
        success[ep] = float(term and env.state[:2] == env.goal)
    if episodes in cps and on_checkpoint:
        on_checkpoint(episodes, Q)
    return {"Q": Q, "returns": returns, "success": success}


def q_policy_dict(Q):
    """policy_fn(obs) -> action; state chưa từng thấy -> action 0."""
    return lambda s: int(np.argmax(Q[s])) if s in Q else 0
