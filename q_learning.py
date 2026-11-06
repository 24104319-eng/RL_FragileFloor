"""
Q-learning dạng bảng (tabular) cho FragileFloorEnv - KHÔNG dùng DQN / function approximation.

Cập nhật:  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') * (1 - terminated) - Q(s,a) ]
- Hết max_steps (truncated) KHÔNG phải terminal thật nên vẫn bootstrap từ Q(s').
- epsilon giảm tuyến tính từ eps_start về eps_end trong eps_decay_frac * episodes episode đầu.
- Q-table kích thước env.n_states x 8 (mã hóa state có sẵn trong env.encode).
"""
import numpy as np


def greedy_action(q_row, rng):
    """argmax, hòa thì chọn ngẫu nhiên (tránh thiên vị action 0 khi Q còn bằng 0)."""
    best = np.flatnonzero(q_row >= q_row.max() - 1e-12)
    return int(rng.choice(best))


def q_learning(env, episodes=5000, alpha=0.1, gamma=0.99,
               eps_start=1.0, eps_end=0.05, eps_decay_frac=0.6,
               seed=0, log_every=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = np.zeros(episodes)
    success = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=np.int32)
    decay_eps = max(1, int(eps_decay_frac * episodes))
    env.reset(seed=seed)                      # đặt seed cho env (trượt, kho báu ngẫu nhiên)

    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
        s, _ = env.reset()
        total, steps = 0.0, 0
        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = greedy_action(Q[s], rng)
            s2, r, term, trunc, _ = env.step(a)
            target = r if term else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            total += r
            steps += 1
            if term or trunc:
                break
        returns[ep] = total
        success[ep] = float(term and env.state[:2] == env.goal)
        lengths[ep] = steps
        if log_every and (ep + 1) % log_every == 0:
            lo = max(0, ep + 1 - log_every)
            print(f"  ep {ep + 1:>7}: eps={eps:.2f}  return TB={returns[lo:ep + 1].mean():8.1f}  "
                  f"thành công={success[lo:ep + 1].mean():.0%}", flush=True)
    return {"Q": Q, "returns": returns, "success": success, "lengths": lengths}
