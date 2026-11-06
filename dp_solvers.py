"""
Dynamic Programming (quy hoạch động) cho FragileFloorEnv:
    1. Value Iteration
    2. Policy Iteration
    3. Truncated Policy Iteration (đánh giá policy chỉ k lần quét mỗi vòng)

Dùng mô hình P từ env.build_model(), chỉ gồm các state reachable.
Hỗ trợ cả môi trường có trượt (slip_prob > 0): mỗi (state, action) có tối đa 4 kết quả.

Lưu ý: gamma phải < 1 để policy evaluation hội tụ với policy xấu
(ví dụ policy đâm tường mãi, return sẽ là -vô cùng nếu gamma = 1).
"""
import sys
import time
import numpy as np

from fragile_floor_env import FragileFloorEnv, ACTION_NAMES, MAPS


# ---------------------------------------------------------------------
# 1. Chuyển mô hình P (dict) sang mảng numpy để tính nhanh
# ---------------------------------------------------------------------
class TabularModel:
    """
    ids[i]    : mã state (int) của state có chỉ số i
    idx[s]    : tra ngược, từ mã state sang chỉ số i
    NS[i,a,k] : chỉ số state kế tiếp của kết quả thứ k
    W[i,a,k]  : xác suất kết quả k, đã nhân (1 - kết thúc): bước terminal không cộng V
    Rexp[i,a] : reward kỳ vọng = tổng (xác suất * reward)
    start_idx : chỉ số các state bắt đầu (random_treasure: mỗi vị trí kho báu một state)
    K         : số kết quả tối đa của một (state, action) (1 nếu không trượt, 4 nếu trượt)
    Q(i,a) = Rexp[i,a] + gamma * tổng_k W[i,a,k] * V[NS[i,a,k]]
    """

    def __init__(self, env):
        states, P = env.build_model()
        self.env = env
        self.ids = np.array(states)
        self.idx = {s: i for i, s in enumerate(states)}
        self.n, self.A = len(states), env.n_actions
        self.start_idx = [self.idx[env.encode(st)] for st in env.initial_states()]
        self.K = max(len(P[s][a]) for s in states for a in range(self.A))
        self.NS = np.zeros((self.n, self.A, self.K), dtype=np.int32)
        self.W = np.zeros((self.n, self.A, self.K))
        self.Rexp = np.zeros((self.n, self.A))
        for s in states:
            i = self.idx[s]
            for a in range(self.A):
                self.NS[i, a, :] = i                    # phần đệm: trỏ về chính nó, xác suất 0
                for k, (p, ns, r, d) in enumerate(P[s][a]):
                    self.NS[i, a, k] = self.idx[ns]
                    self.W[i, a, k] = 0.0 if d else p
                    self.Rexp[i, a] += p * r


# ---------------------------------------------------------------------
# 2. Các hàm dùng chung
# ---------------------------------------------------------------------
def q_values(model, V, gamma):
    """Q[i,a] = reward kỳ vọng + gamma * giá trị kỳ vọng của state kế tiếp."""
    return model.Rexp + gamma * (model.W * V[model.NS]).sum(axis=2)


def greedy_policy(model, V, gamma, old_policy=None, tol=1e-10):
    """
    Policy tham lam theo V. Nếu có old_policy, chỉ đổi action khi action mới
    tốt hơn thật sự (hơn quá tol) để tránh đổi qua lại giữa các action hòa nhau.
    """
    Q = q_values(model, V, gamma)
    best = Q.argmax(axis=1)
    if old_policy is None:
        return best
    ar = np.arange(model.n)
    keep = Q[ar, old_policy] >= Q[ar, best] - tol
    return np.where(keep, old_policy, best)


def evaluate_policy(model, policy, V, gamma, theta=1e-8, max_sweeps=None):
    """
    Policy evaluation: lặp V <- R_pi + gamma * V(NS_pi).
    Dừng khi thay đổi lớn nhất < theta, hoặc khi đủ max_sweeps lần quét.
    Trả về (V, số lần quét).
    """
    ar = np.arange(model.n)
    Rp = model.Rexp[ar, policy]
    Wp = gamma * model.W[ar, policy]          # (n, K)
    NSp = model.NS[ar, policy]                # (n, K)
    sweeps = 0
    while True:
        V_new = Rp + (Wp * V[NSp]).sum(axis=1)
        delta = np.abs(V_new - V).max()
        V = V_new
        sweeps += 1
        if max_sweeps is not None and sweeps >= max_sweeps:
            break
        if delta < theta:
            break
    return V, sweeps


# ---------------------------------------------------------------------
# 3. Ba thuật toán
# ---------------------------------------------------------------------
def value_iteration(model, gamma=0.99, theta=1e-8, max_iter=100000):
    t0 = time.time()
    V = np.zeros(model.n)
    history = []
    for k in range(1, max_iter + 1):
        V_new = q_values(model, V, gamma).max(axis=1)   # Bellman optimality
        delta = np.abs(V_new - V).max()
        V = V_new
        history.append(delta)
        if delta < theta:
            break
    policy = greedy_policy(model, V, gamma)
    info = {"iterations": k, "sweeps": k, "history": history, "time": time.time() - t0}
    return V, policy, info


def policy_iteration(model, gamma=0.99, theta=1e-8, max_iter=10000):
    t0 = time.time()
    V = np.zeros(model.n)
    policy = np.zeros(model.n, dtype=np.int64)          # policy ban đầu: toàn action 0
    total_sweeps, history = 0, []
    for it in range(1, max_iter + 1):
        V, sw = evaluate_policy(model, policy, V, gamma, theta)        # đánh giá đầy đủ
        total_sweeps += sw
        new_policy = greedy_policy(model, V, gamma, old_policy=policy)  # cải thiện
        history.append(int((new_policy != policy).sum()))   # số state đổi action
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    info = {"iterations": it, "sweeps": total_sweeps, "history": history,
            "time": time.time() - t0}
    return V, policy, info


def truncated_policy_iteration(model, gamma=0.99, k=5, theta=1e-8, max_iter=100000):
    """
    Mỗi vòng: cải thiện policy theo V hiện tại, rồi đánh giá chỉ k lần quét.
    k = 1 giống Value Iteration; k lớn (-> vô cùng) giống Policy Iteration.
    """
    t0 = time.time()
    V = np.zeros(model.n)
    policy = np.zeros(model.n, dtype=np.int64)
    history = []
    for it in range(1, max_iter + 1):
        policy = greedy_policy(model, V, gamma, old_policy=policy)      # cải thiện
        V_old = V
        V, _ = evaluate_policy(model, policy, V, gamma, theta=0.0, max_sweeps=k)
        delta = np.abs(V - V_old).max()
        history.append(delta)
        if delta < theta:
            break
    policy = greedy_policy(model, V, gamma, old_policy=policy)
    info = {"iterations": it, "sweeps": it * k, "history": history,
            "time": time.time() - t0}
    return V, policy, info


# ---------------------------------------------------------------------
# 4. Chạy thử policy tìm được
# ---------------------------------------------------------------------
def run_policy(model, policy, env=None, options=None, seed=None):
    """Chơi 1 episode theo policy. Trả về (return, danh sách tên action, tới goal?)."""
    env = env or model.env
    obs, _ = env.reset(seed=seed, options=options)
    total, actions = 0.0, []
    while True:
        a = int(policy[model.idx[obs]])
        actions.append(ACTION_NAMES[a])
        obs, r, term, trunc, _ = env.step(a)
        total += r
        if term or trunc:
            return total, actions, term and env.state[:2] == env.goal


def treasure_decision_map(model, policy, layout_kw):
    """
    Random treasure: với từng ô spawn, chạy policy trên bản KHÔNG trượt xem agent
    có đi nhặt không. In bản đồ: o = đi nhặt, x = bỏ, . = ô không spawn.
    """
    env0 = FragileFloorEnv(**{**layout_kw, "slip_prob": 0.0})
    pid = {int(model.ids[i]): int(policy[i]) for i in range(model.n)}
    grid = [list(row) for row in model.env.grid]
    collect = 0
    for cell in env0.spawn_cells:
        obs, _ = env0.reset(options={"treasure": cell})
        got = False
        for _ in range(200):
            obs, r, te, tr, _ = env0.step(pid[obs])
            got = got or (r > 0 and r < env0.goal_reward)   # reward nhặt kho báu
            if te or tr:
                break
        grid[cell[0]][cell[1]] = "o" if got else "x"
        collect += got
    return ["".join(row) for row in grid], collect, len(env0.spawn_cells)


if __name__ == "__main__":
    # python dp_solvers.py          -> kho báu cố định (T trong map), bắt buộc nhặt
    # python dp_solvers.py random   -> kho báu spawn ngẫu nhiên, nhặt hay không tùy agent
    random_mode = "random" in sys.argv
    gamma = 0.99
    for name in MAPS:
        for slip in (0.0, 0.1):
            kw = dict(layout=name, slip_prob=slip, random_treasure=random_mode,
                      require_all_treasures=not random_mode)
            env = FragileFloorEnv(**kw)
            model = TabularModel(env)
            print(f"\n=== map {name}, slip={slip}: {model.n} state reachable, "
                  f"{env.n_fragile} fragile, "
                  f"{'kho báu ngẫu nhiên (%d ô spawn)' % len(env.spawn_cells) if random_mode else str(env.n_treasure) + ' kho báu cố định'} ===")
            print(f"{'thuật toán':<22}{'vòng lặp':>9}{'tổng quét':>11}{'thời gian':>11}{'V(start)':>11}")
            runs = [
                ("Value Iteration", value_iteration(model, gamma, 1e-6)),
                ("Policy Iteration", policy_iteration(model, gamma, 1e-6)),
                ("Truncated PI (k=5)", truncated_policy_iteration(model, gamma, k=5, theta=1e-6)),
                ("Truncated PI (k=20)", truncated_policy_iteration(model, gamma, k=20, theta=1e-6)),
            ]
            for label, (V, pol, info) in runs:
                v_start = V[model.start_idx].mean()      # trung bình theo vị trí kho báu
                print(f"{label:<22}{info['iterations']:>9}{info['sweeps']:>11}"
                      f"{info['time']:>10.2f}s{v_start:>11.3f}")

            pol = runs[0][1][1]
            if random_mode:
                rows, collect, total = treasure_decision_map(model, pol, kw)
                print(f"Agent đi nhặt kho báu ở {collect}/{total} ô spawn "
                      f"(o = nhặt, x = bỏ):")
                print("\n".join(rows))
            elif slip == 0:
                ret, acts, ok = run_policy(model, pol)
                print(f"Policy tối ưu: return={ret:.0f}, {len(acts)} bước, "
                      f"dùng repair {sum(a.startswith('REPAIR') for a in acts)} lần")
                print(" ".join(acts))
            if slip > 0:   # có trượt: thống kê nhiều ván
                rets, wins = [], 0
                for ep in range(300):
                    ret, _, ok = run_policy(model, pol, seed=ep)
                    rets.append(ret)
                    wins += ok
                print(f"300 ván: return TB={np.mean(rets):.1f}, tới goal {wins / 300:.0%}")
