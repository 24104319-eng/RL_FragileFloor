"""
Demo 1 episode bằng policy đã học (Q-learning hoặc Policy Iteration), in từng bước ra màn hình.

  python demo.py --map medium --algo pi
  python demo.py --map medium --algo q --episodes 30000 --seed 0
  python demo.py --map medium --algo q --episodes 30000 --save q_medium.npy     # lưu Q-table
  python demo.py --map medium --algo q --load q_medium.npy                      # dùng lại, khỏi train
Ký hiệu: A = agent, T = kho báu chưa nhặt, 1/2 = ô fragile còn 1/2 lần, x = ô đã sập, H = hố, G = đích.
"""
import argparse
import time

import numpy as np

from common import load_config, make_env, solve_pi, pi_policy_fn, q_policy_fn
from fragile_floor_env import ACTION_NAMES
from q_learning import q_learning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="medium")
    ap.add_argument("--config", default="default")
    ap.add_argument("--algo", choices=["q", "pi"], default="pi")
    ap.add_argument("--episodes", type=int, default=30000, help="số episode train Q-learning")
    ap.add_argument("--slip", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.0, help="giây chờ giữa các bước")
    ap.add_argument("--save", default=None)
    ap.add_argument("--load", default=None)
    args = ap.parse_args()

    cfg = load_config()
    env = make_env(args.map, cfg=cfg, config=args.config, slip_prob=args.slip)
    if args.algo == "pi":
        policy_fn = pi_policy_fn(solve_pi(env)[4])
        print("Đã giải xong bằng Policy Iteration.")
    else:
        if args.load:
            Q = np.load(args.load)
        else:
            p = dict(cfg["q_learning"]["default"])
            p["episodes"] = args.episodes
            print(f"Đang huấn luyện Q-learning {args.episodes} episode...")
            Q = q_learning(env, seed=args.seed, log_every=max(1, args.episodes // 5), **p)["Q"]
            if args.save:
                np.save(args.save, Q)
        policy_fn = q_policy_fn(Q)

    obs, _ = env.reset(seed=args.seed)
    print("\nTrạng thái ban đầu:")
    print(env.render())
    total = 0.0
    for t in range(1, env.max_steps + 1):
        a = int(policy_fn(obs))
        obs, r, term, trunc, info = env.step(a)
        total += r
        slip = "" if info["action_applied"] == a else f" (bị trượt -> {ACTION_NAMES[info['action_applied']]})"
        print(f"\nBước {t}: {ACTION_NAMES[a]}{slip}   reward={r:+.0f}   tổng={total:+.0f}")
        print(env.render())
        if args.delay:
            time.sleep(args.delay)
        if term or trunc:
            break
    ok = term and env.state[:2] == env.goal
    print("\n=> " + ("ĐẾN ĐÍCH" if ok else "THẤT BẠI (rơi hố / ô sập / hết bước)")
          + f"   return={total:.0f}, {t} bước")


if __name__ == "__main__":
    main()
