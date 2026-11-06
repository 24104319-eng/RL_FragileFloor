# Fragile Floor - chạy toàn bộ thí nghiệm (3 map × 10 kịch bản × 5 thuật toán)

## Chạy trên Colab
```python
# upload thư mục fragile_floor (hoặc giải nén zip) vào /content/fragile_floor
!pip install -q gymnasium pandas
%cd /content/fragile_floor

!python run_all.py --quick                    # 1) chạy thử (vài phút) -> results_quick/, kiểm tra không lỗi
!python run_all.py --seeds 3 --workers 2      # 2) chạy thật -> results/   (khoảng 40-90 phút)
!python run_all.py --exp rewards              # 3) thí nghiệm cấu hình reward (vài phút)
!zip -r results.zip results                   # 4) nén để gửi lại
```
- Có **cache**: mỗi (map, kịch bản) lưu vào `results/cache/`. Colab bị ngắt thì chạy lại đúng lệnh cũ, phần đã xong được bỏ qua.
- Chạy từng phần: `--maps easy medium`, `--scenarios S03 S07`, `--seeds 5`, `--scale 2` (nhân số episode MC/Q-learning).
- Dựng lại bảng/hình từ cache (không train lại): thêm `--plots-only`.
- Số episode MC/Q-learning, θ, k của Truncated PI, số ván đánh giá: mục `run_all` trong `maps.json`.

## 10 kịch bản (gộp từ 16)
| ID        | kho báu | bắt buộc nhặt    | vị trí kho báu         | trượt   |
|---        |---      |---               |---                     |---      |
| S01 / S02 | không   | -                | -                      | 0 / 0.1 |
| S03 / S07 | có      | có               | cố định (T trong map)  | 0 / 0.1 |
| S04 / S08 | có      | không (tùy chọn) | cố định                | 0 / 0.1 |
| S05 / S09 | có      | có               | ngẫu nhiên mỗi episode | 0 / 0.1 |
| S06 / S10 | có      | không (tùy chọn) | ngẫu nhiên             | 0 / 0.1 |
(Khi không có kho báu thì "bắt buộc/ngẫu nhiên" vô nghĩa nên 4 kịch bản trùng nhau được gộp thành 1 cho mỗi mức trượt.)

## 5 thuật toán
Value Iteration, Policy Iteration, Truncated PI (k=5) - quy hoạch động, dùng mô hình; Monte Carlo control (first-visit, ε-greedy), Q-learning - tabular, không DQN.

## Kết quả (thư mục results/)
| File                                                                                                  | Dùng cho phần báo cáo                                                                             |
|---                                                                                                    |---                                                                                                |
| `tables/T1_state_action_space.csv`                                                                    | kích thước state/action space từng map & kịch bản                                                 |
| `tables/T2_dp_convergence.csv`                                                                        | ngưỡng θ, số vòng lặp, số lần quét, residual Bellman, thời gian, vòng sau đó policy không đổi nữa |
| `tables/T3_final_comparison_by_scenario.csv`, `T3b_...by_map.csv`                                     | return, tỉ lệ tới goal, số bước, repair, kho báu, độ chính xác action so với tối ưu, thời gian    |
| `tables/T4_convergence_thresholds_learners.csv`                                                       | số episode để MC/Q-learning đạt 90%/95% return tối ưu và ≥90% tới goal (theo seed)                |
| `tables/T5_policy_vs_steps_checkpoints.csv`                                                           | chất lượng policy sau 0%,1%,...,100% episode (hoặc sau iteration 0,1,2,5,... với DP)              |
| `tables/T6_start_state_Q_values.csv`                                                                  | Q(s0, a) của 8 action tại state đầu: Q* (VI) so với MC/Q-learning theo episode                    |
| `tables/T7_reward_configs.csv`, `rewards_policies_<map>.png`                                          | thí nghiệm cấu hình reward                                                                        |
| `policy_grids/<map>__<kịch bản>.png`                                                                  | **lưới policy**: ban đầu → ~10% → ~50% → cuối → tối ưu (ô đỏ = khác tối ưu) cho cả 5 thuật toán   |
| `trajectories/*.png`                                                                                  | đường đi của policy ban đầu vs policy cuối (R = dùng repair kit, ★ = nhặt kho báu, X = rơi)       |
| `decision_maps/*.png`                                                                                 | kho báu ngẫu nhiên: ô spawn nào agent đi nhặt (o) hay bỏ (x)                                      |
| `curves_train_<map>.png`, `curves_greedy_<map>.png`, `dp_convergence_<map>.png`, `compare_methods.png`| learning curve, hội tụ, so sánh                                                                   |
| `report_tables.md`                                                                                    | các bảng chính ở dạng Markdown                                                                    |

## Định nghĩa "policy ban đầu"
- Value Iteration: policy tham lam theo V₀ = 0 (chỉ nhìn reward tức thời).
- Policy Iteration / Truncated PI: π₀ = luôn đi LÊN (action 0).
- Monte Carlo / Q-learning: Q = 0 → policy ngẫu nhiên (hiển thị bằng dấu "?"); đường đi vẽ theo greedy với Q = 0 (action 0).

## Demo 1 episode bằng policy đã học
`!python demo.py --map medium --algo q --episodes 40000` hoặc `--algo pi`.
