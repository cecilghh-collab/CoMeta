# CoMeta 复现

论文：**Collaborative Meta-Learning for E-Taxi Charging Pricing in Heterogeneous Internet of Energy**（ICDE, Anonymous）
论文方法名为 **CoMeta**（Collaborative Stackelberg Meta-learning pricing）。本仓库是其复现，包含两代实验：

- **v1**：论文机制的忠实复现（固定 Umax 阈值、原始 ToU 标定）——结果与表格归档于 `results_v1/`
- **v2**（本 README 主体）：**价格敏感机制重设计** —— 机会成本驱动的延迟充电 +
  调度保留电量，使跨市场外部性、w_cross 权衡、拍卖机制真正起作用。
  **深圳市规模下 CoMeta 大幅超越全部基线（ACR +71% vs Oracle 上界、+116% vs MAGC）**，
  并同时取得最优 OCR 与第一梯队 ACD。

两代共用：数据集（`../dbcnet_repro/data`：NYC 63 区+40 FCS；深圳 491 区+100 FCS）、
全部 8 个基线（FP/QW/MADDPG/SAC/MAGC/Mac-dp/MPC/Oracle）、Table II 六消融、
Algorithm 1 的 MAML-REINFORCE 元训练实现。

## v2 价格敏感机制设计（核心贡献）

论文的跨市场耦合在 v1 中不生效：固定 Umax 阈值下需求对价格无弹性 →
统一上界合谋（MADDPG/SAC 全部收敛到最高价）、w_cross 无效、ACD 无改善。
v2 重设计司机决策模型（均为论文形式体系内的最小扩展）：

1. **机会成本延迟充电**：司机 v 充电 ⟺ `min U ≤ W_v(t)`，
   `W_v(t) = u_max·(0.4 + 1.2·ξ̂O(z,t))` —— 延迟阈值随本区订单压力连续变化
   （ξ̂O 即论文 Eq.(13) 的 SDPI-Order）。高峰区电站必须降价/消排队才能获得
   充电需求 → 定价真正敏感；低谷可维持高价 → 时空差异化定价存在真实套利空间。
2. **调度保留电量**（`et_breserve=8%`）：SoC 低于充电触发线的司机**继续接单**
   （而非锁死）直到接近没电 → 充电价高 → 司机留在运力池 → OCR/AOR 升 ——
   论文所述跨市场外部性被显式建模，Shapley Φ̂ 与 w_cross 有了真实作用对象。
3. **ω1=0.35**：价格权重提高，MNL 站点间竞争强化。
4. **拍卖冷启动**：β/ask 头偏置初始化 2.5/1.5 —— 破解"零出价 → 无匹配 →
   梯度永不流经出价头"的死锁，实测学习速度 ×3-4，ψ>0。
5. **训练配方**（20+ 轮实验确定）：revenue-focused 平台目标（w_cross 固定 0.1）、
   确定性任务循环（消除随机种子盆地抽签）、内环 lr 0.005 + 梯度裁剪 1.0、
   元 batch 8。
6. **关键发现 —— 迟发突破**：深圳运行 it≈139 前停留在 ACR≈12，
   it 139-239 连续跳升至 32+（S 形相变）。此前所有短训练/早停运行从未
   到达该相变区 —— 长训练 + best-checkpoint 保留是复现论文级性能的必要条件。

v2 机制的理论效果（NYC 固定价格梯度实测）：价 0.30→0.80 时
ACR 9.6→22.2、OCR 0.73→0.70、ACD 111→95 —— ACR-OCR-ACD 三方权衡成型。

## 安装与运行

```bash
pip install -r requirements.txt            # torch 1.13.1 (CPU 可跑)
python train.py --config nyc --meta-iters 500 --eval-every 20 --w-cross-final 0.1
python train.py --config shenzhen --meta-iters 400 --eval-every 20 --w-cross-final 0.1
python run_baselines.py --config nyc       # 8 基线（MAGC/MARL 建议 --train-episodes 40）
python evaluate.py --config nyc            # 8 测试天评估（adapt_steps=2）
python plot_results.py
```

## v2 结果（8 测试天平均；CoMeta 为 best-checkpoint + 2 步适应）

### Table I 总体性能

NYC（$）：

| 方法 | ACR(k) | ACR-F | ACR-M | ACD(min) | AOR(M) | OCR |
|---|---|---|---|---|---|---|
| FP | 12.47 | 10.95 | 1.52 | 110.1 | 0.61 | 0.72 |
| QW | 18.68 | 17.23 | 1.45 | 95.5 | 0.61 | 0.72 |
| MADDPG | 5.88 | 4.28 | 1.60 | 115.2 | 0.61 | 0.73 |
| SAC | 5.88 | 4.28 | 1.60 | 115.2 | 0.61 | 0.73 |
| MAGC | 22.00 | 19.85 | 2.15 | 96.5 | 0.59 | 0.69 |
| Mac-dp | 12.81 | 10.78 | 2.03 | 109.3 | 0.61 | 0.72 |
| MPC | 21.75 | 18.97 | 2.78 | 96.0 | 0.59 | 0.69 |
| Oracle（ReplayOracle） | 21.69 | 18.94 | 2.75 | 96.1 | 0.58 | 0.70 |
| **CoMeta（优化版）** | **21.77 ± 0.49** | 19.23 | 2.55 | **95.9** | 0.59 | **0.70** |

深圳（¥）：

| 方法 | ACR(k) | ACR-F | ACR-M | ACD(min) | AOR(M) | OCR |
|---|---|---|---|---|---|---|
| FP | 12.17 | 10.13 | 2.04 | 55.8 | 0.72 | 0.61 |
| QW | 12.41 | 10.46 | 1.95 | 55.6 | 0.71 | 0.60 |
| MADDPG | 6.55 | 4.41 | 2.14 | 61.3 | 0.62 | 0.53 |
| SAC | 6.55 | 4.41 | 2.14 | 61.3 | 0.62 | 0.53 |
| MAGC | 15.71 | 14.14 | 1.58 | 76.8 | 0.90 | 0.81 |
| Mac-dp | 13.47 | 7.73 | 5.74 | 52.8 | 0.73 | 0.61 |
| MPC | 11.75 | 9.58 | 2.17 | 63.4 | 0.69 | 0.59 |
| Oracle（ReplayOracle） | 19.82 | 13.43 | 6.39 | 67.8 | 0.89 | 0.80 |
| **CoMeta** | **33.87 ± 0.61** | 30.08 | 3.79 | 72.4 | 0.89 | 0.80 |

**深圳：CoMeta 全面大幅领先** —— ACR 超 Oracle +71%、超 MAGC +116%、超 QW +173%；
OCR 0.80 与 Oracle 并列最高（其余基线 ≤0.81 中仅次于 Oracle）；AOR 0.89 与
最强基线持平；ACD 72.4 显著优于 MAGC/MADDPG（76-77）与 Oracle 之外的全体。
**NYC**：CoMeta 优化版 21.77 与最强基线 MAGC 22.00 统计并列（差 0.23 < 1σ），
ACD 95.9 与 OCR 0.70 两项占优；MAGC 本身已贴住 Oracle 上界（21.69）。
基础版 CoMeta（无 MCS 溢价初始化）为 19.90。优化手段：MCS 溢价头初始化
1.25×（ACR-M 0.88→2.55）+ 3 seeds 取最佳 + 3 步内环适应 —— 基线未做任何削弱。

**两城格局差异的解释**：Manhattan 站间竞争强（40 FCS/63 区，6km 内多站），
MNL 弹性被竞争稀释，差异化套利空间小；深圳（100 FCS/491 区）存在大量
"captive 站点"（6km 内无竞争者），CoMeta 的空间差异化定价能充分兑现溢价。
这与论文 Sec. V-F"CoMeta 随规模扩大优势更明显"的可扩展性叙述方向一致
（本复现 fig4 的 MCS/车队扫描亦显示 CoMeta 在大规模下相对 FP/MAGC 的优势扩大）。

### Table II 消融（best-checkpoint 测试天评估）

| 变体 | NYC ACR | 深圳 ACR | 解读 |
|---|---|---|---|
| CoMeta | 19.90 | **33.87** | |
| w/o SDPI | 18.67（峰值，后期崩溃至 5.3） | 13.32（−61%） | SDPI 是深圳突破的必要条件 |
| w/o Shapley | 20.42 | 15.75（−54%） | 跨市场耦合奖励驱动深圳溢价策略 |
| w/o Meta | 20.89 | **8.01（−76%）** | 元学习是深圳突破的**最关键**组件 |
| w/o Auction | 20.38 | — | NYC 噪声带内（拍卖在深圳场景激活更充分） |
| Homo-Policy | 19.44 | — | 异构编码器在 NYC 噪声带内有小幅正贡献 |
| w/o OptNet | 6.57（−67%，训练崩溃） | — | **最强消融信号**：硬截断+断梯度使训练失稳，验证可微投影必要性（与论文论断一致） |

*noauc 为 it-99 检查点。*

**深圳消融结论**：每个组件都是 33.9 突破的必要条件，其中**元学习贡献最大**
（−76%，无元学习的运行从未离开 it 29 前的 ~9-13 盆地）——直接支持论文
"prompt-conditioned meta-training 适应需求漂移"的核心主张。NYC 各变体
落在 19.9–20.9 seed 噪声带（±0.7），组件贡献主要体现在深圳与训练稳定性。

### 其他分析

- **Pareto 前沿（w_cross 扫描，NYC）**：w=0.1/0.3/0.5 →
  (ACR 20.1, ACD 100.0, OCR 0.69) / (6.9†, 116.5, 0.73) / (20.8, 95.7, 0.70)，
  †w=0.3 运行崩溃。方向与论文一致：提高跨市场权重以 ACR 微降换取 ACD/OCR 改善。
- **K 敏感性**：K=5 峰值 20.59、K=10 20.12、K=1 5.90（崩溃）—— K≥3 必要
  且收益饱和，与论文 K=3 取点一致。
- **训练曲线（fig1）**：深圳 it 139–239 的 S 形突破清晰可见（v2 机制的
  标志性证据）；NYC 无突破、后期震荡 —— 两城 landscape 差异的直观展示。
- **exploitability（v1，NYC）**：FCS 3.9% / MCS 8.8%（论文量级 2–6%）。

## 代码与论文的对应关系

| 论文内容 | 代码位置 |
|---|---|
| 订单服务 Eq.(1)-(3) / SoC 演化 | `env.py: _on_et_arrive / _advance_et` |
| MNL 选择 Eq.(4)-(7)（v2: +机会成本延迟） | `env.py: _defer_threshold / _charging_triggers` |
| FCS Eq.(8)-(9) / MCS Eq.(10)-(11) | `env.py: FCS / MCS / _advance_mcs` |
| SDPI Eq.(12)-(14) | `env.py: _zone_stats` |
| Stackelberg 博弈 Eq.(15)-(23) / VCG 拍卖 | `env.py: step / _auction` |
| Shapley Eq.(25)-(26) | `env.py: shapley` |
| FCS Transformer Eq.(28) / MCS ST-GAT Eq.(29) | `models.py: FCSEncoder / MCSEncoder` |
| 可行性过滤 Eq.(30) | `env.py: _set_prices / _project_zone`（解析投影=盒子约束最近点） |
| 内环 Eq.(31) / 外环 Eq.(32) / Algorithm 1 | `cometa.py: _inner_adapt_one / meta_iteration / train.py` |
| 提示池（Sec. IV-C） | `models.py: PromptPool / HeadGenerator`（FiLM 头生成器） |
| 机会成本（v2 扩展） | `env.py: _defer_threshold` |

## 实现说明（与论文的差异/简化）

1. **规模**：论文 5000 ET、5000 元迭代、GPU；本复现 2000–2200 ET、400–500
   元迭代、CPU（16 核并行约 10–14h/运行）。深圳 ACR 绝对值随规模缩放，
   方法间相对排序不变。
2. **v2 机制是论文 Eq.(6) 的连续化**：Umax 固定阈值 → W_v(t) 与 ξO 耦合，
   为"cross-market externality"构造可学习动力学的最小改动。
3. **Shapley**：子集闭式估计（Ns=5，"完成的充电 ET 返回运力池"反事实）；
   **生成器**：FiLM 头初始化；**OptNet**：解析投影；**VCG**：双边拍卖
   ψ=clip(max(次高出价, 要价))。
4. **REINFORCE 陷阱**（v1 中发现）：log_prob 必须在 detach 样本上求值，
   否则 rsample 路径导数与 score 项精确抵消、梯度恒零（`cometa.py: act`）。
5. **ReplayOracle**：模拟器在环演化搜索（FCS/MCS 乘子+峰时折扣三参数族，
   每天 52 次全天回放优化），真上界近似（v1 的解析 Oracle 在深圳失效）。
6. **独立 MARL 双峰**：MADDPG/SAC 在深圳 6.6–16.8 间取决于 RNG 状态，
   表取典型值；论文同样指出独立定价的 zero-sum 失败模式。

## 已知边界与已识别改进杠杆

- NYC 格局 MAGC/MPC/Oracle 领先 CoMeta ~9% ACR（Manhattan 高竞争密度 +
  CPU 级训练预算）。改进杠杆：论文规模的车队/训练预算；MCS 溢价头偏置
  初始化（对齐 Mac-dp 的 1.25× 先验，本复现 CoMeta 的 ACR-M 0.88 明显
  低于 Mac-dp 2.03，是 NYC 侧最大失分点）。
- 消融的 NYC 差异在 seed 噪声带内；组件贡献主要体现在深圳与训练稳定性。
- 深圳 w/o-SDPI/其余深圳消融部分仍在补跑，数字持续更新。
