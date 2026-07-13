# TGRS 风格图件说明与算法对比

## 复现命令

先生成固定随机种子基准，再绘制 IEEE TGRS 双栏图：

```bash
python run_published_model_benchmark.py \
  -o published_model_benchmark --size 18 --seed 20260713 --overwrite
python plot_tgrs_hybrid_comparison.py published_model_benchmark --dpi 600
```

输出包括可编辑的 PDF/SVG，以及 600 dpi 的 TIFF/PNG。空间分布图每一行共享同一色标，因而同一场景内可以直接比较幅值和空间结构；性能图使用对数 RMSE 坐标，并将运行时间写入方法标签。

## 可直接使用的英文图注

**Fig. X. Spatial comparison of DEM-error estimates under four controlled observation regimes.** (a)-(d) Static DEM error mixed with nonlinear deformation: reference, HOMA-DEM, adaptive hypothesis-testing (Adaptive HT) method, and nonparametric ICA method. (e)-(h) Wrapped-phase data with local cycle errors: reference, HOMA-DEM, linear Huber regression, and the engineering-equivalent IGS-CMAES implementation. (i)-(l) Spatially sparse DEM error: reference, HOMA-DEM, the current uncertainty-aware graph model, and the engineering-equivalent phase-gradient direction-consistency (PGDC) implementation. (m)-(p) Height change during the observation period: reference, HOMA-DEM, dynamic-height SBAS, and the static graph model. A common symmetric color scale is used within each row. The experiment uses an 18 x 18 synthetic scene and the fixed random seed 20260713.

**Fig. Y. Accuracy and computational cost of HOMA-DEM and scenario-matched reference methods.** DEM-error or height-change RMSE is shown on a logarithmic scale; the value in parentheses is the wall-clock runtime for the corresponding 18 x 18 deterministic benchmark. HOMA-DEM is marked by circles, the current graph model by diamonds, and published reference methods by squares. HOMA-DEM gives the lowest RMSE in the static nonlinear, sparse-error, and dynamic-height experiments. In the wrapped/cycle-error experiment, specialized linear Huber regression is more accurate and faster, whereas HOMA-DEM remains substantially more accurate and faster than the engineering-equivalent IGS-CMAES implementation. Results from one fixed seed characterize this controlled realization and are not confidence intervals.

## 当前算法与已发表算法的区别

| 方法 | 核心假设或机制 | 与 HOMA-DEM 的主要区别 | 本项目复现状态 | 典型优势与局限 |
|---|---|---|---|---|
| Linear Huber | DEM 误差与基线系数线性相关，形变模型较简单；用 Huber IRLS 抑制离群干涉图 | 单一低复杂度回归；HOMA-DEM 将其作为候选专家之一，并允许非线性形变、动态高度和缠绕域仲裁 | 项目基线 | 简单场景速度快、方差低；模型失配时 DEM 与非线性形变容易混叠 |
| 分形表面正则化（2015） | 利用 DEM/干涉图的分形空间结构联合约束相位和幅度 | 固定空间先验；HOMA-DEM 只对高不确定度像元施加受控图正则，避免全场过平滑 | 适配实现 | 可抑制孤立尖峰并保持结构；原文依赖复数幅度，普通 MintPy 栈不能严格复现 |
| 非参数 ICA（2019） | 用 SVHT 与 ICA 分离时序源，再选取与垂直基线相关的分量 | 尽量减少预设形变函数；HOMA-DEM 仅在 ICA 收敛且基线相关分量通过统计检验时接纳该候选 | 严格流程复现 | 对未知形变形式更灵活；对样本量、源独立性、网络连通性和尺度恢复敏感 |
| Adaptive HT（2021） | 通过 F/t 假设检验自适应选择分段、非线性和周期形变项 | 单一自适应时间模型；HOMA-DEM 将其与稳健回归、ICA、动态高度和缠绕域模型逐像元竞争 | 严格流程复现 | 非线性形变场景表现稳定；无法单独解决空间稀疏性、真实高度突变或相位周期歧义 |
| IGS-CMAES（2021） | 在缠绕相位域以 RI-L1 目标联合搜索 DEM 误差和形变参数 | 全局搜索型估计；HOMA-DEM 仅在周期残差或专家分歧触发时调用，并将其作为仲裁器 | 工程等价实现 | 能直接处理周期歧义；计算量大且存在别名解，本项目局部优化器并非原作者 CMA-ES 依赖 |
| PGDC（2025） | 先用相位梯度方向一致性检测 DEM 误差位置，再进行约束估计 | 检测优先；HOMA-DEM 将 PGDC 与显著性、动态高度和 IGS 结果联合成稀疏门控，不直接采用其单一估计值 | 工程等价实现 | 可减少无误差区过校正；检测阈值、网络组合和参考区选择会导致漏检或幅值低估 |
| Dynamic SBAS（2025） | 枚举高度突变历元，联合估计突变前后高度误差和形变 | 专用动态高度模型；HOMA-DEM 增加 F 检验、最小变化量和不确定度门限，只在证据充分时启用动态分支 | 严格流程复现 | 适合开挖、施工和滑坡等真实高程变化；静态区域可能因额外自由度而过拟合 |
| 当前图模型 | Huber/VCE 时序反演后进行固定图正则 | 全场固定平滑；HOMA-DEM 根据局部不确定度最多混入 35% 图正则结果，并保护 IGS 确认像元 | 当前项目模型 | 对噪声有空间降噪作用；固定正则强度会过平滑边界和稀疏异常 |
| HOMA-DEM | 分层观测感知混合专家：候选估计、统计选择、动态高度检验、PGDC 门控、IGS 仲裁和不确定度图正则 | 不假定一个模型适合所有像元，而是控制复杂模型的触发条件并输出专家来源 | 本项目新方法，尚未发表 | 降低跨场景模型失配风险且可解释；运行时间高于简单回归，仍需多数据集统计验证和消融实验 |

## 定量性能解释

固定种子 `20260713`、`18 x 18` 场景的结果如下：

| 场景 | HOMA-DEM | 主要对照 | 结果解释 |
|---|---:|---:|---|
| 静态 DEM + 非线性形变 | 0.02784 m，0.45 s | Adaptive HT：0.03229 m，0.10 s | HOMA-DEM 的 RMSE 低约 13.8%，但运行时间约为 4.5 倍 |
| 缠绕相位 + 周期错误 | 0.01191 m，1.00 s | Linear Huber：0.00980 m，0.02 s；IGS：0.06304 m，5.51 s | HOMA-DEM 比专用 Linear Huber 高约 21.5%，但比 IGS 低约 81.1%，且约快 5.5 倍 |
| 稀疏 DEM 误差 | 0.09334 m，0.46 s | 当前图模型：1.02815 m；PGDC：5.34007 m | RMSE 分别降低约 90.9% 和 98.3%，说明联合门控比单独检测或固定平滑更适合该模拟 |
| 观测期内高度突变 | 0.00436 m，0.45 s | Dynamic SBAS：0.07712 m；静态图模型：5.62183 m | RMSE 分别降低约 94.4% 和 99.9%，主要来自动态分支检验与静态/动态模型选择 |

这些结果支持的结论是：HOMA-DEM 在四类机制不同的观测条件下具有更低的最坏场景误差，并在三类场景中取得最低 RMSE；它不是每个场景都优于专用方法。尤其在形变简单且解缠结果可直接使用时，Linear Huber 更快且略准确。

## 发表时必须保留的边界条件

1. 当前图仅来自一个固定随机种子，没有重复试验、标准差或置信区间，不能据此声称统计显著优于已发表方法。
2. IGS-CMAES 与 PGDC 是工程等价实现，分形方法是 MintPy 数据约束下的适配实现；论文中应分别标注为 `equivalent` 和 `adapted`，不能写成严格复现。
3. 不同场景的真值幅值、噪声机制和评价对象不同，不应把四个 RMSE 直接合并成一个总排名。
4. 投稿前应增加至少 30-100 个随机种子、不同地形粗糙度/基线网络/相干性/大气噪声水平的统计实验，并报告中位数、四分位数或 95% 置信区间。
5. 至少使用两类真实 MintPy 数据，并以 LiDAR、ICESat-2、稳定区残差或留出干涉图作为独立验证；同时报告消融实验、失败案例、运行时间和峰值内存。

