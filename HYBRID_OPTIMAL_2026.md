# HOMA-DEM：分层观测感知混合专家 DEM 误差校正

`hybrid_optimal_2026` 的建议论文名称为 **HOMA-DEM**（Hierarchical
Observation-aware Mixture of Adaptive experts for DEM-error correction）。它不是宣称存在一个对所有 InSAR 场景都全局最优的估计器，而是通过检测、候选反演、统计检验和受控正则化，在像元层面选择当前观测最支持的模型。

## 1. 统一观测模型

对干涉图 `m` 和像元 `p`：

```text
phi_m(p) = c_m(p) * delta_h(p) + d_m(p) + epsilon_m(p)
c_m(p) = -4*pi/lambda * Bperp_m(p) / (R(p)*sin(theta(p)))
```

其中 `delta_h` 是待估 DEM 误差，`d_m` 是形变，`epsilon_m` 包含大气、轨道、解缠和随机噪声。

## 2. 候选专家

HOMA-DEM 同时构建以下静态 DEM 候选：

1. `adaptive_ht_2021`：通过 F/t 检验选择分段非线性和周期形变项。
2. `adaptive_huber`：线性/二次 BIC 自适应加 Huber IRLS，抵抗异常干涉图。
3. `linear_huber`：低方差简单模型，适合线性形变和少量异常值。
4. `ica_2019`：只有在 FastICA 收敛、基线相关分量通过 F 检验且相关系数绝对值不低于 `0.7` 时才参与竞争。

对每个候选 DEM `h_q`，先扣除其 DEM 相位，再用三次多项式和年周期项剖面拟合形变。评分采用 Huber 加权 BIC：

```text
BIC_q = n*log(RSS_q/n) + k*log(n)
```

像元首先选择 BIC 最小的专家，避免预先固定一种形变函数。

## 3. 动态高度分支

改进 SBAS 对所有允许的突变历元分别拟合前后 `[DEM error, velocity]`。只有同时满足以下条件才接受动态高度模型：

```text
F = ((RSS_static - RSS_dynamic)/2) / (RSS_dynamic/(n-4))
F > F_(1-alpha)(2,n-4)
abs(h_after-h_before) >= max(h_min, 3*sigma_change)
```

默认 `alpha=0.01`、`h_min=2 m`。这样可防止静态区因为增加参数而被误判为高度突变。

## 4. 缠绕相位仲裁

IGS 不再全场运行，只在以下任一条件成立时触发：

- 剖面残差中检测到超过阈值的 `2*pi` 周期错误；
- 多个时域专家的 DEM 候选跨度超过稳健阈值：

```text
T_disagreement = max(0.1 m, median(spread) + 6*MAD(spread))
```

IGS 使用缠绕相位 RI-L1 目标搜索 DEM 和线性形变参数。若 RI-L1 明确改善，则采用 IGS 结果；否则把 IGS 作为相位域仲裁器，在与其 DEM 解一致的候选中优先选择复杂度最低的 `linear_huber`，再依次考虑 Adaptive HT、Adaptive Huber 和 ICA。

## 5. PGDC 稀疏门控

修改 Sobel 和 GDC 用于判断 DEM 误差的空间存在性。HOMA-DEM 不会在所有场景强制使用 PGDC 掩膜。只有显著 DEM 像元比例低于 `0.35` 时进入稀疏模式，并保留：

```text
active = PGDC_detected OR abs(h)/sigma_h >= 2.5
         OR dynamic_height OR IGS_selected
```

非活动像元的 DEM 校正量设为零，从而减少无 DEM 误差区域的过校正。

## 6. 不确定度自适应空间正则化

图正则化不再使用全场固定混合比例。先计算局部 DEM 标准差相对场景中位数的比值 `r`：

```text
blend = clip((r - 1.5)/3, 0, 0.35)
h_final = (1-blend)*h_raw + blend*h_graph
```

正常可信像元的 `blend=0`；只有不确定度明显偏高的像元最多接受 35% 图正则结果。经过 IGS 仲裁的像元不再平滑，避免破坏缠绕域确认结果。

## 7. MintPy 用法

```bash
python run_published_models_on_mintpy.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  --models hybrid_optimal_2026 \
  --dem-bound 200 \
  --velocity-bound 20 \
  --hybrid-graph-lambda 1 \
  --hybrid-dynamic-alpha 0.01 \
  --hybrid-min-height-change 2 \
  -o hybrid_dem_error_ifg \
  --overwrite
```

主要输出：

- `ifgramStack_demErr_hybrid_optimal_2026.h5`：校正后的 MintPy 栈；
- `demComponent_hybrid_optimal_2026.h5`：DEM 误差、DEM 相位以及诊断栅格；
- `diagnostics_hybrid_optimal_2026.json`：专家使用率、ICA/IGS 状态和场景判别摘要。

组件 HDF5 中包含：

| 数据集 | 含义 |
|---|---|
| `demError`, `demPhase` | 最终 DEM 误差和逐干涉图校正相位 |
| `demErrorBefore`, `heightChange`, `dynamicMask`, `changeIndex` | 动态高度估计 |
| `sourceModel` | 每个像元最终采用的专家编号 |
| `demErrorStd`, `demInformation`, `demObservability` | 不确定度与可观测性 |
| `gdc`, `pgdcDetected`, `activeMask` | PGDC 检测和稀疏门控 |
| `unwrapCycleFraction`, `candidateSpread` | 缠绕错误及专家分歧触发量 |
| `graphBlend` | 实际空间正则混合比例 |

`sourceModel` 编号应结合 JSON 中的 `source_names` 读取，因为 ICA 是否满足准入条件会改变候选数量。

## 8. 当前基准结论

固定随机种子 `20260713`、`18x18` 场景下：

| 场景 | HOMA-DEM RMSE | 最佳对照 |
|---|---:|---:|
| 静态 DEM + 非线性形变 | 约 `0.03 m` | Adaptive HT 约 `0.03 m` |
| 缠绕相位 + 局部解缠周期错误 | 约 `0.01 m` | Linear Huber 约 `0.01 m` |
| 稀疏 DEM 误差 | 小于 `0.2 m` | 当前图模型约 `1.0 m` |
| 观测期内高度突变 | 小于 `0.01 m` | Dynamic SBAS 约 `0.08 m` |

不同场景的绝对 RMSE 不能合并为一个总排名。HOMA-DEM 的优势是降低模型失配风险和最坏场景误差，而不是保证每个数据集都严格超过专用模型。

## 9. 可发表创新点

1. 将 DEM 误差检测、静态时域模型选择、动态高度变化和缠绕相位优化统一到一个分层决策框架。
2. 提出“残差周期错误 + 专家分歧”的双触发策略，使昂贵的缠绕域全局搜索只作用于疑难像元。
3. 使用 IGS 作为候选模型仲裁器，而不仅是直接替代估计器。
4. 使用 DEM 显著性与 PGDC 的联合稀疏门控，减少无误差区域过校正。
5. 将图正则化强度与估计不确定度绑定，避免固定正则参数导致地形细节过平滑。
6. 输出逐像元专家来源、动态模型检验量和空间正则比例，支持可解释性与消融实验。

## 10. 发表前仍需完成

- 使用至少两类真实地形、不同传感器和外部 LiDAR/ICESat-2 高程验证；
- 对 PGDC 阈值、动态 F 检验、专家分歧阈值和图正则参数做敏感性分析；
- 进行逐模块消融：去除 ICA、PGDC、IGS、动态分支和不确定度图正则；
- 报告运行时间、内存和触发 IGS 的像元比例；
- 对空间自相关残差使用分块交叉验证，不能只报告训练干涉图残差；
- 当前 IGS 局部优化器是 SciPy 等价映射，若论文声称复现 CMA-ES，应安装并直接使用作者依赖重新验证。
