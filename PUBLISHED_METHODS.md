# 已发表 DEM 误差方法复现与对比

本项目将六类已发表方法映射到统一的 MintPy 干涉图观测模型：

```text
phi_m(p) = c_m(p) * delta_h(p) + phi_deformation_m(p) + epsilon_m(p)
c_m(p) = -4*pi/lambda * Bperp_m(p) / (R(p) * sin(theta(p)))
```

代码实现位于 `published_dem_error_models.py`。实际 MintPy HDF5 入口是
`run_published_models_on_mintpy.py`，场景匹配基准是
`run_published_model_benchmark.py`。

## 复现等级

| 方法 | 入口名称 | 复现等级 | 说明 |
|---|---|---|---|
| 分形表面正则化，2015 | `fractal_2015_adapted` | adapted | 原文需要干涉复数幅度并联合更新幅度与相位；MintPy `ifgramStack` 通常没有该幅度，因此使用显式传入的幅度代理构造四次根距离向梯度先验。 |
| 非参数 DEM 误差估计，2019 | `ica_2019` | strict | 实现干涉网到相邻时段反演、SVHT、FastICA、最大基线相关分量、F 检验和比例恢复。要求连通网，并输出空间中心化的相对 DEM 误差。 |
| 自适应形变模型，2021 | `adaptive_ht_2021` | strict | 实现约一年分组、20% 重叠、总体 F 检验、参数 t 检验、相邻历元差分和重叠伪观测联合解算。 |
| IGS-CMAES，2021 | `igs_cmaes_2021_equivalent` | equivalent | 保留作者代码的 RI-L1 目标、多尺度 IGS 和多初值逻辑；本环境未安装 `cma`，局部 CMA-ES 映射为有界 Powell 加逐级局部网格。 |
| 相位梯度方向一致性，2025 | `pgdc_2025` | equivalent | 修改 Sobel、基线符号归一化和 GDC 检测按原式实现；Delaunay/KNN、相邻等时长干涉图组合、TPC 搜索及约束网平差按论文流程实现，组合配对和边界参考选择为工程等价映射。 |
| 改进 SBAS 动态高度估计，2025 | `dynamic_height_2025` | strict | 对候选突变历元分别拟合前后 DEM 误差与线性形变率，以残差平方和最小选择时刻，输出前后 DEM、变化量和变化日期。 |
| HOMA-DEM 分层混合专家，2026 | `hybrid_optimal_2026` | new hybrid | 以 Adaptive HT、Huber、ICA 为候选，使用动态 F 检验、PGDC 稀疏门控、IGS 缠绕仲裁和不确定度自适应图正则进行逐像元选择。 |

`strict` 表示论文方程可直接映射到当前观测域；`equivalent` 表示核心目标和流程一致，但优化器或工程步骤不同；`adapted` 表示论文所需输入在 MintPy 栈中缺失或原问题域不同。

## 方法原理

### 1. 分形表面正则化

Danudirdjo 与 Hirose 从分形表面小扰动散射模型得到：

```text
m = beta * (c + grad_range(phi))^4
```

并以 `TV[log(m) - 4*log(c + grad_range(phi))]` 作为先验，结合相位高斯似然和幅度 Gamma 似然交替优化。当前适配实现先由普通 DEM 反演得到初值，再通过幅度代理的四次根建立距离向梯度目标，解带数据保真和方向加权的稀疏系统，最后应用论文给出的方位向核 `[0.1, 0.8, 0.1]`。该输出只能用于方法思想对比，不能宣称为原文完整幅相联合 MAP 复现。

### 2. 非参数 ICA

先将多主影像干涉网反演为相邻时段相位 `X`，对中心化数据计算协方差特征值，并用
`tau = 2.858 * median(eigenvalue)` 决定初始主成分数。FastICA 得到混合矩阵 `A` 和空间源 `S` 后，选择与相邻时段垂直基线系数绝对相关性最大的混合向量。若其线性基线关系通过 `F(1,N-1)` 检验，则通过
`f = (b^T b)^-1 b^T a` 和 `delta_h = f*S` 恢复 DEM 误差；否则增加一个主成分后重试。

### 3. 自适应假设检验

每个时间组从完整候选函数开始：常数、线性、二次、三次、年周期正弦和余弦。先用总体 F 检验判断时间相关模型是否显著，再对非恒定参数做双侧 t 检验，默认显著性水平 `alpha=0.01`。随后使用相邻历元差分增强 DEM 项相对比例，并加入重叠区形变一致性伪观测，联合估计每组形变参数与一个静态 DEM 误差。

### 4. IGS-CMAES

直接使用缠绕相位，联合搜索线性形变参数和 DEM 误差。目标函数与作者代码一致：

```text
L = mean(w * (abs(sin(phi_obs)-sin(phi_pred))
            + abs(cos(phi_obs)-cos(phi_pred))))
```

IGS 由粗到细搜索候选初值，局部阶段继续最小化 RI-L1。相位周期性会产生别名解，因此实现保留作者代码的规则：损失四舍五入到四位后相同时，选择归一化参数范数较小的解。

### 5. PGDC

对短时间基线干涉图按垂直基线符号统一相位方向，使用逐邻点缠绕差构造修改 Sobel 梯度。像元的方向一致性为：

```text
GDC_k = abs(sum_m(w_m * exp(i*theta_mk))) / sum_m(w_m)
```

超过阈值的 `3x3` 窗口全部标为疑似 DEM 误差像元。估计阶段在稀疏子网弧段上组合相邻、近似等时长干涉图，通过最大化时间相位相干性搜索相对 DEM 误差，再以弧段 TPC 为权做网平差恢复像元 DEM 误差。该方法的主要优势是先检测后估计，从而减少无误差区过校正；它不保证在理想全场线性模型中优于逐像元最小二乘。

### 6. 动态高度改进 SBAS

对每个候选高度突变历元 `i=3,...,N-3`，排除跨越突变时刻的干涉图，并分别建立前后两组 `[DEM coefficient, temporal baseline]` 设计矩阵。选择残差平方和最小的候选：

```text
i_hat = argmin_i ||phi_before-A_before*x_before||^2
                    + ||phi_after-A_after*x_after||^2
height_change = dem_after - dem_before
```

它针对开挖、填筑和城市建设等观测期内真实表面高度变化。将其放在纯静态场景中排名没有明确意义。

## 在 MintPy 数据上运行

```bash
python run_published_models_on_mintpy.py \
  test_data/HFT473_16x16/ifgramStack.h5 \
  -g test_data/HFT473_16x16/geometryRadar.h5 \
  --mask test_data/HFT473_16x16/maskTempCoh.h5 \
  --models ica_2019 adaptive_ht_2021 pgdc_2025 \
  -o published_dem_error_ifg \
  --overwrite
```

IGS 直接使用 `wrapPhase`；若输入栈没有该数据集，则由 `unwrapPhase` 重缠绕。分形适配方法默认以平均相干性作为幅度代理，因此必须将其视为敏感性实验，而非原文严格复现。

每个方法输出：

- `ifgramStack_demErr_METHOD.h5`：校正后的 MintPy 栈；
- `demComponent_METHOD.h5`：二维 `demError` 和逐干涉图 `demPhase`；
- `diagnostics_METHOD.json`：方法特有统计量和复现等级。

动态高度方法无法由单一高度值校正跨越突变时刻的干涉图，因此这些观测保持原相位，数量记录在 `uncorrected_spanning_observations`。

## 场景匹配基准

```bash
python run_published_model_benchmark.py \
  -o published_model_benchmark \
  --size 18 \
  --seed 20260713 \
  --overwrite
```

基准包括：静态 DEM 加非线性形变、缠绕线性模型、稀疏 DEM 误差、观测期内高度突变、分形地形尖峰五类场景。只应在同一场景内部比较 RMSE，不能把不同场景的方法值合并成总排名。

输出包括：

- `published_method_metrics.csv`：RMSE、MAE、相关系数、运行时间和方法特有指标；
- `published_method_comparison.png/.pdf`：空间结果对比；
- `published_method_rmse.png/.pdf`：按场景分组的 RMSE；
- `published_benchmark_maps.npz`：真值与估计图；
- `benchmark_config.json`：随机种子和网格大小。

## 参考文献与本地来源

1. Danudirdjo, D., Hirose, A. InSAR Image Regularization and DEM Error Correction With Fractal Surface Scattering Model. IEEE TGRS, 2015. [DOI](https://doi.org/10.1109/TGRS.2014.2341254)
2. Liang et al. Nonparametric Estimation of DEM Error in Multitemporal InSAR. IEEE TGRS, 2019. [DOI](https://doi.org/10.1109/TGRS.2019.2930802)
3. Du et al. Adaptive Deformation Model Based on Hypothesis Testing. Remote Sensing, 2021. [DOI](https://doi.org/10.3390/rs13102006)
4. IGS-CMAES. Remote Sensing, 2021. [DOI](https://doi.org/10.3390/rs13132615), [作者代码](https://github.com/Lucklyric/TSInSAR-PF-IGS_CMAES)
5. Song et al. Phase Gradient Direction Consistency. Remote Sensing of Environment, 2025. [DOI](https://doi.org/10.1016/j.rse.2025.115028)
6. Li et al. Estimation of Surface Height Changes and Deformation Time Series With Improved SBAS-InSAR Technique. IEEE TGRS, 2025. [DOI](https://doi.org/10.1109/TGRS.2025.3615234)

对应全文位于仓库 `reference_paper/`。论文全文受原出版协议约束，代码和说明只记录实现所需公式与复现决策，不重新分发提取后的全文文本。

综合模型的完整定义见 [`HYBRID_OPTIMAL_2026.md`](HYBRID_OPTIMAL_2026.md)。
