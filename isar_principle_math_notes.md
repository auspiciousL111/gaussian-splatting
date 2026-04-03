# ISAR-3DGS 原理与数学记录

## 文档用途
本文档用于记录当前几何假设、参数物理意义、参数通路边界、数学原理和公式记录与推导。

## 维护约定
每次涉及几何接口或投影数学变更，都应先更新本文件，再执行实验验证。

---

## 1. 当前 3DGS 默认几何/投影假设

当前基础 3DGS 路径默认采用透视相机模型。

默认链条中的核心假设：
- 相机参数使用 FoVx/FoVy。
- renderer 从 FoV 推导 tanfovx/tanfovy。
- 使用透视 projection matrix。
- CUDA forward 路径使用齐次坐标投影与透视除法。
- 2D 协方差投影由透视 Jacobian 推导。

直接含义：
- ISAR/正交适配若不改数学，就必须先用兼容参数占位。
- 真正切换到正交需要 forward 数学分支明确启用。

---

## 2. A+B+C 分别打通了什么

### A 阶段打通内容
- 在 Python 数据/相机层新增四个投影相关字段：
  - projection_mode
  - ortho_scale_x
  - ortho_scale_y
  - isar_window_size
- reader 到 camera 的最小透传已建立。

### B 阶段打通内容
- renderer settings 可携带上述新字段。
- Python 侧 GaussianRasterizationSettings 结构已扩展。
- 在环境混合期保留兼容过滤，保证运行稳定。

### C 阶段打通内容
- Python binding -> C++ -> CUDA 调用边界签名已统一扩展。
- forward 调用边界可接收新参数。
- backward 调用边界也已对齐接收新参数。
- 仍未启用新的投影数学。

### D 阶段（forward-only）打通内容
- 在 `forward.cu` 中正式启用 `projection_mode` 分支。
- perspective 前向公式保持官方路径不变。
- orthographic/isar 前向路径开始消费：
  - `projection_mode`
  - `ortho_scale_x`
  - `ortho_scale_y`
  - `isar_window_size`（兜底）
- backward 仍未改动。

---

## 3. 新增字段的物理意义

- projection_mode
  - 用于指示投影模型类型。
  - 当前透传采用整数枚举：
    - 0：perspective
    - 1：orthographic/isar

- ortho_scale_x
  - 正交投影在 x 方向的尺度参数。
  - 目标是在正交分支中替代 FoV 派生的横向缩放。

- ortho_scale_y
  - 正交投影在 y 方向的尺度参数。
  - 目标是在正交分支中替代 FoV 派生的纵向缩放。

- isar_window_size
  - 保存 ISAR 数据中的窗口参数。
  - 当前作为元数据透传，后续可用于分支逻辑或标定约束。

---

## 4. 参数通路目前到达了哪里

目前已到达的边界：
- dataset reader 与 camera 对象层。
- renderer settings 组装层。
- Python 扩展调用参数打包层（前向与反向）。
- C++ bridge 入口与转发层。
- Rasterizer 前后向接口层。
- forward preprocess kernel 参数边界层。

目前尚未影响的数值行为：
- backward.cu 尚未引入正交对应梯度分支。

---

## 5. 为什么目前仍不是“完整 ISAR/正交训练”

当前已启用 forward 分支，但尚未完成完整训练语义切换，原因是分阶段策略：
- 已完成：前向分支启用与稳定性验证。
- 未完成：backward 梯度链与正交分支一致化。

因此当前状态是：
- 前向：支持 perspective 与 orthographic/isar 分支。
- 反向：仍是透视假设。
- 训练正确性：尚不能宣称完成 ISAR 正交训练闭环。

---

## 6. 下一轮进入 forward 数学前还需要注意什么

1. 进入 backward 前，先固定并冻结 forward 分支定义。
- 避免 forward/backward 同时漂移导致梯度定位困难。

2. 保留双路径可切换。
- projection_mode 必须保留 perspective 回归路径。

3. 先明确正交坐标约定。
- 轴方向、尺度单位、像素中心映射需先固定。

4. 继续使用最小验证闭环。
- Scene 冒烟。
- forward 渲染合理性检查。
- perspective 下 2-iteration 回归冒烟。
- orthographic/isar 下 forward-only 冒烟（有限性、非全黑、无崩溃）。

5. 临时补丁保持可追踪。
- train.py 中单通道扩 3 通道仅用于 smoke，不是最终建模结论。

6. 占位项需要后续逐步替换。
- ISAR reader 的 FoV 占位映射。
- reader 随机点云初始化。
- renderer 兼容过滤（环境固定后可收敛）。

---

## 7. 阶段 D（forward-only）后的可执行状态

已经就绪：
- 新投影参数可从数据侧一路传到 CUDA 调用边界。
- forward 中 projection_mode 分支已启用并通过冒烟验证。
- 重编译与 smoke 流程已验证可执行。

尚未完成：
- backward.cu 对应梯度链改造。

建议的直接下一步：
- 进入 backward.cu：按已固定的 forward 分支补齐正交/ISAR 梯度链。

---

## 8. 后续更新清单

每次更新本文件应明确写出：
- 哪些几何假设发生变化。
- 哪些公式发生变化。
- 参数通路从“可到达”变为“已消费”的边界位置。
- 哪些临时方案被替换。
- 做了哪些验证、验证证明了什么。

---

## 9. 阶段 D 已启用的前向数学公式（本轮补充）

本节给出当前代码已经对应到的前向公式，方便后续 backward 对齐。

### 9.1 坐标变换

世界坐标到相机坐标：

$$
\mathbf{X}_c = \mathbf{R}\mathbf{X}_w + \mathbf{t},\quad
\mathbf{X}_c = (x, y, z)^\top
$$

其中：
- $\mathbf{R}$ 为旋转矩阵。
- $\mathbf{t}$ 为平移向量。

### 9.2 两种投影模型

1. perspective（projection_mode = 0，保持官方路径）

$$
u = f_x \frac{x}{z} + c_x,\quad
v = f_y \frac{y}{z} + c_y
$$

对应局部 Jacobian：

$$
\mathbf{J}_{\text{persp}} =
\begin{bmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x x}{z^2} \\
0 & \frac{f_y}{z} & -\frac{f_y y}{z^2}
\end{bmatrix}
$$

2. orthographic/isar（projection_mode = 1，阶段 D 新启用）

定义线性尺度（含防零夹紧思想）：

$$
s_x \approx \frac{2}{\max(|\text{ortho\_scale\_x}|, \varepsilon)},\quad
s_y \approx \frac{2}{\max(|\text{ortho\_scale\_y}|, \varepsilon)}
$$

若缺少正交尺度，可用 `isar_window_size` 进行兜底近似。

正交投影可写为：

$$
u = s_x x + c_x,\quad
v = s_y y + c_y
$$

对应 Jacobian：

$$
\mathbf{J}_{\text{ortho}} =
\begin{bmatrix}
s_x & 0 & 0 \\
0 & s_y & 0
\end{bmatrix}
$$

### 9.3 3D 高斯到 2D 椭圆协方差

设 3D 协方差为 $\mathbf{\Sigma}_{3D}$，则屏幕平面协方差统一写法：

$$
\mathbf{\Sigma}_{2D} = \mathbf{J}\,\mathbf{\Sigma}_{3D}\,\mathbf{J}^\top + \lambda \mathbf{I}
$$

其中：
- perspective 分支取 $\mathbf{J}=\mathbf{J}_{\text{persp}}$。
- orthographic 分支取 $\mathbf{J}=\mathbf{J}_{\text{ortho}}$。
- $\lambda \mathbf{I}$ 表示数值稳定所需的小对角正则项。

### 9.4 前向渲染与 alpha 合成

2D 高斯权重（省略归一化常数）可写为：

$$
w_i(\mathbf{p}) = \exp\left(-\frac{1}{2}(\mathbf{p}-\boldsymbol{\mu}_i)^\top
\mathbf{\Sigma}_{2D,i}^{-1}(\mathbf{p}-\boldsymbol{\mu}_i)\right)
$$

像素颜色按前向深度序进行 alpha 累积：

$$
\mathbf{C} = \sum_i T_i\,\alpha_i\,\mathbf{c}_i,\quad
T_i = \prod_{j<i}(1-\alpha_j)
$$

---

## 10. 下一阶段 backward 需要对齐的数学框架

当前状态：forward 已支持双分支，backward 仍是透视假设。

下一阶段需按链式法则补齐正交分支梯度。

### 10.1 均值路径梯度链

以单个高斯中心为例：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{X}_w}
=
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\mu}_{2D}}
\frac{\partial \boldsymbol{\mu}_{2D}}{\partial \mathbf{X}_c}
\frac{\partial \mathbf{X}_c}{\partial \mathbf{X}_w}
$$

其中：

$$
\frac{\partial \mathbf{X}_c}{\partial \mathbf{X}_w}=\mathbf{R}
$$

并且：
- perspective 分支：$\frac{\partial \boldsymbol{\mu}_{2D}}{\partial \mathbf{X}_c}=\mathbf{J}_{\text{persp}}$。
- orthographic 分支：$\frac{\partial \boldsymbol{\mu}_{2D}}{\partial \mathbf{X}_c}=\mathbf{J}_{\text{ortho}}$。

### 10.2 协方差路径梯度链

由

$$
\mathbf{\Sigma}_{2D} = \mathbf{J}\,\mathbf{\Sigma}_{3D}\,\mathbf{J}^\top
$$

可得关键链路：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{\Sigma}_{3D}},\quad
\frac{\partial \mathcal{L}}{\partial \mathbf{J}}
$$

当投影分支不同，$\mathbf{J}$ 的表达不同，需在 backward 中使用与 forward 同一分支的 Jacobian 定义，避免梯度与前向不一致。

### 10.3 参数梯度

对于 orthographic 分支，后续若允许学习或优化尺度，需要明确：

$$
\frac{\partial \mathcal{L}}{\partial s_x},\quad
\frac{\partial \mathcal{L}}{\partial s_y}
$$

若继续使用

$$
s_x = \frac{2}{|\text{ortho\_scale\_x}|},\quad
s_y = \frac{2}{|\text{ortho\_scale\_y}|}
$$

则还会引入绝对值与夹紧点的分段导数，需要在实现上处理不可导点的次梯度或平滑替代。

---

## 13. 阶段 H：训练稳定性与基础投影对照记录

本节只记录实验观察，不引入新的数学改动。

### 13.1 递进训练计划与稳定性

在同一 ISAR 数据集上执行递进训练：100 -> 500 -> 1000 iter。

- 数据：`D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
- 模型：`./output/stage_h_ortho_prog`
- 方式：
  - 100 iter 从零开始。
  - 500 iter 从 `chkpnt100.pth` 续跑。
  - 1000 iter 从 `chkpnt500.pth` 续跑。

结果摘要：
- 三段训练均完成，无 NaN/Inf。
- 训练评估点从 ITER 100 到 ITER 1000，L1 由 `0.104053` 下降到 `0.099753`，PSNR 由 `12.2542` 上升到 `12.4792`。
- 训练后渲染健康检查（100/500/1000）均为 PASS，`finite_ok=True`。

### 13.2 同模型同相机投影对照（perspective vs isar）

在同一训练模型、同一相机下，仅切换 `projection_mode`：

- `perspective`
- `isar`

并输出：`render_perspective.png`、`render_isar.png`、`render_absdiff.png`、`stats.json`。

主要观察：
- `perspective` 输出更接近近全白分布（`near_white_ratio` 约接近 1）。
- `isar` 输出具有更明显的亮度结构变化（`std` 更高、`near_white_ratio` 更低）。
- 在迭代 1000 时，两模式差异的均值绝对差约为 `0.0024`，最大差约为 `0.3425`。

### 13.3 本节结论

Stage H 结果支持以下工程判断：
- 当前分支可在更长一点训练下稳定运行。
- 在同一 ISAR 数据上，`perspective` 与 `orthographic/isar` 的渲染统计与图像分布存在稳定、可复核的差异。
- 本轮目标聚焦“稳定性 + 基础对照”已达成，且未引入新的模型功能或数学变更。

---

## 11. 当前文档结论（更新）

- 阶段 D 后已具备“可运行的前向双投影数学”。
- 现阶段缺口已明确为“反向链与分支 Jacobian 对齐”。
- 后续任何 backward 改动都应以第 9 节公式为前向真值来源进行一致化实现与验证。

---

## 12. 阶段 E 已完成的 backward 对齐数学（本轮新增）

本节记录“已经实现到代码”的最小 backward 对齐，不是最终 ISAR 全量数学。

### 12.1 均值梯度（2D mean 路径）

1. perspective 分支（保持原链路）

$$
\mathbf{m}_{clip} = \mathbf{P}[\mathbf{X}_w, 1]^\top,\quad
\mathbf{m}_{ndc} = \left(\frac{m_x}{m_w}, \frac{m_y}{m_w}\right)
$$

反向时使用透视除法链式求导，当前实现与原始 3DGS 一致。

2. orthographic/isar 分支（本轮补齐）

$$
\mathbf{X}_c = \mathbf{R}\mathbf{X}_w + \mathbf{t}
$$

$$
x_{ndc} = \operatorname{clamp}(s_x x_c),\quad
y_{ndc} = \operatorname{clamp}(s_y y_c)
$$

其中

$$
s_x = \frac{2}{\max(|\text{ortho\_scale\_x}|, \varepsilon)},\quad
s_y = \frac{2}{\max(|\text{ortho\_scale\_y}|, \varepsilon)}
$$

反向采用分段导数：

$$
\frac{\partial x_{ndc}}{\partial x_c} =
\begin{cases}
s_x, & |s_x x_c| \le 1.3 \\
0, & \text{otherwise}
\end{cases},\quad
\frac{\partial y_{ndc}}{\partial y_c} =
\begin{cases}
s_y, & |s_y y_c| \le 1.3 \\
0, & \text{otherwise}
\end{cases}
$$

再由

$$
\frac{\partial \mathbf{X}_c}{\partial \mathbf{X}_w} = \mathbf{R}
$$

把梯度从 view 坐标传回 world 坐标。

### 12.2 协方差梯度（2D conic/cov 路径）

前向统一形式：

$$
\mathbf{\Sigma}_{2D} = \mathbf{J}\mathbf{\Sigma}_{3D}\mathbf{J}^\top
$$

反向仍按

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{\Sigma}_{3D}},\quad
\frac{\partial \mathcal{L}}{\partial \mathbf{J}}
$$

进行链式传播。

- perspective：$\mathbf{J}_{\text{persp}}$ 随 $(x_c,y_c,z_c)$ 变化，因此协方差路径对均值有几何梯度贡献。
- orthographic：$\mathbf{J}_{\text{ortho}}$ 为常数矩阵（由尺度给定），因此该路径对均值的几何梯度为 0（不含深度项）。

### 12.3 inverse-depth 项（两分支共享）

当前渲染中 inverse-depth 仍按

$$
d = \frac{1}{z_c}
$$

因此梯度项保持：

$$
\frac{\partial \mathcal{L}}{\partial z_c}
\mathrel{+}= -\frac{\partial \mathcal{L}}{\partial d}\frac{1}{z_c^2}
$$

该项在 perspective 与 orthographic 分支中都保留。

---

## 13. 阶段 E 后的状态结论

- forward 与 backward 已在 projection_mode 维度完成最小一致化。
- 本轮只覆盖“投影直接相关梯度链”，符合小阶段目标。
- 仍未完成全部 ISAR 数学细化（尺度物理标定、更多建模项），但已可作为下一 baseline 节点。

---

## 14. 阶段 F 最小梯度数值对照验证（本轮新增）

本节只记录“验证方法与结果”，不引入新功能。

### 14.1 对照原则

对每个待验证参数分量，用中心差分近似数值梯度：

$$
g_{num}(\theta_i) = \frac{\mathcal{L}(\theta_i + \varepsilon) - \mathcal{L}(\theta_i - \varepsilon)}{2\varepsilon}
$$

并与解析梯度（autograd/backward）对比：

$$
\Delta_{abs} = |g_{ana} - g_{num}|,
\quad
\Delta_{rel} = \frac{|g_{ana} - g_{num}|}{\max(|g_{ana}|, |g_{num}|, 10^{-8})}
$$

### 14.2 最小验证配置

- Gaussian 数量：1
- 图像尺寸：$9\times 9$
- 损失：中心 $3\times 3$ patch 像素和
- 验证参数子集：
  - 均值链：$\partial \mathcal{L} / \partial (x,y,z)$
  - 协方差链：$\partial \mathcal{L} / \partial (\Sigma_{xx}, \Sigma_{xy}, \Sigma_{yy})$

### 14.3 覆盖案例

1. perspective_minimal
- 用于验证透视分支下 mean/cov 投影链的一致性。

2. orthographic_minimal
- 用于验证正交分支下 mean/cov 投影链的一致性。

3. orthographic_clamp_x
- 设置 $x$ 落在 clamp 饱和区，验证分段导数行为（x 方向梯度应接近 0）。

### 14.4 本轮结果

- 总结：`PASS`

---

## 15. 阶段 P2：最小监督修正（单通道强度语义）

本节只涉及训练监督入口，不涉及 forward/backward 数学改动。

### 15.1 当前最小实现

当满足以下条件时：
- 数据集识别为 ISAR（`poses.csv + images`）
- GT 为单通道 `1xHxW`

训练损失入口改为单通道分支：

$$
I_{gray} = \frac{1}{3}(I_R + I_G + I_B)
$$

并使用：

$$
\mathcal{L} = (1-\lambda)\,\|I_{gray}-I_{gt}\|_1 + \lambda\,(1-\mathrm{SSIM}(I_{gray}, I_{gt}))
$$

其中 $I_{gt}$ 为 `1xHxW` GT 强度图。

### 15.2 为什么选“通道均值”作为 render_gray

- 最小侵入：不改 renderer 表示，不新增网络头。
- 对称中性：当前三通道并无真实颜色语义，均值避免偏向某单一通道。
- 可回退：非 ISAR 或非 1ch GT 时仍走原 RGB 兼容路径，降低回归风险。

### 15.3 本阶段效果判断

在 2-iter 与 100-iter 验证中，训练稳定性正常；
但 100-iter 与旧监督同迭代基线相比，`l1_vs_gt`、`psnr_vs_gt`、强度分位数几乎不变。

工程结论：P2 完成了“监督语义对齐入口”，但单独这一改动尚不足以显著改善当前“能量偏低、结构偏弱”现象。
- perspective_minimal：
  - max_abs_err = $8.030\times 10^{-4}$
  - max_rel_err = $6.333\times 10^{-3}$
- orthographic_minimal：
  - max_abs_err = $3.152\times 10^{-4}$
  - max_rel_err = $1.439\times 10^{-4}$
- orthographic_clamp_x：
  - 解析梯度与数值梯度均为 0（在该构造样本下）
  - clamp 分段导数行为与实现一致

### 14.5 当前仍可疑/仍需继续观察的点

- clamp 阈值附近（$|s_x x| \approx 1.3$）的数值平滑性尚未专门测。
- 当前只验证单高斯与小视野，不包含多高斯重叠/遮挡导致的复杂梯度耦合。
- 当前损失仅为局部像素和，尚未覆盖更复杂 loss 组合（如深度项混合权重变化）。

---

## 15. 阶段 F 后状态结论

- 在不修改 forward/backward 数学的前提下，已完成最小梯度数值一致性验证。
- 投影直接相关梯度链在当前最小案例中与解析梯度一致性良好。
- 该结果可作为 Stage E baseline 之后进入下一验证轮次的依据。

---

## 16. 阶段 G 验证增强（本轮新增）

本节继续只做验证增强，不改 forward/backward 数学。

### 16.1 clamp 边界附近扫描

目标：检查 orthographic 分支在

$$
|s_x x| \approx 1.3,\quad |s_y y| \approx 1.3
$$

附近的分段导数行为。

方法：
- 固定单高斯与固定协方差，扫描阈值两侧样本。
- 对比

$$
g_{ana} \text{ vs } g_{num}
$$

并统计误差。

结果：PASS（当前采样点下梯度一致，且 clamp 外侧梯度近 0）。

### 16.2 多高斯小场景梯度检查

目标：在轻微重叠/遮挡条件下验证投影相关梯度。

方法：
- 构造 3 高斯小场景。
- 对比均值与协方差投影子集梯度（numeric vs analytic）。
- 分别测试 perspective 与 orthographic。

结果：
- orthographic：PASS（在设定阈值内）。
- perspective：FAIL（出现显著不一致）。

### 16.3 20 iter 短训练稳定性 smoke（orthographic/isar）

目标：验证稍长短训练下是否出现崩溃、NaN、明显图像退化。

方法：
- 训练 20 iter。
- 加载第 20 iter 模型渲染一帧，检查 finite 与图像统计。

结果：PASS。

渲染统计示例：

$$
  ext{finite\_ok}=\text{True},\quad
\min=0.654040,\ \max=1.000000,\ \text{mean}=0.996610,\ \text{std}=0.020717
$$

并且：

$$
  ext{nonzero\_ratio}=1.000000,\quad \text{near\_white\_ratio}=0.957237
$$

---

## 17. 阶段 G 后结论与可疑点

- orthographic/isar 分支在“边界扫描 + 多高斯 + 20iter smoke”下保持可运行与可验证。
- 当前主要可疑点集中在 perspective 多高斯梯度数值对照不一致。
- 下一步应继续仅做验证分解，不应立即扩大功能改动。

---

## 18. 阶段 G.1 分解定位结果（本轮新增）

本节只做验证分解，不修改数学实现。

### 18.1 逐步简化对照结果（perspective）

按以下顺序执行：
1. 多高斯原失败风格（3 高斯）
2. 拉大间距、减弱重叠
3. 减小深度差
4. 降为 2 高斯
5. 单高斯对照

结果：上述 5 组在一致判据下均未出现解析梯度异常标签（全部 `ok`）。

### 18.2 $\varepsilon$ sweep 结果

统一测试：

$$
\varepsilon \in \{10^{-2}, 5\times10^{-3}, 10^{-3}, 5\times10^{-4}, 10^{-4}\}
$$

观察到数值梯度随 $\varepsilon$ 变化总体平稳，无大规模符号翻转或发散特征。

### 18.3 分量级定位

按分量统计 fail_ratio：

- `mean_x`: 0.000
- `mean_y`: 0.000
- `cov_xx`: 0.000
- `cov_xy`: 0.000
- `cov_yy`: 0.000

即没有“某一类分量先失稳”的证据。

### 18.4 关键定位结论

此前 Stage G 的 perspective 多高斯 FAIL 主要由验证脚本对照条件不一致触发：

- analytic 使用了 patch=5 的损失；
- numeric 误用默认 patch=3 的损失。

两者比较对象不同，导致误报“梯度不一致”。修复后，perspective 多高斯对照恢复正常。

### 18.5 判定：非平滑还是解析问题

- 非平滑导致“数值梯度不适合作为严格判据”：本轮未观察到主导证据。
- 解析梯度真实问题：本轮未观察到直接证据。

在当前 Stage G.1 范围内，perspective 分支可恢复工程验证层面的基本信心。

---

## 19. 阶段 P2 长训练复核（500/1000）

本节只记录监督入口修正（P2）在更长训练下的实证结果，不涉及 forward/backward 数学改动。

### 19.1 固定评估口径

- 同一数据路径：`D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
- 同一相机：`camera_index=0`
- 同一背景：ISAR 黑底约定
- 主判据：`modes_intensity.isar`

### 19.2 单通道主指标结果

iter 500（P2 vs baseline）：
- `l1` 基本持平（微弱改善）
- `psnr` 基本持平（微弱下降）
- `q95/q99/contrast` 小幅下降

iter 1000（P2 vs baseline）：
- `l1` 下降（更好）
- `psnr` 上升（更好）
- `q95/q99` 上升
- `contrast=q99-q50` 上升

可见，P2 的收益在 1000 iter 才开始显现，且主要体现在高分位能量与对比度。

### 19.3 视觉复核结论

- 500 iter：P2 与 baseline 的主结构几乎无可感差异。
- 1000 iter：P2 相比 baseline 出现更活跃亮散射与更强局部对比。
- 但整体仍偏稀疏散点形态，与 GT 的结构分布差距仍明显。

### 19.4 阶段判定

- P2 不是无效改动：在更长训练下有可测提升。
- 但该提升尚不足以跨越“结构恢复”门槛。
- 下一步建议进入 P2.1（仅监督/loss 侧增强），继续保持“不改 forward/backward 数学”的约束。

---

## 20. 阶段 P2.1 监督侧结论（A/B 最小轮次）

本节只记录监督入口实验结论，不涉及 forward/backward 数学改动。

### 20.1 候选 A 小调参结论

候选 A 形式保持不变：

$$
w = 1 + \alpha I_{gt}^{\gamma},\quad
L_{wL1}=\frac{\sum w|I_{pred}-I_{gt}|}{\sum w}
$$

在已测组合中，`alpha=1.0, gamma=2.0` 在“高分位增强”与“全局 L1 恶化”之间取得最好平衡；
因此当前监督最优版本固定为 `P2.1A(alpha=1.0, gamma=2.0)`。

### 20.2 候选 B 最小验证结论

候选 B 形式：

$$
L = \lambda_1 |I_{pred}-I_{gt}| + \lambda_2 \left|\log(1+\beta I_{pred})-\log(1+\beta I_{gt})\right|
$$

最小接入默认参数：`lambda1=1.0, lambda2=0.2, beta=10.0`。

结论：
- 2-iter smoke 通过，说明候选 B 入口可执行。
- 100 iter 与 P2、P2.1A(a=1.0,g=2.0) 同口径对照中，候选 B 未显示明确优势。
- 因此不继续推进候选 B 的 500/1000，也不扩更多 loss 变体。

### 20.3 阶段切换

在当前节点，监督侧扩展暂时冻结；下一阶段切换为 CUDA 数学公式审查（先审查公式与实现一致性，再决定是否改代码）。

---

## 21. CUDA 数学主线第 1 轮：orthographic Jacobian 像素尺度一致化

本节只记录 forward/backward 投影链路公式对齐，不涉及监督/loss 与数据管线改动。

### 21.1 目标公式

正交前向（当前 surrogate）保持：

$$
x_{ndc}=\operatorname{clamp}\left(\frac{2x_c}{S_x},-1.3,1.3\right),\quad
y_{ndc}=\operatorname{clamp}\left(\frac{2y_c}{S_y},-1.3,1.3\right)
$$

像素映射：

$$
u=\frac{W}{2}(x_{ndc}+1)-\frac{1}{2},\quad
v=\frac{H}{2}(y_{ndc}+1)-\frac{1}{2}
$$

因此 Jacobian 采用像素尺度：

$$
J_{ortho,pix}=
\begin{bmatrix}
W/S_x & 0 & 0 \\
0 & H/S_y & 0 \\
0 & 0 & 0
\end{bmatrix}
$$

### 21.2 代码同步边界

- `forward.cu::computeCov2D(...)`：orthographic Jacobian 改为像素尺度。
- `backward.cu::computeCov2DCUDA(...)`：同口径改为像素尺度 Jacobian。
- `backward.cu::preprocessCUDA(...)`：mean backward 按像素 Jacobian 与 NDC->pixel 映射显式一致化。
- `auxiliary.h::in_frustum(...)`：新增 projection-aware orthographic/isar 分支。

### 21.3 约束确认

- perspective 数学路径保持原样（未改公式）。
- 本轮仍是 orthographic/isar surrogate，不等同于真实 ISAR 物理成像模型。

### 21.4 本轮最小验证结果（真实执行记录）

1) CUDA 扩展重编译
- 已成功重新编译。

2) `stage_f_gradient_check.py`
- `perspective_minimal`：PASS。
- `orthographic_minimal`：PASS。
- `orthographic_clamp_x`：脚本汇总为 FAIL，但该样本为边界 case（`radius <= 0`，未渲染），analytic 与 numeric 梯度均为 0。
- 解释：该项反映的是边界可见性/采样覆盖行为，不应直接归类为主链 Jacobian 数学错误。

3) 20 iter orthographic/isar smoke
- 训练成功跑完。
- 未出现 NaN / Inf / 崩溃。
- post-train 检查 `finite_ok=True`。
- 结论类别：PASS，未见明显 stability / visibility 异常。

### 21.5 本轮判定

- 在“重编译 + Stage F + 20iter smoke”最小验证口径下，CUDA 数学主线第 1 轮总体判定为 PASS。
- `orthographic_clamp_x` 仅作为边界样本记录，结论上不构成主链 mean/covariance 数学失败证据。

### 21.6 math-1 后 500 iter 同口径复测

为判断 math-1 是否对当前最优监督版本产生真实收益，追加执行与历史 `P2.1A(alpha=1.0, gamma=2.0)` 一致的 500 iter 复测：

- 监督口径固定为 candidate-A：
  - `--isar_supervision_mode a`
  - `--isar_l1_weight_alpha 1.0`
  - `--isar_l1_weight_gamma 2.0`
- compare 口径固定为：
  - 同一数据路径
  - 同一 `camera_index=0`
  - 同一黑底约定

对比对象均取 `modes_intensity.isar`：

- 历史 P2.1A@500：
  - `l1_vs_gt=0.1064154`
  - `psnr_vs_gt=12.0270`
  - `q50=0.0000`
  - `q95=0.04117`
  - `q99=0.31792`
  - `contrast = q99 - q50 = 0.31792`

- math-1 后 @500：
  - `l1_vs_gt=0.0790435`
  - `psnr_vs_gt=14.2756`
  - `q50=0.0000`
  - `q95=0.64650`
  - `q99=0.76506`
  - `contrast = q99 - q50 = 0.76506`

### 21.7 复测判读

- 从误差与重建质量指标看，math-1 后 500 iter 相比历史 P2.1A@500 有显著提升：
  - $L_1$ 明显下降；
  - PSNR 明显上升；
  - 高分位响应（$q95, q99$）显著抬升。
- 这说明 orthographic Jacobian 像素尺度一致化与 projection-aware visibility 修正不是“仅数学上自洽”，而是已经对训练结果产生了实质正效应。
- 但从响应分布看，当前结果仍偏“亮斑团/散射团块”形态，而非已经形成紧致、清晰的目标结构；换言之，math-1 有收益，但还不足以单独解决更深层的 ISAR 表示与观测模型问题。

---

## 22. math2 长训练验证（500 / 1000）

本节记录当前 math2 代码状态下的长训练实证结果。注意：本轮不再新增源码改动，只验证现有 math2 状态是否比 `math-1 + candidate-A` 更进一步。

### 22.1 固定实验口径

- supervision 固定为 candidate-A：
  - `--isar_supervision_mode a`
  - `--isar_l1_weight_alpha 1.0`
  - `--isar_l1_weight_gamma 2.0`
- 数据与 compare 口径固定为：
  - `D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
  - `camera_split=train`
  - `camera_index=0`
  - 黑底约定：`isar_source_forces_black`

### 22.2 真实执行结果

#### math2 @500

- post-train render check：PASS
- `finite_ok=True`
- `modes_intensity.isar`：
  - `l1_vs_gt=0.0830301`
  - `psnr_vs_gt=14.0252`
  - `q50=0.0000`
  - `q95=0.65106`
  - `q99=0.76771`
  - `contrast = q99 - q50 = 0.76771`

#### math2 @1000

- post-train render check：PASS
- `finite_ok=True`
- `modes_intensity.isar`：
  - `l1_vs_gt=0.0654826`
  - `psnr_vs_gt=15.2993`
  - `q50=0.0000`
  - `q95=0.59736`
  - `q99=0.78504`
  - `contrast = q99 - q50 = 0.78504`

### 22.3 与 math-1 baseline 的直接对比

对比对象仍取当前主 baseline：

- `math-1 + candidate-A@500`
  - `l1_vs_gt=0.0790435`
  - `psnr_vs_gt=14.2756`
  - `q50=0.0000`
  - `q95=0.64650`
  - `q99=0.76506`
  - `contrast=0.76506`

- `math-1 + candidate-A@1000`
  - `l1_vs_gt=0.0631026`
  - `psnr_vs_gt=15.4842`
  - `q50=0.0000`
  - `q95=0.58482`
  - `q99=0.78731`
  - `contrast=0.78731`

对比判读：

- 500 iter：
  - math2 的 $q95/q99$ 略高；
  - 但 $L_1$ 更差、PSNR 更低；
  - 因而不能判定为优于 math-1。

- 1000 iter：
  - math2 的 $q95$ 略高；
  - 但 $L_1$ 更差、PSNR 更低，且 $q99/contrast$ 也未占优；
  - 因而同样不能判定为优于 math-1。

### 22.4 视觉复核结论

- math2 在 500 / 1000 iter 下都仍主要表现为亮而厚的散射/白点团块。
- 相比 math-1，响应区域并未更集中，反而更有“铺宽”倾向：
  - 500 iter：`nonzero_ratio` 从 `0.2896` 升到 `0.2926`
  - 1000 iter：`nonzero_ratio` 从 `0.2929` 升到 `0.3053`
- 这意味着 math2 当前带来的不是“结构收紧”，而更像“亮响应继续变厚”。

### 22.5 本轮最终判断

- 从长训练结果看，math2 整体应归类为“更差”而不是“更好”或“持平”。
- 它没有继续扩大 math-1 已取得的有效收益，反而暴露出几何数学继续推进的边际收益已经很弱。
- 因此下一步不建议继续把主线押在 math，建议转向更干净的单通道 renderer，再讨论更真实的 ISAR 观测算子。

### 22.6 主线回退说明

在完成 500 / 1000 iter 长训练验证后，math2 未显示出相对 `math-1 + candidate-A` 的稳定优势，因此主线已回退到 math-1。

回退原则：

- 只撤销 math2 额外引入的 coarse visibility 接口透传；
- 不改动 `forward.cu / backward.cu / auxiliary.h` 中已确认有效的 math-1 数学主链；
- 不改动 candidate-A 的监督入口。

回退后追加执行 500 iter 回归复测，得到：

- `l1_vs_gt=0.0806888`
- `psnr_vs_gt=14.2163`
- `q50=0.0000`
- `q95=0.64172`
- `q99=0.77028`
- `contrast=0.77028`

与既有 `math-1 + candidate-A@500`：

- `l1_vs_gt=0.0790435`
- `psnr_vs_gt=14.2756`
- `q95=0.64650`
- `q99=0.76506`

相比仅有轻微数值波动，整体仍处于同一性能区间，因此可以判定当前代码状态已经回到 `math-1 + candidate-A` 主线，而不是残留在 math2。

---

## 23. 单通道 renderer 最小闭环验证说明（R2-min）

本节仅记录已真实执行过的最小闭环验证结论，不引入新实验与新数学改动。

### 23.1 当前单通道 renderer 的实际语义

- 在当前运行时二进制与源码一致后，rasterizer 的有效输出已回到单通道强度语义。
- 上层训练与检查路径按单通道强度进行主判读：
  - 2 iter 训练 smoke 可执行；
  - Stage G 的核心检查量基于单通道 intensity；
  - Stage H 的 `modes_intensity` 统计也是单通道。

当前固定监督口径保持为 candidate-A：

$$
w = 1 + \alpha I_{gt}^{\gamma},\quad (\alpha,\gamma)=(1.0,2.0)
$$

对应参数开关为：
- `--isar_supervision_mode a`
- `--isar_l1_weight_alpha 1.0`
- `--isar_l1_weight_gamma 2.0`

### 23.2 这轮验证已经说明了什么

- 通过重建 `diff_gaussian_rasterization` 扩展并完成一致性复核，render shape/channel 已恢复到单通道主语义。
- 在 `math-1 + candidate-A` 下，R2-min 的最小链路已闭环通过：
  - forward/render shape 复核通过；
  - 2 iter 训练 smoke 通过；
  - Stage G 通过；
  - Stage H 通过。
- 该结果足以支持“当前不回退主线”的工程决策。

### 23.3 这轮验证尚未说明什么

- 尚不能据此证明“长程训练稳定 baseline”已经成立。
- 尚不能据此证明 exposure 分支已经彻底收敛。
  - 观察事实：本轮 exposure 未出现爆炸。
  - 但 `exposure_optimizer.step()` 仍在训练循环中，需后续独立审查其必要性与收敛行为。
- 尚不能据此声称所有兼容问题均已完全收尾。

### 23.4 Stage H 中 intensity 与 raw 分支的区别

- `modes_intensity`：用于单通道强度语义的统计与对照，当前结果为 `shape=(1,1200,1200)`，是本轮主判据。
- `modes`（raw）：保留渲染可视化兼容输出，当前可见 `shape=(3,1200,1200)`。

解释：上述“intensity=1 通道、raw=3 通道”是当前兼容形态，不应在本轮被记为失败信号。

---

## 24. R2-min@500 同口径判定（相对 math-1 baseline）

本节只记录已执行完成的 500 iter 同口径对比结果，不引入新实现与新实验设计。

### 24.1 对比对象与主判据

- baseline：`math-1 + candidate-A@500`
  - `output/stage_m1_retest_p21a_compare_500_train0/stats.json`
- 当前：`R2-min@500`
  - `output/r2min_cons500_compare_20260331/stats.json`
- 主判据统一取：`modes_intensity.isar`

### 24.2 500 iter 对比结果说明了什么

本次结果呈现“分布尾部略增强，但整体误差口径不占优”的结构：

- 误差/重建主指标：
  - `l1_vs_gt` 变大（更差）
  - `psnr_vs_gt` 变小（更差）
- 分布与强响应指标：
  - `q95/q99` 略升
  - `contrast` 略升
  - `nonzero_ratio` 略升

其中对比使用：

$$
	ext{contrast} = q99 - q50
$$

### 24.3 为什么不能判为“真正胜出”

- 若 `q99/contrast` 略升，但 `L1/PSNR` 同时下降，则只能说明响应分布更强或更宽，不足以证明整体重建质量提升。
- 本轮还观察到 `nonzero_ratio` 上升，和“响应覆盖变宽”一致；该现象更接近分布形态变化，而非主指标改进。

因此当前更合理的解释是：
- R2-min 在单通道语义与工程链路上更干净；
- 但在 500 iter 的核心误差口径下暂未优于主 baseline；
- 其定位仍应是“候选实现/分布变化”，不是“更优主线”。

### 24.4 当前主线结论

截至本轮，主线仍应保持 `math-1 + candidate-A`，R2-min 暂不替代 baseline。

## 25. 显式标量散射独立通道的理论动机

### 25.1 为什么要将单通道拆解到表示层
过往实现里，将 ISAR 视作“光度渲染（RGB -> Grayscale）的特例”，这让模型优化必须通过 `SH2RGB` 仿射变换并承受多通道梯度的冗余耦合。真正物理对齐的方式，应该是将散射率定义为单一参数并在渲染时直通物理映射层，消除通道互串。

## 26. 显式标量散射通道的最小激活映射改造

### 26.1 `softplus` 平滑激活的意义
- 历史代码通过 `clamp_min` 防守非正域，存在梯度硬死区（截断会导致梯度丢失从而参数失活）。
- `softplus` 提供了一个从对数空间到线性物理强度的平滑正域激活。

## 27. 去光学化初始化与显式标量映射

### 27.1 光学球谐假设与物理正域映射的矛盾与解耦
原来基于 `RGB2SH` 的逻辑在初始化时认为颜色是一个光度概念。我们在改动了渲染公式后，若渲染链路是 $O_{scatter} = \operatorname{softplus}(x)$，那么参数 $x$ 的初始值如果不经对齐而盲目截断或者沿用 SH 参数，会导致大量参数在网络启动时就处于不利收敛甚至是极其紧绷的梯度空间中。

### 27.2 物理级逆隐层表出
为了保证物理对齐，我们将所有点云在第 0 次 iteration 的强度统一设定为逆向推导的初始分布：
$$ x_{init} = \ln(e^{S_{intensity}} - 1) $$
这个分布配合渲染链路上的 $\operatorname{softplus}$ 映射构成了恒等映射（在强度极小时引入 `1e-4` 的截断避免溢出）：
$$ \operatorname{softplus}(\ln(e^S - 1)) = \ln(1 + e^{\ln(e^S - 1)}) = \ln(1 + e^S - 1) = S $$
这一推导证明了从 `scene` 的初始构建，到 `model` 表示计算，再到 `renderer` 的正向推演，完成了端到端的逻辑闭环，显著弥合了激活引入初期的精度损失。

## 28. 显式物理表示层在长程监督中的收敛极限

### 28.1 1000-iter 全局比对的启示
在阶段 21 对“去光学化 + `softplus` 正域映射”进行的 1000 iterations 压力测试中证实，模型表现出了极高的一致性并彻底排除了训练崩溃问题。但面对基于 `clamp_min` 这种粗暴且不具备良好梯度行为的主线基线，更完善的纯物理表示分支在绝对 L1 与 PSNR 误差上表现旗鼓相当，却依旧微弱落后（L1 相差 $\sim 0.0006$）。

### 28.2 为什么更完善的物理模型没有形成参数压制？
- **观测模型代差**：物理表示被纯化为单通道物理极化数值后，损失函数或观测生成层面仍缺乏雷达特有的物理观测算子（如基于雷达带宽的包络、相位衰落）。
- **优化天花板验证**：这证明 `Representation Layer`（表示层）的内部表达维度（单通道连续正则激活）已达到了它能贡献的最优物理极值上限，逼近了现有的 L1/SSIM（光学视觉遗留）所能监督的解析边界。
- 当前结论指明了下一个理论破局点：我们必须跨过“表示层构建”，朝向 `Observation/Operator Layer`（观测算子层）甚至 `Loss`（损失函数层）的非相干/相干映射改进，才能实质性打破现有的 `15.4dB` 瓶颈。

## 29. 观测层初步：最小观测算子与双域对抗

### 29.1 L1 对数化重构的陷阱
在本实验中，引入了 log1p(I) 将具有重尾、高动态特性的雷达散射信号包络进行了显著压制。
原设想：由软化分布消减强光点带来的梯度爆炸与训练初期主导性方差。
实际结果：3DGS 与传统 MLP 截然不同。\L1 + log1p\ 的直接使用不仅抹除了收敛的高亮误差边界，同时也直接拔除了 Gaussians 球体赖以进行分裂（Split）和克隆（Clone）的核心梯度来源。渲染的高频细节因为误差过早闭合被忽视，导致模型拟合不足。

### 29.2 双域监督（Dual-domain Supervision）的干涉效应
第二阶段引入了强弱并行的组合场：
\$L = \sum W \cdot |I - I_{gt}| + \beta \sum |\log_{e}(1+I) - \log_{e}(1+I_{gt})|$\。
本以为这可以给低能量区域提供辅助引导。但由于两者针对的误差极化方向截然相反，这一混合算子使得原本集中往强烈点中心偏移聚集的 3D 高斯点云发生扩散和迷失。
此项理论实证验证：对于 3D 高斯在雷达图像等高动态（HDR）领域的监督优化，算子的非线性压制不可轻易引入 L1 反向传播主链，它需要采取诸如后续后处理渲染、或者结构上的剥离方法进行解耦研究才更安全。



## 30. Observation Operator Decoupling Principle
Following the confirmed structural failure of blending HDR suppressions (like \log1p\) directly inside the training iteration gradient loops (see 29.1 and 29.2), the system explicitly segments pply_isar_observation_operator to run post-hoc.
The observation transformations are strictly decoupled interfaces applied solely during eval/export steps against strictly RAW-trained geometry. This theoretically preserves Gaussian topological propagation gradients while retaining evaluation layer comparability (modes_intensity_obs) toward future ISAR physics integrations.


### 31. Post-Hoc Prototype: Minimal ISAR Decibel Scaling (\db_radar\)

In order to provide a 'more realistic' evaluation schema outside the training loop, the db_radar operator was initiated. radar signals are primarily evaluated in the Decibel (dB) scale because they encompass enormous dynamic ranges. A basic threshold and dB mapping mimics the physical radar receiver's minimum detectable bounds and normalization:
- **Mathematical Form**: {dB} = 10 \cdot \log_{10}(I + 10^{-4})$
- **Envelope Normalization**: {norm} = \text{clamp}((I_{dB} + 40.0) / 40.0, \text{min}=0.0)$
- **Why it is more realistic than log1p**: log1p(I) mathematically merges a linear regime (for small \I\) and a compressed regime (for large \I\). However, radar inherently treats ratio-based energy tracking through pure  \log_{10}()$ mapping across all bounds. This new prototype explicitly anchors a -40dB noise floor assumption and aligns pixel intensity exclusively as a relative power distribution. It acts strictly as a downstream evaluation rendering method without dragging network geometry out of convergence.



### The DB_RADAR Performance Inversion and Softplus Regularization
During the formal Mainline vs. Candidate branch comparison, an interesting metric inversion occurred. The candidate model (implementing \softplus\ and explicit DC scalar bounds) achieved slightly worse L1 metrics in the linear \
aw\ domain compared to the mainline model, but consistently generated significantly better L1 and PSNR metrics in the \db_radar\ domain.

**Mathematical Rationale:**
The linear \
aw\ L1 metric applies uniform weighting to absolute error. Therefore, models that easily overfit and saturate to extreme bright scattering peaks (which often happens with simple \exp()\ activation) artificially score better in \
aw\ error metrics but sacrifice dynamic range in low-intensity fields.

However, the physical \db_radar = (10 * log10(I + 1e-4) + 40) / 40\ operation applies a violent logarithmic compression. This means a numerical error near the noise floor (^{-4}$) is amplified massively compared to an error near peak values ($-0\text{dB}$). 
Because the Candidate branch utilizes de-optical initialization and \softplus\, it avoids catastrophic gradient explosion and strictly enforces stable, non-negative noise floors. This allows the model to accurately reconstruct the weak structural scatterers, leading to a direct quantitative victory in the DB_RADAR domain. This proves that softplus and explicit scalar modeling physically align far better with inverse SAR rendering properties.


## 32. Observation Context Interface 语义化（context_v1）

本节记录的是接口层语义升级，而不是训练目标改造。

### 32.1 升级前后的抽象差异

升级前：

$$
y = \mathcal{O}_{m}(I)
$$

其中 $m$ 为模式（identity / log1p / db_radar），接口只接收张量与 mode。

升级后：

$$
y = \mathcal{O}_{m}(I; \mathbf{c})
$$

其中 $\mathbf{c}$ 为上下文参数（context），可携带观测语义。

### 32.2 context_v1 当前字段

- 通用语义字段：
  - `normalization_mode`
  - `range_axis`
  - `cross_range_axis`
  - `future_physical_operator_name`
  - `clamp_min`, `clamp_max`
- `log1p` 相关：
  - `epsilon`
- `db_radar` 相关：
  - `epsilon`
  - `noise_floor_db`
  - `dynamic_range_db`
  - `normalization_mode`

### 32.3 db_radar 的参数化表达

当前 context 下，`db_radar` 的默认形式为：

$$
I_{dB} = 10\log_{10}(\max(I,0)+\epsilon),\quad \epsilon=10^{-4}
$$

$$
I_{norm} = \frac{I_{dB}-\text{noise\_floor\_db}}{\text{dynamic\_range\_db}}
$$

默认参数：

$$
	ext{noise\_floor\_db}=-40,\quad \text{dynamic\_range\_db}=40
$$

并采用：

$$
I_{out}=\operatorname{clamp}(I_{norm},\text{min}=0,\text{max}=\varnothing)
$$

其中 `max=None` 表示当前仅下限夹紧，保留高能量上溢供后续统计或可视化策略处理。

### 32.4 与双轨冻结的一致性

本轮只在观测接口层引入参数语义，不将 observation operator 前装进训练 loss，因此不破坏既有结论：

- 轨道 A 继续承担训练基线稳定性。
- 轨道 B 承担真实 ISAR 观测算子的接口挂载与语义扩展。

### 32.5 这一步证明了什么

通过既有 500/1000 模型在 stage_g 与 stage_h 的导出验证，三模式均可在 `mode + context` 下稳定运行，且 json 已能完整审计“模式-上下文-统计”三元关系。这意味着系统已从“观测算子原型函数”进入“观测算子接口语义层”。


## 33. Candidate-A 主线回正与再基线（Rebaseline）

本节记录的是训练监督口径回正，不涉及 CUDA、renderer、evaluation 脚本实现修改。

### 33.1 当前训练主线参数冻结

在当前 clean freeze 快照下，训练主线参数固定为：

- `--isar_supervision_mode a`
- `--isar_l1_weight_alpha 1.0`
- `--isar_l1_weight_gamma 2.0`

对应目标形式保持：

$$
L_A=\frac{\sum w\cdot |I-\hat I|}{\sum w+\epsilon},\quad
w=1+\alpha\cdot (I_{gt}^{+})^{\gamma}
$$

其中 $(\alpha,\gamma)=(1.0,2.0)$。

### 33.2 A@100 对齐结论（入口恢复有效性）

基于同口径对比（历史 vs 当前恢复）：

- $\Delta L1 \approx +7.93\times 10^{-5}$
- $\Delta PSNR \approx -4.58\times 10^{-3}\,\text{dB}$
- $\Delta q99 \approx -9.39\times 10^{-4}$

差异量级很小，说明 candidate-A 训练入口恢复后，A@100 轨迹已基本贴近历史参考。

### 33.3 A@500 / A@1000 正式复现结果

在同一快照、同一数据、同一 seed、同一黑底约定下：

- A@500、A@1000 均成功完成训练与导出。
- stage_g / stage_h 全部 finite，无 NaN / Inf / 崩溃。

相对历史 `math-1 + candidate-A` 基线（`modes_intensity.isar`）：

- A@500：
  - $\Delta L1=+0.002691$
  - $\Delta PSNR=-0.15346\,\text{dB}$
  - $\Delta q99=+0.004856$
  - $\Delta nonzero=+0.010790$
- A@1000：
  - $\Delta L1=+0.002437$
  - $\Delta PSNR=-0.18844\,\text{dB}$
  - $\Delta q99=-0.000237$
  - $\Delta nonzero=+0.002444$

这些差异表明本轮复现与历史主线处于同一监督族轨道，主行为一致，存在可接受重跑波动。

### 33.4 主线与研究分支职责

- 正式训练主线：`math-1 + candidate-A`。
- 研究分支：`deopt + softplus`（current/candidate）继续用于观测层与表示层研究，不作为正式训练主线。


## 34. A@8000 后期退化的最小调度修正实验（opacity reset 单次化）

本节记录的是训练调度实验，不涉及几何模型、投影公式或 CUDA 数学改动。

### 34.1 实验动机

在 36 视角 candidate-A 长训中，A@8000 相比 A@5000 出现后期退化迹象。只读审查后，优先嫌疑集中在 opacity reset 与 densify/prune 的耦合时序。

### 34.2 最小改动定义

仅改动 `train.py` 中 reset 触发条件：

- 原：每 `opacity_reset_interval` 周期触发。
- 新：仅在 `iteration == opacity_reset_interval` 时触发一次。

其余训练口径保持不变：

- `isar_supervision_mode=a`
- `isar_l1_weight_alpha=1.0`
- `isar_l1_weight_gamma=2.0`

### 34.3 结果解读（修正版 vs 原始 A@8000）

基于 `modes_intensity.isar`：

- $\Delta L1=-0.002514$，$\Delta PSNR=+0.265445\,\text{dB}$，表面像素误差有改善。
- 但 $\Delta nonzero=-0.049131$，说明有效回波覆盖下降。
- 同时 $\Delta q99=+0.116032$、$\Delta near\_white=+0.004910$，伴随亮部上冲与饱和风险。
- $\Delta near\_black=+0.056122$，黑场占比上升，与结构断裂/空洞趋势一致。

因此该改动属于“误差指标局部改善，但结构分布恶化”的不均衡改良，不满足“明显更稳且无副作用”。

### 34.4 当前数学层结论

- 本实验不构成几何或投影数学新结论。
- 可以确认：仅靠“opacity reset 单次化”不足以稳定 A@8000 后期形态。
- 该结果应归档为调度层负例证据，用于约束后续实验搜索空间。


## 35. late densify/prune 冻结验证结论（research only, not mainline）

### 35.1 实验定义

在保持 opacity reset 原始周期触发不变的条件下，仅收口 densify/prune 有效窗口：

$$
	ext{densify\_until\_iter}: 15000 \rightarrow 4000
$$

并执行 A@4000 $\rightarrow$ A@8000 的同口径续训对比。

### 35.2 结果要点

相对原始 A@8000（`modes_intensity.isar`）：

- $\Delta L1=-0.003908$
- $\Delta PSNR=+1.003192\,\text{dB}$
- $\Delta near\_black=-0.027185$
- $\Delta near\_white=-0.000214$（回到 0）

同时视觉上碎裂/黑线/黑块显著缓解，说明“4000 后持续 densify/prune”比 opacity reset 更接近主因。

### 35.3 当前可用结论

- 该结论属于训练调度层，不改变几何或投影数学结论。
- 在研究分支范围内，可将“freeze densify after 4000”作为下一阶段默认 schedule 基座。
- 标记：research only，not mainline。


## 36. 多 elevation 监督平衡（research only）

本节只讨论监督侧权重，不涉及几何、reader、CUDA 或投影数学改写。

### 36.1 逐帧动态 balance（已验证，不作为默认）

定义当前视图 GT 亮度均值 $m_t$ 与 EMA $e_t$：

$$
e_t = 0.99 e_{t-1} + 0.01 m_t
$$

逐帧权重：

$$
w_t = \mathrm{clamp}\left(\sqrt{\frac{e_t}{m_t + \varepsilon}},\ 0.5,\ 2.0\right)
$$

总损失外层加权：

$$
L_t' = w_t \cdot L_t
$$

结论：可强化 0° 纬线，但会带来整体亮度/体量不稳，未选为默认。

### 36.2 按 elevation 静态 balance（当前候选基座的一部分）

按训练集统计每条纬线 GT 均值 $\mu_e$，整体均值 $\mu_{all}$：

$$
w_e^{raw} = \sqrt{\frac{\mu_{all}}{\mu_e + \varepsilon}}
$$

执行 clamp（当前实现边界 $[0.67, 1.5]$）与均值归一化后，得到固定权重 $\tilde{w}_e$，训练时仅按当前视图所属 elevation 乘外层标量：

$$
L_t' = \tilde{w}_{e(t)} \cdot L_t
$$

在数据集 `isar_Hubble1_front90_3elev30_20260403` 的一组实际权重示例：

$$
	ilde{w}_{-18.303} \approx 0.8105,\quad
	ilde{w}_{0.0} \approx 1.4018,\quad
	ilde{w}_{+18.303} \approx 0.7877
$$

### 36.3 组合候选：a2g2 + static elevation balance

在 candidate-A 像素内权重保持不变的前提下，组合策略为：

1. 像素内仍用

$$
w_{px} = 1 + \alpha I_{gt}^{\gamma},\quad (\alpha,\gamma)=(2,2)
$$

2. 视图外层再乘固定纬线权重 $\tilde{w}_{e(t)}$。

该组合在当前四方案中表现为更均衡折中：
- 保持较好误差/PSNR。
- 比单独 static elevation balance 明显减轻偏黑副作用。

### 36.4 当前数学层结论

- 当前最优候选仍属于监督侧改造，不触及几何或投影主方程。
- 结论边界：`a2g2 + static elevation balance` 可作为多 elevation 当前默认监督候选继续推进（research only）。
- 下一阶段优先在监督侧做轻量微调，不进入 math2。

