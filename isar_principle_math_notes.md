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
