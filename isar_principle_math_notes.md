# ISAR-3DGS 原理与数学记录

## 文档用途
本文档用于记录当前几何假设、参数物理意义、参数通路边界，以及进入 forward 数学前的关键注意事项。

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
- forward.cu 尚未基于 projection_mode 进入新投影公式分支。
- backward.cu 尚未引入正交对应梯度分支。

---

## 5. 为什么这一轮还没有真正启用 ISAR/正交投影

这是有意的分阶段策略：
- 先稳定接口，再改数学，避免问题定位耦合。
- 如果接口和公式同时改，出错时很难快速定位到责任层。
- 前向与反向需要分开验证，减少一次性风险。

因此当前仍保持：
- 透视 FoV/tanfov 路径有效。
- 透视 projection matrix 与透视除法有效。
- 与此前 smoke 基线可对比。

---

## 6. 下一轮进入 forward 数学前还需要注意什么

1. 首次只改 forward，不改 backward。
- 先验证渲染行为、形状一致性、数值稳定性。

2. 保留双路径可切换。
- projection_mode 必须保留 perspective 回归路径。

3. 先明确正交坐标约定。
- 轴方向、尺度单位、像素中心映射需先固定。

4. 继续使用最小验证闭环。
- Scene 冒烟。
- forward 渲染合理性检查。
- 2-iteration 训练冒烟（保留必要临时兼容）。

5. 临时补丁保持可追踪。
- train.py 中单通道扩 3 通道仅用于 smoke，不是最终建模结论。

6. 占位项需要后续逐步替换。
- ISAR reader 的 FoV 占位映射。
- reader 随机点云初始化。
- renderer 兼容过滤（环境固定后可收敛）。

---

## 7. 阶段 C 后的可执行状态

已经就绪：
- 新投影参数可从数据侧一路传到 CUDA 调用边界。
- 重编译与 smoke 流程已验证可执行。

尚未完成：
- forward.cu 中正交/ISAR 分支公式启用。
- backward.cu 对应梯度链改造。

建议的直接下一步：
- 在 forward.cu 中引入 projection_mode 分支并消费 ortho 参数，完成前向验证后再进入 backward。

---

## 8. 后续更新清单

每次更新本文件应明确写出：
- 哪些几何假设发生变化。
- 哪些公式发生变化。
- 参数通路从“可到达”变为“已消费”的边界位置。
- 哪些临时方案被替换。
- 做了哪些验证、验证证明了什么。
