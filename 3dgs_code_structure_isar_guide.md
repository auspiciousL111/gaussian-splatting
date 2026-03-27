# Gaussian Splatting 官方代码结构与 ISAR 改造指南

> 面向当前工作区 `gaussian-splatting` 的源码梳理文档。目标是帮助你在改造 3DGS 以支持 ISAR 数据集前，快速建立“文件职责 - 数据流 - 依赖关系 - 改动入口”的完整心智模型。

## 1. 项目整体结构概览

### 1.1 根目录职责

- `train.py`：训练主入口（优化循环、损失、densification、保存模型）
- `render.py`：离线渲染入口（加载训练好的点云高斯并导出 render/gt）
- `metrics.py`：评测入口（SSIM/PSNR/LPIPS）
- `full_eval.py`：批量训练+渲染+评测脚本（论文 benchmark 用）
- `convert.py`：COLMAP 预处理脚本（特征提取、匹配、建图、去畸变）
- `scene/`：数据集读取、相机对象、高斯参数模型
- `gaussian_renderer/`：Python 侧渲染封装（调用 CUDA 光栅器）
- `submodules/diff-gaussian-rasterization/`：核心可微高斯光栅化 CUDA 扩展
- `submodules/simple-knn/`：KNN 距离估计（初始化高斯尺度）
- `submodules/fused-ssim/`：加速 SSIM 的 CUDA 扩展
- `utils/`：相机、几何、loss、COLMAP I/O 等工具
- `SIBR_viewers/`：实时可视化 viewer（C++/OpenGL）
- `output/`：训练输出目录（模型、日志、渲染结果）

### 1.2 你当前工作区中的数据组织

你当前有一个单独数据目录：

- `3DGS_DATA/train/`
- `3DGS_DATA/truck/`

每个场景下都有典型 COLMAP 稀疏重建结构：

- `images/`
- `sparse/0/cameras.bin`
- `sparse/0/images.bin`
- `sparse/0/points3D.bin`

这与 `scene/dataset_readers.py` 的读取假设完全一致（优先读取 bin，失败时回退 txt）。

---

## 2. 核心目录详解

## 2.1 `scene/`：数据到可训练对象的桥梁

### 关键文件

- `scene/__init__.py`
- `scene/dataset_readers.py`
- `scene/colmap_loader.py`
- `scene/cameras.py`
- `scene/gaussian_model.py`

### 职责分工

1. `Scene`（`scene/__init__.py`）
- 判定数据类型（COLMAP 或 Blender）
- 构建 `train_cameras` / `test_cameras`
- 初次训练时把输入点云复制成 `output/.../input.ply`
- 负责保存高斯模型（`point_cloud/iteration_x/point_cloud.ply`）和曝光参数（`exposure.json`）

2. `dataset_readers.py`
- 读 COLMAP 相机内外参、点云、深度参数
- 生成 `CameraInfo` / `SceneInfo`
- 计算场景归一化半径 `nerf_normalization["radius"]`
- 决定 train/test 切分（`--eval` 时 LLFF holdout）

3. `colmap_loader.py`
- 二进制/文本格式 COLMAP 文件解析器（`images.bin`、`cameras.bin`、`points3D.bin`）
- 四元数和旋转矩阵互转

4. `cameras.py`
- `Camera` 类保存每张图的几何信息、图像、深度图、投影矩阵
- 构造 `world_view_transform` 与 `projection_matrix`

5. `gaussian_model.py`
- 定义可训练参数：
  - `_xyz`（中心）
  - `_features_dc/_features_rest`（SH 系数）
  - `_opacity`
  - `_scaling`
  - `_rotation`
- 实现 densify/prune、优化器设置、PLY 存取
- `create_from_pcd()` 里用 `simple-knn` 初始化尺度

## 2.2 `gaussian_renderer/`：Python 渲染封装层

### 关键文件

- `gaussian_renderer/__init__.py`
- `gaussian_renderer/network_gui.py`

### 职责

1. `render(...)`
- 从 `Camera` 提取 FOV、变换矩阵
- 构造 `GaussianRasterizationSettings`
- 准备高斯输入（位置、透明度、颜色/SH、尺度/旋转或预计算协方差）
- 调用 `GaussianRasterizer`（C++/CUDA 扩展）
- 输出 `render`, `radii`, `depth`, `viewspace_points`

2. `network_gui.py`
- 与外部 viewer 通信（socket）
- 可在训练中实时查看当前渲染结果

## 2.3 `submodules/diff-gaussian-rasterization/`：核心 CUDA 可微渲染

### 分层结构

1. Python Autograd 包装
- `diff_gaussian_rasterization/__init__.py`

2. PyBind 绑定
- `ext.cpp`

3. C++ 接口层（Torch Tensor 转裸指针、缓冲区管理）
- `rasterize_points.h`
- `rasterize_points.cu`

4. CUDA 渲染实现
- `cuda_rasterizer/forward.cu`
- `cuda_rasterizer/backward.cu`
- `cuda_rasterizer/rasterizer_impl.cu`
- `cuda_rasterizer/auxiliary.h`
- `cuda_rasterizer/forward.h`
- `cuda_rasterizer/backward.h`

### 你后续 ISAR 改造最核心的三个文件

- `cuda_rasterizer/forward.cu`
- `cuda_rasterizer/backward.cu`
- `cuda_rasterizer/auxiliary.h`

## 2.4 `utils/`：通用工具

- `graphics_utils.py`：投影矩阵与 FOV/焦距换算
- `camera_utils.py`：相机加载与缩放
- `loss_utils.py`：L1/SSIM
- `image_utils.py`：PSNR
- `read_write_model.py`：COLMAP 文件读写（可用于你改 `cameras.bin`）
- `general_utils.py`：学习率调度、旋转矩阵构建、随机种子

## 2.5 `submodules/simple-knn/`

- 对每个点估计邻域平均距离（CUDA）
- 在 `gaussian_model.py` 里用于初始化 `scale`

## 2.6 `submodules/fused-ssim/`

- 提供 fused SSIM 前后向 CUDA 实现
- 训练中可替代 Python 版 SSIM 加速

---

## 3. 核心 Python 文件功能与调用路径

## 3.1 `train.py`：训练主循环

主要流程：

1. 参数解析（`arguments/`）
2. 初始化 `GaussianModel` + `Scene`
3. 随机取训练相机
4. 调用 `gaussian_renderer.render()`
5. 计算损失（L1 + DSSIM + 可选深度正则）
6. 反向传播
7. densify & prune
8. 保存 checkpoint / point cloud / TensorBoard 日志

关键点：

- `viewspace_points.grad` 被用于 densification 决策
- `radii` 用于可见性筛选和稀疏优化
- 支持 `SparseGaussianAdam`（加速版扩展）

## 3.2 `render.py`：离线渲染

- 加载某次迭代模型
- 对 train/test 相机批量渲染
- 输出：
  - `.../renders/*.png`
  - `.../gt/*.png`

## 3.3 `metrics.py`

- 读取渲染图与 GT
- 计算 SSIM/PSNR/LPIPS
- 写 `results.json` 与 `per_view.json`

## 3.4 `convert.py`

- 数据准备脚本，调用 COLMAP + ImageMagick
- 把原图转换成 3DGS 可训练输入布局

---

## 4. CUDA 光栅化模块详解（你最可能改动的地方）

## 4.1 Forward 关键路径

入口调用链：

`gaussian_renderer/__init__.py -> diff_gaussian_rasterization/__init__.py -> rasterize_points.cu -> rasterizer_impl.cu -> forward.cu`

核心步骤：

1. `preprocessCUDA`（`forward.cu`）
- 视锥裁剪
- 世界坐标 -> 投影坐标
- 3D 协方差（scale+rotation 或外部给定）
- 3D 协方差 -> 2D 协方差（`computeCov2D`）
- 计算屏幕半径与 tile 覆盖范围
- SH -> RGB（可选）

2. `rasterizer_impl.cu`
- 生成 tile key（tile_id + depth）
- radix sort
- 每 tile 并行混合 splat

3. `renderCUDA`（`forward.cu`）
- 对每像素累计颜色和 inverse depth
- 输出最终图像与辅助 buffer

## 4.2 Backward 关键路径

入口调用链：

`autograd backward -> rasterize_points.cu::RasterizeGaussiansBackwardCUDA -> rasterizer_impl.cu::Rasterizer::backward -> backward.cu`

核心步骤：

1. `BACKWARD::render`（`backward.cu`）
- 反向重走 blending
- 得到对 `mean2D`、`conic2D`、`opacity`、`color`、`invdepth` 的梯度

2. `computeCov2DCUDA`（`backward.cu`）
- 把 `conic2D` 梯度传回 `cov3D` 与部分 `mean3D`
- 这是透视雅可比最强相关位置

3. `preprocessCUDA`（`backward.cu`）
- `mean2D -> mean3D` 的链式梯度
- SH 梯度回传
- scale/rotation 梯度回传

---

## 5. 数据流与处理流程（训练一轮）

## 5.1 数据进入

1. `train.py` 解析参数
2. `Scene(...)` 调用 `dataset_readers.py`
3. `dataset_readers.py` 从 `sparse/0/*.bin` + `images/` 构造相机
4. `GaussianModel.create_from_pcd()` 用 COLMAP 点初始化高斯

## 5.2 前向渲染

1. 选一个 `Camera`
2. `gaussian_renderer.render(...)`
3. CUDA 光栅化输出图像与深度

## 5.3 损失与更新

1. L1 + DSSIM (+ depth regularization)
2. `loss.backward()`
3. 优化器更新 `_xyz/_features/_opacity/_scale/_rotation`
4. densify/prune

## 5.4 输出产物

- `output/<exp>/point_cloud/iteration_x/point_cloud.ply`
- `output/<exp>/cameras.json`
- `output/<exp>/cfg_args`
- `output/<exp>/exposure.json`（若启用曝光补偿）

---

## 6. 模块间依赖关系

## 6.1 训练链路依赖图

```text
train.py
  -> arguments/*
  -> scene/__init__.py
     -> scene/dataset_readers.py
        -> scene/colmap_loader.py
        -> utils/graphics_utils.py
     -> scene/cameras.py
     -> scene/gaussian_model.py
        -> simple_knn._C
  -> gaussian_renderer/__init__.py
     -> diff_gaussian_rasterization (Python)
        -> ext.cpp/PyBind
        -> rasterize_points.cu (C++ bridge)
        -> cuda_rasterizer/*.cu (CUDA core)
  -> utils/loss_utils.py
     -> fused_ssim (optional)
```

## 6.2 最小改动闭环（ISAR）

你如果要把“相机透视成像”改成“雷达正交/近正交成像”，最小闭环通常涉及：

1. 数据读取与相机参数表达
2. 前向投影与 2D 协方差传播
3. 反向梯度公式
4. 输入接口（图像名包含姿态信息）

---

## 7. 修改指南：支持 ISAR 数据集

> 以下给的是“工程改造地图”和“推导模板”。你师兄提到的 `forward.cu/backward.cu` 改动方向是正确的，但通常不止这两个文件。

## 7.1 必改文件清单（按优先级）

### A. 投影模型替换（核心）

1. `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
- `computeCov2D(...)`：当前使用透视 Jacobian
- `preprocessCUDA(...)`：当前用齐次投影 `p_proj = p_hom / w`

2. `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
- `computeCov2DCUDA(...)`：透视链式梯度
- `preprocessCUDA(...)`：`mean2D -> mean3D` 的透视梯度

3. `submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h`
- `in_frustum(...)`：当前按透视近裁剪

### B. 相机/数据接口（高优先）

4. `scene/dataset_readers.py`
- 增加 ISAR 数据读取回调
- 解析“文件名包含视角信息”的逻辑
- 构造 `CameraInfo`（可复用字段，但语义会变化）

5. `scene/__init__.py`
- 扩展数据类型识别（例如 `isar_meta.json` 或自定义目录标记）

6. `scene/cameras.py`
- 增加 ISAR 相机参数（方位角、俯仰角、距离、波束参数等）
- 重新定义 `projection_matrix`（若走正交）

7. `utils/graphics_utils.py`
- 新增正交投影矩阵函数（例如 `getOrthoProjectionMatrix(...)`）

### C. COLMAP 兼容与元数据桥接（视方案）

8. `scene/colmap_loader.py` 或 `utils/read_write_model.py`
- 如果继续走 COLMAP 文件格式壳层：要改 `cameras.bin` 解释方式
- 如果彻底绕过 COLMAP：可以新建 `readISARSceneInfo(...)`，不必强依赖 `cameras.bin`

### D. 训练脚本参数接口（建议）

9. `arguments/__init__.py`
- 增加 `--dataset_type isar`
- 增加 ISAR 参数（例如 `--isar_pose_from_filename`）

10. `train.py` / `render.py`
- 分支逻辑：根据数据类型切换投影模型和加载流程

## 7.2 透视改正交时，Forward 数学改造骨架

当前透视核心是：

$$
\mathbf{p}_{clip}=\mathbf{P}\mathbf{V}\mathbf{X},\quad
\mathbf{p}_{ndc}=\frac{1}{w}(x,y,z)
$$

正交投影可改为（示意）：

$$
u = s_x x_v + b_x,\quad v = s_y y_v + b_y,
$$

其中 $(x_v,y_v,z_v)$ 是 view 坐标。

对应 Jacobian（对 view 坐标）从透视：

$$
\mathbf{J}_{persp}=
\begin{bmatrix}
\frac{f_x}{z} & 0 & -\frac{f_x x}{z^2}\\
0 & \frac{f_y}{z} & -\frac{f_y y}{z^2}
\end{bmatrix}
$$

变成正交：

$$
\mathbf{J}_{ortho}=
\begin{bmatrix}
s_x & 0 & 0\\
0 & s_y & 0
\end{bmatrix}
$$

这意味着你师兄提到“雅可比删除/修改”就是这里：

- `forward.cu::computeCov2D` 里 `J` 需要重写
- `backward.cu::computeCov2DCUDA` 里与 $\partial J/\partial t$ 相关项多数可消失或大幅简化

## 7.3 Backward 公式改造清单（约 10 类）

你说“可能十个左右公式要改”，从代码映射看通常至少有这些：

1. `p_proj = p_hom / w` 的梯度（可删/改）
2. `J` 的定义与其对 $t=(x,y,z)$ 的导数
3. `cov2D = T^T V T` 中 `T = WJ` 的梯度链
4. `dL/dmean` 由投影链传回的部分
5. inverse depth 分支里 `dL/dtz` 相关项
6. 屏幕边界裁剪引入的 `x_grad_mul/y_grad_mul`
7. 半径估计的特征值路径（若正交下你改了核尺度）
8. frustum/near culling 的判定与梯度有效域
9. NDC->像素缩放项
10. 若你把深度定义改成雷达距离（range），则 `depth` 分支梯度需重推

建议按“先跑通前向，再逐步打开反向”策略：

1. 只改 forward（先可渲染）
2. backward 先写近似梯度验证收敛方向
3. 再替换为严格推导公式

## 7.4 文件名携带视角信息的接口改法

你提到“每张图文件名带视角信息”。推荐做法：

1. 在 `dataset_readers.py` 新增解析函数，例如：
- `parse_isar_pose_from_name("az30_el10_r100_img001.png")`

2. 在读取图片列表时直接构造 `CameraInfo`：
- `R/T` 由解析结果转成外参
- 内参可设为统一雷达成像参数

3. 避免强耦合 COLMAP：
- ISAR 数据不一定需要 `cameras.bin/images.bin`
- 可新增 `sceneLoadTypeCallbacks["ISAR"]`

## 7.5 `cameras.bin` 是否必须改

取决于你的数据接入策略：

1. 继续复用 COLMAP 管线
- 需要构造/改写 `cameras.bin` + `images.bin`，并确保 `PINHOLE` 参数能被现有代码接受

2. 新增 ISAR 原生读取管线（推荐）
- 不必依赖 `cameras.bin`
- 直接在 `dataset_readers.py` 生成 `CameraInfo`

对于 ISAR，通常第二种更自然。

## 7.6 推荐改造顺序（可执行）

1. 新建 ISAR 数据读取回调（不改 CUDA）
2. 让 Python 端能用 ISAR 相机加载并进入 `render()`
3. 改 `forward.cu` 投影为正交，先保证前向画面合理
4. 改 `backward.cu`，先保证训练不崩
5. 对照数值梯度做单元验证（建议单高斯场景）
6. 再做完整训练与指标对比

## 7.7 你现在可以直接开始看的“最关键文件 Top 12”

1. `train.py`
2. `scene/__init__.py`
3. `scene/dataset_readers.py`
4. `scene/cameras.py`
5. `scene/gaussian_model.py`
6. `gaussian_renderer/__init__.py`
7. `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
8. `submodules/diff-gaussian-rasterization/rasterize_points.cu`
9. `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`
10. `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
11. `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
12. `utils/graphics_utils.py`

---

## 8. 补充：当前输出目录含义

以 `output/4d0d1594-3` 为例：

- `cfg_args`：训练时参数快照
- `cameras.json`：相机导出信息（便于 viewer）
- `point_cloud/iteration_*/point_cloud.ply`：每个保存步的高斯模型
- `train/ours_*/` 和 `test/ours_*/`：渲染结果
- `exposure.json`：曝光参数

---

## 9. 小结

你后续做 ISAR 改造时，最可能的真实改动集合是：

- 数据读取：`scene/dataset_readers.py`, `scene/__init__.py`, `scene/cameras.py`
- 相机数学：`utils/graphics_utils.py`
- CUDA 前向：`forward.cu`
- CUDA 反向：`backward.cu`
- 训练参数与开关：`arguments/__init__.py`, `train.py`, `render.py`

你师兄给的方向（forward/backward + 接口 + colmap_loader/cameras.bin）是主干正确，但为了项目稳定，建议采用“新增 ISAR 读取分支 + 最小侵入替换投影模型”的方式，不要把 COLMAP 逻辑硬改坏。

如果你愿意，我下一步可以直接在这份文档基础上继续给你出一份“ISAR 改造任务清单（按 commit 切分）”，每个 commit 只改 1-2 个文件，方便你逐步验证。