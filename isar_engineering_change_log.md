# ISAR-3DGS 工程变更记录

## 文档用途
本文档用于按阶段记录 ISAR 适配的真实工程改动与验证结果，便于科研笔记整理、复现实验与后续审计。

## 维护约定
每完成一个新阶段，按本文同样模板新增一节，并明确哪些是最终方案、哪些仍是临时占位。

---

## 阶段 0：RayISAR 多视角导出

### 阶段名称
阶段 0 - RayISAR 多视角数据导出

### 本轮目标
在 RayISAR 外部新增最小导出脚本，生成 3DGS 可直接读取的数据集边界：images + poses.csv。

### 修改文件清单
- 外部项目文件：D:/RayISAR_1.2/setup_isar_multiview.py

### 每个文件改了什么
- D:/RayISAR_1.2/setup_isar_multiview.py
  - 新增 azimuth 扫描导出循环。
  - 新增 All Reflections_Fr.tif 的逐帧导出/复制逻辑。
  - 新增 poses.csv 写出逻辑。
  - 新增默认数据集输出路径。
  - 新增 LOS/角度转换与数据目录初始化辅助函数。

### 为什么这样改
原始 RayISAR 运行模式是单次单视角，不满足 3DGS 训练对多视角图像与每帧位姿元数据的输入要求。

### 仍然是临时/占位方案的地方
- 当前导出流程为最小可用实现，不是完整的批处理工程化框架。
- LOS/up 约定固定为当前实验设定，后续可扩展为可配置策略。

### 验证方式与结果
- 验证导出目录结构正确（images + poses.csv）。
- 验证图像命名符合 img_0000.tif、img_0001.tif 约定。
- 验证 poses.csv 字段齐全且逐帧有值。

### 本轮核心结论
已具备可复用的 ISAR 多视角导出能力，成功建立 3DGS 数据接入的上游边界。

### 下一步建议
进入 3DGS 侧 reader 级接入，先不改 CUDA 数学。

---

## 阶段 1：3DGS 数据接入

### 阶段名称
阶段 1 - ISAR 数据读取与 Scene 路由接入

### 本轮目标
让 3DGS 能直接读取 ISAR 数据格式，并自动识别为 ISAR 数据集进入 Scene 构建。

### 修改文件清单
- scene/dataset_readers.py
- scene/__init__.py
- smoke_test_isar_scene.py

### 每个文件改了什么
- scene/dataset_readers.py
  - 新增 ISAR poses.csv 解析逻辑。
  - 新增 LOS/up/distance 到 R/T 的临时映射函数。
  - 新增 readIsarSceneInfo 读取入口。
  - 新增 sceneLoadTypeCallbacks 中 ISAR 回调注册。
  - 新增按行构造 CameraInfo 的 ISAR 路径。
- scene/__init__.py
  - 新增自动识别规则：存在 poses.csv 且存在 images 目录则判定为 ISAR。
  - 新增 ISAR 场景路由到 ISAR reader。
- smoke_test_isar_scene.py
  - 新增 Scene 构建冒烟脚本。
  - 新增 camera list 路径命中检查与首帧信息打印。

### 为什么这样改
在修改投影数学之前，必须先把数据接入链路稳定下来，保证数据格式、位姿元数据和 Scene 构建可独立验证。

### 仍然是临时/占位方案的地方
- ISAR 路径中的 FoV 仍由 window_size 映射得到，仅用于兼容。
- 点云初始化仍使用临时随机点云方案。
- 当前几何语义是兼容态，不是最终 ISAR 投影定义。

### 验证方式与结果
- 执行 Scene 冒烟测试，结果 PASS。
- 确认 train/test camera 列表构建路径被命中。
- 确认 poses.csv + images 自动识别为 ISAR 数据集。

### 本轮核心结论
ISAR 数据已经可以端到端进入 3DGS Python 场景管线。

### 下一步建议
执行短训练冒烟，先暴露通道与渲染链兼容问题，再进入 CUDA 侧工作。

---

## 阶段 2：单通道兼容训练 smoke

### 阶段名称
阶段 2 - 单通道 GT 临时兼容补丁（仅用于 smoke）

### 本轮目标
在 renderer 与损失仍偏 RGB 假设的前提下，让灰度 ISAR GT 可以完成 1-2 iter 训练冒烟。

### 修改文件清单
- train.py

### 每个文件改了什么
- train.py
  - 在损失计算前新增临时逻辑：若 GT 形状为 1xHxW，则重复为 3xHxW。

### 为什么这样改
当前渲染和损失链条多个位置默认按 RGB 张量工作。为了先跑通最小训练验证，需要一个最小适配补丁。

### 仍然是临时/占位方案的地方
- 该补丁仅用于短程 smoke 验证。
- 不代表最终 ISAR 建模应采用三通道。

### 验证方式与结果
- 执行 2-iteration 训练冒烟，训练流程正常结束。

### 本轮核心结论
在不改投影数学的前提下，训练主循环已可用于快速回归验证。

### 下一步建议
进入参数建模与传递链打通（A+B+C），为 forward 数学切换做接口准备。

---

## 阶段 3：A+B+C 参数建模与传递链打通

### 阶段名称
阶段 3 - 相机参数建模 + Python 到 C++/CUDA 正式透传

### 本轮目标
- A+B：在 Camera/Reader/Renderer 层建立新参数字段并可打包。
- C：将参数正式透传到扩展 binding 与 CUDA 调用边界。
- 明确要求：不改 forward.cu 投影公式，不改 backward.cu。

### 修改文件清单
- scene/dataset_readers.py
- utils/camera_utils.py
- scene/cameras.py
- gaussian_renderer/__init__.py
- submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
- submodules/diff-gaussian-rasterization/rasterize_points.h
- submodules/diff-gaussian-rasterization/rasterize_points.cu
- submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h
- submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
- submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h
- submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu

### 每个文件改了什么
- scene/dataset_readers.py
  - CameraInfo 新增字段：projection_mode、ortho_scale_x、ortho_scale_y、isar_window_size。
  - ISAR reader 填充上述字段。
- utils/camera_utils.py
  - 新字段最小透传到 Camera 构造（含默认兜底）。
- scene/cameras.py
  - Camera 与 MiniCam 新增字段与成员存储。
- gaussian_renderer/__init__.py
  - projection_mode 改为整数枚举映射后打包。
  - 新字段加入 raster settings 打包。
  - 保留兼容过滤逻辑，避免环境不一致导致运行中断。
- diff_gaussian_rasterization/__init__.py
  - GaussianRasterizationSettings 扩展新字段。
  - forward/backward 参数打包加入四个新字段。
  - projection_mode 类型改为 int。
- rasterize_points.h / rasterize_points.cu
  - 扩展 C++ 前后向入口函数签名。
  - 将新参数转发到 Rasterizer 前后向接口。
- rasterizer.h / rasterizer_impl.cu
  - 扩展 Rasterizer 前后向签名。
  - forward 路径将参数传到 FORWARD::preprocess 边界。
  - backward 路径先接收参数但不使用。
- forward.h / forward.cu
  - 扩展 preprocess 与 kernel 边界签名。
  - 仅新增参数占位，不改投影公式。

### 为什么这样改
先把接口链路打通，能把后续数学改动隔离为独立提交，显著降低调试耦合与回归风险。

### 仍然是临时/占位方案的地方
- projection_mode 已透传但尚未驱动 forward.cu 分支数学。
- backward.cu 仍保持原状。
- FoV 兼容路径仍然有效。
- renderer 兼容过滤仍存在（用于环境混合期）。

### 验证方式与结果
- 扩展重编译并重装成功（清理构建缓存后全量编译通过）。
- Scene 冒烟测试 PASS。
- 2-iteration 训练冒烟测试完成。

### 本轮核心结论
新相机参数已正式传到 CUDA 调用边界，且在不改变现有数值行为的前提下通过了完整冒烟回归。

### 下一步建议
下一阶段仅进入 forward 数学改造：基于 projection_mode 与 ortho 参数引入正交分支，先不改 backward.cu。

---

## 阶段 4：forward 前向分支启用（仅前向）

### 阶段名称
阶段 4 - 在 forward.cu 启用 projection_mode 正交/ISAR 分支（不改 backward）

### 本轮目标
- 保持官方 perspective 路径完全不变。
- 在 projection_mode 下新增 orthographic/isar 前向投影分支。
- 仅改前向投影相关数学与前向中和投影直接相关的 2D 协方差映射。
- 不改 backward.cu，不改训练损失逻辑。

### 修改文件清单
- submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
- smoke_test_isar_forward_only.py

### 每个文件改了什么
- submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu
  - `computeCov2D(...)` 新增 `projection_mode` 分支。
  - 保留官方透视公式为独立路径（代码内注释标明 unchanged）。
  - 新增 orthographic/isar 路径：使用 `ortho_scale_x/ortho_scale_y` 进行线性缩放映射，并加入 `isar_window_size` 兜底。
  - `preprocessCUDA(...)` 中 3D 点到 2D 的投影新增分支：
    - perspective：继续使用齐次投影 + 透视除法（不改）。
    - orthographic/isar：使用 view-space 线性映射到 NDC-like 平面。
  - 在前向协方差调用处传入新参数。
- smoke_test_isar_forward_only.py
  - 新增 forward-only 冒烟脚本（仅渲染，不反向）。
  - 输出有限性、最值、均值、标准差、非零像素比例，用于快速排查 NaN/全黑。

### 为什么这样改
阶段 C 已完成参数通路，阶段 4 的目标是最小开启前向分支，让“参数可传”升级为“前向已消费”，同时保持 backward 与训练目标不变，减少耦合风险。

### 仍然是临时/占位方案的地方
- backward 仍是透视梯度链，尚未适配正交分支。
- ISAR reader 中 FoV 与随机点云仍为兼容期占位。
- 正交分支当前优先保证稳定可运行，不代表最终物理标定已完成。

### 验证方式与结果
- 扩展重编译与重装：成功。
- perspective 回归（2-iter train smoke）：成功完成，无回归崩溃。
  - 命令：`train.py --source_path D:/3DGS_new/3DGS_DATA/train --iterations 2 --eval`
- orthographic/isar forward-only smoke：PASS。
  - 输出示例：`finite_ok=True`，`image_shape=(3,1200,1200)`，`nonzero_ratio=1.000000`，无 NaN/Inf/崩溃。

### 本轮核心结论
已在不触碰 backward 的前提下，成功启用并验证 forward 的 projection_mode 双分支：
- perspective 路径保持可用且无回归。
- orthographic/isar 前向分支可运行、输出稳定。

### 下一步建议
进入 backward 分支改造前，先做一个小规模可视化/统计对比（perspective vs orthographic）确认前向几何行为符合预期，再进入 backward 梯度链改造。

---

## 阶段 5：backward 分支对齐（最小梯度链）

### 阶段名称
阶段 5 - 在 backward.cu 补齐 projection_mode 对应的最小梯度链

### 本轮目标
- 只改 backward 相关代码。
- 保持 forward.cu 数学不变。
- 在 backward 中补齐与当前 forward 双分支严格对应的最小梯度链：
  - 2D 协方差路径（computeCov2DCUDA）
  - 2D 均值路径（preprocessCUDA）

### 修改文件清单
- submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h
- submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
- submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu

### 每个文件改了什么
- submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h
  - `BACKWARD::preprocess(...)` 新增参数：
    - `projection_mode`
    - `ortho_scale_x`
    - `ortho_scale_y`
    - `isar_window_size`

- submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu
  - `computeCov2DCUDA(...)` 新增 projection_mode 分支：
    - perspective 分支保留原梯度链。
    - orthographic/isar 分支使用与 forward 一致的常数 Jacobian（由 `ortho_scale_x/y` 或 `isar_window_size` 兜底计算）。
  - 在 covariance 路径中明确：
    - orthographic 下协方差对均值不显式依赖，故该路径对均值的几何梯度为 0。
    - inverse-depth 梯度项保持两分支共享（`invdepth = 1 / z`）。
  - `preprocessCUDA(...)` 新增 projection_mode 分支：
    - perspective：沿用原投影反传公式。
    - orthographic/isar：按 `x_ndc = clamp(scale_x * x_view)`、`y_ndc = clamp(scale_y * y_view)` 反传到 view，再由 view 反传到 world。
  - `BACKWARD::preprocess(...)` 透传新增参数给上述两个 kernel。

- submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu
  - 删除 backward 中对 `projection_mode / ortho_scale_x / ortho_scale_y / isar_window_size` 的 `(void)` 占位。
  - 在调用 `BACKWARD::preprocess(...)` 时转发这些参数。

### 为什么这样改
阶段 4 已启用 forward 双分支；若 backward 继续固定透视链，会造成前后向不一致。阶段 5 只补最小投影梯度链，保证最小可训练闭环，避免一次性扩大改动面。

### 仍然是临时/占位方案的地方
- orthographic 尺度仍使用当前定义（含 `isar_window_size` 兜底），尚未做物理标定精化。
- 仅补齐“投影直接相关”梯度链，未扩展到更大范围的 ISAR 建模优化。

### 验证方式与结果
- 扩展重编译与重装：成功。
  - 命令：`pip install -e submodules/diff-gaussian-rasterization --no-build-isolation`
- perspective 回归 smoke（2-iter）：成功。
  - 命令：`train.py --source_path D:/3DGS_new/3DGS_DATA/train --iterations 2 --eval`
- isar/orthographic 最小 backward smoke（2-iter 短训练）：成功。
  - 命令：`train.py --source_path D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest --images images --iterations 2`

### 本轮核心结论
已完成 backward 与当前 forward 分支的一致化最小闭环：
- perspective 训练回归不退化。
- orthographic/isar 可执行最小反向训练路径。

### 下一步建议
以本节点作为 baseline 后，可进入下一小阶段：
- 在固定当前分支定义的前提下，增加分支间数值对照与梯度统计，逐步替换占位尺度策略。

---

## 阶段 6：梯度数值对照验证（Stage F）

### 阶段名称
阶段 6 - projection_mode 分支的最小梯度数值一致性验证

### 本轮目标
- 仅做验证，不新增功能。
- 仅验证投影直接相关梯度链：
  - 2D mean 对 3D mean 梯度
  - 协方差投影相关梯度
  - orthographic 下 clamp 分段导数行为

### 修改文件清单
- stage_f_gradient_check.py

### 每个文件改了什么
- stage_f_gradient_check.py
  - 新增独立验证脚本，直接调用 `GaussianRasterizer`，不进入训练主流程。
  - 构建最小输入规模：1 个 Gaussian、9x9 图像、3x3 patch loss。
  - 对比 analytic gradient（autograd/backward）与 numeric gradient（中心差分）。
  - 覆盖 3 个最小案例：
    - perspective_minimal
    - orthographic_minimal
    - orthographic_clamp_x（验证 clamp 分段导数）
  - 输出每个指标的 analytic / numeric / abs_err / rel_err 以及 PASS/FAIL 总结。

### 为什么这样改
在 Stage E 后先做最小数值对照，可以在不扩大改动范围的情况下验证 forward/backward 分支一致性，降低后续阶段风险。

### 仍然是临时/占位方案的地方
- 当前是最小局部验证，不覆盖大规模场景、完整训练分布与多高斯相互遮挡。
- orthographic_clamp 案例目前落在“完全平坦区”，可继续补一个接近阈值的边界案例做敏感性检查。

### 验证方式与结果
- 脚本执行命令：
  - `python stage_f_gradient_check.py`
- 结果：`[SUMMARY] stage_f gradient check = PASS`
- 关键数值：
  - perspective_minimal：max_abs_err=8.030e-04，max_rel_err=6.333e-03
  - orthographic_minimal：max_abs_err=3.152e-04，max_rel_err=1.439e-04
  - orthographic_clamp_x：各项梯度为 0，clamp x 分段导数检查通过

### 本轮核心结论
当前 `projection_mode` 下，最小投影相关梯度链在数值上与解析梯度基本一致，Stage F 验证目标达成。

### 下一步建议
保持“只验证不扩展”的策略，可在下一小轮仅新增 1~2 个阈值附近样本，检查 clamp 边界附近数值稳定性。

---

## 当前总体状态
- 阶段 0：完成。
- 阶段 1：完成。
- 阶段 2：完成（临时兼容补丁）。
- 阶段 3：完成（A+B+C 参数传递链完成）。
- 阶段 4：完成（forward 前向分支已启用，backward 未改）。
- 阶段 5：完成（backward 最小投影梯度链与 forward 分支对齐）。
- 阶段 6：完成（最小梯度数值对照验证通过）。

## 后续更新提醒
后续每个阶段都需要同步更新本文件，且必须包含：
- 实际改动文件与函数边界。
- 验证命令与结果。
- 临时方案是否已替换。
- 下一阶段依赖关系。



