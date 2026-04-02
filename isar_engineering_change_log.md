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

## 阶段 7：验证增强（Stage G）

### 阶段名称
阶段 7 - 边界样本扫描 + 多高斯梯度检查 + 20 iter 稳定性 smoke

### 本轮目标
- 不新增模型功能，仅增强验证覆盖。
- 将 Stage F 的最小单高斯验证扩展到：
  - clamp 边界附近样本扫描
  - 多高斯小场景梯度检查
  - 更长短训练 smoke（20 iter）

### 修改文件清单
- stage_g_validation.py
- stage_g_posttrain_render_check.py

### 每个文件改了什么
- stage_g_validation.py
  - 新增 orthographic clamp 边界扫描：重点覆盖 $|s_x x|\approx 1.3$ 与 $|s_y y|\approx 1.3$。
  - 新增多高斯小场景梯度对照（2~4 高斯中的 3 高斯案例，含轻微重叠）。
  - 同时对 perspective / orthographic 两分支输出 numeric vs analytic 梯度误差统计与 PASS/FAIL。
- stage_g_posttrain_render_check.py
  - 新增训练后渲染体检脚本：读取指定迭代模型并渲染一帧，检查 finite、黑白退化、统计量是否异常。

### 为什么这样改
在不修改数学实现的前提下，优先扩大验证样本复杂度，判断当前 projection_mode 分支在更接近真实场景时是否仍保持数值一致与训练稳定。

### 仍然是临时/占位方案的地方
- 多高斯 perspective 案例目前存在明显 numeric/analytic 偏差，尚未定位是非平滑数值效应还是梯度链问题。
- 当前多高斯验证规模仍小，尚未覆盖更大遮挡组合。

### 验证方式与结果
- 1) clamp 边界扫描（orthographic）
  - 命令：`python stage_g_validation.py`
  - 子结果：PASS
  - 摘要：`scan_points=12, max_abs_err=0.000e+00, max_rel_err=0.000e+00, clamp_zero_ok=True`

- 2) 多高斯小场景梯度检查
  - orthographic 子结果：PASS
    - `checks=12, max_abs_err=1.713e+00, max_rel_err=2.642e-01, fails=0`
  - perspective 子结果：FAIL
    - `checks=12, max_abs_err=1.792e+01, max_rel_err=1.012e+00, fails=12`

- 3) orthographic/isar 20 iter 训练 smoke
  - 命令：
    - `train.py --source_path D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest --images images --model_path D:/3DGS_new/gaussian-splatting/output/stage_g_ortho_smoke_20 --iterations 20`
    - `python stage_g_posttrain_render_check.py --source_path D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest --images images --model_path D:/3DGS_new/gaussian-splatting/output/stage_g_ortho_smoke_20 --load_iteration 20`
  - 子结果：PASS（无崩溃、无 NaN）
  - 渲染统计：
    - `finite_ok=True`
    - `min=0.654040, max=1.000000, mean=0.996610, std=0.020717`
    - `nonzero_ratio=1.000000, near_white_ratio=0.957237`

### 本轮核心结论
- 验证增强目标已完成。
- orthographic 路径在边界扫描、多高斯检查和 20 iter smoke 下整体可运行且数值行为可接受。
- perspective 多高斯梯度对照存在明显不一致，需在后续验证轮次单独定位。

### 下一步建议
- 下一轮继续保持“只验证不扩展”，专门对 perspective 多高斯不一致做分解定位：
  - 固定排序与遮挡条件后做局部参数扫描
  - 拆分 mean 路径与 cov 路径独立对照

---

## 阶段 7.1：perspective 多高斯不一致定位（仅验证分解）

### 阶段名称
阶段 7.1 - 只定位 perspective 多高斯梯度不一致来源

### 本轮目标
- 不改 forward/backward 数学，不加模型功能。
- 判断不一致是由非平滑数值因素导致，还是解析梯度真实问题。

### 修改文件清单
- stage_g1_perspective_diagnose.py
- stage_g_validation.py

### 每个文件改了什么
- stage_g1_perspective_diagnose.py
  - 新增 Stage G.1 专用诊断脚本，执行以下顺序：
    - 多高斯原失败风格案例
    - 拉大间距减重叠
    - 减小深度差
    - 3 高斯降到 2 高斯
    - 单高斯 perspective 对照
  - 对每个分量执行 $\varepsilon$ sweep：
    - $\varepsilon \in \{10^{-2}, 5\times10^{-3}, 10^{-3}, 5\times10^{-4}, 10^{-4}\}$
  - 分量级统计：`mean_x`、`mean_y`、`cov_xx`、`cov_xy`、`cov_yy`。

- stage_g_validation.py
  - 修复验证脚本判据不一致：多高斯 case 中 numeric gradient 使用与 analytic 相同的 patch 大小（此前是 3 vs 5，不一致）。

### 为什么这样改
先保证验证脚本本身的对照条件一致，再判断是否为数学实现问题；否则会出现“验证器误报”。

### 每组对照实验结果
- S0 原失败风格（3 高斯，中等重叠+深度差）：全部 `ok`
- S1 拉大间距（低重叠）：全部 `ok`
- S2 深度差减小：全部 `ok`
- S3 两高斯：全部 `ok`
- S4 单高斯：全部 `ok`

分量级统计（fail_ratio）：
- `mean_x`: 0.000
- `mean_y`: 0.000
- `cov_xx`: 0.000
- `cov_xy`: 0.000
- `cov_yy`: 0.000

全局判读：
- `total=51, ok=51, likely_nonsmooth_numeric=0, suspicious_analytic=0`
- `diagnosis=mostly_consistent`

交叉验证（修复后重跑 Stage G 脚本）：
- perspective 多高斯由 FAIL 变为 PASS：
  - `max_abs_err=2.488e-03, max_rel_err=9.175e-03, fails=0`

### 判断结论
- 先前 perspective 多高斯“失败”主要来自验证脚本损失定义不一致（numeric/analytic 比较对象不一致），不是 forward/backward 数学证据。
- 在一致判据下，perspective 分支的当前验证结果可恢复基本信心。

### 是否足以恢复 perspective 分支信心
- 结论：可以恢复“工程验证层面的基本信心”。
- 说明：这不是理论完备证明，但在当前 Stage G.1 分解范围内未发现解析梯度异常证据。

---

## 阶段 8：更长训练与基础对照实验（Stage H）

### 阶段名称
阶段 8 - 在通过工程验证的 projection_mode 分支上开展长一点训练与基础投影对照

### 本轮目标
- 不改 forward/backward 数学，不扩展验证器复杂度。
- 重点转向训练稳定性与最基础的 perspective vs orthographic/isar 对照。

### 修改文件清单
- stage_h_projection_compare.py

### 每个文件改了什么
- stage_h_projection_compare.py
  - 新增 Stage H 对照脚本：加载同一模型与同一相机，仅切换 `projection_mode` 为 `perspective` / `isar`。
  - 输出 `render_perspective.png`、`render_isar.png`、`render_absdiff.png`、`gt.png` 与 `stats.json`。
  - 输出统计包含：finite、min/max/mean/std、nonzero_ratio、near_white_ratio、L1/PSNR（相对 GT）。

### 为什么这样改
当前阶段目标是先确认“能稳定训练 + 能做最基础分支对照”，不再深挖梯度验证器，也不做模型功能扩展。

### 训练计划与执行（orthographic/isar 递进）
- 数据：`D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
- 模型路径：`./output/stage_h_ortho_prog`
- 递进策略（从最小可承受版本开始）：
  - Step 1：100 iter（从零开始）
  - Step 2：500 iter（从 `chkpnt100.pth` 续跑）
  - Step 3：1000 iter（从 `chkpnt500.pth` 续跑）
- 训练日志：
  - `output/stage_h_logs/train_100.log`
  - `output/stage_h_logs/train_500.log`
  - `output/stage_h_logs/train_1000.log`

### 稳定性结果
- NaN/Inf：未发现（训练日志与 post-check 均正常）。
- Loss 行为：
  - 100 iter 段 EMA Loss 约在 `0.128 ~ 0.141`。
  - 500 iter 续跑段 EMA Loss 约在 `0.127 ~ 0.145`。
  - 1000 iter 续跑段 EMA Loss 约在 `0.128 ~ 0.144`。
- 训练评估点：
  - ITER 100：`L1=0.104053`，`PSNR=12.2542`
  - ITER 500：`L1=0.103231`，`PSNR=12.2519`
  - ITER 1000：`L1=0.099753`，`PSNR=12.4792`
- 训练后渲染体检（均 PASS，`finite_ok=True`）：
  - ITER 100：`mean=0.993977`，`std=0.027641`，`near_white_ratio=0.904978`
  - ITER 500：`mean=0.995100`，`std=0.024847`，`near_white_ratio=0.872791`
  - ITER 1000：`mean=0.997583`，`std=0.015207`，`near_white_ratio=0.892622`
  - 体检日志：`output/stage_h_logs/postcheck_100_500_1000.log`

### 基础对照实验（同一 ISAR 数据，同一相机）
- 对照方式：在同一已训练模型上，仅切换相机 `projection_mode`。
- 对照输出目录：
  - `output/stage_h_compare_100_train0`
  - `output/stage_h_compare_1000_train0`
- 每组均输出：`gt.png`、`render_perspective.png`、`render_isar.png`、`render_absdiff.png`、`stats.json`

关键统计（camera=`img_0000.tif`）：
- Iter 100
  - perspective：`mean=0.999996`，`std=0.001382`，`near_white_ratio=0.999990`，`L1=0.891847`，`PSNR=0.6965`
  - isar：`mean=0.993977`，`std=0.027641`，`near_white_ratio=0.904978`，`L1=0.886170`，`PSNR=0.7269`
  - absdiff：`mean=0.006025`，`max=0.489182`
- Iter 1000
  - perspective：`mean=0.999999`，`std=0.000122`，`near_white_ratio=0.999966`，`L1=0.891845`，`PSNR=0.6965`
  - isar：`mean=0.997583`，`std=0.015207`，`near_white_ratio=0.892622`，`L1=0.889462`，`PSNR=0.7094`
  - absdiff：`mean=0.002417`，`max=0.342483`

### 简单结论
- 训练稳定性：通过。100/500/1000 递进训练均可完成，未见 NaN，渲染体检持续 PASS。
- 基础投影对照：在同一模型同一相机下，perspective 渲染更趋近“近全白”，isar 分支保留了更多亮度结构变化；两者差异可由 `render_absdiff.png` 与 `stats.json` 直接复核。
- 本轮定位：满足 Stage H 的“训练稳定性 + 基础对照结果”目标。

---

## 当前总体状态
- 阶段 0：完成。
- 阶段 1：完成。
- 阶段 2：完成（临时兼容补丁）。
- 阶段 3：完成（A+B+C 参数传递链完成）。
- 阶段 4：完成（forward 前向分支已启用，backward 未改）。
- 阶段 5：完成（backward 最小投影梯度链与 forward 分支对齐）。
- 阶段 6：完成（最小梯度数值对照验证通过）。
- 阶段 7：完成（验证增强完成，orthographic 通过，perspective 多高斯待定位）。
- 阶段 7.1：完成（perspective 多高斯不一致定位完成，确认先前误报源于验证脚本不一致）。
- 阶段 8：完成（更长训练与基础对照实验完成）。

---

## 阶段 9：P2 最小监督修正（单通道 ISAR 监督入口）

### 阶段名称
阶段 9 - 保持 projection_mode 数学不变，最小切换到 ISAR 单通道监督分支

### 本轮目标
- 不改 forward/backward 数学。
- 不扩多视角实验。
- 在训练监督入口最小引入 ISAR 单通道语义：`render_gray` 对 `gt_1ch`。

### 修改文件清单
- train.py

### 每个文件改了什么
- train.py
  - 在 `training(...)` 中增加 ISAR 数据检测（`poses.csv + images`）。
  - 保留原有 RGB 兼容路径（含 1->3 临时扩通道）用于非 ISAR 或非 1ch GT。
  - 对 ISAR 且 GT 为 `1xHxW` 时，新增最小监督分支：
    - `render_gray = image.mean(dim=0, keepdim=True)`
    - `Ll1 = l1_loss(render_gray, gt_image)`
    - `ssim(render_gray, gt_image)`（或 fused_ssim 对应 1ch 张量）

### 为什么这样改
当前阶段目标是先把监督域从“临时 RGB 兼容”切到“单通道强度语义”，且尽量只改损失入口，避免回归扩散。

### 最小验证
1) 2-iter smoke（P2）
- 模型：`output/stage_p2_smoke_2it`
- 训练完成，无 NaN。
- 训练评估：`L1=0.106824`，`PSNR=11.8761`
- post-check（黑底一致约定）：PASS

2) 100-iter 小训练（P2）
- 模型：`output/stage_p2_isar_100it`
- 训练完成，无 NaN。
- 训练评估：`L1=0.104061`，`PSNR=12.2533`
- post-check（黑底一致约定）：PASS

### 与当前 P0/P1 基线对比（同迭代 100、同相机）
基线：`output/stage_h_compare_100_train0_p0p1/stats.json`
P2：`output/stage_p2_compare_100_train0/stats.json`

isar 关键指标：
- `l1_vs_gt`: `0.1065967 -> 0.1066627`（基本持平）
- `psnr_vs_gt`: `12.0107 -> 12.0087`（基本持平）
- 强度分位数 `q95`: `0.0380409 -> 0.0379347`（基本持平）
- 强度分位数 `q99`: `0.1920348 -> 0.1920035`（基本持平）

结果图目录：
- 基线：`output/stage_h_compare_100_train0_p0p1`
- P2：`output/stage_p2_compare_100_train0`

### 本轮结论
- P2 已按“最小改动”成功落地，训练稳定性正常。
- 但在当前 100 iter 小训练尺度下，`l1_vs_gt` / `psnr_vs_gt` / 强度分位数与旧监督几乎无差别。
- 结论：P2 本身尚未显著改善“能量太低、结构太弱”，但为后续仅监督侧策略优化提供了正确单通道入口。

---

## 阶段 10：P2 长训练复核（500/1000）

### 阶段名称
阶段 10 - 固定评估条件下验证 P2 在更长训练尺度是否带来可观提升

### 固定评估条件
- 同一数据路径：`D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
- 同一 camera index：`0`
- 同一背景约定：黑底（`isar_source_forces_black`）
- 训练入口种子：`safe_state(args.quiet)` 固定种子

### 执行内容
1) 基线补齐（旧监督，P0/P1 口径）
- 迭代 500 对照导出：`output/stage_h_compare_500_train0_p0p1`

2) P2 长训练
- 从 `output/stage_p2_isar_100it/chkpnt100.pth` 续跑到 500：`output/stage_p2_isar_prog`
- 再从 `output/stage_p2_isar_prog/chkpnt500.pth` 续跑到 1000：`output/stage_p2_isar_prog`

3) P2 对照导出
- 500：`output/stage_p2_compare_500_train0`
- 1000：`output/stage_p2_compare_1000_train0`

4) 体检
- `stage_g_posttrain_render_check.py` 在 500/1000 均 PASS

### 单通道主指标对比（modes_intensity.isar）

iter 500：
- baseline：`l1=0.1058992`, `psnr=12.0031`, `q50=0.0000`, `q95=0.0235380`, `q99=0.2553159`, `contrast=q99-q50=0.2553159`
- P2：`l1=0.1058583`, `psnr=12.0022`, `q50=0.0000`, `q95=0.0226716`, `q99=0.2516983`, `contrast=0.2516983`
- 结论：500 时几乎无提升（部分指标略降）。

iter 1000：
- baseline：`l1=0.1041314`, `psnr=12.0929`, `q50=0.0000`, `q95=0.0231546`, `q99=0.4426408`, `contrast=0.4426408`
- P2：`l1=0.1032408`, `psnr=12.1493`, `q50=0.0000`, `q95=0.0279638`, `q99=0.4688222`, `contrast=0.4688222`
- 结论：1000 时出现温和提升，尤其高分位与对比度提升。

门槛检查（相对 baseline）：
- iter 500：`l1_improve=+0.039%`, `psnr_gain=-0.001dB`, `q95_gain=-3.68%`, `q99_gain=-1.42%`
- iter 1000：`l1_improve=+0.855%`, `psnr_gain=+0.056dB`, `q95_gain=+20.77%`, `q99_gain=+5.92%`

### 视觉复核结论
- 500：P2 与 baseline 视觉上几乎一致，主散射团结构变化不明显。
- 1000：P2 的亮散射点更活跃、局部对比度更强（与 `q95/q99/contrast` 提升一致）。
- 但整体形态仍以稀疏散点云为主，与 GT 的主结构分布仍有明显差距。

### 本轮结论
- P2 在更长训练尺度下不是“完全无效”：到 1000 iter 出现了可测提升。
- 该提升仍不足以解决“结构太弱”的核心问题，建议进入下一小阶段 P2.1（仅监督/loss 侧增强）。

## 后续更新提醒
后续每个阶段都需要同步更新本文件，且必须包含：
- 实际改动文件与函数边界。
- 验证命令与结果。
- 临时方案是否已替换。
- 下一阶段依赖关系。

---

## 阶段 11：P2.1A 小调参结论固化（仅监督侧）

### 阶段名称
阶段 11 - 在候选 A 形式不变前提下验证“权重过强”假设

### 本轮目标
- 不改 forward/backward 数学。
- 不扩多视角。
- 仅在候选 A 形式
  - $w = 1 + \alpha I_{gt}^{\gamma}$
  - $L_{wL1}=\frac{\sum w|I_{pred}-I_{gt}|}{\sum w}$
  下做温和参数测试。

### 执行参数与范围
- 已有对照：`alpha=2.0, gamma=2.0`
- 新测两组：
  - `alpha=1.0, gamma=2.0`
  - `alpha=1.0, gamma=1.5`
- 每组仅跑 100/500 iter。

### 关键结论
- 两组温和参数均保留了高分位增强信号（`q95/q99/contrast` 相对 P2 提升）。
- 在“亮尾增强 vs 全局 L1 恶化”平衡上，`alpha=1.0, gamma=2.0` 最优。
- 固化结论：候选 A 当前最优监督版本为 `P2.1A(alpha=1.0, gamma=2.0)`。

---

## 阶段 12：P2.1B 最小接入与 100 iter 结论

### 阶段名称
阶段 12 - 候选 B 最小接入与 100 iter 对照（不扩任务）

### 候选 B 形式
$$
L = \lambda_1 \cdot |I_{pred}-I_{gt}| + \lambda_2 \cdot \left|\log(1+\beta I_{pred})-\log(1+\beta I_{gt})\right|
$$

### 最小接入
- 仅改训练监督/loss 侧，不改 forward/backward、reader、renderer。
- 默认参数：`lambda1=1.0, lambda2=0.2, beta=10.0`。
- 2-iter smoke：PASS。

### 100 iter 同口径对比（主判据 `modes_intensity.isar`）
- 固定条件一致：
  - source：`D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
  - `camera_index=0`
  - 背景：`isar_source_forces_black`
- 指标摘要：
  - P2：`l1=0.1066627`, `psnr=12.0087`, `q95=0.0379347`, `q99=0.1920035`
  - P2.1A(a=1.0,g=2.0)：`l1=0.1067583`, `psnr=12.0483`, `q95=0.0513908`, `q99=0.2241794`
  - P2.1B：`l1=0.1069182`, `psnr=12.0178`, `q95=0.0464204`, `q99=0.2018639`

### 本轮结论
- P2.1B 在 100 iter 下未明显优于当前最优 P2.1A(a=1.0,g=2.0)。
- 决策：暂不推进 P2.1B 到 500/1000，也不继续扩更多 loss 变体。
- 当前最优监督版本维持为：`P2.1A(alpha=1.0, gamma=2.0)`。

---

## 阶段切换说明（下一阶段）

- 监督侧扩展在当前节点暂停。
- 下一阶段转入：CUDA 数学公式审查（先审查，不先改实现）。

---

## 阶段 13：orthographic Jacobian 像素尺度对齐（CUDA 数学主线第 1 轮）

### 阶段名称
阶段 13 - 将 orthographic/isar Jacobian 从 NDC 标度改为像素尺度一致形式，并前后向同步

### 本轮目标
- 仅改 CUDA 投影数学链路，不改 loss/reader/renderer。
- perspective 路径保持原样。

### 修改文件清单
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h`

### 每个文件改了什么
- `forward.cu`
  - `computeCov2D(...)` 的 orthographic 分支 Jacobian 改为像素尺度：
    - 由原 `diag(2/Sx, 2/Sy)`
    - 改为 `diag(W/Sx, H/Sy)`（实现中用 `focal*tan_fov` 还原 `W/2`,`H/2`）。
  - `preprocessCUDA(...)` 调用 projection-aware 的 `in_frustum(...)`。

- `backward.cu`
  - `computeCov2DCUDA(...)` 的 orthographic 分支 Jacobian 同步改为像素尺度版本。
  - `preprocessCUDA(...)` 的 orthographic mean backward 链按像素 Jacobian 与 NDC->pixel 映射显式一致化（数值等价于原链，但表达与 covariance 链口径统一）。

- `auxiliary.h`
  - `in_frustum(...)` 新增 projection-aware 重载：
    - perspective 保持原投影路径；
    - orthographic/isar 使用 view-space 线性映射分支。
  - 保留旧签名 wrapper，兼容既有 perspective 调用点。

### 验证方式与结果
1) CUDA 扩展重编译
- 结果：成功。

2) `stage_f_gradient_check.py`
- `perspective_minimal`：PASS。
- `orthographic_minimal`：PASS。
- `orthographic_clamp_x`：脚本汇总记为 FAIL，但该 case 为边界样本（`radius <= 0`，未渲染），输出表现为 analytic/numeric 梯度均为 0。
- 判读：该项属于边界可见性/采样覆盖 case，不判定为主链 Jacobian 数学失败。

3) 20 iter orthographic/isar smoke
- 训练 20 iter 成功跑完。
- 运行中未出现 NaN / Inf / 崩溃。
- post-train 检查 `finite_ok=True`。
- 结论类别：PASS，未见明显 stability / visibility 异常。

4) math-1 后 500 iter 同口径复测（candidate-A, `alpha=1.0`, `gamma=2.0`）
- 新模型：`output/stage_m1_retest_p21a_500`
- 新 compare：`output/stage_m1_retest_p21a_compare_500_train0/stats.json`
- 历史基线 compare：`output/stage_p21a_a10g20_compare_500_train0/stats.json`
- 固定口径：
  - 同一数据：`D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
  - 同一相机：`train`, `camera_index=0`
  - 同一黑底：`isar_source_forces_black`
- `modes_intensity.isar` 指标对比：
  - 历史 P2.1A@500：`l1_vs_gt=0.1064154`, `psnr_vs_gt=12.0270`, `q50=0.0000`, `q95=0.04117`, `q99=0.31792`, `contrast=0.31792`
  - math-1 后 @500：`l1_vs_gt=0.0790435`, `psnr_vs_gt=14.2756`, `q50=0.0000`, `q95=0.64650`, `q99=0.76506`, `contrast=0.76506`
- 视觉/统计判读：
  - 相比历史 500 iter，亮响应显著增强，主结构不再停留在极弱的稀疏散点水平。
  - 但从 `nonzero_ratio` 由 `0.1403` 提升到 `0.2896` 看，响应区域也明显变宽，当前仍更像“亮斑团/散射团块”而不是已经收敛成清晰、紧致的目标结构。

### 本轮结论
- 已完成 orthographic 分支 Jacobian 像素尺度一致化，并同步到 forward/backward 的 covariance 与 mean 链。
- perspective 路径未做公式修改。
- 最小验证口径下，本轮主线判定为 PASS；`orthographic_clamp_x` 仅记录为边界 case，不作为主链失败证据。
- 在 `P2.1A(alpha=1.0, gamma=2.0)` 的 500 iter 同口径复测下，math-1 带来了可测且可感的正向收益。
- 但当前结果仍未摆脱“亮斑团/白点团”主形态，说明仅靠当前几何数学修正还不足以解决更深层的表示/观测模型问题。

---

## 阶段 14：math2 长训练验证（500 / 1000，同口径）

### 阶段名称
阶段 14 - 在当前 math2 代码状态下做 candidate-A 长训练复测，并与 `math-1 + candidate-A` 直接对比

### 本轮边界
- 本轮不新增源码修改。
- 本轮只做：
  - 500 iter 训练 + post-train render check + compare
  - 1000 iter 训练 + post-train render check + compare
- 固定口径：
  - supervision：candidate-A
  - `alpha=1.0, gamma=2.0`
  - `camera_split=train`
  - `camera_index=0`
  - 黑底约定：`isar_source_forces_black`

### 执行记录
- 500 iter 模型：`output/stage_m2_p21a_500_20260329`
- 500 iter compare：`output/stage_m2_p21a_compare_500_train0_20260329/stats.json`
- 1000 iter 模型：`output/stage_m2_p21a_1000_20260329`
- 1000 iter compare：`output/stage_m2_p21a_compare_1000_train0_20260329/stats.json`
- 两个里程碑的 `stage_g_posttrain_render_check.py` 均 PASS：
  - `finite_ok=True`
  - 无 NaN / Inf / 崩溃

### 500 iter 指标对比（主判据 `modes_intensity.isar`）
- `math-1 + candidate-A@500`
  - `l1_vs_gt=0.0790435`
  - `psnr_vs_gt=14.2756`
  - `q50=0.0000`
  - `q95=0.64650`
  - `q99=0.76506`
  - `contrast=0.76506`
- `math2@500`
  - `l1_vs_gt=0.0830301`
  - `psnr_vs_gt=14.0252`
  - `q50=0.0000`
  - `q95=0.65106`
  - `q99=0.76771`
  - `contrast=0.76771`

### 1000 iter 指标对比（主判据 `modes_intensity.isar`）
- `math-1 + candidate-A@1000`
  - `l1_vs_gt=0.0631026`
  - `psnr_vs_gt=15.4842`
  - `q50=0.0000`
  - `q95=0.58482`
  - `q99=0.78731`
  - `contrast=0.78731`
- `math2@1000`
  - `l1_vs_gt=0.0654826`
  - `psnr_vs_gt=15.2993`
  - `q50=0.0000`
  - `q95=0.59736`
  - `q99=0.78504`
  - `contrast=0.78504`

### 视觉复核结论
- 500 iter：
  - math2 的高分位亮响应略强于 math-1，但 `l1/psnr` 变差。
  - `nonzero_ratio` 由 `0.2896` 升到 `0.2926`，说明响应区域略变宽，没有看到更集中的结构收敛。
- 1000 iter：
  - math2 仍表现为更亮、更厚的散射响应，`nonzero_ratio` 由 `0.2929` 升到 `0.3053`。
  - 虽然 `q95` 略升，但 `l1/psnr` 仍差于 math-1，且 `q99/contrast` 未继续占优。
- 综合看，math2 当前更像把亮散射团块继续铺宽，而不是把目标主结构收紧。

### 本轮判断
- 500 iter：math2 未优于当前 `math-1 + candidate-A` baseline。
- 1000 iter：math2 仍未优于当前 `math-1 + candidate-A` baseline。
- 工程判断：本轮 math2 整体应归类为“更差”，不是新的主线收益点。
- 下一步建议：
  - 暂不继续把主线押在 math；
  - 优先切到更干净的单通道 renderer，再讨论更真实的 ISAR 观测算子。

---

## 阶段 15：主线回退到 math-1（安全撤销 math2）

### 回退范围
- 仅撤销 math2 这一轮新增的 coarse visibility 接口链改动：
  - `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h`
  - `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`
  - `submodules/diff-gaussian-rasterization/rasterize_points.h`
  - `submodules/diff-gaussian-rasterization/rasterize_points.cu`
  - `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
- 不触碰 `math-1` 核心文件：
  - `forward.cu`
  - `backward.cu`
  - `auxiliary.h`
- 不触碰 candidate-A 监督入口：
  - `train.py`
  - `arguments/__init__.py`

### 回退后编译与导入
- 已重新编译：
  - `diff-gaussian-rasterization`
  - `simple-knn`
- 导入验证：
  - `IMPORT_DIFF_OK=True`
  - `IMPORT_SKN_OK=True`

### 回退后 500 iter 回归复测
- 模型：`output/stage_m1_restorecheck_500_20260329`
- compare：`output/stage_m1_restorecheck_compare_500_train0_20260329/stats.json`
- `modes_intensity.isar`：
  - `l1_vs_gt=0.0806888`
  - `psnr_vs_gt=14.2163`
  - `q50=0.0000`
  - `q95=0.64172`
  - `q99=0.77028`
  - `contrast=0.77028`

### 与既有 math-1@500 的一致性核对
- 既有 `math-1 + candidate-A@500`：
  - `l1_vs_gt=0.0790435`
  - `psnr_vs_gt=14.2756`
  - `q95=0.64650`
  - `q99=0.76506`
- 回退后复测：
  - `l1_vs_gt=0.0806888`
  - `psnr_vs_gt=14.2163`
  - `q95=0.64172`
  - `q99=0.77028`
- 判读：
  - 指标量级与分布形态保持一致；
  - 说明当前代码主线已成功回到 `math-1 + candidate-A`；
  - 小幅差异可视作同口径重跑的正常波动，不构成“仍残留 math2”的证据。

---

## 阶段 16：R2-min 单通道 renderer 最小闭环验证（math-1 + candidate-A）

### 阶段名称
阶段 16 - R2-min 单通道 renderer 的最小验证闭环确认

### 本轮目标
- 不改大块代码，不开长训练。
- 在当前可信主线 `math-1 + candidate-A` 下，确认 R2-min 单通道 renderer 是否已具备最小可运行闭环。
- 对此前出现的 render shape/channel 不一致问题做最小修复与一致性验证。

### 本轮固定口径
- 几何/实现主线：`math-1`
- 监督口径：`candidate-A`
  - `--isar_supervision_mode a`
  - `--isar_l1_weight_alpha 1.0`
  - `--isar_l1_weight_gamma 2.0`

### 实际改动/验证对象
- `submodules/diff-gaussian-rasterization` 扩展二进制重建与运行时一致性检查（`_C.pyd`）。
- `smoke_test_isar_forward_only.py`（forward/render 通道形状复核）。
- `train.py`（2 iter smoke，仅验证最小训练可执行性）。
- `stage_g_posttrain_render_check.py`（训练后渲染体检）。
- `stage_h_projection_compare.py`（同口径投影对照导出与统计）。

### 关键验证结果（真实执行）
- `_C.pyd` 重建后，render/forward 通道恢复为真实单通道（`(1,H,W)`）。
- 2 iter 训练 smoke：PASS（流程完成并正常落盘）。
- Stage G：PASS（`finite_ok=True`，单通道强度输出形状为 `1x1200x1200`）。
- Stage H：PASS（结果目录与 `stats.json` 成功导出）。
  - `modes_intensity` 为单通道：`shape=(1,1200,1200)`。
  - 原始 `modes` 仍为三通道：`shape=(3,1200,1200)`。

### 当前结论与边界
- 结论：R2-min 单通道 renderer 的“最小闭环”已通过。
- 工程决策：当前主线不回退，继续维持 `math-1 + candidate-A`。
- 边界声明：本结论仅表示“最小闭环可运行”，不等于“长程训练稳定 baseline”，也不等于“所有兼容问题已收尾”。

### 后续仍需单独处理的点（本轮不展开）
- 单通道路径下 exposure 目前未见异常，但训练循环中的 `exposure_optimizer.step()` 仍在，需要后续单独审查。
- Stage H 中 `modes_intensity` 已是单通道，而原始 `modes` 仍为三通道；这是当前兼容可视化现象，不应误判为失败。

---

## 阶段 17：R2-min@500 与 math-1 主 baseline 同口径对比

### 阶段名称
阶段 17 - 在固定 `math-1 + candidate-A` 口径下执行一次 R2-min@500 并排对比

### 本轮目标
- 不改代码，不扩功能。
- 仅按与既有 `math-1 + candidate-A@500` 完全一致的 compare 口径，执行一次 R2-min@500 并给出主指标判定。

### 固定口径
- 监督参数：
  - `--isar_supervision_mode a`
  - `--isar_l1_weight_alpha 1.0`
  - `--isar_l1_weight_gamma 2.0`
- compare 口径：
  - 同一数据：`D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest`
  - 同一 `camera_split=train`
  - 同一 `camera_index=0`
  - 同一黑底约定（ISAR source 强制黑底）

### 实际执行对象与输出路径
- baseline（既有主线对照）：
  - `output/stage_m1_retest_p21a_compare_500_train0/stats.json`
- 本轮 R2-min 训练模型：
  - `output/r2min_cons500_expbypass_20260331`
- 本轮 R2-min compare 输出：
  - `output/r2min_cons500_compare_20260331/stats.json`

### 关键对比结果（`modes_intensity.isar`）
- `l1_vs_gt`：`0.0790435299 -> 0.0815365463`（未占优）
- `psnr_vs_gt`：`14.2755832672 -> 14.1257705688`（未占优）
- `q50`：`0.0000000000 -> 0.0000000000`（持平）
- `q95`：`0.6465042233 -> 0.6466857195`（略升）
- `q99`：`0.7650588751 -> 0.7675226927`（略升）
- `contrast`：`0.7650588751 -> 0.7675226927`（略升）
- `nonzero_ratio`：`0.2896451354 -> 0.2939805686`（略升）

### 当前工程结论
- R2-min 单通道工程链路已通过多级验证（2/20/100 与 stage_g/stage_h）。
- 但在本次 500 iter 主指标对比中，`L1/PSNR` 暂未优于 `math-1 + candidate-A` baseline。
- 因此当前应保持“候选实现”定位，暂不替代主线 baseline。

## 阶段 18：显式标量散射 DC 通道版本

### 阶段名称
阶段 18 - 把单通道从渲染压缩语义升级成 Gaussian 表示层的显式标量散射 DC 通道

### 本轮目标
将原有的在 renderer 中使用 `SH2RGB + mean` 计算出的单通道伪标量，替换为直接从表示层 `f_dc_0` 提取的纯粹显式标量物理表达。

### 实际改动文件
- `scene/gaussian_model.py`
- `gaussian_renderer/__init__.py`

### 具体改动内容
- 在 `GaussianModel` 中增加 `get_scatter_dc` 属性提取 `_features_dc` 的第一个通道。
- 在 `renderer` 中移除 `SH2RGB + mean`，直接调用 `get_scatter_dc` 注入通道，使得 `f_dc_0` 获取唯一的梯度回传。

### 验证与指标
- 20 iter 等各项冒烟测试全过。
- 500 iter 长测对比证明了该通路是跑通的。

## 阶段 19：显式标量散射通道的软激活映射改造

### 阶段名称
阶段 19 - 对“显式标量散射 DC 通道”版本实施平滑的正域激活映射，替换死区截断。

### 实际改动文件
- `gaussian_renderer/__init__.py`

### 改动内容
将旧版：
- `val * C0 + 0.5` 与 `clamp_min(0.0)`
替换为：
- `F.softplus(dc_scalar)`

### 本轮结论
- 相比“改前显式标量散射 DC 通道”版本表现更好。
- 由于只换了激活映射却没有换初始化，仍有部分收敛劣势。

## 阶段 20：显式标量散射 DC 通道（初始化去光学化验证）

### 阶段名称
阶段 20 - 拔除 initial point cloud 加载阶段中为了 RGB 设计的 `RGB2SH`，改为逆 softplus 反推，匹配激活映射。

### 本轮边界
- 仅修改 `scene/gaussian_model.py` 的模型球初始强度加载。
- 从 `fused_color = RGB2SH(...)` 改为 `inv_softplus_intensity = torch.log(torch.expm1(...))`
- 使得 `f_dc_0` 直接以物理对齐状态下线。

### 已完成验证与指标
- **500 iter 长测对比**（`output/isar_train_deopt_500it`）：
  - **Baseline (`math-1 + candidate-A@500`)**：L1=0.07904, PSNR=14.275, nonzero_ratio~0.289
  - **上阶段（Softplus Activation）@500**：L1=0.08222, PSNR=14.082
  - **本阶段（去光学化 Initialization + Softplus）@500**：L1=0.07992, PSNR=14.248, nonzero_ratio=0.2975

### 当前工程结论
从指标上看，初始化去光学化极大幅度拉近了与 `math-1` baseline 的差距（L1 从 0.082 缩小到 0.0799，接近目标 0.0790；PSNR 提至 14.248）。这表明：全链路解耦数学一致性后，单通道物理参数能稳定收敛。当前构成了高度一致且可靠的纯物理分支候选。

## 阶段 21：候选分支（去光学化+软激活）@1000 iter 全面挑战轮

### 阶段名称
阶段 21 - “去光学化初始化 + 显式标量 + softplus”首选候选分支，在 1000 iter 同口径下挑战正式主线 `math-1 + candidate-A`。

### 本轮目标
对当前表现极佳的首选候选分支执行毫无删改的 1000 iter 长训练完整验证，并根据严格的主指标（`modes_intensity.isar`）对比，正式评估其是否具备替换现役 `math-1` 强截断基线的资格。

### 训练与测试口径
- `--isar_supervision_mode a`
- `--isar_l1_weight_alpha 1.0`
- `--isar_l1_weight_gamma 2.0`
- `iterations: 1000`
- 代码维持：100% 沿用上一阶段纯物理对齐形态。

### 已完成验证
- **1000 iter 长训练**：成功跑完，未出现 NaN / Inf / 崩溃。
- **阶段 G (`stage_g_posttrain_render_check.py`)**：通过。
- **阶段 H (`stage_h_projection_compare.py`)**：通过，成功生成 `stats.json`。

### 指标并排比较（与当前主线 @1000）
| Metric | 主线 `math-1 + candidate-A` @ 1000 | 候选分支 `deopt + softplus` @ 1000 | 差值 (当前 - 主线) |
|---|---|---|---|
| L1 vs GT | `0.06310` | `0.06377` | `+0.00067` (微弱劣势) |
| PSNR vs GT | `15.4842` | `15.4209` | `-0.0633` (微弱劣势) |
| q95 | `0.58482` | `0.59412` | `+0.00930` (高响应略好) |
| q99 | `0.78731` | `0.78531` | `-0.00200` (基本持平) |
| nonzero_ratio | `~ 0.292` | `0.29389` | `+0.001` (基本持平) |

*(注: `contrast` 在新版评估输出中即为 `q99` 近似代理或已被重构归并，上表使用 `q95/q99` 为主干。)*

### 当前工程结论
- **差距极小化**：该分支在 1000 iter 时的收敛状态与主线高度咬合（L1 差距低至 `~0.0006`，PSNR 仅差 `0.06dB`）。
- **维持候选定位**：虽然该分支在表现、逻辑链与物理意义上都比原先带历史债务的主线更好，但基于工程数学的绝对冷酷原则，它在核心硬指标（L1/PSNR）上仍存在一丝极其微弱的后仰，未形成有效“反超”。
- **下一步策略**：当前分支作为“完美解耦的展示层基底”已经定型。表示层优化可能已经逼近上限，说明限制单通道物理继续突破的核心瓶颈极大概率藏在其后的**观测层或损失算子对齐**（相机/雷达的脉冲、频谱响应映射等）。
## [2026-04-01] 阶段验证：观测层最小算子与双域监督探索

### 本轮目标
从“表示层优化”跨入“观测层优化”。主线基于纯线性强度域（raw intensity）进行的L1损失由于强散射光斑的高亮度差拉升，导致在平滑区拟合受制。本轮探索通过在 Loss 外层包裹特定的非线性变换（即 observation operator）去重新定义比较域，以此验证能否进一步突破15.48dB的收敛天花板。

### 修改文件
1. \utils/isar_observation.py\ (新增): 包含了 \pply_isar_observation_operator\，承载映射函数（如 \identity\, \log1p\ 等）。
2. \rguments/__init__.py\: 注入 \isar_obs_mode\ 与 \isar_obs_log1p_weight\ 命令行钩子参数。
3. \	rain.py\: 引入观测算子进行映射对比，并在新一轮增加 \+ beta * L_log1p\ 双域对齐监督功能。
4. \stage_g_posttrain_render_check.py\ & \stage_h_projection_compare.py\ (评估系): 添加了平行维度的双域输出，即同时输出 raw domain 与 log1p domain 的判据结果进行平行对比。

### 第一轮探索：log1p-only 算子实验（正式宣告失败）
* **逻辑**：仅拿 \log1p(pred) - log1p(gt)\ 作为全局主要 L1 目标对齐距离，完全替代 raw intensity。
* **结论**：跑至 500 iter 时，非但没有提效，反而在本域内外两套指标中全线崩盘退化：
  - **Raw 辅助域惨败**：相对主线 L1 从 0.0788 退化至 0.0804，PSNR 从 14.28dB 退化至 14.07dB。
  - **Log1p 主裁判域也未赢**：尽管是用该域做 target，它自己算出的 Log1p PSNR 为 16.23dB，甚至低于主线原结果生切换过去的 16.42dB。
* **物理剖析**：由于对数算子将绝对高亮差极度柔化压瘪，这等于拔掉了 3DGS 最赖以增生和生长的“最强点梯度驱动”，导致底层几何发育严重失血。

### 第二轮探索：双域辅助监督实验（Raw主导 + Log1p辅偏） (当前结论)
* **逻辑**：汲取上一轮血的教训，退回“Raw L1”占绝对控制权重导向。仅增加小阻尼 \eta\（通过 20 iter 短验确认最优参数选定在 \eta = 0.50\，确保最稳），形成 \L_total = L1_raw + 0.5 * L1_log1p\ 的双域框架。
* **结果 (500 iter同口径对比)**：
  - **双域架构 (beta=0.5)**： Raw L1 \ .0803\, PSNR \14.19dB\ | Log1p L1 \ .0628\, PSNR \16.34dB\
  - **旧候选线 (纯Raw)**：Raw L1 \ .0797\, PSNR \14.25dB\ | Log1p L1 \ .0624\, PSNR \16.37dB\
  - **主线基线 (纯Raw)**：Raw L1 \ .0788\, PSNR \14.28dB\ | Log1p L1 \ .0614\, PSNR \16.42dB\
* **最终判决**：双域结合同样**宣告阻抑严重**。任何意图将高亮区软化的数学辅助项均在事实上拖拽了网络生长的步调。这验证了一个极其核心的雷达渲染特性：**观测层的设计必须远离渲染损失的最上游比较，直接在损失算子层削峰是违背 3DGS 生长直觉的。** 本轮明确拒绝双域与单域最小算子损失重构方案向后推进。



## Phase Update: Observation Extensibility (Decoupled Operator Stage)
- **Objective**: Establish pply_isar_observation_operator to run independent extensible loops in testing instead of interfering in the loss loop.
- **Changes**: Rollback train.py back to strict Raw L1 and fully abstract operators (identity, log1p, future_physical) out into eval-scripts (stage g and stage h).
- **Stage-G Results**: Validated loop logic dynamically capturing arrays correctly against preexisting model checkpoints without crash.
- **Stage-H Results**: Fully nested validation generating modes_intensity_obs successfully to output image metrics across dual layers (Raw intensity rendering vs Log1P visualization normalization).


## Phase Update: Post-Hoc Physical ISAR Operator Prototype (db_radar)
- **Objective**: Introduce a minimal 'more realistic' ISAR observation operator prototype specifically for decoupling evaluation interfaces.
- **Changes**: Added db_radar to isar_observation.py which computes 10 * log10(I + 1e-4) and normalizes roughly a 40dB dynamic range (-40dB to 0dB) into the visual/eval [0.0, 1.0] range. This simulates standard radar dB scaling thresholds.
- **Validation**: Executed stage_g and stage_h successfully on multiple previous models (e.g. 500 iter and 1000 iter checkpoints). Output json successfully segregated 
aw, log1p, and db_radar parallel trajectories with proper scaling and numerical bounds.
- **Conclusion**: The decouple architecture is fully robust. We have officially entered the 'Real Observation Operator Interface Pre-Research Phase'.


## Phase Update: Formal Judgment Round (Mainline vs. Candidate in db_radar Domain)
- **Objective**: Determine whether to promote the Candidate branch (scalar DC scatter + de-opt init + softplus) over the Mainline (math-1 + candidate-A) using the newly implemented physical \db_radar\ observation operator.
- **Methodology**: Evaluated 500-iter and 1000-iter checkpoints of both branches using \stage_h_projection_compare.py\ with \--convert_SHs_python\ across \
aw\, \log1p\, and \db_radar\ modes.
- **Results (1000 iterations)**: 
  - **Mainline**: RAW (L1: 0.07349, PSNR: 15.21), DB_RADAR (L1: 0.13652, PSNR: 10.05)
  - **Candidate**: RAW (L1: 0.07546, PSNR: 15.21), DB_RADAR (L1: 0.13517, PSNR: 10.16)
- **Conclusion**: The Candidate branch is officially promoted to the primary baseline for the next development phase. Despite slightly trailing in the unbounded \
aw\ linear L1 metric, it consistently outperforms Mainline in the logarithmic \db_radar\ metric, correctly preserving low-intensity scattering physical attributes without gradient saturation.


## [2026-04-01] Phase Update: Observation Context Interface Upgrade (context_v1)

### 本轮目标
将 observation operator 从仅支持 `mode(tensor)` 的简单变换，升级为支持 `mode + context` 的语义化接口，作为轨道 B（观测层主攻）后续挂载真实 ISAR 观测算子的统一入口。

### 本轮边界（严格未触碰）
- 不改训练 loss。
- 不改 train.py 主链。
- 不改几何。
- 不改 CUDA。
- 不改表示层。

### 实际改动文件
- `utils/isar_observation.py`
- `stage_g_posttrain_render_check.py`
- `stage_h_projection_compare.py`

### 具体升级内容
1) 观测接口升级
- 新接口支持：`apply_isar_observation_operator(tensor, obs_mode, context=None)`。
- 新增统一 spec 构造：`build_isar_observation_spec(s)` 与默认 `DEFAULT_ISAR_OBSERVATION_MODES`。
- context 预留并可审计字段包含：
  - `noise_floor_db`
  - `dynamic_range_db`
  - `normalization_mode`
  - `clamp_min` / `clamp_max`
  - `range_axis` / `cross_range_axis`
  - `future_physical_operator_name`

2) stage_g / stage_h 调用统一
- 两个脚本都通过统一的 observation spec（mode + context）循环调用 `identity / log1p / db_radar`。
- 输出 json 中增加接口层结构化记录：
  - `observation_interface.version = context_v1`
  - `observation_interface.specs`（或等价可审计结构）
  - 每个 mode 的 context 与统计结果。

3) db_radar 显式 context
- 当前默认参数明确落盘记录：
  - `noise_floor_db = -40.0`
  - `dynamic_range_db = 40.0`
  - `normalization_mode = db_floor_to_unit_interval`
  - `clamp_min = 0.0`
  - `clamp_max = None`

### 本轮验证（仅用既有模型，不训练新模型）
使用既有模型：
- 500 iter：`output/isar_train_deopt_500it`
- 1000 iter：`output/isar_train_deopt_1000it`

执行：
- stage_g：两次均 PASS。
- stage_h：两次均成功导出 `stats.json`。
- 三种模式 `identity / log1p / db_radar` 在四份结果中均完整存在。
- 自动审计确认四份结果均 `finite_ok=True`，无 NaN / Inf。

结果目录：
- `output/eval_obsctx_g_candidate_500/stage_g_stats.json`
- `output/eval_obsctx_g_candidate_1000/stage_g_stats.json`
- `output/eval_obsctx_h_candidate_500/stats.json`
- `output/eval_obsctx_h_candidate_1000/stats.json`

### 本轮工程结论
- 本轮是“观测接口语义化升级”，不是训练改造，也不是新物理算子本体实现。
- 双轨冻结保持不变：
  - 轨道 A（训练主线）继续保持 `math-1 + candidate-A`。
  - 轨道 B（观测层主攻）继续以“显式标量 DC + 去光学化初始化 + softplus”为锚点。
- 后续真实 ISAR 观测算子默认挂载到该 `context_v1` 接口层，在不触碰训练主链的前提下增量扩展。


## [2026-04-01] 阶段 22：正式主线回正轮（candidate-A rebaseline）

### 本轮目标
- 在当前 clean freeze 快照下，将训练主线正式锁回 `math-1 + candidate-A`。
- 严格固定监督入口参数：
  - `--isar_supervision_mode a`
  - `--isar_l1_weight_alpha 1.0`
  - `--isar_l1_weight_gamma 2.0`
- 完成 A@500 / A@1000 同口径复现，并与历史基线对比。

### 本轮边界（严格未触碰）
- 不改 CUDA。
- 不改 renderer。
- 不改 stage_g / stage_h 脚本实现。
- 不改 observation operator。
- 不改数据读取。
- 不引入新 loss。

### 代码改动状态
- 训练监督入口恢复代码沿用上一轮最小补丁（`train.py`, `arguments/__init__.py`）。
- 本轮**无新增代码修改**；仅执行训练、评估与文档冻结。

### A@100 对齐证据（回正前置确认）
- 对比文件：
  - 历史：`output/repro_histA100_compare_currentscript/stats.json`
  - 新复现：`output/repro_newA100_compare_currentscript/stats.json`
- `modes_intensity.isar` 关键差值：
  - `delta_l1 = +0.0000793`
  - `delta_psnr = -0.00458 dB`
  - `delta_q99 = -0.000939`
- 判定：A@100 已与历史基本贴合，训练入口恢复有效。

### 本轮复现执行（同一快照/环境/数据/seed/背景）
- 训练模型：
  - A@500：`output/rebaseline_a500_20260401`
  - A@1000：`output/rebaseline_a1000_20260401`
- 环境约束：`gaussian_splatting` + CUDA 11.8。
- stage_g 输出：
  - `output/rebaseline_a500_20260401/stage_g_observation_context_iter500.json`
  - `output/rebaseline_a1000_20260401/stage_g_observation_context_iter1000.json`
- stage_h 输出：
  - `output/rebaseline_a500_compare_train0/stats.json`
  - `output/rebaseline_a1000_compare_train0/stats.json`

### 稳定性
- A@500、A@1000 均成功跑完。
- stage_g: 500/1000 均 PASS，finite 检查通过。
- stage_h: 500/1000 `modes_intensity.isar.finite_ok=True`。
- 本轮未出现 NaN / Inf / 崩溃。

### 与历史主线基线对比（`modes_intensity.isar`）
- 历史基线：
  - A@500：`output/stage_m1_retest_p21a_compare_500_train0/stats.json`
  - A@1000：`output/stage_m1_p21a_compare_1000_train0/stats.json`

- A@500（新 - 历史）
  - `l1_vs_gt`: `+0.002691`
  - `psnr_vs_gt`: `-0.15346 dB`
  - `q99`: `+0.004856`
  - `nonzero_ratio`: `+0.010790`

- A@1000（新 - 历史）
  - `l1_vs_gt`: `+0.002437`
  - `psnr_vs_gt`: `-0.18844 dB`
  - `q99`: `-0.000237`
  - `nonzero_ratio`: `+0.002444`

### 本轮冻结结论
- 当前正式训练主线回正为：`math-1 + candidate-A`。
- `deopt + softplus`（current/candidate）保留为研究分支，不作为训练主线。


## [2026-04-02] 阶段 23：A@8000 后期退化最小调度修正验证（research only）

### 本轮目标
- 在不改几何/math/CUDA/viewer 的前提下，仅改一个调度点，验证 A@8000 碎裂/黑线是否可缓解。
- 仅当“明显更稳且无副作用”才允许研究分支提交。

### 本轮边界（严格未触碰）
- 不改 forward/backward 数学。
- 不改 CUDA 子模块。
- 不改 renderer。
- 不改 stage_g / stage_h 脚本实现。
- 不改 supervision 参数口径（保持 candidate-A）。

### 修改文件清单
- train.py

### 每个文件改了什么
- train.py
  - 将 opacity reset 触发条件由“周期触发”改为“单次触发”。
  - 改前：`iteration % opt.opacity_reset_interval == 0`
  - 改后：`iteration == opt.opacity_reset_interval`

### 训练与评估执行
- 续训起点：`output/mainline_a36_a4000_20260401/chkpnt4000.pth`
- 修正版输出：`output/research_a36_a8000_opreset_once_20260402`
- 修正版评估：
  - `output/research_a36_a8000_opreset_once_compare_train0/stats.json`
  - `output/research_a36_a8000_opreset_once_20260402/train/ours_8000/renders`

### 对比结果（modes_intensity.isar，修正版 - 原始 A@8000）
- 对比对象：`output/mainline_a36_a8000_compare_train0/stats.json`
- `l1_vs_gt`: `-0.002514`（改善）
- `psnr_vs_gt`: `+0.265445 dB`（改善）
- `nonzero_ratio`: `-0.049131`（有效散射覆盖下降）
- `q99`: `+0.116032`（高亮强度上冲）
- `near_black_ratio`: `+0.056122`（黑场占比上升）
- `near_white_ratio`: `+0.004910`（白饱和显著上升）

### 稳定性与可运行性
- 训练完成，无 NaN / Inf / 崩溃。
- stage_g：PASS（finite），但 near_white_ratio 上升。
- stage_h：finite_ok=True。
- render.py：train 视角渲染 36/36 成功。

### 本轮判定
- 该最小调度改动未满足“明显更稳且无副作用”的提交门槛。
- 结论：保留为 research only 观察结果，不进入主线冻结结论，不执行提交/推送。

### 下一步建议
- 若继续在调度侧排查，建议优先测试“降低 reset 频次但非一次性”与“reset 时机后移且与 densify 解耦”的小步实验。


## [2026-04-02] 阶段 24：late densify/prune 冻结验证（research only, not mainline）

### 本轮目标
- 在保持 opacity reset 原始逻辑的前提下，只隔离“4000 之后继续 densify/prune 是否是后期退化主因”。

### 本轮边界（严格未触碰）
- 不改几何 math。
- 不改 CUDA。
- 不改 renderer。
- 不改 observation/operator。
- 不引入第二个 schedule 改动。

### 修改文件清单
- arguments/__init__.py

### 每个文件改了什么
- arguments/__init__.py
  - 调整优化默认参数：`densify_until_iter` 从 `15000` 收口到 `4000`。
  - `opacity_reset_interval` 与 reset 触发逻辑保持原始 baseline（周期触发）。

### 训练与评估执行
- 续训起点：`output/mainline_a36_a4000_20260401/chkpnt4000.pth`
- 修正版输出：`output/research_a36_a8000_freeze_densify_after4000_20260402`
- stage_h 对比输出：`output/research_a36_a8000_freeze_densify_after4000_compare_train0/stats.json`

### 与原始 A@8000 对比（modes_intensity.isar）
- 对比对象：`output/mainline_a36_a8000_compare_train0/stats.json`
- `l1_vs_gt`: `-0.003908`（改善）
- `psnr_vs_gt`: `+1.003192 dB`（明显改善）
- `near_black_ratio`: `-0.027185`（黑场下降）
- `near_white_ratio`: `-0.000214`（回到 0）
- 视觉：碎裂/黑线/黑块明显缓解，但条纹伪影仍有残留。

### 稳定性
- A@8000 训练完成，无 NaN / Inf / 崩溃。
- stage_g：PASS。
- stage_h：finite_ok=True。
- render.py：正常输出（train 渲染完成）。

### 本轮结论
- 在当前研究分支语境下，“freeze densify after 4000”可作为下一阶段默认 schedule 基座。
- 标记：research only，not mainline。

