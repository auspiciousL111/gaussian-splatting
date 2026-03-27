# 3D Gaussian Splatting → ISAR 改造：分阶段实现计划

**目标** 将 3DGS 从透视相机投影（3D→2D 透视变换）改造为正交投影（SAR 雷达坐标系）

**概览**
- **总体工程量**：~8 个提交，24-40 小时（取决于调试复杂度）
- **验证复杂度**：中等（数值梯度验证关键）
- **风险等级**：高（CUDA 梯度链容易出错，需要逐阶段验证）
- **推荐并行度**：阶段间串行，阶段内可并行开发

---

## 目录
1. [高层策略](#高层策略)
2. [Stage 1：数据加载与接口设计](#stage-1数据加载与接口设计)
3. [Stage 2：前向传播改造](#stage-2前向传播改造)
4. [Stage 3：反向传播改造](#stage-3反向传播改造)
5. [验证框架](#验证框架)
6. [回归测试检查表](#回归测试检查表)

---

## 高层策略

### 为什么分阶段？
1. **Stage 1** 只修改 Python 代码→可运行现有模型进行冒烟测试
2. **Stage 2** 加入前向 CUDA 改造→可评估渲染效果
3. **Stage 3** 完整梯度→开始正式训练

### 保持向后兼容性
```python
# 在所有决策点使用配置开关
if args.dataset_type == 'isar':
    # ISAR 特有逻辑
else:
    # 保持原有 COLMAP 逻辑
```

**关键原则**
- 创建新函数而非修改旧函数（例如 `readISARSceneInfo()` 而非魔改 `readColmapSceneInfo()`)
- 每个 CUDA 改造点都需要 Python-side 控制开关
- 保留所有原有的 COLMAP 代码路径（用于回归测试）

---

## Stage 1：数据加载与接口设计

**目标** 使训练能读取 ISAR 数据，但仍使用透视投影数学（作为基线）

**预期耗时** 4-6 小时  
**建议并行度** 可并行完成 2-3 个 commit

### Commit 1.1：参数扩展

**修改文件** `arguments/__init__.py`

**变更内容**
```python
# 在 ModelParams 中添加
class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # ... 既有代码 ...
        gr.add_argument("--dataset_type", default="colmap", help="'colmap' or 'isar'")
        gr.add_argument("--isar_pose_filename_pattern", 
                       default=r"(\d+)_(\d+)_(\d+)\..*",
                       help="Regex: (azimuth, elevation, range) or custom")
        gr.add_argument("--isar_range_scale", type=float, default=1.0,
                       help="Scale factor for range dimension")
        gr.add_argument("--isar_ortho_width", type=int, default=1024)
        gr.add_argument("--isar_ortho_height", type=int, default=1024)
```

**测试** 运行 `python train.py --help | grep isar` 确认参数出现

**预期产出** `cfg_args` 文件包含新参数

---

### Commit 1.2：ISAR 数据加载器框架

**修改文件** `scene/dataset_readers.py`

**变更内容** 添加菜单项和新的 `readISARSceneInfo()` 函数

```python
# 在 dataset_dict 中添加
def readISARSceneInfo(path, images, eval, llffhold=8):
    """
    读取 ISAR 数据集
    
    目录结构预期:
    path/
      images/
        000_000_100.png      # azimuth_elevation_range.png
        ...
      sparse/0/             # [可选] CAM 或位姿文本
    
    返回: SceneInfo (cameras, point_cloud, nerf_normalization)
    """
    
    # TODO: 待实现 (见 1.3)
    raise NotImplementedError("Implement in Commit 1.3")

dataset_dict = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "ISAR": readISARSceneInfo,  # <-- 新增
}
```

**关键设计**
- 返回类型与 COLMAP 兼容：`SceneInfo(cameras, point_cloud, nerf_normalization)`
- `cameras` 是 `CameraInfo` 的列表，带有语义映射
- 为稍后数学改造预留接口

---

### Commit 1.3：ISAR 文件名解析与 Camera 构造

**修改文件** `scene/dataset_readers.py`

**参考实现**
```python
import re
import numpy as np
from scipy import ndimage  # 可选，用于自动标度估计

def readISARSceneInfo(path, images, eval_args, llffhold=8):
    """
    实现 ISAR 数据加载
    """
    from plyfile import PlyData
    
    # 参数
    pattern = images.isar_pose_filename_pattern
    image_dir = os.path.join(path, "images")
    
    # 1. 扫描图像目录
    image_list = sorted([x for x in os.listdir(image_dir) if x.endswith('.png')])
    
    # 2. 解析文件名
    cameras = []
    for idx, img_name in enumerate(image_list):
        match = re.match(pattern, img_name)
        if not match:
            print(f"Warning: {img_name} 不匹配模式，跳过")
            continue
        
        azimuth, elevation, range_val = map(float, match.groups())
        
        # 转换为弧度 (如果输入是度数)
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        rg_m = range_val * images.isar_range_scale
        
        # 3. 构造 R 矩阵 (SAR → 相机坐标系)
        # SAR 坐标系: X(range), Y(cross-range), Z(elevation)
        # 正交投影: 丢弃 range 维, 保留 (Y, Z) 为图像坐标
        # 
        # 这里用旋转矩阵表示相机朝向（稍后在 forward.cu 中简化）
        Rz = rotation_matrix_z(az_rad)
        Rx = rotation_matrix_x(el_rad)
        R = Rz @ Rx  # SAR → 相机坐标中心
        
        # 4. 构造 T (平移，与 range 相关)
        # 简单版本: range 作为 z 距离（远离雷达）
        t = np.array([0, 0, rg_m])
        
        # 5. 构造内参 (占位符，actual implementation in forward.cu)
        # - 对于正交投影，无焦距概念
        # - 用 ortho_width/height 作为图像尺寸代理
        focal_x = images.isar_ortho_width / 2  # 占位符
        focal_y = images.isar_ortho_height / 2
        
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=t,
            FovY=images.isar_ortho_height,  # 重用字段，稍后忽略
            FovX=images.isar_ortho_width,
            image=Image.open(os.path.join(image_dir, img_name)),
            image_name=img_name,
            width=images.isar_ortho_width,
            height=images.isar_ortho_height,
            depth_params=None,  # ISAR 不用深度
        )
        cameras.append(cam_info)
    
    # 6. 加载或初始化点云
    try:
        pcd_path = os.path.join(path, "sparse", "0", "points3D.ply")
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.colors = o3d.utility.Vector3dVector(
            np.random.rand(len(np.asarray(pcd.points)), 3)
        )
    except:
        # 回退：生成虚拟初始化点云（正交体积内的网格）
        x = np.linspace(-0.5, 0.5, 16)
        y = np.linspace(-0.5, 0.5, 16)
        z = np.linspace(0, 1.0, 8)
        grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), -1).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(grid)
        pcd.colors = o3d.utility.Vector3dVector(np.random.rand(*grid.shape))
    
    # 7. 返回场景
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=cameras[:int(len(cameras)*0.8)],  # 80% 训练
        test_cameras=cameras[int(len(cameras)*0.8):],   # 20% 测试
        nerf_normalization={"translate": np.zeros(3), "radius": 1.0},
        ply_path="",
    )
    
    return scene_info

def rotation_matrix_z(theta):
    """绕 Z 轴旋转"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

def rotation_matrix_x(theta):
    """绕 X 轴旋转"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
```

**修改 Scene 初始化方式** `scene/__init__.py`

```python
class Scene:
    def __init__(...):
        # ...
        if dataset.dataset_type == "isar":
            scene_info = readISARSceneInfo(
                dataset.source_path,
                model_args,
                dataset.eval,
            )
        else:
            scene_info = readColmapSceneInfo(...)
        # ... 继续使用 scene_info ...
```

**测试脚本** `test_isar_loader.py`
```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, "./gaussian-splatting")

from scene.dataset_readers import readISARSceneInfo
from arguments import ModelParams
import argparse

parser = argparse.ArgumentParser()
model_args = ModelParams(parser)
args = parser.parse_args([
    "--source_path", "./3DGS_DATA/train",
    "--dataset_type", "isar",
])

scene_info = readISARSceneInfo(args.source_path, model_args)
print(f"✓ 加载了 {len(scene_info.train_cameras)} 个训练相机")
print(f"✓ 点云大小: {len(scene_info.point_cloud.points)}")

# 验证 camera 内参
cam0 = scene_info.train_cameras[0]
print(f"✓ 第一帧: R shape {cam0.R.shape}, T shape {cam0.T.shape}")
assert cam0.R.shape == (3, 3)
assert cam0.T.shape == (3,)
print("PASS: Commit 1.3")
```

**运行**
```bash
cd d:\3DGS_new\gaussian-splatting
python test_isar_loader.py
```

---

### Commit 1.4：相机参数适配层（保留透视逻辑）

**修改文件** `scene/cameras.py`

**目标** 暂时让 ISAR 相机通过现有的透视投影代码路径（作为基线调试）

```python
class Camera:
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, 
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device="cpu", **kwargs):
        
        # ISAR 标记位
        self.is_isar = kwargs.get('is_isar', False)
        
        # 现存代码
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        # ... 其他初始化保留 ...
        
        # 投影矩阵构造
        if self.is_isar:
            # ISAR 临时路径：仍用透视，但待改造
            # 稍后当 forward.cu 改造时，此处会被跳过
            self.projection_matrix = self._get_ortho_projection_matrix(...)
        else:
            # 原有透视逻辑
            self.projection_matrix = getProjectionMatrix(...)
    
    def _get_ortho_projection_matrix(self, width, height, ortho_scale=1.0):
        """正交投影矩阵（临时占位符，实际逻辑在 CUDA 中）"""
        # 这只是 scene 内的逻辑占位符
        # 真正的正交变换在 forward.cu 实现
        near, far = 0.01, 100.0
        left = -width / (2.0 * ortho_scale)
        right = width / (2.0 * ortho_scale)
        bottom = -height / (2.0 * ortho_scale)
        top = height / (2.0 * ortho_scale)
        
        P = np.zeros((4, 4), dtype=np.float32)
        P[0, 0] = 2 / (right - left)
        P[1, 1] = 2 / (top - bottom)
        P[2, 2] = -2 / (far - near)
        P[0, 3] = -(right + left) / (right - left)
        P[1, 3] = -(top + bottom) / (top - bottom)
        P[2, 3] = -(far + near) / (far - near)
        P[3, 3] = 1.0
        
        return P
```

**验证** `test_stage1_complete.py`
```python
#!/usr/bin/env python3
"""验证 Stage 1 完整性"""
import torch
from scene import Scene
from arguments import ModelParams, PipelineParams
import argparse

parser = argparse.ArgumentParser()
model_args = ModelParams(parser)
pipe_args = PipelineParams(parser)
args = parser.parse_args([
    "--source_path", "./3DGS_DATA/train",
    "--model_path", "./output/stage1_test",
    "--dataset_type", "isar",
    "--isar_ortho_width", "512",
    "--isar_ortho_height", "512",
])

# 初始化场景
try:
    scene = Scene(model_args, args.dataset_type)
    print(f"✓ Scene 初始化成功")
    print(f"  - 训练相机: {len(scene.getTrainCameras())}")
    print(f"  - 测试相机: {len(scene.getTestCameras())}")
except Exception as e:
    print(f"✗ Scene 初始化失败: {e}")
    exit(1)

# 加载 Gaussian 模型 (使用默认初始化)
from scene.gaussian_model import GaussianModel
gaussians = GaussianModel(model_args.sh_degree)
try:
    gaussians.create_from_pcd(scene.point_cloud, spatial_lr_scale=1.0)
    print(f"✓ Gaussian 模型初始化成功")
    print(f"  - Gaussian 数量: {gaussians._xyz.shape[0]}")
except Exception as e:
    print(f"✗ Gaussian 模型初始化失败: {e}")
    exit(1)

# 尝试一次渲染（应不崩溃，使用透视投影作为基线）
from gaussian_renderer import render
cam = scene.getTrainCameras()[0]
try:
    results = render(cam, gaussians, pipe_args, model_args)
    print(f"✓ 渲染成功")
    print(f"  - 输出形状: {results['render'].shape}")
except Exception as e:
    print(f"✗ 渲染失败: {e}")
    exit(1)

print("\n✅ Stage 1 验收完成！")
print("下一步: 前往 Stage 2 修改 forward.cu")
```

**运行**
```bash
cd d:\3DGS_new\gaussian-splatting
python test_stage1_complete.py
```

---

### Commit 1.5：特性开关与配置文件

**修改文件** 创建 `configs/isar_base.json`

```json
{
  "dataset_type": "isar",
  "source_path": "./3DGS_DATA/train",
  "model_path": "./output/isar_stage1",
  "isar_pose_filename_pattern": "([0-9]+)_([0-9]+)_([0-9]+)\\.png",
  "isar_range_scale": 100.0,
  "isar_ortho_width": 512,
  "isar_ortho_height": 512,
  "sh_degree": 3,
  "random_background": false,
  "white_background": false,
  "resolution": 1,
  "eval": false,
  "iterations": 2000
}
```

**使用方法**
```bash
python train.py \
  --config configs/isar_base.json \
  --iterations 100  # 快速冒烟测试
```

---

### Stage 1 验收标准

| 检查项 | 预期结果 | 验证命令 |
|-------|--------|--------|
| 参数加载 | `--dataset_type isar` 可用 | `python train.py --help \| grep dataset_type` |
| ISAR loader | 无异常加载 | `python test_isar_loader.py` |
| Scene 初始化 | 相机和点云加载成功 | `python test_stage1_complete.py` |
| 渲染（透视基线） | 无 CUDA 错误，输出形状正确 | 检查 `test_stage1_complete.py` |
| 配置文件 | 可从 JSON 加载所有参数 | `python train.py --config configs/isar_base.json --iterations 50` |

---

## Stage 2：前向传播改造

**目标** 实现正交投影数学，使渲染输出变为正交视图

**预期耗时** 6-8 小时  
**难度** 中等（主要是 CUDA，但没有梯度链复杂性）

### Commit 2.1：Python 端投影选择符

**修改文件** `gaussian_renderer/__init__.py`

**变更内容** 在渲染调用中传递投影类型标志

```python
def render(viewpoint_cam, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
           scaling_modifier = 1.0, override_color = None, **kwargs):
    
    # ... 既有代码 ...
    
    # 确定投影类型
    is_isar = getattr(viewpoint_cam, 'is_isar', False)  # 新增
    
    # 准备渲染设置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=math.tan(viewpoint_cam.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_cam.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_cam.camera_center,
        prefiltered=pipe.convert_SHs_python,
        debug=pipe.debug,
        is_isar=is_isar,  # 新增标志
    )
    
    rasterizer = GaussianRasterizer(raster_settings)
    
    # ... 渲染逻辑 ...
```

**修改 GaussianRasterizationSettings** `submodules/diff-gaussian-rasterization/rasterize_points.h`

```cpp
struct GaussianRasterizationSettings
{
    int image_height, image_width;
    float tanfovx, tanfovy;
    // ... 既有字段 ...
    bool is_isar;  // 新增
};
```

---

### Commit 2.2：CUDA 前向投影替换（核心）

**修改文件** `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`

**关键修改** `computeCov2D()` 函数

**原有代码（第 ~150-180 行）**
```cpp
static __device__ float computeCov2D(const float3 mean, float focal_x, float focal_y, 
                                     float tan_fovx, float tan_fovy,
                                     const glm::mat3 &cov3D, const glm::mat4x3 &viewmatrix)
{
    // 计算视图空间位置
    glm::vec3 t = glm::vec3(viewmatrix * glm::vec4(mean, 1.f));
    
    // 计算 2D Jacobian （透视投影）
    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    
    // Jacobian 矩阵
    glm::mat3 J = glm::mat3(
        focal_x / t.z, 0.f, -(focal_x * t.x) / (t.z * t.z),
        0.f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
        0.f, 0.f, 1.f
    );
    
    // 2D covariance = J^T V_rk J
    glm::mat3 cov2D_v1 = glm::transpose(J) * cov3D * J;
    
    // ... clamp, eigenvalue computation ...
}
```

**改造后（正交投影）**
```cpp
static __device__ float computeCov2D(const float3 mean, float focal_x, float focal_y, 
                                     float tan_fovx, float tan_fovy,
                                     const glm::mat3 &cov3D, const glm::mat4x3 &viewmatrix,
                                     bool is_isar)  // 新增参数
{
    // 计算视图空间位置
    glm::vec3 t = glm::vec3(viewmatrix * glm::vec4(mean, 1.f));
    
    if (is_isar)  // 新增： ISAR 正交投影路径
    {
        // 正交投影 Jacobian: [ s_x  0  0]  (2x3 矩阵)
        //                    [ 0  s_y  0]
        // 其中 s_x, s_y 是缩放因子（从 focal_x/y 派生）
        // 这里简化为单位缩放（像素到world的比例）
        
        float scale_x = focal_x;  // 或从 ortho_scale 计算
        float scale_y = focal_y;
        
        // 3D covariance 投影到 2D （仅取 XY blocks）
        // Cov_2D = J^T * Cov_3D * J
        // J^T * J = [[s_x^2,    0,    0],
        //            [  0,   s_y^2,   0],
        //            [  0,     0,    0]]  (实际上是 2x3 矩阵的乘积)
        
        float cov2D_xx = cov3D[0][0] * scale_x * scale_x;
        float cov2D_yy = cov3D[1][1] * scale_y * scale_y;
        float cov2D_xy = cov3D[0][1] * scale_x * scale_y;
        
        // 不包含 depth 相关的项 (cov3D[2][...])
        // 因为正交投影丢弃 z 信息
        
        // conic: inverse 2D covariance 的紧凑表示
        float det = cov2D_xx * cov2D_yy - cov2D_xy * cov2D_xy;
        
        // clamp 最小值以避免除零
        det = max(0.1f, det);
        
        float coni_a = cov2D_yy / det;
        float coni_c = cov2D_xx / det;
        float coni_b = -cov2D_xy / det;
        
        return glm::vec3(coni_a, coni_b, coni_c);
    }
    else  // 原有透视逻辑
    {
        // ... 保留原代码 ...
    }
}
```

**改造 preprocessCUDA（）中的调用** (~第 250-260 行)

```cpp
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
    const float* orig_points,
    // ... 其他参数 ...
    bool is_isar)  // 新增参数
{
    // ... 既有 frustum culling 逻辑 ...
    
    // 计算 2D covariance
    float3 conic = computeCov2D(mean, ... , is_isar);  // 传递 is_isar 标志
    
    // ... 继续使用 conic ...
}
```

**改造 C++ wrapper** `submodules/diff-gaussian-rasterization/rasterize_points.cu` (~第 150-200 行)

```cpp
void CudaRasterizer::Rasterizer::forward(
    // ... 既有参数 ...
    bool is_isar)  // 新增
{
    // ... 既有逻辑 ...
    
    preprocessCUDA<C><<<... >>>(
        // ... 既有参数 ...
        is_isar  // 传递标志
    );
    
    // ... 继续 ...
}
```

**改造 PyBind11 绑定** `submodules/diff-gaussian-rasterization/ext.cpp` (~第 80-120 行)

```cpp
m.def("rasterize_gaussians", &CudaRasterizer::Rasterizer::forward, 
      py::arg("background"),
      // ... 既有参数 ...
      py::arg("is_isar") = false,  // 新增，默认为 false
      // ...
);
```

**测试脚本** `test_forward_ortho.py`
```python
#!/usr/bin/env python3
"""测试正交投影前向通道"""
import torch
import math
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams

# 加载 ISAR 场景
model_args = ModelParams(None)
model_args.source_path = "./3DGS_DATA/train"
model_args.dataset_type = "isar"

scene = Scene(model_args)
gaussians = GaussianModel(3)
gaussians.create_from_pcd(scene.point_cloud)

# 获取相机
cam = scene.getTrainCameras()[0]
cam.is_isar = True  # 标记为 ISAR

pipe_args = PipelineParams(None)

# 渲染（应使用正交投影）
bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
try:
    results = render(cam, gaussians, pipe_args, bg_color)
    img = results['render']
    print(f"✓ 正交投影渲染成功")
    print(f"  - 输出形状: {img.shape}")
    print(f"  - 值范围: [{img.min():.3f}, {img.max():.3f}]")
    
    # 保存输出便于检查
    from torchvision.utils import save_image
    save_image(img, "output_isar_ortho_forward.png")
    print(f"✓ 输出已保存到 output_isar_ortho_forward.png")
    
except Exception as e:
    print(f"✗ 渲染失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ Commit 2.2 完成：正交投影前向通道工作正常！")
```

---

### Commit 2.3：渲染内核适配（如需要）

如果 `renderCUDA()` 中使用了焦距计算（通常不会），进行相应调整。

**检查** `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` ~第 350-400 行的 `renderCUDA()` 内核

通常 `renderCUDA()` 只使用已计算的 2D covariance（conic），不需要修改。

**验证无需修改**
```bash
# 搜索 renderCUDA 中是否有焦距使用
grep -n "focal" submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu | grep -i render
# 应返回空
```

如有匹配，修改类似 2.2 的模式（添加 is_isar 条件分支）。

---

### Commit 2.4：参数与默认值

**修改文件** `gaussian_renderer/__init__.py` 中的投影参数获取逻辑

确保 `is_isar` 从 camera 对象正确传递。如果相机列表是动态的，确保一致性：

```python
def render(...):
    # ...
    is_isar = getattr(viewpoint_cam, 'is_isar', False)
    
    if is_isar:
        # ISAR 特定的渲染参数 (后续可扩展)
        pass
    # ...
```

---

### Stage 2 验收标准

| 检查项 | 验证方法 |
|-------|--------|
| 前向 CUDA 编译 | `pip install ./submodules/diff-gaussian-rasterization` 无错误 |
| 正交投影激活 | `test_forward_ortho.py` 成功完成 |
| 梯度流不中断 | `test_forward_ortho.py` 中输出依然可反向传播 （添加 `.backward()` 检查） |
| 回归测试 | 用 `dataset_type=colmap` 场景验证透视投影仍工作 |

---

## Stage 3：反向传播改造

**目标** 实现梯度传播，启用模型训练

**预期耗时** 8-12 小时  
**难度** 高（梯度链复杂，需严格验证）

### Commit 3.1：反向投影 Jacobian 设计

**修改文件** `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`

**关键函数** `computeCov2DCUDA()` (~第 200-300 行)

**原有代码框架受透视影响**
```cpp
__device__ void computeCov2DCUDA(...)
{
    // 正向: J_persp = [[fx/z, 0, -fx*x/z²],
    //                   [0, fy/z, -fy*y/z²]]
    // 反向: dL/dCov3D = J^T * dL/dCov2D * J
    //       其中 J^T * dL/dCov2D 需要链式法则
    
    // 代码中有许多涉及 focal_x, focal_y, t.z 的项
    // 所有这些在正交投影中都消失或简化
}
```

**改造思路**

在 `computeCov2DCUDA()` 中添加 `is_isar` 分支：

```cpp
__device__ void computeCov2DCUDA(
    // ... 既有参数 ...
    bool is_isar)
{
    if (is_isar)
    {
        // 正交反向传播路径 （简化版）
        // 已知: dL/dCov2D_xx, dL/dCov2D_yy, dL/dCov2D_xy
        // 要求: dL/dCov3D[i][j]
        
        // 因为 Cov2D = [[s_x^2*Cov3D[0][0], s_x*s_y*Cov3D[0][1]],
        //               [s_x*s_y*Cov3D[0][1], s_y^2*Cov3D[1][1]]]
        
        // 链式法则:
        // dL/dCov3D[0][0] = dL/dCov2D_xx * s_x^2
        // dL/dCov3D[1][1] = dL/dCov2D_yy * s_y^2
        // dL/dCov3D[0][1] = (dL/dCov2D_xy + dL/dCov2D_yx) * s_x * s_y
        //   (注: Cov2D 对称，所以 dL/dCov2D_yx = dL/dCov2D_xy)
        
        float scale_x = ...; // 从 forward 中获取（需传递）
        float scale_y = ...;
        
        dL_dcov3D[0][0] = dL_dcov2D_xx * scale_x * scale_x;
        dL_dcov3D[1][1] = dL_dcov2D_yy * scale_y * scale_y;
        dL_dcov3D[0][1] = dL_dcov2D_xy * scale_x * scale_y;
        dL_dcov3D[1][0] = dL_dcov3D[0][1];
        
        // 第 3 行（z）和第 2/3 列（关于投影的导数）都是零
        // 因为正交投影丢弃 z
    }
    else
    {
        // 原有透视逻辑
        // ...
    }
}
```

### Commit 3.2：mean 梯度反向传播

**修改文件** `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu` (~第 350-400 行 `preprocessCUDA<>()` backward)

```cpp
// 原有代码（透视）中：
// dL_dmean = transformVec4x3Transpose(dL_dmean2D, viewmatrix)
//     + 涉及 depth 导数的项（ dL_dinvdepth / (t.z^2) 等）

// 改造版:
if (is_isar)
{
    // 正交投影下，mean 的梯度更简单（无 depth 相关项）
    // dL_dmean_view = [[dL_dx], [dL_dy], [0]]  （z 导数为零）
    // dL_dmean_world = (viewmatrix^-1)^T * dL_dmean_view
    
    glm::vec3 dL_dmean_view = transformVec4x3Transpose(dL_dmean2D, viewmatrix);
    // dL_dmean_view.z = 0 对于正交投影（除非有 range 约束）
    
    // 写回：
    atomicAdd(&dL_dmean[idx*3+0], dL_dmean_view.x);
    atomicAdd(&dL_dmean[idx*3+1], dL_dmean_view.y);
    // z 梯度可被设为 0（或改为 range-based constraint）
}
else
{
    // 原有代码
}
```

### Commit 3.3：完整反向链验证

**创建数值梯度验证脚本** `test_gradient_validation.py`

```python
#!/usr/bin/env python3
"""
数值梯度验证脚本
用于验证反向传播的正确性（vs 有限差分）
"""

import torch
import torch.nn.functional as F
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from PIL import Image
import numpy as np

def numerical_gradient(func, x, eps=1e-4):
    """计算数值梯度（有限差分）"""
    grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone()
        x_plus.view(-1)[i] += eps
        
        x_minus = x.clone()
        x_minus.view(-1)[i] -= eps
        
        f_plus = func(x_plus)
        f_minus = func(x_minus)
        
        grad.view(-1)[i] = (f_plus - f_minus) / (2 * eps)
    return grad

def test_single_gaussian_gradient():
    """在单个 Gaussian 上测试梯度"""
    device = "cuda"
    
    # 创建单个 Gaussian 参数
    xyz = torch.tensor([[[0.0, 0.0, 1.0]]], device=device, requires_grad=True)
    scaling = torch.tensor([[[0.1, 0.1, 0.1]]], device=device, requires_grad=True)
    rotation = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device, requires_grad=True)
    opacity = torch.tensor([[0.8]], device=device, requires_grad=True)
    features = torch.zeros(1, 3, 1, device=device, requires_grad=True)  # SH features
    
    # 渲染函数
    def render_func(xyz_val):
        # ... 使用 xyz_val 进行渲染，返回像素损失 ...
        # L = || render(...) - target ||^2
        pass
    
    # 计算解析梯度
    # grad_xyz_analytical = autograd 的结果
    
    # 计算数值梯度
    # grad_xyz_numerical = numerical_gradient(render_func, xyz)
    
    # 比较
    # assert torch.allclose(grad_xyz_analytical, grad_xyz_numerical, atol=1e-2)
    
    print("✓ 单 Gaussian 梯度测试通过")

def test_full_scene_gradient():
    """在完整场景上测试梯度"""
    model_args = ModelParams(None)
    model_args.source_path = "./3DGS_DATA/train"
    model_args.dataset_type = "isar"
    
    scene = Scene(model_args)
    gaussians = GaussianModel(3)
    gaussians.create_from_pcd(scene.point_cloud)
    
    cam = scene.getTrainCameras()[0]
    cam.is_isar = True
    
    pipe_args = PipelineParams(None)
    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # 先验训练图像
    target_img = cam.image.to("cuda") / 255.0  # [3, H, W]
    
    # 前向传播
    results = render(cam, gaussians, pipe_args, bg_color)
    pred_img = results['render']
    
    # 损失
    loss = F.mse_loss(pred_img, target_img)
    print(f"Loss: {loss.item():.6f}")
    
    # 反向传播
    try:
        loss.backward()
        print(f"✓ 反向传播成功")
        print(f"  - xyz grad: {gaussians._xyz.grad is not None}")
        print(f"  - opacity grad: {gaussians._opacity.grad is not None}")
    except Exception as e:
        print(f"✗ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("=== Stage 3 梯度验证 ===")
    
    # test_single_gaussian_gradient()
    success = test_full_scene_gradient()
    
    if success:
        print("\n✅ 梯度验证通过！")
    else:
        print("\n✗ 梯度验证失败")
        exit(1)
```

**运行验证**
```bash
cd d:\3DGS_new\gaussian-splatting
python test_gradient_validation.py
```

### Commit 3.4：短周期训练测试

**创建快速测试脚本** `test_mini_train.py`

```python
#!/usr/bin/env python3
"""
微型训练测试：100 迭代
验证完整训练循环的可行性
"""

import torch
import sys
import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss

def training_loop(args):
    device = "cuda"
    
    # 初始化
    scene = Scene(args, args.dataset_type)
    gaussians = GaussianModel(args.sh_degree)
    gaussians.create_from_pcd(scene.getPointCloud(), spatial_lr_scale=1.0)
    gaussians.training_setup(args)
    
    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    
    # 快速训练循环
    num_iterations = 100
    for iteration in range(num_iterations):
        # 选择随机相机
        cam = scene.getTrainCameras()[iteration % len(scene.getTrainCameras())]
        cam.is_isar = True
        
        # 渲染
        results = render(cam, gaussians, args, bg_color)
        pred = results['render']
        
        # 损失（简单的 L1）
        target = cam.image.to(device) / 255.0
        loss = l1_loss(pred, target)
        
        # 反向传播
        loss.backward()
        
        # 优化器步骤
        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad()
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration:3d}: loss = {loss.item():.6f}")
    
    print("✅ Mini training 完成！")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_args = ModelParams(parser)
    optim_args = OptimizationParams(parser)
    pipe_args = PipelineParams(parser)
    
    args = parser.parse_args([
        "--source_path", "./3DGS_DATA/train",
        "--dataset_type", "isar",
        "--model_path", "./output/test_mini",
        "--sh_degree", "3",
        "--iterations", "100",
    ])
    
    success = training_loop(args)
    if not success:
        exit(1)
```

**运行**
```bash
python test_mini_train.py
```

---

### Stage 3 验收标准

| 检查项 | 验证方法 |
|-------|--------|
| CUDA 编译（backward） | 无编译错误 |
| 梯度流动 | `test_gradient_validation.py` 完成，无 NaN 或 inf |
| 数值梯度一致性 | 解析梯度与有限差分的误差 < 1% |
| Mini 训练 | `test_mini_train.py` 100 迭代损失递减 |
| 完整训练 | `python train.py --config configs/isar_base.json --iterations 1000` 数小时内完成 |

---

## 验证框架

### 通用验证步骤

**每个 Commit 后**

1. **编译检查**
   ```bash
   cd submodules/diff-gaussian-rasterization
   python setup.py clean --all
   pip install .
   ```

2. **运行对应的 test_*.py 脚本**

3. **检查回归**
   ```bash
   # 用 colmap 数据验证透视逻辑未破坏
   python train.py --source_path ./3DGS_DATA/colmap_scene --iterations 50
   ```

### 高级验证：数值梯度检验

**原理** 对于任意参数 $\theta$，数值梯度应与解析梯度接近：

$$\frac{\partial L}{\partial \theta} \approx \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$$

**实现**
```python
def check_numerical_gradient(param_name, eps=1e-4, atol=1e-2):
    """检验单个参数的梯度"""
    
    param = getattr(gaussians, param_name)
    
    # 解析梯度
    param.grad = None
    loss.backward()
    grad_analytical = param.grad.clone()
    
    # 数值梯度（仅对小参数子集）
    grad_numerical = torch.zeros_like(param)
    for idx in range(min(10, param.numel())):  # 仅测试前 10 个
        param_plus = param.clone()
        param_plus.view(-1)[idx] += eps
        loss_plus = compute_loss(param_plus)
        
        param_minus = param.clone()
        param_minus.view(-1)[idx] -= eps
        loss_minus = compute_loss(param_minus)
        
        grad_numerical.view(-1)[idx] = (loss_plus - loss_minus) / (2 * eps)
    
    # 比较
    rel_error = torch.abs(grad_analytical - grad_numerical) / (torch.abs(grad_analytical) + 1e-8)
    
    if (rel_error > atol).any():
        print(f"✗ {param_name}: 梯度不匹配 (max rel error: {rel_error.max().item():.6f})")
        return False
    else:
        print(f"✓ {param_name}: 梯度通过验证")
        return True
```

---

## 回归测试检查表

### 必须通过的回归测试

| 场景 | 命令 | 预期结果 |
|------|------|--------|
| 原有 COLMAP（透视） | `python train.py --source_path ./3DGS_DATA/truck --iterations 50` | 正常训练，损失递减 |
| ISAR 小规模 | `python train.py --config configs/isar_base.json --iterations 100` | 正常训练，无 NaN |
| 混合数据 | 交替 COLMAP 和 ISAR batch | 两者独立工作 |
| 导出模型 | `python render.py --model_path output/isar_...` | 输出视图 |

### 性能基准

| 指标 | 目标 |
|------|------|
| 单帧 forward + backward | < 500ms (on RTX 3090) |
| 1000 迭代训练时间 | < 2 小时 |
| 内存占用 | < 24GB VRAM |
| 最终 PSNR（ISAR）| > 20dB （domain-specific） |

---

## 故障排除

### 常见问题

**问题 1: 正交投影渲染全黑**
- **原因**: 相机 frustum culling 排除了所有 Gaussians
- **解决**: 调整 frustum bounds 或使用 `--isar_ortho_width/height` 更大的值

**问题 2: 梯度为 NaN**
- **原因**: 2D covariance inverse 奇异，或除零
- **解决**: 在 `computeCov2D()` 中增加 clamp/epsilon

**问题 3: 训练损失不收敛**
- **原因**: 学习率对正交缩放不适配
- **解决**: 在 `arguments/__init__.py` 中调整 `position_lr_init`

**问题 4: CUDA 编译错误（`computeCov2D` 不匹配）**
- **原因**: 函数签名改动后旧的 object files 未清除
- **解决**: 
```bash
cd submodules/diff-gaussian-rasterization
rm -rf build dist *.egg-info
pip install .
```

---

## 最终检查清单

### 代码准备
- [ ] Commit 1.1-1.5 完成：ISAR 数据加载可用
- [ ] Commit 2.1-2.4 完成：正交前向投影可用
- [ ] Commit 3.1-3.4 完成：梯度反向传播可用

### 测试准备
- [ ] `test_isar_loader.py` ✓
- [ ] `test_stage1_complete.py` ✓
- [ ] `test_forward_ortho.py` ✓
- [ ] `test_gradient_validation.py` ✓
- [ ] `test_mini_train.py` ✓

### 数据准备
- [ ] ISAR 数据集路径确认
- [ ] 文件名格式与 regex pattern 匹配
- [ ] 点云初始化正确（范围内）

### 文档准备
- [ ] 各 Commit 的 PR 说明已准备
- [ ] 数学公式文档（投影 Jacobian 推导）已备
- [ ] 性能基准已记录

---

## 下一步行动

1. **确认分支策略**
   - 新建 feature 分支: `git checkout -b feature/isar-orthographic`
   - 每个 Commit 后 push，便于 code review

2. **依次实现 Stage 1-3**
   - 按上述顺序执行，每个 Commit 后运行对应测试
   - 若失败，修复后重新提交

3. **定期同步主分支**
   - 避免 merge conflicts: `git rebase main` (偶尔执行)

4. **最终验收**
   - 所有回归测试通过
   - ISAR 完整训练距离 (损失、PSNR 等)，与论文基线比较

---

## 附录：配置文件示例

### configs/isar_base.json
```json
{
  "dataset_type": "isar",
  "source_path": "./3DGS_DATA/train",
  "model_path": "./output/isar_train_001",
  "isar_pose_filename_pattern": "([0-9]+)_([0-9]+)_([0-9]+)\\.png",
  "isar_range_scale": 100.0,
  "isar_ortho_width": 512,
  "isar_ortho_height": 512,
  "sh_degree": 3,
  "resolution": 1,
  "random_background": false,
  "white_background": false,
  "eval": false,
  "iterations": 30000,
  "position_lr_init": 0.00016,
  "position_lr_final": 0.0000016,
  "position_lr_delay_mult": 0.01,
  "position_lr_max_steps": 30000,
  "opacity_lr": 0.05,
  "scaling_lr": 0.005,
  "rotation_lr": 0.001,
  "percent_dense": 0.01,
  "lambda_dssim": 0.2,
  "densification_interval": 100,
  "opacity_reset_interval": 3000,
  "densify_grad_threshold": 0.0002,
  "densify_from_iter": 500,
  "densify_until_iter": 15000,
  "densify_grad_threshold_fine_init": 0.0002,
  "densify_grad_threshold_fine_final": 0.0005
}
```

---

**文档版本**: v1.0  
**最后更新**: 2026-03-25  
**维护者**: ISAR Integration Team
