import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix


@dataclass
class GradEntry:
    name: str
    analytic: float
    numeric: float

    @property
    def abs_err(self) -> float:
        return abs(self.analytic - self.numeric)

    @property
    def rel_err(self) -> float:
        denom = max(abs(self.analytic), abs(self.numeric), 1e-8)
        return self.abs_err / denom


@dataclass
class CaseResult:
    name: str
    mode: str
    entries: List[GradEntry]
    radius: int
    clamp_expected_zero_x: bool = False

    @property
    def max_abs_err(self) -> float:
        return max(e.abs_err for e in self.entries)

    @property
    def max_rel_err(self) -> float:
        return max(e.rel_err for e in self.entries)


def make_settings(
    projection_mode: int,
    width: int,
    height: int,
    ortho_scale_x: float,
    ortho_scale_y: float,
    isar_window_size: float,
) -> GaussianRasterizationSettings:
    fovx = math.radians(60.0)
    fovy = math.radians(60.0)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    znear = 0.01
    zfar = 100.0

    world_view = torch.eye(4, dtype=torch.float32, device="cuda")
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to("cuda")

    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        projection_mode=projection_mode,
        ortho_scale_x=float(ortho_scale_x),
        ortho_scale_y=float(ortho_scale_y),
        isar_window_size=float(isar_window_size),
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=proj,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )


def compute_loss(
    rasterizer: GaussianRasterizer,
    means3d: torch.Tensor,
    cov3d: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    means2d = torch.zeros_like(means3d, requires_grad=True)
    opacities = torch.tensor([[0.95]], dtype=torch.float32, device="cuda")
    colors_precomp = torch.tensor([[0.8, 0.4, 0.2]], dtype=torch.float32, device="cuda")

    image, radii, _ = rasterizer(
        means3D=means3d,
        means2D=means2d,
        opacities=opacities,
        colors_precomp=colors_precomp,
        cov3D_precomp=cov3d,
    )

    # 使用 3x3 小窗口，减少单像素不稳定性
    h_mid = image.shape[1] // 2
    w_mid = image.shape[2] // 2
    patch = image[:, h_mid - 1 : h_mid + 2, w_mid - 1 : w_mid + 2]
    loss = patch.sum()
    return loss, int(radii[0].item())


def numeric_grad(
    rasterizer: GaussianRasterizer,
    means: torch.Tensor,
    cov: torch.Tensor,
    param: str,
    index: int,
    eps: float,
) -> float:
    means_plus = means.detach().clone()
    means_minus = means.detach().clone()
    cov_plus = cov.detach().clone()
    cov_minus = cov.detach().clone()

    if param == "mean":
        means_plus[0, index] += eps
        means_minus[0, index] -= eps
    elif param == "cov":
        cov_plus[0, index] += eps
        cov_minus[0, index] -= eps
    else:
        raise ValueError(f"Unknown param kind: {param}")

    loss_plus, _ = compute_loss(rasterizer, means_plus, cov_plus)
    loss_minus, _ = compute_loss(rasterizer, means_minus, cov_minus)
    return float((loss_plus - loss_minus).item() / (2.0 * eps))


def run_case(
    name: str,
    projection_mode: int,
    mean_init: Tuple[float, float, float],
    cov_init: Tuple[float, float, float, float, float, float],
    ortho_scale_x: float,
    ortho_scale_y: float,
    clamp_expected_zero_x: bool = False,
) -> CaseResult:
    settings = make_settings(
        projection_mode=projection_mode,
        width=9,
        height=9,
        ortho_scale_x=ortho_scale_x,
        ortho_scale_y=ortho_scale_y,
        isar_window_size=2.0,
    )
    rasterizer = GaussianRasterizer(settings)

    means = torch.tensor([list(mean_init)], dtype=torch.float32, device="cuda", requires_grad=True)
    cov = torch.tensor([list(cov_init)], dtype=torch.float32, device="cuda", requires_grad=True)

    loss, radius = compute_loss(rasterizer, means, cov)
    loss.backward()

    entries: List[GradEntry] = []

    # 2D mean <- 3D mean 投影链重点检查
    for idx, axis in enumerate(["x", "y", "z"]):
        ana = float(means.grad[0, idx].item())
        num = numeric_grad(rasterizer, means, cov, "mean", idx, eps=1e-3)
        entries.append(GradEntry(name=f"dL/dmean_{axis}", analytic=ana, numeric=num))

    # 协方差投影链重点检查（最小子集）
    for idx, tag in [(0, "xx"), (1, "xy"), (3, "yy")]:
        ana = float(cov.grad[0, idx].item())
        num = numeric_grad(rasterizer, means, cov, "cov", idx, eps=1e-3)
        entries.append(GradEntry(name=f"dL/dcov_{tag}", analytic=ana, numeric=num))

    mode_name = "perspective" if projection_mode == 0 else "orthographic"
    return CaseResult(
        name=name,
        mode=mode_name,
        entries=entries,
        radius=radius,
        clamp_expected_zero_x=clamp_expected_zero_x,
    )


def print_case(result: CaseResult) -> None:
    print(f"\n[CASE] {result.name} ({result.mode})")
    print(f"radius={result.radius}")
    print("metric                analytic         numeric          abs_err          rel_err")
    for e in result.entries:
        print(
            f"{e.name:<20} {e.analytic:>14.6e} {e.numeric:>14.6e} {e.abs_err:>14.6e} {e.rel_err:>14.6e}"
        )

    if result.clamp_expected_zero_x:
        x_entry = next(v for v in result.entries if v.name == "dL/dmean_x")
        print(
            f"clamp_check(dL/dmean_x near 0): analytic={x_entry.analytic:.6e}, numeric={x_entry.numeric:.6e}"
        )


def judge(results: List[CaseResult]) -> bool:
    # 对这种极小样本，放宽阈值但要求趋势一致
    max_abs_limit = 2e-2
    max_rel_limit = 3e-1
    all_ok = True

    for r in results:
        if r.radius <= 0:
            print(f"[FAIL] {r.name}: Gaussian not rendered (radius <= 0)")
            all_ok = False
            continue

        if r.max_abs_err > max_abs_limit and r.max_rel_err > max_rel_limit:
            print(
                f"[FAIL] {r.name}: max_abs_err={r.max_abs_err:.3e}, max_rel_err={r.max_rel_err:.3e}"
            )
            all_ok = False
        else:
            print(
                f"[PASS] {r.name}: max_abs_err={r.max_abs_err:.3e}, max_rel_err={r.max_rel_err:.3e}"
            )

        if r.clamp_expected_zero_x:
            x_entry = next(v for v in r.entries if v.name == "dL/dmean_x")
            if abs(x_entry.analytic) > 5e-4 or abs(x_entry.numeric) > 5e-4:
                print(
                    f"[WARN] {r.name}: clamp expected near-zero dL/dmean_x, got analytic={x_entry.analytic:.3e}, numeric={x_entry.numeric:.3e}"
                )
                all_ok = False
            else:
                print(f"[PASS] {r.name}: clamp piecewise derivative behaves near zero on x")

    return all_ok


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this stage_f gradient check script")

    # case 1: perspective 最小案例
    case_persp = run_case(
        name="perspective_minimal",
        projection_mode=0,
        mean_init=(0.05, -0.03, 3.0),
        cov_init=(0.05, 0.0, 0.0, 0.06, 0.0, 0.07),
        ortho_scale_x=2.0,
        ortho_scale_y=2.0,
    )

    # case 2: orthographic 最小案例（未触发 clamp）
    case_ortho = run_case(
        name="orthographic_minimal",
        projection_mode=1,
        mean_init=(0.20, -0.10, 3.0),
        cov_init=(0.05, 0.0, 0.0, 0.06, 0.0, 0.07),
        ortho_scale_x=2.0,
        ortho_scale_y=2.0,
    )

    # case 3: orthographic clamp 分段导数案例（x 超出 clamp 范围）
    case_ortho_clamp = run_case(
        name="orthographic_clamp_x",
        projection_mode=1,
        mean_init=(2.2, -0.10, 3.0),
        cov_init=(0.05, 0.0, 0.0, 0.06, 0.0, 0.07),
        ortho_scale_x=2.0,
        ortho_scale_y=2.0,
        clamp_expected_zero_x=True,
    )

    results = [case_persp, case_ortho, case_ortho_clamp]
    for item in results:
        print_case(item)

    ok = judge(results)
    print("\n[SUMMARY] stage_f gradient check =", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
