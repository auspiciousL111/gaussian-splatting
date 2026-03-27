import math
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
class CheckResult:
    name: str
    passed: bool
    summary: str


def make_settings(projection_mode: int, width: int = 11, height: int = 11, ortho_scale_x: float = 2.0, ortho_scale_y: float = 2.0) -> GaussianRasterizationSettings:
    fovx = math.radians(60.0)
    fovy = math.radians(60.0)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    znear = 0.01
    zfar = 100.0

    view = torch.eye(4, dtype=torch.float32, device="cuda")
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to("cuda")

    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        projection_mode=projection_mode,
        ortho_scale_x=float(ortho_scale_x),
        ortho_scale_y=float(ortho_scale_y),
        isar_window_size=2.0,
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=view,
        projmatrix=proj,
        sh_degree=0,
        campos=torch.zeros(3, dtype=torch.float32, device="cuda"),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )


def render_loss(rasterizer: GaussianRasterizer, means: torch.Tensor, cov: torch.Tensor, patch: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    means2d = torch.zeros_like(means, requires_grad=True)
    opacities = torch.full((means.shape[0], 1), 0.9, dtype=torch.float32, device="cuda")

    # 给多高斯不同颜色，提升梯度可观测性
    color_base = torch.tensor(
        [[0.9, 0.2, 0.1], [0.2, 0.9, 0.2], [0.2, 0.3, 0.9], [0.9, 0.8, 0.2]],
        dtype=torch.float32,
        device="cuda",
    )
    colors = color_base[: means.shape[0]].contiguous()

    image, radii, _ = rasterizer(
        means3D=means,
        means2D=means2d,
        opacities=opacities,
        colors_precomp=colors,
        cov3D_precomp=cov,
    )

    h_mid = image.shape[1] // 2
    w_mid = image.shape[2] // 2
    half = patch // 2
    p = image[:, h_mid - half : h_mid + half + 1, w_mid - half : w_mid + half + 1]
    loss = p.sum()
    return loss, radii


def numeric_grad_for_tensor(
    rasterizer: GaussianRasterizer,
    means: torch.Tensor,
    cov: torch.Tensor,
    tensor_name: str,
    index: Tuple[int, int],
    eps: float = 1e-3,
    patch: int = 3,
) -> float:
    means_p = means.detach().clone()
    means_m = means.detach().clone()
    cov_p = cov.detach().clone()
    cov_m = cov.detach().clone()

    if tensor_name == "mean":
        means_p[index] += eps
        means_m[index] -= eps
    elif tensor_name == "cov":
        cov_p[index] += eps
        cov_m[index] -= eps
    else:
        raise ValueError(f"Unsupported tensor_name: {tensor_name}")

    lp, _ = render_loss(rasterizer, means_p, cov_p, patch=patch)
    lm, _ = render_loss(rasterizer, means_m, cov_m, patch=patch)
    return float((lp - lm).item() / (2.0 * eps))


def check_clamp_boundary_scan() -> CheckResult:
    settings = make_settings(projection_mode=1, width=11, height=11, ortho_scale_x=2.0, ortho_scale_y=2.0)
    rasterizer = GaussianRasterizer(settings)

    cov = torch.tensor([[0.05, 0.0, 0.0, 0.06, 0.0, 0.07]], dtype=torch.float32, device="cuda", requires_grad=True)

    # s=2/max(2,eps)=1, 因此阈值在 |x|≈1.3 与 |y|≈1.3
    scan_values = [-1.31, -1.305, -1.301, -1.299, -1.295, -1.29, 1.29, 1.295, 1.299, 1.301, 1.305, 1.31]

    entries: List[GradEntry] = []

    def run_axis(axis: int, fixed_x: float, fixed_y: float, value: float, name: str) -> None:
        mean = [fixed_x, fixed_y, 3.0]
        mean[axis] = value
        means = torch.tensor([mean], dtype=torch.float32, device="cuda", requires_grad=True)

        loss, radii = render_loss(rasterizer, means, cov)
        if int(radii[0].item()) <= 0:
            entries.append(GradEntry(name=f"{name}_{value:.4f}_radius0", analytic=float("nan"), numeric=float("nan")))
            return

        loss.backward()
        ana = float(means.grad[0, axis].item())
        num = numeric_grad_for_tensor(rasterizer, means, cov, "mean", (0, axis))
        entries.append(GradEntry(name=f"{name}_{value:.4f}", analytic=ana, numeric=num))

    for v in scan_values:
        run_axis(axis=0, fixed_x=0.0, fixed_y=0.0, value=v, name="x")
    for v in scan_values:
        run_axis(axis=1, fixed_x=0.0, fixed_y=0.0, value=v, name="y")

    valid = [e for e in entries if not math.isnan(e.abs_err)]
    max_abs = max((e.abs_err for e in valid), default=float("inf"))
    max_rel = max((e.rel_err for e in valid), default=float("inf"))

    # clamp 边界点附近允许较大误差；远离边界要求更严
    relaxed_band = 0.002
    pass_count = 0
    total_count = 0
    suspicious = []
    for e in valid:
        tag, value_str = e.name.split("_")[0], e.name.split("_")[1]
        v = abs(float(value_str))
        near_kink = abs(v - 1.3) < relaxed_band
        total_count += 1
        if near_kink:
            # 不纳入 hard fail，仅记录
            pass_count += 1
            continue

        ok = (e.abs_err < 2e-2) or (e.rel_err < 2.5e-1)
        if ok:
            pass_count += 1
        else:
            suspicious.append(e.name)

    # 额外检查：clamped 区域外侧梯度应接近 0
    clamped_candidates = [e for e in valid if abs(float(e.name.split("_")[1])) >= 1.305]
    clamp_zero_ok = all(abs(e.analytic) < 5e-4 and abs(e.numeric) < 5e-4 for e in clamped_candidates)

    passed = (pass_count == total_count) and clamp_zero_ok
    summary = (
        f"scan_points={len(valid)}, max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e}, "
        f"clamp_zero_ok={clamp_zero_ok}, suspicious={len(suspicious)}"
    )

    print("\n[CLAMP-SCAN] around |s_x x|≈1.3 and |s_y y|≈1.3")
    print(summary)
    if suspicious:
        print("suspicious_samples=", ", ".join(suspicious[:8]))

    return CheckResult(name="clamp_boundary_scan", passed=passed, summary=summary)


def run_multi_gaussian_case(name: str, projection_mode: int) -> CheckResult:
    settings = make_settings(projection_mode=projection_mode, width=13, height=13, ortho_scale_x=2.0, ortho_scale_y=2.0)
    rasterizer = GaussianRasterizer(settings)

    means = torch.tensor(
        [
            [0.00, 0.00, 2.7],
            [0.06, 0.015, 2.85],
            [-0.05, -0.02, 3.0],
        ],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    cov = torch.tensor(
        [
            [0.09, 0.006, 0.0, 0.085, 0.0, 0.09],
            [0.085, -0.005, 0.0, 0.09, 0.0, 0.095],
            [0.08, 0.004, 0.0, 0.08, 0.0, 0.09],
        ],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )

    loss, radii = render_loss(rasterizer, means, cov, patch=5)
    if int((radii > 0).sum().item()) < 2:
        return CheckResult(name=name, passed=False, summary="insufficient visible gaussians")

    loss.backward()

    checks: List[Tuple[str, str, Tuple[int, int]]] = []
    for gi in range(3):
        checks.append((f"mean_g{gi}_x", "mean", (gi, 0)))
        checks.append((f"mean_g{gi}_y", "mean", (gi, 1)))

    checks.extend(
        [
            ("cov_g0_xx", "cov", (0, 0)),
            ("cov_g0_xy", "cov", (0, 1)),
            ("cov_g0_yy", "cov", (0, 3)),
            ("cov_g1_xx", "cov", (1, 0)),
            ("cov_g1_xy", "cov", (1, 1)),
            ("cov_g1_yy", "cov", (1, 3)),
        ]
    )

    entries: List[GradEntry] = []
    for nm, tensor_name, idx in checks:
        if tensor_name == "mean":
            ana = float(means.grad[idx].item())
        else:
            ana = float(cov.grad[idx].item())
        num = numeric_grad_for_tensor(rasterizer, means, cov, tensor_name=tensor_name, index=idx, eps=5e-4, patch=5)
        entries.append(GradEntry(name=nm, analytic=ana, numeric=num))

    max_abs = max(e.abs_err for e in entries)
    max_rel = max(e.rel_err for e in entries)
    fail_list = [e.name for e in entries if not ((e.abs_err < 2.5e-2) or (e.rel_err < 3.5e-1))]

    passed = len(fail_list) == 0
    summary = f"checks={len(entries)}, max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e}, fails={len(fail_list)}"

    print(f"\n[MULTI-GAUSS] {name}")
    print(summary)
    if fail_list:
        print("failed_metrics=", ", ".join(fail_list[:8]))

    return CheckResult(name=name, passed=passed, summary=summary)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for stage_g_validation")

    print("[Stage G] validation started")
    r1 = check_clamp_boundary_scan()
    r2 = run_multi_gaussian_case("multi_gauss_perspective", projection_mode=0)
    r3 = run_multi_gaussian_case("multi_gauss_orthographic", projection_mode=1)

    all_results = [r1, r2, r3]
    for r in all_results:
        print(f"[RESULT] {r.name}: {'PASS' if r.passed else 'FAIL'} | {r.summary}")

    overall = all(r.passed for r in all_results)
    print("\n[SUMMARY] stage_g_validation =", "PASS" if overall else "FAIL")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
