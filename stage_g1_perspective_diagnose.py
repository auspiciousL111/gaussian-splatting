import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix


EPS_LIST = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]


@dataclass
class ParamEval:
    metric: str
    analytic: float
    numeric_by_eps: Dict[float, float]

    def abs_err(self, eps: float) -> float:
        return abs(self.analytic - self.numeric_by_eps[eps])

    def rel_err(self, eps: float) -> float:
        v = self.numeric_by_eps[eps]
        denom = max(abs(self.analytic), abs(v), 1e-8)
        return abs(self.analytic - v) / denom


@dataclass
class ScenarioResult:
    name: str
    gaussians: int
    overlap_hint: str
    depth_hint: str
    evals: List[ParamEval]


@dataclass
class MetricClassSummary:
    metric_class: str
    count: int
    fail_count: int
    fail_ratio: float


def make_settings(width: int = 15, height: int = 15) -> GaussianRasterizationSettings:
    fovx = math.radians(60.0)
    fovy = math.radians(60.0)
    tanfovx = math.tan(fovx * 0.5)
    tanfovy = math.tan(fovy * 0.5)

    view = torch.eye(4, dtype=torch.float32, device="cuda")
    proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).to("cuda")

    return GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        projection_mode=0,  # perspective only
        ortho_scale_x=2.0,
        ortho_scale_y=2.0,
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


def render_loss(rasterizer: GaussianRasterizer, means: torch.Tensor, cov: torch.Tensor, patch: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    means2d = torch.zeros_like(means, requires_grad=True)
    opacities = torch.full((means.shape[0], 1), 0.9, dtype=torch.float32, device="cuda")
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
    return p.sum(), radii


def numeric_grad(rasterizer: GaussianRasterizer, means: torch.Tensor, cov: torch.Tensor, tensor_name: str, idx: Tuple[int, int], eps: float) -> float:
    means_p = means.detach().clone()
    means_m = means.detach().clone()
    cov_p = cov.detach().clone()
    cov_m = cov.detach().clone()

    if tensor_name == "mean":
        means_p[idx] += eps
        means_m[idx] -= eps
    else:
        cov_p[idx] += eps
        cov_m[idx] -= eps

    lp, _ = render_loss(rasterizer, means_p, cov_p)
    lm, _ = render_loss(rasterizer, means_m, cov_m)
    return float((lp - lm).item() / (2.0 * eps))


def evaluate_scenario(name: str, means_init: Sequence[Sequence[float]], cov_init: Sequence[Sequence[float]], overlap_hint: str, depth_hint: str) -> ScenarioResult:
    settings = make_settings()
    rasterizer = GaussianRasterizer(settings)

    means = torch.tensor(means_init, dtype=torch.float32, device="cuda", requires_grad=True)
    cov = torch.tensor(cov_init, dtype=torch.float32, device="cuda", requires_grad=True)

    loss, radii = render_loss(rasterizer, means, cov)
    visible = int((radii > 0).sum().item())
    if visible < max(1, means.shape[0] - 1):
        print(f"[WARN] {name}: low visibility ({visible}/{means.shape[0]})")

    loss.backward()

    checks: List[Tuple[str, str, Tuple[int, int]]] = []
    for g in range(means.shape[0]):
        checks.append((f"mean_x_g{g}", "mean", (g, 0)))
        checks.append((f"mean_y_g{g}", "mean", (g, 1)))

    # 重点按要求覆盖 cov_xx/cov_xy/cov_yy
    for g in range(min(2, means.shape[0])):
        checks.append((f"cov_xx_g{g}", "cov", (g, 0)))
        checks.append((f"cov_xy_g{g}", "cov", (g, 1)))
        checks.append((f"cov_yy_g{g}", "cov", (g, 3)))

    out: List[ParamEval] = []
    for metric, tname, idx in checks:
        ana = float(means.grad[idx].item()) if tname == "mean" else float(cov.grad[idx].item())
        nums = {eps: numeric_grad(rasterizer, means, cov, tname, idx, eps) for eps in EPS_LIST}
        out.append(ParamEval(metric=metric, analytic=ana, numeric_by_eps=nums))

    return ScenarioResult(name=name, gaussians=means.shape[0], overlap_hint=overlap_hint, depth_hint=depth_hint, evals=out)


def classify_param(ev: ParamEval) -> str:
    # 取最常用 eps=1e-3 作为主判据
    abs_main = ev.abs_err(1e-3)
    rel_main = ev.rel_err(1e-3)

    # 看 eps sweep 稳定性：若 numeric 随 eps 波动/符号翻转大，偏向非平滑
    vals = [ev.numeric_by_eps[e] for e in EPS_LIST]
    signs = [0 if abs(v) < 1e-10 else (1 if v > 0 else -1) for v in vals]
    sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1] and 0 not in (signs[i], signs[i - 1]))
    spread = max(vals) - min(vals)
    scale = max(max(abs(v) for v in vals), abs(ev.analytic), 1e-6)
    spread_ratio = spread / scale

    if abs_main < 2e-2 or rel_main < 3e-1:
        return "ok"

    if sign_changes >= 2 or spread_ratio > 1.0:
        return "likely_nonsmooth_numeric"

    return "suspicious_analytic"


def summarize_by_class(results: List[ScenarioResult], key_prefix: str) -> MetricClassSummary:
    labels = []
    for sc in results:
        for ev in sc.evals:
            if ev.metric.startswith(key_prefix):
                labels.append(classify_param(ev))

    fail = sum(1 for x in labels if x != "ok")
    cnt = len(labels)
    return MetricClassSummary(metric_class=key_prefix, count=cnt, fail_count=fail, fail_ratio=(fail / cnt if cnt else 0.0))


def print_scenario(sc: ScenarioResult) -> None:
    print(f"\n[SCENARIO] {sc.name} | gaussians={sc.gaussians} | overlap={sc.overlap_hint} | depth={sc.depth_hint}")
    print("metric           analytic     num@1e-2    num@5e-3    num@1e-3    num@5e-4    num@1e-4    abs@1e-3    rel@1e-3    label")
    for ev in sc.evals:
        label = classify_param(ev)
        print(
            f"{ev.metric:<15} {ev.analytic:>11.4e} {ev.numeric_by_eps[1e-2]:>11.4e} {ev.numeric_by_eps[5e-3]:>11.4e} "
            f"{ev.numeric_by_eps[1e-3]:>11.4e} {ev.numeric_by_eps[5e-4]:>11.4e} {ev.numeric_by_eps[1e-4]:>11.4e} "
            f"{ev.abs_err(1e-3):>11.4e} {ev.rel_err(1e-3):>11.4e} {label}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    # 1) 原失败风格：3高斯，重叠+深度差
    s0 = evaluate_scenario(
        name="S0_original_fail_like",
        means_init=[[0.00, 0.00, 2.7], [0.06, 0.015, 2.85], [-0.05, -0.02, 3.0]],
        cov_init=[[0.09, 0.006, 0.0, 0.085, 0.0, 0.09], [0.085, -0.005, 0.0, 0.09, 0.0, 0.095], [0.08, 0.004, 0.0, 0.08, 0.0, 0.09]],
        overlap_hint="medium",
        depth_hint="separated",
    )

    # 2) 多高斯但拉大间距，减弱重叠
    s1 = evaluate_scenario(
        name="S1_spread_apart",
        means_init=[[-0.45, -0.45, 2.8], [0.45, -0.45, 2.9], [0.00, 0.45, 3.0]],
        cov_init=[[0.08, 0.0, 0.0, 0.08, 0.0, 0.09], [0.08, 0.0, 0.0, 0.08, 0.0, 0.09], [0.08, 0.0, 0.0, 0.08, 0.0, 0.09]],
        overlap_hint="low",
        depth_hint="separated",
    )

    # 3) 多高斯但减少深度差/遮挡敏感性
    s2 = evaluate_scenario(
        name="S2_small_depth_gap",
        means_init=[[0.00, 0.00, 2.85], [0.07, 0.01, 2.87], [-0.05, -0.015, 2.89]],
        cov_init=[[0.085, 0.005, 0.0, 0.085, 0.0, 0.09], [0.085, -0.004, 0.0, 0.085, 0.0, 0.09], [0.085, 0.003, 0.0, 0.085, 0.0, 0.09]],
        overlap_hint="medium",
        depth_hint="close",
    )

    # 4) 3 -> 2 Gaussian
    s3 = evaluate_scenario(
        name="S3_two_gaussians",
        means_init=[[0.00, 0.00, 2.8], [0.08, 0.02, 2.9]],
        cov_init=[[0.09, 0.006, 0.0, 0.085, 0.0, 0.09], [0.085, -0.004, 0.0, 0.09, 0.0, 0.095]],
        overlap_hint="medium",
        depth_hint="moderate",
    )

    # 5) 单高斯 perspective 对照
    s4 = evaluate_scenario(
        name="S4_single_gaussian",
        means_init=[[0.05, -0.03, 3.0]],
        cov_init=[[0.05, 0.0, 0.0, 0.06, 0.0, 0.07]],
        overlap_hint="none",
        depth_hint="single",
    )

    scenarios = [s0, s1, s2, s3, s4]
    for sc in scenarios:
        print_scenario(sc)

    # 分量级定位统计
    sum_mean_x = summarize_by_class(scenarios, "mean_x")
    sum_mean_y = summarize_by_class(scenarios, "mean_y")
    sum_cov_xx = summarize_by_class(scenarios, "cov_xx")
    sum_cov_xy = summarize_by_class(scenarios, "cov_xy")
    sum_cov_yy = summarize_by_class(scenarios, "cov_yy")

    print("\n[CLASS-SUMMARY]")
    for s in [sum_mean_x, sum_mean_y, sum_cov_xx, sum_cov_xy, sum_cov_yy]:
        print(f"{s.metric_class}: count={s.count}, fail_count={s.fail_count}, fail_ratio={s.fail_ratio:.3f}")

    # 全局判读
    labels = []
    for sc in scenarios:
        for ev in sc.evals:
            labels.append(classify_param(ev))

    n_ok = sum(1 for x in labels if x == "ok")
    n_nonsmooth = sum(1 for x in labels if x == "likely_nonsmooth_numeric")
    n_sus = sum(1 for x in labels if x == "suspicious_analytic")
    total = len(labels)

    print("\n[GLOBAL-JUDGEMENT]")
    print(f"total={total}, ok={n_ok}, likely_nonsmooth_numeric={n_nonsmooth}, suspicious_analytic={n_sus}")

    if n_sus == 0 and n_nonsmooth > 0:
        print("diagnosis=majority_non_smooth_numeric_effect")
    elif n_sus > 0:
        print("diagnosis=contains_suspicious_analytic_items")
    else:
        print("diagnosis=mostly_consistent")


if __name__ == "__main__":
    main()
