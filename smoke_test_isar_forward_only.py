import os
import traceback
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import Scene, GaussianModel


def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    args = parser.parse_args([
        "--source_path", "D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest",
        "--model_path", "D:/3DGS_new/gaussian-splatting/output/isar_forward_only_smoke",
        "--images", "images",
    ])

    os.makedirs(args.model_path, exist_ok=True)

    print("[FWD-SMOKE] Building ISAR Scene...")
    print(f"[FWD-SMOKE] source_path={args.source_path}")

    try:
        gaussians = GaussianModel(args.sh_degree)
        scene_obj = Scene(args, gaussians, load_iteration=None, shuffle=False, resolution_scales=[1.0])
        train_cams = scene_obj.getTrainCameras(1.0)
        if not train_cams:
            raise RuntimeError("No train cameras loaded for ISAR forward smoke test")

        cam = train_cams[0]
        bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipeline_params.extract(args), bg)
            image = render_pkg["render"]

        finite_ok = bool(torch.isfinite(image).all().item())
        min_v = float(image.min().item())
        max_v = float(image.max().item())
        mean_v = float(image.mean().item())
        std_v = float(image.std().item())
        nonzero_ratio = float((image.abs() > 1e-6).float().mean().item())

        print("[FWD-SMOKE] PASS")
        print(f"projection_mode={getattr(cam, 'projection_mode', 'unknown')}")
        print(f"image_shape={tuple(image.shape)}")
        print(f"finite_ok={finite_ok}")
        print(f"min={min_v:.6f}, max={max_v:.6f}, mean={mean_v:.6f}, std={std_v:.6f}")
        print(f"nonzero_ratio={nonzero_ratio:.6f}")

        if not finite_ok:
            raise RuntimeError("Forward output contains NaN/Inf")

    except Exception as exc:
        print("[FWD-SMOKE] FAIL")
        print(f"error_type={type(exc).__name__}")
        print(f"error_message={exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
