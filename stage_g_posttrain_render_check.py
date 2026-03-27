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
    parser.add_argument("--load_iteration", type=int, default=20)

    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise RuntimeError(f"model_path does not exist: {args.model_path}")

    print("[POST-TRAIN-CHECK] loading scene/model...")
    print(f"source_path={args.source_path}")
    print(f"model_path={args.model_path}")
    print(f"load_iteration={args.load_iteration}")

    try:
        gaussians = GaussianModel(args.sh_degree)
        scene_obj = Scene(args, gaussians, load_iteration=args.load_iteration, shuffle=False, resolution_scales=[1.0])
        cams = scene_obj.getTrainCameras(1.0)
        if not cams:
            raise RuntimeError("No train cameras found for post-train render check")

        cam = cams[0]
        bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

        with torch.no_grad():
            pkg = render(cam, gaussians, pipeline_params.extract(args), bg)
            image = pkg["render"]

        finite_ok = bool(torch.isfinite(image).all().item())
        min_v = float(image.min().item())
        max_v = float(image.max().item())
        mean_v = float(image.mean().item())
        std_v = float(image.std().item())
        nonzero_ratio = float((image.abs() > 1e-6).float().mean().item())
        near_white_ratio = float((image > 0.999).float().mean().item())

        black_like = nonzero_ratio < 1e-3
        white_like = near_white_ratio > 0.999

        print("[POST-TRAIN-CHECK] metrics")
        print(f"finite_ok={finite_ok}")
        print(f"shape={tuple(image.shape)}")
        print(f"min={min_v:.6f}, max={max_v:.6f}, mean={mean_v:.6f}, std={std_v:.6f}")
        print(f"nonzero_ratio={nonzero_ratio:.6f}, near_white_ratio={near_white_ratio:.6f}")

        if finite_ok and (not black_like) and (not white_like):
            print("[POST-TRAIN-CHECK] PASS")
        else:
            print("[POST-TRAIN-CHECK] FAIL")
            print(f"black_like={black_like}, white_like={white_like}")
            raise RuntimeError("Post-train render check failed due to invalid image statistics")

    except Exception as exc:
        print("[POST-TRAIN-CHECK] EXCEPTION")
        print(f"type={type(exc).__name__}")
        print(f"message={exc}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
