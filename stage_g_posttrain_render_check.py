import os
import traceback
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.isar_observation import apply_isar_observation_operator


def resolve_background(args) -> tuple[torch.Tensor, dict]:
    source_path = str(getattr(args, "source_path", "") or "")
    source_lower = source_path.lower()
    is_isar_source = "isar" in source_lower

    white_background_cfg = bool(getattr(args, "white_background", False))
    allow_white_for_isar = bool(getattr(args, "allow_white_background_for_isar", False))

    if is_isar_source and not allow_white_for_isar:
        effective_white_background = False
        reason = "isar_source_forces_black"
    else:
        effective_white_background = white_background_cfg
        reason = "cfg_white_background" if effective_white_background else "cfg_black_background"

    bg_color = [1.0, 1.0, 1.0] if effective_white_background else [0.0, 0.0, 0.0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    info = {
        "source_path": source_path,
        "is_isar_source": is_isar_source,
        "white_background_cfg": white_background_cfg,
        "allow_white_background_for_isar": allow_white_for_isar,
        "white_background_effective": effective_white_background,
        "reason": reason,
        "background_rgb": bg_color,
    }
    return bg, info


def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=20)
    parser.add_argument("--allow_white_background_for_isar", action="store_true")

    args = get_combined_args(parser)

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
        bg, bg_info = resolve_background(args)
        print(f"[POST-TRAIN-CHECK] background={bg_info}")

        with torch.no_grad():
            pkg = render(cam, gaussians, pipeline_params.extract(args), bg)
            image_raw = pkg["render"]

        # Stats for raw
        finite_ok = bool(torch.isfinite(image_raw).all().item())
        min_v = float(image_raw.min().item())
        max_v = float(image_raw.max().item())
        mean_v = float(image_raw.mean().item())
        std_v = float(image_raw.std().item())
        nonzero_ratio = float((image_raw.abs() > 1e-6).float().mean().item())
        near_white_ratio = float((image_raw > 0.999).float().mean().item())

        black_like = nonzero_ratio < 1e-3
        white_like = near_white_ratio > 0.999

        print("[POST-TRAIN-CHECK] RAW metrics")
        print(f"finite_ok={finite_ok}")
        print(f"shape={tuple(image_raw.shape)}")
        print(f"min={min_v:.6f}, max={max_v:.6f}, mean={mean_v:.6f}, std={std_v:.6f}")
        print(f"nonzero_ratio={nonzero_ratio:.6f}, near_white_ratio={near_white_ratio:.6f}")

        # Extensible Observation checks
        all_obs_finite_ok = True
        for obs_mode in ["log1p"]:
            image_obs = apply_isar_observation_operator(image_raw, obs_mode)
            obs_finite_ok = bool(torch.isfinite(image_obs).all().item())
            if not obs_finite_ok:
                all_obs_finite_ok = False
            obs_min_v = float(image_obs.min().item())
            obs_max_v = float(image_obs.max().item())
            obs_mean_v = float(image_obs.mean().item())
            obs_std_v = float(image_obs.std().item())
            print(f"[POST-TRAIN-CHECK] OBS ({obs_mode}) metrics")
            print(f"obs_finite_ok={obs_finite_ok}")
            print(f"min={obs_min_v:.6f}, max={obs_max_v:.6f}, mean={obs_mean_v:.6f}, std={obs_std_v:.6f}")

        if finite_ok and all_obs_finite_ok and (not black_like) and (not white_like):
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
