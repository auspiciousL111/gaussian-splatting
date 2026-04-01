import os
import json
import traceback
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.isar_observation import (
    DEFAULT_ISAR_OBSERVATION_MODES,
    apply_isar_observation_operator,
    build_isar_observation_specs,
    to_isar_intensity,
)


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


def parse_observation_modes(obs_modes_csv: str) -> list[str]:
    modes = [m.strip() for m in str(obs_modes_csv).split(",") if m.strip()]
    if not modes:
        raise ValueError("obs_modes must include at least one mode")
    return modes


def build_observation_specs_from_args(args) -> list[dict]:
    modes = parse_observation_modes(getattr(args, "obs_modes", ",".join(DEFAULT_ISAR_OBSERVATION_MODES)))
    base_context = {
        "range_axis": getattr(args, "obs_range_axis", None),
        "cross_range_axis": getattr(args, "obs_cross_range_axis", None),
        "future_physical_operator_name": getattr(args, "obs_future_physical_operator_name", None),
    }
    context_overrides = {mode: dict(base_context) for mode in modes}
    if "db_radar" in context_overrides:
        context_overrides["db_radar"].update(
            {
                "noise_floor_db": getattr(args, "obs_noise_floor_db", -40.0),
                "dynamic_range_db": getattr(args, "obs_dynamic_range_db", 40.0),
                "normalization_mode": getattr(args, "obs_normalization_mode", "db_floor_to_unit_interval"),
                "clamp_min": getattr(args, "obs_clamp_min", 0.0),
                "clamp_max": getattr(args, "obs_clamp_max", None),
            }
        )
    if "db_cfar" in context_overrides:
        context_overrides["db_cfar"].update(
            {
                "epsilon": getattr(args, "obs_cfar_epsilon", 1e-4),
                "nonzero_threshold": getattr(args, "obs_cfar_nonzero_threshold", 1e-6),
                "q_low": getattr(args, "obs_cfar_q_low", 0.2),
                "q_high": getattr(args, "obs_cfar_q_high", 0.995),
                "min_points": getattr(args, "obs_cfar_min_points", 64),
                "fallback_noise_floor_db": getattr(args, "obs_cfar_fallback_noise_floor_db", -40.0),
                "fallback_dynamic_range_db": getattr(args, "obs_cfar_fallback_dynamic_range_db", 40.0),
                "normalization_mode": getattr(args, "obs_cfar_normalization_mode", "nonzero_percentile_window"),
                "clamp_min": getattr(args, "obs_cfar_clamp_min", 0.0),
                "clamp_max": getattr(args, "obs_cfar_clamp_max", 1.0),
            }
        )
    return build_isar_observation_specs(modes, context_overrides)


def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=20)
    parser.add_argument("--allow_white_background_for_isar", action="store_true")
    parser.add_argument("--obs_modes", type=str, default=",".join(DEFAULT_ISAR_OBSERVATION_MODES))
    parser.add_argument("--obs_noise_floor_db", type=float, default=-40.0)
    parser.add_argument("--obs_dynamic_range_db", type=float, default=40.0)
    parser.add_argument("--obs_normalization_mode", type=str, default="db_floor_to_unit_interval")
    parser.add_argument("--obs_clamp_min", type=float, default=0.0)
    parser.add_argument("--obs_clamp_max", type=float, default=None)
    parser.add_argument("--obs_range_axis", type=str, default=None)
    parser.add_argument("--obs_cross_range_axis", type=str, default=None)
    parser.add_argument("--obs_future_physical_operator_name", type=str, default=None)
    parser.add_argument("--obs_cfar_epsilon", type=float, default=1e-4)
    parser.add_argument("--obs_cfar_nonzero_threshold", type=float, default=1e-6)
    parser.add_argument("--obs_cfar_q_low", type=float, default=0.2)
    parser.add_argument("--obs_cfar_q_high", type=float, default=0.995)
    parser.add_argument("--obs_cfar_min_points", type=int, default=64)
    parser.add_argument("--obs_cfar_fallback_noise_floor_db", type=float, default=-40.0)
    parser.add_argument("--obs_cfar_fallback_dynamic_range_db", type=float, default=40.0)
    parser.add_argument("--obs_cfar_normalization_mode", type=str, default="nonzero_percentile_window")
    parser.add_argument("--obs_cfar_clamp_min", type=float, default=0.0)
    parser.add_argument("--obs_cfar_clamp_max", type=float, default=1.0)
    parser.add_argument("--output_json", type=str, default="")

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
            image_raw = to_isar_intensity(pkg.get("intensity", pkg["render"]))

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

        obs_specs = build_observation_specs_from_args(args)
        observation_results = {}

        # Extensible Observation checks
        all_obs_finite_ok = True
        for spec in obs_specs:
            obs_mode = spec["mode"]
            obs_context = spec["context"]
            image_obs = apply_isar_observation_operator(image_raw, obs_mode, obs_context)
            obs_finite_ok = bool(torch.isfinite(image_obs).all().item())
            if not obs_finite_ok:
                all_obs_finite_ok = False
            obs_min_v = float(image_obs.min().item())
            obs_max_v = float(image_obs.max().item())
            obs_mean_v = float(image_obs.mean().item())
            obs_std_v = float(image_obs.std().item())
            observation_results[obs_mode] = {
                "mode": obs_mode,
                "context": obs_context,
                "stats": {
                    "finite_ok": obs_finite_ok,
                    "shape": list(image_obs.shape),
                    "min": obs_min_v,
                    "max": obs_max_v,
                    "mean": obs_mean_v,
                    "std": obs_std_v,
                    "nonzero_ratio": float((image_obs.abs() > 1e-6).float().mean().item()),
                    "near_white_ratio": float((image_obs > 0.999).float().mean().item()),
                },
            }
            print(f"[POST-TRAIN-CHECK] OBS ({obs_mode}) metrics")
            print(f"context={obs_context}")
            print(f"obs_finite_ok={obs_finite_ok}")
            print(f"min={obs_min_v:.6f}, max={obs_max_v:.6f}, mean={obs_mean_v:.6f}, std={obs_std_v:.6f}")

        loaded_iteration = int(getattr(scene_obj, "loaded_iter", args.load_iteration))
        export_payload = {
            "model_path": args.model_path,
            "source_path": args.source_path,
            "loaded_iteration": loaded_iteration,
            "background": bg_info,
            "raw": {
                "finite_ok": finite_ok,
                "shape": list(image_raw.shape),
                "min": min_v,
                "max": max_v,
                "mean": mean_v,
                "std": std_v,
                "nonzero_ratio": nonzero_ratio,
                "near_white_ratio": near_white_ratio,
            },
            "observation_interface": {
                "version": "context_v1",
                "modes": [spec["mode"] for spec in obs_specs],
            },
            "observation_outputs": observation_results,
        }

        if args.output_json:
            out_json = args.output_json
        else:
            out_json = os.path.join(args.model_path, f"stage_g_observation_context_iter{loaded_iteration}.json")
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        print(f"[POST-TRAIN-CHECK] observation_export_json={out_json}")

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
