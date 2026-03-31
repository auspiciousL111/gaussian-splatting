import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss
from utils.isar_observation import apply_isar_observation_operator


def to_3ch(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3 and img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img


def to_gray(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(img.shape)}")
    if img.shape[0] == 1:
        return img
    if img.shape[0] >= 3:
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=img.dtype, device=img.device).view(3, 1, 1)
        return (img[:3] * weights).sum(dim=0, keepdim=True)
    raise ValueError(f"Unsupported channel count for grayscale conversion: {img.shape[0]}")


def save_tensor_image(path: str, image_chw: torch.Tensor) -> None:
    image = torch.clamp(image_chw.detach().cpu(), 0.0, 1.0)
    image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def save_tensor_gray_image(path: str, image_1hw: torch.Tensor) -> None:
    image = torch.clamp(image_1hw.detach().cpu(), 0.0, 1.0)
    if image.dim() != 3 or image.shape[0] != 1:
        raise ValueError(f"Expected 1xHxW gray tensor, got shape={tuple(image.shape)}")
    image = (image.squeeze(0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def histogram_summary(image: torch.Tensor, bins: int = 16) -> dict:
    image_clamped = torch.clamp(image.detach().cpu(), 0.0, 1.0)
    hist = torch.histc(image_clamped, bins=bins, min=0.0, max=1.0)
    total = float(hist.sum().item())
    if total <= 0.0:
        total = 1.0

    quantile_probs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=torch.float32)
    quantiles = torch.quantile(image_clamped.view(-1), quantile_probs).tolist()

    return {
        "bins": int(bins),
        "counts": [int(round(v)) for v in hist.tolist()],
        "ratios": [float(v / total) for v in hist.tolist()],
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }


def summarize(image: torch.Tensor, gt: torch.Tensor) -> dict:
    finite_ok = bool(torch.isfinite(image).all().item())
    return {
        "finite_ok": finite_ok,
        "shape": list(image.shape),
        "min": float(image.min().item()),
        "max": float(image.max().item()),
        "mean": float(image.mean().item()),
        "std": float(image.std().item()),
        "nonzero_ratio": float((image.abs() > 1e-6).float().mean().item()),
        "near_white_ratio": float((image > 0.999).float().mean().item()),
        "l1_vs_gt": float(l1_loss(image, gt).item()),
        "psnr_vs_gt": float(psnr(image, gt).mean().item()),
    }


def summarize_intensity(image: torch.Tensor, gt: torch.Tensor, hist_bins: int) -> dict:
    finite_ok = bool(torch.isfinite(image).all().item())
    return {
        "finite_ok": finite_ok,
        "shape": list(image.shape),
        "min": float(image.min().item()),
        "max": float(image.max().item()),
        "mean": float(image.mean().item()),
        "std": float(image.std().item()),
        "nonzero_ratio": float((image.abs() > 1e-6).float().mean().item()),
        "near_black_ratio": float((image < 1e-3).float().mean().item()),
        "near_white_ratio": float((image > 0.999).float().mean().item()),
        "l1_vs_gt": float(l1_loss(image, gt).item()),
        "psnr_vs_gt": float(psnr(image, gt).mean().item()),
        "histogram": histogram_summary(image, bins=hist_bins),
    }


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


def main() -> None:
    parser = ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--camera_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./output/stage_h_compare")
    parser.add_argument("--hist_bins", type=int, default=16)
    parser.add_argument("--allow_white_background_for_isar", action="store_true")
    args = get_combined_args(parser)

    os.makedirs(args.output_dir, exist_ok=True)

    gaussians = GaussianModel(args.sh_degree)
    scene_obj = Scene(args, gaussians, load_iteration=args.load_iteration, shuffle=False, resolution_scales=[1.0])
    cams = scene_obj.getTrainCameras(1.0) if args.camera_split == "train" else scene_obj.getTestCameras(1.0)
    if not cams:
        raise RuntimeError(f"No cameras in split: {args.camera_split}")

    if args.camera_index < 0 or args.camera_index >= len(cams):
        raise ValueError(f"camera_index out of range: {args.camera_index}, total={len(cams)}")

    cam = cams[args.camera_index]
    pipe = pipeline_params.extract(args)
    bg, bg_info = resolve_background(args)

    gt = to_3ch(cam.original_image.cuda())
    gt_intensity = to_gray(gt)
    save_tensor_image(os.path.join(args.output_dir, "gt.png"), gt)
    save_tensor_gray_image(os.path.join(args.output_dir, "gt_intensity.png"), gt_intensity)

    results = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "loaded_iteration": scene_obj.loaded_iter,
        "camera_split": args.camera_split,
        "camera_index": args.camera_index,
        "camera_name": cam.image_name,
        "background": bg_info,
        "gt_intensity": summarize_intensity(gt_intensity, gt_intensity, args.hist_bins),
        "modes": {},
        "modes_intensity": {},
        "modes_intensity_obs": {},
    }

    original_mode = getattr(cam, "projection_mode", "perspective")
    original_ox = float(getattr(cam, "ortho_scale_x", 1.0))
    original_oy = float(getattr(cam, "ortho_scale_y", 1.0))
    original_isar_window = float(getattr(cam, "isar_window_size", 1.0))

    mode_images = {}
    mode_intensity_images = {}

    try:
        for mode in ["perspective", "isar"]:
            cam.projection_mode = mode
            cam.ortho_scale_x = original_ox
            cam.ortho_scale_y = original_oy
            cam.isar_window_size = original_isar_window

            with torch.no_grad():
                out = render(cam, gaussians, pipe, bg)
                img = to_3ch(out["render"])
                img_intensity = to_gray(img)

            save_tensor_image(os.path.join(args.output_dir, f"render_{mode}.png"), img)
            save_tensor_gray_image(os.path.join(args.output_dir, f"render_{mode}_intensity.png"), img_intensity)
            mode_images[mode] = img
            mode_intensity_images[mode] = img_intensity
            results["modes"][mode] = summarize(img, gt)
            results["modes_intensity"][mode] = summarize_intensity(img_intensity, gt_intensity, args.hist_bins)
            
            # Loop over extensible observation operators
            for obs_op in ["log1p"]: 
                if obs_op not in results["modes_intensity_obs"]:
                    results["modes_intensity_obs"][obs_op] = {}
                    
                img_obs = apply_isar_observation_operator(img_intensity, obs_op)
                gt_obs = apply_isar_observation_operator(gt_intensity, obs_op)
                results["modes_intensity_obs"][obs_op][mode] = summarize_intensity(img_obs, gt_obs, args.hist_bins)
                
                # Normalize log1p image to 0-1 roughly for visualization
                if obs_op == "log1p":
                    img_obs_vis = img_obs / 0.693147
                    save_tensor_gray_image(os.path.join(args.output_dir, f"render_{mode}_intensity_{obs_op}.png"), img_obs_vis)
        diff = torch.abs(mode_images["perspective"] - mode_images["isar"])
        save_tensor_image(os.path.join(args.output_dir, "render_absdiff.png"), diff)
        results["perspective_vs_isar_absdiff"] = {
            "mean": float(diff.mean().item()),
            "max": float(diff.max().item()),
            "std": float(diff.std().item()),
        }

        diff_intensity = torch.abs(mode_intensity_images["perspective"] - mode_intensity_images["isar"])
        save_tensor_gray_image(os.path.join(args.output_dir, "render_absdiff_intensity.png"), diff_intensity)
        results["perspective_vs_isar_absdiff_intensity"] = {
            "mean": float(diff_intensity.mean().item()),
            "max": float(diff_intensity.max().item()),
            "std": float(diff_intensity.std().item()),
            "histogram": histogram_summary(diff_intensity, bins=args.hist_bins),
        }
    finally:
        cam.projection_mode = original_mode
        cam.ortho_scale_x = original_ox
        cam.ortho_scale_y = original_oy
        cam.isar_window_size = original_isar_window

    out_json = os.path.join(args.output_dir, "stats.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[STAGE-H-COMPARE] saved outputs to:", args.output_dir)
    print("[STAGE-H-COMPARE] background:", json.dumps(bg_info))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()