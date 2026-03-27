import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss


def to_3ch(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 3 and img.shape[0] == 1:
        return img.repeat(3, 1, 1)
    return img


def save_tensor_image(path: str, image_chw: torch.Tensor) -> None:
    image = torch.clamp(image_chw.detach().cpu(), 0.0, 1.0)
    image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


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


def main() -> None:
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--camera_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./output/stage_h_compare")
    args = parser.parse_args()

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
    bg = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    gt = to_3ch(cam.original_image.cuda())
    save_tensor_image(os.path.join(args.output_dir, "gt.png"), gt)

    results = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "loaded_iteration": scene_obj.loaded_iter,
        "camera_split": args.camera_split,
        "camera_index": args.camera_index,
        "camera_name": cam.image_name,
        "modes": {},
    }

    original_mode = getattr(cam, "projection_mode", "perspective")
    original_ox = float(getattr(cam, "ortho_scale_x", 1.0))
    original_oy = float(getattr(cam, "ortho_scale_y", 1.0))
    original_isar_window = float(getattr(cam, "isar_window_size", 1.0))

    mode_images = {}

    try:
        for mode in ["perspective", "isar"]:
            cam.projection_mode = mode
            cam.ortho_scale_x = original_ox
            cam.ortho_scale_y = original_oy
            cam.isar_window_size = original_isar_window

            with torch.no_grad():
                out = render(cam, gaussians, pipe, bg)
                img = to_3ch(out["render"])

            save_tensor_image(os.path.join(args.output_dir, f"render_{mode}.png"), img)
            mode_images[mode] = img
            results["modes"][mode] = summarize(img, gt)

        diff = torch.abs(mode_images["perspective"] - mode_images["isar"])
        save_tensor_image(os.path.join(args.output_dir, "render_absdiff.png"), diff)
        results["perspective_vs_isar_absdiff"] = {
            "mean": float(diff.mean().item()),
            "max": float(diff.max().item()),
            "std": float(diff.std().item()),
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
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()