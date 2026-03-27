import os
import traceback
from argparse import ArgumentParser

from arguments import ModelParams
import scene as scene_module
from scene import Scene, GaussianModel


def main():
    parser = ArgumentParser()
    model_params = ModelParams(parser)

    args = parser.parse_args([
        "--source_path", "D:/3DGS_new/3DGS_DATA/isar_Hubble1_aztest",
        "--model_path", "D:/3DGS_new/gaussian-splatting/output/isar_scene_smoke",
        "--images", "images",
    ])

    os.makedirs(args.model_path, exist_ok=True)

    call_counter = {"count": 0}
    original_camera_list_fn = scene_module.cameraList_from_camInfos

    def wrapped_camera_list_from_camInfos(*fn_args, **fn_kwargs):
        call_counter["count"] += 1
        return original_camera_list_fn(*fn_args, **fn_kwargs)

    scene_module.cameraList_from_camInfos = wrapped_camera_list_from_camInfos

    print("[SMOKE] Building Scene from ISAR dataset...")
    print(f"[SMOKE] source_path={args.source_path}")
    print(f"[SMOKE] model_path={args.model_path}")

    try:
        gaussians = GaussianModel(args.sh_degree)
        scene_obj = Scene(args, gaussians, load_iteration=None, shuffle=False, resolution_scales=[1.0])

        train_cams = scene_obj.getTrainCameras(1.0)
        test_cams = scene_obj.getTestCameras(1.0)

        print("[SMOKE] PASS")
        print(f"train cameras={len(train_cams)}")
        print(f"test cameras={len(test_cams)}")

        first_image = train_cams[0].image_name if train_cams else "none"
        print(f"first image_name={first_image}")
        print(f"entered cameraList_from_camInfos={call_counter['count'] > 0} (calls={call_counter['count']})")

        if train_cams:
            cam0 = train_cams[0]
            print(f"first camera image_name={cam0.image_name}")
            print(f"first camera width={cam0.image_width}, height={cam0.image_height}")

            # original_image is produced by camera_utils.loadCam -> PILtoTorch path.
            img = cam0.original_image
            print(f"loaded image tensor shape={tuple(img.shape)}")

            channels = int(img.shape[0]) if img.ndim >= 3 else 1
            print(f"loaded image channels={channels}")

            # Temporary compatibility behavior check:
            # current Camera path expects RGB tensor layout [3, H, W].
            if channels == 3:
                print("single-channel tif compatibility note: current chain is using RGB-compatible tensor (3 channels).")
            elif channels == 1:
                print("single-channel tif compatibility note: image remained single-channel in current chain.")
            else:
                print("single-channel tif compatibility note: unexpected channel count, please inspect loading path.")

    except Exception as e:
        print("[SMOKE] FAIL")
        print(f"error_type={type(e).__name__}")
        print(f"error_message={e}")
        traceback.print_exc()
    finally:
        scene_module.cameraList_from_camInfos = original_camera_list_fn


if __name__ == "__main__":
    main()
