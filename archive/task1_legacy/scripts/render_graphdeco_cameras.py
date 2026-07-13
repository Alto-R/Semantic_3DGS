#!/usr/bin/env python3
"""Render RGB sanity frames from a GraphDeco pretrained 3DGS model.

The official pretrained model folders include `cameras.json`, but the stock
GraphDeco renderer still tries to load source images for ground-truth output.
This utility renders directly from `cameras.json` so a pretrained model can be
validated without the original image tree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
DEFAULT_GRAPHDECO_ROOT = WORKSPACE_ROOT / "external" / "gaussian-splatting"


def parse_indices(value: str) -> List[int]:
    indices: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def evenly_spaced_indices(total: int, count: int) -> List[int]:
    if total <= 0:
        raise ValueError("No cameras found")
    if count <= 0:
        raise ValueError("--count must be positive")
    if count >= total:
        return list(range(total))
    values = np.linspace(0, total - 1, count)
    return sorted({int(round(value)) for value in values})


def scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width, height
    scale = max_width / float(width)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def tensor_to_image(render: torch.Tensor) -> Image.Image:
    image = render.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def load_graphdeco(graphdeco_root: Path) -> Dict[str, Any]:
    rasterizer_root = graphdeco_root / "submodules" / "diff-gaussian-rasterization"
    sys.path.insert(0, str(rasterizer_root))
    sys.path.insert(0, str(graphdeco_root))
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    return {
        "GaussianRasterizationSettings": GaussianRasterizationSettings,
        "GaussianRasterizer": GaussianRasterizer,
        "MiniCam": MiniCam,
        "GaussianModel": GaussianModel,
        "focal2fov": focal2fov,
        "getProjectionMatrix": getProjectionMatrix,
        "getWorld2View2": getWorld2View2,
    }


def make_camera(camera_json: Dict[str, Any], modules: Dict[str, Any], max_width: int) -> Any:
    width = int(camera_json["width"])
    height = int(camera_json["height"])
    render_width, render_height = scaled_size(width, height, max_width)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.asarray(camera_json["rotation"], dtype=np.float64)
    c2w[:3, 3] = np.asarray(camera_json["position"], dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3].T
    translation = w2c[:3, 3]

    fovx = modules["focal2fov"](float(camera_json["fx"]), width)
    fovy = modules["focal2fov"](float(camera_json["fy"]), height)
    world_view_transform = torch.tensor(
        modules["getWorld2View2"](rotation, translation),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    projection_matrix = modules["getProjectionMatrix"](
        znear=0.01,
        zfar=100.0,
        fovX=fovx,
        fovY=fovy,
    ).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)

    return modules["MiniCam"](
        render_width,
        render_height,
        fovy,
        fovx,
        0.01,
        100.0,
        world_view_transform,
        full_proj_transform,
    )


def selected_cameras(cameras: List[Dict[str, Any]], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.camera_indices:
        indices = parse_indices(args.camera_indices)
    else:
        indices = evenly_spaced_indices(len(cameras), args.count)
    for index in indices:
        if index < 0 or index >= len(cameras):
            raise IndexError(f"Camera index {index} is outside 0..{len(cameras) - 1}")
        yield cameras[index]


def render_rgb(viewpoint_camera: Any, pc: Any, modules: Dict[str, Any], pipe: Any, bg_color: torch.Tensor) -> Dict[str, Any]:
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device="cuda")
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    settings_kwargs = {
        "image_height": int(viewpoint_camera.image_height),
        "image_width": int(viewpoint_camera.image_width),
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": bg_color,
        "scale_modifier": 1.0,
        "viewmatrix": viewpoint_camera.world_view_transform,
        "projmatrix": viewpoint_camera.full_proj_transform,
        "sh_degree": pc.active_sh_degree,
        "campos": viewpoint_camera.camera_center,
        "prefiltered": False,
        "debug": pipe.debug,
    }
    if "antialiasing" in getattr(modules["GaussianRasterizationSettings"], "_fields", ()):
        settings_kwargs["antialiasing"] = False
    raster_settings = modules["GaussianRasterizationSettings"](**settings_kwargs)
    rasterizer = modules["GaussianRasterizer"](raster_settings=raster_settings)

    cov3D_precomp = None
    scales = None
    rotations = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(1.0)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    if pipe.convert_SHs_python:
        from utils.sh_utils import eval_sh

        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        directions = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        directions = directions / directions.norm(dim=1, keepdim=True)
        colors_precomp = torch.clamp_min(eval_sh(pc.active_sh_degree, shs_view, directions) + 0.5, 0.0)
        shs = None
    else:
        colors_precomp = None
        shs = pc.get_features

    raster_output = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    rendered_image, radii = raster_output[:2]
    return {
        "render": rendered_image.clamp(0, 1),
        "radii": radii,
        "visibility_filter": (radii > 0).nonzero(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--graphdeco-root",
        default=DEFAULT_GRAPHDECO_ROOT,
        type=Path,
    )
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--sh-degree", default=3, type=int)
    parser.add_argument("--count", default=20, type=int)
    parser.add_argument("--camera-indices", default="")
    parser.add_argument("--max-width", default=1280, type=int)
    parser.add_argument("--white-background", action="store_true")
    args = parser.parse_args()

    camera_path = args.model_path / "cameras.json"
    ply_path = args.model_path / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not camera_path.exists():
        raise FileNotFoundError(camera_path)
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = json.loads(camera_path.read_text(encoding="utf-8"))
    modules = load_graphdeco(args.graphdeco_root)

    gaussians = modules["GaussianModel"](args.sh_degree)
    gaussians.load_ply(str(ply_path))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    pipe = SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
        antialiasing=False,
    )

    manifest = {
        "model_path": str(args.model_path),
        "ply_path": str(ply_path),
        "camera_json": str(camera_path),
        "iteration": args.iteration,
        "max_width": args.max_width,
        "frames": [],
    }

    with torch.no_grad():
        for output_index, camera_json in enumerate(selected_cameras(cameras, args)):
            camera = make_camera(camera_json, modules, args.max_width)
            result = render_rgb(camera, gaussians, modules, pipe, background)
            filename = f"{output_index:05d}_cam{int(camera_json['id']):04d}.png"
            tensor_to_image(result["render"]).save(args.output_dir / filename)
            manifest["frames"].append(
                {
                    "file": filename,
                    "camera_id": int(camera_json["id"]),
                    "image_name": camera_json.get("img_name", ""),
                    "source_width": int(camera_json["width"]),
                    "source_height": int(camera_json["height"]),
                    "render_width": int(camera.image_width),
                    "render_height": int(camera.image_height),
                }
            )
            print(f"wrote {filename}")

    (args.output_dir / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
