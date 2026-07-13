"""Shared helpers for running FlashSplat directly from GraphDeco cameras.json."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


CameraItem = Tuple[int, Dict[str, Any]]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_GRAPHDECO_ROOT = WORKSPACE_ROOT / "external" / "gaussian-splatting"


def parse_indices(value: str) -> List[int]:
    indices: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            indices.append(int(part))
    return indices


def evenly_spaced_indices(total: int, count: int) -> List[int]:
    if total <= 0:
        raise ValueError("No cameras found")
    if count <= 0:
        raise ValueError("--count must be positive")
    if count >= total:
        return list(range(total))
    return sorted({int(round(value)) for value in np.linspace(0, total - 1, count)})


def selected_camera_items(cameras: Sequence[Dict[str, Any]], camera_indices: str, count: int) -> List[CameraItem]:
    if camera_indices:
        indices = parse_indices(camera_indices)
    else:
        indices = evenly_spaced_indices(len(cameras), count)

    items: List[CameraItem] = []
    for index in indices:
        if index < 0 or index >= len(cameras):
            raise IndexError(f"Camera index {index} is outside 0..{len(cameras) - 1}")
        items.append((index, cameras[index]))
    return items


def ensure_camera_item(items: List[CameraItem], cameras: Sequence[Dict[str, Any]], camera_index: int) -> List[CameraItem]:
    if any(index == camera_index for index, _camera in items):
        return items
    if camera_index < 0 or camera_index >= len(cameras):
        raise IndexError(f"Camera index {camera_index} is outside 0..{len(cameras) - 1}")
    return [(camera_index, cameras[camera_index]), *items]


def camera_filename(output_index: int, camera_json: Dict[str, Any]) -> str:
    return f"{output_index:05d}_cam{int(camera_json['id']):04d}.png"


def scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if max_width <= 0 or width <= max_width:
        return width, height
    scale = max_width / float(width)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def load_cameras(model_path: Path) -> List[Dict[str, Any]]:
    camera_path = model_path / "cameras.json"
    if not camera_path.exists():
        raise FileNotFoundError(camera_path)
    return json.loads(camera_path.read_text(encoding="utf-8"))


def point_cloud_path(model_path: Path, iteration: int) -> Path:
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    return ply_path


def load_flashsplat(
    flashsplat_root: Path,
    graphdeco_root: Path = DEFAULT_GRAPHDECO_ROOT,
) -> Dict[str, Any]:
    flashsplat_root = flashsplat_root.resolve()
    graphdeco_root = graphdeco_root.resolve()
    paths = [
        flashsplat_root / "submodules" / "flashsplat-rasterization",
        graphdeco_root / "submodules" / "diff-gaussian-rasterization",
        flashsplat_root,
    ]
    for path in reversed(paths):
        sys.path.insert(0, str(path))

    from gaussian_renderer import flashsplat_render
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import focal2fov, getProjectionMatrix, getWorld2View2

    return {
        "flashsplat_render": flashsplat_render,
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


def default_pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
        antialiasing=False,
    )


def load_gaussians(modules: Dict[str, Any], ply_path: Path, sh_degree: int) -> Any:
    gaussians = modules["GaussianModel"](sh_degree)
    gaussians.load_ply(str(ply_path))
    return gaussians


def background_tensor(white_background: bool) -> torch.Tensor:
    return torch.tensor(
        [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )


def render_flashsplat(
    camera: Any,
    gaussians: Any,
    modules: Dict[str, Any],
    pipeline: SimpleNamespace,
    background: torch.Tensor,
    gt_mask: Optional[torch.Tensor] = None,
    override_color: Optional[torch.Tensor] = None,
    obj_num: int = 1,
) -> Dict[str, Any]:
    return modules["flashsplat_render"](
        camera,
        gaussians,
        pipeline,
        background,
        override_color=override_color,
        gt_mask=gt_mask,
        obj_num=obj_num,
    )


def tensor_to_rgb_array(render: torch.Tensor) -> np.ndarray:
    image = render.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0 + 0.5).astype(np.uint8)


def iter_camera_items(
    cameras: Sequence[Dict[str, Any]],
    camera_indices: str,
    count: int,
) -> Iterable[tuple[int, int, Dict[str, Any]]]:
    for output_index, (camera_index, camera_json) in enumerate(
        selected_camera_items(cameras, camera_indices, count)
    ):
        yield output_index, camera_index, camera_json
