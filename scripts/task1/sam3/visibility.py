"""Cache per-camera rendered mass and select informative Gaussian observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def informative_observations(indices, mass, peak_mass, absolute_floor, relative_floor):
    """Use the same visibility gate for both support votes and their denominator."""
    if not np.isfinite(absolute_floor) or absolute_floor < 0:
        raise ValueError('absolute visibility floor must be finite and nonnegative')
    if not np.isfinite(relative_floor) or not 0 <= relative_floor <= 1:
        raise ValueError('relative visibility floor must be in [0, 1]')
    indices = np.asarray(indices, dtype=np.uint32)
    mass = np.asarray(mass, dtype=np.float32)
    if indices.shape != mass.shape or not np.all(np.isfinite(mass)) or np.any(mass < 0):
        raise ValueError('invalid visibility mass')
    threshold = np.maximum(absolute_floor, relative_floor * peak_mass[indices])
    return indices[(mass > 0) & (mass >= threshold)]


def observation_reliability(indices, mass, peak_mass, absolute_scale, relative_scale):
    """Continuous counterpart of the visibility gate; weak views retain small votes."""
    informative_observations(indices, mass, peak_mass, absolute_scale, relative_scale)
    scale = np.maximum(absolute_scale, relative_scale * peak_mass[indices])
    return np.minimum(1., np.asarray(mass, np.float64) / np.maximum(scale, 1e-30))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--votes-manifest', required=True, type=Path)
    ap.add_argument('--flashsplat-root', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(args.output_dir)
    import torch
    from scripts.task1.common.flashsplat_cameras import (
        load_cameras, load_flashsplat, load_gaussians, make_camera,
        render_flashsplat, default_pipeline, background_tensor)
    from scripts.task1.dinov3.lift_dense_view_votes import flashsplat_class_rows
    from scripts.task1.sam3.provenance import sha256_file
    manifest = json.loads(args.votes_manifest.read_text())
    n = manifest['gaussian_count']
    modules = load_flashsplat(args.flashsplat_root)
    gaussians = load_gaussians(modules, Path(manifest['ply_path']), 3)
    assert len(gaussians.get_xyz) == n
    cameras = load_cameras(Path(manifest['model_path']))
    pipeline, bg = default_pipeline(), background_tensor(False)
    args.output_dir.mkdir(parents=True, exist_ok=args.overwrite)
    peak = np.zeros(n, np.float32)
    frames = []
    with torch.no_grad():
        for k, frame in enumerate(manifest['frames']):
            camera = make_camera(cameras[frame['camera_index']], modules, manifest['render_max_width'])
            mask = torch.zeros((camera.image_height, camera.image_width), device='cuda')
            pkg = render_flashsplat(camera, gaussians, modules, pipeline, bg, gt_mask=mask, obj_num=1)
            mass = flashsplat_class_rows(pkg['used_count'].detach().float().cpu().numpy(), 1, n)[0]
            indices = np.flatnonzero(mass > 0).astype(np.uint32)
            with np.load(args.votes_manifest.parent / frame['vote_file']) as z:
                if not np.array_equal(indices, z['observed']):
                    raise RuntimeError('cached mass observation set differs from the source lift')
            if not np.all(np.isfinite(mass)):
                raise RuntimeError('nonfinite visibility')
            np.maximum(peak, mass, out=peak)
            name = Path(frame['file']).stem + '.npz'
            np.savez_compressed(args.output_dir / name, indices=indices, mass=mass[indices])
            frames.append({'file': frame['file'], 'camera_index': frame['camera_index'], 'mass_file': name})
            del pkg, mask
            if (k+1) % 16 == 0:
                print('VISIBILITY', k+1, '/', len(manifest['frames']), flush=True)
    np.save(args.output_dir / 'peak_mass.npy', peak)
    result = {'source': 'sam3_rendered_mass_v1', 'votes_manifest': str(args.votes_manifest.resolve()),
              'votes_manifest_sha256': sha256_file(args.votes_manifest), 'gaussian_count': n,
              'mass_units': 'sum of alpha-compositing contributions across pixels',
              'peak_mass_file': 'peak_mass.npy', 'frames': frames}
    (args.output_dir / 'visibility_manifest.json').write_text(json.dumps(result, indent=2))
    print('VISIBILITY_COMPLETE', json.dumps({'peak_quantiles': np.quantile(peak[peak>0], [0,.1,.5,.9,1]).tolist()}), flush=True)


if __name__ == '__main__':
    main()
