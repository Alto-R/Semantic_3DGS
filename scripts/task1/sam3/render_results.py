"""Render final SAM3 semantic/instance labels through the original 3D Gaussians."""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def overlay(rgb, labels):
    result = rgb.copy()
    keep = labels.max(axis=-1) > 8
    result[keep] = (.35 * rgb[keep] + .65 * labels[keep]).astype(np.uint8)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--render-manifest', type=Path, required=True)
    ap.add_argument('--rgb-dir', type=Path, required=True)
    ap.add_argument('--flashsplat-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--layers', default='window,signboard,storefront')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(args.output_dir)

    import torch
    from scripts.task1.common.flashsplat_cameras import (
        load_cameras, load_flashsplat, load_gaussians, make_camera,
        render_flashsplat, default_pipeline, background_tensor, tensor_to_rgb_array)
    from scripts.task1.common.semantic_palette import rgb_for_class
    from scripts.task1.sam3.instance_membership import load_membership
    from scripts.task1.sam3.provenance import sha256_file

    run = args.run_dir
    votes = json.loads((run/'stages/02_mask_votes/sam3_vote_manifest.json').read_text())
    render_manifest = json.loads(args.render_manifest.read_text())
    graph = json.loads((run/'stages/05_scene_graph/scene_graph.json').read_text())
    instances = np.load(run/'stages/05_scene_graph/gaussian_instances.npy')
    n = votes['gaussian_count']
    if instances.shape != (n,):
        raise ValueError('instance labels differ from reconstruction size')
    instance_lut = np.zeros((int(instances.max(initial=0))+1,3),np.float32)
    fallback_lut = instance_lut.copy()
    for node in graph['nodes']:
        i = node['instance_id']
        if i < len(instance_lut):
            instance_lut[i] = colorsys.hsv_to_rgb(i*.61803398875%1,.7,.95)
            fallback_lut[i] = rgb_for_class(node['concept'])
    semantic_dir = run/'stages/04_semantic_consensus'
    layers = {}
    if semantic_dir.exists():
        labels = np.load(semantic_dir/'gaussian_labels.npy')
        priority = np.load(semantic_dir/'gaussian_labels_objects_first.npy')
        mapping = json.loads((semantic_dir/'label_map.json').read_text())['labels']
        lut = np.zeros((max(x['id'] for x in mapping)+1,3),np.float32)
        for row in mapping:
            lut[row['id']] = row['rgb']
        membership = load_membership(semantic_dir/'concept_membership.npz')
        gs = np.repeat(np.arange(n),np.diff(membership.indptr))
        for row in mapping:
            if row['concept'] in args.layers.split(','):
                layers[row['concept']] = (gs[(membership.status==1)&(membership.instance_ids==row['id'])],row['rgb'])
        del membership, gs
        semantic_colors, priority_colors = lut[labels], lut[priority]
        label_path = semantic_dir/'gaussian_labels.npy'
    else:
        labels = instances
        semantic_colors = priority_colors = fallback_lut[instances]
        label_path = run/'stages/05_scene_graph/gaussian_instances.npy'
    if labels.shape != (n,):
        raise ValueError('semantic labels differ from reconstruction size')
    cameras = load_cameras(Path(votes['model_path']))
    modules = load_flashsplat(args.flashsplat_root)
    gaussians = load_gaussians(modules,Path(votes['ply_path']),3)
    if len(gaussians.get_xyz) != n:
        raise ValueError('PLY differs from labels')
    pipeline,bg = default_pipeline(),background_tensor(False)
    colors = {name:torch.from_numpy(array).cuda() for name,array in [
        ('semantic',semantic_colors),('objects_first',priority_colors),('instance',instance_lut[instances])]}
    mass_colors = torch.from_numpy(np.column_stack([labels>0,instances>0,np.ones(n)]).astype(np.float32)).cuda()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name in colors:
        for suffix in ['labels','overlays']:
            (args.output_dir/(name+'_'+suffix)).mkdir(exist_ok=True)
    for name in layers:
        (args.output_dir/('layer_'+name)).mkdir(exist_ok=True)
    frames,tiles = [],[]
    by_file = {f['file']:f for f in render_manifest['frames']}
    with torch.no_grad():
        for pos, f in enumerate(votes['frames']):
            if f['file'] not in by_file or by_file[f['file']]['camera_index'] != f['camera_index']:
                raise ValueError('render cameras differ from lift cameras')
            camera = make_camera(cameras[f['camera_index']],modules,votes['render_max_width'])
            rgb = np.asarray(Image.open(args.rgb_dir/f['file']).convert('RGB'))
            if rgb.shape[:2] != (camera.image_height,camera.image_width):
                raise ValueError('RGB size differs from lift size')
            previews = {}
            for name,color in colors.items():
                pkg = render_flashsplat(camera,gaussians,modules,pipeline,bg,override_color=color)
                label = tensor_to_rgb_array(pkg['render']);del pkg
                over = overlay(rgb,label)
                Image.fromarray(label).save(args.output_dir/(name+'_labels')/f['file'])
                Image.fromarray(over).save(args.output_dir/(name+'_overlays')/f['file'])
                if name in ['semantic','objects_first']:previews[name] = (label,over)
            for name,(support,color) in layers.items():
                array = np.zeros((n,3),np.float32);array[support] = color
                pkg = render_flashsplat(camera,gaussians,modules,pipeline,bg,override_color=torch.from_numpy(array).cuda())
                label = tensor_to_rgb_array(pkg['render']);del pkg
                Image.fromarray(overlay(rgb,label)).save(args.output_dir/('layer_'+name)/f['file'])
            pkg = render_flashsplat(camera,gaussians,modules,pipeline,bg,override_color=mass_colors)
            mass = pkg['render'].detach().cpu().numpy();del pkg
            visible = mass[2]>.05
            record = {'file':f['file'],'camera_index':f['camera_index'],
                'mean_semantic_mass_fraction':float((mass[0]/np.maximum(mass[2],1e-8))[visible].mean()) if visible.any() else 0,
                'mean_instance_mass_fraction':float((mass[1]/np.maximum(mass[2],1e-8))[visible].mean()) if visible.any() else 0}
            frames.append(record)
            tile = Image.new('RGB',(600,216),'white')
            for col,arr in enumerate([rgb,previews['semantic'][0],previews['objects_first'][1]]):
                tile.paste(Image.fromarray(arr).resize((200,200)),(col*200,16))
            ImageDraw.Draw(tile).text((3,1),f"{f['file']} semantic mass {record['mean_semantic_mass_fraction']:.1%}",fill='black')
            tiles.append(tile)
            print('RENDERBACK',pos+1,'/',len(votes['frames']),flush=True)
    for start in range(0,len(tiles),24):
        group = tiles[start:start+24]
        sheet = Image.new('RGB',(1800,((len(group)+2)//3)*216),'white')
        for j,tile in enumerate(group):sheet.paste(tile,((j%3)*600,(j//3)*216))
        sheet.save(args.output_dir/f'contact_{start//24+1:02d}.jpg',quality=90)
    summary = {'source':'sam3_final_3d_renderback','gaussian_count':n,'camera_count':len(frames),
        'semantic_labeled_gaussians':int((labels>0).sum()),'instance_labeled_gaussians':int((instances>0).sum()),
        'semantic_labels_sha256':sha256_file(label_path),
        'instance_labels_sha256':sha256_file(run/'stages/05_scene_graph/gaussian_instances.npy'),
        'mean_view_semantic_mass_fraction':float(np.mean([f['mean_semantic_mass_fraction'] for f in frames])),
        'opacity':.65,'frames':frames,'note':'coverage is not ground-truth accuracy'}
    (args.output_dir/'render_summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
