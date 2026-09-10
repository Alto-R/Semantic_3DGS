"""Validate fresh-camera and reused-camera entry contracts without GPU work."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest

pytestmark=pytest.mark.skipif(shutil.which('bash') is None,reason='requires bash')
ROOT=Path(__file__).resolve().parents[1]


def config(tmp_path,**settings):
    env={k:v for k,v in os.environ.items() if k not in ['RGB_DIR','RENDER_MANIFEST']}
    env.update(PROJECT_ROOT=str(ROOT),SCENE='old_street',SAM3_MODEL_REVISION='test',
               TASK1_ROOT=str(tmp_path),CONFIG_ONLY='1')
    env.update(settings)
    return subprocess.run(['bash','scripts/slurm/slurm_task1_sam3_instance_scene.sbatch'],
                          cwd=ROOT,env=env,capture_output=True,text=True)


def test_pipeline_can_start_from_reconstruction(tmp_path):
    p=config(tmp_path)
    assert p.returncode==0,p.stderr
    assert 'render_inputs=1' in p.stdout
    assert '/stages/00_real_camera_views/rgb_renders' in p.stdout


def test_pipeline_reuses_both_rgb_and_manifest(tmp_path):
    p=config(tmp_path,RGB_DIR='/existing/rgb',RENDER_MANIFEST='/existing/views.json')
    assert p.returncode==0,p.stderr
    assert 'render_inputs=0' in p.stdout
    assert 'rgb_dir=/existing/rgb' in p.stdout


def test_pipeline_rejects_partial_reuse_configuration(tmp_path):
    p=config(tmp_path,RGB_DIR='/existing/rgb')
    assert p.returncode==2
    assert 'Set both RGB_DIR and RENDER_MANIFEST' in p.stderr
