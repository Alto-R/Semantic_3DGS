"""Download and verify the fixed ModelScope checkpoint used by the old_street run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    spec=json.loads((Path(__file__).resolve().parents[2]/'configs/sam3_modelscope_files.json').read_text())
    if not args.verify_only:
        subprocess.run([sys.executable,'-m','modelscope.cli.cli','download',spec['repo_id'],
            '--revision',spec['revision'],'--local-dir',str(args.output_dir),
            '--exclude','sam3.pt','--max-workers','2'],check=True)
    for name,expected in spec['verified_files'].items():
        path=args.output_dir/name
        if path.stat().st_size != expected['bytes']:
            raise RuntimeError(f'size mismatch: {name}')
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest != expected['sha256']:raise RuntimeError(f'hash mismatch: {name}')
    (args.output_dir/'verified_provenance.json').write_text(json.dumps(spec,indent=2))
    print('CHECKPOINT_VERIFIED',len(spec['verified_files']),'files; SAM3_MODEL_REVISION='+spec['revision'])


if __name__=='__main__':main()
