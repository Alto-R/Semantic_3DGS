"""Pool same-concept evidence without inventing a resolved global instance ID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.task1.sam3.instance_membership import MembershipCSR, load_membership, save_membership


def pool_concepts(membership, concept_lookup, threshold=.4):
    if not np.isfinite(threshold) or not 0 <= threshold < 1:
        raise ValueError('semantic threshold must be in [0, 1)')
    n = len(membership.indptr)-1
    g = np.repeat(np.arange(n,dtype=np.int64),np.diff(membership.indptr))
    concepts = np.asarray(concept_lookup)[membership.instance_ids]
    if np.any(concepts <= 0):
        raise ValueError('every instance must map to a positive concept id')
    stride = int(np.max(concept_lookup,initial=0))+1
    key = g*stride+concepts
    order = np.argsort(key,kind='stable')
    key = key[order]
    starts = np.r_[0,np.flatnonzero(key[1:]!=key[:-1])+1] if len(key) else np.zeros(0,np.int64)
    obs = membership.observe_counts[order][starts]
    counts = np.add.reduceat(membership.support_counts[order].astype(np.int32),starts) if len(starts) else np.zeros(0,np.int32)
    # S4 emits at most one winning mask per (Gaussian, camera, concept).
    # Therefore pooling IDs does not double-count synonym detections.
    if np.any(counts > obs):
        raise RuntimeError('pooled concept votes exceed observing cameras')
    sw = membership.support_weights if membership.support_weights is not None else membership.support_counts.astype(float)
    ow = membership.observe_weights if membership.observe_weights is not None else membership.observe_counts.astype(float)
    support = np.add.reduceat(sw[order],starts) if len(starts) else np.zeros(0)
    observed = ow[order][starts]
    if np.any(support > observed + 1e-8):
        raise RuntimeError('pooled reliability exceeds observing reliability')
    scores = support/observed
    status = np.where(scores > threshold,1,3).astype(np.uint8)
    status[counts < 2] = 2
    pooled_g = key[starts]//stride
    ptr = np.r_[0,np.cumsum(np.bincount(pooled_g,minlength=n))]
    return MembershipCSR(ptr,(key[starts]%stride).astype(np.uint16),counts.astype(np.uint16),
                         obs,scores.astype(np.float32),status,support,observed)


def project_labels(membership, object_ids=()):
    n = len(membership.indptr)-1
    g = np.repeat(np.arange(n,dtype=np.int64),np.diff(membership.indptr))
    keep = membership.status==1
    g,ids,scores = g[keep],membership.instance_ids[keep],membership.scores[keep]
    sizes = np.bincount(ids,minlength=int(ids.max(initial=0))+1)
    priority = np.isin(ids,object_ids).astype(np.int8)
    order = np.lexsort((ids,sizes[ids],-scores,-priority,g))
    g,ids = g[order],ids[order]
    labels = np.zeros(n,np.uint16)
    first = np.r_[True,g[1:]!=g[:-1]] if len(g) else np.zeros(0,bool)
    labels[g[first]] = ids[first]
    return labels


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--membership',type=Path,required=True)
    ap.add_argument('--registry',type=Path,required=True)
    ap.add_argument('--vocabulary',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--threshold',type=float,default=.4)
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args(argv)
    if args.output_dir.exists() and not args.overwrite:raise FileExistsError(args.output_dir)
    from scripts.task1.sam3.provenance import sha256_file
    from scripts.task1.common.semantic_palette import rgb_for_class
    membership=load_membership(args.membership)
    reg=json.loads(args.registry.read_text());vocab=json.loads(args.vocabulary.read_text())
    roles={p['phrase']:p['role'] for p in vocab['phrases']}
    names=sorted(roles);mapping={c:i+1 for i,c in enumerate(names)}
    lut=np.zeros(65536,np.uint16)
    for inst in reg['instances']:lut[inst['instance_id']]=mapping[inst['concept']]
    pooled=pool_concepts(membership,lut,args.threshold)
    labels=project_labels(pooled)
    object_ids=[mapping[c] for c in names if roles[c]=='gnn_node']
    object_first=project_labels(pooled,object_ids)
    original_instance=project_labels(membership)
    unresolved=(labels>0)&(original_instance==0)
    args.output_dir.mkdir(parents=True,exist_ok=args.overwrite)
    save_membership(args.output_dir/'concept_membership.npz',pooled)
    np.save(args.output_dir/'gaussian_labels.npy',labels)
    np.save(args.output_dir/'gaussian_labels_objects_first.npy',object_first)
    np.save(args.output_dir/'instance_unresolved.npy',unresolved)
    legend=[{'id':mapping[c],'concept':c,'role':roles[c],'rgb':list(rgb_for_class(c))} for c in names]
    (args.output_dir/'label_map.json').write_text(json.dumps({'source':'sam3_pooled_concept_labels','labels':legend},indent=2))
    g=np.repeat(np.arange(len(labels)),np.diff(pooled.indptr));accepted=pooled.status==1
    stats=[]
    for row in legend:
        c=row['id'];stats.append({**row,'flat_gaussians':int((labels==c).sum()),
            'objects_first_gaussians':int((object_first==c).sum()),
            'multilabel_gaussians':int(((pooled.instance_ids==c)&accepted).sum())})
    sweep=[]
    for t in [.5,.4,.3]:
        flag=np.zeros(len(labels),bool);flag[g[(pooled.support_counts>=2)&(pooled.support_weights>t*pooled.observe_weights)]]=True
        sweep.append({'threshold':t,'labeled_gaussians':int(flag.sum()),'coverage':float(flag.mean())})
    summary={'source':'sam3_pooled_concept_consensus','contract':'visibility_weighted_per_concept_two_camera_v1',
        'membership_sha256':sha256_file(args.membership),'registry_sha256':sha256_file(args.registry),
        'threshold':args.threshold,'comparison':'strictly_greater','minimum_support_cameras':2,
        'gaussian_count':len(labels),'labeled_gaussians':int((labels>0).sum()),'coverage':float((labels>0).mean()),
        'instance_labeled_gaussians':int((original_instance>0).sum()),
        'class_labeled_but_instance_unresolved':int(unresolved.sum()),
        'pooling':'sum same-concept instance support weights; one winner per Gaussian/camera/concept before pooling',
        'display':'score-first and objects-first projections supplied separately; same accepted support coverage',
        'concepts':stats,'threshold_sweep':sweep}
    (args.output_dir/'semantic_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['concepts']},indent=2))


if __name__=='__main__':main()
