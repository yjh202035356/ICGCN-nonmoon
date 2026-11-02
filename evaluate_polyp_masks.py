"""Evaluate grayscale prediction maps using SAM2-UNet's metric implementation.

Published-style metrics average over 256 thresholds and then samples.
Fixed-0.5 metrics refer to the supplied stored map; this may be min/max
normalized and is not necessarily the model's original sigmoid probability.
"""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import py_sod_metrics
from scripts.prepare_polyp1450 import TEST_COUNTS as DATASETS, index as image_index

def evaluate(pred_root, gt_root, method, provenance, output, raw_root=None, raw_key=None):
    if bool(raw_root) != bool(raw_key):
        raise ValueError('Supply both raw_root and raw_key, or neither')
    results=[]
    for dataset, expected in DATASETS.items():
        preds=image_index(Path(pred_root)/dataset)
        gts=image_index(Path(gt_root)/dataset/'masks')
        if len(gts)!=expected or preds.keys()!=gts.keys():
            raise ValueError(f'{dataset}: expected {expected} complete matching predictions; '
                             f'got {len(preds)} predictions / {len(gts)} masks; '
                             f'missing={sorted(gts.keys()-preds.keys())[:5]}, '
                             f'extra={sorted(preds.keys()-gts.keys())[:5]}')
        fm=py_sod_metrics.FmeasureV2(metric_handlers={
            'dice':py_sod_metrics.DICEHandler(with_adaptive=False,with_dynamic=True),
            'iou':py_sod_metrics.IOUHandler(with_adaptive=False,with_dynamic=True)})
        fixed_dice, fixed_iou = [], []
        raw_dice, raw_iou = [], []
        for stem in sorted(gts):
            with Image.open(preds[stem]) as im: pred=np.asarray(im.convert('L'))
            with Image.open(gts[stem]) as im: gt=np.asarray(im.convert('L'))
            if pred.shape!=gt.shape:
                raise ValueError(f'{dataset}/{stem}: predictions must be at GT size, got {pred.shape}/{gt.shape}')
            fm.step(pred=pred, gt=gt)
            p, g = pred.astype(np.float64)/255 >= .5, gt>127
            inter=np.logical_and(p,g).sum()
            total=p.sum()+g.sum(); union=np.logical_or(p,g).sum()
            fixed_dice.append(float(2*inter/total) if total else 1.)
            fixed_iou.append(float(inter/union) if union else 1.)
            if raw_root:
                with np.load(Path(raw_root)/dataset/f'{stem}.npz') as arrays:
                    probability=arrays[raw_key]
                if probability.shape != gt.shape or not np.isfinite(probability).all():
                    raise ValueError(f'Invalid raw probability: {dataset}/{stem}')
                if probability.min()<0 or probability.max()>1:
                    raise ValueError(f'Probability outside [0,1]: {dataset}/{stem}')
                raw_pred=probability>=.5
                raw_inter=np.logical_and(raw_pred,g).sum()
                raw_total=raw_pred.sum()+g.sum()
                raw_union=np.logical_or(raw_pred,g).sum()
                raw_dice.append(float(2*raw_inter/raw_total) if raw_total else 1.)
                raw_iou.append(float(raw_inter/raw_union) if raw_union else 1.)
        measured=fm.get_results()
        row={'method':method,'provenance':provenance,'dataset':dataset,'n':expected,
             'threshold_mean_dice':float(measured['dice']['dynamic'].mean()),
             'threshold_mean_iou':float(measured['iou']['dynamic'].mean()),
             'stored_map_dice_at_0.5':float(np.mean(fixed_dice)),
             'stored_map_iou_at_0.5':float(np.mean(fixed_iou))}
        if raw_root:
            row.update(raw_probability_dice_at_0_5=float(np.mean(raw_dice)),
                       raw_probability_iou_at_0_5=float(np.mean(raw_iou)))
        results.append(row); print(json.dumps(row),flush=True)
    out=Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    out.with_suffix('.json').write_text(json.dumps({
        'pysodmetrics_version':importlib.metadata.version('pysodmetrics'),
        'protocol':'SAM2-UNet eval.py: FmeasureV2 dynamic Dice/IoU averaged across 256 thresholds and images',
        'fixed_threshold_note':'0.5 applies to stored grayscale map; not guaranteed raw sigmoid probability',
        'results':results,
        'macro_mean':{k:float(np.mean([r[k] for r in results])) for k in
                      ['threshold_mean_dice','threshold_mean_iou','stored_map_dice_at_0.5','stored_map_iou_at_0.5']
                      + (['raw_probability_dice_at_0_5','raw_probability_iou_at_0_5'] if raw_root else [])}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pred-root',required=True)
    p.add_argument('--gt-root',default='data/TestDataset')
    p.add_argument('--method',required=True)
    p.add_argument('--provenance',choices=['author_precomputed_maps','local_inference'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--raw-root')
    p.add_argument('--raw-key',choices=['semantic','structural','naive_average','resolver_v2'])
    a=p.parse_args();evaluate(a.pred_root,a.gt_root,a.method,a.provenance,a.output,a.raw_root,a.raw_key)
