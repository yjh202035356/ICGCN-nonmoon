#!/usr/bin/env python3
"""Select real Kvasir cases with similarly high Dice for 3 segmentation models.

Uses the *same 256x256 padded canvas and mask decoding* as
scripts/make_three_model_qualitative_kvasir.py. Runs on CPU, without inference.

This script ranks only genuine saved predictions, never modifies masks or images.
The selection is descriptive, and should be identified as a chosen qualitative
example rather than a representative estimate of each model's performance.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from make_three_model_qualitative_kvasir import (
    IMAGE_EXT,
    PRED_EXT,
    align_sources,
    dice,
    index_files,
    read_gt,
    read_prediction,
    read_rgb,
    render_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', default='data/Kvasir/test/images')
    parser.add_argument('--gts', default='data/Kvasir/test/labels')
    parser.add_argument('--simple', default='qualitative_outputs/simple_vilmedsam')
    parser.add_argument('--psf', default='qualitative_outputs/psf_sam')
    parser.add_argument('--crss', default='qualitative_outputs/crss')
    for name in ('simple', 'psf', 'crss'):
        parser.add_argument(f'--{name}-kind', default='auto',
                            choices=['auto', 'probability', 'logits', 'binary'])
        parser.add_argument(f'--{name}-key', default=None)
    parser.add_argument('--min-dice', type=float, default=0.89,
                        help='Every model must score at least this Dice (default: 0.89)')
    parser.add_argument('--max-gap', type=float, default=None,
                        help='Optional maximum (highest - lowest) model Dice for a case')
    parser.add_argument('--top', type=int, default=3,
                        help='Number of best qualifying cases to render (default: 3)')
    parser.add_argument('--out', default='qualitative_outputs/three_model_high_dice')
    parser.add_argument('--dpi', type=int, default=240)
    parser.add_argument('--no-dice', action='store_true', default=True,
                        help='Hide per-row Dice in figures (default: hidden)')
    parser.add_argument('--show-dice', action='store_true',
                        help='Show Dice on figure (otherwise only printed and in CSV)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_dice <= 1.0:
        raise ValueError('--min-dice must be in [0,1]')
    if args.max_gap is not None and not 0 <= args.max_gap <= 1:
        raise ValueError('--max-gap must be in [0,1]')
    if args.top < 1:
        raise ValueError('--top must be >= 1')

    specs = [
        ('Input', args.images, IMAGE_EXT),
        ('Ground Truth', args.gts, IMAGE_EXT),
        ('Simple-ViLMedSAM', args.simple, PRED_EXT),
        ('PSF-SAM', args.psf, PRED_EXT),
        ('CRSS-SAM (Ours)', args.crss, PRED_EXT),
    ]
    idx = [index_files(path, extensions, label) for label, path, extensions in specs]
    cases = sorted(set.intersection(*(set(x) for x in idx)))
    if not cases:
        raise RuntimeError('No common test image IDs across images, GT and all 3 predictions.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for stem in cases:
        rgb = read_rgb(idx[0][stem])
        gt = read_gt(idx[1][stem], rgb.shape[:2])
        raw_preds = [
            read_prediction(idx[2][stem], args.simple_key, args.simple_kind),
            read_prediction(idx[3][stem], args.psf_key, args.psf_kind),
            read_prediction(idx[4][stem], args.crss_key, args.crss_kind),
        ]
        _, gt, preds, canvas = align_sources(
            rgb, gt, raw_preds, ['Simple-ViLMedSAM', 'PSF-SAM', 'CRSS-SAM']
        )
        scores = [dice(pred, gt) for pred in preds]
        spread = max(scores) - min(scores)
        qualified = (min(scores) >= args.min_dice and
                     (args.max_gap is None or spread <= args.max_gap))
        rows.append({
            'case': stem,
            'simple_dice': scores[0],
            'psf_dice': scores[1],
            'crss_dice': scores[2],
            'mean_dice': sum(scores) / 3,
            'min_dice': min(scores),
            'max_minus_min': spread,
            'canvas': canvas,
            'passes_filter': qualified,
        })

    # Smallest gap first, then highest mean Dice. No use of image-level content in ranking.
    rows.sort(key=lambda r: (not r['passes_filter'], r['max_minus_min'],
                             -r['mean_dice'], r['case']))
    csv_file = out / 'all_200_cases_dice.csv'
    with csv_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    good = [row for row in rows if row['passes_filter']]
    print(f'Common cases: {len(cases)}')
    print(f'Cases with all three Dice >= {args.min_dice:.3f}' +
          (f' and gap <= {args.max_gap:.3f}' if args.max_gap is not None else '') +
          f': {len(good)}')
    print(f'All case scores saved: {csv_file}')
    if not good:
        best = sorted(rows, key=lambda r: (-r['min_dice'], r['max_minus_min']))[:5]
        print('No cases meet filter. Best cases by minimum model Dice:')
        for row in best:
            print(f"  {row['case']}: simple={row['simple_dice']:.4f}, "
                  f"psf={row['psf_dice']:.4f}, crss={row['crss_dice']:.4f}, "
                  f"gap={row['max_minus_min']:.4f}")
        print('Do not present a lower-scoring case as meeting the threshold.')
        return

    print('Best matching cases (minimum gap among high-Dice predictions):')
    for i, row in enumerate(good[:args.top], start=1):
        print(f"  {i}. {row['case']} | Simple={row['simple_dice']:.4f} "
              f"PSF={row['psf_dice']:.4f} CRSS={row['crss_dice']:.4f} "
              f"gap={row['max_minus_min']:.4f}")
        methods = [
            ('Simple-ViLMedSAM', idx[2][row['case']], args.simple_key, args.simple_kind),
            ('PSF-SAM', idx[3][row['case']], args.psf_key, args.psf_kind),
            ('CRSS-SAM (Ours)', idx[4][row['case']], args.crss_key, args.crss_kind),
        ]
        render_case(row['case'], idx[0][row['case']], idx[1][row['case']],
                    methods, out / f"{row['case']}_three_models.png",
                    args.dpi, thickness=2, show_dice=args.show_dice)


if __name__ == '__main__':
    main()
