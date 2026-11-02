"""Run the pinned official SAM2-UNet on this server's canonical polyp split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
sys.dont_write_bytecode = True

COMMIT = '01598e5e9912ffb23f965ecbebf4d1dfecbaa56e'
COUNTS = {'Kvasir': 100, 'CVC-ClinicDB': 62, 'CVC-ColonDB': 380,
          'ETIS-LaribPolypDB': 196, 'CVC-300': 60}
EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def pair(images, masks, expected, official_loader=False):
    def index(folder):
        out = {}
        for p in sorted(Path(folder).iterdir()):
            if not p.is_file() or p.name.startswith('.') or p.suffix.lower() not in EXT: continue
            if p.stem in out: raise RuntimeError(f'Duplicate stem: {p}')
            out[p.stem] = p
        return out
    im, gt = index(images), index(masks)
    if len(im) != expected or im.keys() != gt.keys():
        raise RuntimeError(f'Expected {expected} matching pairs: {images}, {masks}; got {len(im)}, {len(gt)}')
    if official_loader:
        a, b = sorted(im.values()), sorted(gt.values())
        if any(p.suffix not in {'.jpg', '.png'} for p in a) or any(p.suffix != '.png' for p in b):
            raise RuntimeError('Official loader only supports jpg/png images and png masks')
        if [p.stem for p in a] != [p.stem for p in b]:
            raise RuntimeError('Official loader sorted filenames do not form matching pairs')
    return im, gt

def write(path, content):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train', 'export'])
    p.add_argument('--crss-root', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--train-root', required=True)
    p.add_argument('--test-root', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--work', required=True)
    a = p.parse_args()
    crss, repo, work = [Path(x).resolve() for x in [a.crss_root, a.repo, a.work]]
    current = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if current != COMMIT: raise RuntimeError(f'Wrong official source version: {current}')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Official source must remain unmodified')
    sys.path.insert(0, str(crss))
    sys.path.insert(0, str(repo))
    from PIL import Image
    import numpy as np
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in baseline environment')
    os.chdir(repo)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / 'checkpoints' / 'SAM2-UNet-20.pth'
    manifest = work / 'training_completed.json'
    if a.stage == 'train':
        train = Path(a.train_root).resolve()
        im, gt = pair(train/'images', train/'labels', 1450, official_loader=True)
        from dataset import FullDataset
        ds = FullDataset(str(train/'images')+'/', str(train/'labels')+'/', 352, mode='train')
        if len(ds) != 1450 or len(ds.gts) != 1450: raise RuntimeError('Official loader did not load all 1450 images')
        for stem in im:
            with Image.open(im[stem]) as x, Image.open(gt[stem]) as y:
                if x.size != y.size: raise RuntimeError(f'Image/mask size mismatch: {stem}')
        save = checkpoint.parent
        if save.exists() and any(save.iterdir()):
            raise RuntimeError(f'Existing training output; choose a fresh WORK_ROOT: {save}')
        random.seed(123); np.random.seed(123); torch.manual_seed(123); torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        settings = {'method': 'SAM2-UNet', 'source_commit': COMMIT, 'seed': 123,
                    'epochs': 20, 'batch_size': 12, 'train_images': 1450,
                    'size': 352, 'lr': .001, 'weight_decay': .0005,
                    'selection': 'fixed_final_epoch', 'training_code': 'official train.py unchanged',
                    'loss_note': 'Preserves official structure_loss including legacy reduce=none argument',
                    'pretrained_sha256': sha(a.pretrained),
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0),
                    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'training_image_paths': [str(im[k].resolve()) for k in sorted(im)],
                    'training_mask_paths': [str(gt[k].resolve()) for k in sorted(gt)]}
        write(work/'training_started.json', settings)
        sys.argv = [str(repo/'train.py'), '--hiera_path', str(Path(a.pretrained).resolve()),
                    '--train_image_path', str(train/'images')+'/',
                    '--train_mask_path', str(train/'labels')+'/', '--save_path', str(save),
                    '--epoch', '20', '--lr', '0.001', '--batch_size', '12', '--weight_decay', '0.0005']
        runpy.run_path(str(repo/'train.py'), run_name='__main__')
        if not checkpoint.is_file(): raise RuntimeError('Final checkpoint missing')
        settings['checkpoint_sha256'] = sha(checkpoint)
        settings['status'] = 'completed'
        write(manifest, settings)
        print('TRAIN COMPLETE:', checkpoint, flush=True)
        return
    if not manifest.exists(): raise RuntimeError('Complete local 20-epoch training first')
    provenance = json.loads(manifest.read_text())
    if provenance.get('status') != 'completed' or provenance.get('checkpoint_sha256') != sha(checkpoint):
        raise RuntimeError('Training provenance/checkpoint mismatch')
    output = work/'predictions'
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Prediction output exists; do not overwrite: {output}')
    test = Path(a.test_root).resolve()
    pairs = {}
    for name, count in COUNTS.items():
        base = test/name
        candidates = [base/x for x in ['images', 'image'] if (base/x).is_dir()]
        if len(candidates) != 1: raise RuntimeError(f'Missing/ambiguous images: {base}')
        pairs[name] = pair(candidates[0], base/'masks', count)
    from SAM2UNet import SAM2UNet
    import torch.nn.functional as F
    from tqdm import tqdm
    # Exactly the author's test transform, with explicit stem pairing.
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((352,352)), transforms.ToTensor(),
                                   transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    model = SAM2UNet().cuda()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True); model.eval()
    report = {'status': 'incomplete', 'provenance': 'local_inference',
              'checkpoint_sha256': provenance['checkpoint_sha256'], 'source_commit': COMMIT,
              'epoch': 20, 'counts': {}, 'raw_key': 'semantic',
              'raw_key_note': 'Single SAM2-UNet output uses the existing evaluator semantic array key',
              'restore': '352 logits -> bilinear GT size -> sigmoid; per-image minmax uint8 PNG'}
    write(output/'export_manifest.json', report)
    with torch.no_grad():
        for name, count in COUNTS.items():
            im, gt = pairs[name]
            pngdir, rawdir = output/name, output/'raw_probabilities'/name
            pngdir.mkdir(parents=True, exist_ok=True); rawdir.mkdir(parents=True, exist_ok=True)
            for stem in tqdm(sorted(im), desc=name):
                with Image.open(im[stem]) as x, Image.open(gt[stem]) as g:
                    if x.size != g.size: raise RuntimeError(f'Image/mask size mismatch: {name}/{stem}')
                    tensor = transform(x.convert('RGB')).unsqueeze(0).cuda()
                    hw = (g.height, g.width)
                logits, _, _ = model(tensor)
                raw = F.interpolate(logits, size=hw, mode='bilinear', align_corners=False).sigmoid()[0,0].cpu().numpy().astype(np.float32)
                normalized = (raw-raw.min())/(raw.max()-raw.min()+1e-8)
                Image.fromarray((normalized*255).astype(np.uint8)).save(pngdir/f'{stem}.png')
                np.savez_compressed(rawdir/f'{stem}.npz', semantic=raw)
            report['counts'][name] = count
            write(output/'export_manifest.json', report)
    report['status'] = 'complete'; write(output/'export_manifest.json', report)
    print('EXPORT COMPLETE: 798 local test predictions', flush=True)

if __name__ == '__main__': main()
