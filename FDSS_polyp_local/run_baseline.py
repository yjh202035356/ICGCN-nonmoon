"""FDSS-Net under the canonical local Polyp1450 protocol, without test selection."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
COMMIT = 'ebe9ccaee6c96b5c23a311c14dde3f3222bfee52'
PVT_SHA = '80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d'
COUNTS = {'Kvasir': 100, 'CVC-ClinicDB': 62, 'CVC-ColonDB': 380,
          'ETIS-LaribPolypDB': 196, 'CVC-300': 60}
EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def pair(images, masks, expected, official_loader=False):
    def index(folder):
        out = {}
        for p in sorted(Path(folder).iterdir()):
            if not p.is_file() or p.name.startswith('.') or p.suffix.lower() not in EXT:
                continue
            if p.stem in out:
                raise RuntimeError(f'Duplicate stem: {p}')
            out[p.stem] = p
        return out
    im, gt = index(images), index(masks)
    if len(im) != expected or im.keys() != gt.keys():
        raise RuntimeError(f'Expected {expected} matching pairs; got {len(im)} images / {len(gt)} masks')
    if official_loader:
        a, b = sorted(im.values()), sorted(gt.values())
        if any(p.suffix not in {'.jpg', '.png'} for p in a + b):
            raise RuntimeError('Official training loader only supports lower-case jpg/png')
        if [p.stem for p in a] != [p.stem for p in b]:
            raise RuntimeError('Official loader sorted filenames do not form matching pairs')
    return im, gt


def load_training_module(repo):
    """Preserve the official batch/scale/loss loop; remove only its save/test tail."""
    load_model_module(repo)
    spec = importlib.util.spec_from_file_location('fdss_official_training', repo / 'train.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = ast.parse((repo / 'train.py').read_text())
    function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'train')
    # The final if block calls test() and chooses test-score checkpoints at epoch >= 50.
    tail = function.body[-1]
    if not isinstance(tail, ast.If) or 'meandice = test(' not in ast.unparse(tail):
        raise RuntimeError('Unexpected official training tail; source review required')
    function.body.pop()
    if any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == 'test'
           for x in ast.walk(function)):
        raise RuntimeError('Test call remains in adapted training function')
    adapted = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(adapted)
    exec(compile(adapted, str(repo / 'train.py') + ':no_test_selection', 'exec'), module.__dict__)
    return module, ast.unparse(adapted) + '\n'


def load_model_module(repo):
    if 'lib.pvt' in sys.modules:
        return sys.modules['lib.pvt']
    path = Path(repo) / 'lib/pvt.py'
    source = path.read_text()
    bad = 'nn.BatchNorm2d(self.in_channels)\n'
    if source.count(bad) != 1:
        raise RuntimeError('Unexpected official model syntax; review source instead of guessing')
    # One missing separator in an unused Sequential constructor. No layer/forward changes.
    patched = source.replace(bad, 'nn.BatchNorm2d(self.in_channels),\n', 1)
    spec = importlib.util.spec_from_file_location('lib.pvt', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['lib.pvt'] = module
    exec(compile(patched, str(path) + ':missing_comma_fix', 'exec'), module.__dict__)
    return module


def seed_all(torch, np):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(pretrained, device):
    import torch
    load_model_module(Path.cwd())
    from lib.pvt import OURSNet
    model = OURSNet()
    weights = torch.load(pretrained, map_location='cpu', weights_only=True)
    reference = model.backbone.state_dict()
    if (not reference.keys() <= weights.keys()
        or weights.keys() - reference.keys() != {'head.bias', 'head.weight'}
        or any(reference[k].shape != weights[k].shape for k in reference)):
        raise RuntimeError('PVT pretrained keys/shapes do not cover the whole backbone')
    # The official constructor loaded these same weights. Check that it loaded them exactly.
    if any(not torch.equal(reference[k], weights[k]) for k in reference):
        raise RuntimeError('Official backbone did not load the expected pretrained weights')
    return model.to(device)


def atomic_save(torch, payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'train', 'resume', 'export'])
    for name in ['repo', 'train-root', 'test-root', 'pretrained', 'work']:
        parser.add_argument('--' + name, required=True)
    a = parser.parse_args()
    repo, train, test, pretrained, work = [Path(x).resolve() for x in
        [a.repo, a.train_root, a.test_root, a.pretrained, a.work]]
    current = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if current != COMMIT:
        raise RuntimeError('Unexpected official source revision: ' + current)
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Tracked official source was modified')
    if sha(pretrained) != PVT_SHA:
        raise RuntimeError('Wrong PVT ImageNet pretrained file')
    # Satisfy the official model's hard-coded relative path through one explicit symlink.
    link = repo / 'pretrained_pth' / 'pvt_v2_b2.pth'
    link.parent.mkdir(exist_ok=True)
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != pretrained:
            raise RuntimeError(f'Existing PVT path differs from expected symlink: {link}')
    else:
        link.symlink_to(pretrained)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from PIL import Image
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in baseline environment')
    if torch.__version__.split('+')[0] != '2.4.0':
        raise RuntimeError('This protocol pins torch 2.4.0')
    seed_all(torch, np)
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work / 'checkpoints' / 'FDSS-Net-100.pth'
    manifest = work / 'training_completed.json'
    if a.stage == 'export':
        if not manifest.exists():
            raise RuntimeError('Complete local 100-epoch training first')
        provenance = json.loads(manifest.read_text())
        if (provenance.get('status') != 'completed' or provenance.get('source_commit') != COMMIT
            or provenance.get('epochs') != 100 or provenance.get('checkpoint_sha256') != sha(checkpoint)):
            raise RuntimeError('Training provenance/checkpoint mismatch')
        output = work / 'predictions'
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f'Predictions already exist; refusing overwrite: {output}')
        pairs = {}
        for name, count in COUNTS.items():
            base = test / name
            candidates = [base / x for x in ['images', 'image'] if (base / x).is_dir()]
            if len(candidates) != 1:
                raise RuntimeError(f'Missing/ambiguous test image directory: {base}')
            pairs[name] = pair(candidates[0], base / 'masks', count)
        model = build_model(pretrained, 'cuda')
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        model.eval()
        from torchvision import transforms
        import torch.nn.functional as F
        from tqdm import tqdm
        transform = transforms.Compose([transforms.Resize((352, 352)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        report = {'status': 'incomplete', 'method': 'FDSS-Net', 'provenance': 'local_inference',
            'checkpoint_sha256': provenance['checkpoint_sha256'], 'source_commit': COMMIT,
            'epoch': 100, 'counts': {}, 'raw_key': 'semantic',
            'raw_key_note': 'Single FDSS final output uses the existing evaluator semantic array key',
            'restore': 'P1 logits -> bilinear original size -> sigmoid; minmax uint8 PNG'}
        write(output / 'export_manifest.json', report)
        with torch.no_grad():
            for name, count in COUNTS.items():
                im, gt = pairs[name]
                pngdir, rawdir = output / name, output / 'raw_probabilities' / name
                pngdir.mkdir(parents=True, exist_ok=True)
                rawdir.mkdir(parents=True, exist_ok=True)
                for stem in tqdm(sorted(im), desc=name):
                    with Image.open(im[stem]) as image, Image.open(gt[stem]) as mask:
                        if image.size != mask.size:
                            raise RuntimeError(f'Image/mask dimension mismatch: {name}/{stem}')
                        x = transform(image.convert('RGB')).unsqueeze(0).cuda()
                        hw = (mask.height, mask.width)
                    logits = model(x)[0]
                    raw = F.interpolate(logits, size=hw, mode='bilinear', align_corners=False)
                    raw = raw.sigmoid()[0, 0].cpu().numpy().astype(np.float32)
                    if not np.isfinite(raw).all():
                        raise RuntimeError(f'Non-finite output: {name}/{stem}')
                    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                    Image.fromarray((normalized * 255).astype(np.uint8)).save(pngdir / f'{stem}.png')
                    np.savez_compressed(rawdir / f'{stem}.npz', semantic=raw)
                report['counts'][name] = count
                write(output / 'export_manifest.json', report)
        report['status'] = 'complete'
        write(output / 'export_manifest.json', report)
        print('EXPORT COMPLETE: 798 local test predictions', flush=True)
        return
    im, gt = pair(train / 'images', train / 'labels', 1450, official_loader=True)
    for stem in im:
        with Image.open(im[stem]) as image, Image.open(gt[stem]) as mask:
            if image.size != mask.size:
                raise RuntimeError(f'Image/mask dimension mismatch: {stem}')
    module, adapted_source = load_training_module(repo)
    module.opt = SimpleNamespace(epoch=100, lr=1e-4, optimizer='AdamW', augmentation=True,
        batchsize=16, trainsize=352, clip=.5, decay_rate=.5,
        train_save=str(work / 'checkpoints') + '/')
    module.best = 0
    from utils.dataloader import get_loader
    loader = get_loader(str(train / 'images') + '/', str(train / 'labels') + '/',
        batchsize=16, trainsize=352, augmentation=True)
    if len(loader.dataset) != 1450 or len(loader.dataset.gts) != 1450:
        raise RuntimeError('Official loader dropped training samples')
    module.total_step = len(loader)
    model = build_model(pretrained, 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if a.stage == 'smoke':
        # Full intended batch at all three scales; no checkpoints or trained weights retained.
        batch = next(iter(loader))
        if batch[0].shape[0] != 16:
            raise RuntimeError('Smoke needs the full planned batch of 16')
        module.total_step = 1
        torch.cuda.reset_peak_memory_stats()
        module.train([batch], model, optimizer, 1, '__TEST_ACCESS_FORBIDDEN__')
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Non-finite parameters after smoke')
        print('SMOKE PASS: batch=16, scales=256/352/448, three optimizer steps; peak_allocated_GiB=',
              round(torch.cuda.max_memory_allocated() / 2**30, 3), flush=True)
        return
    audit_path = work / 'train_test_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'PASS' or audit.get('train_count') != 1450 or audit.get('exact_train_test_overlap') != 0:
        raise RuntimeError('Complete train/test audit required')
    recipe = {'method': 'FDSS-Net', 'source_commit': COMMIT, 'seed': 123, 'epochs': 100,
        'batch_size': 16, 'size': 352, 'lr': 1e-4, 'weight_decay': 1e-4,
        'optimizer': 'AdamW', 'scale_rates': [.75, 1, 1.25], 'scale_sizes': [256, 352, 448],
        'loss_weights': [1.2, .6, .3, .15, .1], 'gradient_element_clip': .5,
        'lr_halved_before_epochs': [50, 90], 'augmentation': True,
        'selection': 'fixed_final_epoch', 'train_images': 1450,
        'pretrained_sha256': PVT_SHA, 'data_audit_sha256': sha(audit_path),
        'training_function_sha256': hashlib.sha256(adapted_source.encode()).hexdigest(),
        'official_deviations': ['No test access or test-score checkpoint selection during training',
            'In-memory syntax repair: one missing comma after BatchNorm in lib/pvt.py line 85',
            'Exactly 100 epochs; official range(1,100) performs 99',
            'Python 3.10 / Torch 2.4 cu121 rather than README Python 3.12 / Torch 2.8',
            'Seed 123 explicitly fixed; provenance/resume/native-size export added'],
        'loss_note': 'Official scalar reduction=mean BCE preserved; not a corrected per-pixel weighted BCE',
        'train_image_paths': [str(im[k].resolve()) for k in sorted(im)],
        'train_mask_paths': [str(gt[k].resolve()) for k in sorted(gt)]}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    settings = dict(recipe, protocol_sha256=fingerprint, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    resume_file = work / 'checkpoints' / 'resume.pth'
    start = 1
    if a.stage == 'resume':
        payload = torch.load(resume_file, map_location='cpu', weights_only=False)
        if payload['protocol_sha256'] != fingerprint:
            raise RuntimeError('Resume protocol differs from original run')
        if payload['epoch'] >= 100:
            raise RuntimeError('Already reached epoch 100; use export instead of resume')
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        random.setstate(payload['python_rng'])
        np.random.set_state(payload['numpy_rng'])
        torch.set_rng_state(payload['torch_rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
        start = payload['epoch'] + 1
        del payload
        print('RESUME FROM EPOCH:', start, flush=True)
    elif (work / 'checkpoints').exists() and any((work / 'checkpoints').iterdir()):
        raise RuntimeError('Training output exists; use resume or a fresh WORK_ROOT')
    write(work / 'training_started.json', settings)
    (work / 'adapted_training_function.py').write_text(adapted_source)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (work / 'pip_freeze.txt').write_text(freeze)
    for epoch in range(start, 101):
        if epoch in [50, 90]:
            module.adjust_lr_d(optimizer, 1e-4, epoch, .5)
        module.train(loader, model, optimizer, epoch, '__TEST_ACCESS_FORBIDDEN__')
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload = {'epoch': epoch, 'protocol_sha256': fingerprint, 'model': state,
            'optimizer': optimizer.state_dict(), 'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all()}
        atomic_save(torch, payload, resume_file)
        atomic_save(torch, state, work / 'checkpoints' / 'latest.pth')
        write(work / 'progress.json', {'epoch': epoch, 'epochs': 100, 'train_batches': len(loader),
            'optimizer_steps_per_epoch': 3 * len(loader), 'selection': 'fixed_final_epoch'})
        if epoch == 100:
            atomic_save(torch, state, checkpoint)
        del state, payload
        gc.collect()
    settings['checkpoint_sha256'] = sha(checkpoint)
    settings['status'] = 'completed'
    write(manifest, settings)
    print('TRAIN COMPLETE:', checkpoint, flush=True)


if __name__ == '__main__':
    main()
