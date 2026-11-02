from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SAM_PIXEL_MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)

def _index_by_stem(folder: Path):
    result = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith('.'):
            if p.stem in result:
                raise RuntimeError(f"Duplicate stem in {folder}: {p.stem}")
            result[p.stem] = p
    return result

def resize_longest_and_pad(arr, image_size=1024, is_mask=False, image_pad_value=None):
    h, w = arr.shape[:2]
    scale = image_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (nw, nh), interpolation=interp)

    if arr.ndim == 3:
        if image_pad_value is None:
            image_pad_value = np.zeros((arr.shape[2],), dtype=np.float32)
        out = np.empty((image_size, image_size, arr.shape[2]), dtype=np.float32)
        out[...] = np.asarray(image_pad_value, dtype=np.float32)
        out[:nh, :nw] = resized.astype(np.float32)
    else:
        out = np.zeros((image_size, image_size), dtype=np.float32)
        out[:nh, :nw] = resized.astype(np.float32)
    return out

class KvasirTeacherDataset(Dataset):
    def __init__(self, split_root, image_size=1024, augment=False, require_teacher=True):
        self.root = Path(split_root)
        self.image_size = image_size
        self.augment = augment
        self.require_teacher = require_teacher

        self.images = _index_by_stem(self.root / "images")
        self.labels = _index_by_stem(self.root / "labels")
        self.maps = _index_by_stem(self.root / "attribution_map") if require_teacher else {}
        expected = set(self.images)
        for name, index in [("labels", self.labels)] + ([("attribution_map", self.maps)] if require_teacher else []):
            if set(index) != expected:
                missing = sorted(expected - set(index))
                extra = sorted(set(index) - expected)
                raise RuntimeError(f"Incomplete {name}: missing={len(missing)} {missing[:5]}, extra={len(extra)} {extra[:5]}")
        self.stems = sorted(expected)
        if not self.stems:
            raise RuntimeError(f"No matched triplets in {self.root}")

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]

        image = cv2.imread(str(self.images[stem]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.labels[stem]), cv2.IMREAD_GRAYSCALE)
        teacher = cv2.imread(str(self.maps[stem]), cv2.IMREAD_GRAYSCALE) if self.require_teacher else None
        if image is None or mask is None or (self.require_teacher and teacher is None):
            raise RuntimeError(f"Read failure: {stem}")
        if image.shape[:2] != mask.shape or (self.require_teacher and teacher.shape != mask.shape):
            raise RuntimeError(f"Image/mask/teacher dimension mismatch: {stem}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_longest_and_pad(
            image, self.image_size, is_mask=False,
            image_pad_value=SAM_PIXEL_MEAN_RGB
        )
        mask = resize_longest_and_pad(mask, self.image_size, is_mask=True)
        if self.require_teacher:
            teacher = resize_longest_and_pad(teacher, self.image_size, is_mask=False)

        mask = (mask > 127).astype(np.float32)
        if self.require_teacher:
            teacher = teacher.astype(np.float32) / 255.0
            tmin, tmax = float(teacher.min()), float(teacher.max())
            if tmax > tmin + 1e-6:
                teacher = (teacher - tmin) / (tmax - tmin)

        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, 1).copy()
                mask = np.flip(mask, 1).copy()
                if self.require_teacher:
                    teacher = np.flip(teacher, 1).copy()
            if random.random() < 0.5:
                image = np.flip(image, 0).copy()
                mask = np.flip(mask, 0).copy()
                if self.require_teacher:
                    teacher = np.flip(teacher, 0).copy()

        sample = {
            "image": torch.from_numpy(image).permute(2,0,1).float(),
            "mask": torch.from_numpy(mask)[None].float(),
            "name": stem,
        }
        if self.require_teacher:
            sample["teacher"] = torch.from_numpy(teacher)[None].float()
        return sample
