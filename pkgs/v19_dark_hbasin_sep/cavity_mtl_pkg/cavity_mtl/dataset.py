from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# =========================
# v19 dark+hbasin 3ch patch
# =========================
import os as _v19_os
import numpy as _v19_np

try:
    from scipy.ndimage import gaussian_filter as _v19_gaussian_filter
    from scipy.ndimage import grey_closing as _v19_grey_closing
except Exception:
    _v19_gaussian_filter = None
    _v19_grey_closing = None

try:
    from skimage.morphology import reconstruction as _v19_reconstruction
except Exception:
    _v19_reconstruction = None


def _v19_norm01(a):
    a = a.astype(_v19_np.float32)
    lo = float(_v19_np.percentile(a, 1.0))
    hi = float(_v19_np.percentile(a, 99.0))
    if hi <= lo + 1e-6:
        lo = float(a.min())
        hi = float(a.max())
    if hi <= lo + 1e-6:
        return _v19_np.zeros_like(a, dtype=_v19_np.float32)
    a = (a - lo) / (hi - lo)
    return _v19_np.clip(a, 0.0, 1.0).astype(_v19_np.float32)


def _v19_dark_residual(x, sigma=5.0):
    x = x.astype(_v19_np.float32)
    if _v19_gaussian_filter is None:
        return _v19_np.zeros_like(x, dtype=_v19_np.float32)
    blur = _v19_gaussian_filter(x, sigma=float(sigma))
    dark = blur - x
    dark = _v19_np.clip(dark, 0.0, None)
    return _v19_norm01(dark)


def _v19_hbasin_black_hdome(x, h=0.08):
    x = x.astype(_v19_np.float32)
    x = _v19_np.clip(x, 0.0, 1.0)
    h = float(h)

    if _v19_reconstruction is not None:
        try:
            seed = _v19_np.clip(x + h, 0.0, 1.0).astype(_v19_np.float32)
            rec = _v19_reconstruction(seed, x, method="erosion").astype(_v19_np.float32)
            basin = rec - x
            basin = _v19_np.clip(basin, 0.0, None)
            basin = basin / max(h, 1e-6)
            return _v19_np.clip(basin, 0.0, 1.0).astype(_v19_np.float32)
        except Exception:
            pass

    if _v19_grey_closing is not None:
        size = int(_v19_os.environ.get("CAVITY_V19_HBASIN_CLOSE_SIZE", "17"))
        closed = _v19_grey_closing(x, size=(size, size))
        basin = closed - x
        return _v19_norm01(basin)

    return _v19_np.zeros_like(x, dtype=_v19_np.float32)


def _v19_make_3ch_from_first_channel(arr):
    arr = _v19_np.asarray(arr)

    if arr.ndim == 2:
        x = arr.astype(_v19_np.float32)
    elif arr.ndim == 3:
        x = arr[0].astype(_v19_np.float32)
    else:
        raise ValueError(f"Unexpected v19 image shape: {arr.shape}")

    x = _v19_norm01(x)
    sigma = float(_v19_os.environ.get("CAVITY_RESIDUAL_SIGMA", "5.0"))
    h = float(_v19_os.environ.get("CAVITY_V19_HBASIN_H", "0.08"))

    dark = _v19_dark_residual(x, sigma=sigma)
    hbasin = _v19_hbasin_black_hdome(x, h=h)

    return _v19_np.stack([x, dark, hbasin], axis=0).astype(_v19_np.float32)
# =========================
# end v19 patch
# =========================


# Supports 2D image files, DICOM, numpy masks, and NIfTI masks.
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.dcm')
MASK_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy', '.nii', '.nii.gz')


def _has_ext(path: Path, exts: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(name.endswith(e) for e in exts)


def _clean_stem(path: str | Path) -> str:
    """Return stem while treating .nii.gz as one extension."""
    p = Path(path)
    name = p.name
    low = name.lower()
    for ext in ('.nii.gz', '.nii', '.dcm', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy'):
        if low.endswith(ext):
            return name[: -len(ext)]
    return p.stem


def _list_files(root: str | Path, exts: tuple[str, ...]) -> List[Path]:
    root = Path(root)
    files: List[Path] = []
    if not root.exists():
        raise FileNotFoundError(f'Directory does not exist: {root}')
    for p in root.rglob('*'):
        if p.is_file() and _has_ext(p, exts):
            files.append(p)
    return sorted(files)


def minmax01(img: np.ndarray) -> np.ndarray:
    """Baseline-compatible min-max normalization."""
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(img)) if img.size else 0.0
    mx = float(np.max(img)) if img.size else 0.0
    if mx > mn:
        img = (img - mn) / (mx - mn + 1e-8)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)




def _robust_rescale01(img: np.ndarray, lo_q: float = 1.0, hi_q: float = 99.0) -> np.ndarray:
    """Robustly rescale a non-negative derived channel to [0, 1]."""
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.size == 0:
        return img.astype(np.float32)
    lo, hi = np.percentile(img, [lo_q, hi_q])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
    if hi > lo:
        img = (img - lo) / (hi - lo + 1e-8)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_residual_channel(base_norm: np.ndarray, mode: str = 'abs', sigma: float = 5.0) -> np.ndarray:
    """Create a second channel emphasizing local CXR structure.

    Args:
        base_norm: min-max normalized CXR in [0, 1].
        mode:
            abs  = |X - GaussianBlur(X)|. Safest default; emphasizes local edges/textures.
            dark = ReLU(GaussianBlur(X) - X). Emphasizes local dark/lucent pockets.
        sigma: Gaussian sigma in pixels before resize/pad. 4~8 is usually reasonable for 512 CXR.
    """
    x = np.clip(base_norm.astype(np.float32), 0.0, 1.0)
    sigma = float(sigma)
    if sigma <= 0:
        blur = x
    else:
        blur = cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    mode = str(mode).lower()
    if mode in ('abs', 'absolute', 'abs_residual'):
        r = np.abs(x - blur)
    elif mode in ('dark', 'dark_residual', 'local_dark'):
        r = np.maximum(blur - x, 0.0)
    elif mode in ('bright', 'bright_residual', 'local_bright'):
        r = np.maximum(x - blur, 0.0)
    else:
        raise ValueError(f'Unknown residual_mode={mode}. Use abs, dark, or bright.')
    return _robust_rescale01(r, 1.0, 99.0)


def make_residual_channels(base_norm: np.ndarray, mode: str = 'abs', sigma: float = 5.0) -> np.ndarray:
    """Return HxWx2 tensor [original minmax, residual]."""
    base_norm = np.clip(base_norm.astype(np.float32), 0.0, 1.0)
    residual = make_residual_channel(base_norm, mode=mode, sigma=sigma)
    return np.stack([base_norm, residual], axis=-1).astype(np.float32)

def _normalize01(img: np.ndarray) -> np.ndarray:
    """Robust percentile scaling; kept for optional experiments, not baseline default."""
    img = img.astype(np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.size == 0:
        raise ValueError('Empty image array')
    lo, hi = np.percentile(img, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def resize_and_pad(img: np.ndarray, target_size: int | tuple[int, int] = 512, is_mask: bool = False) -> np.ndarray:
    """Resize with aspect ratio preserved, then zero-pad to target size.

    This mirrors the official baseline notebook. Direct square resizing stretches
    CXR anatomy and masks, which is harmful for small cavity segmentation.
    """
    if isinstance(target_size, int):
        th = tw = target_size
    else:
        th, tw = target_size
    h, w = img.shape[:2]
    scale = min(th / h, tw / w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if img.ndim == 3:
        padded = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
    else:
        padded = np.zeros((th, tw), dtype=img.dtype)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    padded[top:top + nh, left:left + nw] = resized
    return padded


def _to_2d(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if is_mask:
            smallest_axis = int(np.argmin(arr.shape))
            return np.max(arr, axis=smallest_axis)
        else:
            smallest_axis = int(np.argmin(arr.shape))
            mid = arr.shape[smallest_axis] // 2
            return np.take(arr, indices=mid, axis=smallest_axis)
    raise ValueError(f'Unsupported array shape after squeeze: {arr.shape}')


def read_dicom_raw(path: str | Path) -> np.ndarray:
    """Read DICOM raw 2D pixels using the official baseline convention.

    The baseline applies MONOCHROME1 inversion and plain min-max scaling. It does
    not apply VOI LUT/windowing or percentile clipping before generating channel 0/1.
    """
    try:
        import pydicom
    except Exception as e:
        raise ImportError('DICOM reading requires pydicom. Run: pip install pydicom') from e

    ds = pydicom.dcmread(str(path), force=True)
    img = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        img = np.max(img) - img
    img = _to_2d(img, is_mask=False)
    return img.astype(np.float32)


def make_pseudo_bone_suppression_channel(base_norm: np.ndarray) -> np.ndarray:
    """Deterministic pseudo bone-suppression / soft-tissue channel.

    This is not a learned bone-suppression model.
    It suppresses local bright ridge-like components and keeps the soft-tissue contrast.
    """
    x = np.clip(base_norm.astype(np.float32), 0.0, 1.0)

    # Bright local structures are a crude proxy for ribs / clavicles.
    local = cv2.GaussianBlur(
        x, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0,
        borderType=cv2.BORDER_REFLECT_101
    )
    bright_ridge = np.maximum(x - local, 0.0)
    bright_ridge = _robust_rescale01(bright_ridge, 1.0, 99.0)

    # Suppress bright ridges, then lightly smooth and robustly rescale.
    soft = x - 0.55 * bright_ridge
    soft = np.clip(soft, 0.0, 1.0)
    soft = cv2.GaussianBlur(
        soft, ksize=(0, 0), sigmaX=0.6, sigmaY=0.6,
        borderType=cv2.BORDER_REFLECT_101
    )
    return _robust_rescale01(soft, 1.0, 99.0).astype(np.float32)


def make_baseline_channels(base_norm: np.ndarray) -> np.ndarray:
    """Create 3-channel CXR tensor: [minmax CXR, pseudo bone suppression, dark residual]."""
    base_norm = np.clip(base_norm.astype(np.float32), 0.0, 1.0)
    ch0 = base_norm
    ch1 = make_pseudo_bone_suppression_channel(base_norm)
    ch2 = make_residual_channel(base_norm, mode='dark', sigma=5.0)
    return np.stack([ch0, ch1, ch2], axis=-1).astype(np.float32)


def read_image(path: str | Path, baseline_channels: bool = False) -> np.ndarray:
    """Read CXR.

    Returns HxW float if baseline_channels=False, HxWx3 if True.
    """
    path = Path(path)
    if path.name.lower().endswith('.dcm'):
        raw = read_dicom_raw(path)
        base_norm = minmax01(raw)
        if baseline_channels:
            return make_baseline_channels(base_norm)
        return base_norm

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(str(path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = _to_2d(img, is_mask=False)
    base_norm = minmax01(img)
    if baseline_channels:
        return make_baseline_channels(base_norm)
    return base_norm


def read_nifti_mask(path: str | Path) -> np.ndarray:
    """Read NIfTI mask.

    Prefer SimpleITK because the official baseline uses sitk.GetArrayFromImage.
    nibabel often exposes axes in a different order, which can look like an H/W
    transpose relative to pydicom.pixel_array.
    """
    try:
        import SimpleITK as sitk
        obj = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(obj)
    except Exception:
        try:
            import nibabel as nib
        except Exception as e:
            raise ImportError('NIfTI reading requires SimpleITK or nibabel. Run: pip install SimpleITK nibabel') from e
        arr = nib.load(str(path)).get_fdata(dtype=np.float32)
    arr = _to_2d(arr, is_mask=True)
    return (arr > 0).astype(np.float32)


def read_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    low = path.name.lower()
    if low.endswith('.nii') or low.endswith('.nii.gz'):
        mask = read_nifti_mask(path)
    elif low.endswith('.npy'):
        mask = np.load(path)
        mask = _to_2d(mask, is_mask=True)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(str(path))
        mask = _to_2d(mask, is_mask=True)
    return (mask > 0).astype(np.float32)


def align_mask_to_image(mask: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, str]:
    """Align a 2D mask to a 2D CXR image array.

    With SimpleITK this should usually be 'same'. If a remaining exact H/W
    reversal is detected, transpose the mask. If the mask is already 512x512 or
    another resolution, keep its display-space contents and later resize/pad.
    """
    mask = np.asarray(mask)
    image = np.asarray(image)
    image2d = image[..., 0] if image.ndim == 3 else image
    if mask.ndim != 2:
        mask = _to_2d(mask, is_mask=True)
    if image2d.ndim != 2:
        image2d = _to_2d(image2d, is_mask=False)
    if mask.shape == image2d.shape:
        return mask.astype(np.float32), 'same'
    if mask.shape == image2d.shape[::-1]:
        return mask.T.astype(np.float32), 'transpose'
    return mask.astype(np.float32), 'keep_size_resize_pad_later'


def build_image_mask_table(images_dir: str | Path, masks_dir: Optional[str | Path] = None) -> List[Dict]:
    """Match images and masks by cleaned file stem. If masks_dir is None, creates test rows."""
    image_paths = _list_files(images_dir, IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f'No images found in {images_dir}. Supported: {IMAGE_EXTS}')

    mask_by_stem: Dict[str, Path] = {}
    if masks_dir is not None:
        mask_paths = _list_files(masks_dir, MASK_EXTS)
        if not mask_paths:
            raise FileNotFoundError(f'No masks found in {masks_dir}. Supported: {MASK_EXTS}')
        for m in mask_paths:
            mask_by_stem.setdefault(_clean_stem(m), m)

    rows = []
    missing = []
    for img in image_paths:
        image_id = _clean_stem(img)
        row = {'image_id': image_id, 'image_path': str(img), 'mask_path': None, 'label': None}
        if masks_dir is not None:
            mp = mask_by_stem.get(image_id)
            if mp is None:
                missing.append(image_id)
                continue
            mask = read_mask(mp)
            label = int(mask.sum() > 0)
            row['mask_path'] = str(mp)
            row['label'] = label
        rows.append(row)

    if masks_dir is not None and missing:
        preview = ', '.join(missing[:10])
        raise FileNotFoundError(
            f'{len(missing)} images had no matching mask by stem under {masks_dir}. '
            f'Examples: {preview}. If masks are named like case001_mask.nii.gz, rename them or tell me the pattern.'
        )
    return rows


class CavityDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict],
        img_size: int = 512,
        augment: Optional[Callable] = None,
        in_chans: int = 3,
        has_masks: bool = True,
        baseline_preprocess: bool = True,
        residual_mode: str = 'abs',
        residual_sigma: float = 5.0,
    ):
        self.rows = list(rows)
        self.img_size = int(img_size)
        self.augment = augment
        self.in_chans = int(in_chans)
        self.has_masks = has_masks
        self.baseline_preprocess = bool(baseline_preprocess)
        self.residual_mode = str(residual_mode)
        self.residual_sigma = float(residual_sigma)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def labels(self) -> np.ndarray:
        vals = [0 if r.get('label') is None else int(r['label']) for r in self.rows]
        return np.asarray(vals, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | int]:
        r = self.rows[idx]
        # Channel policy:
        #   1 channel: original min-max CXR
        #   2 channels: [original min-max CXR, local residual map]
        #   3 channels: [minmax, pseudo bone suppression, dark residual]
        image = read_image(r['image_path'], baseline_channels=(self.in_chans == 3 and self.baseline_preprocess))
        if self.in_chans == 2:
            base2d = image[..., 0] if image.ndim == 3 else image
            image = make_residual_channels(base2d, mode=self.residual_mode, sigma=self.residual_sigma)
        image2d = image[..., 0] if image.ndim == 3 else image

        label = 0
        if self.has_masks and r.get('mask_path') is not None:
            mask = read_mask(r['mask_path'])
            label = int(mask.sum() > 0)
        else:
            mask = np.zeros_like(image2d, dtype=np.float32)

        mask, align_action = align_mask_to_image(mask, image2d)

        if self.baseline_preprocess:
            # Official baseline: resize each image channel and mask with aspect-ratio preservation + zero padding.
            if image.ndim == 2:
                image = resize_and_pad(image, self.img_size, is_mask=False)
            else:
                chans = [resize_and_pad(image[..., c], self.img_size, is_mask=False) for c in range(image.shape[-1])]
                image = np.stack(chans, axis=-1)
            mask = resize_and_pad(mask, self.img_size, is_mask=True)
        else:
            # Experimental mode: direct square resizing.
            if image.shape[:2] != (self.img_size, self.img_size):
                image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            if mask.shape[:2] != (self.img_size, self.img_size):
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if image.ndim == 2:
            image_hwc = image[..., None]
        else:
            image_hwc = image
        mask_hw = mask.astype(np.float32)

        if self.augment is not None:
            out = self.augment(image=image_hwc, mask=mask_hw)
            image_hwc = out['image']
            mask_hw = out['mask']

        if image_hwc.ndim == 2:
            image_hwc = image_hwc[..., None]
        image_hwc = image_hwc.astype(np.float32)
        mask_hw = (mask_hw > 0.5).astype(np.float32)

        if self.in_chans == 3 and image_hwc.shape[-1] == 1:
            image_hwc = np.repeat(image_hwc, 3, axis=-1)
        elif self.in_chans == 2:
            if image_hwc.shape[-1] == 1:
                # Fallback: construct residual after augmentation if a single channel slipped through.
                image_hwc = make_residual_channels(image_hwc[..., 0], mode=self.residual_mode, sigma=self.residual_sigma)
            elif image_hwc.shape[-1] != 2:
                image_hwc = image_hwc[..., :2]
        elif self.in_chans == 1 and image_hwc.shape[-1] != 1:
            image_hwc = image_hwc[..., :1]

        image_t = torch.from_numpy(image_hwc.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask_hw[None]).float()
        label_t = torch.tensor(float(label), dtype=torch.float32)

        return {
            'idx': idx,
            'image_id': r['image_id'],
            'image_path': r['image_path'],
            'image': image_t,
            'mask': mask_t,
            'label': label_t,
        }


# ============================================================
# V19_RUNTIME_DATASET_WRAPPER
# Force dataset output to:
#   ch0 = normalized CXR
#   ch1 = dark residual
#   ch2 = black h-dome / h-basin
# This wrapper is activated only when:
#   CAVITY_FORCE_V19_DARK_HBASIN_3CH=1
# ============================================================

def _v19_is_image_like_obj(obj):
    try:
        import torch as _v19_torch
        if isinstance(obj, _v19_torch.Tensor):
            return obj.ndim in (2, 3)
    except Exception:
        pass

    try:
        import numpy as _v19_np2
        if isinstance(obj, _v19_np2.ndarray):
            return obj.ndim in (2, 3)
    except Exception:
        pass

    return False


def _v19_convert_image_obj(obj):
    if _v19_os.environ.get("CAVITY_FORCE_V19_DARK_HBASIN_3CH", "0") != "1":
        return obj

    try:
        import torch as _v19_torch
        if isinstance(obj, _v19_torch.Tensor):
            arr = obj.detach().cpu().numpy()
            arr3 = _v19_make_3ch_from_first_channel(arr)
            out = _v19_torch.from_numpy(arr3)
            if obj.dtype.is_floating_point:
                out = out.to(dtype=obj.dtype)
            return out
    except Exception:
        pass

    try:
        import numpy as _v19_np2
        if isinstance(obj, _v19_np2.ndarray):
            return _v19_make_3ch_from_first_channel(obj)
    except Exception:
        pass

    return obj


def _v19_convert_sample_output(sample):
    if _v19_os.environ.get("CAVITY_FORCE_V19_DARK_HBASIN_3CH", "0") != "1":
        return sample

    # dict output
    if isinstance(sample, dict):
        sample = dict(sample)
        preferred_keys = [
            "image", "img", "x", "cxr", "input",
            "image_tensor", "img_tensor", "pixel_values"
        ]

        for k in preferred_keys:
            if k in sample and _v19_is_image_like_obj(sample[k]):
                sample[k] = _v19_convert_image_obj(sample[k])
                return sample

        # fallback: first image-like value
        for k, v in sample.items():
            if _v19_is_image_like_obj(v):
                sample[k] = _v19_convert_image_obj(v)
                return sample

        return sample

    # tuple/list output: usually first item is image
    if isinstance(sample, tuple):
        items = list(sample)
        if len(items) > 0 and _v19_is_image_like_obj(items[0]):
            items[0] = _v19_convert_image_obj(items[0])
        return tuple(items)

    if isinstance(sample, list):
        items = list(sample)
        if len(items) > 0 and _v19_is_image_like_obj(items[0]):
            items[0] = _v19_convert_image_obj(items[0])
        return items

    # direct image output
    if _v19_is_image_like_obj(sample):
        return _v19_convert_image_obj(sample)

    return sample


def _v19_wrap_dataset_classes():
    import inspect as _v19_inspect

    wrapped = []
    for _name, _cls in list(globals().items()):
        if not _v19_inspect.isclass(_cls):
            continue
        if getattr(_cls, "__module__", None) != __name__:
            continue
        if not hasattr(_cls, "__getitem__"):
            continue
        if getattr(_cls, "_v19_runtime_wrapped", False):
            continue

        _orig_getitem = _cls.__getitem__

        def _make_wrapped_getitem(orig_func):
            def _wrapped_getitem(self, idx):
                sample = orig_func(self, idx)
                return _v19_convert_sample_output(sample)
            return _wrapped_getitem

        _cls.__getitem__ = _make_wrapped_getitem(_orig_getitem)
        _cls._v19_runtime_wrapped = True
        wrapped.append(_name)

    if _v19_os.environ.get("CAVITY_FORCE_V19_DARK_HBASIN_3CH", "0") == "1":
        print("[V19 DATASET WRAPPER] wrapped classes:", wrapped)


_v19_wrap_dataset_classes()
# ============================================================
# END V19_RUNTIME_DATASET_WRAPPER
# ============================================================
