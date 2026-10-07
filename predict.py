import os
import sys
import csv
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch
from scipy import ndimage as ndi


IMG_SIZE = 512
PRUNE_TOP_K = 3
PRUNE_MAX_DROP = 0.25
PRUNE_P95_DROP = 0.20
PRUNE_DROP_AREA = 200
V18_HDOME_H = 0.08
V19_HBASIN_H = 0.08
W_V18 = 0.38
IN_CHANS = 2
RESIDUAL_MODE = "dark"
RESIDUAL_SIGMA = 5.0

W_V12 = 0.00
W_V12_2 = 0.49
W_V19 = 0.13

T_CLS = 0.30
T_MASK = 0.12
MIN_AREA = 0


def clear_pkg_modules():
    for k in list(sys.modules.keys()):
        if k == "cavity_mtl" or k.startswith("cavity_mtl."):
            del sys.modules[k]


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "ema", "ema_state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    out = {}
    for k, v in sd.items():
        kk = str(k)
        for prefix in ["module.", "model.", "net."]:
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        out[kk] = v
    return out


def import_preprocess_functions(pkg_dir):
    clear_pkg_modules()
    pkg_dir = str(Path(pkg_dir).resolve())
    sys.path.insert(0, pkg_dir)
    try:
        from cavity_mtl.dataset import read_image, make_residual_channels, resize_and_pad
    finally:
        try:
            sys.path.remove(pkg_dir)
        except ValueError:
            pass
    return read_image, make_residual_channels, resize_and_pad


def load_model_group(pkg_dir, weight_dir, device):
    clear_pkg_modules()
    pkg_dir = str(Path(pkg_dir).resolve())
    sys.path.insert(0, pkg_dir)

    try:
        from cavity_mtl.model import CavityMTLNet

        models = []
        for fold in range(5):
            ckpt_path = Path(weight_dir) / f"fold{fold}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(str(ckpt_path))

            model = CavityMTLNet(in_chans=IN_CHANS, pretrained=False)
            sd = get_state_dict(ckpt_path)
            missing, unexpected = model.load_state_dict(sd, strict=False)

            print(f"[load] {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

            model.to(device)
            model.eval()
            models.append(model)

    finally:
        try:
            sys.path.remove(pkg_dir)
        except ValueError:
            pass

    return models


def find_dicom_files(patient_dir):
    files = sorted(glob.glob(os.path.join(patient_dir, "*.dcm")))
    if not files:
        files = sorted(glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True))
    if not files:
        raise RuntimeError(f"No DICOM files found in: {patient_dir}")
    return files


def load_reference_image(patient_dir):
    files = find_dicom_files(patient_dir)

    # Prefer series reader. For CXR this is usually a one-slice series.
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(patient_dir)
        if series_ids:
            names = reader.GetGDCMSeriesFileNames(patient_dir, series_ids[0])
            if names:
                reader.SetFileNames(names)
                img = reader.Execute()
                return img, list(names)
    except Exception as e:
        print(f"[warn] ImageSeriesReader failed for {patient_dir}: {repr(e)}", flush=True)

    # Fallback: read first DICOM as 2D image.
    img = sitk.ReadImage(files[0])
    return img, files


def make_input_tensor(dcm_path, read_image, make_residual_channels, resize_and_pad):
    # read_image applies the same normalization / DICOM handling as training dataset.
    base = read_image(dcm_path, baseline_channels=False)

    if base.ndim == 3:
        if base.shape[-1] == 1:
            base = base[..., 0]
        else:
            base = base[..., 0]

    base = np.asarray(base, dtype=np.float32)
    orig_h, orig_w = base.shape[:2]

    ch = make_residual_channels(base, mode=RESIDUAL_MODE, sigma=RESIDUAL_SIGMA)

    chans = []
    for c in range(ch.shape[-1]):
        chans.append(resize_and_pad(ch[..., c], IMG_SIZE, is_mask=False))

    x = np.stack(chans, axis=0).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)

    return x, (orig_h, orig_w)


# ---- FINAL_V12_V124_V18_HELPERS_START ----

def _as_2d_float_image(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    mn = float(arr.min())
    mx = float(arr.max())
    if mn < -1e-4 or mx > 1.0001:
        arr = (arr - mn) / max(mx - mn, 1e-6)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def make_hdome_channel(x, h=0.08):
    x = _as_2d_float_image(x)
    h = float(h)
    try:
        from skimage.morphology import reconstruction
        seed = np.clip(x - h, 0.0, 1.0).astype(np.float32)
        rec = reconstruction(seed, x, method="dilation").astype(np.float32)
        hd = x - rec
    except Exception:
        from scipy.ndimage import grey_opening
        rec = grey_opening(x, size=(15, 15)).astype(np.float32)
        hd = x - rec
    hd = np.clip(hd, 0.0, h) / max(h, 1e-6)
    return hd.astype(np.float32)


def make_v18_hdome_input_tensor(dcm_path, read_image, make_residual_channels, resize_and_pad):
    x, orig_shape = make_input_tensor(
        dcm_path,
        read_image,
        make_residual_channels,
        resize_and_pad,
    )
    base = read_image(dcm_path, baseline_channels=False)
    base = _as_2d_float_image(base)
    hd = make_hdome_channel(base, h=V18_HDOME_H)
    hd512 = resize_and_pad(hd, IMG_SIZE, is_mask=False).astype(np.float32)

    if x.ndim != 4 or x.shape[1] < 2:
        raise RuntimeError(f"Expected Bx2xHxW tensor for v18, got {tuple(x.shape)}")

    x = x.clone()
    x[0, 1] = torch.from_numpy(hd512)
    return x, orig_shape


def unpad_resize_mask(mask512, orig_shape):
    h, w = orig_shape
    scale = min(IMG_SIZE / float(w), IMG_SIZE / float(h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    left = (IMG_SIZE - nw) // 2
    top = (IMG_SIZE - nh) // 2
    crop = mask512[top:top + nh, left:left + nw].astype(np.uint8)
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
    return (out > 0).astype(np.uint8)


def prune_components_drop_small_lowprob(mask, prob, drop_area=200, p95_drop=0.20, max_drop=0.25, top_k=3):
    mask = mask.astype(bool)
    prob = prob.astype(np.float32)

    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    labels, ncomp = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if ncomp <= 0:
        return np.zeros_like(mask, dtype=np.uint8)

    feats = []
    for lab in range(1, int(ncomp) + 1):
        m = labels == lab
        area = int(m.sum())
        if area <= 0:
            continue

        vals = prob[m]
        p95 = float(np.percentile(vals, 95))
        mx = float(vals.max())
        score = float(p95 * np.sqrt(max(area, 1)))

        drop = (area <= int(drop_area)) and (p95 < float(p95_drop)) and (mx < float(max_drop))
        if not drop:
            feats.append({"label": lab, "area": area, "p95": p95, "max": mx, "score": score})

    if not feats:
        return np.zeros_like(mask, dtype=np.uint8)

    if int(top_k) > 0 and len(feats) > int(top_k):
        feats = sorted(feats, key=lambda z: z["score"], reverse=True)[:int(top_k)]

    keep_labels = np.array([int(f["label"]) for f in feats], dtype=np.int32)
    return np.isin(labels, keep_labels).astype(np.uint8)

# ---- FINAL_V12_V124_V18_HELPERS_END ----



def unpad_resize_prob(prob512, orig_shape):
    h, w = orig_shape
    scale = min(IMG_SIZE / float(w), IMG_SIZE / float(h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    left = (IMG_SIZE - nw) // 2
    top = (IMG_SIZE - nh) // 2

    crop = prob512[top:top + nh, left:left + nw]
    prob = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prob.astype(np.float32)


def to_output_mask_shape(mask2d, ref_img):
    arr = sitk.GetArrayFromImage(ref_img)

    if ref_img.GetDimension() == 2:
        return mask2d.astype(np.uint8)

    if arr.ndim == 3:
        out = np.zeros_like(arr, dtype=np.uint8)
        if arr.shape[0] == 1:
            out[0] = mask2d.astype(np.uint8)
        else:
            # Task 1 is X-ray; this branch is only a safety fallback.
            z = arr.shape[0] // 2
            out[z] = mask2d.astype(np.uint8)
        return out

    return mask2d.astype(np.uint8)




# ---- FINAL_V124_V18_V19_HELPERS_START ----

def make_hbasin_channel(base, h=0.08):
    base = np.asarray(base, dtype=np.float32)
    base = np.clip(base, 0.0, 1.0)

    try:
        from skimage.morphology import reconstruction

        seed = np.clip(base + float(h), 0.0, 1.0)
        rec = reconstruction(seed, base, method="erosion")
        basin = rec.astype(np.float32) - base
        basin = np.clip(basin, 0.0, float(h)) / (float(h) + 1e-6)

    except Exception:
        from scipy.ndimage import grey_closing

        rec = grey_closing(base, size=(15, 15))
        basin = rec.astype(np.float32) - base
        basin = np.clip(basin, 0.0, None)
        mx = float(basin.max())
        if mx > 1e-6:
            basin = basin / mx

    return basin.astype(np.float32)


def make_v19_hbasin_input_tensor_from_dark(x_dark):
    if x_dark.ndim != 4 or x_dark.shape[1] != 2:
        raise RuntimeError(f"Expected Bx2xHxW tensor for dark input, got {tuple(x_dark.shape)}")

    base = x_dark[0, 0].detach().cpu().numpy().astype(np.float32)
    hbasin = make_hbasin_channel(base, h=V19_HBASIN_H)

    hbasin_t = torch.from_numpy(hbasin).to(dtype=x_dark.dtype, device=x_dark.device)
    hbasin_t = hbasin_t.unsqueeze(0).unsqueeze(0)

    x_v19 = torch.cat([x_dark, hbasin_t], dim=1)

    if x_v19.shape[1] != 3:
        raise RuntimeError(f"Expected Bx3xHxW tensor for v19, got {tuple(x_v19.shape)}")

    return x_v19.float()

# ---- FINAL_V124_V18_V19_HELPERS_END ----



class V124V18V19PrunedEnsemble:
    def __init__(self, weights_root):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[device] {self.device}", flush=True)

        self.root = Path(__file__).resolve().parent
        self.weights_root = Path(weights_root)

        self.v12_2_pkg = self.root / "pkgs" / "v12_4_ropad_lite_gate_boostonly" / "cavity_mtl_pkg"
        self.v18_pkg = self.root / "pkgs" / "v18_hdome2ch_boostonly" / "cavity_mtl_pkg"
        self.v19_pkg = self.root / "pkgs" / "v19_dark_hbasin_sep" / "cavity_mtl_pkg"

        self.read_image, self.make_residual_channels, self.resize_and_pad = import_preprocess_functions(self.v12_2_pkg)

        print("[final config] A/B/C = v12-4/v18/v19 = 0.49/0.38/0.13", flush=True)
        print("[final config] t_cls=0.30 t_mask=0.12", flush=True)
        print("[final config] pruning: drop_area=200 p95=0.20 max=0.25 top_k=3 on 512 grid", flush=True)
        print("[final config] final mask is restored to original DICOM grid", flush=True)

        self.models_v124 = load_model_group(
            self.v12_2_pkg,
            self.weights_root / "A",
            self.device,
        )

        os.environ["CAVITY_FORCE_HDOME2CH"] = "1"
        os.environ["CAVITY_HDOME_H"] = str(V18_HDOME_H)

        self.models_v18 = load_model_group(
            self.v18_pkg,
            self.weights_root / "B",
            self.device,
        )

        os.environ["CAVITY_FORCE_V19_DARK_HBASIN_3CH"] = "1"
        os.environ["CAVITY_V19_HBASIN_H"] = str(V19_HBASIN_H)
        os.environ["CAVITY_RESIDUAL_SIGMA"] = "5.0"
        os.environ["CAVITY_V19_FULL_DECOUPLE"] = "1"

        old_in_chans = globals().get("IN_CHANS", 2)
        globals()["IN_CHANS"] = 3
        try:
            self.models_v19 = load_model_group(
                self.v19_pkg,
                self.weights_root / "C",
                self.device,
            )
        finally:
            globals()["IN_CHANS"] = old_in_chans

        torch.set_grad_enabled(False)

    def _predict_group(self, models, x):
        cls_probs = []
        mask_probs = []

        x = x.to(self.device, non_blocking=True)

        with torch.no_grad():
            for model in models:
                out = model(x)
                cls = torch.sigmoid(out["cls_logit"]).reshape(-1)[0].detach().float().cpu().item()
                mask = torch.sigmoid(out["mask_logit"])[0, 0].detach().float().cpu().numpy()
                cls_probs.append(float(cls))
                mask_probs.append(mask.astype(np.float32))

        return float(np.mean(cls_probs)), np.mean(mask_probs, axis=0).astype(np.float32)

    def predict_case(self, patient_dir):
        ref_img, dicom_files = load_reference_image(patient_dir)
        dcm_path = dicom_files[len(dicom_files) // 2]

        x_dark, orig_shape = make_input_tensor(
            dcm_path,
            self.read_image,
            self.make_residual_channels,
            self.resize_and_pad,
        )

        x_hdome, orig_shape_v18 = make_v18_hdome_input_tensor(
            dcm_path,
            self.read_image,
            self.make_residual_channels,
            self.resize_and_pad,
        )

        if tuple(orig_shape) != tuple(orig_shape_v18):
            raise RuntimeError(f"orig_shape mismatch: dark={orig_shape} hdome={orig_shape_v18}")

        x_hbasin = make_v19_hbasin_input_tensor_from_dark(x_dark)

        cls_v124, mask_v124_512 = self._predict_group(self.models_v124, x_dark)
        cls_v18, mask_v18_512 = self._predict_group(self.models_v18, x_hdome)
        cls_v19, mask_v19_512 = self._predict_group(self.models_v19, x_hbasin)

        cls_prob = W_V12_2 * cls_v124 + W_V18 * cls_v18 + W_V19 * cls_v19
        mask_prob_512 = W_V12_2 * mask_v124_512 + W_V18 * mask_v18_512 + W_V19 * mask_v19_512

        if cls_prob < T_CLS:
            mask2d = np.zeros(orig_shape, dtype=np.uint8)
            cavity = 0
        else:
            raw512 = (mask_prob_512 >= T_MASK).astype(np.uint8)

            pruned512 = prune_components_drop_small_lowprob(
                raw512,
                mask_prob_512,
                drop_area=PRUNE_DROP_AREA,
                p95_drop=PRUNE_P95_DROP,
                max_drop=PRUNE_MAX_DROP,
                top_k=PRUNE_TOP_K,
            )

            mask2d = unpad_resize_mask(pruned512, orig_shape)

            if tuple(mask2d.shape) != tuple(orig_shape):
                raise RuntimeError(f"mask shape mismatch after inverse resize: {mask2d.shape} vs {orig_shape}")

            cavity = int(int(mask2d.sum()) > 0)

        mask_out = to_output_mask_shape(mask2d, ref_img)
        return mask_out, cavity, cls_prob, ref_img

def save_mask(mask_arr, ref_img, out_path):
    pred_img = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    pred_img.CopyInformation(ref_img)
    sitk.WriteImage(pred_img, out_path)


def run_one_case(model, patient_dir, output_dir):
    our_id = os.path.basename(os.path.normpath(patient_dir))

    mask, cavity, cls_prob, ref_img = model.predict_case(patient_dir)

    out_path = os.path.join(output_dir, f"{our_id}.nii.gz")
    save_mask(mask, ref_img, out_path)

    print(f"[case] {our_id} cavity={cavity} cls={cls_prob:.5f} mask_vox={int(mask.sum())}", flush=True)
    return our_id, cavity


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--weights", default="/workspace/weights")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    model = V124V18V19PrunedEnsemble(args.weights)

    patient_dirs = sorted(
        d for d in glob.glob(os.path.join(args.input, "*"))
        if os.path.isdir(d)
    )
    if not patient_dirs:
        raise RuntimeError(f"No patient folders found under: {args.input}")

    rows = []
    for d in tqdm(patient_dirs, desc="Inference"):
        rows.append(run_one_case(model, d, args.output))

    csv_path = os.path.join(args.output, "prediction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["our_id", "cavity"])
        writer.writerows(rows)

    print(f"[saved] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
