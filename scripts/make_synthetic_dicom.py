"""Write a synthetic chest-radiograph-like DICOM for a smoke test of predict.py.

    python scripts/make_synthetic_dicom.py --out input/synthetic_case/image.dcm

The image is a smooth synthetic pattern with a dark round 'cavity-like' region; it is NOT a real
radiograph and the model output on it is meaningless. It only exercises DICOM reading, preprocessing,
model loading, post-processing and output writing end to end.
"""
import argparse
import datetime
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

CR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.1"


def synthetic_image(h=2048, w=2048, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    r = np.sqrt(((yy - cy) / (0.45 * h)) ** 2 + ((xx - cx) / (0.35 * w)) ** 2)
    body = np.clip(1.2 - r, 0, 1)                       # bright thorax on dark background
    lungs = np.exp(-(((xx - 0.33 * w) / (0.14 * w)) ** 2 + ((yy - 0.5 * h) / (0.28 * h)) ** 2))
    lungs += np.exp(-(((xx - 0.67 * w) / (0.14 * w)) ** 2 + ((yy - 0.5 * h) / (0.28 * h)) ** 2))
    img = body - 0.45 * lungs
    ribs = 0.04 * np.sin(yy / h * 40 * np.pi) * lungs
    cavity = 0.30 * np.exp(-(((xx - 0.30 * w) / (0.03 * w)) ** 2 + ((yy - 0.38 * h) / (0.03 * h)) ** 2))
    img = img + ribs - cavity + rng.normal(0, 0.01, (h, w)).astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 4000 + 100).astype(np.uint16)          # MONOCHROME2, 12-bit-like range


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="input/synthetic_case/image.dcm")
    ap.add_argument("--size", type=int, default=2048)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    px = synthetic_image(args.size, args.size)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CR_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(out), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID = CR_IMAGE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality, ds.PatientName, ds.PatientID = "CR", "SYNTHETIC^PHANTOM", "SYNTHETIC-0001"
    ds.StudyDate = ds.SeriesDate = ds.ContentDate = datetime.date.today().strftime("%Y%m%d")
    ds.BodyPartExamined, ds.ViewPosition = "CHEST", "PA"
    ds.Rows, ds.Columns = px.shape
    ds.PixelSpacing = ["0.14", "0.14"]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = px.tobytes()
    ds.save_as(str(out), write_like_original=False)
    print(f"[saved] {out}  {px.shape[1]}x{px.shape[0]} uint16")


if __name__ == "__main__":
    main()
