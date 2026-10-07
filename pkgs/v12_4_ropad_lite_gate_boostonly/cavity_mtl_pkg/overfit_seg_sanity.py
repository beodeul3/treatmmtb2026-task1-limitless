from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cavity_mtl.dataset import build_image_mask_table, CavityDataset
from cavity_mtl.losses import dice_loss, focal_bce_with_logits
from cavity_mtl.model import CavityMTLNet
from cavity_mtl.utils import autocast_context, move_to_device, seed_everything


def main():
    ap = argparse.ArgumentParser(description='Overfit a few positive samples to verify mask alignment/loss/model can learn.')
    ap.add_argument('--images_dir', required=True)
    ap.add_argument('--masks_dir', required=True)
    ap.add_argument('--out_dir', default='runs/overfit_seg_sanity')
    ap.add_argument('--model', default='convnext_tiny.fb_in22k_ft_in1k')
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--in_chans', type=int, default=1, choices=[1, 3])
    ap.add_argument('--n_pos', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    seed_everything(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_image_mask_table(args.images_dir, args.masks_dir)
    pos_rows = [r for r in rows if int(r['label']) == 1]
    if len(pos_rows) == 0:
        raise RuntimeError('No positive rows found.')
    # choose small/medium masks, not huge masks, to test real cavity learning
    areas = []
    tmp_ds = CavityDataset(pos_rows, img_size=args.img_size, augment=None, in_chans=args.in_chans, has_masks=True)
    for i in range(len(tmp_ds)):
        it = tmp_ds[i]
        areas.append(float(it['mask'].mean()))
    order = np.argsort(areas)
    lo = max(0, int(0.20 * len(order)))
    hi = max(lo + 1, int(0.80 * len(order)))
    cand = order[lo:hi]
    chosen = cand[:min(args.n_pos, len(cand))]
    small_rows = [pos_rows[int(i)] for i in chosen]
    print('Selected image_ids:', [r['image_id'] for r in small_rows])
    ds = CavityDataset(small_rows, img_size=args.img_size, augment=None, in_chans=args.in_chans, has_masks=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CavityMTLNet(args.model, pretrained=not args.no_pretrained, in_chans=args.in_chans, mask_to_cls_stopgrad=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    for epoch in range(args.epochs):
        model.train(); losses=[]; dices=[]; maxps=[]
        for batch in loader:
            bd = move_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with autocast_context(args.amp):
                out = model(bd['image'])
                seg = dice_loss(out['mask_logit'], bd['mask']) + 0.5 * focal_bce_with_logits(out['mask_logit'], bd['mask'], gamma=2.0)
                # weak cls auxiliary only; sanity target is segmentation overfit
                cls = focal_bce_with_logits(out['cls_logit'], bd['label'].float(), gamma=2.0)
                loss = 2.0 * seg + 0.1 * cls
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad():
                p = torch.sigmoid(out['mask_logit'])
                pred = (p > 0.3).float()
                inter = (pred * bd['mask']).sum(dim=(1,2,3))
                den = pred.sum(dim=(1,2,3)) + bd['mask'].sum(dim=(1,2,3))
                dice = ((2*inter+1e-6)/(den+1e-6)).mean().item()
                losses.append(float(loss.detach().cpu())); dices.append(dice); maxps.append(float(p.max().detach().cpu()))
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f'epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.4f} dice@0.3={np.mean(dices):.4f} mask_max={np.mean(maxps):.4f}')
    # save final overlays
    model.eval()
    with torch.no_grad():
        for i in range(len(ds)):
            it = ds[i]
            img = it['image'].unsqueeze(0).to(device)
            out = model(img)
            prob = torch.sigmoid(out['mask_logit'])[0,0].cpu().numpy()
            x = it['image'].numpy()[0]
            gt = it['mask'].numpy()[0]
            x_u8 = ((x - x.min())/(x.max()-x.min()+1e-6)*255).astype(np.uint8)
            base = cv2.cvtColor(x_u8, cv2.COLOR_GRAY2BGR)
            gt_color = base.copy(); gt_color[gt>0.5] = (0,255,0)
            pr_color = base.copy(); pr_color[prob>0.3] = (0,0,255)
            heat = cv2.applyColorMap((np.clip(prob,0,1)*255).astype(np.uint8), cv2.COLORMAP_JET)
            canvas = np.concatenate([base, gt_color, pr_color, heat], axis=1)
            cv2.imwrite(str(out_dir / f'{i:02d}_{it["image_id"]}.png'), canvas)
    print('Saved sanity overlays to', out_dir)

if __name__ == '__main__':
    main()
