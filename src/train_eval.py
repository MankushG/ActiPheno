"""
train_eval.py  —  ActiPheno
LOPO cross-validation: one model per fold, immediate CSV dumps so
nothing is lost when Colab inevitably dies.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from data_loader import (
    ActiDataset, DepresjonLoader, PatientRecord,
    RandomAugment, WINDOW_SIZE, compute_fold_stats,
)
from model import ActiPheno, ActiPhenoConfig
from evaluate import compute_metrics, print_global

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# each test fold is a single patient — all windows are the same class,
# so sklearn screams about undefined AUC and single-label CM every fold
warnings.filterwarnings("ignore", message="Only one class is present", category=UserWarning)
warnings.filterwarnings("ignore", message="A single label was found",   category=UserWarning)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_STRIDE = 720          # overlapping — roughly doubles training samples per patient
TEST_STRIDE  = WINDOW_SIZE  # non-overlapping — one prediction per distinct window
THR          = 0.5          # fixed; tuning over 91 thresholds on 4 val patients is just noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_val_patients(
    records: List[PatientRecord], seed: int, n_per_class: int = 2
) -> Tuple[List[PatientRecord], List[PatientRecord]]:
    """Hold out n_per_class patients from each class for early stopping.

    Stratified so val always contains both classes — if all 4 val patients
    are healthy, the MCC early-stopping signal is blind to depression quality.
    Fold-specific seed means different folds see different val patients.
    """
    rng  = random.Random(seed)
    deps = [r for r in records if r.label == 1]
    ctls = [r for r in records if r.label == 0]
    val  = []
    if len(deps) >= n_per_class:
        val += rng.sample(deps, n_per_class)
    if len(ctls) >= n_per_class:
        val += rng.sample(ctls, n_per_class)
    val_ids = {r.pid for r in val}
    tr = [r for r in records if r.pid not in val_ids]
    return tr, val


def make_loader(ds: ActiDataset, bs: int, nw: int, train: bool, balance: bool) -> DataLoader:
    if train and balance:
        labels  = np.array(ds.labels, dtype=np.int64)
        counts  = np.bincount(labels, minlength=2)
        w0, w1  = 1.0 / max(counts[0], 1), 1.0 / max(counts[1], 1)
        weights = np.where(labels == 1, w1, w0).astype(np.float64)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=nw,
                          pin_memory=DEVICE.type == "cuda")
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=nw,
                      pin_memory=DEVICE.type == "cuda")


def train_epoch(model, loader, criterion, opt, scaler, amp: bool) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x = x.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)

        if amp and DEVICE.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            # unscale before clip_grad_norm_ — the scaler inflates gradients
            # by ~65536x to keep them in fp16 range; clipping them before
            # unscaling zeroes out almost every gradient and kills learning
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total += loss.item() * y.shape[0]
        n     += y.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def get_probs(model, loader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    probs, ys, pids = [], [], []
    for x, y, pid in loader:
        p = torch.sigmoid(model(x.to(DEVICE, dtype=torch.float32))).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
        pids.extend(list(pid))
    return np.concatenate(probs), np.concatenate(ys), pids


def patient_agg(
    probs: np.ndarray, pids: List[str], ys: np.ndarray, mode: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Group window probabilities by patient and return one score per patient.

    'frac' needs a threshold to compute, but using THR inside the val loop
    defeats the point of a continuous early-stopping signal. Falls back to
    mean for val; THR is applied correctly at final test time in agg_test_patient.
    """
    bucket: Dict[str, List[float]] = defaultdict(list)
    label:  Dict[str, int]         = {}
    for p, pid, y in zip(probs, pids, ys):
        bucket[pid].append(float(p))
        label[pid] = int(y)

    scores, labels, ids = [], [], []
    for pid, ps in bucket.items():
        arr = np.array(ps)
        s   = float(np.median(arr)) if mode == "median" else float(np.mean(arr))
        scores.append(s)
        labels.append(label[pid])
        ids.append(pid)

    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int64), ids


def agg_test_patient(probs: np.ndarray, mode: str) -> float:
    if mode == "median":
        return float(np.median(probs))
    if mode == "frac":
        # fraction of windows above THR — e.g. "60% of this patient's day looks depressed"
        return float(np.mean(probs >= THR))
    return float(np.mean(probs))


def completed_folds(fold_csv: Path) -> Set[int]:
    if not fold_csv.exists():
        return set()
    try:
        return set(pd.read_csv(fold_csv)["fold"].astype(int).tolist())
    except Exception:
        return set()  # partial/corrupted file from a previous crash — start clean


# ---------------------------------------------------------------------------
# Main LOPO loop
# ---------------------------------------------------------------------------

def run_lopo(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    log.info("=" * 70)
    log.info("ActiPheno  |  device=%s  seed=%d  agg=%s  thr=%.2f (fixed)",
             DEVICE, args.seed, args.agg, THR)
    log.info("=" * 70)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv   = out_dir / f"fold_preds_seed{args.seed}_agg{args.agg}.csv"
    window_csv = out_dir / f"window_preds_seed{args.seed}_agg{args.agg}.csv"

    done = completed_folds(fold_csv)
    if done:
        log.info("resuming — %d fold(s) already done: %s", len(done), sorted(done))

    data = DepresjonLoader(data_dir=args.data_dir, min_windows=1)

    win_cm = np.zeros((2, 2), dtype=np.int64)
    pat_cm = np.zeros((2, 2), dtype=np.int64)

    for fold_idx, (train_recs, test_rec) in enumerate(data.lopo_splits(), start=1):
        if args.max_folds > 0 and fold_idx > args.max_folds:
            break
        if fold_idx in done:
            log.info("fold %d — already done, skipping", fold_idx)
            continue

        # ── patient split ──────────────────────────────────────────────────
        tr, val = split_val_patients(train_recs, args.seed + fold_idx, args.val_per_class)
        if not val:
            tr, val = train_recs, []

        # ── normalisation on train patients only — test rec never touches this
        mu, std = compute_fold_stats(tr, use_delta=args.use_delta)

        aug = RandomAugment(args.jitter, args.scale, args.shift) if args.augment else None

        train_ds = ActiDataset(tr,          mu, std, args.use_delta, aug,  TRAIN_STRIDE)
        val_ds   = (ActiDataset(val, mu, std, args.use_delta, None, TEST_STRIDE)
                    if val else None)
        test_ds  = ActiDataset([test_rec],  mu, std, args.use_delta, None, TEST_STRIDE)

        log.info("fold %d | held-out=%s (label=%d) | train=%d val=%s test=%d",
                 fold_idx, test_rec.pid, test_rec.label,
                 len(train_ds), len(val_ds) if val_ds else 0, len(test_ds))

        train_dl = make_loader(train_ds, args.batch_size, args.num_workers, True,  args.balance)
        val_dl   = (make_loader(val_ds, args.batch_size, args.num_workers, False, False)
                    if val_ds else None)
        test_dl  = make_loader(test_ds,  args.batch_size, args.num_workers, False, False)

        # ── model ──────────────────────────────────────────────────────────
        in_ch = 2 if args.use_delta else 1
        model = ActiPheno(ActiPhenoConfig(in_channels=in_ch)).to(DEVICE)

        if args.pos_weight == "auto" and not args.balance:
            labels_arr = np.array(train_ds.labels)
            n_pos = labels_arr.sum()
            n_neg = len(labels_arr) - n_pos
            pw = float(n_neg) / max(n_pos, 1)
        else:
            pw = 1.0

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=DEVICE))
        opt    = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and DEVICE.type == "cuda"))

        # ── training ───────────────────────────────────────────────────────
        best_state: Optional[Dict] = None
        best_mcc   = -1e9
        patience   = args.patience

        for ep in range(1, args.epochs + 1):
            train_epoch(model, train_dl, criterion, opt, scaler, args.amp)

            if val_dl is not None:
                vp, vy, vpids = get_probs(model, val_dl)
                vscores, vlabels, _ = patient_agg(vp, vpids, vy, args.agg)
                mcc = compute_metrics(vlabels, vscores, thr=THR)["mcc"]

                if mcc > best_mcc:
                    best_mcc   = mcc
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    patience = args.patience
                else:
                    patience -= 1

                if patience <= 0:
                    log.info("  early stop ep %d  (best val_mcc=%.3f)", ep, best_mcc)
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

        # ── test ───────────────────────────────────────────────────────────
        tp_arr, ty, _ = get_probs(model, test_dl)
        win_metrics   = compute_metrics(ty, tp_arr, thr=THR)
        win_cm       += win_metrics["cm"].astype(np.int64)

        pat_score = agg_test_patient(tp_arr, args.agg)
        pat_pred  = int(pat_score >= THR)
        pat_true  = int(test_rec.label)
        pat_cm[pat_true, pat_pred] += 1

        log.info("  true=%d pred=%d | score=%.3f | win_f1=%.3f win_mcc=%.3f",
                 pat_true, pat_pred, pat_score,
                 win_metrics["f1"], win_metrics["mcc"])

        # dump immediately — Colab keeps dying around fold 45 ───────────────
        fold_row = {
            "seed": args.seed, "fold": fold_idx, "patient_id": test_rec.pid,
            "patient_true": pat_true, "patient_pred": pat_pred,
            "patient_score": float(pat_score), "n_windows": len(tp_arr),
            "window_f1": float(win_metrics["f1"]),
            "window_mcc": float(win_metrics["mcc"]),
        }
        win_rows = [
            {"seed": args.seed, "fold": fold_idx, "patient_id": test_rec.pid,
             "y_true": int(ty[i]), "p": float(tp_arr[i])}
            for i in range(len(tp_arr))
        ]

        pd.DataFrame([fold_row]).to_csv(
            fold_csv, mode="a", header=not fold_csv.exists(), index=False
        )
        pd.DataFrame(win_rows).to_csv(
            window_csv, mode="a", header=not window_csv.exists(), index=False
        )
        done.add(fold_idx)

    # rebuild global CMs from CSVs to handle partial resumes cleanly ────────
    if fold_csv.exists() and window_csv.exists():
        df_f = pd.read_csv(fold_csv)
        df_w = pd.read_csv(window_csv)
        pat_cm = np.zeros((2, 2), dtype=np.int64)
        win_cm = np.zeros((2, 2), dtype=np.int64)
        for _, row in df_f.iterrows():
            pat_cm[int(row["patient_true"]), int(row["patient_pred"])] += 1
        for _, row in df_w.iterrows():
            win_cm[int(row["y_true"]), int(row["p"] >= THR)] += 1

    print_global(f"WINDOW-LEVEL   (seed={args.seed}, agg={args.agg})", win_cm)
    print_global(f"PATIENT-LEVEL  (seed={args.seed}, agg={args.agg})", pat_cm)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ActiPheno LOPO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",      default="data")
    p.add_argument("--out_dir",       default="runs")
    p.add_argument("--epochs",        type=int,   default=25)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--wd",            type=float, default=1e-3)
    p.add_argument("--batch_size",    type=int,   default=32)
    # num_workers 0 on Colab — WeightedRandomSampler + multiprocessing
    # deadlocks silently and you only find out 40 folds in
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--use_delta",     action="store_true")
    p.add_argument("--augment",       action="store_true")
    p.add_argument("--jitter",        type=float, default=0.03)
    p.add_argument("--scale",         type=float, default=0.05)
    p.add_argument("--shift",         type=int,   default=30)
    p.add_argument("--val_per_class", type=int,   default=2)
    p.add_argument("--patience",      type=int,   default=5)
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--agg",           default="mean", choices=["mean", "median", "frac"])
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--max_folds",     type=int,   default=-1,
                   help="-1 runs all folds")

    # both fight class imbalance but via different mechanisms — using both
    # doubly compensates and sends the model into depression-prediction overdrive
    imb = p.add_mutually_exclusive_group()
    imb.add_argument("--balance",    action="store_true",
                     help="WeightedRandomSampler on minority class")
    imb.add_argument("--pos_weight", default="auto", choices=["auto", "none"],
                     help="BCEWithLogitsLoss pos_weight — auto sets n_neg/n_pos per fold")

    return p.parse_args()


if __name__ == "__main__":
    run_lopo(parse_args())
