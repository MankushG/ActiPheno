"""
data_loader.py  —  ActiPheno
Raw CSV loading, windowing, fold normalisation, augmentation, LOPO splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

WINDOW_SIZE  = 1440   # one full day at 1-minute resolution
ACTIVITY_COL = "activity"
LABEL_DEP    = 1
LABEL_CTL    = 0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PatientRecord:
    pid:     str
    label:   int
    raw:     np.ndarray
    madrs:   Optional[float] = None
    # non-overlapping windows kept for min_windows gating and fold norm stats;
    # actual train/test windowing always re-slices from raw with its own stride
    windows: List[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> np.ndarray:
    """Load one patient CSV and return a clean float32 activity array.

    log1p is not applied here — it runs in __getitem__ so augmentation
    operates on the log-scale signal, same as what the model sees.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    if ACTIVITY_COL not in df.columns:
        raise KeyError(f"no '{ACTIVITY_COL}' col in {path.name}; got {list(df.columns)}")

    s = df[ACTIVITY_COL].astype(float)
    if s.isna().any():
        s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # watch physically cannot record negative counts
    return np.clip(s.to_numpy(dtype=np.float32), 0.0, None)


def _make_windows(arr: np.ndarray, stride: int = WINDOW_SIZE) -> List[np.ndarray]:
    # non-overlapping (stride == WINDOW_SIZE) for test — no duplicate predictions
    # overlapping (stride < WINDOW_SIZE) for train — more samples, lower variance
    out, start = [], 0
    while start + WINDOW_SIZE <= len(arr):
        out.append(arr[start: start + WINDOW_SIZE])
        start += stride
    return out


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class DepresjonLoader:
    """Loads the Depresjon dataset and exposes a LOPO split iterator.

    Skipped patients are printed explicitly — in a 55-patient dataset,
    losing two patients silently shifts the LOPO confusion matrix enough
    to matter.
    """

    def __init__(self, data_dir: str | Path = "data", min_windows: int = 1) -> None:
        self.data_dir    = Path(data_dir)
        self.min_windows = min_windows
        self._scores: Dict[str, float] = {}
        self.records:  Dict[str, PatientRecord] = {}
        self._skipped: List[Tuple[str, str]]    = []

        self._load_scores()
        self._load_group("condition", LABEL_DEP)
        self._load_group("control",   LABEL_CTL)
        self._report()

    def _load_scores(self) -> None:
        p = self.data_dir / "scores.csv"
        if not p.exists():
            log.warning("scores.csv not found — MADRS unavailable")
            return
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip().str.lower()
        id_col    = next(
            (c for c in df.columns if c in {"number", "id", "patient_id", "patient"}),
            df.columns[0],
        )
        score_col = next((c for c in df.columns if "madrs" in c), None)
        if score_col is None:
            return
        for _, row in df.iterrows():
            self._scores[str(row[id_col]).strip()] = (
                float(row[score_col]) if pd.notna(row[score_col]) else None
            )

    def _load_group(self, group: str, label: int) -> None:
        folder = self.data_dir / group
        if not folder.exists():
            log.warning("folder not found: %s", folder)
            return

        for csv_path in sorted(folder.glob("*.csv")):
            pid = csv_path.stem
            try:
                raw = _read_csv(csv_path)
            except Exception as e:
                print(f"  skipping broken file {csv_path.name}: {e}")
                self._skipped.append((pid, f"read error: {e}"))
                continue

            wins = _make_windows(raw)
            if len(wins) < self.min_windows:
                reason = f"only {len(wins)} full day(s) (need >= {self.min_windows})"
                self._skipped.append((pid, reason))
                continue

            madrs = self._scores.get(pid) or self._scores.get(pid.lstrip("0") or "0")
            self.records[pid] = PatientRecord(
                pid=pid, label=label, raw=raw, madrs=madrs, windows=wins
            )

    def _report(self) -> None:
        n_dep = sum(r.label == LABEL_DEP for r in self.records.values())
        n_ctl = sum(r.label == LABEL_CTL for r in self.records.values())
        log.info("loaded %d patients  (%d depressed | %d healthy)", len(self.records), n_dep, n_ctl)
        if self._skipped:
            log.warning("%d patient(s) SKIPPED:", len(self._skipped))
            for pid, why in self._skipped:
                log.warning("  ✗  %s  —  %s", pid, why)
        # print(f"loaded {len(self.records)} patients... finally")

    @property
    def patient_ids(self) -> List[str]:
        return list(self.records.keys())

    def lopo_splits(self) -> Iterator[Tuple[List[PatientRecord], PatientRecord]]:
        """Yield (train_records, test_record) for every LOPO fold.

        Order is deterministic (sorted at load time). Windows are never
        shuffled across patient boundaries — shuffling leaks the test
        patient's data into training and invalidates the whole evaluation.
        """
        ids = self.patient_ids
        for test_id in ids:
            train = [self.records[i] for i in ids if i != test_id]
            yield train, self.records[test_id]


# ---------------------------------------------------------------------------
# Fold normalisation — Welford online algorithm
# ---------------------------------------------------------------------------

def compute_fold_stats(
    records: List[PatientRecord],
    use_delta: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over training-fold patients only.

    Welford online algorithm — updates running stats one window at a time,
    no need to materialise the full dataset in RAM.

    Training fold only — using all 55 patients leaks the test patient's
    activity distribution into the normalisation, which is a subtle but
    real form of data leakage in LOPO.

    Non-overlapping windows used for stats so patients with longer
    recordings don't dominate the estimates.
    """
    C = 2 if use_delta else 1
    n, mu, M2 = np.zeros(C), np.zeros(C), np.zeros(C)

    def _update(c: int, batch: np.ndarray) -> None:
        nonlocal n, mu, M2
        b   = batch.astype(np.float64)
        m   = b.size
        if m == 0:
            return
        bm  = b.mean()
        bM2 = b.var(ddof=0) * m
        if n[c] == 0:
            n[c], mu[c], M2[c] = m, bm, bM2
            return
        d      = bm - mu[c]
        total  = n[c] + m
        mu[c] += d * (m / total)
        M2[c] += bM2 + d**2 * (n[c] * m / total)
        n[c]   = total

    for rec in records:
        for w in rec.windows:
            x0 = np.log1p(w.astype(np.float32))
            _update(0, x0)
            if use_delta:
                # first diff of log-activity — wake/sleep transitions are
                # far more visible here than in the raw magnitude signal
                _update(1, np.diff(x0, prepend=x0[0]).astype(np.float32))

    std = np.sqrt(M2 / np.maximum(n, 1.0))
    return mu.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

class RandomAugment:
    """Circular shift, amplitude scale, and additive jitter — training only.

    Circular shift simulates different watch start times across recording days.
    Amplitude scale captures day-to-day variability in absolute activity level.
    Jitter is mostly just regularisation.

    None of these break circadian structure — no time-axis flips, no random
    masking, nothing that produces a biologically implausible signal.
    """

    def __init__(self, jitter: float = 0.03, scale: float = 0.05, shift: int = 30):
        self.jitter = jitter
        self.scale  = scale
        self.shift  = shift

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.shift > 0:
            k = np.random.randint(-self.shift, self.shift + 1)
            if k:
                x = np.roll(x, k)
        if self.scale > 0:
            x = x * np.float32(np.random.normal(1.0, self.scale))
        if self.jitter > 0:
            x = x + np.random.normal(0.0, self.jitter, size=x.shape).astype(np.float32)
        return x.astype(np.float32)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ActiDataset(Dataset):
    """Windows from a list of PatientRecords, normalised and channel-stacked.

    Returns (x, y, pid):
      x   — float32 tensor (C, WINDOW_SIZE), z-scored with fold-level stats
      y   — float32 scalar; BCEWithLogitsLoss needs float labels, not long
      pid — patient ID string for grouping window predictions at eval time

    Two channels:
      ch 0 = log1p(raw activity)   absolute activity level
      ch 1 = delta of ch 0         moment-to-moment transitions

    The delta channel makes wake/sleep edges salient to the CNN. Depression-
    linked circadian disruption shows up most clearly at those transitions —
    the model should not have to learn to compute differences from scratch.
    """

    def __init__(
        self,
        records:   List[PatientRecord],
        mu:        np.ndarray,
        std:       np.ndarray,
        use_delta: bool = True,
        augment:   Optional[RandomAugment] = None,
        stride:    int = WINDOW_SIZE,
    ) -> None:
        self.mu        = mu.astype(np.float32)
        self.std       = std.astype(np.float32)
        self.use_delta = use_delta
        self.augment   = augment

        # pre-slice so DataLoader workers don't need PatientRecord refs
        self.items: List[Tuple[np.ndarray, int, str]] = []
        for rec in records:
            for w in _make_windows(rec.raw, stride=stride):
                self.items.append((w.astype(np.float32), rec.label, rec.pid))

        self.labels = [lbl for _, lbl, _ in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        w, y, pid = self.items[idx]

        x0 = np.log1p(w)
        if self.augment is not None:
            x0 = self.augment(x0)

        if self.use_delta:
            x1 = np.diff(x0, prepend=x0[0]).astype(np.float32)
            x  = np.stack([x0, x1], axis=0)   # (2, T)
        else:
            x  = x0[np.newaxis, :]             # (1, T)

        x = (x - self.mu[:, np.newaxis]) / self.std[:, np.newaxis]
        return (
            torch.from_numpy(x).float(),
            torch.tensor(float(y), dtype=torch.float32),
            pid,
        )
