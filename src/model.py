"""
model.py  —  ActiPheno
1D-CNN  →  BiLSTM  →  Attention pooling  →  logit
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


@dataclass
class ActiPhenoConfig:
    in_channels:  int = 2
    window_size:  int = 1440
    cnn_channels: Optional[List[int]] = None
    cnn_kernel:   int   = 7
    cnn_pool:     int   = 2
    cnn_dropout:  float = 0.10
    lstm_hidden:  int   = 96
    lstm_layers:  int   = 1
    lstm_dropout: float = 0.0
    attn_hidden:  int   = 64
    fc_hidden:    int   = 64
    head_dropout: float = 0.35

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]


class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU → Dropout → MaxPool.

    GroupNorm normalises within each sample independently, so it stays
    stable when LOPO fold batch sizes drop to single digits. BatchNorm
    over 4–8 samples produces noisy statistics that destabilise training.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int, dropout: float):
        super().__init__()
        n_groups = 8 if out_ch % 8 == 0 else (4 if out_ch % 4 == 0 else 1)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.GroupNorm(n_groups, out_ch),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(kernel_size=pool, stride=pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ActiPheno(nn.Module):
    """1D-CNN + BiLSTM + Attention pooling for binary depression classification.

    Input  (B, C, 1440)  —  C=1 raw only, C=2 raw + delta
    Output (B,)          —  raw logits; sigmoid applied externally at inference

    Stage 1 — CNN:
      Kernel size 7 = ±3.5 min receptive field at 1-minute resolution.
      Right scale for local motifs: alarm-waking spikes, brief sedentary
      periods, meal-time bursts. Three pool(2) stages compress 1440 → 180
      before the LSTM, keeping sequence length manageable.

    Stage 2 — BiLSTM:
      Circadian rhythms are full-sequence phenomena. The BiLSTM sees all
      180 CNN outputs in both directions, capturing how morning activity
      patterns relate to evening ones — the phase relationships where
      depression-linked disruption is most consistent in the literature.

    Stage 3 — Attention pooling:
      Mean-pooling 180 LSTM outputs dilutes signal from the clinically
      discriminative windows (nocturnal inactivity, early-morning restlessness).
      Learned per-timestep weights let the model concentrate on those periods.

    Output logits not probabilities — BCEWithLogitsLoss fuses sigmoid + BCE
    numerically, avoiding log(0) on saturated outputs.
    """

    def __init__(self, cfg: Optional[ActiPhenoConfig] = None) -> None:
        super().__init__()
        cfg = cfg or ActiPhenoConfig()
        self.cfg = cfg

        ch = [cfg.in_channels] + cfg.cnn_channels
        self.cnn = nn.Sequential(*[
            ConvBlock(ch[i], ch[i + 1], cfg.cnn_kernel, cfg.cnn_pool, cfg.cnn_dropout)
            for i in range(len(cfg.cnn_channels))
        ])

        self.lstm = nn.LSTM(
            input_size=cfg.cnn_channels[-1],
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )
        lstm_dim = cfg.lstm_hidden * 2

        self.attn = nn.Sequential(
            nn.Linear(lstm_dim, cfg.attn_hidden),
            nn.Tanh(),
            nn.Linear(cfg.attn_hidden, 1),
        )

        self.head = nn.Sequential(
            nn.Dropout(p=cfg.head_dropout),
            nn.Linear(lstm_dim, cfg.fc_hidden),
            nn.GELU(),
            nn.Dropout(p=cfg.head_dropout / 2),
            nn.Linear(cfg.fc_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p.data)
                    elif "weight_hh" in name:
                        # orthogonal recurrent weights keep gradient norms
                        # near 1 through long sequences
                        nn.init.orthogonal_(p.data)
                    elif "bias" in name:
                        nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)           # (B, 128, 180)
        x = x.permute(0, 2, 1)   # (B, 180, 128)

        h, _ = self.lstm(x)       # (B, 180, 192)

        scores  = self.attn(h).squeeze(-1)      # (B, 180)
        weights = torch.softmax(scores, dim=1)  # (B, 180)
        pooled  = torch.sum(h * weights.unsqueeze(-1), dim=1)  # (B, 192)

        return self.head(pooled).squeeze(-1)    # (B,)
