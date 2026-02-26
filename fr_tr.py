from typing import Optional, List, Set, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from fr import FERN
import numpy as np
from other_models.PatchTST import Model as PatchTST
from other_models.DLinear import Model as DLinear
from other_models.TimeMixer import Model as TimeMixer
from typing import Tuple, Union, Sequence, Optional, Literal
import study.fr_cfg as configs
import study.fr_data_mgr as data_mgr
from pydantic import (
    Field, BaseModel, ConfigDict, 
    computed_field, field_validator, PositiveInt, 
)
import matplotlib.pyplot as plt

import study.fr_tr_metrics as metrics 

from torch.optim.lr_scheduler import (
    _LRScheduler, ReduceLROnPlateau,
    LinearLR, CosineAnnealingLR,
    SequentialLR, OneCycleLR, CosineAnnealingWarmRestarts, 
) 
from loguru import logger
import time
import gc
import matplotlib.pyplot as plt 
from pathlib import Path
from safetensors.torch import save_file, load_file, save_model, load_model
from torch.cuda.amp import autocast, GradScaler
from colored import fg, attr
import study.fr_gen as fgen
import study.fr_core as fcore

# region EarlyStopping
class EarlyStopper(BaseModel):
    """A Pydantic model to manage early stopping logic.""" 
    patience: PositiveInt
    best_val: float = float("inf")
    counter: int = 0

    def check(self, val_obj: float) -> bool:
        """Checks the validation objective and updates the counter."""
        improved = val_obj < self.best_val 
        log_parts = [metrics.cprint("Val obj", "light_magenta")]

        if improved:
            log_parts.extend(
                [
                    metrics.cprint("improved", "light_green", bold=True),
                    metrics.cprint(f"{self.best_val:.4f}", "white"),
                    metrics.cprint("→", "light_green"),
                    metrics.cprint(f"{val_obj:.4f}", "light_green", bold=True),
                    metrics.cprint("(saving checkpoint)", "light_blue"),
                ]
            )
            self.best_val = val_obj
            self.counter = 0

        else:
            log_parts.extend(
                [
                    metrics.cprint("did not improve", "light_red"),
                    metrics.cprint(f"({val_obj:.4f} > {self.best_val:.4f})", "white"),
                    metrics.cprint(
                        f"counter {self.counter + 1}/{self.patience}", "light_yellow"
                    ),
                ]
            )
            self.counter += 1

        final_log_message = " ".join(log_parts)
        print(f"    -> {final_log_message}")

        return improved

    @property
    def should_stop(self) -> bool:
        """Determines if training should stop."""
        return self.counter >= self.patience

    def reset(self):
        """Resets the stopper to its initial state."""
        self.best_val = float("inf")
        self.counter = 0
# endregion EarlyStopping


def to_cpu_pinned(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.is_cuda:
        out = torch.empty_like(t, device='cpu', pin_memory=True)
        out.copy_(t, non_blocking=True)     # async HtoD if dst is pinned
        return out
    else:
        return t.clone()  # keep semantics similar to .detach().cpu()
 
"""
_weight utilities
Intent/contract:
- Take any 1D weight-like array (length T or T-1) and turn it into a high-contrast [0,1] array of length T.
- Steps: nan-safe → optional smoothing → robust percentile clip → min-max → gamma stretch.
- This yields weights the viewer can actually see on long polylines.
"""

def _nan_to_num_1d(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if not np.isfinite(w).all():
        w = np.nan_to_num(w, nan=np.nanmedian(w), posinf=np.max(np.isfinite(w) and w or [0]), neginf=np.min(np.isfinite(w) and w or [0]))
    return w

def _smooth_ema(w: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    if alpha <= 0: 
        return w
    out = w.copy()
    for i in range(1, len(w)):
        out[i] = alpha * out[i] + (1 - alpha) * out[i-1]
    return out

def _normalize_weights(
    w: np.ndarray,
    T: int,
    *,
    smooth_alpha: float = 0.15,
    clip_lo: float = 5.0,
    clip_hi: float = 95.0,
    gamma: float = 0.8,
) -> np.ndarray:
    """
    Input: w length T or T-1 or anything; Output: length T in [0,1] with contrast.
    - smooth_alpha in [0,1]: 0=no smoothing, larger=more smoothing
    - clip_lo/clip_hi: robust percentile range
    - gamma < 1 boosts mid-highs (more contrast); gamma > 1 compresses highs
    """
    if w is None:
        return None
    w = _nan_to_num_1d(np.asarray(w))
    # Align to T points (same contract as _add_gradient_line)
    if w.size == T - 1:
        w = np.concatenate([w, w[-1:]], axis=0)  # pad last segment value
    elif w.size != T:
        w = np.interp(
            np.linspace(0, w.size - 1, T),
            np.arange(w.size, dtype=np.float32),
            w.astype(np.float32),
        )
    # optional smoothing
    if smooth_alpha > 0:
        w = _smooth_ema(w, alpha=smooth_alpha)
    # robust clipping
    lo = np.percentile(w, clip_lo)
    hi = np.percentile(w, clip_hi)
    if hi <= lo:
        hi = lo + 1e-6
    w = np.clip(w, lo, hi)
    # min-max → [0,1]
    w = (w - lo) / (hi - lo + 1e-12)
    # gamma for contrast
    if gamma != 1.0:
        w = np.clip(w, 1e-9, 1.0) ** gamma
    return w.astype(np.float32)

def _normalize_weights_with_fixed_range(
    w: np.ndarray,
    T: int,
    *,
    lo: float | None = None,
    hi: float | None = None,
    smooth_alpha: float = 0.15,
    gamma: float = 0.8,
) -> np.ndarray:
    """
    Contract:
    - w: 1D array length T or T-1 or arbitrary; returns length T in [0,1].
    - If lo/hi are provided, we DO NOT recompute percentiles per series.
      We clip to [lo, hi] globally so same absolute values share the same color.
    - We keep smoothing and gamma for visibility but never change lo/hi.
    """
    if w is None:
        return None
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    # align to T points
    if w.size == T - 1:
        w = np.concatenate([w, w[-1:]], axis=0)
    elif w.size != T:
        w = np.interp(np.linspace(0, w.size - 1, T),
                      np.arange(w.size, dtype=np.float32),
                      w.astype(np.float32))
    # optional smoothing for less flicker
    if smooth_alpha > 0:
        out = w.copy()
        for i in range(1, T):
            out[i] = smooth_alpha * out[i] + (1 - smooth_alpha) * out[i-1]
        w = out
    # fixed range clamp
    if lo is None or hi is None or hi <= lo:
        lo = float(np.percentile(w, 5.0))
        hi = float(np.percentile(w, 95.0))
        if hi <= lo:
            hi = lo + 1e-6
    w = np.clip(w, lo, hi)
    w = (w - lo) / (hi - lo + 1e-12)
    if gamma != 1.0:
        w = np.clip(w, 1e-9, 1.0) ** gamma
    return w.astype(np.float32)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

 
def _norm_01(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-series min-max normalize to [0,1]."""
    v = v.astype(np.float32)
    lo, hi = np.min(v), np.max(v)
    den = max(hi - lo, eps)
    return (v - lo) / den

def _add_gradient_line(
    ax: plt.Axes,                 # where to draw
    y: np.ndarray,                # 1D array of y-values (length T)
    *,
    w_alpha: np.ndarray | None = None,  # per-point weights in [0,1] to drive transparency
    w_lw:    np.ndarray | None = None,  # per-point weights in [0,1] to drive thickness
    color: str = "#E69F00",             # base color when not using a colormap
    cmap: str | None = None,            # optional colormap name; if set, color varies by w_alpha
    alpha_base: float = 0.15,           # minimum transparency
    alpha_scale: float = 0.85,          # how much transparency can increase from base
    lw_min: float = 1.25,               # minimum line width
    lw_max: float = 3.25,               # maximum line width
    zorder: int = 4,                    # draw above curves with lower zorder
):
    """
    _add_gradient_line
    Intent/contract:
    - Draw a polyline whose per-segment alpha/line-width are modulated by weights.
    - y: length T; there are (T-1) segments.
    - w_alpha / w_lw: accept length T or length T-1. If neither matches, resample to T.
    - This guard prevents broadcast errors if upstream series lengths ever drift.
    """
    T = len(y)
    x = np.arange(T, dtype=np.float32)

    def _align_weight(w, name):
        if w is None:
            return None
        w = np.asarray(w).reshape(-1)
        if len(w) == T:
            return w
        if len(w) == T - 1:
            # per-segment → per-point by padding last
            return np.concatenate([w, w[-1:]], axis=0)
        # Fallback: resample to T with linear interpolation
        w_lin = np.interp(
            np.linspace(0, len(w) - 1, T),
            np.arange(len(w), dtype=np.float32),
            w.astype(np.float32),
        )
        print(f"⚠️  {name} length {len(w)} ≠ T={T}; resampled to T.")
        return w_lin

    w_alpha = _align_weight(w_alpha, "w_alpha")
    w_lw    = _align_weight(w_lw,    "w_lw")

    if w_alpha is not None:
        assert len(w_alpha) == T, f"w_alpha has len {len(w_alpha)}, expected {T}"
    if w_lw is not None:
        assert len(w_lw) == T, f"w_lw has len {len(w_lw)}, expected {T}"

    """
    Convert the polyline into segments for LineCollection.
    Shape becomes [Nseg, 2, 2] where Nseg = T-1. Each segment is [(x_k,y_k), (x_{k+1},y_{k+1})].
    This is why later we index weights with [:-1]: there are T-1 segments.
    """
    # segments: [Nsegments, 2, 2] → [[(x0,y0),(x1,y1)], ...]
    segs = np.stack([np.column_stack([x[:-1], y[:-1]]),
                     np.column_stack([x[1:],  y[1:]])], axis=1)
    
    nseg = segs.shape[0] # Cache the number of segments.
    assert nseg == T - 1, f"nseg={nseg}, but T-1={T-1}"
    """
    Colormap branch: if a cmap is provided, color each segment by w_alpha (after slicing to T-1).
    colors = cmap_obj(weights) returns RGBA per segment based on the colormap.

    Then we override the alpha channel (colors[:, 3]) to be alpha_base + alpha_scale * weight, 
    clipped to [0,1]. This keeps transparency meaningful even when the colormap has its own alpha.
    """
    if cmap:
        cmap_obj = plt.get_cmap(cmap) 
        colors = cmap_obj((w_alpha[:-1] if w_alpha is not None else np.zeros(nseg)))
        # override alpha from weights
        colors[:, 3] = (alpha_base + alpha_scale * (w_alpha[:-1] if w_alpha is not None else 0)).clip(0, 1)
    else: 
        """
        Solid-color branch: all segments start with the same RGBA. If w_alpha is given, 
        vary only the alpha channel per segment using the same base + scale * weight rule.
        """
        rgba = np.array(mpl.colors.to_rgba(color), dtype=np.float32)
        colors = np.tile(rgba, (nseg, 1))
        if w_alpha is not None:
            colors[:, 3] = (alpha_base + alpha_scale * w_alpha[:-1]).clip(0, 1)
    """
    Compute per-segment line widths. Default to lw_min. If w_lw is supplied, map it linearly 
    into [lw_min, lw_max] (again using [:-1] to match segment count).
    """
    lws = np.full(nseg, lw_min, dtype=np.float32)
    if w_lw is not None:
        lws = lw_min + (lw_max - lw_min) * w_lw[:-1]
    """
    Create the LineCollection with your segments, per-segment colors, and per-segment widths.
    Add it to the Axes. (You can call ax.autoscale_view() after this if there 
    wasn’t already a line plotted that set limits.)
    """
    lc = LineCollection(segs, colors=colors, linewidths=lws, zorder=zorder)
    ax.add_collection(lc)

#region SequenceCollector
class SequenceCollector(BaseModel): 
    """
    Contract: shapes for SequenceCollector
    - Incoming pred/target windows (append to pred_lst/target_lst):
    pred:   [B, H, D]
    target: [B, H, D]
    - Incoming eigen windows (append to temp_eig_*_seq_lst):
    Emax/Esum/Elog: [B, D, H]

    Finalize stitching:
    - Pred/Target: permute to [B, D, H] before tde_to_seq(stride, offset).
    - Eigen: pass as-is [B, D, H] to tde_to_seq(stride, offset).

    Resulting stitched sequences:
    - pred_seq, truth_seq:          [D, T]
    - eig_patch_{max,sum,logdet}_seq: [D, T]

    Sanity asserts (recommended at append time):
    - pred.ndim == target.ndim == 3 and pred.shape == target.shape
    - eig.ndim == 3 and eig.shape[-1] == pred.shape[1]  # eig H matches pred H
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)
    model_type:  str
    seed: int = Field(default=1955)
    results_path: Path

    pred_lst: List[torch.Tensor] = Field(default_factory=list) # [B, H, D]
    target_lst: List[torch.Tensor] = Field(default_factory=list) 

    eig_patches_mags:   list[torch.Tensor] = Field(default_factory=list)  #NOTE [B,D,H]
    eig_patches_max:    list[torch.Tensor] = Field(default_factory=list)  #NOTE [B,D,H]
    eig_patches_sum:    list[torch.Tensor] = Field(default_factory=list)  #NOTE [B,D,H]
    eig_patches_logdet: list[torch.Tensor] = Field(default_factory=list)  #NOTE [B,D,H]

    pred_seq:   Optional[np.ndarray] = None # [D, Full_length]
    truth_seq:  Optional[np.ndarray] = None
    eig_max:    Optional[np.ndarray] = None   # [D, T]
    eig_sum:    Optional[np.ndarray] = None
    eig_logdet: Optional[np.ndarray] = None

    pred_samples:  Optional[np.ndarray] = None   # will be set in finalize()
    truth_samples: Optional[np.ndarray] = None # [5, D, H]
    eig_max_samples: np.ndarray | None = None   # [5, D, H]
    eig_sum_samples: np.ndarray | None = None   # [5, D, H]
    sample_mode: Literal["quantile", "random"] = "quantile"
    
    # for stitching into seq, dataset are all default to stride=1, offset=0
    stride: int = 1
    offset: int = 0

    pred_len: Optional[int] = None
    patch_size: Optional[int] = None
      
    def reset_temp(self) -> None:
        """Clear ALL per-phase buffers and counters. Call at the START of each val/test phase."""
        self.pred_lst.clear()
        self.target_lst.clear()
        self.eig_patches_max.clear()
        self.eig_patches_sum.clear()
        self.eig_patches_logdet.clear()
     
    @classmethod
    def from_npz(cls, data: "np.lib.npyio.NpzFile") -> "SequenceCollector":
        stride = int(data["stride"]) if "stride" in data.files else 1
        offset = int(data["offset"]) if "offset" in data.files else 0
        return cls(
            pred_seq=data["pred_seq"],
            truth_seq=data["truth_seq"],         # map truth_seq -> truth_seq
            pred_samples=data["pred_samples"],
            truth_samples=data["truth_samples"],
            stride=stride,
            offset=offset,
        )

    def add_batch(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError(f"pred/target must be [B,D,H]; got pred {tuple(pred.shape)}, target {tuple(target.shape)}")
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")
        
        self.pred_lst.append(to_cpu_pinned(pred))
        self.target_lst.append(to_cpu_pinned(target))
    
    def add_eigs(self, coef_eigs: fgen.CoefEigenMonitor):  
        assert coef_eigs.eig_patches_mags is not None,   "eig_patches_mags is None; update_from_states() missing?"
        assert coef_eigs.eig_patches_max is not None,    "eig_patches_max is None; update_from_states() missing?"
        assert coef_eigs.eig_patches_sum is not None,    "eig_patches_sum is None; update_from_states() missing?"
        assert coef_eigs.eig_patches_logdet is not None, "eig_patches_logdet is None; update_from_states() missing?"

        self.eig_patches_max.append(coef_eigs.eig_patches_max)
        self.eig_patches_sum.append(coef_eigs.eig_patches_sum)
        self.eig_patches_logdet.append(coef_eigs.eig_patches_logdet)
        self.eig_patches_mags.append(coef_eigs.eig_patches_mags)
         
    def finalize(self) -> "SequenceCollector":
        """
        SequenceCollector.finalize
        Intent/contract:
        - Pred windows: [B, H, D] → (permute) [B, D, H] → tde_to_seq → [D, T].
        - Eigen ribbons arrive as [B, D, H] where H is the window length. DO NOT permute to [B, H, C].
        Pass [B, D, H] directly to tde_to_seq so S=H and stitched T matches preds.
        - After stitching: T_pred == T_sum == T_max must hold.
        - TDE overlays use the same 'picks' indices and have shape [5, D, H].
        """
        # List[B, H, D] -> [B, D, H]
        pred_tde = torch.cat(self.pred_lst, dim=0).permute(0, 2, 1).contiguous().cpu().numpy()
        target_tde = torch.cat(self.target_lst, dim=0).permute(0, 2, 1).contiguous().cpu().numpy()
        N_pred = pred_tde.shape[0] 
        if self.sample_mode == "quantile":
            picks = [0, N_pred//4, N_pred//2, 3*N_pred//4, N_pred-1]
        elif self.sample_mode == "random":
            rng = np.random.default_rng(1955)
            picks = rng.choice(N_pred, 5, replace=False)
        else:
            raise ValueError(f"Unknown sample mode: {self.sample_mode}")
        self.pred_samples = pred_tde[picks]
        self.truth_samples = target_tde[picks]
 
        # --- 2) Stitch preds/targets to full sequences [D, T] ---
        self.pred_seq = tde_to_seq(pred_tde, stride=self.stride, offset=self.offset)
        self.truth_seq = tde_to_seq(target_tde, stride=self.stride, offset=self.offset)
        self.pred_samples = pred_tde[picks]    # [5, D, H]
        self.truth_samples = target_tde[picks] # [5, D, H] 

        self.pred_lst.clear() #  clear these now if you want to free memory
        self.target_lst.clear()
        
        # --- 3) Eigen ribbons 
        if (self.eig_patches_max):
            Emax  = torch.cat(self.eig_patches_max, dim=0)   # [B,D,H]
            Esum  = torch.cat(self.eig_patches_sum, dim=0)
            Elog  = torch.cat(self.eig_patches_logdet, dim=0)

            N_eig = Emax.shape[0]
            if N_eig != N_pred:
                raise RuntimeError(f"Pred/eigen window counts differ: pred B={N_pred}, eig B={N_eig}.")

            # stitch like preds: permute to [B,H,C] → numpy → tde_to_seq(...)
            def _stitch(E: torch.Tensor) -> np.ndarray:
                E_bdh = E.contiguous().cpu().numpy()  # [B, D, H] → treat H as S
                return tde_to_seq(E_bdh, stride=self.stride, offset=self.offset)  # [D,T]
            self.eig_max    = _stitch(Emax) # [D, T]
            self.eig_sum    = _stitch(Esum)
            self.eig_logdet = _stitch(Elog)
        
            # TDE overlays for the same windows
            self.eig_max_samples = Emax[picks].detach().cpu().numpy()   # [5, C, H]
            self.eig_sum_samples = Esum[picks].detach().cpu().numpy()   # [5, C, H]

            T_pred = self.pred_seq.shape[1]
            T_sum  = self.eig_sum.shape[1]
            T_max  = self.eig_max.shape[1]
  
            if not (T_pred == T_sum == T_max):
                raise RuntimeError(
                f"Eigen ribbons length mismatch: pred_seq T={T_pred}, sum T={T_sum}, max T={T_max}. "
                "Ensure eigen windows stayed [B,D,H] (no permute) so S=H for tde_to_seq."
            )
        return self
    
    def plot_seq(self, 
        feature_idx: int = 0,  
        save_path: str = "seq_plot_ggplot.png",
        *,
        overlay_eigs: bool = True,
        eig_alpha_from: Literal["sum","max"] = "sum",
        eig_lw_from:    Literal["max","sum"] = "max",
        eig_cmap: str | None = "viridis",          # e.g. "viridis"; None uses solid color
        draw_base_pred: bool = False,         # 有渐变时默认不再画纯色基准线
    ):
        """Generates a ggplot-style plot of the full sequence."""

        ## --- Style Definitions ---
        PRED_COLOR = "#E69F00"  # A vivid orange
        TRUTH_COLOR = "#56B4E9"  # A sky blue

        # Use 'ggplot' style context
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            if draw_base_pred:
                # 没有渐变时，或明确要求画基准线时，才画纯色预测
                ax.plot(
                    self.pred_seq[feature_idx], label="Prediction",
                    color=PRED_COLOR, linewidth=1.5,
                )
            ax.plot(
                self.truth_seq[feature_idx], label="Ground Truth",
                color=TRUTH_COLOR, linewidth=1.8, alpha=0.8,
            )

            if overlay_eigs and self.eig_sum is not None and self.eig_max is not None:
                # choose sources
                series_alpha = self.eig_sum[feature_idx] if eig_alpha_from == "sum" else self.eig_max[feature_idx]
                series_lw    = self.eig_max[feature_idx] if eig_lw_from    == "max" else self.eig_sum[feature_idx]
                # normalize per-feature
                w_alpha = _norm_01(series_alpha)
                w_lw    = _norm_01(series_lw)

                if getattr(self, "eig_sum", None) is not None:
                    w_drive = self.eig_sum[feature_idx]              # [T]
                else:
                    w_drive = np.abs(self.truth_seq[feature_idx] - self.pred_seq[feature_idx])  # [T]

                T = self.pred_seq.shape[1]
                w = _normalize_weights(w_drive, T, smooth_alpha=0.15, clip_lo=5, clip_hi=95, gamma=0.8)

                # overlay on top of the prediction line (you can also overlay on truth)
                _add_gradient_line(
                    ax, self.pred_seq[feature_idx],
                    w_alpha=w_alpha, w_lw=w_lw,
                    color=PRED_COLOR, cmap=eig_cmap,
                    alpha_base=0.22, alpha_scale=0.88, lw_min=1.2, lw_max=2.90, zorder=5,
                )
            """
            If you’d rather draw a separate visualization, add a sibling plot_seq_eigs(feature_idx=0, kind="both") 
            that just renders the two normalized series as simple lines or as a filled band.
            """ 
            ax.set_title( ## --- Aesthetics and Labels ---
                f"Full Sequence Comparison (Feature {feature_idx}, Seed {self.seed})",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
 
            ax.legend(fontsize=11) # The ggplot style has a nice legend frame by default
 
            fig.tight_layout() ## --- Final Touches ---
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🖼️  Sequence plot saved to: {save_path}")
            plt.show()

    def plot_tde(self, 
        feature_idx: int = 0,
        batch_idx: int = 0, 
        to_save: bool = False, 
        save_path: str = "tde_plot_ggplot.png",
        *,
        overlay_eigs: bool = True,
        eig_alpha_from: Literal["sum","max"] = "sum",
        eig_lw_from:    Literal["max","sum"] = "max",
        eig_cmap: str | None = 'viridis',          # e.g. "viridis"; None uses solid color    
        draw_base_pred: bool = False,
    ):
        """Generates a ggplot-style plot of a single TDE sample."""

        ## --- Style Definitions ---
        PRED_COLOR = "#009E73"  # A bluish green
        TRUTH_COLOR = "#D55E00"  # A vermilion

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(9, 5))
 
            if draw_base_pred: ## --- Plot Data --- 
                ax.plot(
                    self.pred_samples[batch_idx, feature_idx, :],
                    label="Reconstructed Prediction",
                    color=PRED_COLOR, linewidth=2.0, linestyle="--",
                )
            ax.plot(
                self.truth_samples[batch_idx, feature_idx, :],
                label="True Target", color=TRUTH_COLOR,
                linewidth=2.2, marker="o", markersize=6,
                markevery=10, markerfacecolor="white",  # Contrasts well with gray background
                markeredgewidth=1.5,
            )
            if overlay_eigs and self.eig_sum_samples is not None and self.eig_max_samples is not None:
                series_alpha = self.eig_sum_samples[batch_idx, feature_idx, :] if eig_alpha_from == "sum" else self.eig_max_samples[batch_idx, feature_idx, :]
                series_lw    = self.eig_max_samples[batch_idx, feature_idx, :] if eig_lw_from    == "max" else self.eig_sum_samples[batch_idx, feature_idx, :]
                w_alpha = _norm_01(series_alpha)
                w_lw    = _norm_01(series_lw)

                if getattr(self, "eig_sum", None) is not None:
                    w_drive = self.eig_sum[feature_idx]              # [T]
                else:
                    w_drive = np.abs(self.truth_seq[feature_idx] - self.pred_seq[feature_idx])  # [T]

                T = self.pred_seq.shape[1]
                w = _normalize_weights(w_drive, T, smooth_alpha=0.15, clip_lo=5, clip_hi=95, gamma=0.8)

                _add_gradient_line(
                    ax, self.pred_samples[batch_idx, feature_idx, :],
                    w_alpha=w_alpha, w_lw=w_lw,
                    color=PRED_COLOR, cmap=eig_cmap,
                    alpha_base=0.22, alpha_scale=0.88, lw_min=1.2, lw_max=2.9, zorder=5,
                )
 
            ax.set_title( ## --- Aesthetics and Labels ---
                "Time-Delay Embedding Reconstruction",
                fontsize=15, fontweight="bold",
                pad=15,
            )
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
 
            ax.legend(fontsize=10) # Keep the default framed legend which works well in ggplot
 
            fig.tight_layout() ## --- Final Touches ---
            if to_save:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"🖼️  TDE plot saved to: {save_path}") 
            plt.show()
#endregion SequenceCollector

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.0+: "high" = allow TF32

class PhaseTime(BaseModel):
    """Accumulate wall-clock time (seconds) for a logical phase.

    Intent:
        Summarize how much time is spent in e.g. training, validation, test.
    Assumptions:
        Caller passes non-negative durations from time.time() deltas.
    Returns:
        total_s: float
            Accumulated wall-clock seconds for this phase.
    """
    name: str
    total_s: float = 0.0

    def add(self, delta_s: float) -> None: 
        self.total_s += float(delta_s) # REQUIRE: delta_s >= 0

class Trainer(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,validate_assignment=False,populate_by_name=True,   
        extra="forbid",
        ) # <— enables parsing by alias names below
    
    # --- inputs ---
    cfg: configs.BaseTrainingConfig = Field(alias="config")
    db: data_mgr.DataBundle = Field(alias="data_bundle")
    seed: int
    
    # --- lightweight fields set immediately ---
    device: str = "cuda"
    experiment_name: Optional[str] = None
    model_type: Optional[str] = None
    
    # --- heavy runtime objects (set in model_post_init) ---
    model: Optional[nn.Module] = None
    opt: torch.optim.Optimizer | None = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    mgr: Optional[metrics.MetricManager] = None
    earlystopper: EarlyStopper = Field(default_factory=lambda: EarlyStopper(patience=5))
    
    results_path: Optional[Path] = None
    collector: Optional[SequenceCollector] = None
    
    # AMP
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float32
    scaler: torch.amp.GradScaler | None = None
    
    # per-batch ephemera
    epoch: int = 0
    x: torch.Tensor | None = None
    y: torch.Tensor | None = None
    pred: torch.Tensor | None = None
    last_states: fgen.States | None = None
    
    stage_objective: torch.Tensor | None = None

    # Auxiliary loss (model-specific)
    aux_loss: dict[str, torch.Tensor] | None = None  # CHG: optional model-specific scalar
     # Data scaling flags/stats (copied from DataBundle)
    no_scale: bool = False 
    
    # Time
    time_train: PhaseTime = Field(default_factory=lambda: PhaseTime(name="train"))
    time_val: PhaseTime = Field(default_factory=lambda: PhaseTime(name="val"))
    time_test_epochs: PhaseTime = Field(default_factory=lambda: PhaseTime(name="test_during_training"))
 
    training_time: float | None = None  # keep as alias for total run inside train_model
     
        
    def model_post_init(self, __ctx): 
        torch.manual_seed(self.seed); np.random.seed(self.seed) # seeds + device 
        self.device = self.db.device

        # build model/opt/scheduler
        self.model = ModelFactory(self.cfg)                  # nn.Module adapter
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate,
                                    weight_decay=0.0, fused=False)
        self.scheduler = SchedulerFactory.build(self.cfg, self.opt)

        # metrics manager uses global std depending on scaling mode
        global_std = (self.db.train_std if self.db.no_scale
                    else np.array([1.0]*self.db.train_std.shape[0]))
        self.mgr = metrics.MetricManager.setup_metric_manager(cfg=self.cfg, global_std=global_std,
                                        seed=self.seed, use_proj_swd=self.cfg.use_proj_swd)

        # early stopping
        self.earlystopper = EarlyStopper(patience=self.cfg.patience)
        
        # experiment identity (FIX: no trailing commas)
        self.experiment_name = self.db.experiment_name
        self.model_type = self.cfg.model_type

        # paths
        self.results_path = Path("results") / self.model_type / str(self.seed)
        (self.results_path / "models").mkdir(parents=True, exist_ok=True)
        (self.results_path / "predictions").mkdir(parents=True, exist_ok=True)
        (self.results_path / "csv").mkdir(parents=True, exist_ok=True)

        # collector
        self.collector = SequenceCollector(
            model_type=self.cfg.model_type,
            results_path=self.results_path,  seed=self.seed,
            pred_len=self.cfg.pred_len if self.cfg.model_type == "fr" else None,      # critical
            patch_size=self.cfg.patch_size if self.cfg.model_type == "fr" else None,  # critical
        )
        if isinstance(self.cfg, configs.FERNConfig) and self.cfg.patch_size is not None:
            self.collector.pred_len = self.cfg.pred_len
            self.collector.patch_size = self.cfg.patch_size

        # AMP setup
        self.amp_enabled = (self.cfg.device.startswith("cuda") and getattr(self.cfg, "amp_enabled", True))
        self.amp_dtype = (torch.bfloat16 if (self.amp_enabled and torch.cuda.is_bf16_supported()) else torch.float32)
        self.scaler = torch.amp.GradScaler(enabled=self.amp_enabled and self.amp_dtype is torch.float16)
        # (GradScaler is unnecessary for bf16; PyTorch skips it when enabled=False)
  
        print(f"Post-init: Creating directory at {self.models_path} and {self.predictions_path}")
 
    @computed_field
    @property
    def models_path(self) -> Path:
        return (self.results_path / "models" / f"{self.model_type}_{self.seed}.safetensors")

    @computed_field
    @property
    def predictions_path(self) -> Path:
        return (self.results_path / "predictions" / f"{self.model_type}_{self.seed}_pred_pack.npz")

    @computed_field
    @property
    def csv_path(self) -> Path:
        return (self.results_path / "csv" / f"{self.model_type}_{self.seed}.csv")
    
    @field_validator("experiment_name", "model_type")
    def name_must_not_be_whitespace(cls, v: str) -> str:
        """Ensure that string fields are not just whitespace."""
        if not v.strip():
            raise ValueError("Field must not be empty or contain only whitespace.")
        return v
    
    def save_model(self) -> Path:
        from safetensors.torch import save_model
        path = self.models_path
        save_model(self.model, path)  # preserves shared tensors
        print(f"✅ Model saved to: {path}")
        return path
    
    def load_model(self) -> Path:
        from safetensors.torch import load_model
        path = self.models_path
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")
        load_model(self.model, str(path), strict=True, device=self.device)  # model-first API
        print(f"✅ Model loaded from: {path}")
        return path
    
    def save_predictions_npz(self) -> Path:
        """Save all prediction artifacts (seq + 5 samples) into one compressed NPZ."""
        path = self.predictions_path
        np.savez_compressed(
            path,
            pred_seq=self.collector.pred_seq,          # type: ignore[union-attr]
            truth_seq=self.collector.target_seq,       # type: ignore[union-attr]
            pred_samples=self.collector.pred_samples,  # [5, D, T]
            truth_samples=self.collector.target_samples,
            stride=self.collector.stride,
            offset=self.collector.offset,
        )
        print(f"✅ Predictions bundle saved to: {path}")
        return path

    def load_predictions_npz(self) -> "SequenceCollector":
        """Load the compressed predictions bundle."""
        path = self.predictions_path 
        if not path.exists():
            raise FileNotFoundError(f"No predictions bundle at {path}")
        with np.load(path, allow_pickle=False) as data:
            sc = SequenceCollector.from_npz(data)
        print(f"✅ Predictions bundle loaded from: {path}")
        return sc
   
    def lighten(self):
        """Keep this Trainer for plots/debug, but free heavy state."""
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.opt = None
        self.scheduler = None
        # self.db = None   # (or keep just self.db.device if you still need it)
        # optional: clear per-batch tensors
        self.x = self.y = self.pred = None
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self
    
    def __or__(self, fn):
        """Pipeline operator: `self | step1 | step2`."""
        result = fn(self)
        return result if result is not None else self
    
    def _reset_batch(self):
        """Clear ephemeral fields before loading a new batch."""
        self.x = self.y = self.pred = None
        
    def check_pred_exists(self):
        """Raises an error if the 'pred' field has not been populated."""
        if self.pred is None:
            raise ValueError("'pred' is None — ensure forward ran before metrics/backward.")
    
    @property
    def best_val_metrics(self) -> torch.Tensor:
        return self.mgr.best_val_metrics

    @property
    def best_test_metrics(self) -> torch.Tensor:
        return self.mgr.best_test_metrics
 
    @property
    def training_time_in_minutes_and_seconds(self) -> None:   
        if self.training_time is None:
            print("Training not completed yet")
            return
        minutes = int(self.training_time // 60)
        seconds = self.training_time % 60
        print(f"Time taken: {minutes}m {seconds:.2f}s")   
        
    def train_model(self):
        self.epoch = 0
        start_total = time.time()
        self.earlystopper.reset()  # work across epochs

        for epoch in range(self.cfg.epochs):
            self.epoch=epoch
            # ====== TRAINING PHASE ======
            t0 = time.time()
            self = (self 
                    | phase_before_train
                    )
            with torch.set_grad_enabled(True):
                for batch_x, batch_y in self.db.loader_data.train_loader: # , ch
                    self.x=batch_x.to( self.db.device, non_blocking=True ) 
                    self.y=batch_y.to(self.db.device, non_blocking=True) 
                    self = (self 
                            | batch_train_forward 
                            | batch_train_metrics 
                            | batch_train_backward
                            )
            self = (self 
                    | phase_train
                    )
            self.time_train.add(time.time() - t0)  # CHG: accumulate train time

            # --- VALIDATION PHASE ---
            t0 = time.time()
            self = self | phase_before_val
            with torch.set_grad_enabled(False):
                for batch_x, batch_y in self.db.loader_data.val_loader: # , ch
                    self.x=batch_x.to( self.db.device, non_blocking=True ) 
                    self.y=batch_y.to(self.db.device, non_blocking=True)
                    self = (self 
                            | batch_val_forward 
                            | batch_val_metrics 
                            )
            self = (self 
                    | phase_val
                    )
            self.time_val.add(time.time() - t0)  # CHG: accumulate val time

            if self.earlystopper.should_stop:
                print(f"Early stopping at epoch {self.epoch}")
                break

            # --- TESTING PHASE ---
            t0 = time.time()
            self = (self 
                    | phase_before_test
                    )
            with torch.set_grad_enabled(False):
                for batch_x, batch_y in self.db.loader_data.test_loader: # , ch
                    self.x=batch_x.to( self.db.device, non_blocking=True ) 
                    self.y=batch_y.to(self.db.device, non_blocking=True)
                    self = (self 
                            | batch_test_forward 
                            | batch_test_metrics 
                            | batch_test_collect
                            )
            self = (self 
                    | phase_test
                    )
            self.time_test_epochs.add(time.time() - t0)  # CHG: accumulate per-epoch test time


        # --- FINAL TESTING PHASE --- 
        print(" ---- TESTING FINAL MODEL ----")
        load_path = self.load_model()
        self = (self 
                | phase_before_test
                )
        with torch.set_grad_enabled(False):
            for i, (batch_x, batch_y) in enumerate(self.db.loader_data.test_loader): # , ch
                self.x=batch_x.to( self.db.device, non_blocking=True ) 
                self.y=batch_y.to(self.db.device, non_blocking=True)
                self = (self 
                        | batch_test_forward 
                        | batch_test_metrics 
                        | batch_test_collect # add to collector
                        )
   
        self = (self 
                | phase_final
                ) 
        self.training_time = time.time() - start_total  # CHG: single total wall-clock
        return self
# region BatchBundle Functions

def _assert_lengths(tr):
    exp_seq = tr.model.config.seq_len
    exp_pred = tr.model.config.pred_len
    assert tr.x.shape[1] == exp_seq, f"x length {tr.x.shape[1]} ≠ cfg.seq_len {exp_seq}"
    assert tr.y.shape[1] == exp_pred, (
        f"y length {tr.y.shape[1]} ≠ cfg.pred_len {exp_pred}"
    )
    assert tr.pred.shape[1] == exp_pred, (
        f"pred length {tr.pred.shape[1]} ≠ cfg.pred_len {exp_pred}"
    )

def batch_test_collect(tr: "Trainer")->"Trainer":  
    tr.collector.add_batch(tr.pred, tr.y) 
    if tr.cfg.model_type == "FERN":
        st = tr.last_states    
        tr.collector.add_eigs(st.monitor)
        return tr 
    else:
        return tr

def phase_before_train(tr: "Trainer")->"Trainer":
    tr.model.train()
    tr.mgr.train.reset()
    tr._reset_batch()  
    return tr
    
def phase_before_val(tr: "Trainer")->"Trainer":
    tr.model.eval()
    tr.mgr.val.reset()
    tr._reset_batch()  
    return tr

def phase_before_test(tr: "Trainer")->"Trainer":
    tr.model.eval()
    tr.mgr.test.reset()
    tr._reset_batch()  
    tr.collector.reset_temp()
    return tr
    
def batch_train_forward(tr: "Trainer")->"Trainer":
    assert tr.x is not None and tr.y is not None, "Call after loading batch"
    with torch.autocast(device_type="cuda", dtype=tr.amp_dtype, enabled=tr.amp_enabled):
        out = tr.model(tr.x)                # your ModelAdapter
        if isinstance(out, dict):
            tr.last_states = out.get("states", None)
            tr.aux_loss = out.get("aux_loss", None)
        else:
            tr.last_states = None
            tr.aux_loss = None
        tr.pred = tr.model._extract_prediction(out)
    tr.check_pred_exists()
    _assert_lengths(tr)  # <--- add
    return tr

def _to_tensor_like(x: torch.Tensor, a: Union[float, Sequence, torch.Tensor]) -> torch.Tensor:
    """Convert numpy/list/scalar/torch to a tensor on x's device/dtype."""
    return torch.as_tensor(a, dtype=x.dtype, device=x.device)


def batch_val_forward(tr: "Trainer")->"Trainer":
    with torch.autocast(device_type="cuda", dtype=tr.amp_dtype, enabled=tr.amp_enabled):
        out = tr.model(tr.x, update=False)
        if isinstance(out, dict) and "states" in out:
            tr.last_states = out["states"]
        else:
            tr.last_states = None
        tr.pred = tr.model._extract_prediction(out)   
    tr.check_pred_exists()
    _assert_lengths(tr)  # <--- add
    return tr


def batch_test_forward(tr: "Trainer")->"Trainer":
    tr.model.eval()
    with torch.autocast(device_type="cuda", dtype=tr.amp_dtype, enabled=tr.amp_enabled):
        out = tr.model(tr.x, update=False)
        if isinstance(out, dict) and "states" in out:
            tr.last_states = out["states"]
        else:
            tr.last_states = None
        tr.pred = tr.model._extract_prediction(out)  
    tr.check_pred_exists()
    _assert_lengths(tr)  # <--- add

    return tr


def batch_train_metrics(tr: "Trainer")->"Trainer":
    with torch.autocast(device_type="cuda", dtype=tr.amp_dtype, enabled=tr.amp_enabled):
        if tr.cfg.model_type == "PFNN":
            # PFNN: use one-step MSE as training objective; still update metrics for logging.
            first_pred = tr.pred[:, 0, :]   # [B, D]
            first_true = tr.y[:, 0, :]      # [B, D]
            tr.stage_objective = F.mse_loss(first_pred, first_true, reduction='mean')
            # Optional: still log full-horizon metrics for analysis
            _ = tr.mgr.train.update_and_calc_objective(tr.pred, tr.y)
        else:
            tr.stage_objective = tr.mgr.train.update_and_calc_objective(tr.pred, tr.y)
        # print(f"{tr.amp_dtype} {tr.amp_enabled}")
        # tr.stage_objective = tr.mgr.train.update_and_calc_objective(tr.pred, tr.y)
    return tr


def batch_val_metrics(tr: "Trainer")->"Trainer":
    with torch.autocast(device_type="cuda", dtype=tr.amp_dtype, enabled=tr.amp_enabled):
        tr.stage_objective = tr.mgr.val.update_and_calc_objective(tr.pred, tr.y)
    return tr


def batch_test_metrics(tr: "Trainer")->"Trainer":
    with torch.autocast(device_type="cuda", dtype=tr.amp_dtype, enabled=tr.amp_enabled):
        tr.stage_objective = tr.mgr.test.update_and_calc_objective(tr.pred, tr.y)
    return tr


def batch_train_backward(tr: "Trainer")->"Trainer":
    tr.amp_enabled = (tr.cfg.device.startswith("cuda") and getattr(tr.cfg, "amp_enabled", True)) 
    tr.opt.zero_grad(set_to_none=True)  #!!! close for optimization
    
    # CHG: combine main objective with any aux losses using cfg.aux_loss_weights
    total_loss = tr.stage_objective
    if tr.cfg.model_type == "PFNN" and tr.cfg.aux_loss_weights is None:
        raise ValueError("PFNN model requires aux_loss_weights and it is not there")
    if (tr.cfg.model_type == "PFNN" 
        and tr.aux_loss is None 
        and (tr.aux_loss['pfnn_id'] is None or tr.aux_loss['pfnn_contr'] is None)
    ):
        raise ValueError("PFNN model requires aux_loss and it is not there")
    if tr.aux_loss:
        for name, loss in tr.aux_loss.items():
            if name not in tr.cfg.aux_loss_weights:
                raise ValueError(f"Auxiliary loss {name} not found in cfg.aux_loss_weights")
            if tr.cfg.model_type != "PFNN": 
                w = tr.cfg.aux_loss_weights[name]
                if w != 0.0:
                    total_loss = total_loss + w * loss
            else: # PFNN training requires two stages, handled manually
                # NOTE: PFNN logic is here
                if tr.epoch < 6: #TODO a sensible setting
                    if name == "pfnn_contr":
                        """Early Stages when not on attractor, use minimal contractive loss 
                        to allow >1 eigenvalue stretches; later we enforce <= 1 eigenvalues"""
                        w = tr.cfg.aux_loss_weights[name] 
                        total_loss = total_loss + 0.001 * loss 
                    if name == "pfnn_id":
                        w = tr.cfg.aux_loss_weights[name] 
                        total_loss = total_loss + w * loss 
                else:
                    w = tr.cfg.aux_loss_weights[name]
                    total_loss = total_loss + w * loss

                
    # CHG: combine main objective with any aux losses using cfg.aux_loss_weights

    if tr.scaler and tr.scaler.is_enabled():
        tr.scaler.scale(total_loss).backward()
        # after total_loss.backward()
        max_g = 0.0
        max_name = None
        for name, p in tr.model.named_parameters():
            if p.grad is not None:
                g = p.grad.abs().max().item()
                if g > max_g:
                    max_g, max_name = g, name
        if max_g < 1e-6 or max_g > 3e3:
            print(f"[WARNING] max |grad|={max_g:.2e} on param {max_name}")
        # max_grad = max(p.grad.abs().max().item() 
        #             for p in tr.model.parameters() if p.grad is not None)
        # if max_grad < 1e-6 or max_grad > 1e3:
        #     print(f"[WARNING] Gradient magnitude: {max_grad:.2e}") 
        tr.scaler.unscale_(tr.opt) # if you clip grads, unscale first:
        # torch.nn.utils.clip_grad_norm_(tr.model.parameters(), max_norm)
        tr.scaler.step(tr.opt)
        tr.scaler.update() 
    else:
        total_loss.backward()
        # after total_loss.backward()
        max_g = 0.0
        max_name = None
        for name, p in tr.model.named_parameters():
            if p.grad is not None:
                g = p.grad.abs().max().item()
                if g > max_g:
                    max_g, max_name = g, name
        if max_g < 1e-6 or max_g > 1e3:
            print(f"[WARNING] max |grad|={max_g:.2e} on param {max_name}")
        # max_grad = max(p.grad.abs().max().item() 
        #             for p in tr.model.parameters() if p.grad is not None)
        # if max_grad < 1e-6 or max_grad > 1e3:
        #     print(f"[WARNING] Gradient magnitude: {max_grad:.2e}")
        # max_norm = 2.0
        # total_norm = torch.nn.utils.clip_grad_norm_(tr.model.parameters(), max_norm)
        # print(f"Total norm: {total_norm}")
        tr.opt.step() 
    return tr
  

def phase_train(tr: "Trainer")->"Trainer":
    report = tr.mgr.train.collect_metrics()  
    # print(f"Epoch {tr.epoch + 1} | {report}")
    if tr.cfg.model_type == "PFNN":
        # CHG: debug PFNN first-step training loss
        with torch.no_grad():
            print(f"[PFNN] epoch {tr.epoch+1} first-step MSE (stage_objective) = "
                  f"{tr.stage_objective.item():.4f}")
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), colorize=True,
           format="{message}")
    logger.info(f"Epoch {tr.epoch + 1} | {report}")
    return tr


def phase_val(tr: "Trainer")->"Trainer":
    report = tr.mgr.val.collect_metrics()          # compute once
    val_obj_smooth = tr.mgr.commit_val(report)  # or b.mgr.commit_val(report) if you combined them
    tr.mgr.last_val = report                       # store canonical copy

 
    print(f"Epoch {tr.epoch + 1} | {report} | val_obj_smooth: {val_obj_smooth}")
    if tr.scheduler:
        if isinstance(tr.scheduler, ReduceLROnPlateau):
            tr.scheduler.step(val_obj_smooth)          # end of val epoch
        elif isinstance(tr.scheduler, OneCycleLR):
            pass  # stepped per-batch elsewhere
        else:
            tr.scheduler.step()                         # end of train epoch
    # if b.scheduler:
    #     b.scheduler.step(b.stage_objective)
    if tr.epoch > 2: #TODO this is the place setting grace period
        improved: bool = tr.earlystopper.check(val_obj_smooth)
        if improved: 
            tr.save_model() # save_path = save_model(tr.checkpoint, tr.seed, tr.model)
            tr.mgr.best_val_metrics = report 
    return tr


def phase_test(tr: "Trainer")->"Trainer":
    report = tr.mgr.test.collect_metrics() 
    print(f"Epoch {tr.epoch + 1} | {report}")
    return tr


def phase_final(tr: "Trainer")->"Trainer":
    report = tr.mgr.test.collect_metrics() 
    print(f"Best Test ReRun | {tr.model.config.model_type} | {report}") 
    tr.collector.finalize() 
    tr.mgr.best_test_metrics = report 
    return tr
# endregion BatchBundle Functions
 
# region ModelAdapter
class ModelAdapter(nn.Module):
    """A single, generic adapter that wraps any model."""

    def __init__(self, config: configs.BaseTrainingConfig, underlying_model: nn.Module):
        super().__init__()
        self.config = config
        self.model = underlying_model

    def _extract_prediction(self, raw_output: Union[torch.Tensor, dict]) -> torch.Tensor:
        """Internal method to handle the different output formats."""
        sig = self.config.output_signature
        if sig == "tensor":
            if not isinstance(raw_output, torch.Tensor):
                raise TypeError(
                    f"Model was configured for 'tensor' output but returned {type(raw_output)}"
                )
            return raw_output
        elif sig == "dict:pred":
            if not isinstance(raw_output, dict):
                raise TypeError(f""""Model was configured for 'dict:pred' output but 
                                returned {type(raw_output)}""")
            pred = raw_output.get("pred")
            if pred is None:
                raise ValueError("Model output is dict but has no 'pred' key")
            return pred
        # Add other cases like 'obj.pred' or 'tuple[0]' if needed
        else:
            raise NotImplementedError(f"Output signature '{sig}' is not implemented.")

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """ Calls the underlying model's forward pass using the signature
        specified in the Pydantic configuration. """
        sig = self.config.forward_signature

        if sig == "x,update":  # For models like FERN
            return self.model(x, update=update)

        elif sig == "x,none,none,none":  # For models like PatchTST
            return self.model(x, None, None, None)

        elif sig == "x":  # For models like DLinear
            return self.model(x)

        elif sig == "naive_repeat":  # The model itself is just this adapter.
            last = x[:, -1:, :]
            return last.repeat(1, self.config.pred_len, 1)

        else:
            raise NotImplementedError(f"Forward signature '{sig}' is not implemented.")
# endregion ModelAdapter


# region ModelFactory
def ModelFactory(config: configs.BaseTrainingConfig) -> ModelAdapter:
    """wraps underlying model for a unified interface"""

    model_type = config.model_type

    if model_type == "FERN":
        underlying_model = FERN(config)
    elif model_type == "PatchTST":
        underlying_model = PatchTST(config)
    elif model_type == "DLinear":
        underlying_model = DLinear(config)
    elif model_type == "TimeMixer":
        underlying_model = TimeMixer(config)
    # elif model_type == "HoloMorph":
    #     underlying_model = HoloMorph(config)
    elif model_type == "Attraos":
        from study.other_models.Attraos import Model as Attraos
        underlying_model = Attraos(config)
    elif model_type == "Koopa":
        from study.other_models.Koopa import Model as Koopa
        underlying_model = Koopa(config)
    elif model_type == "ModernTCN":
        from study.other_models.MordernTCN import Model as ModernTCN
        underlying_model = ModernTCN(config)
    elif model_type == "PFNN":
        from study.other_models.Pfnn import Model as PFNNModel
        underlying_model = PFNNModel(config) 
    elif model_type == "naive":  # For the naive case, the adapter IS the model.
        underlying_model = (
            nn.Identity()
        )  # Note: We pass nn.Identity() as a placeholder, it won't be used.
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return ModelAdapter(config=config, underlying_model=underlying_model).to(config.device)
# endregion ModelFactory


def debug_check_model_gradients(model):
    grads = [
        p.grad.abs().max().item() for p in model.parameters() if p.grad is not None
    ]
    if not grads or any(g is None for g in grads):
        raise RuntimeError("""[Train Batch] No gradients found on any parameters after backward!""")
    print(f"  [Train Batch] Max abs grad among params: {max(grads):.6f}")




def tde_to_seq(
    windows: np.ndarray,  # Shape: (B, D, S) -> (Batch, Dimensions/Features, Sequence Length)
    stride: int = 1,
    offset: int = 0,
    mode: Literal["average", "sum"] = "average",
) -> np.ndarray:
    """  
    [[[10, 10, 10]]],  # b=0
    [[[20, 20, 20]]],  # b=1  

    T = offset + (B-1)*stride + S = 0 + (4-1)*1 + 3 = 6 
    acc = [[0, 0, 0, 0, 0, 0]] (Shape [1, 6]) 
    count = [[0, 0, 0, 0, 0, 0]] (Shape [1, 6]) 

    b = 0 (Window [10, 10, 10]) 
    start=0, end=3 
    acc = [[10, 10, 10, 0, 0, 0]] 
    count = [[ 1, 1, 1, 0, 0, 0]] 
    b = 1 (Window [20, 20, 20]) 
    start=1, end=4 
    acc = [[10, 30, 30, 20, 0, 0]] (e.g., 10+20=30) 
    count = [[ 1, 2, 2, 1, 0, 0]] 
    """
    B, D, S = windows.shape
    T = offset + (B - 1) * stride + S  # Calculate the full length of the series
    acc = np.zeros((D, T), dtype=windows.dtype)
    count = np.zeros((D, T), dtype=np.int32)

    for b in range(B):
        start = offset + b * stride
        end = start + S
        acc[:, start:end] += windows[b]
        count[:, start:end] += 1
    if mode == "sum":
        return acc 
    count = np.maximum(count, 1) # Prevent division by zero  
    return acc / count
 
# region SchedulerFactory
class SchedulerFactory:
    """A factory class for creating PyTorch learning rate schedulers based on a configuration object."""

    @staticmethod
    def build(
        cfg: configs.BaseTrainingConfig, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Args:
            cfg: A configuration object that contains scheduler settings. 
            It must have attributes like 'scheduler_type', 'learning_rate', 'warmup_epochs', etc.
            optimizer: The PyTorch optimizer to which the scheduler will be attached.

        Returns:
            A PyTorch learning rate scheduler instance, or None if the scheduler_type is 'none'
            or not recognized.
        """
        scheduler_type = getattr(cfg, "scheduler_type", "none")

        if scheduler_type == "plateau":
            print(
                f"""[*] ReduceLROnPlateau scheduler enabled with 
                patience={cfg.lr_scheduler_patience} and factor={cfg.lr_scheduler_factor}"""
            )
            assert hasattr(cfg, "lr_scheduler_patience"), "cfg must have 'lr_scheduler_patience'"
            assert hasattr(cfg, "lr_scheduler_factor"), "cfg must have 'lr_scheduler_factor'"
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode="min",
                factor=cfg.lr_scheduler_factor, patience=cfg.lr_scheduler_patience,
            )

        elif scheduler_type == "cosine":
            print(f"""[*] Cosine scheduler with {cfg.warmup_epochs}-epoch warmup enabled.""")
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.001,  # Start LR at 0.1% of the initial LR
                total_iters=cfg.warmup_epochs,
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.eta_min
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[cfg.warmup_epochs],
            )

        elif scheduler_type == "none":
            print("[*] No learning rate scheduler will be used.")
            return None

        else:
            print(f"""[!] Warning: Scheduler type '{scheduler_type}' not recognized. 
                No scheduler will be used.""")
            return None
# endregion SchedulerFactory

def inv_standardize(
    x: torch.Tensor,
    mean: Union[float, Sequence, torch.Tensor],
    std:  Union[float, Sequence, torch.Tensor],
    ch: Optional[torch.Tensor] = None,      # [N] channel ids; required for CI with per-channel stats
    *,
    feature_axis: int = -1,                 # which axis is the 'channel/feature' axis in x
    sample_axis: int = 0,                   # which axis aligns with ch (CI uses 0)
    strict: bool = True                     # if True, raise when shapes would expand
) -> torch.Tensor:
    """
    Unified inverse-standardization that NEVER changes x.shape.

    Cases:
    - Non-CI: x[..., F], mean/std shape [F] or scalar -> broadcast along feature_axis.
    - CI:     x[..., 1] with ch given and mean/std shape [D] -> gather per-sample stats via ch.
    - Global: mean/std scalar -> broadcast everywhere.

    Returns: tensor with SAME shape as x.
    """
    x_shape = x.shape
    F = x.shape[feature_axis]

    mean_t = _to_tensor_like(x, mean)
    std_t  = _to_tensor_like(x, std)

    # 1) Global scalar stats -> trivial, preserves shape
    if mean_t.ndim == 0 and std_t.ndim == 0:
        return x * std_t + mean_t

    # 2) Non-CI or per-feature broadcast: mean/std are 1D of length F
    if mean_t.ndim == 1 and std_t.ndim == 1 and mean_t.numel() == F and std_t.numel() == F:
        view_shape = [1] * x.ndim
        view_shape[feature_axis] = F
        mean_b = mean_t.view(*view_shape)
        std_b  = std_t.view(*view_shape)
        out = x * std_b + mean_b
        if strict and out.shape != x_shape:
            raise RuntimeError(f"inv_standardize: shape changed {x_shape} -> {out.shape} in non-CI branch.")
        return out

    # 3) CI (feature dim == 1) with per-channel stats: gather by ch along sample_axis
    if F == 1 and mean_t.ndim == 1 and std_t.ndim == 1 and mean_t.numel() > 1 and std_t.numel() > 1:
        if ch is None:
            raise ValueError("inv_standardize: CI tensor with feature size 1 but mean/std are vectors; pass `ch`.")
        if ch.ndim != 1:
            raise ValueError("inv_standardize: `ch` must be 1-D [N].")
        if ch.shape[0] != x.shape[sample_axis]:
            raise ValueError(f"inv_standardize: len(ch)={ch.shape[0]} must equal x.shape[sample_axis]={x.shape[sample_axis]}.")

        m = mean_t[ch]  # [N]
        s = std_t[ch]   # [N]
 
        expand = [1] * x.ndim # reshape m,s to broadcast over all non-sample, non-feature dims
        expand[sample_axis] = m.shape[0]   # N
        
        m = m.view(*expand) # feature_axis already 1; no need to set expand[feature_axis]
        s = s.view(*expand)

        out = x * s + m
        if strict and out.shape != x_shape:
            raise RuntimeError(f"inv_standardize: shape changed {x_shape} -> {out.shape} in CI branch.")
        return out

    # 4) Fallback: try safe broadcast ONLY if it preserves shape (rare)
    mean_b, std_b = mean_t, std_t
    while mean_b.dim() < x.dim():
        mean_b = mean_b.unsqueeze(0)
        std_b  = std_b.unsqueeze(0)
    out = x * std_b + mean_b
    if strict and out.shape != x_shape:
        raise RuntimeError(
            f"inv_standardize: would change shape {x_shape} -> {out.shape}. "
            f"Check `feature_axis`, `ch`, and stats shapes (mean={tuple(mean_t.shape)}, std={tuple(std_t.shape)})."
        )
    return out