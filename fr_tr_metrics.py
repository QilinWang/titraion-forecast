from typing import Optional, Literal

import torch
import torch.nn as nn

from typing import Callable, List, Dict, Union, Any
import numpy as np
from pydantic import (
    BaseModel,  Field, ConfigDict, config, model_validator, field_validator, computed_field,
)
import study.fr_cfg as configs 
from collections import deque
from statistics import median
from colored import fg, attr


 
def cprint(text, color=None, bold=False):
    parts = []
    if color:
        parts.append(fg(color))
    if bold:
        parts.append(attr("bold"))
    parts.append(text)
    parts.append(attr("reset"))
    return "".join(parts)

def _to_float(x) -> float:
    if isinstance(x, torch.Tensor):
        x = x.item()
    return float(x)

def _alpha_from_half_life(h: int) -> float:
    # (1 - α)^h = 0.5  =>  α = 1 - 0.5**(1/h)
    h = max(1, int(h))
    return 1.0 - 0.5 ** (1.0 / h)



class MetricRegistry:
    """
    intent: map metric-name -> builder(spec, **kwargs) → loss fn (Callable[T,T]->T)
    assumptions: builders are pure w.r.t. inputs (cfg/device can be passed via kwargs); names are unique.
    returns: registry utilities; supports decorator registration and table dispatch.
    """
    _builders: Dict[str, Callable[..., Any]] = {}

    # === API ===
    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """intent: decorator to register a builder under a unique name"""
        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            # CHG: overwrite is allowed only intentionally; comment if you want strict
            cls._builders[name] = fn
            return fn
        return _wrap

    @classmethod
    def build(cls, name: str, spec: configs.MetricSpec, **kwargs: Any) -> Callable:
        """
        intent: build a metric function from (name, spec) using registered builders
        assumptions: 'quantileXX' routed to 'quantile' builder; spec.params carry options
        returns: a callable loss fn
        """
        # direct hit
        if name in cls._builders:
            return cls._builders[name](spec, **kwargs)

        # pattern route: quantile70 -> 'quantile'
        if name.startswith("quantile") and "quantile" in cls._builders:
            return cls._builders["quantile"](spec, **kwargs)

        # GUARD-OK: explicit config error
        raise ValueError(f"Unknown metric: {name}")

# === Step 2: Register default builders ===

@MetricRegistry.register("mse")
def build_mse(spec: configs.MetricSpec, **kwargs: Any) -> nn.Module:
    """intent: standard MSE"""
    return nn.MSELoss()

@MetricRegistry.register("mae")
def build_mae(spec: configs.MetricSpec, **kwargs: Any) -> nn.Module:
    """intent: standard L1"""
    return nn.L1Loss()

@MetricRegistry.register("huber")
def build_huber(spec: configs.MetricSpec, **kwargs: Any) -> nn.Module:
    """intent: Huber(delta) from spec.params['delta']"""
    delta = float(spec.params.get("delta", 1.0))
    return nn.HuberLoss(delta=delta)

@MetricRegistry.register("quantile")
def build_pinball(spec: configs.MetricSpec, **kwargs: Any) -> nn.Module:
    """intent: Pinball(quantile) from spec.params['quantile']"""
    q = float(spec.params["quantile"])  # REQUIRE: caller provides it in spec
    return PinballLoss(quantile=q)

@MetricRegistry.register("ept")
def build_ept(spec: configs.MetricSpec, **kwargs: Any) -> nn.Module:
    """intent: EPT uses global_std + device passed via kwargs"""
    global_std = kwargs["global_std"]         # REQUIRE: provided by caller
    device = kwargs["device"]                 # REQUIRE: provided by caller
    return EPTMetric(global_std=global_std, device=device)

@MetricRegistry.register("swd")
def build_swd(spec: configs.MetricSpec, **kwargs: Any) -> nn.Module:
    """intent: SWD from cfg-dependent dims and spec params"""
    cfg = kwargs["cfg"]                       # REQUIRE: cfg supplies channels/CI flags
    seed = int(kwargs["seed"])
    use_proj = bool(kwargs.get("use_proj_swd", False))
    feature_dim = 1 if cfg.channel_independent else cfg.channels
    return SWDMetric(
        feature_dim=feature_dim,
        num_proj=cfg.num_proj_swd,
        seed=seed,
        feature_axis=2,
        point_axis=1,
        use_proj=use_proj,
        return_type=spec.params.get("return_type", "sq_swd2"),
    ).to(cfg.device)
 
class MetricRecorder(BaseModel):
    """Tracks a single metric's statistics at batch level""" 
    model_config = ConfigDict( arbitrary_types_allowed=True,  validate_assignment=False, extra="forbid",)

    name: str
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    weight_backward: float = Field(default=0.0, ge=0.0)
    weight_validate: float = Field(default=0.0, ge=0.0)

    total: float = Field(default=0.0, ge=0.0)
    count: int = Field(default=0, ge=0)

    @classmethod
    def from_spec(cls, name: str, spec: configs.MetricSpec, **kwargs: Any) -> "MetricRecorder":
        """intent: build tracker from registry and spec; returns MetricRecorder"""
        fn = MetricRegistry.build(name, spec, **kwargs)
        return cls(
            name=name, fn=fn,
            weight_backward=float(spec.weight_backward),
            weight_validate=float(spec.weight_validate),
        )

    def acc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raw = self.fn(pred, target)
        v = raw.item()
        if np.isfinite(v):
            self.total += v
            self.count += 1
        else:
            print(f"[{self.name}] warning: non-finite value {v} skipped")
        return raw # keep tensor for backprop when in train

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0
  
    def __str__(self) -> str:
        return f"MetricMeter(mean={self.mean:.4f}, count={self.count})"
    
      
class ModeMetrics(BaseModel):
    """Manages all metric calculations for a single mode (e.g., 'train').""" 
    model_config = ConfigDict( arbitrary_types_allowed=True, validate_assignment=False, extra="forbid", )

    # --- INPUTS ---
    mode: Literal["train", "val", "test"]
    cfg: configs.BaseTrainingConfig
    global_std: np.ndarray 
    seed: int 
    use_proj_swd: bool = False  
 
    # --- State ---
    metric_recorders: Dict[str, MetricRecorder] = Field(default_factory=dict) 

    # --- Counters --- 
    running_obj_total: float = 0.0
    running_obj_count: int = 0

    @classmethod 
    def setup_phase_metric_recorders(
        cls,
        mode: Literal["train", "val", "test"], 
        cfg: configs.BaseTrainingConfig,
        global_std: np.ndarray,
        seed: int,
        use_proj_swd: bool, 
    ) -> "ModeMetrics":
        """Build a ModeMetrics with metric_recorders initialized from cfg."""
        metric_recorders: Dict[str, "MetricRecorder"] = {}
 
        for name, metric_spec in cfg.metric_configs.items():
            rec = MetricRecorder.from_spec(
            name, metric_spec,
            cfg=cfg,
            seed=seed,
            use_proj_swd=use_proj_swd,
            global_std=global_std,
            device=cfg.device,
            )
            metric_recorders[name] = rec
        return cls(
            mode=mode, 
            cfg=cfg,
            global_std=global_std,
            seed=seed,
            use_proj_swd=use_proj_swd, 
            metric_recorders=metric_recorders,
        )  
     
    @property
    def running_obj_mean(self) -> float:
        return (  self.running_obj_total / max(self.running_obj_count, 1)) 

    @model_validator(mode="after")
    def setup_metric_recorders(self) -> "ModeMetrics":  
        for name, metric_spec in self.cfg.metric_configs.items():
            metric_recorder = MetricRecorder.from_spec(
            name, metric_spec,
            cfg=self.cfg,
            seed=self.seed,
            use_proj_swd=self.use_proj_swd,
            global_std=self.global_std,
            device=self.cfg.device,
            )
            self.metric_recorders[name] = metric_recorder
        return self

    def update_and_calc_objective(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  
        total = torch.zeros((), device=pred.device, requires_grad=True if self.mode in ["train"] else False)  # seed
        
        for tracker in self.metric_recorders.values():
            raw = tracker.acc_loss(pred, targets)
            weight = ( tracker.weight_backward if self.mode in ["train"] else tracker.weight_validate )
            if weight > 0:
                total = total + weight * raw 

        self.running_obj_total += float(total.item())
        self.running_obj_count += 1
        return total
        

    def collect_metrics(self) -> "MetricsReport":
        # build a name->value dict from metric_recorders (exactly what you already do)
        metrics_data = {name: tracker.mean for name,tracker in self.metric_recorders.items()} 

        # hand the dict to MetricsReport; computed-field lenses (mse/mae/...) will read from it
        metrics_report = MetricsReport(
            mode=self.mode,
            metrics=metrics_data,         # <--- key change (no ** expansion)
            running_obj=self.running_obj_mean, 
        )
        return metrics_report

    def reset(self):
        for tracker in self.metric_recorders.values():
            tracker.reset()
        self.running_obj_total = 0.0
        self.running_obj_count = 0


def half_life_to_alpha(h: int) -> float:
    # (1 - alpha)^h = 0.5  -> alpha = 1 - 0.5**(1/h)
    return 1.0 - 0.5 ** (1.0 / max(h, 1))


class MetricManager(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # CUDA tensors, generators, …
        validate_assignment=False,  # mutate without re-validation
    ) 
    # --- INPUTS ---
    cfg: configs.BaseTrainingConfig
    global_std: np.ndarray
    seed: int
    use_proj_swd: bool = False

    # --- STATE (Initialized by model_validator) ---
    train: ModeMetrics | None = None
    val: ModeMetrics | None = None
    test: ModeMetrics | None = None
    
    # last committed reports (per epoch) 
    last_train: Optional["MetricsReport"] = None
    last_val:   Optional["MetricsReport"] = None
    last_test:  Optional["MetricsReport"] = None

    # best metrics (across epochs)
    best_val_metrics: Optional["MetricsReport"] = None
    best_test_metrics: Optional["MetricsReport"] = None
    
    # state (across epochs)
    val_obj_hist: deque = Field(default_factory=lambda: deque(maxlen=3)) 
    val_obj_smooth: Optional[float] = None  # last smoothed scalar
    
    # smoothing policy (tweak as you like)
    smooth_kind: Literal["none", "sma", "ema", "median"] = "none"
    smooth_window: int = 3         # window for SMA/median
    smooth_half_life: Optional[int] = None  # if using EMA
    smooth_alpha: Optional[float] = None    # override EMA alpha
    
    @classmethod
    def setup_metric_manager(cls, cfg, global_std, seed, use_proj_swd) -> "MetricManager":
        train = ModeMetrics.setup_phase_metric_recorders(
            mode="train", cfg=cfg, global_std=global_std, seed=seed, use_proj_swd=use_proj_swd
        )
        val = ModeMetrics.setup_phase_metric_recorders(
            mode="val", cfg=cfg, global_std=global_std, seed=seed, use_proj_swd=use_proj_swd
        )
        test = ModeMetrics.setup_phase_metric_recorders(
            mode="test", cfg=cfg, global_std=global_std, seed=seed, use_proj_swd=use_proj_swd
        )
        return cls(
            cfg=cfg,
            global_std=global_std,
            seed=seed,
            use_proj_swd=use_proj_swd,
            train=train,
            val=val,
            test=test,
        )
    
    # --- update reports from per epoch container to cross epoch container --- 
    def commit_train(self, report: "MetricsReport") -> None:
        self.last_train = report

    def commit_val(self, report: "MetricsReport") -> float:
        self.last_val = report
        # one-scalar smoothing (SMA/EMA/median) over report.running_obj
        return self.compute_smoothed_val_objective(report)   # returns the smoothed scalar

    def commit_test(self, report: "MetricsReport") -> None:
        self.last_test = report
    
    # --- smoothing ---
    def _ema_alpha(self) -> float:
        if self.smooth_alpha is not None:
            return self.smooth_alpha
        if self.smooth_half_life:
            return 1.0 - 0.5 ** (1.0 / max(self.smooth_half_life, 1))
        # default: relate to window
        return 1.0 - 0.5 ** (1.0 / max(self.smooth_window, 1))
    
    def compute_smoothed_val_objective(self, report: "MetricsReport") -> float:
        """Update and return the smoothed validation objective from a MetricsReport."""
        x = float(report.running_obj)  # assume it is set for val

        kind = self.smooth_kind
        if kind == "sma": 
            self.val_obj_hist.append(x)
            self.val_obj_smooth = sum(self.val_obj_hist) / len(self.val_obj_hist)
        elif kind == "median": 
            self.val_obj_hist.append(x)
            self.val_obj_smooth = float(median(self.val_obj_hist))
        elif kind == "none":
            self.val_obj_smooth = x
        else:  # "ema"
            a = self._ema_alpha()
            prev = self.val_obj_smooth if self.val_obj_smooth is not None else x
            self.val_obj_smooth = a * x + (1.0 - a) * prev 
        return self.val_obj_smooth
    
    # --- reset ---
    def reset_val_smoothing(self):
        self.val_obj_hist.clear()  ; self.val_obj_smooth = None
        
    def reset_all(self):
        self.train.reset() ; self.val.reset() ; self.test.reset()
        
    # --- ergonomics ---
    def _empty(self, mode: str) -> "MetricsReport":
        return MetricsReport(mode=mode, metrics={}, running_obj=None)

    @property
    def train_report(self) -> "MetricsReport":
        return self.last_train or self._empty("train")

    @property
    def val_report(self) -> "MetricsReport":
        return self.last_val or self._empty("val")

    @property
    def test_report(self) -> "MetricsReport":
        return self.last_test or self._empty("test")


class MetricsReport(BaseModel):
    """A structured container for the results of one epoch for one mode."""
    mode: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    running_obj: float
    display_digits: int = 3
    display_digits_aux: int = 3
    
    # Back-compat lenses so val.mse etc. still work
    @computed_field(return_type=float)
    def mse(self) -> float:   return float(self.metrics.get("mse", 0.0))
    @computed_field(return_type=float)
    def mae(self) -> float:   return float(self.metrics.get("mae", 0.0))
    @computed_field(return_type=float)
    def huber(self) -> float: return float(self.metrics.get("huber", 0.0))
    @computed_field(return_type=float)
    def swd(self) -> float:   return float(self.metrics.get("swd", 0.0))
    @computed_field(return_type=float)
    def ept(self) -> float:   return float(self.metrics.get("ept", 0.0))
    @computed_field(return_type=float)
    def quantile70(self) -> float: return float(self.metrics.get("quantile70", 0.0))
    @computed_field(return_type=float)
    def quantile30(self) -> float: return float(self.metrics.get("quantile30", 0.0))

    def __str__(self) -> str:
        parts = [f"{cprint(self.mode.upper(), color='cyan', bold=True)}"]
        metric_order = ["mse", "mae", "huber", "swd", "ept", 
                        "q70", "q30", "obj"]
        name_map = {
            "mse": "MSE", "mae": "MAE", "huber": "HUB", "swd": "SWD", "ept": "EPT",
            "quantile70": "Q70", "quantile30": "Q30",
            "running_obj": "OBJ"
        }
        values = self.model_dump()
        for key, display_name in name_map.items():
            value = values.get(key)
            if value is None: continue

            if key == "ept":
                val_str = f"{value:.0f}"
            elif "quantile" in key or key == "running_obj":
                val_str = f"{value:.{self.display_digits_aux}f}"
            else:
                val_str = f"{value:.{self.display_digits}f}"
                
            parts.append(f"{display_name}={cprint(val_str, 
                    color='light_magenta', bold=True)}")
 
        # mae_str = f"{self.mae:.{self.display_digits}f}"
        #     f"MAE={cprint(mae_str, color='light_magenta', bold=True)}", 
        return " | ".join(parts)
 

# ---------------------------
# Projections
# ---------------------------
def rand_proj(
    input_dim: int, num_proj: int, seed: int, requires_grad: bool = False
) -> torch.Tensor:
    """Generate a (input_dim x num_proj) matrix whose columns are unit-length projection directions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    proj = torch.randn(
        input_dim, num_proj, device=device, requires_grad=requires_grad, generator=g
    )
    proj = proj / (
        proj.norm(dim=0, keepdim=True) + 1e-9
    )  # normalize columns (unit vectors)
    return proj  # (F, L)


# ---------------------------
# Core SWD slice + sort
# ---------------------------
def _project_along_axis(
    x: torch.Tensor, proj_mat: torch.Tensor, feature_axis: int
) -> torch.Tensor:
    """
    Project x along 'feature_axis' with proj_mat (F x L), returning a tensor where that axis is replaced by L.
    Works by moving 'feature_axis' to the last dim, matmul, then moving L back to that position.
    Intuitively,
    Always ask yourself: "What is the set of things I'm trying to compare?"
    The answer defines your "point cloud."
    The axis you sort over represents the different points in the cloud.
    The axis you destroy (project) represents the dimensions/features of each individual point.
    """
    # Bring 'feature_axis' to last
    x_feat_last = torch.moveaxis(x, feature_axis, -1)  # (..., F)
    z = x_feat_last @ proj_mat  # (..., L)
    # Put L back where 'feature_axis' was
    z = torch.moveaxis(z, -1, feature_axis)  # (..., L, ...)
    return z


class SWDMetric(nn.Module):
    """
    Sliced Wasserstein with axis control.

    Args:
      feature_dim:  size of the axis to project (F)
      num_proj:     number of projection directions (L)
      seed:         RNG seed for projections
      feature_axis: which axis in (y_pred, y_real) is the feature space to project (e.g., D)
      point_axis:   which axis indexes the set of points to sort (e.g., S, R, or B)
      use_proj: if False, use identity projections (no cross-feature mixing);
                    if True, use random unit directions in feature space
      return_type:  "sq_swd2" (mean squared), "swd2" (sqrt of mean squared), or "swd1" (mean abs)

    Typical choices:
      * Horizon-shape (per-sample): feature_axis=D, point_axis=S
      * Dataset cloud:               feature_axis=(S*D) after flatten, point_axis=B
      * Multi-realization per-x:     feature_axis=(S*D) or D, point_axis=R
    """

    def __init__(
        self,
        feature_dim: int,
        num_proj: int,
        seed: int,
        feature_axis: int,  # which axis in input tensors is the feature space to project
        point_axis: int,  # which axis in input tensors is the set of points to sort
        use_proj: bool = False,
        return_type: Literal["sq_swd2", "swd2", "swd1"] = "sq_swd2",
    ):
        super().__init__()
        self.train_seed = seed
        self.eval_seed = seed
        self.num_proj = num_proj
        self.feature_axis = feature_axis
        self.point_axis = point_axis
        self.return_type = return_type
        self.use_proj = use_proj
        self.feature_dim = feature_dim
        
        self._printed_shapes = True

        if use_proj:
            proj_mat = rand_proj( feature_dim, num_proj, seed=self.eval_seed, requires_grad=False )
        else:
            proj_mat = torch.eye(feature_dim, feature_dim)
        self.register_buffer("proj_mat", proj_mat)

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        """
        y_pred, y_real: same shape tensors (e.g., (B, S, D) etc.).
        We project along 'feature_axis' and sort along 'point_axis'.
        """
        assert y_pred.shape == y_real.shape, "Shapes of y_pred and y_real must match."
        # optionally resample during training
        # if self._printed_shapes:
        #     print("y shape sample:", {y_real.shape}, "feature_axis=", {self.feature_axis})
            
        if self.training and self.use_proj:
            self.train_seed += 1
            newP = rand_proj(
                self.feature_dim,
                self.num_proj,
                seed=self.train_seed,
                requires_grad=False,
            )
            self.proj_mat.copy_(newP)

        # 1) project along 'feature_axis'
        z_pred = _project_along_axis(
            y_pred, self.proj_mat, self.feature_axis
        )  # (..., L, ...)
        z_real = _project_along_axis(
            y_real, self.proj_mat, self.feature_axis
        )  # (..., L, ...)

        """
        Generic sliced 1-D OT:
        1) project along 'feature_axis' with proj_mat (F x L),
        2) sort along 'point_axis' (the set index),
        3) return sorted difference (pred - real).
        """
        # After projection, 'feature_axis' has been replaced by L (or F in identity mode).
        proj_axis = self.feature_axis % z_pred.ndim
        point_axis = self.point_axis % z_pred.ndim
        # guard: don't sort along the projections axis
        assert point_axis != proj_axis, (
            "point_axis must index the set of points, not the projections."
        )

        # 2) sort along the 'point_axis' (quantile matching)
        z_pred_sorted, _ = torch.sort(z_pred, dim=point_axis)
        z_real_sorted, _ = torch.sort(z_real, dim=point_axis)
        diff = z_pred_sorted - z_real_sorted

        # 3) reduce: mean over points, over projections, then over remaining batch-like axes
        # adjust projection axis index if point_axis < proj_axis (sorting removes/permutes nothing but keep indices aligned)
        if self.return_type == "sq_swd2":
            val = diff.pow(2).mean(dim=point_axis)
            swd = val.mean()
        elif self.return_type == "swd2":
            val = diff.pow(2).mean(dim=point_axis).sqrt()
            swd = val.mean()
        elif self.return_type == "swd1":
            val = diff.abs().mean(dim=point_axis)
            swd = val.mean()
        else:
            raise ValueError(f"Invalid return_type: {self.return_type}")

        return swd


"""
Imagine diff has axes [ ... point_axis ..., ... proj_axis ... ].
First, you reduce (.mean) over point_axis.
Now every axis that comes after point_axis in the order shifts left by one.
If proj_axis was to the right of point_axis, its index is now off by -1.
So proj_axis - (proj_axis > point_axis) is just a quick way to fix the index.

Example:

Suppose diff.shape = (B, S, L) with axes (0=B, 1=S=point_axis, 2=L=proj_axis).
You reduce over axis=1 (S). New shape (B, L).
Now the projections axis is at index 1, not 2.
proj_axis=2, point_axis=1 ⇒ proj_axis > point_axis = True ⇒ 2-1=1. Perfect.
"""
# usage
# Inputs y_* shaped (B, S, D). Project channels (feature_axis=2), sort horizon (point_axis=1):
# metric = SWDMetric(feature_dim=D, num_proj=1500, seed=0, feature_axis=2, point_axis=1, return_type="sq_swd2")

# Univariate (D=1), exact 1D OT (no need for many projections)
# metric = SWDMetric(feature_dim=1, num_proj=1, seed=0, feature_axis=2, point_axis=1, return_type="sq_swd2")
# score = metric(y_pred, y_true)


class TargetStdMetric(nn.Module):
    """Compute the standard deviation of target data."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        # y_real shape: [batch_size, seq_len, channels] or [batch_size, channels, seq_len]
        return torch.std(y_real, dim=-1).mean()  # Average across batch and channels


class EPTMetric(nn.Module):
    """
    Effective Prediction Time:
        • global_std: 1D tensor [D]  (per-channel threshold)
        • y_pred, y_true: [B, S, D]
    Returns a scalar: mean T_{b,d} over all sequences and channels.
    """

    def __init__(self, global_std: np.ndarray, device: torch.device):
        super().__init__()
        # shape to [1, D, 1] so it broadcasts against [B, D, S]
        # print(f"EPT global_std shape: {global_std.shape}")
        self.register_buffer(
            "thr",
            torch.tensor(global_std, dtype=torch.float32, device=device)[None, :, None],
        )

    def forward(self, y_pred: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.permute(0, 2, 1)
        y_real = y_real.permute(0, 2, 1)
        # print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_real.shape}, thr shape: {self.thr.shape}")
        # print(f"self.thr: {self.thr}")
        err = (y_pred - y_real).abs()  # [B,D,S]
        crossed = err > self.thr  # [B,D,S] bool
        # first index along S where crossed is True; if never, returns 0
        first = (crossed.float()).argmax(-1)  # [B,D] int64
        # first index along S where value==1 (PyTorch argmax returns first max)
        never = ~crossed.any(-1)  # [B,D] bool
        # If any time‑step is True, it returns True; otherwise False. → [B,D].
        first[never] = y_pred.size(-1)  # set to S when never crossed
        return first.float().mean()  # scalar like MSE/MAE
 

class PinballLoss(nn.Module):
    r"""
    Quantile (pinball) loss for data shaped (B, S, D).

    Canonical definition (always ≥ 0):
        Let u = y_true - y_pred and τ ∈ (0,1).
        ρ_τ(u) = max( τ·u, (τ-1)·u )

    Equivalent piecewise with error = y_pred - y_true:
        if error ≥ 0 (over-predict):    (1-τ)·error
        else (under-predict):           -τ·error

    Args:
        quantile: τ in (0, 1)
        reduction: "mean" | "sum" | "none"
    """

    def __init__(self, quantile: float, reduction: Literal["mean", "sum", "none"] = "mean"):
        super().__init__()
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile τ must be in (0, 1).")
        self.register_buffer(
            "q", torch.tensor(quantile, dtype=torch.float32), persistent=False
        )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.shape != y_true.shape or y_pred.ndim != 3:
            raise ValueError(
                f"y_pred and y_true must both be (B, S, D); got {y_pred.shape=} {y_true.shape=}"
            )
        q = self.q.to(device=y_pred.device, dtype=y_pred.dtype)

        # canonical form (diff = y_true - y_pred)
        diff = y_true - y_pred  # (B, S, D)
        loss = torch.maximum(q * diff, (q - 1) * diff)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # (B, S, D)


class CRPSApprox(nn.Module):
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32).view(1, -1, 1, 1)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred: [B, num_quantiles, S, D]  predicted quantiles
        y_true: [B, S, D]                 true values
        """
        y_true = y_true.unsqueeze(1)  # [B,1,S,D]
        error = y_true - y_pred  # note: consistent sign with pinball def
        loss = torch.maximum(self.quantiles * error, (self.quantiles - 1) * error)
        return loss.mean()  # scalar CRPS estimate
