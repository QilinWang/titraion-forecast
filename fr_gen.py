from typing import Literal, List, Tuple, Optional, Iterable, Dict, Callable, Sequence, TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops 
from pydantic import ConfigDict, model_validator, computed_field
from pydantic import BaseModel, Field 
import study.fr_cfg as configs

import study.fr_core as fcore 

# ------------------------------------------------------------
# region Op
class Op(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # <--- Add this line
        arbitrary_types_allowed=True,  # accept tensors
        json_encoders={  # compact dumps
            torch.Tensor: lambda t: f"Tensor{tuple(t.shape)}"
        },
    )
    name: str = "Op"
 
    def __call__(self, x):
        return self.apply(x)

    def __ror__(self, left):
        return self._pipe(left, self)

    def __or__(self, right):
        return self._pipe(self, right)
 
    def apply(self, x):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError
 
    def _summary(self, k=0):
        return f"{'  ' * k}- {self.name}\n"

    def __str__(self): 
        return self._summary()
 
    @staticmethod
    def _pipe(a, b): # helper
        if isinstance(a, torch.Tensor) and isinstance(b, Op):
            return b.apply(a)
        if isinstance(a, Op) and isinstance(b, torch.Tensor):
            return a.apply(b)
        if isinstance(a, Op) and isinstance(b, Op):
            return Chain([a, b])
        print(f"type(a): {type(a)}, type(b): {type(b)}")
        raise TypeError

    @property
    def inv(self) -> "InverseOp":
        return InverseOp(base=self)

    def __invert__(self) -> "InverseOp":
        """enables ~op so  x | self.shift.inv = x | ~self.shift"""
        return self.inv


class Chain(Op):
    """constructor flattens nested composites. 
    Then (~Chain) works and is equivalent to chaining each part's inverse in reverse order.
    """
    name: str = "Chain"
    parts: list[Op] = Field(default_factory=list)
 
    def __init__(self, parts: Iterable[Op]):
        flat = []
        for p in parts:
            flat.extend(p.parts) if isinstance(p, Chain) else flat.append(p)
        super().__init__(parts=flat)
 
    def __iter__(self):
        return iter(self.parts)

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, i):
        return self.parts[i]
 
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.parts:
            x = op.apply(x)
        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for op in reversed(self.parts):
            x = op.inverse(x)
        return x

    @property
    def inv(self): 
        return Chain([p.inv for p in reversed(self.parts)])
 
    def _summary(self, k=0):
        inner = "".join(p._summary(k + 1) for p in self.parts)
        return f"{'  ' * k}* {self.name}:\n{inner}"


class InverseOp(Op):
    base: Op

    @property
    def name(self):  # override for nicer print
        return f"{self.base.name}^-1"

    def apply(self, x):
        return self.base.inverse(x)

    def inverse(self, x):
        # inverse of the inverse is the forward
        return self.base.apply(x)

    def _summary(self, k=0):
        return f"{'  ' * k}- {self.name}\n" 
# endregion Op

"""
In this section, we define a list of convient wrapper of tensors 
that allows you to 'mutate' tensors with coefficients 
like x | scale | shift | ... or x | rotation | ... 
This is the 'micro use' of Op and Chain DSL.
It allows us to reuse the same API in G and T (where we send States 
to States), the 'macro use'.
"""
# region Scale
def apply_block_diagonal(x, scale):
    """
    Apply a block-diagonal transformation with 2×2 blocks to a vector.
    # [a  b] [x_even]
    # [-b a] [x_odd ]

    Args:
        x: Input tensor of shape [..., dim]
        a: Scaling parameters of shape [..., dim//2]
        b: Rotation parameters of shape [..., dim//2]

    Returns:
        Transformed tensor of same shape as x
    """
    dim = x.shape[-1]
    assert dim % 2 == 0, "Dimension must be even for 2×2 blocks"
    # For example, if x has shape [32, 128], it becomes [32, 64, 2], 64 is the number of 2×2 blocks, 2 is the number of elements per block
    x_reshaped = einops.rearrange(
        x, "b d (sn e) -> b d sn e", sn=dim // 2, e=2
    )  # Reshape tensor to group adjacent pairs [..., dim//2, 2]
    x_even = x_reshaped[..., 0]  # [..., dim//2]
    x_odd = x_reshaped[..., 1]  # [..., dim//2]
    scale_reshaped = einops.rearrange(scale, "b d (sn e) -> b d sn e", sn=dim // 2, e=2)
    scale_even = scale_reshaped[..., 0]
    scale_odd = scale_reshaped[..., 1]
    y_even = scale_even * x_even - scale_odd * x_odd  # [..., dim//2]
    y_odd = scale_odd * x_even + scale_even * x_odd  # [..., dim//2]

    return einops.rearrange(torch.stack([y_even, y_odd], dim=-1), "b d sn e -> b d (sn e)")


def apply_tridiagonal(
    x: torch.Tensor,
    diag_lower: torch.Tensor,
    diag_main: torch.Tensor,
    diag_upper: torch.Tensor,
) -> torch.Tensor:
    """
    Applies a tridiagonal matrix transformation without materializing the matrix.

    This function computes `y = T @ x` where T is a tridiagonal matrix defined
    by the three diagonal coefficient tensors. It is highly efficient as it
    only uses slicing, padding, and element-wise multiplication.

    The operation for each element `y[i]` is:
    y[i] = diag_lower[i-1]*x[i-1] + diag_main[i]*x[i] + diag_upper[i]*x[i+1]

    Args:
        x (torch.Tensor): The input tensor of shape (..., S).
        diag_lower (torch.Tensor): The lower diagonal coefficients, shape (..., S-1).
        diag_main (torch.Tensor): The main diagonal coefficients, shape (..., S).
        diag_upper (torch.Tensor): The upper diagonal coefficients, shape (..., S-1).

    Returns:
        torch.Tensor: The transformed tensor `y` of the same shape as `x`.
        
    # 1. Main diagonal term (no shift needed) y_i += main_i * x_i
    # 2. Lower diagonal term (applies to x_{i-1}) y_i += lower_{i-1} * x_{i-1}
    # 3. Upper diagonal term (applies to x_{i+1}) y_i += upper_i * x_{i+1}
    """
    S = x.shape[-1]
    if diag_main.shape[-1] != S:
        raise ValueError(f"Shape mismatch: x last dim is {S} but diag_main is {diag_main.shape[-1]}")
    if diag_lower.shape[-1] != S - 1 or diag_upper.shape[-1] != S - 1:
        raise ValueError(f"Shape mismatch: Off-diagonals must have length {S - 1}")
 
    main_term = diag_main * x 
    
    x_shifted_right = x[..., :-1]  # x_0, x_1, ..., x_{S-2}
    lower_prod = diag_lower * x_shifted_right 
    lower_term = F.pad(lower_prod, (1, 0)) # Pad on the left to align with y_1, y_2, ...

    
    x_shifted_left = x[..., 1:]  # x_1, x_2, ..., x_{S-1}
    upper_prod = diag_upper * x_shifted_left 
    upper_term = F.pad(upper_prod, (0, 1))  # Pad on the right to align with y_0, y_1, ...
 
    y = main_term + lower_term + upper_term 
    return y
# region softclip functions


# ===== PostProcessConfig with semantic names =====
class PostProcessConfig(BaseModel):
    post_process: Literal[
        "relu6", "softplus", "sigmoid", "tanh", "hardtanh", "none",
        "relu6_softcap_leaky", "softclip_hinge", "softclip_leaky",
        "bounded_tanh", "positive_log_exp",
    ] = "none"  
    low: float = 0.0
    high: float = 6.0
    factor: float = 1.0
    softplus_sharpness: float = 1.0  # β: higher = sharper knee
    leak: float = 0.05               # Leaky identity fraction
    tanh_temp: float | None = None   # Temperature for bounded_tanh
    logexp_max_log: float = 3.0  # log-exp max cap on log-range; exp in (e^-max_log, e^max_log)
    
    """Configuration for eigenvalue-based sparsification."""
    use_eigen_sparse: bool = False
    top_k_gaps: int = 5          # Number of largest gaps to consider
    min_keep: int = 1            # Minimum eigenvalues to keep per patch
    patch_spec: fcore.PatchSpec = Field(..., description="Patch specification for sparsification")


# ===== Registry =====
_POST_PROCESSORS: Dict[str, Callable[[torch.Tensor, PostProcessConfig], torch.Tensor]] = {}

def register_postproc(name: str):
    def decorator(fn):
        _POST_PROCESSORS[name] = fn
        return fn
    return decorator


# ===== Simple ones (factor scaling only) =====
def _simple(transform):
    def proc(s, cfg):
        return transform(s) * cfg.factor
    return proc

register_postproc("relu6")(_simple(F.relu6)) 
register_postproc("sigmoid")(_simple(torch.sigmoid))
register_postproc("tanh")(_simple(torch.tanh))


# ===== Complex ones with parameters =====
@register_postproc("softplus")
def _softplus(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    return F.softplus(s, beta=cfg.softplus_sharpness) * cfg.factor

@register_postproc("hardtanh")
def _hardtanh(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    """Hard clamp: ℝ → [low, high] with zero gradient outside."""
    return F.hardtanh(s, min_val=cfg.low, max_val=cfg.high) * cfg.factor


@register_postproc("softclip_hinge")
def _softclip_hinge(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    """
    Smooth clamp to [low, high] with differentiable knee.
    Inside: grad≈1 (identity-like). Outside: grad→0 (clamp-like).
    """
    return (
        s 
        - F.softplus(s - cfg.high, beta=cfg.softplus_sharpness) 
        + F.softplus(cfg.low - s, beta=cfg.softplus_sharpness)
    ) * cfg.factor


@register_postproc("softclip_leaky")
def _softclip_leaky(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    """
    Soft clamp with leaky identity: (1-α)·softclip + α·s.
    Inside: grad≈1. Outside: grad≈leak (nonzero for stability).
    """
    clamped = (
        s 
        - F.softplus(s - cfg.high, beta=cfg.softplus_sharpness)
        + F.softplus(cfg.low - s, beta=cfg.softplus_sharpness)
    )
    return (cfg.leak * s + (1.0 - cfg.leak) * clamped) * cfg.factor


@register_postproc("relu6_softcap_leaky")
def _relu6_softcap_leaky(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    """
    Positive soft-cap: ℝ → [0, high] with leaky passthrough.
    • Softplus(s) ensures positive
    • Soft-cap at 'high' with smooth knee
    • Leak allows small gradient beyond cap
    Formula: (1-α)·cap(softplus(s)) + α·softplus(s)
    """
    x_pos = F.softplus(s)  # Ensure positive
    capped = x_pos - F.softplus(x_pos - cfg.high, beta=cfg.softplus_sharpness)
    return (capped + cfg.leak * (x_pos - capped)) * cfg.factor


@register_postproc("bounded_tanh")
def _bounded_tanh(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    """
    Map ℝ → [low, high] via tanh with optional temperature scaling.
    
    Formula: midpoint + radius·tanh(s/τ)
    • midpoint = (low + high)/2  ← Output center
    • radius   = (high - low)/2  ← Output range
    • τ (tau)  = temperature     ← Controls sharpness (lower = sharper)
    
    Examples:
    • s=0  → midpoint (always)
    • s→∞  → high
    • s→-∞ → low
    • τ=1  → standard tanh
    • τ→0  → approaches step function
    • τ→∞  → approaches linear
    """
    midpoint = 0.5 * (cfg.low + cfg.high)
    radius = 0.5 * (cfg.high - cfg.low)
    
    # Optional temperature scaling for controlling transition sharpness
    x_scaled = s if cfg.tanh_temp is None else s / cfg.tanh_temp
    
    return (midpoint + radius * torch.tanh(x_scaled)) * cfg.factor


@register_postproc("positive_log_exp")
def _positive_log_exp(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    """
    Smooth positive mapping: ℝ → (0, ∞) with bounded log.
    
    Transforms s via: exp(3·tanh(s))
    • Range: [e^(-3), e^(3)] ≈ [0.05, 20.1]
    • Center: exp(0) = 1 when s=0
    • Smooth saturation at extremes (sigmoid-like)
    • Numerically stable (no overflow/underflow)
    
    Use case: Scale factors that must stay positive but shouldn't explode.
    """
    max_log = cfg.logexp_max_log
    bounded_log = max_log * torch.tanh(s)  # s → [-3, 3]
    return (torch.expm1(bounded_log) + 1.0) * cfg.factor  # e^bounded_log


# ===== Simplified PostProcessor =====
class PostProcessor:
    @staticmethod
    def process(s: torch.Tensor, config: PostProcessConfig) -> torch.Tensor:
        if config.post_process == "none":
            return s
        
        processor = _POST_PROCESSORS.get(config.post_process)
        if processor is None:
            raise ValueError(f"Unknown post_process: {config.post_process}")
        
        return processor(s, config)


@staticmethod 
def sparsify_safely(s: torch.Tensor, cfg: PostProcessConfig) -> torch.Tensor:
    """
    Sparsify eigenvalues by zeroing out values after largest spectral gaps.
    Algorithm (per patch or globally):
    ────────────────────────────────────
    1. Sort eigenvalues descending: [λ₁, λ₂, ..., λₚ] where λ₁ ≥ λ₂ ≥ ... ≥ λₚ
    2. Compute gaps: gap[i] = λᵢ - λᵢ₊₁  (size P-1)
    3. Find top-K largest gaps (e.g., K=3)
    4. Set cutoff = rightmost position among top-K gaps + 1
    5. Keep ranks < cutoff, zero out rest
    6. Restore original order
    Example (1D, no patches):
    ─────────────────────────
    Input:     [0.9, 0.1, 0.85, 0.05, 0.02]
    Sorted:    [0.9, 0.85, 0.1, 0.05, 0.02]  (descending)
    Gaps:      [0.05, 0.75, 0.05, 0.03]      (differences)
               [gap0, gap1, gap2, gap3]
    Top-2:     gap1=0.75 (rank 1), gap0=0.05 (rank 0)
    Rightmost: rank 1 → cutoff = 2
    Keep:      [0.9, 0.85, X, X, X]          (ranks 0,1 kept)
    Reorder:   [0.9, X, 0.85, X, X]          (back to original positions)
    Patched Mode:
    ─────────────
    If cfg.patch_spec defines patches (L = G×P):
      - Input [..., L] → [..., G, P]
      - Sparsify INDEPENDENTLY per patch [..., G, P]
      - Output [..., L] (unpatchified)
    Args:
        s: Eigenvalues [..., L]
        cfg: Config with patch_spec, top_k_gaps, min_keep, use_abs
    Returns:
        Sparsified eigenvalues, same shape as input
    """
    orig_shape = s.shape
    x = s.abs() if cfg.use_abs else s
    # ═══════════════════════════════════════════════════════════
    # STEP 1: Patchify (if enabled)
    # ═══════════════════════════════════════════════════════════
    # Input:  [..., L]
    # Output: [..., G, P] where P is at dim=-1
    x = fcore.patchify(x, patch_spec=cfg.patch_spec, axis=-1)
    P = cfg.patch_spec.P
    if P <= 1:
        return s  # nothing to sparsify
    # ═══════════════════════════════════════════════════════════
    # STEP 2: Sort descending within each patch
    # ═══════════════════════════════════════════════════════════
    # sorted_vals: [..., G, P] - eigenvalues in descending order
    # sorted_idx:  [..., G, P] - indices that did the sorting
    sorted_vals, sorted_idx = torch.sort(x, dim=-1, descending=True)
    # ═══════════════════════════════════════════════════════════
    # STEP 3: Compute spectral gaps
    # ═══════════════════════════════════════════════════════════
    # gaps[i] = sorted_vals[i] - sorted_vals[i+1]
    # Shape: [..., G, P-1]
    gaps = sorted_vals[..., :-1] - sorted_vals[..., 1:]
    # ═══════════════════════════════════════════════════════════
    # STEP 4: Find cutoff based on top-K largest gaps
    # ═══════════════════════════════════════════════════════════
    K = max(1, min(cfg.top_k_gaps, gaps.shape[-1]))
    # Get K-th largest gap value (the smallest of the top-K)
    kth_vals = torch.topk(gaps, k=K, dim=-1, largest=True).values[..., -1:]  # [..., G, 1]
    # Mask: True where gap is in top-K
    top_mask = (gaps >= kth_vals)  # [..., G, P-1]
    # Find RIGHTMOST position among top-K gaps
    # (This is the gap AFTER which we cut)
    ranks = torch.arange(gaps.shape[-1], device=gaps.device)  # [P-1]
    ranks = ranks.view(*(1,) * (gaps.ndim - 1), -1)  # [1, ..., 1, P-1] 
    # rightmost[..., 0] = max rank where top_mask is True
    rightmost = torch.where(
        top_mask, 
        ranks, 
        torch.tensor(-1, device=gaps.device)
    ).max(dim=-1, keepdim=True).values  # [..., G, 1] 
    # Cutoff = rightmost gap position + 1 (keep everything before the gap)
    # Clamp to [min_keep, P] to ensure we keep at least min_keep values

    cutoff = torch.clamp(
        rightmost + 1, 
        min=cfg.min_keep, 
        max=P
    )  # [..., G, 1] 
    # ═══════════════════════════════════════════════════════════
    # STEP 5: Apply cutoff mask in SORTED space
    # ═══════════════════════════════════════════════════════════
    # ranks: [P] for each position in sorted order
    ranks_p = torch.arange(P, device=x.device)  # [P]
    ranks_p = ranks_p.view(*(1,) * (x.ndim - 1), -1)  # [1, ..., 1, P] 
    # Keep only ranks < cutoff
    keep = (ranks_p < cutoff)  # [..., G, P]
    spars_sorted = sorted_vals * keep.to(sorted_vals.dtype) 
    # ═══════════════════════════════════════════════════════════
    # STEP 6: Restore original order (UNSORT)
    # ═══════════════════════════════════════════════════════════
    # Invert the sorting permutation
    inv = torch.argsort(sorted_idx, dim=-1)  # [..., G, P]
    out_patched = torch.gather(spars_sorted, dim=-1, index=inv) 
    # ═══════════════════════════════════════════════════════════
    # STEP 7: Restore signs (if we used abs)
    # ═══════════════════════════════════════════════════════════ 
    if cfg.use_abs:
        out_patched = out_patched * torch.sign(x) 
    # ═══════════════════════════════════════════════════════════
    # STEP 8: Unpatchify back to original shape
    # ═══════════════════════════════════════════════════════════
    out = fcore.unpatchify(out_patched, cfg.patch_spec, axis=-1)
    return out.view(orig_shape)  # Ensure exact shape match

 
class Scale(Op):
    """Some Scale cannot be inversed"""
    coef: torch.Tensor
    post_config: PostProcessConfig 

    def process(self, s: torch.Tensor) -> torch.Tensor:
        processed = PostProcessor.process(s, self.post_config)
        # if self.post_config.use_eigen_sparse:
        #     processed = EigenDiver.sparsify(processed, self.post_config)
        return processed
   
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the apply method.")
 

class DiagScale(Scale): 
    def apply(self, x: torch.Tensor) -> torch.Tensor: 
        return x * self.process(self.coef) 

    def inverse(self, x: torch.Tensor) -> torch.Tensor: 
        return x / (self.process(self.coef) + 1e-9)


class ComplexScale(Scale): 
    """Inverse not implemented, transposition to be defined later if needed."""
    def apply(self, x: torch.Tensor) -> torch.Tensor: 
        return apply_block_diagonal(x, self.process(self.coef))
    

class TriScale(Scale): 
    upper_coef: torch.Tensor = Field(..., description="REQUIRED: Upper diagonal coefficients")
    lower_coef: torch.Tensor = Field(..., description="REQUIRED: Lower diagonal coefficients")
    off_post_config: PostProcessConfig | None = None
 
    @model_validator(mode="after") 
    def validate_coefs(self) -> "TriScale": 
        expected = self.coef.shape[-1] - 1
        if self.upper_coef.shape[-1] != expected or self.lower_coef.shape[-1] != expected:
            raise ValueError(
                f"Off-diagonal coefs must be length {expected}, "
                f"got upper={self.upper_coef.shape[-1]}, lower={self.lower_coef.shape[-1]}"
            )
        return self 
     
    def _process_off_diagonal(self, coef: torch.Tensor) -> torch.Tensor:
        """Process off-diagonal with off_post_config if provided, else return raw."""
        if self.off_post_config is None:
            return coef
        return PostProcessor.process(coef, self.off_post_config)
 
    def apply(self, x: torch.Tensor) -> torch.Tensor: 
        main = self.process(self.coef)  # Inherited from Scale, uses self.post_config
        lower = self._process_off_diagonal(self.lower_coef)
        upper = self._process_off_diagonal(self.upper_coef)
        return apply_tridiagonal(x=x, diag_lower=lower, diag_main=main, diag_upper=upper)
# endregion Scale Op


# region Shift
class Shift(Op):
    coef: torch.Tensor 
    post_config: PostProcessConfig 
 
    def process(self, s: torch.Tensor) -> torch.Tensor:
            return PostProcessor.process(s, self.post_config)
        
    def apply(self, x: torch.Tensor) -> torch.Tensor: 
        return x + self.process(self.coef)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.process(self.coef)
# endregion Shift

# region Rotation
class Rotation(Op):
    """
    vecs for data-dependent rotations
    (v/||v||)ᵀx ; [B,D,1]
    x - 2(v/||v||)((v/||v||)ᵀx) ; [B,D,S]
    """
    unit_vecs: List[torch.Tensor]
    patch_spec: fcore.PatchSpec
    data_lst: List[torch.Tensor] | None = None

    def apply(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.rotate(x, update=update)

    def inverse(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.rotate_back(x, update=update)

    def apply_reflect(
        self, 
        unit_vec: torch.Tensor, 
        x: torch.Tensor,  
        patch_spec: fcore.PatchSpec,
        update: bool = True 
    ) -> torch.Tensor:
        """Apply single Householder reflection: H = I - 2vv^T."""  
        with torch.set_grad_enabled(update):
            x_gp = fcore.patchify(x.contiguous(), patch_spec=patch_spec, axis=-1) 
            v_gp = fcore.patchify(unit_vec.contiguous(), patch_spec=patch_spec, axis=-1)
            dot = (v_gp * x_gp).sum(dim=-1, keepdim=True)       # [B,C,G,1]
            y_gp = x_gp - 2.0 * v_gp * dot                      # [B,C,G,P]
            y_gp = fcore.unpatchify(y_gp, patch_spec, axis=-1)        # [B,C,L] 
        return y_gp 

    def rotate(self, data: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Apply sequence of Householder reflections H_n(...H_2(H_1(x)))."""
        with torch.set_grad_enabled(update):
            for i, unit_vec in enumerate(self.unit_vecs):
                data = self.apply_reflect(unit_vec=unit_vec, x=data, patch_spec=self.patch_spec, update=update)
        return data

    def rotate_back(self, data: torch.Tensor, update: bool = True) -> torch.Tensor:
        """Apply reverse sequence H_1(H_2(...H_n(x)))."""
        nR = len(self.unit_vecs)
        with torch.set_grad_enabled(update):
            for i, unit_vec in enumerate(reversed(self.unit_vecs)):
                data = self.apply_reflect(unit_vec=unit_vec, x=data, patch_spec=self.patch_spec, update=update)
        return data
# endregion Rotation

# region RotateFactory
class RotateFactory(nn.Module):
    """Real-valued Data-dependent rotation matrix via Householder reflections""" 
    def __init__( self, schema: fcore.RotationSchema, ):
        super().__init__()
        self.num_reflects = schema.num_reflects 
        self.schema = schema
        self.block_size = schema.block_size
        self.foundry = schema.build_rotation_heads()

        self._init_random_unit_vec_buffer(schema.device, schema.dtype)
         
        # schedule pointer: which block of block_size is active this step
        self.current_block_idx = 0
        
        if self.block_size < self.num_reflects:
            assert self.num_reflects % self.block_size == 0, "For the minimal rotating schedule, require num_reflects % block_size == 0."
            self._num_blocks = self.num_reflects // self.block_size
        else:
            self._num_blocks = 1
            
    def _init_random_unit_vec_buffer(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
        ) -> None:
        """
        Allocate a cached buffer of random unit vectors shaped (1, C, R, out_dim).
        If patching is enabled (P != None), normalize *within each patch* of length P.
        """
        C, out_dim = self.schema.channels, self.schema.out_dim
        R          = self.schema.num_reflects
        patch_spec       = self.schema.patch_spec

        # 1) sample raw noise (no strings for dtype/device)
        raw = torch.randn((1, C, R, patch_spec.L), device=device, dtype=dtype, generator=generator)

        # 2) per-patch normalization if patched; otherwise whole-axis normalize 
        raw = fcore.patchify(raw, patch_spec=patch_spec, axis=-1)   # (1, C, R, G, P) with P at -1
        with torch.autocast(device_type=str(device).split(":")[0], enabled=False): 
            raw = F.normalize(raw, p=2, dim=-1, eps=1e-9)                            # unit vectors *inside each patch*
        raw = fcore.unpatchify(raw, patch_spec, axis=-1)                                     # back to (1, C, R, out_dim)
   
        # 3) cache
        self.register_buffer("_rand_u", raw, persistent=True) # (1, C, R, out_dim)
 
    def gen_random_unit_vecs(self, src: torch.Tensor) -> list[torch.Tensor]:
        R          = self.schema.num_reflects

        if not hasattr(self, "_rand_u"):
            self._init_random_unit_vec_buffer(src.device, src.dtype) 
        raw = self._rand_u.expand(src.shape[0], -1, -1, -1)  # (B,C,R,out_dim)
        return [raw[..., r, :] for r in range(R)]

    # ---- core: data-dependent unit vectors with sparse-backward gating ----
    def gen_unit_vecs(self, src: torch.Tensor) -> List[torch.Tensor]:
        C, out_dim = self.schema.channels, self.schema.out_dim
        R          = self.schema.num_reflects
        patch_spec       = self.schema.patch_spec

        u = self.foundry[fcore.CoefName.ROTATION.name](src)  # (B,C, out_dim * R) 
        u = u.view(*u.shape[:-1], self.num_reflects, -1)  # (B, C, R, out_dim)

        # per-patch normalization if needed (keep P at last axis) 
        u = fcore.patchify(u, patch_spec=patch_spec, axis=-1) 
        with torch.autocast(device_type=str(src.device).split(":")[0], enabled=False):
            u = F.normalize(u, p=2, dim=-1, eps=1e-9)  # (B, C, R, G, P) normalize over size -P patches
        u = fcore.unpatchify(u, patch_spec=patch_spec, axis=-1) 
        
        # ---- sparse-backward schedule (minimal rotating blocks) ----
        grad_on = torch.is_grad_enabled()  
        if grad_on and self.training and self.block_size < R:
            # active block [start : start + block_size)
            start = (self.current_block_idx * self.block_size) % R
            end   = start + self.block_size
            mask = torch.zeros(R, dtype=torch.bool, device=u.device)
            mask[start:end] = True # designate active block of reflectors

            # broadcast mask to (B,C,R,D)
            m = mask.view(1, 1, R, 1).to(u.dtype)
            u_eff = u.detach() + (u - u.detach()) * m      # stop-grad outside the active block

            # advance pointer once per forward in training
            self.current_block_idx = (self.current_block_idx + 1) % self._num_blocks
        else:
            u_eff = u  # all reflectors trainable (eval mode or block_size==self.num_reflects)
        return [u_eff[..., r, :] for r in range(self.num_reflects)]
 
    def forward(self, source: torch.Tensor) -> Rotation:
        return Rotation( unit_vecs=self.gen_unit_vecs(source), patch_spec=self.schema.patch_spec,  )
# endregion RotateFactory
 
 
# region States

 

class Histogram(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False, extra="forbid",)

    edges: torch.Tensor = Field(..., description="Histogram bin edges")
    counts_lst: List[int] = Field(..., description="Counts per bin")
    total_counts: int = 0
    total_values: float = 0.0

    @classmethod
    def setup_histogram_with_edges(cls, device, dtype, edges: Sequence[float]=(1e-6, 1e-1, 1.0, 3.0, 10.0)) -> "Histogram":
        assert len(edges) >= 1, "Histogram requires at least one edge." 
        edge_tensor = torch.as_tensor(edges, device=device, dtype=dtype)
        return cls(edges=edge_tensor, counts_lst=[0] * (len(edges) + 1), total_counts=0, total_values=0.0)
        
    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x_flat_abs = x.detach().abs().reshape(-1) 
        edges = self.edges.to(device=x_flat_abs.device, dtype=x_flat_abs.dtype)  # no mutation of self.edges

        idx = torch.bucketize(x_flat_abs,  edges, right=False).tolist()
        for i in idx:
            self.counts_lst[int(i)] += 1
        self.total_counts += x_flat_abs.numel()
        self.total_values += x_flat_abs.sum().item()

    def reset(self) -> None:
        self.counts_lst = [0] * (len(self.edges) + 1)
        self.total_counts = 0
        self.total_values = 0.0

    def histogram(self) -> List[int]:
        return list(self.counts_lst)

    def n(self) -> int:
        return int(self.total_counts)
    
    def mean(self) -> float:
        return self.total_values / self.total_counts if self.total_counts > 0 else 0.0

class CoefEigenMonitor(BaseModel):
    """ 
    - Track three things per update:
    1) scale magnitudes (|scale|) [B, C, H]
    2) shift magnitudes (|shift|)
    3) eigen "strength" proxy derived from scale op:
        - DiagScale:       eig = |main|
        - ComplexScale:    pairs (a,b) -> |a+ib|, repeated to length H
        - TriScale:        Gershgorin radius ≈ |main| + |lower|_shifted + |upper|_shifted
    - For each tensor: compute zero-fraction, bucket fractions by |·| using bins,
    update running totals, and offer print/report utilities. 

    Usage:
        mon = CoefEigenMonitor(bins=(1., 3., 6., 10.) )
        ...
        mon.update_from_states(states)             # after scale/shift have been set
        mon.print_batch_summary(prefix="TRAIN")    # optional per-step print
        ...
        mon.print_running_summary(prefix="EPOCH")  # end-of-epoch summary
        stats = mon.report()                       # dict for logging
    """
    model_config = ConfigDict( arbitrary_types_allowed=True, validate_assignment=False, extra="forbid", ) 
    device: str | torch.device ='cuda'   # CHG: store only what you use
    dtype: torch.dtype = torch.float32
 
    scale_hist: Histogram = Histogram.setup_histogram_with_edges(
        device='cpu', dtype=torch.float32, edges=(1e-6,1e-1,1.0,3.0,10.0))
    scale_delta_hist: Histogram = Histogram.setup_histogram_with_edges(
        device='cpu', dtype=torch.float32, edges=(1e-6,1e-1,1.0,3.0,10.0))
    shift_hist: Histogram = Histogram.setup_histogram_with_edges(
        device='cpu', dtype=torch.float32, edges=(1e-6,1e-1,1.0,3.0,10.0))
    shift_delta_hist: Histogram = Histogram.setup_histogram_with_edges(
        device='cpu', dtype=torch.float32, edges=(1e-6,1e-1,1.0,3.0,10.0))
    eigen_hist: Histogram = Histogram.setup_histogram_with_edges(
        device='cpu', dtype=torch.float32, edges=(1e-6,1e-1,1.0,3.0,10.0))
 
    eig_patches_max: torch.Tensor | None  = None      # [B,C,H]
    eig_patches_sum: torch.Tensor | None  = None      # [B,C,H]
    eig_patches_logdet: torch.Tensor | None  = None   # [B,C,H]
    eig_patches_mags: torch.Tensor | None  = None             # [B,C,H]

    @classmethod
    def create_monitor(cls, 
        device: str | torch.device ='cuda', 
        dtype: torch.dtype = torch.float32,
        bins: Sequence[float]=(1e-6, 1e-1, 1.0, 3.0, 10.0)
    ) -> "CoefEigenMonitor":
        scale_hist = Histogram.setup_histogram_with_edges(device=device, dtype=dtype, edges=bins)
        shift_hist = Histogram.setup_histogram_with_edges(device=device, dtype=dtype, edges=bins)
        eigen_hist = Histogram.setup_histogram_with_edges(device=device, dtype=dtype, edges=bins)
        scale_delta_hist = Histogram.setup_histogram_with_edges(device=device, dtype=dtype, edges=bins)
        shift_delta_hist = Histogram.setup_histogram_with_edges(device=device, dtype=dtype, edges=bins)
        return cls(device=device, dtype=dtype, scale_hist=scale_hist, shift_hist=shift_hist, eigen_hist=eigen_hist, scale_delta_hist=scale_delta_hist, shift_delta_hist=shift_delta_hist)
    # ---------- public API ---------- 
    @torch.no_grad()
    def update_scale(self, op: "Scale") -> None:
        proc = op.process(op.coef)
        delta = proc - op.coef
        self.scale_hist.update(proc) 
        self.scale_delta_hist.update(delta)
        
    @torch.no_grad()
    def update_shift(self, op: "Shift") -> None: 
        proc = op.process(op.coef)
        delta = proc - op.coef
        self.shift_hist.update(proc)
        self.shift_delta_hist.update(delta)

    @torch.no_grad()
    def update_eigen(self, op: "Op") -> None:
        eig_abs = self._eigen_magnitudes_from_op(op)
        self.eigen_hist.update(eig_abs)

    def epoch_histograms(self) -> Dict[str, List[int]]:
        return {
            "scale": self.scale_hist.histogram(),
            "scale_delta": self.scale_delta_hist.histogram(),
            "shift": self.shift_hist.histogram(),
            "shift_delta": self.shift_delta_hist.histogram(),
            "eigen": self.eigen_hist.histogram(), 
        }
      
    @torch.no_grad()
    def _eigen_magnitudes_from_op(self, op: "Op") -> torch.Tensor:
        kind = type(op).__name__.lower()
        coef_proc = op.process(op.coef).detach()
        if "diagscale" in kind:
            return coef_proc.abs()
        if "complexscale" in kind:
            b, c, h = coef_proc.shape
            z = coef_proc.view(b, c, h // 2, 2)
            mag = torch.linalg.vector_norm(z, ord=2, dim=-1)
            return mag.repeat_interleave(2, dim=-1).abs()
        if "triscale" in kind:
            main = coef_proc.abs()  
            if op.lower_coef is None or op.upper_coef is None:
                raise ValueError("TriScale missing off-diagonal coefficients.")
            lower = op._process_off_diagonal(op.lower_coef).detach().abs()
            upper = op._process_off_diagonal(op.upper_coef).detach().abs()  
            lower_pad = F.pad(lower, (1, 0))  # shift right
            upper_pad = F.pad(upper, (0, 1))  # shift left
            return main + lower_pad + upper_pad
        return coef_proc.abs()

    
    @torch.no_grad()
    def _compute_patch_reductions(self, states: "States", *, eps: float = 1e-6) -> None:
        assert states.scale is not None, "States.scale is None; cannot compute eigen patch reductions."

        eig_abs = self._eigen_magnitudes_from_op(states.scale)  # [B,C,H]

        self.eigen_hist.update(eig_abs)
        B, C, H = eig_abs.shape

        patch_spec = states.patch_spec

        v = fcore.patchify(eig_abs, patch_spec=patch_spec, axis=-1)  # [B,C,G,P]
        mx_gp = v.max(dim=-1, keepdim=True).values  # [B,C,G,1]
        sm_gp = v.sum(dim=-1, keepdim=True)         # [B,C,G,1] TODO: trace is NOT for abs, though in all pos case doesn't matter
        lg_gp = torch.log(v.clamp_min(eps)).sum(dim=-1, keepdim=True)  # [B,C,G,1]
        mx = fcore.unpatchify(mx_gp.expand(B, C, patch_spec.G, patch_spec.P), patch_spec, axis=-1)
        sm = fcore.unpatchify(sm_gp.expand(B, C, patch_spec.G, patch_spec.P), patch_spec, axis=-1)
        lg = fcore.unpatchify(lg_gp.expand(B, C, patch_spec.G, patch_spec.P), patch_spec, axis=-1)
  
        self.eig_patches_mags   = eig_abs
        self.eig_patches_max    = mx # [B,C,H]
        self.eig_patches_sum    = sm
        self.eig_patches_logdet = lg  
 
    def print_epoch_summary(self) -> None:
        """Print histogram summary at end of epoch."""
  
        edges = self.scale_hist.edges  # Assuming all use same edges
        
        def format_hist(name: str, hist: Histogram) -> str:
            counts = hist.histogram()
            total = sum(counts)
            if total == 0:
                return f"{name}: [no data]"
            
            # Format bins
            bins_str = []
            for i, count in enumerate(counts):
                pct = 100 * count / total
                if i == 0:
                    label = f"<{edges[0]:.1g}"
                elif i == len(counts) - 1:
                    label = f"≥{edges[-1]:.1g}"
                else:
                    label = f"{edges[i-1]:.1g}–{edges[i]:.1g}"
                bins_str.append(f"{label}:{pct:5.1f}%")
            
            return f"{name:12} [{total:8d}]: " + "  ".join(bins_str)
        
        print("\n=== Coefficient Histograms ===")
        print(format_hist("scale", self.scale_hist))
        print(format_hist("scale_delta", self.scale_delta_hist))
        print(format_hist("shift", self.shift_hist))
        print(format_hist("shift_delta", self.shift_delta_hist))
        print(format_hist("eigen", self.eigen_hist))
        print()
    
    def reset(self) -> None:
        """Clear all histograms."""
        self.scale_hist.reset()
        self.scale_delta_hist.reset()
        self.shift_hist.reset()
        self.shift_delta_hist.reset()
        self.eigen_hist.reset()
        # Clear patch data
        self.eig_patches_mags = None
        self.eig_patches_max = None
        self.eig_patches_sum = None
        self.eig_patches_logdet = None
      
#region States
class States(BaseModel):
    model_config = ConfigDict( arbitrary_types_allowed=True, validate_assignment=False, extra="forbid", ) 
    # System information 
    is_training: bool = True 
    patch_spec: fcore.PatchSpec
    
    # Temporary fields for data flow
    x: torch.Tensor
    z: torch.Tensor
    y: Optional[torch.Tensor] = None 
    h: Optional[torch.Tensor] = None 

    scale: Optional["Op"] = None
    shift: Optional["Op"] = None
    rotation: Optional["Rotation"] = None 
     
    monitor: CoefEigenMonitor = Field(default_factory=CoefEigenMonitor)

    # recorder
    """A fancier way to say self.scale = scale, recording raw and post-processed coefs for  later analysis"""
    @torch.no_grad()
    def set_scale(self, scale: "Op", detach: bool = True) -> "States":
        self.scale = scale 
        return self
    
    @torch.no_grad()
    def set_shift(self, shift: "Op", detach: bool = True) -> "States":
        self.shift = shift 
        return self
    
    @torch.no_grad()
    def set_rotation(self, rotation: "Rotation") -> "States":
        self.rotation = rotation
        return self 
     
    def __or__(self, op):  # allow states | op style
        return op(self)

    def set_field(self, field_name: str, field_value: torch.Tensor | Op) -> "States":
        return self.model_copy(update={field_name: field_value})

    def get_field(self, field_name: str) -> torch.Tensor | Op | None:
        return getattr(self, field_name)

    def set_x(self, x: torch.Tensor) -> "States":
        return self.model_copy(update={"x": x})

    def set_y(self, y: torch.Tensor) -> "States":
        return self.model_copy(update={"y": y})

    def set_z(self, z: torch.Tensor) -> "States":
        return self.model_copy(update={"z": z})
    
    def set_h(self, h: torch.Tensor) -> "States":
        return self.model_copy(update={"h": h}) 
# endregion States
  
# region HiddenFactory
class HiddenFactory(nn.Module):
    def __init__(self, schema: fcore.HiddenSchema):
        super().__init__()
        self.schema = schema
        self.channels = schema.channels 
        self.hidden_net = self.schema.build_heads() 
  
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        h = self.hidden_net(x)  # (B, C, H)
        return h
# endregion HiddenFactory


# region CoefFactory
class CoefFactory(nn.Module):
    """ The architecture is defined in the Pydantic config, and this class builds it. """
    def __init__(self, schema: fcore.CoefSchema):
        super().__init__()
        self.schema = schema 

        foundry = self.schema.build_heads()
        self.foundry = foundry  # ModuleDict[fcore.CoefName, nn.Module] 
    
    def _pp_cfg(self, kind, low=0.0, high=6.0, 
                factor=1.0, beta=1.0, leak=0.05, tanh_temp=None, logexp_max_log=3.0,) -> PostProcessConfig:
        return PostProcessConfig(
            post_process=kind,
            low=low, high=high, factor=factor,
            softplus_sharpness=beta, 
            leak=leak, 
            tanh_temp=tanh_temp,
            logexp_max_log=logexp_max_log,
            top_k_gaps=self.schema.top_k_gaps, min_keep=1,  
            patch_spec=self.schema.patch_spec,
            use_eigen_sparse=self.schema.use_eigen_sparse,
        )
        
    # (1-leak)*cap + leak*identity
    def _build_shift_op(self, raw_coefs: dict) -> Shift:
        return Shift(
            coef=raw_coefs[fcore.CoefName.SHIFT],
            post_config=self._pp_cfg(kind="softclip_leaky", low=-15.0, high=15.0, 
                factor=1.0, beta=0.8, leak=0.1, tanh_temp=None),
        )  # softclip_leaky bounded_tanh none

    # region Build Scale Op
    def _build_scale_op(self, raw_coefs: dict) -> Scale:
        scale_structure = self.schema.scale_structure
        # print(f"scale_structure: {scale_structure}",flush=True)
        coef_dim = raw_coefs[fcore.CoefName.SCALE].shape[-1]
        if fcore.CoefName.OFF_SCALE in raw_coefs:
            off_dim = raw_coefs[fcore.CoefName.OFF_SCALE].shape[-1]
            if off_dim != coef_dim - 1:
                raise ValueError(
                    f"Off-scale coef shape mismatch: coef dim is {coef_dim}, off-scale coef dim should be {coef_dim - 1}, but is {off_dim}"
                ) 

        scale = raw_coefs[fcore.CoefName.SCALE]

        if scale_structure == "diagonal":
            return DiagScale(
                coef=scale,
                post_config=self._pp_cfg(kind="softclip_leaky", low=-1.0, high=4.5, 
                    factor=1.0, beta=0.8, leak=0.1, tanh_temp=None,
                    ),
            )  # relu6_softcap_leaky

        elif scale_structure == "complex":
            return ComplexScale(
                coef=scale,
                post_config=self._pp_cfg(kind="softclip_leaky", low=-4.5, high=4.5, 
                    factor=1.0, beta=0.8, leak=0.1, tanh_temp=None),
            )  # softclip_leaky

        elif scale_structure in {"tri_anti", "tri_sym"}:
            upper = raw_coefs[fcore.CoefName.OFF_SCALE]   
            return TriScale(
                coef=scale,
                lower_coef=(-upper if scale_structure in {"tri_anti"} else upper),
                upper_coef=upper,
                post_config=self._pp_cfg(kind="softclip_leaky", low=-5.0, high=5.0, 
                    factor=1.0, beta=0.8, leak=0.1, tanh_temp=None),
            )

        else:
            raise NotImplementedError(f"Unknown scaling scale_structure: {scale_structure}") 
    # endregion _build_scaling_op
    
    def forward(
        self,
        h: torch.Tensor,  # (B, C, H)
        *,
        want: tuple[fcore.CoefName | str, ...] | None = None,
        update: bool = True,
    ) -> Tuple[Scale, Shift]:
        # ---- normalize 'want' to a set of fcore.CoefName ----
        if want is None:
            requested = {fcore.CoefName.SCALE, fcore.CoefName.SHIFT}
        else:
            requested = {  w if isinstance(w, fcore.CoefName) else fcore.CoefName[w.upper()] for w in want }

        # ---- schema dims ----
        B, C, H = h.shape
        assert C == self.schema.channels and H == self.schema.hid_dim, (
            f"h (B,{C},{H}) != schema (C={self.schema.channels}, H={self.schema.hid_dim})"
        ) #TODO uncomment later

        with torch.set_grad_enabled(update):
            raw_coefs = {} 
            for head, spec in   self.schema.head_specs.items():   
                coef = self.foundry[head.name](h)  # (B,C,H_head)
                if self.schema.stitch:
                    coef = stitch(
                        coef,
                        patch_len=self.schema.stitch.patch_len,
                        stride=self.schema.stitch.stride,
                        input_len=self.schema.out_dim,
                        num_patches=self.schema.stitch.num_patches,
                        window=self.schema.stitch.window,  # "ones" | "hann" | None
                        normalize=self.schema.stitch.normalize,  # "avg" | "sum"
                    ) 
                # ---- post thresholds/clips if present on spec.post ----
                post = getattr(spec, "post", None)
                if post is not None:
                    zero_threshold = getattr(post, "zero_threshold", None)
                    clip_upper = getattr(post, "clip_upper_bound", None)  
                    clip_lower = getattr(post, "clip_lower_bound", None)

                    if zero_threshold is not None and zero_threshold > 0.0: 
                        coef = F.threshold(coef, zero_threshold, 0.0)

                    # Only clamp if at least one bound is provided
                    if clip_lower is not None or clip_upper is not None:
                        if clip_lower is None:
                            coef = torch.clamp(coef, max=float(clip_upper))
                        elif clip_upper is None:
                            coef = torch.clamp(coef, min=float(clip_lower))
                        else:
                            coef = torch.clamp(
                                coef, min=float(clip_lower), max=float(clip_upper)
                            )

                raw_coefs[head] = coef

            scale = self._build_scale_op(raw_coefs)
            shift = self._build_shift_op(raw_coefs)
        return (scale, shift)
# endregion CoefFactory

def stitch(
    patches: torch.Tensor,  # (B, C, H)
    *,
    patch_len: int,
    stride: int,
    input_len: int,
    num_patches: int,
    window: Optional[Literal["ones", "hann"]] = "ones",
    normalize: Literal["avg", "sum"] = "avg",
) -> torch.Tensor:
    """
    1D stitch implemented via F.fold by pretending height=1.
    This is compact and fast when P is large, but requires an exact sliding grid.
    """ 
    B, C, _ = patches.shape
    patches = patches.view(B, C, num_patches, patch_len)
    G = num_patches
    P = patch_len

    # Compute the number of sliding blocks F.fold expects and enforce exact cover.
    expected_G = (input_len - patch_len) // stride + 1
    if (input_len - patch_len) % stride != 0 or expected_G != G:
        raise ValueError(
            f"fold requires exact grid: got P={P}, but input_len={input_len}, "
            f"patch_len={patch_len}, stride={stride} imply P={expected_G}."
        )

    dev, dt = patches.device, patches.dtype

    # region patch_stitch  
    w = torch.ones(P, device=dev, dtype=dt) if window  == "ones" else (
        torch.hann_window(P, periodic=False, device=dev, dtype=dt)
    )   
    # Prepare the 'im2col' style tensor for fold:
    # (N, C * kernel_height(=1) * kernel_width(=P), n_blocks) .
    num = patches * w  # (B,C,G,P) * (P,)
    num = num.permute(0, 1, 3, 2).reshape(B, C * P, G)  # (B, C*P, G)
 
    # [B,C,1,T]
    out_num = F.fold( num, output_size=(1, input_len), kernel_size=(1, P), stride=(1, stride) )  
    

    if normalize == "sum": return out_num.squeeze(2)

    # Denominator via folding window ones
    den_cols = torch.ones((B, C, G, P), device=dev, dtype=dt) * w  # (B,C,G,P)
    den_cols = den_cols.permute(0, 1, 3, 2).reshape(B, C * P, G)  # (B, C*P, G)
    out_den = F.fold( den_cols, output_size=(1, input_len), kernel_size=(1, P), stride=(1, stride) )  # (B, C, 1, target_len)

    return (out_num / torch.clamp_min(out_den, 1e-12)).squeeze(2)

