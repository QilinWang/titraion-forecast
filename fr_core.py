
from re import L, S
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from pydantic import (
    BaseModel, ConfigDict, model_validator, computed_field, Field, PositiveInt,
)
from typing import (
    List, Tuple, Union, Callable, Dict, Optional, Any, Self, Literal,
    Annotated, Iterable, Set, Sequence, assert_never,
)
from enum import StrEnum, auto
import math
from dataclasses import dataclass
 
# ---- Helper functions ---- 
# region patching 
class PatchSpec(BaseModel):
    L: int                  # full length on the working axis
    P: PositiveInt         # patch_size 

    @computed_field
    @property
    def G(self) -> PositiveInt:
        return self.L // self.P  # num_patches

    @model_validator(mode="after")
    def _validate(self) -> "PatchSpec":
        if self.L % self.P != 0:
            raise ValueError("Length L must be divisible by patch_size P")
        return self

    @classmethod
    def from_len(cls, L: int, *, patch_size: PositiveInt) -> "PatchSpec": 
        return cls(L=L, P=patch_size)

def patchify(x: torch.Tensor, *, patch_spec: PatchSpec, axis: int = -1,) -> torch.Tensor:
    """  
    Convert length-`L` axis into two trailing axes (..., G, P) with P at -1.
    Shapes: input (..., L) (on `axis`) → output (..., G, P) with P at -1, G at -2. 
    """ 
    x = x.movedim(axis, -1) # Move the working axis to the end so we can split it cleanly. 
    x = x.reshape(*x.shape[:-1], patch_spec.G, patch_spec.P) # Split the last axis: (..., L) → (..., G, P) with P LAST.
    return x  # P@-1, G@-2 (by construction)

def unpatchify(x: torch.Tensor, patch_spec: PatchSpec, *, axis: int = -1,) -> torch.Tensor:
    """ 
    - Input:  Merge the trailing (..., G, P) back to (..., L), then move L to `axis`. (from `patchify`)  
    - Output: (..., L) placed on the same `axis` you originally passed to patchify
    """ 
    x = x.reshape(*x.shape[:-2], patch_spec.L)   # (..., L) # Merge (G, P) → L on the tail, then put L back to `axis`.
    return x.movedim(-1, axis)

def reduce_patchwise(data: torch.Tensor, patch_spec: PatchSpec, *, reduce: str = "sum", axis: int = -1,) -> torch.Tensor:
    """ Reduce within each patch and return COMPACT shape (..., G, 1).
    - Input s: (..., L) where L = G*P if patched; otherwise L=spec.L
    - Output:   (..., G, 1)   (if not patched, returns (..., 1, 1))
    """
    x = patchify(data, patch_spec=patch_spec, axis=axis)  # (..., G, P) with P at -1
    if reduce == "sum":
        r = x.sum(dim=-1, keepdim=True)            # (..., G, 1)
    elif reduce == "max":
        r = x.max(dim=-1, keepdim=True).values     # (..., G, 1)
    else:
        raise ValueError(f"reduce must be 'sum' or 'max', got {reduce!r}")
    return r

def expand_compact(r_bcg1: torch.Tensor, patch_spec: PatchSpec, *, axis: int = -1,) -> torch.Tensor:
    """
    Expand COMPACT (..., G, 1) back to (..., L) by broadcasting along P, then unpatchify. 
    """ 
    r_gp = r_bcg1.expand(*r_bcg1.shape[:-1], patch_spec.P)   # (..., G, P)
    return unpatchify(r_gp, patch_spec, axis=axis)           # (..., L)

def make_patch_ranks_like(
    x: torch.Tensor,
    patch_spec: PatchSpec,
    *,
    axis: int = -1,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    Build a broadcastable rank index along the **patch axis** P for a tensor shaped like `x`.

    Contract:
      - We first `patchify(x, axis=axis)` so that the last axis is P and the second-to-last is G.
      - We return a tensor of shape [1, 1, ..., 1, P] that broadcasts over all leading dims,
        including batch/channel/... and also over G (the patch index).
      - If not patched, P == L and the result is [1,..,1,L], still broadcastable to x.

    Example:
      x       : [B, C,   G,   P]   after patchify
      ranks   : [1, 1,   1,   P]   broadcasts across B,C,G
      cutoff  : [B, C,   G,   1]   typical per-patch cutoff shape
      keep    : (ranks < cutoff)   → [B, C,   G,   P]
    """
    x_p = patchify(x, patch_spec=patch_spec, axis=axis)  # (..., G, P) with P at -1
    P = patch_spec.P  
    r = torch.arange(P, device=x_p.device, dtype=dtype)   # (P,)
    for _ in range(x_p.dim() - 1):                        # → [1, 1, ..., 1, P]
        r = r.unsqueeze(0)
    return r
# endregion patching

#---- ModuleSpec / SpecKey system ------------------------------------------------
# region SpecKey enums
class IDLayer(StrEnum):
    MODULE = "MODULE"
    FLOW  = "FLOW"

class ModuleFlavour(StrEnum):
    HID  = "HID"   # hidden factory
    SS   = "SS"    # scale/shift (coef) factory
    ROT  = "ROT"   # rotation-in-space factory

class Space(StrEnum):
    X = auto(); H = auto(); Y = auto(); Z = auto()
 
DIM_BY_SPACE = {
    Space.X: "seq_len", Space.Y: "pred_len", Space.Z: "dim_augment", Space.H: "dim_hidden",
}

# region FlowType
class FlowType(StrEnum):
    """A enum for the geometric operations"""
    SHIFT_ONLY = "shift_only"; SCALE_ONLY = "scale_only"
    SCALE_SHIFT = "scale_shift" ; PROB_SCALE_SHIFT = "prob_scale_shift" ; SHIFT_SCALE = "shift_scale"  # shift then scale (rare but handy)
    R = "r"; R_SCALE_SHIFT_RB = "r_scale_shift_rb"; R_SCALE_RB_SHIFT = "r_scale_rb_shift"; R_SCALE_SHIFT = "r_scale_shift"; MOBIUS = "mobius"
# endregion FlowType

class SpecKey(BaseModel):
    """ A typed, extensible identifier that round-trips to a stable string key. """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="forbid")
    layer: IDLayer = Field(..., description="MODULE or FLOW")
    flavour: ModuleFlavour | FlowType = Field(..., description="HID/SS/ROT or flowtype like SCALE_SHIFT")
    src: "Space" = Field(..., description="Source space")
    dst: "Space" = Field(..., description="Destination space")
    version: str = Field(..., description="Version string like v0")
  
    def key(self) -> str: # MODULE:SS:Z<-X:v=v3;in=dst;struct=complex    
        arrow = f"{self.dst.name.lower()}<-{self.src.name.lower()}"
        core  = f"{self.layer}:{self.flavour.name.lower()}:{arrow}:v={self.version}" 
        return core

    @staticmethod
    def from_brick(b: "ModuleSpec") -> "SpecKey":
        if isinstance(b, CoefID):
            return SpecKey(
                layer=IDLayer.MODULE, flavour=ModuleFlavour.SS,
                src=b.src, dst=b.dst, version=b.version,
            )
        elif isinstance(b, RotID):
            return SpecKey(
                layer=IDLayer.MODULE, flavour=ModuleFlavour.ROT,
                src=b.src, dst=b.dst, version=b.version,
            )
        elif isinstance(b, HidID):
            return SpecKey(
                layer=IDLayer.MODULE, flavour=ModuleFlavour.HID,
                src=b.src, dst=b.dst, version=b.version,
            ) 
        assert_never(b)

    @staticmethod
    def from_flow(s: "FlowSpec") -> "SpecKey": # For FLOWs, we keep dst<-src and use flowtype name as flavour.
        return SpecKey(
            layer=IDLayer.FLOW, flavour=s.flowtype,
            src=s.src, dst=s.dst,
            version=s.version,   
        )
#endregion SpecKey enums

class FlowSpec(BaseModel):
    """A declarative flow: src → dst via a geometry, carrying a version tag."""
    src: Space
    dst: Space
    flowtype: FlowType
    version: str = "v0" 

    @property
    def name(self) -> str: # Canonical human name used by FlowRegistry
        return f"{self.src.name.upper()}_to_{self.dst.name.upper()}_{self.version.upper()}"

# For each FlowType, which factory flavours are required?
#   hid=True  -> HID_{src->H}_v0       (flow_templates hardcode HID v0)
#   ss=True   -> SS_{dst<-src}_{v}     (coef version follows FlowSpec.version)
#   rot=True  -> ROT_IN_{dst}_{vrot}   (see vrot rule below)
_FLOW_NEEDS: Dict[FlowType, Dict[str, bool]] = {
    FlowType.SHIFT_ONLY:        {"hid": True, "ss": True,  "rot": False},
    FlowType.SCALE_ONLY:        {"hid": True, "ss": True,  "rot": False},
    FlowType.SCALE_SHIFT:       {"hid": True, "ss": True,  "rot": False},
    FlowType.PROB_SCALE_SHIFT:  {"hid": True, "ss": True,  "rot": False},
    FlowType.SHIFT_SCALE:       {"hid": True, "ss": True,  "rot": False},
    FlowType.R:                 {"hid": True, "ss": False, "rot": True},
    FlowType.R_SCALE_RB_SHIFT:  {"hid": True, "ss": True,  "rot": True},
    FlowType.R_SCALE_SHIFT_RB:  {"hid": True, "ss": True,  "rot": True},
    FlowType.R_SCALE_SHIFT:     {"hid": True, "ss": True,  "rot": True},
    # FlowType.MOBIUS: define when you add a recipe
}

def modulespecs_from_flowspecs(flowspecs: List[FlowSpec]) -> List["ModuleSpec"]: 
    need: Dict[str, "ModuleSpec"] = {}
    for flowspec in flowspecs:
        req = _FLOW_NEEDS.get(flowspec.flowtype)
        if req is None:
            raise ValueError(f"No factory requirements registered for flowtype={flowspec.flowtype}")
 
        if req["hid"]: # adding ModuleSpecs to a key
            hid = HidID(src=flowspec.src, dst=Space.H, version="v0")  # flow_templates use HID v0
            need[hid.module_key] = hid

        if req["ss"]: 
            coef = CoefID.from_src_dst(
                src=flowspec.src,
                dst=flowspec.dst,
                version=flowspec.version,
            )
            need[coef.module_key] = coef

        if req["rot"]:
            rv = _rot_version_for(flowspec.flowtype, flowspec.version)
            rot = RotID(src=flowspec.dst, dst=flowspec.dst, version=rv)  # rotation lives in dst-space
            need[rot.module_key] = rot

    return list(need.values())

def _rot_version_for(flowtype: FlowType, spec_version: str) -> str:
    if flowtype in (FlowType.R_SCALE_RB_SHIFT, FlowType.R_SCALE_SHIFT_RB):
        return "v0"
    return spec_version


class FactoryFlavour(StrEnum):
    HIDDEN = "hid" ; COEF = "ss" ; ROTATION = "rot"

class ScaleMode(StrEnum):
    DYN = "dyn"; FIXED = "fixed"; MIX = "mix"; NONE = "none"

class ShiftMode(StrEnum):
    DYN = "dyn"; FIXED = "fixed"; MIX = "mix"; NONE = "none"

class ScaleStructure(StrEnum):
    COMPLEX = "complex"; DIAGONAL = "diagonal"
    TRI_ANTI = "tri_anti"; TRI_SYM = "tri_sym";  


 
class CoefID(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", validate_assignment=True)

    src: "Space"
    dst: "Space"
    version: str

    scale_structure: ScaleStructure = Field(..., description="scale_structure of the scale") 
    num_experts: PositiveInt = Field(1, description="number of mixture of experts")
    
    @classmethod
    def from_src_dst(
        cls,
        src: Space,
        dst: Space,
        version: str,
        *,
        scale_structure: ScaleStructure | None = None,
        num_experts: PositiveInt = 1,
    ) -> "CoefID":
        if scale_structure is None:
            if dst == Space.Y:
                scale_structure = ScaleStructure.DIAGONAL
            elif dst == Space.X and version not in ("v0", "v3"):
                scale_structure = ScaleStructure.COMPLEX
            else:
                scale_structure = ScaleStructure.DIAGONAL
        return cls(
            src=src,
            dst=dst,
            version=version,
            scale_structure=scale_structure,
            num_experts=num_experts,
        )

    @property
    def flavour(self):
        return FactoryFlavour.COEF

    def __str__(self):
        return f"SS_{self.dst.value.upper()}_GIVEN_{self.src.value.upper()}_{self.version.upper()}"

    @property
    def module_key(self) -> str: # MODULE:SS:dst<-src:v=...
        return SpecKey.from_brick(self).key()
 
    def add_to_ModuleDict(self, cfg: "FERNConfig", moduledict: nn.ModuleDict) -> nn.ModuleDict:
        key = self.module_key
        if key not in moduledict:  # Only build if not present
            moduledict[key] = CoefSchema.coefschema_from_cfg_and_id(cfg, self).build_factory()
        return moduledict
       

class RotID(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 
    src: "Space"
    dst: "Space"
    version: str # Note: no cfg-dependent knobs here. Rotation details come from cfg/schema.

    @property
    def flavour(self):
        return FactoryFlavour.ROTATION

    def __str__(self):
        return f"ROT_IN_{self.dst.value.upper()}_{self.version.upper()}"

    @property
    def module_key(self) -> str:
        # MODULE:ROT:dst<-dst:v=...;in=dst
        return SpecKey.from_brick(self).key()
 
    def add_to_ModuleDict(self, cfg: "FERNConfig", moduledict: nn.ModuleDict) -> nn.ModuleDict:
        key = self.module_key
        if key not in moduledict:  # Only build if not present
            moduledict[key] = RotationSchema.rotation_schema_from_cfg_and_id(cfg, self).build_factory()
        return moduledict
    

class HidID(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    src: "Space"
    dst: "Space"
    version: str

    @property
    def flavour(self):
        return FactoryFlavour.HIDDEN

    def __str__(self):
        return f"HID_GIVEN_{self.src.value.upper()}_{self.version.upper()}"

    @property
    def module_key(self) -> str: # MODULE:HID:H<-src:v=...
        return SpecKey.from_brick(self).key()
 
    def add_to_ModuleDict(self, cfg: "FERNConfig", moduledict: nn.ModuleDict) -> nn.ModuleDict:
        key = self.module_key
        if key not in moduledict:  # Only build if not present
            moduledict[key] = HiddenSchema.hidden_schema_from_cfg_and_id(cfg, self).build_factory()
        return moduledict
    
ModuleSpec = Union[CoefID, RotID, HidID]

# region CoefName
class CoefName(StrEnum):
    """Enum for coefficient names used in CoefSchema and CoefFactory."""
    SCALE = auto() ; SHIFT = auto() ; OFF_SCALE = auto() ;  
    MAGNITUDE = auto() ;  GATE = auto() ; ROTATION= auto()

#region DynamicTanh
class DynamicTanh(nn.Module):
    def __init__(self, channels: int, dim: int,):
        super().__init__()
        self.dim = dim
        self.a = nn.Parameter(torch.ones(channels, self.dim))
        self.b = nn.Parameter(torch.zeros(channels,self.dim))   
        self.c = nn.Parameter(torch.ones(channels, self.dim))
        self.d = nn.Parameter(torch.zeros(channels, self.dim))
          
    def forward(self, x):
        core_in =  x * self.a  + self.b  
        core = torch.tanh(core_in)  
        y = core* self.c  + self.d
        return y 

class ResidualBlock(nn.Module):
    """
    A custom nn.Module that implements the core ResNet idea: y = x + F(x). 
    """
    def __init__(self, f: nn.Module, skip_connection: nn.Module | None = None,
                 dim: int | None = None, channels: int | None = None):
        """
        Args:
            f: The sequence of layers that define the transformation `F(x)`, a small nn.Sequential.
            shortcut: A module (like a Linear layer or 1x1 Conv) to make the dimensions of `x` match the dimensions
                of `F(x)`. This is only needed if `F(x)` changes the shape. Defaults to an identity connection.
        """
        super().__init__()
        self.f, self.dim, self.channels = f, dim, channels
        self.skip_connection = skip_connection if skip_connection is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Defines the forward pass: applies the transformation and adds the skip connection. """ 
        identity = self.skip_connection(x) 
        fx = self.f(x) 
        return identity + fx
#endregion DynamicTanh

class Act(StrEnum):
    IDENTITY = "identity"; RELU = "relu"; RELU6 = "relu6"; LEAKY_RELU = "leaky_relu"
    SOFTPLUS = "softplus"; GELU = "gelu"; ELU = "elu"; CELU = "celu"
    TANH = "tanh"; HARDTANH = "hardtanh"; SIGMOID = "sigmoid"; SOFTSIGN = "softsign"
    SOFTMAX = "softmax"; LOGSOFTMAX = "logsoftmax"; SOFTSHRINK = "softshrink"
    HARDSHRINK = "hardshrink"; RRELU = "rrelu"; SELU = "selu"; TANHSHRINK = "tanhshrink"
    HARDSWISH = "hardswish"; SILU = "silu"; MISH = "mish"
    DYNAMIC_TANH = "dynamic_tanh"; RESBLOCK = "resblock"; LOGSIGMOID = "logsigmoid" 
    
# ---- registry core ----------------------------------------------------------
"""
A global registry: keys are enum values (e.g., Act.RELU), values are builder functions 
that take an ActSetup and return an nn.Module.
"""
_BUILDERS: Dict[Act, Callable[["ActSetup"], nn.Module]] = {}

def register_activation(kind: Act) -> Callable[[Callable[["ActSetup"], nn.Module]], Callable[["ActSetup"], nn.Module]]:
    """A decorator factory. You call it with an enum (e.g., Act.RESBLOCK) 
    and it returns a decorator."""
    def deco(fn: Callable[["ActSetup"], nn.Module]):
        if kind in _BUILDERS:
            raise ValueError(f"Activation {kind} already registered.")
        _BUILDERS[kind] = fn
        return fn
    return deco

"""
@register_activation(Act.RESBLOCK) over a function:
When Python executes this definition, it immediately registers that function in _BUILDERS[kind] 
and returns the function unchanged. 
"""

def register_simple(kind: Act, cls: type[nn.Module], **fixed_kwargs):
    """for activations that have no special kwargs, it auto-wraps the class ctor in a tiny builder and registers that."""
    @register_activation(kind)
    def _build(_: ActSetup) -> nn.Module:
        return cls(**fixed_kwargs)
  
class ActSetup(BaseModel):
    """At runtime, this looks up the builder in _BUILDERS by self.act_type, and calls it with self to produce the actual nn.Module."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    act_type: Act
    # Only needed for DynamicTanh; optional otherwise
    dim: int | None = None
    channels: int | None = None

    # Optional tunables (use when relevant) 
    leaky_relu_negative_slope: float | None = None         # LeakyReLU
    softplus_beta: float | None = None                   # Softplus
    softplus_threshold: float | None = None              # Softplus
    gelu_approximate: Literal["none","tanh"] | None = None  # GELU
    elu_alpha: float | None = None                  # ELU
    celu_alpha: float | None = None                  # CELU
    hard_tanh_min: float | None = None
    hard_tanh_max: float | None = None
    softshrink_lambd: float | None = None
    hardshrink_lambd: float | None = None

    def finalized(self, *, dim: int, channels: int) -> "ActSetup":
        """Return a copy with dim/channels filled only if needed."""
        if self.act_type == Act.DYNAMIC_TANH:
            return self.model_copy(update={
                "dim": self.dim or dim,
                "channels": self.channels or channels,
            })
        return self
    
    def build(self, *, dim: int | None = None, channels: int | None = None) -> nn.Module:
        """Create the nn.Module; optional dim/channels can be supplied at call time."""
        # fill missing shape info if caller provides it
        if dim is not None and self.dim is None:
            object.__setattr__(self, "dim", dim)
        if channels is not None and self.channels is None:
            object.__setattr__(self, "channels", channels)

        kind = self.act_type
        if kind not in _BUILDERS:
            raise ValueError(f"No builder registered for {kind}")
        return _BUILDERS[kind](self)

# ---- register simple ones in one line each ----------------------------------
register_simple(Act.IDENTITY, nn.Identity)
register_simple(Act.RELU, nn.ReLU)
register_simple(Act.RELU6, nn.ReLU6)
register_simple(Act.TANH, nn.Tanh)
register_simple(Act.SIGMOID, nn.Sigmoid)
register_simple(Act.SOFTSIGN, nn.Softsign)
register_simple(Act.MISH, nn.Mish)
register_simple(Act.SELU, nn.SELU)
register_simple(Act.TANHSHRINK, nn.Tanhshrink)
register_simple(Act.HARDSWISH, nn.Hardswish)
register_simple(Act.SILU, nn.SiLU)
register_simple(Act.RRELU, nn.RReLU)            # uses default params
register_simple(Act.SOFTSHRINK, nn.Softshrink)  # uses default params
register_simple(Act.HARDSHRINK, nn.Hardshrink)  # uses default params
register_simple(Act.LOGSIGMOID, nn.LogSigmoid)
# ---- register the few with config-dependent kwargs --------------------------

@register_activation(Act.LEAKY_RELU)
def _build_leaky(cfg: ActSetup) -> nn.Module:
    slope = cfg.leaky_relu_negative_slope if cfg.leaky_relu_negative_slope is not None else 0.07
    return nn.LeakyReLU(negative_slope=slope, inplace=False)

@register_activation(Act.SOFTPLUS)
def _build_softplus(cfg: ActSetup) -> nn.Module:
    beta =  cfg.softplus_beta if cfg.softplus_beta is not None else 1.0
    thr  =  cfg.softplus_threshold if cfg.softplus_threshold is not None else 20.0
    return nn.Softplus(beta=beta, threshold=thr)

@register_activation(Act.GELU)
def _build_gelu(cfg: ActSetup) -> nn.Module:
    approx = cfg.gelu_approximate if cfg.gelu_approximate is not None else "none"
    return nn.GELU(approximate=approx)

@register_activation(Act.ELU)
def _build_elu(cfg: ActSetup) -> nn.Module:
    alpha = cfg.elu_alpha if cfg.elu_alpha is not None else 1.0
    return nn.ELU(alpha=alpha)

@register_activation(Act.CELU)
def _build_celu(cfg: ActSetup) -> nn.Module:
    alpha = cfg.celu_alpha if cfg.celu_alpha is not None else (cfg.elu_alpha or 1.0)
    return nn.CELU(alpha=alpha)

@register_activation(Act.HARDTANH)
def _build_hardtanh(cfg: ActSetup) -> nn.Module:
    mn = cfg.hard_tanh_min if cfg.hard_tanh_min is not None else -1.0
    mx = cfg.hard_tanh_max if cfg.hard_tanh_max is not None else 1.0
    return nn.Hardtanh(min_val=mn, max_val=mx)

@register_activation(Act.SOFTMAX)
def _build_softmax(cfg: ActSetup) -> nn.Module:
    if cfg.dim is None:
        raise ValueError("Softmax requires `dim` (which axis to normalize).")
    return nn.Softmax(dim=cfg.dim)

@register_activation(Act.LOGSOFTMAX)
def _build_logsoftmax(cfg: ActSetup) -> nn.Module:
    if cfg.dim is None:
        raise ValueError("LogSoftmax requires `dim` (which axis to normalize).")
    return nn.LogSoftmax(dim=cfg.dim)

@register_activation(Act.DYNAMIC_TANH)
def _build_dynamic_tanh(cfg: ActSetup) -> nn.Module:
    if cfg.channels is None or cfg.dim is None:
        raise ValueError("DynamicTanh requires `channels` and `dim`.")
    return DynamicTanh(cfg.channels, cfg.dim)

@register_activation(Act.RESBLOCK)
def _build_resblock(cfg: ActSetup) -> nn.Module: 
    inner = nn.ELU(alpha=1.0) # default inner activation
    return ResidualBlock(f=inner)

"""
# leaky relu with slope
cfg = ActSetup(act_type=Act.LEAKY_RELU, leaky_relu_negative_slope=0.1)
act = cfg.build()

# dynamic tanh (needs channels & dim)
cfg = ActSetup(act_type=Act.DYNAMIC_TANH)
act = cfg.build(dim=128, channels=32)  # supply at call-time or set in cfg
"""

#region 
class FixedScale(nn.Module):
    """
    intent/contract:
        # REQUIRE: input is [B,D,H]
        Return Fixed learned scale γ[B,D,H]. No x-dependence.
    """
    def __init__(self, channels: int, length: int):
        super().__init__()  
        self.gamma = nn.Parameter(torch.ones(channels, length))
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        B, D, H = x.shape
        if (D, H) != self.gamma.shape:
            raise ValueError(f"FixedScale: got x.shape={tuple(x.shape)} but γ.shape={tuple(self.gamma.shape)}")
        return self.gamma.unsqueeze(0).expand(B, -1, -1)  # [B,D,H]
#endregion

#region FixedSpec
class FixedSpec(BaseModel):
    out_dim: int = Field(..., description="Unused, for bookkeeping only.")
    channels: int
    has_scale: bool = True
    has_shift: bool = True

    def build_fixed(self) -> FixedScale:
        return FixedScale(
            channels=self.channels,
            length=self.out_dim,
        ) 
# endregion FixedSpec

# region ConvSpec   
class ConvSpec(BaseModel):
    """Specification for depthwise/grouped convolutions for shape [B D S]"""  
    out_dim: int = Field(..., description="Unused, for bookkeeping only.")
    in_channels: int
    out_channels: int
    kernel_size: int = Field(..., description="Size of the convolution kernel.")
    groups: int = Field(..., description="Number of groups for grouped convolution.")
    bias: bool = True
    is_depthwise: bool = True
 
    def build_conv(self) -> nn.Conv1d: 
        conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding="same",
            groups=self.groups,
            bias=self.bias
        ) 
        return conv 
#endregion ConvSpec

#region LinearSpec
class LinearSpec(BaseModel):
    out_dim: int
    use_spec_norm: bool = False
    use_bias: bool =False# True

    def build_linear(self, in_dim: int) -> nn.Module:
        lin = nn.Linear(in_dim, self.out_dim, bias=self.use_bias)
        if self.use_spec_norm:
            lin = spectral_norm(lin)
        return lin
#endregion LinearSpec

#region BlockSpec
class BlockSpec(BaseModel):
    pre: LinearSpec | ConvSpec | FixedSpec
    act: ActSetup | Act
    norm: Literal["layer", "rms", "none"] | None = None
    drop_out: float | None = None
    post: LinearSpec | ConvSpec | FixedSpec | None = None

    @property
    def out_dim(self) -> int:
        """The block's final output width."""
        return self.post.out_dim if self.post is not None else self.pre.out_dim
 
    @model_validator(mode='before')
    @classmethod
    def _promote_activation(cls, data: dict) -> dict:
        act = data.get('act')
        if isinstance(act, Act):
            data['act'] = ActSetup( act_type=act, dim=data.get('out_dim'), channels=data.get('channels'), )
        return data
    
    def build_block(self, in_dim: int, channels: int) -> nn.Module:
        layers: list[nn.Module] = []
        if isinstance(self.pre, FixedSpec):
            entry1 = self.pre.build_fixed()
        elif isinstance(self.pre, LinearSpec):
            entry1 = self.pre.build_linear(in_dim)
        elif isinstance(self.pre, ConvSpec):
            entry1 = self.pre.build_conv() 
        else:
            raise ValueError(f"Unknown pre layer spec type: {type(self.pre)}")
        layers.append(entry1)

        if self.norm is not None:
            if self.norm == "layer":
                layers.append(nn.LayerNorm(self.pre.out_dim, elementwise_affine=False)) 
            elif self.norm == "rms":
                layers.append(nn.RMSNorm(self.pre.out_dim, elementwise_affine=False)) 
            elif self.norm == "none":
                pass
        
        act_cfg: ActSetup = (
            self.act if isinstance(self.act, ActSetup)
            else ActSetup(act_type=self.act)
        ).finalized(dim=self.pre.out_dim, channels=channels)

        layers.append(act_cfg.build())
        
        if self.drop_out is not None and self.drop_out > 0.0:
            layers.append(nn.Dropout(self.drop_out))
        
        # optional post linear: pre.out_dim -> post.out_dim
        if self.post is not None:
            if isinstance(self.post, FixedSpec):
                entry2 = self.post.build_fixed()
            elif isinstance(self.post, LinearSpec):
                entry2 = self.post.build_linear(self.pre.out_dim)
            elif isinstance(self.post, ConvSpec):
                entry2 = self.post.build_conv()
            else:
                raise ValueError(f"Unknown post layer spec type: {type(self.post)}")
            layers.append(entry2)

        return nn.Sequential(*layers)
#endregion BlockSpec

#region HiddenSchema  
class HiddenSchema(BaseModel):
    """
    Builds an MLP mapping (B, C, in_dim) → (B, C, hid_dim).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    channels: int
    in_dim: int
    hid_dim: int
    out_dim: int  # optional bookkeeping for downstream; not used by build() 
    
    architecture: list[BlockSpec] | None = None
    use_conv: bool = False # False
    use_params: bool = False # True
    device: str = 'cuda' 
  
    @model_validator(mode="after")
    def build_default_architecture(self) -> "HiddenSchema":
        # Default: in -> hid, then hid -> hid, hid -> hid
        if self.architecture is None:
            self.architecture = [
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActSetup(act_type=Act.IDENTITY, elu_alpha=0.5), norm='none', drop_out=0.0),  # IDENTITY
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActSetup(act_type=Act.ELU, elu_alpha=1.0), norm='none', drop_out=0.0), 
                # BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActSetup(act_type=Act.RESBLOCK, elu_alpha=1.0), norm='none', drop_out=0.0),
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActSetup(act_type=Act.ELU, softshrink_lambd=0.07), norm='none', drop_out=0.0),  
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActSetup(act_type=Act.LOGSIGMOID, elu_alpha=1.0), norm='none', drop_out=0.0),
                BlockSpec(pre=LinearSpec(out_dim=self.hid_dim), act=ActSetup(act_type=Act.ELU, elu_alpha=1.0), norm='none', drop_out=0.0,) 
            ]
            assert len(self.architecture) >= 1, "Hidden architecture must have ≥1 block"
            
            first = self.architecture[0]
            assert first.pre.out_dim == self.hid_dim, f"""first block must map to hid_dim={self.hid_dim}, 
                got {first.pre.out_dim}"""

            for i, blk in enumerate(self.architecture[1:], start=1):
                assert blk.out_dim == self.hid_dim, f"""block {i} must preserve hid_dim={self.hid_dim},
                    got out_dim={blk.out_dim}"""
        return self
 

    def build_heads(self) -> nn.Sequential:
        layers = []
        in_w = self.in_dim
        for blk in self.architecture:
            seq = blk.build_block(in_dim=in_w, channels=self.channels)
            layers.append(seq)
            in_w = blk.out_dim                  # <- becomes hid_dim after block 1
        
        return nn.Sequential(*layers)
    
    @classmethod 
    def hidden_schema_from_cfg_and_id(cls, cfg: "FERNConfig", hid_id: "HidID") -> "HiddenSchema":
        return HiddenSchema(
            channels=cfg.channels,
            in_dim=cfg.dim_of(hid_id.src),
            out_dim=cfg.dim_hidden,
            hid_dim=cfg.dim_hidden,
            device=cfg.device,
        )

    def build_factory(self) -> nn.Module:
        from study.fr_gen import HiddenFactory
        return HiddenFactory(schema=self)
#endregion HiddenSchema


#region HeadSpec
class HeadSpec(BaseModel):
    """One coefficient head = core block."""
    core: list[BlockSpec]                   # H -> out_elems 
     
    @classmethod
    def scale_default(cls, out_elems: int, channels: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup(act_type=Act.ELU),),
                BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup( act_type=Act.ELU, elu_alpha=1.0),)
            ], 
        )
    
    @classmethod
    def scale_fixed_default(cls, out_elems: int, channels: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=FixedSpec(out_dim=out_elems, channels=channels),act=ActSetup(act_type=Act.ELU),norm='none', drop_out=0.0,), 
            ], 
        )
        
    @classmethod
    def complex_scale_default(cls, out_elems: int, channels: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup(act_type=Act.ELU),),
                BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup( act_type=Act.ELU), post=LinearSpec(out_dim=out_elems),),
            ], 
        )
    
    @classmethod
    def shift_default(cls, out_elems: int, channels: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup(act_type=Act.ELU), norm='none', drop_out=0.0, ),
                BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup(act_type=Act.ELU), norm='none', drop_out=0.0,post=LinearSpec(out_dim=out_elems), ),
            ], 
        )
    
    @classmethod
    def off_scale_default(cls, out_elems: int, channels: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup(act_type=Act.ELU,),post=LinearSpec(out_dim=out_elems),),
            ], 
        )
      
    @classmethod 
    def rotation_default(cls, out_elems: int, channels: int) -> "HeadSpec":
        return cls(
            core=[
                BlockSpec(pre=LinearSpec(out_dim=out_elems, use_spec_norm=False, use_bias=False),act=ActSetup(act_type=Act.RELU),),
                # BlockSpec(pre=LinearSpec(out_dim=out_elems),act=ActSetup(act_type=Act.ELU), post=LinearSpec(out_dim=out_elems),),
            ], 
        )
#endregion HeadSpec

#region StitchSpec
class StitchSpec(BaseModel):
    target_len: PositiveInt = Field(..., description="""
        What you WANT; The desired output length after stitching.
    """)
    patch_len: PositiveInt = 48
    stride: PositiveInt = 24 
    normalize: str = "avg"           # "avg" | "sum"
    window: Literal["ones", "hann"] = "ones"  # "ones" | "hann"
   
    @computed_field
    @property
    def overlap(self) -> int:
        return max(0, self.patch_len - self.stride)

    @computed_field
    @property
    def num_patches(self) -> int:
        # ceil(max(0, T-L)/S) + 1
        return math.ceil(max(0, self.target_len - self.patch_len) / self.stride) + 1

    @computed_field
    @property
    def covered_len(self) -> int:
        """
        The actual range on the timeline that the patches span when placed with the given stride.
        **Formula explanation:**
        - First patch covers `[0, patch_len)`
        - Each subsequent patch starts at `stride` offset from previous
        - Last patch starts at `(num_patches - 1) * stride`
        - Last patch ends at `(num_patches - 1) * stride + patch_len`

        Position:    0         24        48        72        96
                    |---------|---------|---------|---------|
        Patch 0:     [--------48--------]
        Patch 1:                [--------48--------]
        Patch 2:                          [--------48--------]

                    |<------------covered_len = 96----------->|

        Patch 0: position 0              → ends at 48
        Patch 1: position 0 + 1*stride   → ends at 24 + 48 = 72
        Patch 2: position 0 + 2*stride   → ends at 48 + 48 = 96
        """
        # (G-1)*S + L
        return (self.num_patches - 1) * self.stride + self.patch_len

    @computed_field
    @property
    def out_elems(self) -> int:
        """What you GENERATE: Total number of elements across all patches (flattened), 
        BEFORE stitching. This is the size of the tensor your model generates.
        Patch 0: 48 elements
        Patch 1: 48 elements  (24 overlap with patch 0)
        Patch 2: 48 elements  (24 overlap with patch 1)
        Total:   144 elements (with 48 elements of overlap)
        """ 
        return self.num_patches * self.patch_len # G * L

    @model_validator(mode="after")
    def _check(self) -> "StitchSpec": # REQUIRE: basic feasibility
        if self.target_len < self.patch_len:
            raise ValueError("target_len must be >= patch_len")
        if self.stride >= self.patch_len:
            raise ValueError("stride must be < patch_len")
        if self.covered_len < self.target_len:
            """covered_len must be >= target_len: 
            It ensures you have enough patches to reach target_len"""
            raise ValueError(f"covered_len={self.covered_len} < target_len={self.target_len}")
        return self
     
    @classmethod
    def from_dims(
        cls, *, out_dim: int,  patch_len: int,  stride: int,  
        normalize: str = "avg",
        window: Literal["ones", "hann"] = "ones",
    ) -> "StitchSpec":   
        return cls(target_len=out_dim, patch_len=patch_len, stride=stride, normalize=normalize, window=window)
#endregion StitchSpec

#region CoefSchema
class CoefSchema(BaseModel):   
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)  
    in_dim: int  # usually = hid_dim of HiddenSchema
    out_dim: int # target field dimension (e.g., seq_len/pred_len)
    hid_dim: int # explicit for clarity if needed elsewhere
    channels: int 
      
    head_specs: Dict[CoefName, HeadSpec] | None = None 
    scale_structure: ScaleStructure = ScaleStructure.DIAGONAL
     
    scale_structure: ScaleStructure = Field(..., description="diagonal or complex scale_structure")
    num_experts: PositiveInt = Field(1, description="number of mixture of experts")

    stitch: StitchSpec | None = None
      
    patch_spec: PatchSpec
    
    use_eigen_sparse: bool = False
    top_k_gaps: int = 10 

    @classmethod
    def coefschema_from_cfg_and_id(cls, cfg: "FERNConfig", coef_id: CoefID) -> "CoefSchema":
        # patch spec
        out_dim = cfg.dim_of(coef_id.dst)  
        if coef_id.dst == Space.Y:
            patch_spec = PatchSpec(L=out_dim, P=cfg.patch_size)
        else:
            patch_spec = PatchSpec(L=out_dim, P=1) # no patching for non-y spaces

        # === (Optional) stitch preset; leave None unless you want overlap-add
        stitch: StitchSpec | None = None
        
        # stitch = StitchSpec.from_dims(
        #     out_dim=cfg.dim_of(coef_id.dst), 
        #     patch_len=8, 
        #     stride=4
        #     )
 
        # === structure → required heads (use enum keys, not strings) 
        STRUCTURE_SPECS = {
            'diagonal': [
                (CoefName.SHIFT, HeadSpec.shift_default), 
                (CoefName.SCALE, HeadSpec.scale_default)
            ],
            'complex': [
                (CoefName.SHIFT, HeadSpec.shift_default), 
                (CoefName.SCALE, HeadSpec.complex_scale_default)
            ],
            'tri_sym': [
                (CoefName.SHIFT, HeadSpec.shift_default), 
                (CoefName.SCALE, HeadSpec.scale_default), 
                (CoefName.OFF_SCALE, HeadSpec.off_scale_default)
            ],
            'tri_anti': [
                (CoefName.SHIFT, HeadSpec.shift_default), 
                (CoefName.SCALE, HeadSpec.scale_default), 
                (CoefName.OFF_SCALE, HeadSpec.off_scale_default)
            ],
        } 
        required_specs = STRUCTURE_SPECS[coef_id.scale_structure]  

        head_specs = {}
        for spec in required_specs: 
            name, builder = spec  
            dst_dim = stitch.out_elems if stitch else out_dim
            dst_dim = dst_dim - 1 if name == CoefName.OFF_SCALE else dst_dim 
            head_specs[name] = builder(out_elems=dst_dim, channels=cfg.channels)
        
        # Validate
        missing = set(s[0] for s in required_specs) - set(head_specs.keys())
        if missing:
            raise ValueError(f"""Missing required heads {missing} for scale_structure '{coef_id.scale_structure}'""")
        

        return CoefSchema(
            # src=coef_id.src,
            # dst=coef_id.dst,
            channels=cfg.channels,
            in_dim=cfg.dim_hidden,
            out_dim=out_dim,
            hid_dim=cfg.dim_hidden,
            patch_spec=patch_spec,
            use_eigen_sparse=cfg.use_eigen_sparse,
            top_k_gaps=cfg.top_k_gaps,
 
            scale_structure=coef_id.scale_structure,
            num_experts=coef_id.num_experts,
            stitch=stitch,                       # CHG: pass it in
            head_specs=head_specs,               # CHG: pass it in
        ) 
      
    def build_heads(self) -> nn.ModuleDict:
        foundry = nn.ModuleDict()  

        assert self.head_specs is not None
        for head, spec in self.head_specs.items():  
            layers_core = []
            in_w = self.hid_dim
            for blk in spec.core:
                seq = blk.build_block(in_dim=in_w, channels=self.channels)
                layers_core.append(seq)
                in_w = blk.out_dim         
            foundry[head.name] = nn.Sequential(*layers_core)  

        return foundry 

    @computed_field
    @property
    def out_elems(self) -> int:
        """
        Raw width emitted by each head BEFORE stitch.
        - no stitch: equals out_dim
        - stitch   : equals num_patches * patch_len (concatenated raw width)
        """
        return self.stitch.out_elems if self.stitch else self.out_dim
       
    def build_factory(self) -> nn.Module:
        from study.fr_gen import CoefFactory
        return CoefFactory(schema=self)
#endregion CoefSchema

#region RotationSchema
class RotationSchema(BaseModel):
    """Schema for data-dependent Householder rotations."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    patch_spec: PatchSpec
  
    channels: int
    in_dim: int
    out_dim: int
    num_reflects: int  
    patch_spec: PatchSpec
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    block_size: int =8 

    head_specs: dict[CoefName, HeadSpec] = Field(..., description="Heads to produce rotation parameters.")

    @classmethod
    def rotation_schema_from_cfg_and_id(cls, cfg: "FERNConfig", rot_id: RotID) -> "RotationSchema":
        out_dim = cfg.dim_of(rot_id.dst)
        patch_spec = PatchSpec.from_len(L=out_dim, patch_size=cfg.patch_size)
        return RotationSchema(
            channels=cfg.channels,
            in_dim=cfg.dim_hidden,
            out_dim=out_dim,
            num_reflects=cfg.num_reflects, 
            patch_spec=patch_spec,
            device=cfg.device,
            dtype=torch.float32,
            block_size=cfg.block_size,
            head_specs={
                CoefName.ROTATION: HeadSpec.rotation_default(
                        out_elems=out_dim * cfg.num_reflects,
                        channels=cfg.channels,
                    )
                }
            )

    def build_rotation_heads(self) -> nn.ModuleDict:
        foundry = nn.ModuleDict()  

        assert self.head_specs is not None
        for head, spec in self.head_specs.items():  
            layers_core = []
            in_w = self.in_dim
            for blk in spec.core:
                seq = blk.build_block(in_dim=in_w, channels=self.channels)
                layers_core.append(seq)
                in_w = blk.out_dim
            foundry[head.name] = nn.Sequential(*layers_core)

        for m in foundry.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        return foundry 
 
    def build_factory(self) -> nn.Module:
        from study.fr_gen import RotateFactory  # avoid top-level circular import
        return RotateFactory(schema=self)
#endregion RotationSchema

#region GateSchema
class GateSchema(BaseModel):
    enabled: bool = True
    in_dim: int = Field(..., description="Input channel dim the MLP consumes (e.g., y channels).")
    out_dim: int = Field(..., description="Output channel dim (usually == in_dim for y->y).")
    hid_dim: int = Field(..., description="Hidden width; None -> single affine.")
    sharp: float = 8.0
    tau: float = 0.8

    def build_factory(self) -> nn.Module: 
        from study.fr_gen import GateFactory  # avoid top-level circular import
        return GateFactory(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            hidden=self.hid_dim,
            sharp=self.sharp,
            tau=self.tau,
        )
# endregion GateSchema