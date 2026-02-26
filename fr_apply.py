from enum import StrEnum, auto
from pydantic import ConfigDict, model_validator
from pydantic import BaseModel, Field
from typing import Union, Literal, List, Tuple, Optional, Sequence, Dict, Annotated
import torch
import torch.nn as nn
import study.fr_gen as fgen 
import sys
import study.fr_core as fcore  # needed for ModuleSpec synthesis


 


# region FlowRegistry
class FlowRegistry:
    def __init__(self, fac: nn.ModuleDict, specs: List[fcore.FlowSpec]):
        self.fac = fac
        self.cache: Dict[str, fgen.Op] = {}
        self.display_to_canonical: Dict[str, str] = {}
        seen_disp = set()
        for spec in specs:
            if spec.name in seen_disp:
                raise ValueError(f"Duplicate display name in flow: {spec.name}")
            seen_disp.add(spec.name) 
            op = compile_flow(spec, fac)
            disp = spec.name                 # e.g., "X_to_Z_V0"
            canon = canonical_flow_key(spec) # e.g., "FLOW:SCALE_SHIFT:Z<-X:v=v0"

            # [ADD] index by BOTH keys
            self.cache[canon] = op
            self.cache[disp]  = op
            self.display_to_canonical[disp] = canon

    def op(self, name: str) -> fgen.Op:
        # Programmatic: op = registry.op("Z_to_Y_V4") is a runnable Op
        return self.cache[name]

    def apply(self, states: fgen.States, sequence: List[str]) -> fgen.States:
        # FLOW → recipe via FLOW_TEMPLATES, recipe → op via compile_chain
        for name in sequence:
            states = states | self.op(name)
        return states
    
def canonical_flow_key(spec: fcore.FlowSpec) -> str:
    """Canonical, typed key for a flow (not used for lookup yet)."""
    return fcore.SpecKey.from_flow(spec).key()
# endregion FlowRegistry

# region StrEnum
class ParamKey(StrEnum):
    """Abstract names for which parameter head you need from your factory dict."""
    HID = "HID"  # hidden from src
    SS = "SS"  # scale/shift params
    ROT = "ROT"  # rotation params
# endregion StrEnum


# region G and T 

class G(fgen.Op):
    field: fcore.Space 
    fac: fgen.CoefFactory | fgen.RotateFactory | fgen.HiddenFactory = None
    augment: Literal[
        "none", "odd", "odd_odds", "odd_evens",
        "even", "even_evens", "even_odds", 
        "boot_odd_keep_even", "boot_even_keep_odd",
    ] = "none"

    def apply(self, state: fgen.States) -> fgen.States:
        source = getattr(state, self.field)
        if self.augment == "boot_odd_keep_even":
            source = bootstrap_odds_keep_evens(source)
        elif self.augment == "boot_even_keep_odd":
            source = bootstrap_even_keep_odd(source)
        elif self.augment != "none":
            source = keep_positions(x=source, keep_mode=self.augment)

        if isinstance(self.fac, fgen.CoefFactory):
            scale, shift = self.fac(source)  # unpack (h)
            state.set_scale(scale)
            state.set_shift(shift)
            return state
        elif isinstance(self.fac, fgen.RotateFactory):
            rotation = self.fac(source)
            state.set_rotation(rotation)
            return state
        elif isinstance(self.fac, fgen.HiddenFactory):
            h = self.fac(source)
            state.h = h
            return state
        elif isinstance(self.fac, fgen.GateFactory):
            gate = self.fac(source)  # Gate
            # optional: store both logits and probs for logging
            state.gate = gate
            state.gate_prob = gate.prob().detach()
            return state

        else:
            raise ValueError(f"Unknown factory: {type(self.fac)}")


class TOp(StrEnum):
    """The actions your T(...) op can apply to the destination field."""
    ROTATION = auto() ; SCALE = auto() ; SHIFT = auto() ; GATE_HARD = auto() ; 


class T(fgen.Op):
    """APPLIES a pre-generated operator from the States to a data field."""
    field: fcore.Space
    op_name: TOp
    # inverse: bool = False  # Whether to apply the inverse operation
    is_inverse: bool = Field(
        False,
        alias="inverse",                 # input/output key
        validation_alias="inverse",      # accept 'inverse' on load
        serialization_alias="inverse",   # emit 'inverse' on dump
        description="apply inverse operation; aliased as 'inverse' in I/O",
    )

    def apply(self, state: fgen.States) -> fgen.States:
        if self.op_name == TOp.GATE_HARD:
            gate = state.gate
            if gate is None:
                raise RuntimeError("Gate not set in state before T(GATE_HARD).")
            # Terminal hard mask: exact 0-or-identity (no residual)
            val = getattr(state, self.field)
            out = gate.apply(val)
            setattr(state, self.field, out)
            return state

        op = getattr(state, self.op_name)
        val  = getattr(state, self.field)
        temp_out = val | op

        if self.op_name in [TOp.SCALE]: 
            temp_out = temp_out + val
            out = temp_out
            # final_target = new_target + target  # TODO + 1e-5
        else:
            out = temp_out
        setattr(state, self.field, out)
        return state
# endregion G and T
class I(fgen.Op):
    """Identity operator"""
    def apply(self, state):  # just returns the states unchanged
        return state


# region ProbG
class ProbG(fgen.Op):
    """A Op operator that probabilistically chooses one of several G operators to apply.
    A Op operator that probabilistically chooses one of several G operators to apply.
    During evaluation (is_training=False), it always chooses the first operator in the list.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    choices: List[Tuple[G, float]]

    @model_validator(mode="after")
    def validate_probabilities(self) -> "ProbabilisticG":
        """Ensures the probabilities sum to 1.0."""
        total_prob = sum(prob for _, prob in self.choices)
        if not torch.isclose(torch.tensor(total_prob), torch.tensor(1.0)):
            raise ValueError(f"Probabilities must sum to 1.0, but got {total_prob}")
        return self

    def apply(self, state: fgen.States) -> fgen.States:
        """
        Randomly selects and applies a G operator during training.
        Applies the first G operator during evaluation.
        """
        # During inference/evaluation, always use the first (default) option for reproducibility.
        if not state.is_training:
            default_op, _ = self.choices[0]
            return default_op.apply(state)

        # During training, randomly select an operator based on the probabilities.
        p = torch.rand(1).item()
        cumulative_prob = 0.0
        for g_op, prob in self.choices:
            cumulative_prob += prob
            if p < cumulative_prob:
                return g_op.apply(state)

        # Fallback to the last operator in case of floating point inaccuracies
        last_op, _ = self.choices[-1]
        return last_op.apply(state)
# endregion ProbG

# def specialize_prob(spec: ChainSpec, p):
#     steps = list(spec.steps)
#     steps[0] = steps[0].model_copy(update={"type": "ProbG", "probg_choices": p})
#     return ChainSpec(name=f"{spec.name}(custom)", steps=steps, version=spec.version, augment=spec.augment)
"""
usage: 
flows["X_to_Z_PROB"] = fapply.compile_chain(
    specialize_prob(fapply.FLOW_TEMPLATES[fcore.FlowType.SCALE_SHIFT],
                    [("none",0.4), ("boot_odd_keep_even",0.3), ("even",0.3)]),
    src="x", dst="z", fac=self.NNs, version="v0",
)
"""

 
# region GStep and TStep
class BaseStep(BaseModel):
    model_config = ConfigDict()

class GStep(BaseStep):
    """GStep (“generate”) Says: “Run a generator G(...) on either the source field ('src')
    or the hidden field ('h'), using a particular ParamKey.”
    "src" means: use the source field string passed at compile time
    "h" means: use the hidden real field 
    """
    type: Literal["G", "ProbG"] = "G"
    field: Literal["src", "h"]
    param_key: ParamKey
    
    ss_version: str | None = None
    rot_version: str | None = None
    hid_version: str | None = None 
    
    probg_choices: List[Tuple[str, float]] = None

    @model_validator(mode="after")
    def validate_probg_choices(self) -> "GStep":
        if self.type == "ProbG":
            if self.probg_choices is None:
                self.probg_choices = [
                    ("none", 0.5),  # boot_odd_keep_even
                    ("boot_odd_keep_even", 0.25),  # boot_odd_keep_even # odd
                    ("even", 0.25),  # boot_odd_keep_even
                ]
        return self

class TStep(BaseStep):
    """TStep (“transform”) Says: “Apply a transform T(...) on the destination field ('dst')
    with a given TOp, optionally with inverse=True (only valid for rotation).”"""
    type: Literal["T"] = "T"
    field: Literal["dst"]  # we always transform the destination field
    op_name: TOp
    inverse: bool = False
# endregion GStep and TStep

# region StepSpec and ChainSpec
StepSpec = Union[GStep, TStep]

class ChainSpec(BaseModel):
    """
    A Pydantic model that wraps:
    name: for logging,
    steps: a list of GStep | TStep,
    optional defaults: version and augment (so you can omit them at call sites).
    """ 
    name: str
    steps: List[StepSpec]  
    version: Optional[str] = None # Optional compile-time defaults (don’t have to set here)
    augment: Literal[
        "none", "odd", "even", "even_evens", "even_odds",
        "odd_odds", "odd_evens", "boot_odd_keep_even",
    ] = "none"
# endregion StepSpec and ChainSpec


# region Key builders
def _factory_key(param_key: "ParamKey", *, src: fcore.Space, dst: fcore.Space, version: str) -> str:
    """
    Emit canonical factory keys that match FERNConfig.build_factories() outputs.
    - HID: HID_GIVEN_{SRC}_{V}
    - SS : SS_{DST}_GIVEN_{SRC}_{V}
    - ROT: ROT_IN_{DST}_{V}
    """  
    if param_key is ParamKey.HID: # hidden lives in H, built from SRC
        return fcore.HidID(src=src, dst=fcore.Space.H, version=version).module_key
    if param_key is ParamKey.SS: # coef maps SRC -> DST 
        return fcore.CoefID.from_src_dst(src=src, dst=dst, version=version).module_key
    if param_key is ParamKey.ROT: # rotation is in-place in DST-space
        return fcore.RotID(src=dst, dst=dst, version=version).module_key
    raise KeyError(f"Unknown ParamKey: {param_key}")


def pick_ver(param_key: ParamKey, step: GStep, default: str | None) -> str:
    if param_key is ParamKey.SS:
        return step.ss_version or default or "v0"
    if param_key is ParamKey.ROT:
        return step.rot_version or default or "v0"
    if param_key is ParamKey.HID:
        return step.hid_version or default or "v0"
# endregion Key builders


# region compile_chain
# --- compile_chain: ChainSpec -> fgen.Op ---
def compile_chain(
    spec: ChainSpec, *, src: fcore.Space, dst: fcore.Space, fac: "nn.ModuleDict",
    version: str | None = None,  # chain-level default
    augment: Optional[str] = None,
) -> "fgen.Op": 
    """ Takes a ChainSpec + runtime context and returns executable plan:
    Start with plan = I().
    For each step:
    GStep: resolve field ('src' → actual src, otherwise 'h'), compute the factory key from ParamKey, and append G(...).
    Augment is applied only when field == src (your current design).
    TStep: append T(field=dst, op_name=..., inverse=...).
    Return the composed fgen.Op. """
    plan = I()
    aug = augment if augment is not None else spec.augment

    for step in spec.steps:
        if step.type == "G" or step.type == "ProbG":
            field = src if step.field == "src" else "h"
            v = pick_ver(step.param_key, step, version)
            key = _factory_key(step.param_key, src=src, dst=dst, version=v)
            if field == src:
                if step.type == "G":
                    plan = plan | G(field=field, fac=fac[key], augment=aug)
                elif step.type == "ProbG":
                    plan = plan | ProbG(
                        choices=[
                            (G(field=field, fac=fac[key], augment=aug), prob)
                            for aug, prob in step.probg_choices
                        ]
                    )
            else:
                plan = plan | G(field=field, fac=fac[key])
        elif step.type == "T":  # T
            assert step.field == "dst"
            plan = plan | (
                T(field=dst, op_name=step.op_name, inverse=True)
                if step.inverse
                else T(field=dst, op_name=step.op_name)
            )
        elif step.type == "Print":  # Print
            plan = plan | CoefMonitorOp(
                message=step.message,
                coef=step.coef,
                use_processed=step.use_processed,
                bins=step.bins,
            )
    return plan
# endregion Compiler

# region FLOW BUILDER
# fcore.FlowType -> ChainSpec registry (one-liners)
def compile_flow(flowspec: fcore.FlowSpec, fac: nn.ModuleDict) -> fgen.Op:
    return compile_chain(
        FLOW_TEMPLATES[flowspec.flowtype], 
        src=flowspec.src, dst=flowspec.dst, fac=fac, version=flowspec.version)
    
    
"""A dict mapping your fcore.FlowType enum → ChainSpec."""
FLOW_TEMPLATES: dict[fcore.FlowType, ChainSpec] = {
    fcore.FlowType.SCALE_SHIFT: ChainSpec(
        name="SCALE_SHIFT",
        steps=[
            GStep(field="src", param_key=ParamKey.HID, hid_version="v0"), 
            GStep(field="h", param_key=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SCALE),
            TStep(field="dst", op_name=TOp.SHIFT),  # TODO
        ],
    ),
    
    fcore.FlowType.SCALE_ONLY: ChainSpec(
        name="SCALE_ONLY",
        steps=[
            GStep(field="src", param_key=ParamKey.HID, hid_version="v0"),
            GStep(field="h", param_key=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SCALE),
        ],
    ),
    fcore.FlowType.SHIFT_ONLY: ChainSpec(
        name="SHIFT_ONLY",
        steps=[
            GStep(field="src", param_key=ParamKey.HID, hid_version="v0"),
            GStep(field="h", param_key=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SHIFT),
        ],
    ),
    fcore.FlowType.R_SCALE_RB_SHIFT: ChainSpec(
        name="R_SCALE_RB_SHIFT",
        steps=[
            GStep(field="src", param_key=ParamKey.HID, hid_version="v0", type="G"),  # ProbG
            GStep(field="h", param_key=ParamKey.SS),
            GStep(field="h", param_key=ParamKey.ROT, rot_version="v0"),
            
            TStep(field="dst", op_name=TOp.ROTATION),
            TStep(field="dst", op_name=TOp.SCALE),
            TStep(field="dst", op_name=TOp.ROTATION, inverse=True),
            TStep(field="dst", op_name=TOp.SHIFT),
        ],
    ),
    fcore.FlowType.R_SCALE_SHIFT: ChainSpec(
        name="R_SCALE_SHIFT",
        steps=[
            GStep(field="src", param_key=ParamKey.HID, hid_version="v0"),
            GStep(field="h", param_key=ParamKey.SS),
            GStep(field="h", param_key=ParamKey.ROT, rot_version="v0"),
            TStep(field="dst", op_name=TOp.ROTATION),
            TStep(field="dst", op_name=TOp.SCALE),
            TStep(field="dst", op_name=TOp.SHIFT),
        ],
    ),
    fcore.FlowType.R: ChainSpec(
        name="R",
        steps=[
            GStep(field="src", param_key=ParamKey.HID, hid_version="v0"),
            GStep(field="h", param_key=ParamKey.ROT),
            TStep(field="dst", op_name=TOp.ROTATION),
        ],
    ),
    fcore.FlowType.PROB_SCALE_SHIFT: ChainSpec(
        name="PROB_SCALE_SHIFT",
        steps=[
            GStep(field="src", param_key=ParamKey.HID, hid_version="v0", type="ProbG",
                probg_choices=[("none", 0.4), ("boot_odd_keep_even", 0.3), ("even", 0.3)]),
            GStep(field="h", param_key=ParamKey.SS),
            TStep(field="dst", op_name=TOp.SCALE),
            TStep(field="dst", op_name=TOp.SHIFT),
        ],
    ), 
}
# endregion FLOW BUILDER

# region bootstrap
# -------------------------------
# region bootstrap
def _indices_from_mode(T: int, mode: str, device: torch.device) -> torch.Tensor:
    if mode not in _MODE_TO_SLICE:
        raise ValueError(f"Unknown mode '{mode}'. Valid: {list(_MODE_TO_SLICE.keys())}")
    start, step = _MODE_TO_SLICE[mode]
    return torch.arange(start, T, step, device=device)


def _complement_indices(T: int, keep_idx: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(T, dtype=torch.bool, device=keep_idx.device)
    mask[keep_idx] = False
    return mask.nonzero(as_tuple=False).squeeze(-1)


_MODE_TO_SLICE = {
    "odd": (1, 2),  # 1,3,5,...
    "even": (0, 2),  # 0,2,4,...
    "even_evens": (0, 4),  # 0,4,8,...
    "even_odds": (2, 4),  # 2,6,10,...
    "odd_evens": (1, 4),  # 1,5,9,...
    "odd_odds": (3, 4),  # 3,7,11,...
}


def make_keep_mask(x: torch.Tensor, start: int, step: int) -> torch.Tensor:
    """Binary mask with ones at positions [start::step] along the last dim."""
    if not isinstance(start, int) or not isinstance(step, int) or step <= 0:
        raise ValueError("`start` must be int and `step` a positive int.")
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., start::step] = True
    return mask

# region bootstrap functions
def keep_positions(
    x: torch.Tensor,
    keep_mode: Literal[
        "odd", "even", "even_evens", "even_odds", "odd_odds", "odd_evens"
    ],
    return_mask: bool = False,
) -> torch.Tensor:
    """Keep ONLY the selected positions; others are zeroed. Optionally return mask."""
    if keep_mode not in _MODE_TO_SLICE:
        raise ValueError(f"keep_mode must be one of {list(_MODE_TO_SLICE)}")
    start, step = _MODE_TO_SLICE[keep_mode]
    mask = make_keep_mask(x, start, step)
    if return_mask:
        return mask
    return x * mask.to(dtype=x.dtype)


def drop_positions(
    x: torch.Tensor,
    drop_mode: Literal[
        "odd", "even", "even_evens", "even_odds", "odd_odds", "odd_evens"
    ],
) -> torch.Tensor:
    """Zero OUT the selected positions; keep the rest."""
    if drop_mode not in _MODE_TO_SLICE:
        raise ValueError(f"drop_mode must be one of {list(_MODE_TO_SLICE)}")
    start, step = _MODE_TO_SLICE[drop_mode]
    y = x.clone()
    y[..., start::step] = 0
    return y
# endregion bootstrap functions

def bootstrap_keep(
    x: torch.Tensor,
    *,
    keep_mode: str = "even",
    resample_from: Optional[str] = None,  # e.g. 'odd'; default = complement of keep_mode
    generator: Optional[torch.Generator] = None,
    patch_size: int = 24,
) -> torch.Tensor:
    """
    Bootstrap selected positions along the last dim, keeping others unchanged.

    Args:
        x: Tensor of shape (..., T). Works with any dtype/device.
        keep_mode: which positions to keep as-is (see _MODE_TO_SLICE).
        resample_from: which positions form the bootstrap pool and are
            *also* the positions to be replaced. If None, use the complement
            of keep_mode. Typical pairs: keep='even', resample_from='odd'.
        generator: optional torch.Generator for reproducible sampling.

    Returns:
        y: same shape as x. Positions in `keep_mode` are copied from x.
           Positions in `resample_from` are resampled (with replacement)
           from x restricted to that same set.
    """
    *lead, T = x.shape
    device = x.device

    keep_idx = _indices_from_mode(T, keep_mode, device)

    if resample_from is None:
        res_idx = _complement_indices(T, keep_idx)  # default: complement
    else:
        res_idx = _indices_from_mode(T, resample_from, device)

    if res_idx.numel() == 0:
        return x.clone()
    
    spec = fcore.PatchSpec.from_len(T, patch_size=patch_size)
    G, P = spec.G, spec.P

    y = x.clone() 

    # 3) Iterate over patches g = 0..G-1 and resample only inside each patch.
    for g in range(G):
        start = g * P
        end   = start + P
        # res_idx holds the indices to be replaced (e.g., “odd” or “even” positions).
        # Here we restrict to those that lie inside the current patch slice.
        block_res_idx = res_idx[(res_idx >= start) & (res_idx < end)]
        # If there is nothing to replace in this patch, skip to the next one.
        if block_res_idx.numel() == 0:
            continue
        
        # 4) Build the sampling pool: the values at the positions we will replace.
        #    This keeps the bootstrap local to the patch.
        pool = x.index_select(-1, block_res_idx)  # (..., K) 
        
        # 5) Draw K indices with replacement from 0..K-1, then gather those values.
        K = block_res_idx.numel()
        draw = torch.randint(0, K, (K,), device=device, generator=generator)
        boot = pool.index_select(-1, draw)  # (..., K)

        # 6) Write the resampled values back into y at the original indices.
        y.index_copy_(-1, block_res_idx, boot)

    return y


# Convenience alias matching your original function:
def bootstrap_odds_keep_evens(x: torch.Tensor) -> torch.Tensor:
    return bootstrap_keep(x, keep_mode="even", resample_from="odd")


def bootstrap_even_keep_odd(x: torch.Tensor) -> torch.Tensor:
    return bootstrap_keep(x, keep_mode="odd", resample_from="even")
# endregion bootstrap