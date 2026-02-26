
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from typing import Dict, Optional, List, Literal, Union, Any, Mapping, Sequence, TYPE_CHECKING
from typing_extensions import Self
from pydantic import (
    BaseModel, Field, computed_field, model_validator, ConfigDict,
)
from enum import Enum, StrEnum, auto 
from functools import cached_property  
import math
import study.fr_core as fcore
import study.fr_apply as fapply
from functools import partial 
from typing import TypedDict, Required, NotRequired, Self,TYPE_CHECKING
from typing_extensions import Unpack  # Py <3.12 



#region BaseTrainingConfig 
class MetricSpec(BaseModel):
    """intent: weights + params for one metric; [no guards besides ge=0]"""
    weight_backward: float = Field(default=0.0, ge=0.0)   # CHG: replace nonneg validator
    weight_validate: float = Field(default=0.0, ge=0.0)
    params: Dict[str, Any] = Field(default_factory=dict)

def _default_metric_specs() -> dict[str, MetricSpec]:
    """intent: default per-metric weights/params; returns dict[name->MetricSpec]"""
    return {
        "mse":        MetricSpec(weight_backward=0.0, weight_validate=0.1),
        "mae":        MetricSpec(weight_backward=0.0, weight_validate=1.0),
        "huber":      MetricSpec(weight_backward=1.0, weight_validate=0.0, params={"delta": 1.0}),
        "swd":        MetricSpec(weight_backward=0.0, weight_validate=0.1, params={"return_type": "sq_swd2"}),
        "ept":        MetricSpec(weight_backward=0.0, weight_validate=0.0),
        "quantile70": MetricSpec(weight_backward=0.0, weight_validate=0.0, params={"quantile": 0.70}),
        "quantile30": MetricSpec(weight_backward=0.0, weight_validate=0.0, params={"quantile": 0.30}),
    }

class BaseTrainingConfig(BaseModel):
    """A Pydantic model for shared, common training parameters."""
    # --- Essential Identifiers ---
    model_config = ConfigDict(validate_assignment=True, extra="forbid", arbitrary_types_allowed=True)

    seq_len: int
    pred_len: int
    label_len: int = 0
    channels: int
    batch_size: int = 128
    channel_independent: bool = False
    use_proj_swd: bool = False
    amp_enabled: bool = False
    
    # --- Core Training Hyperparameters ---
    learning_rate: float = 3e-4
    epochs: int = 50
    patience: int = 5
    device: str = "cuda"
    # dtype: torch.dtype = torch.float32

    num_proj_swd: int = 500
    seeds: List[int] = Field(..., description="List of random seeds for training runs.",)
    task_name: str = "long_term_forecast"

    scheduler_type: Literal["none", "plateau", "cosine"] = Field(
        default="none",
        description="Type of LR scheduler to use ('none', 'plateau', 'cosine').",)
    
    lr_scheduler_patience: int = Field(default=2,description="Patience (in epochs) for ReduceLROnPlateau before reducing LR.",)
    lr_scheduler_factor: float = Field(default=0.7,description="Factor by which to reduce LR for ReduceLROnPlateau (new_lr = lr * factor).",)
    warmup_epochs: int = Field(default=3,description="Number of epochs for warmup.",)
    eta_min: float = Field(default=1e-5,description="Minimum LR for cosine scheduler.",)

    # --- Losses ---
    metric_configs: dict[str, "MetricSpec"] = Field(default_factory=_default_metric_specs)

    aux_loss_weights: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_common(cls, *, seq_len: int, pred_len: int, channels: int, seeds: List[int], **specifics: Any) -> Self: 
        # CHG: ignore any accidental attempts to override metrics via factory args
        if "metric_configs" in specifics:
            # GUARD-OK: keep metric source of truth inside config construction
            specifics.pop("metric_configs")
        # Start from required keys, layer overrides on top; Pydantic validates once.
        base = dict(seq_len=seq_len, pred_len=pred_len, channels=channels, seeds=seeds)
        return cls(**{**base, **specifics})
    
    # optional convenience when you *explicitly* want to change metrics:
    def with_metrics(self, metric_configs: dict[str, "MetricSpec"]) -> "BaseTrainingConfig":
        """intent: explicit API to replace metric set; returns a new config"""
        return self.model_copy(update={"metric_configs": metric_configs}, deep=True)

    @model_validator(mode="after")
    def _print_CI(self) -> "BaseTrainingConfig":
        if self.channel_independent:
            print(f"BaseTrainingConfig: channel_independent is True, it is {self.channel_independent}")
        return self

"""
usage
# minimal + a couple of overrides
fr = FERNConfig.from_common(seq_len=96, pred_len=192, channels=7,
                             dim_hidden=192, patience=8, seeds=[7, 1955, 2023, 4])

# later tweak safely
fr2 = fr.with_overrides(learning_rate=1e-3, epochs=80)
"""

#  "dim_hidden": 144,
#     "dim_augment": 144, 
#     "num_reflects": 8,
# endregion

DIM_BY_LETTER = {"x": "seq_len",   # length of x-series
                 "y": "pred_len",  # horizon
                 "z": "dim_augment"}
 
SCHEMA_BY_FLAVOUR = {"ss": fcore.CoefSchema, 
                     "hid": fcore.HiddenSchema, 
                    "rot": fcore.RotationSchema,
                     }

#region FERNConfig
class FERNConfig(BaseTrainingConfig):
    """FLAT configuration for FERN model. Contains ALL base and specific fields."""
    # --- Meta ---
    model_type: Literal["FERN"] = "FERN"
    forward_signature: Literal["x,update"] = "x,update"
    output_signature: Literal["tensor","dict:pred"] = "dict:pred"

    # --- FERN Specific Fields ---
    dim_augment: int = 128
    dim_hidden: int = 128 
    num_reflects: int = Field(
        8, description=""" number of Householder reflectors to use for data rotation; Note that ANY reflections 
        for an e.g. dim=24 patch can be decomposed into at most 24 reflectors.
    """)
    block_size: int = Field(
        8, description=""" block size of Householder reflectors with Gradient: if block_size = num_reflects,
        all reflectors are trainable; if block_size < num_reflects, the reflections are divided into 
        blocks: during training, one block has gradient, the rest don't. This is implemented to cut backprop memory cost
        to enable larger number of reflectors. 
    """)

    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    factory_schemas: Dict[str, Union[fcore.CoefSchema, fcore.HiddenSchema, fcore.RotationSchema]] = Field(
        default_factory=dict, repr=False 
    )
    
    log_train_coef_stats: bool = False  # Master toggle
    log_train_coef_prob: float = 0.1   # Sampling probability (0.0 to 1.0) 
 
    use_eigen_sparse: bool = False
    top_k_gaps: int = 10

    patch_size: int = Field(24, description="patch size for patch-wise coef") #TODO 

    

    @computed_field
    @property
    def flow_phases(self) -> Dict[str, List[fcore.FlowSpec]]:
        return FERNConfig.default_flow_phases()

    @computed_field
    @property
    def patch_spec(self) -> fcore.PatchSpec:
        """intent: derive spec from pred_len & patch_len; REQUIRE: pred_len % patch_len == 0"""
        return fcore.PatchSpec.from_len(L=self.pred_len, patch_size=self.patch_size)

    @computed_field(return_type=int)
    @property
    def module_spec(self) -> List[fcore.ModuleSpec]:
        return fcore.modulespecs_from_flowspecs(FERNConfig.flatten_flow_phases(self.flow_phases))
 
    def dim_of(self, field: fcore.Space) -> int:
        return {
            fcore.Space.X: self.seq_len,
            fcore.Space.Y: self.pred_len,
            fcore.Space.Z: self.dim_augment,
            fcore.Space.H: self.dim_hidden,
        }[field]
 
    def with_overrides(self, **overrides: Any) -> "FERNConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)
      
    def build_factories(self) -> nn.ModuleDict:
        factories = nn.ModuleDict() 
        ordered = sorted(self.module_spec, key=lambda b: b.module_key) # [ADD] stable order by canonical key
        for spec in ordered:
            spec.add_to_ModuleDict(self, factories) 
        return factories
    
    @staticmethod
    def flatten_flow_phases(flow_phases: Dict[str, List[fcore.FlowSpec]]) -> List[fcore.FlowSpec]:
        out: List[fcore.FlowSpec] = []
        for phase in ("encoding", "processing", "decoding"):
            out.extend(flow_phases.get(phase, []))
        return out

    @staticmethod
    def add_bidir(lst, src, dst, geom_sd, geom_ds, versions):
        for v in versions:
            lst.append(fcore.FlowSpec(src=src, dst=dst, flowtype=geom_sd, version=v))
            lst.append(fcore.FlowSpec(src=dst, dst=src, flowtype=geom_ds, version=v))

    @staticmethod
    def default_flow_phases() -> Dict[str, List[fcore.FlowSpec]]:
        # [ADD] default flow program (mirrors your fr.py sequences)
        enc: List[fcore.FlowSpec] = [] 
        proc: List[fcore.FlowSpec] = []
        dec: List[fcore.FlowSpec] = []
        # X ↔ Z for v0, v2, v3, v4, v5
        FERNConfig.add_bidir(enc, fcore.Space.X,fcore.Space.Z, fcore.FlowType.SCALE_SHIFT, 
                fcore.FlowType.SCALE_SHIFT, 
                ["v0","v2","v3","v4","v5"])
        # Y ↔ Z for v2, v3, v4 (dst→src is SHIFT_ONLY per your current intent)
        FERNConfig.add_bidir(proc, fcore.Space.Y,fcore.Space.Z, 
                fcore.FlowType.SCALE_SHIFT, 
                fcore.FlowType.SHIFT_ONLY, 
                ["v2","v3","v4"])
        FERNConfig.add_bidir(dec, fcore.Space.Y,fcore.Space.Z,
                fcore.FlowType.SCALE_SHIFT, 
                fcore.FlowType.SHIFT_ONLY,
                [  "v5"], # "v4",
                )

        # Equivalent to:
        # proc: List[fcore.FlowSpec] = [
        #     fcore.FlowSpec(src="z", dst="y", flowtype=fcore.FlowType.SCALE_SHIFT, version="v2"),
        #     fcore.FlowSpec(src="y", dst="z", flowtype=fcore.FlowType.SHIFT_ONLY,  version="v2"),
        #     fcore.FlowSpec(src="z", dst="y", flowtype=fcore.FlowType.SCALE_SHIFT, version="v3"),
        #     fcore.FlowSpec(src="y", dst="z", flowtype=fcore.FlowType.SHIFT_ONLY,  version="v3"),
        # ]

        # NOTE OT step handwritten in fr.py, leave blank here; the code allow two level of writing.
        # dec: List[fcore.FlowSpec] = [
        #     # fcore.FlowSpec(src="z", dst="y", flowtype=fcore.FlowType.R_SCALE_RB_SHIFT, version="v0"),
        # ]  
        return {"encoding": enc, "processing": proc, "decoding": dec}
# endregion

#region PatchTSTConfig
class PatchTSTConfig(BaseTrainingConfig):
    """FLAT configuration for PatchTST model."""
    # --- Meta ---
    model_type: Literal["PatchTST"] = "PatchTST"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"

    # --- PatchTST Specific Fields ---
    d_model: int = 128
    e_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    activation: str = "gelu"
    patch_len: int = 16
    stride: int = 8
    factor: int = 3
    task_name: str = "long_term_forecast"

    @computed_field(return_type=int)
    @property
    def enc_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def dec_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def c_out(self) -> int:
        return self.channels
    
    def with_overrides(self, **overrides: Any) -> "PatchTSTConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)


class DLinearConfig(BaseTrainingConfig):
    """FLAT configuration for DLinear model."""
    # --- Meta ---
    model_type: Literal["DLinear"] = "DLinear"
    forward_signature: Literal["x"] = "x"
    output_signature: Literal["tensor"] = "tensor"

    # --- DLinear Specific Fields ---
    individual: Optional[bool] = False

    def with_overrides(self, **overrides: Any) -> "DLinearConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)
    
    # @model_validator(mode="before")
    # @classmethod
    # def _derive_individual(cls, data: dict):
    #     """If user didn't specify `individual`, set it based on CI:
    #     - CI=True  -> individual=False (shared linear)
    #     - CI=False -> individual=True  (per-channel heads)
    #     """
    #     if isinstance(data, dict):
    #         if data.get("individual", None) is None:
    #             ci = data.get("channel_independent", True)
    #             data["individual"] = (not ci)
    #     return data

    # @model_validator(mode="after")
    # def _check_consistency(self):
    #     # In CI mode, per-channel heads don't make sense
    #     if self.channel_independent and self.individual:
    #         raise ValueError(
    #             f"DLinearConfig: `individual=True` conflicts with `channel_independent=True`. and it is {self.individual} and {self.channel_independent}"
    #         )
    #     # (Optional) If you require CI tensors to be [B·D, S, 1], enforce channels==1
    #     # if self.channel_independent and self.channels != 1:
    #     #     raise ValueError(
    #     #         f"DLinearConfig: self.channel_independent is True and it is {self.channel_independent} \
    #     #             yet` CI mode expects `channels==1` to match [B·D, S, 1] inputs. and it is {self.channels:1.0f} "
    #     #     )
    #     return self

class TimeMixerConfig(BaseTrainingConfig):
    """FLAT configuration for TimeMixer model."""
    # --- Meta ---
    model_type: Literal["TimeMixer"] = "TimeMixer"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"

    # --- TimeMixer Specific Fields ---
    embed: Optional[str] = None  # must keep None
    freq: Optional[str] = None  # must keep None
    use_norm: Optional[bool] = None  # must keep None
    channel_independence: bool = True
    e_layers: int = 2
    down_sampling_layers: int = 3
    down_sampling_window: int = 2
    d_model: int = 16
    d_ff: int = 32
    dropout: float = 0.1
    decomp_method: str = "moving_avg"
    moving_avg: int = 25
    down_sampling_method: str = "avg"

    def with_overrides(self, **overrides: Any) -> "TimeMixerConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)

    @computed_field(return_type=int)
    @property
    def enc_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def dec_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def c_out(self) -> int:
        return self.channels


class NaiveConfig(BaseTrainingConfig):
    """FLAT configuration for Naive baseline model."""
    # --- Meta ---
    model_type: Literal["naive"] = "naive"
    forward_signature: Literal["naive_repeat"] = "naive_repeat"
    output_signature: Literal["tensor"] = "tensor"

    def with_overrides(self, **overrides: Any) -> "NaiveConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)

class AttraosConfig(BaseTrainingConfig):
    """FLAT configuration for Attraos model."""
    # --- Meta ---
    model_type: Literal["Attraos"] = "Attraos"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"
    
    # --- Attraos-specific (used by the model) ---
    patch_len: int = 24
    e_layers: int = 1

    # Phase Space Reconstruction (PSR)
    PSR_dim: int = 3                            # embedding dimension D
    PSR_type: Literal["indep","merged_seq","merged"] = "merged_seq"
    PSR_delay: int = 1                          # delay τ

    # MDMU / SSM widths
    dt_rank: int = 8                           # low-rank for Δ
    d_state: int = 16                           # SSM state size N

    # Δ parameterization (Mamba-style)
    dt_scale: float = 1.0
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4

    # Evolution options
    FFT_evolve: bool = False                     # frequency-enhanced local evolution
    multi_res: bool = False                     # enables H-branch (pscan_H)
    
    # derived
    @computed_field(return_type=int)
    @property
    def d_inner(self) -> int:
        # used by MDMU blocks: D_inner = PSR_dim * patch_len
        return self.PSR_dim * self.patch_len

    def with_overrides(self, **overrides: Any) -> "AttraosConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)

class KoopaConfig(BaseTrainingConfig):
    """FLAT configuration for Koopa model."""
    # --- Meta ---
    model_type: Literal["Koopa"] = "Koopa"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"
    mask_spectrum: Optional[List[int]] = None
    
    # --- Koopa Specific Fields ---
    # Frequency mask: list of rfft indices to zero out (DC-only by default)
    seg_len: int = 24
    num_blocks: int = 2
    dynamic_dim: int = 32
    hidden_dim: int = 128
    hidden_layers: int = 2
    multistep: bool = True

    @computed_field(return_type=int)
    @property
    def enc_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def dec_in(self) -> int:
        return self.channels

    @computed_field(return_type=int)
    @property
    def c_out(self) -> int:
        return self.channels
    
    def with_overrides(self, **overrides: Any) -> "KoopaConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)

    @model_validator(mode="after")
    def _debug_lr(self) -> "KoopaConfig":
        # CHG: temporary sanity check for override plumbing
        print(f"[KoopaConfig] final lr = {self.learning_rate}")
        return self
     
    
class ModernTCNConfig(BaseTrainingConfig):
    model_type: Literal["ModernTCN"] = "ModernTCN"
    forward_signature: Literal["x,none,none,none"] = "x,none,none,none"
    output_signature: Literal["tensor"] = "tensor"

    # backbone / head
    patch_size: int = 8
    patch_stride: int = 4
    downsample_ratio: int = 2    # per-stage downsample; fixed to 4 stages inside
    ffn_ratio: int = 2
    num_blocks: Union[int, List[int]] = 1      # int → broadcast to 4 stages
    large_size: Union[int, List[int]] = 51     # kernel size of DW conv per stage
    small_size: Union[int, List[int]] = 5
    dims: List[int] = [64, 64, 64, 64]         # per-stage channel dims
    small_kernel_merged: bool = False
    dropout: float = 0.3
    head_dropout: float = 0.0
    use_multi_scale: bool = False

    # normalization / extras
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    individual: bool = False
    decomposition: bool = False
    kernel_size: int = 25   # for series_decomp moving average when used

    @computed_field(return_type=int)
    @property
    def enc_in(self) -> int:  # ModernTCN expects enc_in==channels
        return self.channels

    @computed_field(return_type=int)
    @property
    def c_out(self) -> int:
        return self.channels
    
    def with_overrides(self, **overrides: Any) -> "ModernTCNConfig":
        """ Return a new config with updates applied; original unchanged. """
        return self.model_copy(update=overrides, deep=True)

class PFNNConfig(BaseTrainingConfig):
    """FLAT configuration for 1D PFNN-style Koopman AE baseline."""
    # --- Meta ---
    model_type: Literal["PFNN"] = "PFNN"
    forward_signature: Literal["x"] = "x"
    output_signature: Literal["dict:pred"] = "dict:pred"

    # --- PFNN specific hyperparams ---
    latent_dim: int = 96       # latent_dim was = channels * factor; Change to direct int
    hidden_dim: int = 96      # hidden_dim was = channels * factor
    pfnn_init_scale: float = 1.0      # scale on orthogonal W at init

    # CHG: default aux loss weights for this model
    aux_loss_weights: dict[str, float] = Field(
        default_factory=lambda: {"pfnn_id":1.0, "pfnn_contr": 0.3} #TODO 
    )
    # per their ablation "pfnn_id": 0.5, "pfnn_contr": 0.3 but performance not good
    # https://arxiv.org/pdf/2503.14702 page 10

    # optional: convenience constructor, same pattern as others
    @classmethod
    def from_common(cls, *, seq_len: int, pred_len: int, channels: int, seeds: List[int], **specifics: Any) -> "PFNNConfig":
        base = dict(seq_len=seq_len, pred_len=pred_len, channels=channels, seeds=seeds  )
        return cls(**{**base, **specifics})

ModelConfigType = Union[
    FERNConfig,
    TimeMixerConfig,
    PatchTSTConfig,
    DLinearConfig,
    NaiveConfig,
    KoopaConfig,
    ModernTCNConfig,
    AttraosConfig,
    PFNNConfig,
]

REGISTRY = {
    "fr": FERNConfig,
    "tm": TimeMixerConfig,   # same pattern: subclass BaseModelConfig, set defaults
    "dl": DLinearConfig,
    'tst': PatchTSTConfig,
    'tm': TimeMixerConfig,
    'naive': NaiveConfig,
    'attr': AttraosConfig,
    'kp': KoopaConfig,
    'mtcn': ModernTCNConfig,
    'pfnn': PFNNConfig,
}

