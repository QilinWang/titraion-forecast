import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import study.fr_cfg as configs
import study.fr_gen as fgen
import study.fr_apply as fapply 
from torch.distributions import MultivariateNormal
import numpy as np
import math
from typing import List
from pydantic import BaseModel, ConfigDict
import study.fr_core as fcore

class PipelinePhase:
    def __init__(self, flows: dict, sequence: list[str]):
        self.flows = flows
        self.sequence = sequence

    def apply(self, states: fgen.States) -> fgen.States:
        for name in self.sequence:
            if name:
                states = states | self.flows[name]
        return states

    def __repr__(self):
        return f"{self.__class__.__name__}({' → '.join(self.sequence)})"

class EncodingPhase(PipelinePhase):
    pass

class ProcessingPhase(PipelinePhase):
    pass

class DecodingPhase(PipelinePhase):
    pass

def key_ss(dst: fcore.Space, src: fcore.Space, v: str) -> str:
    # MODULE:SS:dst<-src:v=... 
    return fcore.CoefID.from_src_dst(src=src, dst=dst, version=v).module_key

def key_rot(dst: fcore.Space, v: str) -> str:
    # MODULE:ROT:dst<-dst:v=...;in=dst  (rotation lives in dst-space)
    return fcore.RotID(src=fcore.Space.H, dst=dst, version=v).module_key

def key_hid(src: fcore.Space, v: str="v0") -> str:
    # MODULE:HID:H<-src:v=...
    return fcore.HidID(src=src, dst=fcore.Space.H, version=v).module_key

def _ensure_modules(model: "FERN", bricks):
    for b in sorted(bricks, key=lambda x: x.module_key):
        if b.module_key not in model.NNs:
            b.add_to_ModuleDict(model.cfg, model.NNs)

class FERN(nn.Module):
    """ """
    def __init__(self, cfg: configs.FERNConfig):
        super().__init__()
        self.cfg = cfg
        self.NNs = cfg.build_factories()
        # print("NNs keys:", list(self.NNs.keys())[:30])
        # self.print_factory_keys()
        self.print_num_factory_keys()   
        
        # === SEMANTIC PIPELINE PHASES ===
        self.phases = self._build_pipeline_phases()
        self.print_encoding_keys(cfg)  
        
        self.revin_enabled = bool(getattr(cfg, "revin", False))
        self.revin_affine = bool(getattr(cfg, "revin_affine", True))
        self.revin_subtract_last = bool(getattr(cfg, "revin_subtract_last", False))

        if self.revin_enabled:
            self.revin_layer = RevIN(
                num_features=cfg.channels,
                affine=self.revin_affine,
                subtract_last=self.revin_subtract_last,
            )
            
        _ensure_modules(self, [
            fcore.RotID(src=fcore.Space.H, dst=fcore.Space.Y, version="v0"),
            # fcore.CoefID(src=fcore.Space.Z, dst=fcore.Space.Y, version="v0"),
            fcore.CoefID.from_src_dst(src=fcore.Space.Z, dst=fcore.Space.Y, version="v0"),
            # optionally:
            # fcore.HidID(src=fcore.Space.Z, dst=fcore.Space.H, version="v0"),
        ])

        monitor = fgen.CoefEigenMonitor.create_monitor(
                    device=self.cfg.device, dtype=self.cfg.dtype, 
                    bins=(1e-6, 1e-1, 1.0, 3.0, 10.0)
                ),
        
    def print_encoding_keys(self, cfg) -> None:
        enc_keys = [fapply.canonical_flow_key(s) for s in self.cfg.flow_phases["encoding"]]
        dec_keys = [fapply.canonical_flow_key(s) for s in self.cfg.flow_phases["decoding"]]
        print("Encoding (canonical flow keys):", enc_keys)
        print("Decoding (canonical flow keys):", dec_keys)
        print("Encoding seq:", [s.name for s in cfg.flow_phases["encoding"]])
        print("Processing seq:", [s.name for s in cfg.flow_phases["processing"]])
        print("Decoding seq:", [s.name for s in cfg.flow_phases["decoding"]])
        
    def print_factory_keys(self) -> None:
        print("Factory keys (canonical):")
        for k in self.NNs.keys():
            print("  ", k)
    
    def print_num_factory_keys(self) -> None:
        print("Number of factory keys:", len(self.NNs.keys()))
            
    def _build_pipeline_phases(self) -> dict:
        """Build semantically meaningful pipeline phases"""
        # [ADD] compile the config-declared program
        flow_phases = self.cfg.flow_phases                      # Dict[str, List[fcore.FlowSpec]]
        vocab_specs = []
        for phase in ("encoding", "processing", "decoding"):
            vocab_specs.extend(flow_phases.get(phase, []))

        links = fapply.FlowRegistry(self.NNs, vocab_specs)
        flows = links.cache  # dict[name -> Op]

        encoding_seq  = [fapply.canonical_flow_key(s) for s in flow_phases.get("encoding", [])]
        processing_seq = [fapply.canonical_flow_key(s) for s in flow_phases.get("processing", [])]
        decoding_seq  = [fapply.canonical_flow_key(s) for s in flow_phases.get("decoding", [])]

        return {
            "encoding": EncodingPhase(flows, encoding_seq),
            "processing": ProcessingPhase(flows, processing_seq),
            "decoding": DecodingPhase(flows, decoding_seq),
            "flows": flows,
        }
          
    def forward(
        self, x_bsd: torch.Tensor, update: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Chain example : plan = Chain([ rotate, op, ~rotate, params_for_z_given_z.shift] )
        states.z = states.z | plan
        """
        x = x_bsd.permute(0, 2, 1)
        # --- RevIN normalize (per-series, no cross-batch coupling) ---
        # if getattr(self, "revin_enabled", False):
        #     x = self.revin_layer(x, "norm")
        y_shape = dynamic_size(x, self.cfg.pred_len)
        z_shape = dynamic_size(x, self.cfg.dim_augment)
        z = sample_base(z_shape, x.device, x.dtype, kind="gauss") *0.1
        y = sample_base(y_shape, x.device, x.dtype, kind="gauss") *0.1
        # y = self.mvn(y)
        y = y  
        # y = y * self.param_scale_y + self.param_shift_y
        # states = states.set_y(y)

        with torch.set_grad_enabled(update): 
            
            states = fgen.States( # Initialize state 
                x=x, z=z, y=y, is_training=self.training, 
                patch_spec=self.cfg.patch_spec,
            ) 

            # === PHASE 1: ENCODE X → Z ===
            states = self.phases["encoding"].apply(states)  # TODO
            # states2 = self.phases["encoding"].apply(states2)
            # === PHASE 2: INITIALIZE Y ===

            # === PHASE 3: PROCESS Y ↔ Z ===
            states = self.phases["processing"].apply(states) #TODO
            # states2 = self.phases["processing"].apply(states2)
            pre_y = states.y

            # === PHASE 4: DECODE Z → Y ===
            states.h = self.NNs[key_hid(fcore.Space.Z, "v0")](states.z)
            rotation = self.NNs[key_rot(fcore.Space.Y, "v0")](states.h)
            scale, shift = self.NNs[key_ss(fcore.Space.Y, fcore.Space.Z, "v0")](states.h) # Z_to_Y_V0
            states.set_scale(scale)
            states.set_shift(shift)
            states.set_rotation(rotation) 
            states.monitor._compute_patch_reductions(states)

            constant = pre_y   | shift    |  rotation # | shift | self.ky shift
            
            accu = constant | scale
            # accu = (constant + torch.sigmoid(accu)) #| scale
            accu = accu  |  rotation.inv     # .inv #TODO  |  # self.ky.inv | 
            states.y = accu   
            # states = self.phases["decoding"].apply(states) #TODO
            # states = states | fapply.G(field="z", fac=self.NNs["gate_from_y"]) | fapply.T(op_name="gate_hard", field="y")
            # states.monitor.update_from_states(states)
            states.monitor.update_scale(op=states.scale)
            states.monitor.update_shift(op=states.shift)  
            # states = self.phases["processing2"].apply(states) #TODO
            # states = self.phases["processing"].apply(states) #TODO 
            y_bds = states.y 
            # if getattr(self, "revin_enabled", False):
            #     y_bds = self.revin_layer(y, "denorm")
            pred = y_bds.permute(0, 2, 1) # back to [B, pred_len, C]
        return {"pred": pred, "states": states}  
# endregion FERN


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True, subtract_last: bool = False):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = eps
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        # caches per forward (not persistent)
        self.register_buffer("_mean", torch.tensor(0.0), persistent=False)
        self.register_buffer("_stdev", torch.tensor(1.0), persistent=False)
        self.register_buffer("_last", torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: [B, C, L]
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == "denorm":
            return self._denormalize(x)
        else:
            raise NotImplementedError("RevIN mode must be 'norm' or 'denorm'")

    def _get_statistics(self, x: torch.Tensor) -> None:
        # reduce over all dims except channel
        if self.subtract_last:
            self._last = x[:, :, -1:].detach()
        else:
            self._mean = x.mean(dim=tuple(range(2, x.ndim)), keepdim=True).detach()
        self._stdev = torch.sqrt(x.var(dim=tuple(range(2, x.ndim)), keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - (self._last if self.subtract_last else self._mean)) / self._stdev
        if self.affine:
            y = y * self.affine_weight.view(1, -1, 1) + self.affine_bias.view(1, -1, 1)
        return y

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.affine:
            y = (y - self.affine_bias.view(1, -1, 1)) / (self.affine_weight.view(1, -1, 1) + self.eps * self.eps)
        y = y * self._stdev + (self._last if self.subtract_last else self._mean)
        return y

#    SS_Z_GIVEN_X_V0 → "MODULE:SS:Z<-X:v=v0"
#    HID_H_GIVEN_X_V0 → "MODULE:HID:H<-X:v=v0"
#    ROT_IN_Y_V0 → "MODULE:ROT:Y<-Y:v=v0;in=dst"
        
# Pretty-printer: spec -> "I | G[src:HID] | G[h:SS_Y_GIVEN_Z_V2] | R | Sc | Ri | Sh"
def render_chain(spec: fapply.ChainSpec, *, src: str, dst: str, version: str | None) -> str:
    parts = ["I"]
    for st in spec.steps:
        if st.type == "G":
            f = "src" if st.field == "src" else "h"
            parts.append(f"G[{f}:{st.param.name}]")
        else:
            op = st.op_name
            parts.append(
                "~R"
                if (op == "rotation" and st.inverse)
                else {
                    "rotation": "R",
                    "scale": "Sc",
                    "complex_scale": "CSc",
                    "shift": "Sh",
                }[op]
            )
    return " | ".join(parts)


# Example in logs:
# print(render_chain(fapply.FLOW_TEMPLATES[fapply.FlowType.R_SCALE_RB_SHIFT], src="z", dst="y", version="v2"))
# I | G[src:HID] | G[h:SS] | G[h:ROT] | R | Sc | ~R | Sh
# endregion Pretty-printer

# region moving average
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:,:,0:1,].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:,:,-1:,].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        # x = x.permute(0, 2, 1)
        return x
# endregion moving average


# region dynamic size
def dynamic_size(data: torch.Tensor, last_dim_size: int) -> List[int]:
    return list(data.shape[:-1]) + [last_dim_size]
# endregion dynamic size

# region constants
_EULER_GAMMA = 0.5772156649015329
_GUMBEL_STD = np.pi / np.sqrt(6.0)
# endregion constants

# region trunc normal

def trunc_normal(mu, sigma, shape, k=3.0, device=None, dtype=None):
    # Sample u in [Phi(-k), Phi(k)] then invert
    low  = 0.5 * (1 + math.erf(-k / math.sqrt(2)))
    high = 0.5 * (1 + math.erf( k / math.sqrt(2)))
    u = torch.rand(*shape, device=device, dtype=dtype) * (high - low) + low
    # Inverse CDF via erfinv
    eps = math.sqrt(2) * torch.erfinv(2*u - 1)
    return mu + sigma * eps
# endregion trunc normal

# region sample base
def sample_base(
    shape, device, dtype, kind="gauss", *, gen: torch.Generator | None = None
):
    if kind == "gauss":
        # return torch.randn(shape, device=device, dtype=dtype, generator=gen)
        return trunc_normal(0.0, 1.0, shape, 3.0, device, dtype)
    elif kind == "gumbel":
        # Standard Gumbel(0,1): g = -log(-log U)  (equivalently -log Exp(1))
        g = (-torch.empty(shape, device=device, dtype=dtype).exponential_(generator=gen).log())
        g = (g - _EULER_GAMMA) / _GUMBEL_STD # Optional: standardize to ~N(0,1)-like scale if you want parity with Gaussian
        return g
    elif kind == "laplace":
        return torch.distributions.Laplace(
            loc=torch.tensor(0.0, device=device, dtype=dtype),
            scale=torch.tensor(1.0, device=device, dtype=dtype),
        ).sample(shape, generator=gen)
    else:
        raise ValueError(f"Unknown base kind: {kind}")
# endregion sample base

# region Parametrizations *unused*
class SymmetricParametrization(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

class MatrixExponentialParametrization(nn.Module):
    def forward(self, X):
        return torch.matrix_exp(X)

class CayleyMapParametrization(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, X):  # (I + X)(I - X)^{-1}
        Id = self.Id.to(X.device)
        return torch.linalg.solve(Id - X, Id + X)

class SkewParametrization(nn.Module):
    def forward(self, X):
        A = X.triu(1)
        return A - A.transpose(-1, -2)

class LowerTriangularParametrization(nn.Module):
    def forward(self, X):
        return torch.tril(X, diagonal=-1) + torch.eye(X.size(-1), device=X.device) 
# endregion Parametrizations