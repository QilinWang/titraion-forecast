from pathlib import Path
import time
import torch
import numpy as np 
from typing import List, Dict, Optional, Any, Union, Literal
import study.fr_cfg as configs
import study.fr_tr as train
import study.fr_data_mgr as data_mgr 
import study.fr_data_gen as data_gen
from pydantic import BaseModel, ConfigDict

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
 
from pathlib import Path 
from torch.distributions import exp_family

# IPython-specific imports for setup
try:
    from IPython.display import Markdown, display  # type: ignore
except ImportError:
    Markdown = None

    def display(*args, **kwargs):
        print("Warning: display() is not available outside of an IPython environment.")

SAVE_DIR = Path("results")
from enum import StrEnum, auto
from functools import partial
from typing import Callable, Dict, Optional

from typing import Any, Dict, List, Optional
# from . import fr_tr as train
# from . import fr_cfg as configs
# from . import fr_data_mgr as data_mgr
import polars as pl


class PremadeID(StrEnum):
    LORENZ_BASE       = auto() ; LORENZ_PARAM      = auto() ; LORENZ_STATE      = auto() ; LORENZ_SWITCH     = auto()
    ROSSLER_BASE      = auto() ; ROSSLER_PARAM     = auto() ; ROSSLER_STATE      = auto() ; ROSSLER_SWITCH     = auto()
    HYPER_ROSSLER_BASE = auto() ; HYPER_ROSSLER_PARAM     = auto() ; HYPER_ROSSLER_STATE      = auto() ; HYPER_ROSSLER_SWITCH     = auto()
    LOGISTIC_BASE     = auto() ; LOGISTIC_PARAM     = auto() ; LOGISTIC_STATE      = auto() ; LOGISTIC_SWITCH     = auto()
    DUFFING_BASE      = auto() ; DUFFING_PARAM     = auto() ; DUFFING_STATE      = auto() ; DUFFING_SWITCH     = auto()
    LORENZ96_BASE     = auto() ; LORENZ96_PARAM     = auto() ; LORENZ96_STATE      = auto() ; LORENZ96_SWITCH     = auto()
    CHUA_BASE         = auto() ; CHUA_PARAM     = auto() ; CHUA_STATE      = auto() ; CHUA_SWITCH     = auto()
    HENON_BASE        = auto() ; HENON_PARAM     = auto() ; HENON_STATE      = auto() ; HENON_SWITCH     = auto()
    OU_BASE           = auto() ; OU_PARAM     = auto() ; OU_STATE      = auto() ; OU_SWITCH     = auto()
    SLDS_BASE         = auto() ; SLDS_PARAM     = auto() ; SLDS_STATE      = auto() ; SLDS_SWITCH     = auto()
    DOUBLEWELL_BASE   = auto() ; DOUBLEWELL_PARAM     = auto() ; DOUBLEWELL_STATE      = auto() ; DOUBLEWELL_SWITCH     = auto()
    SEASONAL_AR_BASE  = auto() ; SEASONAL_AR_PARAM     = auto() ; SEASONAL_AR_STATE      = auto() ; SEASONAL_AR_SWITCH     = auto()
    GARCH_BASE        = auto() ; GARCH_PARAM     = auto() ; GARCH_STATE      = auto() ; GARCH_SWITCH     = auto()
    KS_BASE           = auto() ; KS_PARAM     = auto() ; KS_STATE      = auto() ; KS_SWITCH     = auto()

    ETTH2_BASE = auto() ; ETTM2_BASE = auto() ; ETTH1_BASE = auto() ; ETTM1_BASE = auto() ; WEATHER_BASE = auto()
 


def create_source(
    dataset_id: data_mgr.SynDataID, 
    common: dict[str, Any],
    shock: dict[str, Any],
    overrides: dict[str, Any] = {},
    ) -> data_mgr.SourceConfig:
    # CHG: copy to avoid mutating caller / shared defaults
    common = dict(common or {})
    shock = dict(shock or {})
    overrides = dict(overrides or {})
    
    params = data_mgr.ID_to_Params[dataset_id]
    
    merged = {**common, **shock, **overrides,}

    return data_mgr.SourceConfig(dataset_id=dataset_id, params=params(**merged))
  

def make_source(
    shock_id: PremadeID, 
    common: dict[str, Any] = {},
    shock: dict[str, Any] = {},
    overrides: dict[str, Any] = {}, 
    ) -> data_mgr.SourceConfig: 
    
    # CHG: copy to avoid mutating caller / shared defaults
    common = dict(common or {})
    shock = dict(shock or {})
    overrides = dict(overrides or {})

    common.setdefault("steps", 35999)
    common.setdefault("dt", 1e-2)
    common.setdefault("dtype", np.float64)
    shock.setdefault("shock_frac", 0.35)

    if shock_id == PremadeID.ETTH2_BASE:
        return data_mgr.SourceConfig(
            dataset_id=(ds:=data_mgr.KnownDataID.ETTH2),
            path=f"{ds}.csv",  date_column="date", 
        )
    elif shock_id == PremadeID.ETTM2_BASE:
        return data_mgr.SourceConfig(
            dataset_id=(ds:=data_mgr.KnownDataID.ETTM2),
            path=f"{ds}.csv", date_column="date", 
        )
    elif shock_id == PremadeID.ETTH1_BASE:
        return data_mgr.SourceConfig(
            dataset_id=(ds:=data_mgr.KnownDataID.ETTH1),
            path=f"{ds}.csv", date_column="date", 
        )
    elif shock_id == PremadeID.ETTM1_BASE:
        return data_mgr.SourceConfig(
            dataset_id=(ds:=data_mgr.KnownDataID.ETTM1),
            path=f"{ds}.csv", date_column="date", 
        )

    elif shock_id == PremadeID.WEATHER_BASE:
        return data_mgr.SourceConfig(
            dataset_id=(ds:=data_mgr.KnownDataID.WEATHER),
            path=f"{ds}.csv", date_column="date", 
        )
 
    elif shock_id == PremadeID.LORENZ_BASE:
        return create_source(
            dataset_id=data_mgr.SynDataID.LORENZ,
            common=common,
            shock={}, #NOTE: in base case we override shock params to empty
            overrides=overrides,
        )
    elif shock_id == PremadeID.LORENZ_PARAM:
        shock.update({
            "shock_kind": "param", 
            "sigma_after": 10.1, # was 10.0 
            "rho_after": 28.1, # was 28.0 
            "beta_after": 8.1/3, # was 8/3
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.LORENZ,
            common=common,
            shock=shock, # NOTE: in param shock we keep shock frac=0.35
            overrides=overrides
        ) 

    elif shock_id == PremadeID.LORENZ_STATE:
        shock.update({
            "shock_kind": "state_eps",
            "shock_eps": 0.9, #
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.LORENZ,
            common=common,
            shock=shock,
            overrides=overrides
        )  
 
    elif shock_id == PremadeID.LORENZ_SWITCH:
        shock.update({
            "shock_kind": "switch",
            "switch_update": {
                "rho": 28.1, # was 28.0 
                "initial_cond": [1.002, 0.982, 1.102], # was [1.0, 0.98, 1.1] 
            },
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.LORENZ,
            common=common,
            shock=shock,
            overrides=overrides
        )  
    
    elif shock_id == PremadeID.ROSSLER_BASE:
        return create_source(
            dataset_id=data_mgr.SynDataID.ROSSLER,
            common=common,
            shock={},
            overrides=overrides,
        )
    
    elif shock_id == PremadeID.ROSSLER_PARAM:
        shock.update({
            "shock_kind": "param", 
            "a_after": 0.25, # was 0.2
            "b_after": 0.25, # was 0.2
            "c_after": 5.75, # was 5.7
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.ROSSLER,
            common=common,
            shock=shock,
            overrides=overrides,
        )
    
    elif shock_id == PremadeID.LORENZ96_BASE:
        common.update({
            "steps": 55000, "dt": 7e-3, "method": 'rk4', "dim": 6,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.LORENZ96,
            common=common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.LORENZ96_SWITCH:
        common.update({
            "steps": 55000, "dt": 7e-3, "method": 'rk4', "dim": 6,
        })
        shock.update({
            "shock_kind": "switch", 
            "switch_update": {
                "forcing": 9.0, # was 8.0
                'initial_cond': np.array([0.99, 1.02, 1.02, 1.03, 1.01, 1.01], dtype=np.float64)
            },
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.LORENZ96,
            common=common,
            shock=shock,
            overrides=overrides,
        )
    elif shock_id == PremadeID.CHUA_BASE:
        common.update({
            "dt": 5e-3,
        }) 
        return create_source(
            dataset_id=data_mgr.SynDataID.CHUA,
            common=common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.CHUA_PARAM:
        common.update({
            "dt": 5e-3,
        })
        shock.update({
            "shock_kind": "param", 
            "alpha_after": 15.9, # was 15.6
            "beta_after": 28.5, # was 28
            "m0_after": -8.1/7, # was -8/7
            "m1_after": -5.2/7    , # was -5/7
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.CHUA,
            common=common,
            shock=shock,
            overrides=overrides,
        )
    elif shock_id == PremadeID.CHUA_SWITCH:
        common.update({
            "dt": 5e-3,
        })
        shock.update({
            "shock_kind": "switch",
            "switch_update": {
                'initial_cond': np.array([0.11, 0.01, 0.02], dtype=np.float64)
            },
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.CHUA,
            common=common,
            shock=shock,
            overrides=overrides,
        )
    
    elif shock_id == PremadeID.OU_BASE:
        common.update({
            "initial_cond": [0.0], "steps": 25000, "dt": 0.5, "method": 'euler',
            "theta": 0.2, "mu": 0.0, "sigma": 0.3,
        })  
        return create_source(
            dataset_id=data_mgr.SynDataID.OU,
            common= common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.OU_PARAM:
        common.update({
            "initial_cond": [0.0], "steps": 25000, "dt": 0.5, "method": 'euler',
            "theta": 0.2, "mu": 0.0, "sigma": 0.3,
        })
        shock.update({
            "shock_kind": "param",
            "mu_after": 0.5,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.OU,
            common=common,
            shock=shock,
            overrides=overrides,
        )
    
    elif shock_id == PremadeID.SLDS_BASE:
        common.update({
            "steps": 25000, "dt": 0.01, "method": 'euler',
            "A1":0.9, "Q1":0.05, "A2":0.98, "Q2":0.35, "p11":0.94, "p22":0.95,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.SLDS,
            common=common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.SLDS_PARAM:
        common.update({
            "steps": 25000, "dt": 0.01, "method": 'euler',
            "A1":0.9, "Q1":0.05, "A2":0.98, "Q2":0.35, "p11":0.94, "p22":0.95,
        })
        shock.update({
            "shock_kind": "param",
            "A1_after": 0.83, "Q1_after": 0.50, "A2_after": 0.97, "Q2_after": 0.30, 
            "p11_after": 0.96, "p22_after": 0.92,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.SLDS,
            common=common,
            shock=shock,
            overrides=overrides,
        )
    elif shock_id == PremadeID.SLDS_SWITCH:
        common.update({
            "steps": 25000, "dt": 0.01, "method": 'euler',
            "A1":0.9, "Q1":0.05, "A2":0.98, "Q2":0.35, "p11":0.94, "p22":0.95,
        })
        shock.update({
            "shock_kind": "switch",
            "switch_update": {
                "A1": 0.87, "Q1": 0.07, "A2": 0.99, "Q2": 0.45,
                "p11": 0.90, "p22": 0.95,
            },
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.SLDS,
            common=common,
            shock=shock,
            overrides=overrides,
        )
    elif shock_id == PremadeID.DOUBLEWELL_BASE:
        common.update({
            "steps": 25000, "dt": 0.5, "method": 'euler', # dt/method kept for API; not used here
            "a": 1.5, "sigma": 0.25, 'seed':1955
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.DOUBLEWELL,
            common=common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.DOUBLEWELL_PARAM:
        common.update({
            "steps": 25000, "dt": 0.5, "method": 'euler', # dt/method kept for API; not used here
            "a": 1.5, "sigma": 0.25, 'seed':1955
        })
        shock.update({
            "shock_kind": "param",
            "a_after": 1.0, "sigma_after": 0.35,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.DOUBLEWELL,
            common=common,
            shock=shock,
            overrides=overrides,
        ) 
    elif shock_id == PremadeID.DOUBLEWELL_SWITCH:
        common.update({
            "steps": 25000, "dt": 0.5, "method": 'euler', # dt/method kept for API; not used here
            "a": 1.5, "sigma": 0.25, 'seed':1955,
        })
        shock.update({
            "shock_kind": "switch",
            "switch_update": {
                'a': 1.0, 'sigma': 0.35,
            },
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.DOUBLEWELL,
            common=common,
            shock=shock,
            overrides=overrides,
        ) 
    elif shock_id == PremadeID.SEASONAL_AR_BASE:
        common.update({
            "steps": 25000, "dt": 0.01, "method": 'euler', # dt/method kept for API; not used here
            "S": 24, "phi": 0.5, "sigma": 0.2, "a0": 1.0, "amp_drift_per_step": 0e-4,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.SEASONAL_AR,
            common=common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.SEASONAL_AR_PARAM:
        common.update({
            "steps": 25000, "dt": 0.01, "method": 'euler', # dt/method kept for API; not used here
            "S": 24, "phi": 0.5, "sigma": 0.2, "a0": 1.0, "amp_drift_per_step": 0e-4,
        })
        shock.update({
            "shock_kind": "param",
            "a0_after": 1.4, "sigma_after": 0.35, "phi_after": 0.8,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.SEASONAL_AR,
            common=common,
            shock=shock,
            overrides=overrides,
        ) 
    elif shock_id == PremadeID.GARCH_BASE:
        common.update({
            "steps": 25000, "dt": 0.01, "method": 'euler', # dt/method kept for API; not used here
            "omega": 0.01, "alpha": 0.06, "beta": 0.90,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.GARCH,
            common=common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.GARCH_PARAM:
        common.update({
            "steps": 25000, "dt": 0.01, "method": 'euler', # dt/method kept for API; not used here
            "omega": 0.01, "alpha": 0.06, "beta": 0.90,
        })
        shock.update({
            "shock_kind": "param",
            "omega_after": 0.03, "alpha_after": 0.15, "beta_after": 0.70, 
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.GARCH,
            common=common,
            shock=shock,
            overrides=overrides,
        ) 
    elif shock_id == PremadeID.KS_BASE:
        common.update({
            "steps": 25000, "dt": 1e-2, "method": 'etdrk4', # dt/method kept for API; not used here
            "nx": 64, "Lx": 22.0, "nu": 1.0,
        })
        return create_source(
            dataset_id=data_mgr.SynDataID.KS,
            common=common,
            shock={},
            overrides=overrides,
        )
    elif shock_id == PremadeID.KS_PARAM:
        common.update({
            "steps": 25000, "dt": 1e-2, "method": 'etdrk4', # dt/method kept for API; not used here
            "nx": 64, "Lx": 22.0, "nu": 1.0,
        })
        shock.update({
            "shock_kind": "param",
            "nu_after": 0.80,
        }) # nx=128=>128 dim
        return create_source(
            dataset_id=data_mgr.SynDataID.KS,
            common=common,
            shock=shock,
            overrides=overrides,
        )  
    else:
        raise NotImplementedError(f"Shock scenario {shock_id} not implemented.") 
     
# region get_default_cfg_dict
def get_default_cfg_dict(
    *,
    seq_len: int,
    pred_len: int,
    channels: int,
    seeds: List[int],
    common_overrides: dict[str, Any] | None = None,
    per_model_specifics: dict[str, dict[str, Any]] | None = None,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> Dict[str, configs.ModelConfigType]:
    """Intent: build {model_name -> ModelConfig} via registry + from_common.
    
    Assumptions:
        - configs.REGISTRY: model_name -> ConfigClass (Pydantic BaseModel subclass).
        - Each ConfigClass implements from_common(seq_len, pred_len, channels, **overrides).
        - If include is not None, only models in `include` are built.
        - exclude removes names from all_names if include is None.
    Returns:
        - dict[model_name, ModelConfigInstance], sorted by model name.
    """
    # === Inputs / setup ===
    common_overrides = common_overrides or {}
    per_model_specifics = per_model_specifics or {}
    
    out: Dict[str, configs.ModelConfigType] = {}

    # === Select target models ===
    all_names = set(configs.REGISTRY.keys())
    if include is not None: 
        unknown = include - all_names
        if unknown:
            raise KeyError(f"Unknown models in include: {sorted(unknown)}")
        target = sorted(include)
    else:
        target = sorted(all_names - (exclude or set()))
 
    for name in target: # === Construct each config ===
        cls = configs.REGISTRY[name]
        if not hasattr(cls, "from_common"): 
            raise AttributeError(f"{name}: registry class {cls.__name__} must implement from_common(...)")

        # Merge precedence: common_overrides < per_model_specifics[name]
        
        specifics = {**common_overrides, **per_model_specifics.get(name, {})}
        cfg = cls.from_common(seq_len=seq_len, pred_len=pred_len, channels=channels, seeds=seeds, **specifics)
        out[name] = cfg 
    return out 
# endregion

def format_latex_rows_for_model(
    df: pl.DataFrame,
    model_key: str,
    source_name: str,
    std_horizons: list[int],
    digits: int = 3,
) -> str:
    """Intent: build LaTeX row block for a single model.
    
    Assumptions:
        - df has columns: 'seed','MSE','MAE','SWD','EPT','Model','pred_len'.
        - Seed column may be int (for per-seed rows) or str (for summary rows).
        - Summary rows already have 'MSE','MAE','SWD','EPT' as formatted strings
          (e.g. "13.675 ± 0.184") from run_multiple_seeds.
    Returns:
        - A LaTeX string with lines like:
            \\midrule
            7 & 13.87 & 2.52 & 9.50 & 71.19 & fr & 96 \\\\
            ...
    """
    # Filter rows for this model and desired horizons
    mdl = df.filter(
        (pl.col("Model") == model_key)
        & (pl.col("pred_len").is_in(std_horizons))
    )
    if mdl.is_empty():
        return ""

    lines: list[str] = []
    lines.append(r"\\midrule")

    # We iterate in the order they appear (seeds first, summary mixed in)
    for row in mdl.iter_rows(named=True):
        seed = str(row["seed"])
        model = str(row["Model"])
        pred_len = str(row["pred_len"])

        if seed.isdigit():
            # CHG: per-seed numeric rows → format with fixed decimals
            mse = float(row["MSE"])
            mae = float(row["MAE"])
            swd = float(row["SWD"])
            ept = float(row["EPT"])
            line = (
                f"{seed} & "
                f"{mse:.{digits}f} & "
                f"{mae:.{digits}f} & "
                f"{swd:.{digits}f} & "
                f"{ept:.{digits}f} & "
                f"{model} & {pred_len} \\\\"
            )
        else:
            # CHG: summary rows (e.g. '96-FERN') – metrics already formatted
            mse = str(row["MSE"])
            mae = str(row["MAE"])
            swd = str(row["SWD"])
            ept = str(row["EPT"])
            line = (
                f"{seed} & "
                f"{mse} & "
                f"{mae} & "
                f"{swd} & "
                f"{ept} & "
                f"{model} & {pred_len} \\\\"
            )

        lines.append(line)

    return "\n".join(lines)

def format_latex_summary_line_for_model(
    df: pl.DataFrame,
    model_key: str,
    std_horizons: list[int],
    digits: int = 3,
) -> str:
    """Intent: one LaTeX line with simple average across horizons for a model.

    Assumptions:
        - df has columns: 'seed','MSE','MAE','SWD','EPT','Model','pred_len'.
        - Seed rows: seed is all digits (e.g. '7','1955'); summary rows have
          non-digit seeds (e.g. '96-FERN').
    Returns:
        - A single LaTeX row string, e.g.
          'FERN-AVG & 13.62 & 2.49 & 9.02 & 71.00 & fr & ALL \\\\'
          or empty string if no data.
    """
    # Filter model + horizons
    mdl = df.filter(
        (pl.col("Model") == model_key)
        & pl.col("pred_len").is_in(std_horizons)
    )
    if mdl.is_empty():
        return ""

    # Seed rows: seed is all digits
    seeds = mdl.filter(
        pl.col("seed").cast(pl.Utf8).str.contains(r"^[0-9]+$")
    )
    if seeds.is_empty():
        return ""
 
    by_h = ( # Per-horizon mean across seeds
        seeds.group_by("pred_len")
        .agg(
            MSE_mean=pl.col("MSE").mean(),
            MAE_mean=pl.col("MAE").mean(),
            SWD_mean=pl.col("SWD").mean(),
            EPT_mean=pl.col("EPT").mean(),
        )
    )
 
    avg = by_h.select( # Simple average of these per-horizon means (unweighted)
        pl.col("MSE_mean").mean().alias("MSE"),
        pl.col("MAE_mean").mean().alias("MAE"),
        pl.col("SWD_mean").mean().alias("SWD"),
        pl.col("EPT_mean").mean().alias("EPT"),
    ).row(0, named=True)

    mse = f"{avg['MSE']:.{digits}f}"
    mae = f"{avg['MAE']:.{digits}f}"
    swd = f"{avg['SWD']:.{digits}f}"
    ept = f"{avg['EPT']:.1f}"
 
    label = f"{model_key}-AVG" # You can choose the label; here I use '<model>-AVG' and pred_len 'ALL'
    line = f"{label} & {mse} & {mae} & {swd} & {ept} & {model_key} & ALL \\\\"
    return line

def sync_cfg_channels_to_bundle(cfg: configs.BaseTrainingConfig, bundle: data_mgr.DataBundle) -> configs.BaseTrainingConfig:
    """
    Intent/contract:
    - Align cfg.channels with the dataset's *active* feature count after known-fix drops.
    - In-place, idempotent. Returns cfg for chaining.
    - Does not touch seq_len/pred_len or any other fields.

    Why:
    - stage_known_problem_fixed may drop columns → HiddenFactory/States carry C = len(active_feature_names).
    - FERN factories/schemas are built from cfg.channels. Keep them consistent to avoid CoefFactory's
      shape assertion.
    """
    names = bundle.feature_names
    if cfg.channels != len(names):
        print(f"[cfg-sync] channels: {cfg.channels} -> {len(names)}")
        cfg.channels = len(names)
    return cfg

class SeedRun(BaseModel):
    """One training run for a specific (model, pred_len, seed).
    Intent:
        Flatten nested {pred_len -> model -> [Trainer]} into a simple list.
    Assumptions:
        - trainer.best_test_metrics is populated.
    Returns:
        - Access to trainer and its metrics via .trainer / .metrics.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid', validate_assignment=False)

    model_key: str
    pred_len: int
    seed: int
    trainer: train.Trainer  # [*] single-seed Trainer object

    @property
    def metrics(self) -> Dict[str, Any]:
        """Best test metrics as a small dict."""
        m = self.trainer.best_test_metrics
        return {
            "Model": self.model_key,
            "pred_len": self.pred_len,
            "seed": self.seed,
            "MSE": float(m.mse),
            "MAE": float(m.mae),
            "SWD": float(m.swd),
            "EPT": float(m.ept),
        }
    
    @property
    def training_time(self) -> float:
        t = self.trainer.training_time
        assert t is not None, "Trainer.training_time should be set after training."
        return float(t)

class ExpResults(BaseModel):
    """Container for all runs + metrics from exp_runner.
    Assumptions:
        - runs contains one SeedRun per (model_key, pred_len, seed) actually executed.
    Returns:
        - master_df: long-form polars DF for LaTeX / CSV.
        - helpers to fetch Trainers by model/pred_len/seed.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_name: str
    seq_len: int
    channels: int
    pred_len_lst: list[int]
    seeds: list[int]

    runs: list[SeedRun]
    master_df: pl.DataFrame | None = None # [N_rows, *] long-form metrics table
    
    total_training_time: float = 0.0
    
    def model_post_init(self, __context: Any) -> None:  # CHG: pydantic v2 hook
        rows = []
        total = 0.0
        for run in self.runs:
            total += run.training_time
            rows.append(run.metrics)
        self.total_training_time = total
        self.master_df = pl.from_dicts(rows) if rows else pl.DataFrame()
         

    def filter_models(self, model_key: str, pred_len: int | None = None) -> list[train.Trainer]:
        """All Trainer objects for a model, optionally restricted to one pred_len."""
        out: list[train.Trainer] = []
        for r in self.runs:
            if r.model_key != model_key:
                continue
            if pred_len is not None and r.pred_len != pred_len:
                continue
            out.append(r.trainer)
        return out

    def filter_model(self, model_key: str, pred_len: int, seed: int) -> train.Trainer:
        """Unique Trainer for (model_key, pred_len, seed).

        REQUIRE:
            There is exactly one such run; raise KeyError otherwise.
        """
        for r in self.runs:
            if r.model_key == model_key and r.pred_len == pred_len and r.seed == seed:
                return r.trainer
        raise KeyError(f"No run for model={model_key}, pred_len={pred_len}, seed={seed}")


def exp_runner( 
    train_source_config: data_mgr.SourceConfig,
    seq_len: int,
    pred_len_lst: Optional[List[int]] = None,
    channels: int = 7,
    seeds: List[int] = [7, 1955, 2023, 4],
    no_scale=True,
    
    common_overrides: dict[str, Any] = {
        # use_proj_swd=False,
        # channel_independent=True, 
        # patience: int = 5,
        # batch_size: int = 96,
        # known_fix: Optional[data_mgr.KnownFixConfig] = None, 
    },
    per_model_specifics: dict[str, dict[str, Any]] = {
        "fr": {
            # num_reflects=8,
        },
    },
    
    save_csv: bool = True,
    to_latex: bool = True,
    latex_digits: int = 3,
     
    include: set[str] | None = None,   # CHG: use same knobs as get_default_cfg_dict
    exclude: set[str] | None = None, 
)->ExpResults:
    """Intent: run all requested models × horizons, aggregate metrics in Polars.

    Returns:
        ExpResults:
            - .runs: list[SeedRun]
            - .master_df: long-form metrics table
    """
    if pred_len_lst is None:
        pred_len_lst = [96, 192, 336, 720]

    # CHG: flat collection of runs instead of nested dict
    run_records: list[SeedRun] = []
    
    # Long-form rows in polars (per pred_len × model) 

    for pred_len in pred_len_lst:
        print(f"\n{'#' * 60}\n>>> Running all models @ pred_len={pred_len}\n{'#' * 60}\n")
        
        # === Build model configs (all at once) ======================
        cfgs = get_default_cfg_dict(
            seq_len=seq_len,
            pred_len=pred_len,
            channels=channels,
            seeds=seeds,
            common_overrides=common_overrides,
            per_model_specifics=per_model_specifics,
            include=include,
            exclude=exclude,
        )
        
        # === Create data pipeline / bundle =======================
        # REQUIRE: data_bundle provides dataset and inferred channel count.
        for cfg_name, cfg in cfgs.items(): 
            data_bundle = data_mgr.create_data_pipeline(
                train_source_config, 
                cfg,
                no_scale=no_scale,
            )
            # Sync channel count from bundle into cfg (so model/channel dims match).
            cfg = sync_cfg_channels_to_bundle(cfg, data_bundle)

            # === Run multi-seed experiment =========================== 
            trainer_lst: list[train.Trainer] = []
            local_rows: list[dict[str, Any]] = []  # CHG: for per-model CSV

            for i, seed in enumerate(cfg.seeds):
                trainer = train.Trainer(config=cfg, data_bundle=data_bundle, seed=seed)
                trainer = trainer.train_model()
                trainer.lighten()
                trainer_lst.append(trainer) 
            
                run = SeedRun(
                    model_key=cfg_name,
                    pred_len=pred_len,
                    seed=trainer.seed,
                    trainer=trainer,
                )
                run_records.append(run)
                local_rows.append(run.metrics)  # CHG: link per-run metrics
            
            if save_csv:
                csv_path = trainer_lst[0].csv_path
                df_pl = pl.from_dicts(local_rows)
                df_pl.write_csv(csv_path)
                print(f"Saved CSV: {csv_path}")

    # ===== AFTER ALL HORIZONS =====
    source_name = train_source_config.dataset_id.value  # CHG: define before ExpResults

    results = ExpResults(
        source_name=source_name,
        seq_len=seq_len,
        channels=channels,
        pred_len_lst=list(pred_len_lst),
        seeds=list(seeds),
        runs=run_records, 
    )
    print(f"Total training time: {results.total_training_time:.2f} seconds")

    # === LaTeX export (still via pandas for formatting) ============== 
    if to_latex and not results.master_df.is_empty(): 
        std_horizons = [96, 192, 336, 720]  # adjust if you want fewer

        models_in_results = sorted(set(results.master_df["Model"].unique().to_list()))
        for model_key in models_in_results:
            latex_rows = format_latex_rows_for_model(
                df=results.master_df,
                model_key=model_key,
                source_name=source_name,
                std_horizons=std_horizons,
                digits=latex_digits,
            )
            if not latex_rows:
                continue

            avg_line = format_latex_summary_line_for_model(
                df=results.master_df,
                model_key=model_key,
                std_horizons=std_horizons,
                digits=latex_digits,
            )

            block_parts = [f"% Rows for {model_key} on {source_name}", latex_rows]
            if avg_line: 
                block_parts.append(avg_line)

            block = "\n".join(block_parts)
            display(Markdown(f"```latex\n{block}\n```"))
    return results
"""
exp = exp_runner(...)

# flat view
for run in exp.runs:
    print(run.model_key, run.pred_len, run.seed, run.metrics["MSE"])

# specific Trainer
tr_fr_96_seed7 = exp.trainer_for("fr", pred_len=96, seed=7)
"""

def run_many_datasets(
    source_cfgs: dict[str, data_mgr.SourceConfig], 
    **exp_kwargs: Any,
) -> tuple[dict[str, ExpResults], pl.DataFrame]:
    """Run exp_runner on multiple datasets and aggregate master_df."""

    results_by_ds: dict[str, ExpResults] = {}
    dfs: list[pl.DataFrame] = []

    for ds_key, src_cfg in source_cfgs.items():
        res = exp_runner(train_source_config=src_cfg,  **exp_kwargs)
        # tag dataset in the DF for later global tables
        df_tagged = res.master_df.with_columns(
            pl.lit(ds_key).alias("dataset")
        )
        dfs.append(df_tagged)
        results_by_ds[ds_key] = res

    combined_df = pl.concat(dfs, how="vertical") if dfs else pl.DataFrame()
    return results_by_ds, combined_df

"""
USAGE:
sources = {
    "ETTh2": etth2_cfg,
    "ETTm1": ettm1_cfg,
    "Lorenz": lorenz_cfg,
}

results_map, df_all = run_many_datasets(
    source_cfgs=sources,
    seq_len=336,
    pred_len_lst=[96, 192],
    channels=7,
    seeds=[7, 1955, 2023, 4],
    no_scale=True,
)
"""


def plot_final_iclr_figure(full_series, input_len, patch_len):
    """
    Generates the final, publication-quality figure with a more plausible
    forecast series that aligns better with the geometric guides.
    """
    # --- 1. ICLR-Ready Setup and Styling ---
    plt.style.use("seaborn-v0_8-paper")
    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("FERN's Geometric Forecasting Mechanism", weight="bold")

    # Colorblind-friendly palette
    COLOR_INPUT = "#377eb8"  # Blue
    COLOR_PREDICT = "#ff7f00"  # Orange
    COLOR_TRUTH = "#4daf4a"  # Green
    COLOR_LATENT = "#984ea3"  # Purple
    LINEWIDTH = 2.2

    # --- 2. Panel (a): Input Series ---
    ax1 = axes[0]
    input_series = full_series[:input_len]
    ax1.plot(
        np.arange(input_len),
        input_series,
        color=COLOR_INPUT,
        linewidth=LINEWIDTH,
        label="Input Series",
    )
    num_input_patches = input_len // patch_len
    for i in range(1, num_input_patches):
        ax1.axvline(
            x=i * patch_len - 0.5,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
    ax1.set_title("(a) Input Series", weight="bold")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, input_len)

    # --- 3. Panel (b): Latent Space Transformation ---
    ax2 = axes[1]
    np.random.seed(0)
    points = np.random.randn(200, 2)
    scale = np.array([[2.8, 0], [0, 0.9]])
    angle = np.pi / 4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transform = rotation @ scale
    transformed_points = (transform @ points.T).T
    ax2.scatter(
        points[:, 0],
        points[:, 1],
        alpha=0.6,
        s=20,
        color=COLOR_INPUT,
        label="Isotropic Gaussian Latent",
    )
    ax2.scatter(
        transformed_points[:, 0],
        transformed_points[:, 1],
        alpha=0.6,
        s=20,
        color=COLOR_LATENT,
        label="Anisotropic Encoded Latent",
    )
    iso_radius = 2.5
    circ = Ellipse(
        xy=(0, 0),
        width=2 * iso_radius,
        height=2 * iso_radius,
        angle=0,
        facecolor=COLOR_INPUT,
        alpha=0.1,
    )
    ax2.add_patch(circ)
    cov = np.cov(transformed_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    ell_scale = 2.5
    ell_width = 2 * ell_scale * np.sqrt(eigvals.max())
    ell_height = 2 * ell_scale * np.sqrt(eigvals.min())
    ell_angle_deg = np.degrees(
        np.arctan2(eigvecs[1, eigvals.argmax()], eigvecs[0, eigvals.argmax()])
    )
    ell = Ellipse(
        xy=(0, 0),
        width=ell_width,
        height=ell_height,
        angle=ell_angle_deg,
        facecolor=COLOR_LATENT,
        alpha=0.15,
    )
    ax2.add_patch(ell)
    ax2.set_title("(b) Latent Space Transformation", weight="bold")
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("Latent Dimension 1")
    ax2.set_ylabel("Latent Dimension 2")
    ax2.legend(loc="upper right")

    # --- 4. Panel (c): Patchwise Geometric Forecast ---
    ax3 = axes[2]
    forecast_len = patch_len * 3
    forecast_x = np.arange(input_len, input_len + forecast_len)
    last_val = input_series[-1]

    # **MODIFIED LINE**: Create a U-shaped forecast to better match the ground truth trend
    # This ensures the forecast line aligns visually with all three ellipsoids.
    t_poly = np.linspace(-1.5, 1.5, forecast_len)
    parabolic_trend = 3.0 * (t_poly**2)
    t_sin = np.linspace(0, 3 * np.pi, forecast_len)
    forecast_y = (
        last_val
        + parabolic_trend
        - 2.5
        + np.sin(t_sin * 1.2) * 0.8
        + (np.random.randn(forecast_len) * 0.4)
    )

    ground_truth_y = full_series[input_len : input_len + forecast_len]

    context_x = np.arange(input_len - patch_len, input_len + 1)
    context_y = full_series[input_len - patch_len : input_len + 1]
    ax3.plot(
        context_x, context_y, color=COLOR_INPUT, linewidth=LINEWIDTH, label="Input"
    )
    ax3.plot(
        forecast_x,
        forecast_y,
        color=COLOR_PREDICT,
        linewidth=LINEWIDTH,
        linestyle="--",
        label="Forecast",
    )
    ax3.plot(
        forecast_x,
        ground_truth_y,
        color=COLOR_TRUTH,
        linewidth=LINEWIDTH,
        linestyle="-",
        label="Ground Truth",
    )

    global_pred_std = np.std(forecast_y) + 1e-6
    for i in range(forecast_len // patch_len):
        start_idx, end_idx = i * patch_len, (i + 1) * patch_len
        patch_x_coords = forecast_x[start_idx:end_idx]
        patch_y_truth = ground_truth_y[start_idx:end_idx]
        patch_y_pred = forecast_y[start_idx:end_idx]

        center_x, center_y = np.mean(patch_x_coords), np.mean(patch_y_pred)

        slope = np.polyfit(patch_x_coords, patch_y_truth, 1)[0]
        angle_deg = np.degrees(np.arctan(slope))

        patch_pred_std = np.std(patch_y_pred)
        rel_unc = np.clip(patch_pred_std / global_pred_std, 0.7, 1.5)
        height = 2.5 * rel_unc

        ellipse = Ellipse(
            xy=(center_x, center_y),
            width=patch_len * 1.2,
            height=height,
            angle=angle_deg,
            facecolor=COLOR_PREDICT,
            alpha=0.3,
        )
        ax3.add_patch(ellipse)
        ax3.axvline(
            x=end_idx + input_len - patch_len - 0.5,
            color="grey",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

    ax3.set_title("(c) Patchwise Geometric Forecast", weight="bold")
    ax3.set_xlabel("Time Step")
    ax3.legend(loc="upper left")
    ax3.set_xlim(input_len - patch_len, input_len + forecast_len)

    # --- 5. Final Polishing ---
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(width=1.2)
        ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="gainsboro")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("fern_final_figure.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

"""
if __name__ == '__main__':
    np.random.seed(42)
    input_length = 96
    patch_length = 24
    total_length = 200
    t = np.linspace(0, 10 * np.pi, total_length)
    # Using the same data generation as before
    time_series = 3 * np.sin(t * 0.5) + np.cos(t * 2.5) + t * 0.15 + np.random.randn(total_length) * 0.4

    print("Generating the final, revised ICLR-style figure...")
    plot_final_iclr_figure(time_series, input_length, patch_length)
    print("Figure saved as 'fern_final_figure.pdf'")
"""

#region Text processing helpers



from dataclasses import dataclass
from typing import List, Dict

LONG_NAMES = {
    "dl": "DLinear",    # CHG: human-readable model names
    "fr": "Fern",       # CHG: used for pretty print in rebuttal blocks
    "kp": "Koopa",
    "mtcn": "ModernTCN",
    "tm": "TimeMixer",
    "tst": "PatchTST",
    "pfnn": "PFNN",
}

# === Data containers ===

@dataclass
class ModelRow:
    """Intent: container for per-model metrics.
    
    Assumptions:
        - short_name is an identifier like 'tm', 'fr'.
        - All metrics are scalar floats (MSE, MAE, SWD, EPT).
    Returns:
        - Simple struct; conceptual shape [1] per field.
    """
    short_name: str
    mse: float
    mae: float
    swd: float
    ept: float


# === Parsing ===

def parse_latex_block(raw: str) -> List[ModelRow]:
    """Intent: parse latex-style lines into ModelRow entries.
    
    Assumptions:
        - Each metric line looks like:
          name-AVG & mse & mae & swd & ept & short_name & split \\
        - Column order is: tag, mse, mae, swd, ept, short_name, split.
        - Non-metric lines are empty or contain no '&'.
    Returns:
        - rows: list[ModelRow] [N], preserving input order.
    """
    rows: List[ModelRow] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "&" not in line:
            continue  # skip header / blank lines

        parts = [p.strip().rstrip("\\") for p in line.split("&")]
        # REQUIRE: at least 6 parts: tag, mse, mae, swd, ept, short_name, ...
        tag, mse_s, mae_s, swd_s, ept_s, short_name, *_ = parts
        short = short_name.split()[0]

        rows.append(
            ModelRow(
                short_name=short,
                mse=float(mse_s),
                mae=float(mae_s),
                swd=float(swd_s),
                ept=float(ept_s),
            )
        )
    return rows


# === Ranking ===

def compute_metric_ranks(rows: List[ModelRow]) -> Dict[str, Dict[str, float]]:
    """Intent: compute per-model ranks for MSE, SWD, and combined rank.
    
    Assumptions:
        - Lower MSE and SWD are better.
        - rows: list[ModelRow] [N] with unique short_name.
    Returns:
        - ranks: short_name -> {'mse_rank', 'swd_rank', 'rank} (float).
    """
    short_names = [r.short_name for r in rows]  # [N]
    mse_vals = [r.mse for r in rows]           # [N]
    swd_vals = [r.swd for r in rows]           # [N]

    order_mse = sorted(range(len(rows)), key=lambda i: mse_vals[i])
    order_swd = sorted(range(len(rows)), key=lambda i: swd_vals[i])

    mse_rank: Dict[str, int] = {}
    swd_rank: Dict[str, int] = {}

    for rank, idx in enumerate(order_mse, start=1):
        mse_rank[short_names[idx]] = rank
    for rank, idx in enumerate(order_swd, start=1):
        swd_rank[short_names[idx]] = rank

    ranks: Dict[str, Dict[str, float]] = {}
    for name in short_names:
        r_mse = mse_rank[name]
        r_swd = swd_rank[name]
        ranks[name] = {
            "mse_rank": float(r_mse),
            "swd_rank": float(r_swd),
            "rank": (r_mse + r_swd) / 2.0,
        }
    return ranks


# === Formatting for rebuttal ===

def format_block(
    raw: str,
    header: str,
    base_short: str = "tm",
    ours_short: str = "fr",
    decimals_metric: int = 3,
) -> str:
    """Intent: turn a latex metrics block into a markdown-friendly summary.
    
    Assumptions:
        - `raw` contains one dataset block with `&`-separated rows.
        - `header` is the heading line (e.g. 'Lorenz-Base (MSE ↓ / SWD ↓)...').
        - `base_short` is the short_name used as the relative baseline (e.g. 'tm').
        - `ours_short` is the method to label as '(ours)'.
    Returns:
        - Multi-line markdown string ready to paste.
    """
    rows = parse_latex_block(raw)  # [N]

    # GUARD-OK: base model must exist to define relative factors
    by_name: Dict[str, ModelRow] = {r.short_name: r for r in rows}
    if base_short not in by_name:
        raise ValueError(f"Base model '{base_short}' not found in rows.")

    base_row = by_name[base_short]
    ranks = compute_metric_ranks(rows)

    # Identify best and second-best per metric for ** / *
    mse_sorted = sorted(rows, key=lambda r: r.mse)
    swd_sorted = sorted(rows, key=lambda r: r.swd)
    best_mse = mse_sorted[0].short_name
    second_mse = mse_sorted[1].short_name if len(mse_sorted) > 1 else None
    best_swd = swd_sorted[0].short_name
    second_swd = swd_sorted[1].short_name if len(swd_sorted) > 1 else None

    def format_name(short: str) -> str:
        return LONG_NAMES.get(short, short)
        # if short == ours_short:
        #     return f"{short} (ours)"
        # if short == base_short:
        #     return f"{short} (base)"
        # return short

    display_names: Dict[str, str] = {r.short_name: format_name(r.short_name) for r in rows}
    name_width = max(len(name) for name in display_names.values())

    def fmt_metric(value: float) -> str:
        return f"{value:.{decimals_metric}f}"

    def fmt_rel(numer: float, denom: float) -> str:
        return f"{numer / denom:.2f}×"

    lines: List[str] = [header, ""]

    for r in rows:  # preserves input order
        short = r.short_name
        name_field = display_names[short].ljust(name_width)

        mse_str = fmt_metric(r.mse)
        swd_str = fmt_metric(r.swd)

        # Highlight best / second-best with ** / *
        if short == best_mse:
            mse_str = f"**{mse_str}**"
        elif second_mse is not None and short == second_mse:
            mse_str = f"*{mse_str}*"

        if short == best_swd:
            swd_str = f"**{swd_str}**"
        elif second_swd is not None and short == second_swd:
            swd_str = f"*{swd_str}*"

        rel_mse = fmt_rel(r.mse, base_row.mse)
        rel_swd = fmt_rel(r.swd, base_row.swd)
        rank_val = ranks[short]["rank"]

        line = (
            f"{name_field}:  MSE {mse_str}, SWD {swd_str}  |  "
            f"{rel_mse} / {rel_swd} {base_short}  | rank {rank_val:.2f}"
        )
        lines.append(line)

    return "\n".join(lines)


# raw_slds = """SLDS BASE 
# dl-AVG & 4.446 & 1.465 & 3.469 & 64.897 & dl & ALL \\
# fr-AVG & 5.090 & 1.645 & 2.613 & 47.905 & fr & ALL \\
# kp-AVG & 6.059 & 1.927 & 4.242 & 34.419 & kp & ALL \\
# mtcn-AVG & 7.733 & 1.945 & 5.549 & 36.239 & mtcn & ALL \\
# tm-AVG & 4.675 & 1.650 & 3.008 & 31.379 & tm & ALL \\
# tst-AVG & 5.619 & 1.770 & 3.654 & 43.615 & tst & ALL \\
# """

# print(
#     format_block(
#         raw_slds,
#         header="SLDS-Base (MSE ↓ / SWD ↓). Base = TimeMixer (tm).",
#         base_short="tm",
#         ours_short="fr",
#     )
# )

# region Claude ver helpers
def format_nonstationarity_table(data_str, base_model='tm', title=''):
    """
    Parse raw LaTeX table data and format as markdown with relative performance.
    
    Args:
        data_str: String containing LaTeX table rows (model-AVG & MSE & MAE & SWD & EPT...)
        base_model: Model code to use as baseline (default 'tm')
        title: Optional title for the table
    """
    import re
    
    # Parse the data
    lines = [l.strip() for l in data_str.strip().split('\n') if l.strip()]
    results = []
    
    for line in lines:
        # Match pattern: model-AVG & MSE & MAE & SWD & EPT & model & ALL \\
        parts = [p.strip() for p in line.split('&')]
        if len(parts) >= 4:
            model_name = parts[0].replace('-AVG', '').strip()
            mse = float(parts[1].strip())
            mae = float(parts[2].strip())
            swd = float(parts[3].strip())
            ept = float(parts[4].strip()) if len(parts) > 4 else 0.0
            
            results.append({
                'model': model_name,
                'mse': mse,
                'mae': mae,
                'swd': swd,
                'ept': ept
            })
    
    # Find base model values
    base = next((r for r in results if r['model'] == base_model), None)
    if not base:
        raise ValueError(f"Base model '{base_model}' not found in data")
    
    base_mse = base['mse']
    base_swd = base['swd']
    
    # Calculate ranks (lower is better for MSE and SWD)
    mse_sorted = sorted(results, key=lambda x: x['mse'])
    swd_sorted = sorted(results, key=lambda x: x['swd'])
    
    for r in results:
        r['mse_rank'] = mse_sorted.index(r) + 1
        r['swd_rank'] = swd_sorted.index(r) + 1
        r['avg_rank'] = (r['mse_rank'] + r['swd_rank']) / 2
        r['mse_vs_base'] = r['mse'] / base_mse
        r['swd_vs_base'] = r['swd'] / base_swd
    
    # Sort by average rank
    results.sort(key=lambda x: x['avg_rank'])
    
    # Find best and second best for each metric
    best_mse = min(results, key=lambda x: x['mse'])
    best_swd = min(results, key=lambda x: x['swd'])
    second_mse = sorted(results, key=lambda x: x['mse'])[1]
    second_swd = sorted(results, key=lambda x: x['swd'])[1]
    
    # Format output
    output = []
    if title:
        output.append(f"{title}")
        output.append("")
    
    model_name_map = {
        'dl': 'DLinear',
        'fr': 'FERN (ours)',
        'kp': 'Koopman',
        'mtcn': 'ModernTCN',
        'pfnn': 'PFNN',
        'tm': 'TimeMixer',
        'tst': 'Transformer'
    }
    
    for r in results:
        model_display = model_name_map.get(r['model'], r['model'])
        
        # Format MSE with emphasis
        if r['model'] == best_mse['model']:
            mse_str = f"**{r['mse']:.2f}**"
        elif r['model'] == second_mse['model']:
            mse_str = f"*{r['mse']:.2f}*"
        else:
            mse_str = f"{r['mse']:.2f}"
        
        # Format SWD with emphasis
        if r['model'] == best_swd['model']:
            swd_str = f"**{r['swd']:.2f}**"
        elif r['model'] == second_swd['model']:
            swd_str = f"*{r['swd']:.2f}*"
        else:
            swd_str = f"{r['swd']:.2f}"
        
        # Format relative performance
        if r['model'] == base_model:
            rel_str = "1.00× / 1.00×"
        else:
            rel_str = f"{r['mse_vs_base']:.2f}× / {r['swd_vs_base']:.2f}× {base_model}"
        
        # Build the line with proper spacing
        line = f"{model_display:18s} MSE {mse_str:>8s}, SWD {swd_str:>8s}  |  {rel_str:20s} | rank {r['avg_rank']:.1f}"
        output.append(line)
    
    return '\n'.join(output)


# Usage example:
# data = """
# dl-AVG & 4.446 & 1.465 & 3.469 & 64.897 & dl & ALL \\
# fr-AVG & 5.090 & 1.645 & 2.613 & 47.905 & fr & ALL \\
# kp-AVG & 6.059 & 1.927 & 4.242 & 34.419 & kp & ALL \\
# mtcn-AVG & 7.733 & 1.945 & 5.549 & 36.239 & mtcn & ALL \\
# tm-AVG & 4.675 & 1.650 & 3.008 & 31.379 & tm & ALL \\
# tst-AVG & 5.619 & 1.770 & 3.654 & 43.615 & tst & ALL \\
# """

# print(format_nonstationarity_table(
#     data, 
#     base_model='tm',
#     title='SLDS-Base (MSE ↓ / SWD ↓). Base = TimeMixer (tm).'
# ))
#endregion