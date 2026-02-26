import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple
import polars as pl
import numpy as np
from typing import (
    Optional, List, Literal, Union, Callable, Dict, Tuple, Any, Protocol, runtime_checkable
) 
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import study.fr_cfg as configs
import pendulum
from functools import cached_property, partial 
from pathlib import Path 
from enum import StrEnum, auto
import functools
from torch.utils.data import default_collate
import einops
import study.fr_data_gen as data_gen

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

# --- in data_mgr.py ---
def _stack_and_slice_tde(batch, pred_len: int):
    """
    intent/contract:
        Stack a batch of (x:[S,D], y:[L+H,D]) pairs and slice prediction horizon.
        returns: xs:[B,S,D], ys:[B,H,D]
    """
    # === Stack both tensors in one go (less Python work) ===
    xs, ys = default_collate(batch)                       # CHG: use default_collate to stack both
    ys = ys[:, -pred_len:, :]                             # [B,H,D]
    return xs, ys


def collate_tde(batch, pred_len: int):
    """
    intent/contract:
        Default TDE collate. Each item is (seq_x:[S,D], seq_y:[L+H,D]).
        returns: xs:[B,S,D], ys:[B,H,D]
    """
    xs, ys = _stack_and_slice_tde(batch, pred_len)        # CHG: deduplicate logic via helper
    return xs, ys


def collate_tde_ci(batch, pred_len: int):
    """
    intent/contract:
        Channel-independent collate: fold channel into batch.
        shapes: x:[B,S,D]→[B·D,S,1], y:[B,H,D]→[B·D,H,1]
    """
    xs, ys = _stack_and_slice_tde(batch, pred_len)        # CHG: share stack/slice
    B, S, D = xs.shape
    H = ys.shape[1]

    # === Fold channel into batch (no unnecessary copies) ===
    xs = einops.rearrange(xs, 'b s d -> (b d) s 1')  # [B·D, S, 1]
    ys = einops.rearrange(ys, 'b h d -> (b d) h 1')  # [B·D, H, 1]
    return xs, ys


class TimeSeriesDataset(Dataset):
    """A PyTorch Dataset for time series data.""" 
    def __init__(self, *, data_x, data_y, seq_len, label_len, pred_len):
        self.data_x = data_x
        self.data_y = data_y
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        if seq_len <= 0 or label_len < 0 or pred_len <= 0:
            raise ValueError("Sequence lengths must be positive integers.")

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        input_start = index
        input_end = input_start + self.seq_len
        target_start = input_end - self.label_len
        target_end = target_start + self.label_len + self.pred_len

        seq_x = self.data_x[input_start:input_end]
        seq_y = self.data_y[target_start:target_end] 
        return seq_x, seq_y

class TimeSeriesDatasetStrided(Dataset):  
    def __init__(self, *, data_x: torch.Tensor, data_y: torch.Tensor, 
                 seq_len: int, label_len: int, pred_len: int):  
        if not (data_x.is_contiguous() and data_y.is_contiguous()):
            raise ValueError("TimeSeriesDatasetStrided requires contiguous torch tensors.")
        N, D = data_x.shape  # N = time steps, D = features/channels
        n_windows = N - seq_len - pred_len + 1
        if n_windows <= 0:
            raise ValueError(f"Not enough length: N={N}, S={seq_len}, H={pred_len}")
        """
        data_x.as_strided is making a [N D] into [B S D]/[n_windows, seq_len, D] where each B is a window,
        there is N-seq-pred+1 of them, each moving [0, ... 10] of seq=3 is [a0,a1,a2] [a1,a2,a3] 
        is your ORIGINAL stride on [N, D] which is 3 since you cross [b0 b1 b2];
        S is the movement again from a0 to a1 which by pass b0 b1 b2, And D dim is a0 to b0 so it is 1
         
        as_strided parameters:
        - size=(n_windows, seq_len, D): Shape of the NEW VIEW
        - stride=(row_stride, row_stride, col_stride): How to navigate this view
          
        stride[0] = row_stride: Moving to next WINDOW shifts start by 1 time step
        stride[1] = row_stride: Moving within a window shifts by 1 time step
        stride[2] = col_stride: Moving to next feature shifts by 1 element 
        """
        row_stride, col_stride = data_x.stride()  # Original strides of [N, D]
        self.X = data_x.as_strided(
            size=(n_windows, seq_len, D),
            stride=(row_stride, row_stride, col_stride),
            storage_offset=0,  # Start from beginning of data_x
        )

        total_t = label_len + pred_len  # Total target sequence length
        
        self.Y = data_y.as_strided(
            size=(n_windows, total_t, D),
            stride=(row_stride, row_stride, col_stride),  # Same stride pattern as X
            storage_offset=(seq_len - label_len) * row_stride,  # But shifted start position
        )
        self.data_x = data_x
        self.data_y = data_y
        self.n_windows = n_windows
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
    
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, index: int): 
        return self.X[index], self.Y[index]

##### Config Classes ##### 
class SynDataID(StrEnum):
    LORENZ = auto() ; HENON = auto() ; ROSSLER = auto() ; HYPER_ROSSLER = auto() ; LOGISTIC = auto()  # noqa: E702
    DUFFING = auto() ; LORENZ96 = auto() ; CHUA = auto() ;  
    # STOCHASTIC ONES
    OU = auto() ; SLDS = auto() ; DOUBLEWELL = auto(); SEASONAL_AR = auto() ; GARCH = auto(); KS = auto();

ID_to_Params = {
    SynDataID.LORENZ: data_gen.LorenzParams,
    SynDataID.ROSSLER: data_gen.RosslerParams,
    SynDataID.HYPER_ROSSLER: data_gen.HyperRosslerParams,
    SynDataID.LOGISTIC: data_gen.LogisticParams,
    SynDataID.DUFFING: data_gen.DuffingParams,
    SynDataID.LORENZ96: data_gen.Lorenz96Params,
    SynDataID.CHUA: data_gen.ChuaParams,
    
    SynDataID.HENON: data_gen.HenonParams,
    
    SynDataID.LOGISTIC: data_gen.LogisticParams,
    
    SynDataID.OU: data_gen.OUParams,
    SynDataID.SLDS: data_gen.SLDSParams,
    SynDataID.DOUBLEWELL: data_gen.DoubleWellParams,
    SynDataID.SEASONAL_AR: data_gen.SeasonalARParams,
    SynDataID.GARCH: data_gen.GARCHParams,
    SynDataID.KS: data_gen.KSParams,
    
}

class KnownDataID(StrEnum):
    ETTH1 = auto() ; ETTH2 = auto() ; ETTM1 = auto() ; ETTM2 = auto() ; WEATHER = auto()
 
class SourceConfig(BaseModel):
    dataset_id: SynDataID | KnownDataID = Field(..., description="Source dataset identifier.")

    # for csv
    path: Optional[Path] = Field(None, description="Single filename assumed to be in the './datasets/' directory.")
    date_column: Optional[str] = None #"date"
    random_feature_num_seed_pair: Optional[Tuple[int, int]] = None
    hard_sentinels: list[float | int] | None = None

    # for synthetic
    params: BaseModel | None = None  # e.g., LorenzParams, HenonParams, etc.
    
    @field_validator("path", mode="before")
    @classmethod
    def assemble_path(cls, v): 
        if v is None: # CHG: allow None (synthetic mode), str, or Path
            return None
        if isinstance(v, Path):
            return v
        
        p = Path(v) # v is str
        if len(p.parts) == 1:  # CHG: no-op print unless you want loud logging everywhere
            # print(f"-> Path '{v}' treated as filename; prefixing './datasets/'.")
            return Path("./datasets") / p
        return p
    
    @model_validator(mode="after")
    def fill_param_and_path(self) -> "SourceConfig": 
        if isinstance(self.dataset_id, SynDataID):
            # REQUIRE: params provided
            if self.params is None:
                raise ValueError(f"Params must be provided for synthetic dataset '{self.dataset_id}'.")
            self.path = None  # CHG: enforce no path in synthetic mode 
            self.date_column = None # date_column irrelevant in synthetic mode
        elif isinstance(self.dataset_id, KnownDataID):
            # REQUIRE: path provided
            if self.path is None:
                raise ValueError(f"Path must be provided for CSV dataset '{self.dataset_id}'.")
            self.params = None  # CHG: enforce no params in CSV mode
            if self.date_column is None:
                self.date_column = "date"  # CHG: default once, here
            if self.hard_sentinels is None:
                self.hard_sentinels = [-9999, -9999.0]  # CHG: default once, here
        else:
            raise ValueError(f"Unknown datasetID: {self.dataset_id}")
        return self

# final_bundle = (
#         initial_bundle
#         | stage_load_raw_data
#         | stage_known_problem_fixed          # <— NEW
#         | stage_split_and_scale_data
#         | stage_move_to_window_view
#         | stage_create_dataloaders
#         | stage_compute_koopa_mask         # <— NEW 
#     )

class KnownDatasetMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    sampling_seconds: int = Field(..., gt=0, description="e.g., 3600 (hourly), 900 (15-min), 600 (10-min)")
    sentinels: Dict[str, List[float]]         # { column_name: [value1, value2, ...] } 

    # --- fix step 1: should you drop columns? ---
    lv1_threshold_sentinel_density_for_drop_col: float = Field(1.0, ge=0.0, le=1.0, 
        description="""Drop column if >x% of values are sentinels (ANY sentinel). 1 = disable.""")

    lv2_threshold_zero_density_for_drop_col: float = Field(1.0, ge=0.0, le=1.0, 
        description="""Drop column if >x% of values are near-zero. 1 = disable.""")
    
    # or just transform the sparse columns
    lv1_threshold_zero_density_for_transform_col: float = Field(1.0, ge=0.0, le=1.0, 
        description="""Transform column if >x% of values are near-zero. 1 = disable.""")
    
    sparse_transform_method: Literal["asinh", "none"] = "asinh"

    lv3_threshold_hours_for_drop_col: int = Field(24*7, ge=0,
        description="""Drop column if longerst zero runs > x hours. 0 = disable.
        E.g. 3 hours min 6 hours max => ETTh2 6+ consecutive zeroes in one column = column gets removed.
        """)
    
    # --- fix step 2: delete rows? --- 
    # over long continguous zeroes? We cannot impute very long hours.
    lv2_threshold_hours_for_drop_row: int = Field(12, ge=0,
        description="""Drop ANY zero run rows if lv2_threshold_hours_for_drop_row < zero run < lv3_threshold_hours_for_drop_col. 0 = disable. 
        E.g. 3 hours min 6 hours max => ETTh2 3-6 consecutive zeroes in one column = row gets removed. """)

    # --- fix step 3: impute zero rows? Not for rainfalls but some occasional sensor failures. ---
    lv1_threshold_hours_for_impute: int = Field(3, ge=0,
        description="""Impute ANY zero run if zero run ≤ lv1_threshold_hours_for_impute. 0 = disable.
        If=3 then in ETTh1 1 or 2 consecutive zeroes in ETTm2 under 4*2=8 consecutive zeroes get imputed. """)
    
    impute_method: Literal["ffill_bfill", "linear", "nearest", "none"] = "ffill_bfill"
 
    @computed_field
    @property
    def steps_per_hour(self) -> int:
        return int(round(3600 / self.sampling_seconds))

    @computed_field
    @property
    def lv3_threshold_steps_for_drop_col(self) -> int:
        return int(round(self.lv3_threshold_hours_for_drop_col * self.steps_per_hour))
    
    @computed_field
    @property
    def lv2_threshold_steps_for_drop_row(self) -> int:
        return int(round(self.lv2_threshold_hours_for_drop_row * self.steps_per_hour))

    @computed_field
    @property
    def lv1_threshold_steps_for_impute(self) -> int:
        return int(round(self.lv1_threshold_hours_for_impute * self.steps_per_hour))
 
     
# Minimal registry you can extend anytime 
# === Defaults ===
# intent: tiny factory with shared defaults; avoids repetition without hiding fields.
def default_meta(sampling_seconds: int, sentinels: Optional[Dict[str, List[float]]] = None) -> KnownDatasetMeta:
    return KnownDatasetMeta(
        sampling_seconds=sampling_seconds,
        sentinels=(sentinels or {}),            # REQUIRE: dict[str, list[float]]
        lv1_threshold_sentinel_density_for_drop_col=0.10,
        lv2_threshold_zero_density_for_drop_col=0.15,
        lv1_threshold_zero_density_for_transform_col=0.10,
        lv3_threshold_hours_for_drop_col=24 * 7,
        lv2_threshold_hours_for_drop_row=3, # less than 3 imputed; more than 3 and less then 1 week dropped.
        lv1_threshold_hours_for_impute=3,
        impute_method="ffill_bfill",
        sparse_transform_method="asinh",
    )

# === Table ===
KNOWN_DATASETS: Dict[KnownDataID, KnownDatasetMeta] = {
    KnownDataID.ETTH2: default_meta(
        3600,
        {"LULL": [-31.46199989, -28.15699959], "MUFL": [88.29799652]},
    ),
    KnownDataID.ETTM2: default_meta(
        900,
        {"LULL": [-31.46199989, -28.15699959], "MUFL": [88.29799652]},
    ),
    KnownDataID.WEATHER: default_meta(
        600,
        {"OT": [-9999]},
    ),
    KnownDataID.ETTH1: default_meta(3600),
    KnownDataID.ETTM1: default_meta(900),
}

class KnownFixConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)
 
    dataset_id: KnownDataID = Field(..., description="Dataset identifier for known fix.")
    meta: KnownDatasetMeta = Field(..., description="Metadata for the known dataset.")

    value_tol: float = Field(1e-6, ge=0.0, description="Absolute tolerance for near-zero/sentinel tests.")
 
    @classmethod
    def construct_known_fix_with_default(
        cls,
        dataset_id: KnownDataID,
        meta: KnownDatasetMeta,
        *,
        long_run_hours: int = 7*24,
        value_tol: float = 1e-6,
    ) -> "KnownFixConfig":
        """Factory that pulls defaults from the registry, allowing overrides."""
        reg = KNOWN_DATASETS[dataset_id]  # CHG: use [] to fail loudly if missing
        return cls(
            # enable=True,
            dataset_id=dataset_id, 
            meta=reg,
            value_tol=value_tol, 
        )

class RawData(BaseModel):
    """
    Intent/contract:
        Load raw feature matrix + names from a SourceConfig.
        - Synthetic: use the config's generator.
        - CSV: Lazy-scan with Polars, drop the date column if specified, drop nulls, return polars dataframe. 
    Returns:
        RawData(data: [N, D] float32, feature_names: List[str])
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    dataset_id: SynDataID | KnownDataID = Field(..., description="Source dataset identifier.")
    backend: Literal["numpy", "polars"]  # CHG: explicit backend tag
    data_np: Optional[np.ndarray] = Field(default=None, description="[N, D] if backend='numpy'")
    data_pl: Optional[pl.DataFrame] = Field(default=None, description="DataFrame if backend='polars'")
    feature_names: list[str]

    @property
    def channels(self) -> int:
        # CHG: unified channel count
        if self.backend == "numpy":
            return int(self.data_np.shape[1])  # REQUIRE: data_np not None
        return len(self.data_pl.columns)

    def to_numpy(self) -> np.ndarray:
        # CHG: single, explicit materialization point
        if self.backend == "numpy":
            return self.data_np.astype(np.float32, copy=False)

        arr = self.data_pl.to_numpy()
        if arr.ndim != 2:
            raise ValueError(f"Table must be 2D; got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)
    # === Loader ===
    @classmethod
    def from_config(cls, source_cfg: "SourceConfig") -> "RawData":
        """
        # REQUIRE:
        # - source_cfg.type in {"synthetic"} or isinstance(source_cfg, CSVSourceConfig)
        # - CSV path exists and is readable
        # - If date_column is set, it must exist in the CSV header
        """
        print(f"-> Loading from {getattr(source_cfg, 'name', '<unnamed>')}")

        # --- Synthetic (NumPy backend) ---
        if isinstance(source_cfg.dataset_id, SynDataID):
            data = source_cfg.params.generate()  # REQUIRE: np.ndarray [N, D]
            if not isinstance(data, np.ndarray) or data.ndim != 2:
                raise TypeError(f"Synth generator must return 2D numpy array; got {type(data)} shape={getattr(data,'shape',None)}")
            feature_names = [f"f{i}" for i in range(data.shape[1])]  # CHG: transparent default names
            return cls(dataset_id=source_cfg.dataset_id, backend="numpy", data_np=data, data_pl=None, feature_names=feature_names)

        # --- CSV (Polars backend) ---
        if isinstance(source_cfg.dataset_id, KnownDataID):
            print(f"   Scanning {source_cfg.path} with Polars...")
            
            # null_values = str(source_cfg.hard_sentinels) if source_cfg.hard_sentinels else None
            # # CHG: accept one or many hard sentinels; pass through as list or dict (strings) 
            if getattr(source_cfg, "hard_sentinels", None) is not None:
                hs = source_cfg.hard_sentinels
                null_values = [str(x) for x in (hs if isinstance(hs, (list)) else [hs])]     # global
            else:
                null_values = None 
            assert source_cfg.path is not None, "Path must be set for CSV dataset."
            lf = pl.scan_csv(source_cfg.path, null_values=null_values)

            date_col = getattr(source_cfg, "date_column", None)
            if date_col:
                cols_now = lf.columns
                if date_col not in cols_now:
                    raise ValueError(f"date_column='{date_col}' not found in CSV header: {cols_now}")
                print(f"   Excluding date column: '{date_col}'")
                lf = lf.select(pl.exclude(date_col))

            df = lf.drop_nulls().collect()
            feature_names = df.columns
            print(f"   Final features being used: {feature_names}")

            # CHG: keep Polars backend here; convert later via .to_numpy()
            return cls(dataset_id=source_cfg.dataset_id, backend="polars", data_np=None, data_pl=df, feature_names=feature_names)

        raise ValueError(f"Unsupported dataset id type: {type(source_cfg.dataset_id).__name__}")
 

class CleanReport(BaseModel):
    """
    intent/contract:
        Summary of cleaning actions.
        returns: dropped_cols (list[str]), rows_dropped (int), imputed_cols (list[str]).
    """
    dropped_cols: list[str]
    rows_dropped: int
    imputed_cols: list[str]
    rows_masked_any: int
    rows_masked_per_col: Dict[str, int]
    
class CleanData(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    dataset_id: Union[SynDataID, KnownDataID] = Field(..., description="Source dataset identifier.")
    backend: Literal["numpy", "polars"]  # CHG: explicit backend tag
    data_np: Optional[np.ndarray] = Field(default=None, description="[N, D] if backend='numpy'")
    data_pl: Optional[pl.DataFrame] = Field(default=None, description="DataFrame if backend='polars'")
    feature_names: list[str]
    known_fix: Optional[KnownFixConfig] = Field(default=None, description="Known fix configuration for csv datasets.")
    clean_report: CleanReport = Field(default_factory=lambda: CleanReport(dropped_cols=[], rows_dropped=0, imputed_cols=[], rows_masked_any=0, rows_masked_per_col={}), description="Data cleaning report.")
    
    @model_validator(mode="after")
    def validate_backend_data(self) -> "CleanData":
        if self.backend == "numpy" and self.data_np is None:
            raise ValueError("backend='numpy' requires data_np to be set.")
        if self.backend == "polars" and self.data_pl is None:
            raise ValueError("backend='polars' requires data_pl to be set.")
        return self

    @property
    def channels(self) -> int: 
        if self.backend == "numpy":
            assert self.data_np is not None 
            return int(self.data_np.shape[1])  # REQUIRE: data_np not None
        assert self.data_pl is not None
        return len(self.data_pl.columns)
    
    def to_numpy(self) -> np.ndarray:
        # CHG: single, explicit materialization point
        if self.backend == "numpy":
            assert self.data_np is not None
            return self.data_np.astype(np.float32, copy=False)

        assert self.data_pl is not None
        arr = self.data_pl.to_numpy()
        if arr.ndim != 2:
            raise ValueError(f"Table must be 2D; got shape {arr.shape}")
        return arr.astype(np.float32, copy=False)

    # === Core building blocks ===
    @staticmethod
    def _max_run_len(df: pl.DataFrame, cond: pl.Expr) -> pl.Expr:
            """
            === Run-length via change cumsum ===
            Step 1: Boolean condition - which rows are near-zero?
            Polars: pl.col(c) creates an expression for column c 
            Step 2: Run-Length Encoding via cumsum trick
            
            Breakdown:
            - boolean series [True, True, False, True, False, ...] 
            - cond.shift(1): Shift down by 1 row [null, True, True, False, True, ...]
            - cond != cond.shift(1): True where value CHANGES [T, F, T, T, T] 
            - .cast(pl.Int32): Convert boolean to 0/1 [1, 0, 1, 1, 1]
            - .cumsum(): Running sum [1, 1, 2, 3, 4]
            Result: Each contiguous segment gets same group ID
            
            Step 3: Compute run length for each group, assign back to rows
            Breakdown:
            - pl.len().over(grp): For each group, count how many rows in that group
              Example: grp=[1,1,2,3,4] → lengths=[2,2,1,1,1]
            - pl.when(cond).then(...).otherwise(0): Only keep lengths where cond=True (near-zero)
              If cond=False (not near-zero), set length to 0
            
            Example walkthrough:
            col:    [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0]
            cond:   [T,   T,   T,   F,   T,   T,   F  ]
            shift:  [N,   T,   T,   T,   F,   T,   T  ]
            !=:     [T,   F,   F,   T,   T,   F,   T  ]  (change points)
            cumsum: [1,   1,   1,   2,   3,   3,   4  ]  (group IDs)
            len:    [3,   3,   3,   1,   2,   2,   1  ]  (group sizes)
            runlen: [3,   3,   3,   0,   2,   2,   0  ]  (0 for non-zero groups)
            Step 4: Extract maximum run length
            df.select(...).item(): Execute the expression and get scalar result
            In our example: max([3,3,3,0,2,2,0]) = 3
            """
            change = (cond != cond.shift(1)).fill_null(True) 
            grp    = change.cast(pl.Int32).cum_sum()
            runlen = pl.when(cond).then(pl.len().over(grp)).otherwise(0) 
            return runlen.max() # a lazy expr, compute via df.select(...).item() later

    @staticmethod
    def _runlen_expr(cond: pl.Expr) -> pl.Expr:
        """Expr that returns per-row run length where `cond` True, else 0."""
        change = (cond != cond.shift(1)).fill_null(True)
        grp    = change.cast(pl.Int32).cum_sum()
        return pl.when(cond).then(pl.len().over(grp)).otherwise(0)
        """example usage
        df = df.filter(~(r >= pl.lit(3))) # Drop long runs filter out rows where run length is >= 3
        """

    @staticmethod
    def _cond_any(col: str, zero_target:float, sentinels: Dict[str, List[float]] | None, tol: float, ) -> Tuple[pl.Expr, pl.Expr, pl.Expr]:
        """Build zero/all sentinel/their combination expr for column c."""
        cond_zero = (pl.col(col) - pl.lit(zero_target)).abs() <= tol
        if sentinels and (vals := sentinels.get(col)): # := is walrus
            cond_sentinel_list = [ (pl.col(col) - pl.lit(v)).abs() <= tol for v in vals ]
            cond_sentinel = pl.fold(
                acc=pl.lit(False), 
                function=lambda acc, e: acc | e,
                exprs=cond_sentinel_list
                )
        else:
            cond_sentinel = pl.lit(False)
        cond_any = cond_zero | cond_sentinel
        return cond_zero, cond_sentinel, cond_any
    
    # === Stage 1: Analyze (single select for all columns) ===
    @staticmethod
    def _analyze_columns(
        df: pl.DataFrame,
        *,
        sentinels: Dict[str, List[float]] | None,
        zero_target: float,
        tol: float,
    ) -> Dict[str, Dict[str, Any]]:
        N = int(df.height or 0)
        cols = list(df.columns)

        all_analysis_exprs: List[pl.Expr] = []
        for c in cols:
            cond_zero, cond_sentinel, cond_any = CleanData._cond_any(c, zero_target, sentinels, tol)
            all_analysis_exprs.extend([
                cond_zero.sum().alias(f"{c}__zero_count"),
                cond_sentinel.sum().alias(f"{c}__sentinel_count"),
                cond_any.sum().alias(f"{c}__any_count"),
                CleanData._runlen_expr(cond_zero).max().alias(f"{c}__max_zero_run"),
                CleanData._runlen_expr(cond_sentinel).max().alias(f"{c}__max_sentinel_run"),
                CleanData._runlen_expr(cond_any).max().alias(f"{c}__max_any_run"),
            ])
        if not all_analysis_exprs:
            return {}

        results_df = df.select(all_analysis_exprs)  # 1 × (6·D)

        metrics: Dict[str, Dict[str, Any]] = {}
        for c in cols:
            zero_count      = int(results_df[f"{c}__zero_count"].item())
            sentinel_count  = int(results_df[f"{c}__sentinel_count"].item())
            any_count       = int(results_df[f"{c}__any_count"].item())
            max_zero_run    = int(results_df[f"{c}__max_zero_run"].item())    if zero_count     else 0
            max_sentinel_run= int(results_df[f"{c}__max_sentinel_run"].item())if sentinel_count else 0
            max_any_run     = int(results_df[f"{c}__max_any_run"].item())     if any_count      else 0

            zero_frac     = (zero_count / N)    if N else 0.0
            sentinel_frac = (sentinel_count / N)if N else 0.0
            any_frac      = (any_count / N)     if N else 0.0

            metrics[c] = {
                "zero_count": zero_count, "zero_frac": zero_frac, "max_zero_run": max_zero_run,
                "sentinel_count": sentinel_count, "sentinel_frac": sentinel_frac, "max_sentinel_run": max_sentinel_run,
                "any_count": any_count, "any_frac": any_frac, "max_any_run": max_any_run,
            }
        return metrics
    
    # === Stage 2: Drop columns ===
    @staticmethod
    def _drop_bad_columns(
        df: pl.DataFrame,
        metrics: Dict[str, Dict[str, Any]],
        *,
        lv1_threshold_sentinel_density_for_drop_col: float,
        lv2_threshold_zero_density_for_drop_col: float,
        lv3_threshold_steps_for_drop_col: int,
        verbose: bool,
    ) -> Tuple[pl.DataFrame, List[str]]:
        drop_columns: List[str] = []
        for c, m in metrics.items():
            drop_by_sentinel_frac = (m["sentinel_frac"] >= lv1_threshold_sentinel_density_for_drop_col)
            drop_by_zero_frac     = (m["zero_frac"]     >= lv2_threshold_zero_density_for_drop_col)
            drop_by_long_run      = (lv3_threshold_steps_for_drop_col > 0 and m["max_any_run"] >= lv3_threshold_steps_for_drop_col)
            if drop_by_sentinel_frac or drop_by_zero_frac or drop_by_long_run:
                drop_columns.append(c)

        if drop_columns:
            if verbose:
                print(f"🧊 Drop columns: {drop_columns}", flush=True)
            df2: pl.DataFrame = df.drop(drop_columns)
        else:
            df2 = df
        return df2, drop_columns

    # === Stage 3: Drop rows with long runs ===
    @staticmethod
    def _drop_bad_rows(
        df: pl.DataFrame,
        *,
        sentinels: Dict[str, List[float]] | None,
        zero_target: float,
        tol: float,
        lv2_threshold_steps_for_drop_row: int,
        verbose: bool,
    ) -> Tuple[pl.DataFrame, int]:
        if lv2_threshold_steps_for_drop_row <= 0 or df.width == 0:
            return df, 0

        rowmask_parts: List[pl.Expr] = []
        for c in df.columns:
            _, _, cond_any = CleanData._cond_any(c, zero_target, sentinels, tol)
            runlen = CleanData._runlen_expr(cond_any)
            rowmask_parts.append(runlen >= pl.lit(lv2_threshold_steps_for_drop_row))

        rowmask: pl.Expr = pl.fold(acc=pl.lit(False), function=lambda acc, e: acc | e, exprs=rowmask_parts) if rowmask_parts else pl.lit(False)
        rows_to_drop = int(df.select(rowmask.sum()).item())
        if rows_to_drop > 0:
            df = df.filter(~rowmask)
            if verbose:
                print(f"✂️ Drop rows (runs ≥ {lv2_threshold_steps_for_drop_row}): {rows_to_drop}", flush=True)
        
        # === One-line fail-fast after filtering as well (paranoia check) ===
        if df.width and df.select(pl.any_horizontal(pl.all().is_null().sum() == df.height)).item():
            raise ValueError("All-null column detected after row-drop.")
        return df, rows_to_drop

# === Stage 4: Impute short runs ===
    @staticmethod
    def _impute_short_runs(
        df: pl.DataFrame,
        *,
        sentinels: Dict[str, List[float]] | None,
        zero_target: float,
        tol: float,
        lv1_threshold_steps_for_impute: int,
        impute_method: Literal["ffill_bfill","linear","nearest", "none"],
        verbose: bool,
    ) -> Tuple[pl.DataFrame, int, Dict[str, int]]:
        if impute_method == "none" or lv1_threshold_steps_for_impute <= 0:
            return df, 0, {}

        masked_exprs: List[pl.Expr] = []
        for c in df.columns:
            _, _, cond_any = CleanData._cond_any(c, zero_target, sentinels, tol)
            runlen = CleanData._runlen_expr(cond_any)
            short_mask = (runlen > 0) & (runlen <= pl.lit(lv1_threshold_steps_for_impute))
            masked_exprs.append(pl.when(short_mask).then(None).otherwise(pl.col(c)).alias(c))

        masked_df = df.with_columns(masked_exprs) 

        # === One-line fail-fast after filtering as well (paranoia check) ===
        if masked_df.select(pl.any_horizontal(pl.all().is_null().sum() == masked_df.height)).item():
            raise ValueError("All-null column detected after row-drop.")

        # rows with ANY null (i.e., will be imputed somewhere)
        rows_masked_any = int(
            masked_df.select(pl.any_horizontal(pl.all().is_null())).sum().item()
        )  # count of rows where any column is null

        # per-column null counts (how many cells will be imputed per column)
        per_col_null_counts = (
            masked_df.select([pl.col(c).is_null().sum().alias(c) for c in masked_df.columns])
                    .row(0)  # one-row frame → tuple of counts
        )
        # map to {col: count}
        rows_masked_per_col = {c: int(per_col_null_counts[i]) for i, c in enumerate(masked_df.columns)}
        
        if impute_method == "ffill_bfill":
            filled = masked_df.with_columns([
                pl.all().fill_null(strategy="forward").fill_null(strategy="backward")
            ])
            if verbose:
                print(f"🩹 Imputed short runs (≤ {lv1_threshold_steps_for_impute}) on: {list(filled.columns)}", flush=True)
         
        elif impute_method == "linear":
            # CHG: linear interpolation between non-null points; edges may remain null → pad with ffill/bfill
            # === Linear ===
            filled = masked_df.with_columns([
                pl.col(c)
                .interpolate(method="linear")        # fill interior gaps
                .fill_null(strategy="forward")       # pad tail
                .fill_null(strategy="backward")      # pad head
                .alias(c)
                for c in masked_df.columns
            ])
            if verbose:
                print(f"🩹 Imputed short runs (≤ {lv1_threshold_steps_for_impute}) via linear interpolation (+edge pad) on: {list(filled.columns)}", flush=True)  

        elif impute_method == "nearest":
            # CHG: nearest-value interpolation; ties resolved by nearest neighbor; pad edges to remove residual nulls
            # === Nearest ===
            filled = masked_df.with_columns([
                pl.col(c)
                .interpolate(method="nearest")       # nearest-value fill for interior
                .fill_null(strategy="forward")
                .fill_null(strategy="backward")
                .alias(c)
                for c in masked_df.columns
            ])
            if verbose:
                print(f"🩹 Imputed short runs (≤ {lv1_threshold_steps_for_impute}) via nearest interpolation (+edge pad) on: {list(filled.columns)}", flush=True)
        else:
            raise ValueError(f"Unknown impute method: {impute_method}")
        # REQUIRE: 'filled' is the DataFrame after ffill/bfill or interpolation
        assert int(filled.select(pl.any_horizontal(pl.all().is_null())).sum().item()) == 0, \
            "Imputation left nulls behind unexpectedly."
        return filled, rows_masked_any, rows_masked_per_col
 
 
    @classmethod
    def from_raw_data_and_known_fix(cls, raw: RawData, use_known_fix: bool = True) -> "CleanData":
        if isinstance(raw.dataset_id, SynDataID):
            if not raw.backend == "numpy" or raw.data_np is None: 
                raise ValueError("Synthetic Data must be in numpy backend; Synthetic Data must have data_np set.")
            return cls(
                    dataset_id=raw.dataset_id,
                    backend="numpy",
                    data_np=raw.data_np.astype(np.float32, copy=False),  # [N, D] ,
                    data_pl=None,
                    feature_names=raw.feature_names, known_fix=None
                ) 
        if not use_known_fix:
            return cls(
                dataset_id=raw.dataset_id,
                backend="polars",
                data_np=None,
                data_pl=raw.data_pl,
                feature_names=raw.feature_names, known_fix=None
            ) 
        assert isinstance(raw.dataset_id, KnownDataID), "Known fix only applies to known datasets."
        registry_meta: KnownDatasetMeta = KNOWN_DATASETS[raw.dataset_id]  # Crashes if not found (loud)
        known_fix = KnownFixConfig.construct_known_fix_with_default(
            dataset_id=raw.dataset_id,
            meta=registry_meta,
        )
        # --- Apply known fix steps --- 
        meta = known_fix.meta      
        sentinels = meta.sentinels 
        tol = known_fix.value_tol

        if raw.data_pl is None or not raw.backend == "polars":
            raise ValueError("Known fix only applies to polars backend; RawData must have data_pl set.")
        df = raw.data_pl  # CHG: source table to mutate via policy
        zero_target = 0.0  # CHG: used below in analyze/drop rows
        verbose = True     # CHG: matches prior printing behavior; set False if you want silence

        report: Dict[str, Any] = {"per_column": {}, "dropped_cols": [], "rows_dropped": 0, "imputed_cols": []}

        # 1) analyze
        metrics = CleanData._analyze_columns(df, sentinels=sentinels, zero_target=zero_target, tol=tol)
        report["per_column"] = metrics

        # 2) drop columns
        df, dropped_cols = CleanData._drop_bad_columns(
            df, metrics,
            lv1_threshold_sentinel_density_for_drop_col=meta.lv1_threshold_sentinel_density_for_drop_col,
            lv2_threshold_zero_density_for_drop_col=meta.lv2_threshold_zero_density_for_drop_col,
            lv3_threshold_steps_for_drop_col=meta.lv3_threshold_steps_for_drop_col,
            verbose=verbose,
        )
 
        if sentinels: # keep only active sentinel keys
            sentinels = {c: sentinels[c] for c in sentinels if c in df.columns}

        # 3) drop rows
        df, rows_dropped = CleanData._drop_bad_rows(
            df,
            sentinels=sentinels,
            zero_target=zero_target,
            tol=tol,
            lv2_threshold_steps_for_drop_row=meta.lv2_threshold_steps_for_drop_row,
            verbose=verbose,
        )
        # 4) impute short runs
        df2, rows_masked_any, rows_masked_per_col = CleanData._impute_short_runs(
            df,
            sentinels=sentinels,
            zero_target=zero_target,
            tol=tol,
            lv1_threshold_steps_for_impute=meta.lv1_threshold_steps_for_impute,
            impute_method=meta.impute_method,
            verbose=verbose,
        )
        clean_report = CleanReport(
            dropped_cols=dropped_cols, 
            rows_dropped=rows_dropped, 
            imputed_cols=list(df2.columns) if (meta.impute_method != "none" and meta.lv1_threshold_steps_for_impute > 0) else [],
            rows_masked_any=rows_masked_any,
            rows_masked_per_col=rows_masked_per_col,
        )
        return cls(
            dataset_id=raw.dataset_id,
            backend="polars",
            data_np=None,
            data_pl=df2,
            feature_names=list(df2.columns),
            known_fix=known_fix,
            clean_report=clean_report,
        )

def stage_load_raw_data(bundle: "DataBundle") -> "DataBundle":
    print("\n--- Stage: Load Raw Data ---") 
    raw_data = RawData.from_config(bundle.source_config) 
    bundle.raw_data = raw_data   
    return bundle

def stage_known_problem_fixed(bundle: "DataBundle") -> "DataBundle":
    print("\n--- Stage: Known Problem Fixes ---")
    assert bundle.raw_data is not None, "Raw data must be loaded before known fixes."
    clean_data= CleanData.from_raw_data_and_known_fix(
        raw=bundle.raw_data, use_known_fix=True,
    )
    bundle.clean_data = clean_data
    print(f"-> Completed known problem fixes. Report:\n{clean_data.clean_report}\n")
    return bundle

class SplitConfig(BaseModel):
    """intent: fractional split policy + optional blank gaps to reduce leakage.
    assumptions: fractions in [0,1], ordered train < val < test; gaps are counts.
    returns: concrete index ranges via .resolve(n).
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    # === Fractions (policy only) ===
    train_start_fraction: float = Field(0.0, ge=0.0, le=1.0)
    train_end_fraction:   float = Field(0.7, ge=0.0, le=1.0)
    val_start_fraction:   float = Field(0.7, ge=0.0, le=1.0)
    val_end_fraction:     float = Field(0.8, ge=0.0, le=1.0)
    test_start_fraction:  float = Field(0.8, ge=0.0, le=1.0)
    test_end_fraction:    float = Field(1.0, ge=0.0, le=1.0)

    # === Gaps (blank samples inserted between splits) ===
    val_gap:  int = Field(0, ge=0, description="Blank samples between train and val.")
    test_gap: int = Field(0, ge=0, description="Blank samples between val and test.")

    @classmethod
    def preset(cls, name: str) -> "SplitConfig":
        """intent: named ratios for quick experiments; returns cfg with defaults."""
        # REQUIRE: known preset
        if name == "702010":
            return cls(train_start_fraction=0.0, train_end_fraction=0.7, val_start_fraction=0.7, val_end_fraction=0.9,
                       test_start_fraction=0.9, test_end_fraction=1.0, val_gap=0, test_gap=0)
        if name == "601525":
            return cls(train_start_fraction=0.0, train_end_fraction=0.6, val_start_fraction=0.6, val_end_fraction=0.75,
                       test_start_fraction=0.75, test_end_fraction=1.0, val_gap=0, test_gap=0)
        raise ValueError(f"unknown preset={name}")

    @model_validator(mode="after")
    def _check_splits(self) -> "SplitConfig":
        """intent: enforce basic ordering & bounds (fractions only)."""
        t0, t1 = self.train_start_fraction, self.train_end_fraction
        v0, v1 = self.val_start_fraction,   self.val_end_fraction
        s0, s1 = self.test_start_fraction,  self.test_end_fraction

        # REQUIRE: 0 ≤ start < end ≤ 1 for each; train ≤ val ≤ test in order
        ok_bounds = (0.0 <= t0 < t1 <= 1.0) and (0.0 <= v0 < v1 <= 1.0) and (0.0 <= s0 < s1 <= 1.0)  # CHG: fix boolean logic
        ok_order  = (t1 <= v0) and (v1 <= s0)  # CHG: enforce non-overlap in fraction space
        if not (ok_bounds and ok_order):
            raise ValueError("Invalid split fractions: must satisfy 0≤start<end≤1 and train < val < test.")
        return self

    # === Resolver (single place where n is used) ===
    def compute_split_indices(self, n: int) -> dict[str, tuple[int, int]]:
        """intent: map fractional policy (+gaps) to concrete half-open index ranges.
        assumptions: n is length AFTER cleaning; gaps are applied between splits.
        returns: dict with (start, end) for 'train','val','test'.
        """
        # --- base indices from fractions ---
        t0 = int(n * self.train_start_fraction) ; t1 = int(n * self.train_end_fraction)
        v0 = int(n * self.val_start_fraction) ; v1 = int(n * self.val_end_fraction)
        s0 = int(n * self.test_start_fraction) ; s1 = int(n * self.test_end_fraction)

        # REQUIRE: basic monotone bounds (fractions validator already enforces order)
        # GUARD-OK: explicit bound check here to surface rounding issues early
        if not (0 <= t0 <= t1 <= n and 0 <= v0 <= v1 <= n and 0 <= s0 <= s1 <= n):
            # CHG: fail fast on any rounding-induced out-of-bounds
            raise ValueError(f"Index bounds invalid for n={n}: "
                            f"train=({t0},{t1}), val=({v0},{v1}), test=({s0},{s1})")
        
        # === Strict gap constraints (no auto “max/min” fixes) === 
        if (v0 - t1) < self.val_gap: 
            raise ValueError(f"val_gap={self.val_gap} not feasible: train_end={t1}, val_start={v0}, need ≥ {t1 + self.val_gap}.")
        
        if (s0 - v1) < self.test_gap: 
            raise ValueError(f"test_gap={self.test_gap} not feasible: val_end={v1}, test_start={s0}, need ≥ {v1 + self.test_gap}.")

        # === Non-empty slices (no silent empties) ===
        if (t1 - t0) <= 0: 
            raise ValueError(f"Empty train split: ({t0},{t1})")
        if (v1 - v0) <= 0:
            raise ValueError(f"Empty val split: ({v0},{v1})")
        if (s1 - s0) <= 0:
            raise ValueError(f"Empty test split: ({s0},{s1})")
        return {
                "train": (t0, t1),
                "val":   (v0, v1),
                "test":  (s0, s1),
            }
 

class SplitData(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    dataset_id: Union[SynDataID, KnownDataID] = Field(..., description="Source dataset identifier.") 
    feature_names: list[str]
    scaler_type: Literal["standard", "robust_iqr","robust_mad",
        "center_mean","center_median","none"] = "none"


    train_data: np.ndarray = Field(..., description="[N_train, D] training data.")
    val_data: np.ndarray = Field(..., description="[N_val, D] validation data.")
    test_data: np.ndarray = Field(..., description="[N_test, D] test data.") 

    data_stats: Dict[str, np.ndarray] = Field(
        ..., description="Per-split data statistics: quantiles, mean, std, quantiles, MAD."
    )

    @staticmethod
    def split_stats_fast(
        X: np.ndarray,
        train: Tuple[int, int],
        val: Tuple[int, int],
        test: Tuple[int, int],
        *,
        ddof: int = 0,
        dtype = None,
    ) -> Dict[str, np.ndarray]:
        """
        intent: compute per-split stats in one go: mean/std via matrix ops; quantiles/MAD via slice quantiles.
        assumptions: X is 2D [N, D], cleaned (no NaN if using np.quantile), disjoint half-open splits.
        returns: dict of arrays shaped as noted; columns are [train, val, test].

        Symbols:
        N := number of rows; D := number of features.
        """
        # === Shapes & slicing ===
        # REQUIRE: X.ndim == 2
        N, D = X.shape
        t0, t1 = train; v0, v1 = val; s0, s1 = test
        # GUARD-OK: avoid UB/NaN from empty splits or bad bounds
        if not (0 <= t0 <= t1 <= N and 0 <= v0 <= v1 <= N and 0 <= s0 <= s1 <= N):
            raise ValueError(f"bad split bounds for N={N}: train={train}, val={val}, test={test}")

        Xv = np.asarray(X, dtype=dtype, order="C")  # CHG: fix comment spelling: contiguous; optional dtype narrowing

        # === Fast path: mean/std via masked matrix ops ===
        train_mask = np.zeros(N, dtype=np.float32); train_mask[t0:t1] = 1.0
        val_mask   = np.zeros(N, dtype=np.float32); val_mask[v0:v1]   = 1.0
        test_mask  = np.zeros(N, dtype=np.float32); test_mask[s0:s1]  = 1.0
        M = np.column_stack([train_mask, val_mask, test_mask])                     # [N, 3]

        sums   = Xv.T @ M                                                          # [D, 3]
        counts = M.sum(axis=0).astype(np.float32)                                  # [3]
        if np.any(counts == 0):
            # GUARD-OK: prevent divide-by-zero; strict failure on empty split
            raise ValueError(f"empty split detected; counts={counts.tolist()}")

        means  = sums / counts                                                     # [D, 3]
        sums2  = (Xv * Xv).T @ M                                                   # [D, 3]
        vars_  = (sums2 / counts) - means**2                                       # [D, 3]
        if ddof:
            vars_ *= counts / np.maximum(counts - ddof, 1.0)                       # CHG: explicit sample-var correction
        stds   = np.sqrt(np.maximum(vars_, 0.0))                                   # [D, 3]

        # === Quantiles + MAD (per-split selection; unavoidable order stats) ===
        X_train, X_val, X_test = Xv[t0:t1], Xv[v0:v1], Xv[s0:s1]                   # views

        def _q_and_mad(A: np.ndarray):
            # REQUIRE: A.ndim == 2 [n_i, D]
            q01, q05, q10, q25, q50, q75, q90, q95, q99 = np.quantile(
                A, [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99],
                axis=0,
                method="linear",                                                   # CHG: reproducible interpolation
            )
            mad = np.median(np.abs(A - q50), axis=0)                               # [D]
            return q01, q05, q10, q25, q50, q75, q90, q95, q99, mad

        tq01, tq05, tq10, tq25, tq50, tq75, tq90, tq95, tq99, tmad = _q_and_mad(X_train)
        vq01, vq05, vq10, vq25, vq50, vq75, vq90, vq95, vq99, vmad = _q_and_mad(X_val)
        sq01, sq05, sq10, sq25, sq50, sq75, sq90, sq95, sq99, smad = _q_and_mad(X_test)

        # === Per-split min/max (reuse views) ===
        _max = np.column_stack([X_train.max(axis=0), X_val.max(axis=0), X_test.max(axis=0)])  # CHG: reuse views
        _min = np.column_stack([X_train.min(axis=0), X_val.min(axis=0), X_test.min(axis=0)])  # CHG: reuse views

        # === Pack results (columns = [train, val, test]) ===
        return {
            "mean":   means,                                                       # [D, 3]
            "std":    stds,                                                        # [D, 3]
            "q01":    np.column_stack([tq01, vq01, sq01]),                         # [D, 3]
            "q05":    np.column_stack([tq05, vq05, sq05]),                         # [D, 3]
            "q10":    np.column_stack([tq10, vq10, sq10]),                         # [D, 3]
            "q25":    np.column_stack([tq25, vq25, sq25]),                         # [D, 3]
            "median": np.column_stack([tq50, vq50, sq50]),                         # [D, 3]
            "q75":    np.column_stack([tq75, vq75, sq75]),                         # [D, 3]
            "q90":    np.column_stack([tq90, vq90, sq90]),                         # [D, 3]
            "q95":    np.column_stack([tq95, vq95, sq95]),                         # [D, 3]
            "q99":    np.column_stack([tq99, vq99, sq99]),                         # [D, 3]
            "mad":    np.column_stack([tmad, vmad, smad]),                         # [D, 3]
            "max":    _max,                                                        # [D, 3]
            "min":    _min,                                                        # [D, 3]
            "count":  counts.astype(np.int64),                                     # [3]
        }
 
    def __str__(self) -> str:
        """
        intent: human-readable, comprehensive split report using precomputed `self.stats`.
        assumptions: self.train_data/val_data/test_data are [N, D] torch.Tensors; self.stats keys exist and match shapes.
        returns: multi-line string; no mutation, no device transfers beyond `.cpu()` for printing.
        """
        # === Display options ===
        torch.set_printoptions(precision=3, sci_mode=False)
        np.set_printoptions(precision=3, suppress=True)

        # === Aliases (no device moves except for tiny samples) ===
        stats = self.data_stats  # REQUIRE: dict with keys: mean,std,median,mad,q01,q05,q25,q75,q90,q95,q99,min,max,count
        train, val, test = self.train_data, self.val_data, self.test_data
        D = train.shape[1]
        counts = stats["count"]  # [3]

        # === Basic checks (fail loud) ===
        # GUARD-OK: rank check prevents nonsense formatting
        if any(x.ndim != 2 for x in (train, val, test)):
            raise ValueError(f"[SplitData.__str__] tensors must be [N,D]; "
                            f"got train={tuple(train.shape)}, val={tuple(val.shape)}, test={tuple(test.shape)}")

        # === Section 1: Split overview ===
        lines = ["=" * 80, "📊 DATA SPLITS SUMMARY", "=" * 80]
        lines.append("\n┌─ Split Shapes & Counts")
        lines.append(f"│  Train: {train.shape} ({int(counts[0]):,} samples)")
        lines.append(f"│  Val:   {val.shape} ({int(counts[1]):,} samples)")
        lines.append(f"│  Test:  {test.shape} ({int(counts[2]):,} samples)")
        total = int(counts.sum())
        # GUARD-OK: protect division in edge cases; you already forbid empty splits upstream
        ratio_train = counts[0] / total if total else 0.0
        ratio_val   = counts[1] / total if total else 0.0
        ratio_test  = counts[2] / total if total else 0.0
        lines.append(f"│  Ratios: {ratio_train:.1%} / {ratio_val:.1%} / {ratio_test:.1%}")
        lines.append(f"└─ Device: {train.device}")

        # === Section 2: Train-based normalization params ===
        lines.append("\n┌─ Normalization Parameters (from train split)")
        train_mean   = stats["mean"][:, 0]
        train_std    = stats["std"][:, 0]
        mask = train_std > 1e-6
        std_min = train_std[mask].min() if mask.any() else 0.0  # CHG: precompute min with mask

        train_median = stats["median"][:, 0]
        train_mad    = stats["mad"][:, 0]

        if D <= 10:
            lines.append("│  Channel │    Mean │     Std │  Median │     MAD")
            lines.append("│  " + "─" * 55)
            for i in range(D):
                lines.append(
                    f"│     {i:2d}   │ {train_mean[i]:7.3f} │ {train_std[i]:7.3f} │ "
                    f"{train_median[i]:7.3f} │ {train_mad[i]:7.3f}"
                )
        else:
            lines.append(f"│  {D} channels total")
            lines.append(f"│  Mean   range: [{train_mean.min():.3f}, {train_mean.max():.3f}]")
            lines.append(f"│  Std    range: [{std_min:.3f}, {train_std.max():.3f}]")
            lines.append(f"│  Median range: [{train_median.min():.3f}, {train_median.max():.3f}]")
        lines.append("└─")

        # === Section 3: Distribution health (train) ===
        lines.append("\n┌─ Distribution Health (train split)")
        q01 = stats["q01"][:, 0]; q05 = stats["q05"][:, 0]; q25 = stats["q25"][:, 0]
        q75 = stats["q75"][:, 0]; q95 = stats["q95"][:, 0]; q99 = stats["q99"][:, 0]
        iqr = q75 - q25
        # tail wedges vs IQR
        right_tail_ratio = (q99 - q95) / np.maximum(iqr, 1e-8)
        left_tail_ratio  = (q05 - q01) / np.maximum(iqr, 1e-8)
        heavy_tails_mask = (right_tail_ratio > 1.5) | (left_tail_ratio > 1.5)

        # near-constant channels (train std)
        near_constant_mask = (train_std < 1e-6)

        # sparsity by zeros (train)
        zero_fraction = (train == 0).astype(np.float32).mean(axis=0)
        sparse_mask = zero_fraction > 0.3

        # scale ratio
        finite_stds = train_std[train_std > 1e-6] 
        scale_ratio = float(finite_stds.max() / finite_stds.min()) if finite_stds.size > 0 else 1.0


        lines.append(f"│  Heavy-tailed channels:  {int(heavy_tails_mask.sum())}/{D}")
        if heavy_tails_mask.any():
            idx = np.where(heavy_tails_mask)[0]
            lines.append(f"│    → Channels: {idx[:10].tolist()}" + ("..." if len(idx) > 10 else ""))
        lines.append(f"│  Near-constant channels: {int(near_constant_mask.sum())}/{D}")
        if near_constant_mask.any():
            idx = np.where(near_constant_mask)[0]
            lines.append(f"│    → Channels: {idx[:10].tolist()}" + ("..." if len(idx) > 10 else ""))
        lines.append(f"│  Sparse channels (>30% zeros): {int(sparse_mask.sum())}/{D}")
        if sparse_mask.any():
            idx = np.where(sparse_mask)[0]
            lines.append(f"│    → Channels: {idx[:10].tolist()}" + ("..." if len(idx) > 10 else ""))
        lines.append(f"│  Scale variation: {scale_ratio:.1f}x (max std / min std)")
        if scale_ratio > 100:
            lines.append("│    ⚠️  Large scale differences detected — consider robust scaling or center-only.")
        lines.append("└─")

        # === Section 4: Cross-split comparison (vs train) ===
        lines.append("\n┌─ Cross-Split Distribution Comparison")
        mean_shift_abs = np.abs(stats["mean"][:, 1:] - stats["mean"][:, 0:1])  # [D,2]
        mean_shift_val = float(mean_shift_abs[:, 0].mean())
        mean_shift_test = float(mean_shift_abs[:, 1].mean())
        lines.append(f"│  Mean shift (avg |Δ| vs train): Val={mean_shift_val:.3f} | Test={mean_shift_test:.3f}")

        std_ratio_val  = np.median((stats["std"][:, 1] / np.maximum(stats["std"][:, 0], 1e-8)))
        std_ratio_test = np.median((stats["std"][:, 2] / np.maximum(stats["std"][:, 0], 1e-8)))
        lines.append(f"│  Std ratio (median vs train):   Val={std_ratio_val:.2f}× | Test={std_ratio_test:.2f}×")

        if abs(std_ratio_val - 1.0) > 0.3 or abs(std_ratio_test - 1.0) > 0.3:
            lines.append("│    ⚠️  Variance differs significantly across splits")
        lines.append("└─")

        # === Section 5: Per-channel details (small D only) ===
        if D <= 5:
            lines.append("\n┌─ Per-Channel Detailed Statistics")
            for ch in range(D):
                lines.append(f"│  Channel {ch}:")
                lines.append(f"│    Train: μ={stats['mean'][ch,0]:.3f}, σ={stats['std'][ch,0]:.3f}, "
                            f"range=[{stats['min'][ch,0]:.3f}, {stats['max'][ch,0]:.3f}]")
                lines.append(f"│           q01={stats['q01'][ch,0]:.3f}, q50={stats['median'][ch,0]:.3f}, q99={stats['q99'][ch,0]:.3f}")
                lines.append(f"│    Val:   μ={stats['mean'][ch,1]:.3f}, σ={stats['std'][ch,1]:.3f}, "
                            f"range=[{stats['min'][ch,1]:.3f}, {stats['max'][ch,1]:.3f}]")
                lines.append(f"│    Test:  μ={stats['mean'][ch,2]:.3f}, σ={stats['std'][ch,2]:.3f}, "
                            f"range=[{stats['min'][ch,2]:.3f}, {stats['max'][ch,2]:.3f}]")
                if ch < D - 1:
                    lines.append("│    " + "─" * 60)
            lines.append("└─")

        # === Section 6: Sample vectors ===
        lines.append("\n┌─ Sample Data (first sample, first 5 dims)")
        dims_to_show = min(5, D)
        lines.append(f"│  Train: {train[0, :dims_to_show]}" + ("..." if D > 5 else ""))
        lines.append(f"│  Val:   {val[0, :dims_to_show]}"   + ("..." if D > 5 else ""))
        lines.append(f"│  Test:  {test[0, :dims_to_show]}"  + ("..." if D > 5 else ""))
        lines.append("└─")

        # === Footer ===
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    @property
    def channels(self) -> int:
        return len(self.feature_names)

    @staticmethod
    def normalize(data: np.ndarray, center: np.ndarray, scale: np.ndarray, *, eps: float = 1e-8) -> np.ndarray: 
        return (data - center) / (np.maximum(scale, eps))
    
    @staticmethod
    def center_and_scale(
        data_stats: Dict[str, np.ndarray], 
        normalize: Literal["standard", "robust_iqr","robust_mad","center_mean",
            "center_median","none"]='none') -> Tuple[np.ndarray, np.ndarray]:

        if normalize == "none":
            # CHG: needs D; infer from one stat we always have
            D = int(data_stats["mean"].shape[0])
            return np.zeros(D, dtype=np.float32), np.ones(D, dtype=np.float32)

        if normalize == "standard":
            center = data_stats["mean"][:, 0]
            scale  = data_stats["std"][:, 0]
        elif normalize == "robust_iqr":
            center = data_stats["median"][:, 0]
            scale  = (data_stats["q75"][:, 0] - data_stats["q25"][:, 0])
        elif normalize == "robust_mad":
            center = data_stats["median"][:, 0]
            scale  = data_stats["mad"][:, 0] * 1.4826  # approx std
        elif normalize == "center_mean":
            center = data_stats["mean"][:, 0]
            scale  = np.ones_like(center, dtype=np.float32)
        elif normalize == "center_median":
            center = data_stats["median"][:, 0]
            scale  = np.ones_like(center, dtype=np.float32)
        else:
            raise ValueError(f"Unknown normalization method: {normalize}")

        return center.astype(np.float32, copy=False), scale.astype(np.float32, copy=False)
     

    @classmethod
    def from_clean_data(
        cls,
        clean: CleanData,
        split_preset_name: str = "702010",
        normalize: Literal["standard", "robust_iqr","robust_mad","center_mean","center_median","none"] = "none",
    ) -> "SplitData":
        """intent: build SplitData from CleanData (numpy or polars), compute stats once, then optional normalize."""
        # === Resolve splits ===

        split_cfg = SplitConfig.preset(name=split_preset_name)

        if clean.backend == "numpy":
            assert clean.data_np is not None, "[SplitData.from_clean_data] numpy backend but data_np is None"
            data = np.ascontiguousarray(clean.data_np, dtype=np.float32)  # ✅ Ensure contiguous + dtype
        else:
            assert clean.data_pl is not None, "[SplitData.from_clean_data] polars backend but data_pl is None"
            data = np.ascontiguousarray(clean.data_pl.to_numpy(), dtype=np.float32)  # ✅ Same

        feature_names = clean.feature_names
        length = data.shape[0]
        # CHG: align with resolver that returns {"train":(s,e),...}
        indices = split_cfg.compute_split_indices(n=length)  # CHG: use one API consistently

        t0,t1 = indices["train"]; v0,v1 = indices["val"]; s0,s1 = indices["test"]
        train_data = data[t0:t1, :]
        val_data   = data[v0:v1, :]
        test_data  = data[s0:s1, :]

        # === Compute stats once on RAW cleaned data ===
        data_stats = SplitData.split_stats_fast(
            X=data, train=indices["train"], val=indices["val"], test=indices["test"], dtype=np.float32
        )

        # === Optional normalization (fit on train stats only) ===
        center, scale = SplitData.center_and_scale(data_stats=data_stats, normalize=normalize)
        if normalize != "none":
            train_data = SplitData.normalize(train_data, center=center, scale=scale)
            val_data   = SplitData.normalize(val_data,   center=center, scale=scale)
            test_data  = SplitData.normalize(test_data,  center=center, scale=scale)

        # === Construct ===
        return cls(
            dataset_id=clean.dataset_id,
            feature_names=feature_names,
            scaler_type=normalize,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            data_stats=data_stats,  # CHG: field name aligned
        )
 
# final_bundle = (
#         initial_bundle
#         | stage_load_raw_data
#         | stage_known_problem_fixed          # <— NEW
#         | stage_split_and_scale_data
#         | stage_move_to_window_view
#         | stage_create_dataloaders
#         | stage_compute_koopa_mask         # <— NEW 
#     )

def stage_split_and_scale_data(bundle: "DataBundle") -> "DataBundle":
    """Splits data into train/val/test sets without scaling."""
    print("\n--- Stage: Splitting Data ---")
    assert bundle.clean_data is not None, "[stage_split_and_scale_data] clean_data is None"
    split_data = SplitData.from_clean_data(
        clean=bundle.clean_data,
        split_preset_name='702010'
    )
    bundle.split_data = split_data
    print(f"-> Completed data splitting. Split summary:\n{split_data}\n")
    print(bundle.split_data.data_stats)
 
    return bundle
   


@runtime_checkable
class TimeSeriesDatasetProtocol(Protocol):
    """Protocol defining the interface for time series datasets."""
    seq_len: int
    label_len: int
    pred_len: int
    
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class WindowedData(BaseModel):
    """
    intent: container for windowed datasets (TDE) before loaders.
    assumptions: datasets yield (x_win [S,D], y_win [L+H,D]).
    returns: self-printable summary; pure data, no loader state.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    train_dataset: TimeSeriesDatasetProtocol
    val_dataset: TimeSeriesDatasetProtocol
    test_dataset: TimeSeriesDatasetProtocol
    mode: Literal["lazy","strided"]="strided"

    @classmethod
    def from_splits(cls, splits, *, seq_len: int, label_len: int, pred_len: int, mode: Literal["lazy","strided"]="lazy"):
        """intent: uniform constructor from numpy/torch splits.
        assumptions: splits.* are [N,D]; no device moves here.
        returns: WindowedData
        """  
        train_tensor = torch.as_tensor(splits.train_data, dtype=torch.float32)
        val_tensor = torch.as_tensor(splits.val_data, dtype=torch.float32)
        test_tensor = torch.as_tensor(splits.test_data, dtype=torch.float32)
        DS = TimeSeriesDatasetStrided if mode == "strided" else TimeSeriesDataset
        return cls(
            train_dataset=DS(data_x=train_tensor, data_y=train_tensor, seq_len=seq_len, label_len=label_len, pred_len=pred_len),
            val_dataset=DS(data_x=val_tensor, data_y=val_tensor, seq_len=seq_len, label_len=label_len, pred_len=pred_len),
            test_dataset=DS(data_x=test_tensor, data_y=test_tensor, seq_len=seq_len, label_len=label_len, pred_len=pred_len),
        )
    
    def __str__(self) -> str:
        """intent: show first-item shapes without side effects."""
        x_tr, y_tr = self.train_dataset[0]
        x_va, y_va = self.val_dataset[0]
        x_te, y_te = self.test_dataset[0]
        return "\n".join([
            "📊 TDE Data shapes",
            f"   Train: x{tuple(x_tr.shape)}, y{tuple(y_tr.shape)}",
            f"   Val  : x{tuple(x_va.shape)}, y{tuple(y_va.shape)}",
            f"   Test : x{tuple(x_te.shape)}, y{tuple(y_te.shape)}",
        ])

def stage_move_to_window_view(bundle: "DataBundle") -> "DataBundle": 
    """Applies time-delay embedding to create TimeSeriesDatasets.""" 
    print("\n--- Stage: Moving to Window View ---") 
    cfg = bundle.training_config  
    bundle.windowed_data = WindowedData.from_splits(
        splits=bundle.split_data,
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,    
        pred_len=cfg.pred_len,
        mode="strided"  # CHG: fix to strided for efficiency
    ) 
    print(bundle.windowed_data) 
    return bundle
 
class LoaderData(BaseModel):
    """
    intent: thin wrapper around three LoaderData.
    assumptions: datasets are ready; collate fn resolves CI vs non-CI.
    returns: printable preview of batch shapes.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    @classmethod
    def from_windowed(cls, wd: WindowedData, 
        *, 
        batch_size: int, 
        pred_len: int, 
        channel_independent: bool, 
        num_workers: int = 0):
        # === choose collate once ===
        collate = functools.partial(collate_tde_ci, pred_len=pred_len) if channel_independent \
                  else functools.partial(collate_tde,    pred_len=pred_len)

        mk = lambda ds, shuffle: DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            pin_memory=True, 
            # persistent_workers=True,
            num_workers=num_workers, #prefetch_factor=4,
            drop_last=False, collate_fn=collate,
        )
        return cls(
            train_loader=mk(wd.train_dataset, True),
            val_loader=mk(wd.val_dataset, False),
            test_loader=mk(wd.test_dataset, False),
        )

    def __str__(self) -> str:
        """intent: show batch shapes; fail loud if empty."""
        torch.set_printoptions(precision=3, sci_mode=False); np.set_printoptions(precision=3, suppress=True)
        parts = [f"Number of batches in train_loader: {len(self.train_loader)}"]
        try:
            xb, yb = next(iter(self.train_loader))[:2]  # REQUIRE: dataset returns (x,y)
            parts.append(f"First batch: X{tuple(xb.shape)} | Y{tuple(yb.shape)}")
            parts.append(f"Sample X[0, :1]: {xb[0, :1].detach().cpu().numpy() if isinstance(xb, torch.Tensor) else xb[0, :1]}")
            parts.append(f"Sample Y[0, :1]: {yb[0, :1].detach().cpu().numpy() if isinstance(yb, torch.Tensor) else yb[0, :1]}")
        except StopIteration:
            parts.append("Train loader is empty.")
        return "\n".join(parts)


def stage_create_dataloaders(bundle: "DataBundle") -> "DataBundle":
    """intent: wrap datasets in loaders; runtime stage."""
    print("\n--- Stage: Creating LoaderData ---")
    cfg = bundle.training_config
    assert bundle.windowed_data is not None, "[stage_create_dataloaders] windowed_data is None"
    bundle.loader_data = LoaderData.from_windowed(
        bundle.windowed_data,
        batch_size=cfg.batch_size,
        pred_len=cfg.pred_len,
        channel_independent=cfg.channel_independent,
        num_workers=getattr(cfg, "num_workers", 0),
    )
    print(bundle.loader_data)
    return bundle

def stage_compute_koopa_mask(bundle: "DataBundle") -> "DataBundle":
    """
    intent: if model_type=='Koopa' and mask_spectrum unset, estimate once from TRAIN loader.
    assumptions: loader yields xs shaped [B,S,D] or [B·D,S,1] (CI); seq_len = S = L.
    returns: bundle with cfg.mask_spectrum = sorted list of rFFT bin indices.
    """     
    cfg = bundle.training_config
    if getattr(cfg, "model_type", None) != "Koopa":
        return bundle
    # only auto-compute when not preset
    if getattr(cfg, "mask_spectrum", None) not in (None, [], "auto"):
        return bundle
 
    alpha = getattr(cfg, "alpha", 0.20)
    L = int(cfg.seq_len)

    print(f"[Koopa] Computing mask_spectrum from train loader (alpha={alpha}, L={L})...")
    ms = _compute_mask_spectrum_from_loader(bundle.loader_data.train_loader, L=L, alpha=alpha)
    cfg.mask_spectrum = ms
    print(f"[Koopa] mask_spectrum bins: {ms[:10]}{'...' if len(ms)>10 else ''} (len={len(ms)})")
    return bundle

def _compute_mask_spectrum_from_loader(
    loader: torch.utils.data.DataLoader,
    L: int,
    alpha: float = 0.2,
    window: str | None = "hann",
    device: str | torch.device | None = None,
) -> list[int]:
    """
    intent/contract:
        One-pass mean magnitude spectrum over training batches.
        Shapes: x:[B,S,D] or [B·D,S,1] with S==L; rFFT(dim=1) ⇒ F=L//2+1 bins.
    returns:
        Sorted unique top-α bin indices (DC=0 always included).
    """ 
    # === Accumulator ===
    F = L // 2 + 1
    amps_sum = torch.zeros(F, dtype=torch.float32, device=device)  # zero-init
    batches = 0

    # === Optional window (leakage reduction) ===
    w = None
    if window is not None:
        if window == "hann":
            w = torch.hann_window(L, periodic=True, device=device)
        else:
            raise ValueError(f"Unknown window={window!r}")  # GUARD-OK: unsupported window

    # === Accumulate ===
    for xs, _ys in loader:
        x = xs.to(dtype=torch.float32, device=device) if device is not None else xs.to(torch.float32)
        # REQUIRE: time dim is -2 and equals L
        if x.shape[-2] != L:  # GUARD-OK: wrong seq_len silently corrupts F
            raise ValueError(f"L={L} expected at dim(-2); got shape {tuple(x.shape)}")

        if w is not None:
            x = x * w.view(1, L, 1)  # [1,L,1] → [B,S,1] or [B·D,S,1]

        xf = torch.fft.rfft(x, dim=1)   # [B,F,D] or [B,F,1]
        mag = torch.abs(xf)             # [B,F,D] or [B,F,1]
        m = mag.mean(dim=(0, 2))        # [F]; mean over batch and channel   # CHG: drop ndim branch

        amps_sum += m
        batches += 1

    # === Average & top-α bins ===
    denom = max(1, batches)             # zero-safe per your preference
    amps = amps_sum / denom             # [F]

    k = max(1, int(round(F * float(alpha))))
    topk = torch.topk(amps, k).indices.tolist()
    if 0 not in topk:
        topk.append(0)                  # include DC
    return sorted(set(i for i in topk if 0 <= i < F))

class DataBundle(BaseModel):
    """
    A single, stateful container that flows through the data processing pipeline.
    It holds all inputs, configurations, and intermediate/final results.
    """ 
    def __or__(self, func: Callable[["DataBundle"], "DataBundle"]): 
        return func(self) # Enables the elegant `|` pipeline syntax.

    model_config = ConfigDict( extra='forbid', arbitrary_types_allowed=True, validate_assignment=False, ) 
    source_config: SourceConfig   # SourceConfig   
    training_config: configs.BaseTrainingConfig | None  # e.g., BaseTrainingConfig holding seq_len, etc.
    no_scale: bool = True  # Whether to skip scaling during data prep.

    # --- Fields to be populated by the pipeline --- 
    raw_data: Optional[RawData] = None
    clean_data: Optional[CleanData] = None 
    split_data: Optional[SplitData] = None 
    windowed_data: Optional[WindowedData] = None

    loader_data: Optional[LoaderData] = None  # Will hold LoadersContainer

    run_started_at: pendulum.DateTime = Field(  # CHG: anchor timestamp once; avoids drifting names
        default_factory=lambda: pendulum.now()
    )
   
    # --- Properties --- 
    @field_validator("experiment_name")
    def name_must_not_be_whitespace(cls, v: str) -> str:
        """Ensure that string fields are not just whitespace."""
        if not v.strip():
            raise ValueError("Field must not be empty or contain only whitespace.")
        return v

    @property
    def dataset_id(self) -> str: 
        return self.source_config.dataset_id

    @property
    def train_std(self) -> np.ndarray:
        assert self.split_data is not None, "[DataBundle.train_std] split_data is None"
        return self.split_data.data_stats["std"][:, 0]  # train split

    @property
    def device(self) -> str:
        return self.training_config.device
    
    @computed_field  # Pydantic v2: read-only, recomputed on access
    @property
    def experiment_name(self) -> str:  
        ts = self.run_started_at.format("YYYYMMDD_HHmm")
        cfg = self.training_config
        return f"{cfg.model_type}_{self.dataset_id}_seq{cfg.seq_len}_pred{cfg.pred_len}_{ts}"
    
    @property
    def feature_names(self) -> list[str]:
        assert self.clean_data is not None, "[DataBundle.feature_names] clean_data is None"
        return self.clean_data.feature_names
        

def create_data_pipeline(
    source_config: SourceConfig, 
    model_config: configs.BaseTrainingConfig,
    no_scale: bool = True,  
) -> DataBundle:
    """
    Initializes and runs the entire data processing pipeline.

    Returns:
        A fully populated DataBundle containing all data artifacts.
    """ 
    # 1. Create the initial bundle with all necessary configurations.
    initial_bundle = DataBundle(
        source_config=source_config,
        training_config=model_config,
        no_scale=no_scale, 
    ) 
    # 2. Execute the full pipeline in a clear, readable sequence.
    final_bundle = (
        initial_bundle
        | stage_load_raw_data
        | stage_known_problem_fixed          # <— NEW
        | stage_split_and_scale_data
        | stage_move_to_window_view
        | stage_create_dataloaders
        | stage_compute_koopa_mask         # <— NEW 
    ) 
    print("\n✅ Data pipeline finished successfully.")
    return final_bundle

def sample_features(cols: list[str], k_seed: tuple[int, int]) -> list[str]:
    """
    intent/contract:
        Randomly sample k distinct feature names from cols without replacement.
        assumptions: cols is a list of strings [D]; k_seed=(k, seed) with 1 <= k <= D.
        returns: list[str] of length k.
    """
    # === Unpack ===
    k, seed = k_seed  # CHG: clearer tuple name + unpack for readability

    # === Deterministic draw ===
    np.random.seed(seed)  # CHG: stable randomness via explicit seed
    picked = np.random.choice(cols, size=k, replace=False)  # [k] names

    # === Log & return ===
    print(f"Original Num Columns: {len(cols)}, Selected Num Columns: {len(picked)}")
    return picked.tolist()  # CHG: ensure List[str] (not ndarray)

import pandas as pd

def analyze_zero_patterns(filepath: str):
    """
    Analyzes the ETTh2.csv file to determine the nature of zero values
    in each feature column.
    
    It classifies zeros as 'Isolated' (likely missing data) or
    'Clustered' (likely true intermittent data).
    """
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please ensure 'ETTh2.csv' is in the same directory.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # The 7 feature columns
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    # Drop rows where ALL features are 0, as these might be padding/empty
    data = data.dropna(how='all', subset=features)
    
    print(f"--- Analyzing Zero Patterns in {filepath} ---")
    print(f"Total records loaded: {len(data)}\n")
    
    report = []

    for col in features:
        if col not in data.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
            
        col_series = data[col]
        
        # 1. Total zero count
        total_zeros = (col_series == 0).sum()
        
        if total_zeros == 0:
            report.append({
                "Column": col,
                "Total Zeros": 0,
                "% Zeros": "0.0%",
                "Isolated Zeros": 0,
                "Clustered Zeros": 0,
                "Classification": "Continuous (No Zeros)"
            })
            continue

        # 2. Identify isolated zeros
        # An isolated zero has non-zero neighbors (prev and next)
        is_zero = (col_series == 0)
        prev_is_nonzero = (col_series.shift(1) != 0)
        next_is_nonzero = (col_series.shift(-1) != 0)
        
        # Count isolated zeros
        isolated_zeros = (is_zero & prev_is_nonzero & next_is_nonzero).sum()
        
        # 3. Clustered zeros are the remainder
        clustered_zeros = total_zeros - isolated_zeros
        
        percent_zeros = (total_zeros / len(data)) * 100
        
        # 4. Classify the column
        classification = ""
        if isolated_zeros > clustered_zeros:
            # More isolated implies missing data
            classification = "Type 2 (Missing Data)"
        elif clustered_zeros >= isolated_zeros and total_zeros > 0:
            # More clustered implies true intermittency
            classification = "Type 1 (Intermittent)"
        
        report.append({
            "Column": col,
            "Total Zeros": total_zeros,
            "% Zeros": f"{percent_zeros:.2f}%",
            "Isolated Zeros": isolated_zeros,
            "Clustered Zeros": clustered_zeros,
            "Classification": classification
        })

    # Print the final report
    print_report(report)

def print_report(report: list):
    """Helper function to format and print the analysis report."""
    
    # Find max col width for alignment
    max_col = max(len(r['Column']) for r in report)
    max_class = max(len(r['Classification']) for r in report)
    
    header = (
        f"{'Column':<{max_col}} | {'Total Zeros':>11} | {'% Zeros':>7} | "
        f"{'Isolated Zeros':>15} | {'Clustered Zeros':>16} | "
        f"{'Classification':<{max_class}}"
    )
    print(header)
    print("-" * len(header))
    
    for r in report:
        print(
            f"{r['Column']:<{max_col}} | {r['Total Zeros']:>11} | {r['% Zeros']:>7} | "
            f"{r['Isolated Zeros']:>15} | {r['Clustered Zeros']:>16} | "
            f"{r['Classification']:<{max_class}}"
        )
    
    print("\n--- Recommendations ---")
    print("Type 1 (Intermittent): Data is truly zero. Use your 'Gate Model' idea for these features.")
    print("Type 2 (Missing Data):  Data is null/missing. **Linearly interpolate** these features before training.")
    print("Continuous:             No special handling needed.")
# analyze_zero_patterns('./datasets/ETTh2.csv')