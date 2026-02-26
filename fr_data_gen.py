import numpy as np 
from typing import Union, Tuple, Dict, Optional, Any, List, Callable, Literal, Sequence 
from pydantic import BaseModel, Field, model_validator, field_validator, PositiveInt, ConfigDict
from numpy.typing import DTypeLike


import numpy as np
from pydantic import BaseModel  # for parity with your setup
 
class ODEParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, 
                              validate_assignment=False)
    initial_cond: np.ndarray                # will be coerced by validator
    steps: PositiveInt = Field(..., description="Must be at least one step")  
    dt: float = Field(..., description="Simulation time increment. Must be > 0.")
    method: Literal['euler', 'rk4'] = 'rk4'
    time_dependent: bool = False
    t0: float = 0.0 
    dtype: DTypeLike = np.float64
    
    def calc_diff(self, state: np.ndarray, t: float | None = None) -> np.ndarray:
        raise NotImplementedError
    
    # --- NEW: single shock control --- 
    """
    ODE system relies on state x and param p.  
    state_eps: at shock_step s, nudge the state x by shock_eps
    param: at shock_step s, x stays, but mutate p.
    switch: run two full sims (A and B) and splice: prefix from A up to s, 
        suffix from B from s onward, i.e. change x and change p to completely different values.
    """

    shock_frac: Optional[float] = None   # e.g., 0.72 means at step round(0.72*steps)
    shock_step: Optional[int] = None     # overrides shock_frac if set
    shock_kind: Optional[Literal['state_eps','param','switch']] = None
    shock_eps: float = 1e-2              # ~1e-3 is tiny for Lorenz/Rössler/Chua
    switch_update: Optional[Dict] = None  # <--- new: kwargs for the B version

    @field_validator('initial_cond', mode='before')
    @classmethod
    def force_convert_to_dtype(cls, v):
        # CHG: access dtype via model_fields to avoid AttributeError on cls.dtype (pydantic v2)
        field = getattr(cls, "model_fields", {}).get("dtype")
        target_dtype = getattr(field, "default", np.float64)
        arr = np.asarray(v, dtype=target_dtype).reshape(-1)
        if arr.size == 0:
            raise ValueError("initial_cond cannot be empty")
        return arr
    # def force_convert_to_dtype(cls, v):
    #     arr = np.asarray(v, dtype=cls.dtype).reshape(-1)
    #     if arr.size == 0:
    #         raise ValueError("initial_cond cannot be empty")
    #     return arr
    
    # hook: default no-op
    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        return x


def run_ode(params: ODEParams) -> np.ndarray:
    """
    Integrate x' = f(x)              if params.time_dependent == False
           or x' = f(x, t)           if params.time_dependent == True
    using 'euler' or 'rk4' for `steps` steps of size `dt`.

    Required on `params`:
      - initial_cond: array-like or scalar
      - steps: int
      - dt: float
      - method: 'euler' or 'rk4'
      - calc_diff: callable; signature f(x) or f(x, t)
    Optional:
      - time_dependent: bool (default False)
      - t0: float (default 0.0)
    """
    # --- init ---
    x = params.initial_cond.copy()   # already float64, 1-D
    dt, steps, t = params.dt, params.steps, params.t0
    f = params.calc_diff
    traj = np.empty((steps + 1, x.size), dtype=params.dtype)
    traj[0] = x
    
    # compute the (single) shock step index (1..steps)
    s = params.shock_step
    if s is None and params.shock_frac is not None:
        s = int(round(params.shock_frac * steps))
        s = max(1, min(steps, s))
    shocked = False

    # --- Euler ---
    if params.method == "euler":
        for i in range(1, steps + 1):
            if s is not None and i == s and not shocked:
                x = params.on_shock(x, t); shocked = True
                
            k1 = f(x, t) if params.time_dependent else f(x) 
            x = x + k1 * dt
            t = t + dt  # harmless for autonomous; useful for logs

            if not np.isfinite(x).all():
                raise ValueError(f"Numerical instability at step {i}: state={x}, t={t}")
            traj[i] = x

    # --- RK4 ---
    elif params.method == "rk4": 
        for i in range(1, steps + 1):
            if s is not None and i == s and not shocked:
                x = params.on_shock(x, t); shocked = True
             
            if params.time_dependent:
                k1 = f(x,             t)
                k2 = f(x + 0.5*dt*k1, t + 0.5*dt)
                k3 = f(x + 0.5*dt*k2, t + 0.5*dt)
                k4 = f(x + dt*k3,     t + dt)
            else:
                k1 = f(x)
                k2 = f(x + 0.5*dt*k1)
                k3 = f(x + 0.5*dt*k2)
                k4 = f(x + dt*k3)
                    
            x = x + (k1 + 2*k2 + 2*k3 + k4) * (dt / 6.0)
            t = t + dt
            if not np.isfinite(x).all():
                raise ValueError(f"Numerical instability at step {i}: state={x}, t={t}")
            traj[i] = x
    else:
        raise ValueError(f"Unknown method: {params.method!r}. Expected 'euler' or 'rk4'.")

    traj = np.asarray(traj, dtype=np.float32) # always convert to float32
    return traj 


# --- helper: two-run splice for 'switch' ---
def run_ode_switch(m: ODEParams) -> np.ndarray:
    if m.switch_update is None:
        raise ValueError("switch_update must be set for shock_kind='switch'.")

    s = m.shock_step
    if s is None and m.shock_frac is not None:
        s = int(round(m.shock_frac * m.steps))
    if s is None:
        raise ValueError("Provide shock_step or shock_frac for 'switch'.")
    s = max(1, min(m.steps, s))  # clamp

    # A: base; B: updated params; disable shocks inside each run
    A = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None})
    B = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None, **m.switch_update})
    
    A = type(m).model_validate(A.model_dump())
    B = type(m).model_validate(B.model_dump())
    
    trajA = run_ode(A)
    trajB = run_ode(B)

    combo = trajB.copy()
    combo[:s+1] = trajA[:s+1]  # include state at step s
    return combo


class OUParams(ODEParams):
    """
    OU SDE (Itô):  dx_t = θ (μ - x_t) dt + σ dW_t
    We integrate with Euler–Maruyama:
      x_{t+Δ} = x_t + θ(μ - x_t)Δ + σ sqrt(Δ) ξ_t,  ξ_t ~ N(0, I)
    Notes:
      - We ignore 'rk4' (not meaningful for SDEs). Require method='euler'.
      - Shocks:
          'state_eps' : x <- x + shock_eps at step s
          'param'     : θ/μ/σ <- *_after at step s (in-place) and continue
          'switch'    : two full sims (A base, B with updates) then splice
    """
    # OU parameters
    theta: float = Field(0.2, gt=0.0)
    mu:    float = 0.0
    sigma: float = Field(0.3, gt=0.0)
    seed: int = Field(1955, description="Random seed") 

    # Optional post-shock replacements for 'param' or 'switch'
    theta_after: Optional[float] = None
    mu_after:    Optional[float] = None
    sigma_after: Optional[float] = None
    
 

    # OU is 1-D by default here; enforce length-1 state to keep it simple & clear
    initial_cond: np.ndarray = Field(
        default_factory=lambda: np.array([0.0], dtype=np.float64)
    )

    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_1d_len1(cls, arr: np.ndarray):
        if arr.size != 1:
            raise ValueError("OU requires 1D initial_cond of length 1.") 
        return arr

    def _param_flip(self):
        if self.theta_after is not None: self.theta = self.theta_after
        if self.mu_after    is not None: self.mu    = self.mu_after
        if self.sigma_after is not None: self.sigma = self.sigma_after

    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.shock_kind == 'state_eps':
            x = x.copy(); x[0] += self.shock_eps
        elif self.shock_kind == 'param':
            self._param_flip()
        return x

    def generate(self) -> np.ndarray:
        # Guard: E-M only
        if self.method != 'euler':
            raise ValueError("OU uses Euler–Maruyama; set method='euler'.")
        if self.shock_kind == 'switch':
            return run_ou_switch(self)
        return run_ou_em(self)


def run_ou_em(params: OUParams) -> np.ndarray:
    """
    Euler–Maruyama for OU:
      x_{t+Δ} = x_t + θ(μ - x_t)Δ + σ sqrt(Δ) ξ,  ξ ~ N(0,1)
    Shock semantics: same as run_ode (single shock at step s).
    Returns: (steps+1, 1) float64 trajectory.
    """
    x = params.initial_cond.copy()              # shape (1,)
    dt, steps = params.dt, params.steps
    traj = np.empty((steps + 1, 1), dtype=params.dtype)
    traj[0, 0] = x[0]

    # determine shock index (1..steps), like run_ode
    s = params.shock_step
    if s is None and params.shock_frac is not None:
        s = int(round(params.shock_frac * steps))
        s = max(1, min(steps, s))
    shocked = False

    rng = np.random.default_rng(params.seed)

    for i in range(1, steps + 1):
        if s is not None and i == s and not shocked:
            x = params.on_shock(x, t=(i-1)*dt); shocked = True  # apply state nudge or param flip

        # drift + diffusion
        drift = params.theta * (params.mu - x[0]) * dt
        diffusion = params.sigma * np.sqrt(dt) * rng.standard_normal()

        x[0] = x[0] + drift + diffusion
        traj[i, 0] = x[0]

        if not np.isfinite(x[0]):
            raise ValueError(f"Numerical instability at step {i}: state={x[0]}")

    return traj


def run_ou_switch(m: OUParams) -> np.ndarray:
    """
    'switch' mode: run two independent OU sims and splice:
      A (base params) provides prefix up to step s; B (updated params) provides suffix.
    """
    if m.switch_update is None:
        raise ValueError("switch_update must be set for shock_kind='switch'.")

    # compute splice index (1..steps)
    s = m.shock_step
    if s is None and m.shock_frac is not None:
        s = int(round(m.shock_frac * m.steps))
    if s is None:
        raise ValueError("Provide shock_step or shock_frac for 'switch'.")
    s = max(1, min(m.steps, s))

    # A: base; B: updated; disable inner shocks & re-validate to coerce types
    A = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None})
    B = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None, **m.switch_update})
    A = type(m).model_validate(A.model_dump())
    B = type(m).model_validate(B.model_dump())

    # Both must use EM
    if A.method != 'euler' or B.method != 'euler':
        raise ValueError("OU switch requires method='euler' for both runs.")

    trajA = run_ou_em(A)
    trajB = run_ou_em(B)

    combo = trajB.copy()
    combo[:s+1] = trajA[:s+1]  # include state at s
    return combo

class LorenzParams(ODEParams): 
    sigma: float = Field(10.0, gt=0, description="Prandtl number, related to fluid viscosity (must be > 0)")
    rho: float = Field(28.0, gt=0, description="Rayleigh number, related to temperature gradient (must be > 0)")
    beta: float = Field(8.0 / 3, gt=0, description="Geometric constant (must be > 0)")
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([1.0, 0.98, 1.1], dtype=np.float64))
    
    # optional “after” values for param override / regime switch
    sigma_after: Optional[float] = None
    rho_after:   Optional[float] = None
    beta_after:  Optional[float] = None
    
    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_3d(cls, arr: np.ndarray):
        if arr.size != 3:
            raise ValueError("Lorenz requires 3D initial_cond: [x, y, z].")
        return arr

    def calc_diff(self, state: np.ndarray) -> np.ndarray: 
        x, y, z = state 
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z
        ], dtype=np.float64)
    
    def generate(self) -> np.ndarray: 
        print(f"-> Generating Lorenz trajectory with params: rho={self.rho}")  
        if self.shock_kind == 'switch':
            return run_ode_switch(self)
        #TODO run_ode_switch(self)[3000:,...] is temporary to drop not-on-attractor data
        return run_ode(self)
    
    # --- NEW: what to do at the shock ---
    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        kind = self.shock_kind
        if kind == 'state_eps':
            x = x.copy(); x[0] += self.shock_eps  # nudge the x-coordinate
            return x
        if kind in ('param',):
            if self.sigma_after is not None: self.sigma = self.sigma_after
            if self.rho_after   is not None: self.rho   = self.rho_after
            if self.beta_after  is not None: self.beta  = self.beta_after
            return x
        return x

class RosslerParams(ODEParams): 
    a: float = Field(0.2, ge=0.0, description="Linear feedback in y; typically ≥ 0")
    b: float = Field(0.2, ge=0.0, description="Offset in z; typically ≥ 0")
    c: float = Field(5.7, gt=0.0, description="Nonlinear chaos parameter; typically > 0")
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float64))
    
    a_after: Optional[float] = None
    b_after: Optional[float] = None
    c_after: Optional[float] = None

    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_3d(cls, arr: np.ndarray):
        if arr.size != 3:
            raise ValueError("Rössler needs 3D initial_cond.")
        return arr

    def calc_diff(self, state: np.ndarray) -> np.ndarray: 
        x, y, z = state  
        return np.array([
            -y - z,
            x + self.a * y,
            self.b + z * (x - self.c)
        ], dtype=np.float64) 
        
    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.shock_kind == 'state_eps':
            x = x.copy(); x[0] += self.shock_eps
        elif self.shock_kind in ('param'):
            if self.a_after is not None: self.a = self.a_after
            if self.b_after is not None: self.b = self.b_after
            if self.c_after is not None: self.c = self.c_after
        return x

    def generate(self) -> np.ndarray:  
        if self.shock_kind == 'switch':
            return run_ode_switch(self)
        return run_ode(self) 

class HyperRosslerParams(ODEParams):
    a: float = Field(0.25, ge=0.0, description="Linear feedback in y; typically ≥ 0")
    b: float = Field(3.0, ge=0.0, description="Offset in z; typically ≥ 0")
    c: float = Field(0.5, gt=0.0, description="Coupling strength from z to w; must be > 0")
    d: float = Field(0.05, gt=0.0, description="Growth term for w; must be > 0")
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([1.0, 1.0, 4.0, 1.0], dtype=np.float64))
    
    a_after: Optional[float] = None
    b_after: Optional[float] = None
    c_after: Optional[float] = None
    d_after: Optional[float] = None

    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_4d(cls, arr: np.ndarray):
        if arr.size != 4:
            raise ValueError("Hyper-Rössler requires 4D initial_cond: [x, y, z, w].")
        return arr

    def calc_diff(self, state: np.ndarray) -> np.ndarray:
        x, y, z, w = state 
        dx = -y - z
        dy = x + self.a * y + w
        dz = self.b + x * z
        dw = -self.c * z + self.d * w
        return np.array([dx, dy, dz, dw], dtype=np.float64)
    
    
    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.shock_kind == 'state_eps':
            x = x.copy(); x[0] += self.shock_eps
        elif self.shock_kind in ('param'):
            if self.a_after is not None: self.a = self.a_after
            if self.b_after is not None: self.b = self.b_after
            if self.c_after is not None: self.c = self.c_after
            if self.d_after is not None: self.d = self.d_after
        return x
    
    
    def generate(self) -> np.ndarray:
        if self.shock_kind == 'switch':
            return run_ode_switch(self)
        return run_ode(self) 
"""
# A) tiny state nudge at 72% of the way (x → x + 1e-3 on coord 0)
p = LorenzParams(
    initial_cond=[1.0,0.98,1.1], dt=0.01, steps=10000, method='rk4',
    sigma=10.0, rho=28.0, beta=8/3,
    shock_frac=0.72, shock_kind='state_eps', shock_eps=1e-3
)
traj = p.generate()

# B) parameter bump at 35% (only beta changes)
p = LorenzParams(
    initial_cond=[1,1,1], dt=0.01, steps=10000, method='rk4',
    shock_frac=0.35, shock_kind='param', beta_after=3.0
)
traj = p.generate()

# C) splice at 72%: A = beta=10, B = beta=11 (and maybe different initial_cond)
p = LorenzParams(
    initial_cond=[1.0,0.98,1.1], dt=0.01, steps=20000, method='rk4',
    sigma=10.0, rho=28.0, beta=10.0,
    shock_frac=0.72, shock_kind='switch',
    switch_update={'beta': 11.0, 'initial_cond': [1.5, -0.2, 2.0]}  # optional IC change
)
traj = p.generate()
"""
class DuffingParams(ODEParams):
    alpha: float = Field(1.0, ge=0.0, description="Linear stiffness; typically ≥ 0")
    beta: float = Field(-1.0, description="Nonlinear stiffness; can be < 0 for chaotic behavior")
    delta: float = Field(0.2, gt=0.0, description="Damping coefficient; must be > 0")
    gamma: float = Field(0.3, gt=0.0, description="Amplitude of forcing term; must be > 0")
    omega: float = Field(1.0, gt=0.0, description="Frequency of forcing term; must be > 0")
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([0.1, 0.1], dtype=np.float64))
    time_dependent: bool = True

    # enforce 2D state [x, v]
    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_2d(cls, arr: np.ndarray):
        if arr.size != 2:
            raise ValueError("Duffing requires 2D initial_cond: [x, v].")
        return arr

    def calc_diff(self, state: np.ndarray, t: float) -> np.ndarray:
        x, v = state 
        dx = v
        dv = self.gamma * np.cos(self.omega * t) - self.delta * v - self.alpha * x - self.beta * x**3
        return np.array([dx, dv], dtype=np.float64)
    
    def generate(self) -> np.ndarray: 
        if self.shock_kind == 'switch':
            return run_ode_switch(self)
        return run_ode(self) 

 
class Lorenz96Params(ODEParams):
    dim: int = Field(20, gt=3, description="Number of variables in the system; must be greater than 3")
    forcing: float = Field(8.0, description="Forcing term F; chaos typically occurs when F ≥ 8")
    initial_cond: Optional[np.ndarray] = Field(None, description="Initial condition of length dim")

  
    @model_validator(mode='after')
    def _fill_or_check_ic(self):
        if self.initial_cond is None:
            # arr = np.full((self.dim,), self.forcing, dtype=np.float64)
            arr = np.ones((self.dim,), dtype=np.float64)
            arr[0] += 0.01  # small perturbation
            object.__setattr__(self, 'initial_cond', arr)
        else:
            arr = np.asarray(self.initial_cond, dtype=np.float64).reshape(-1)
            if arr.size != self.dim:
                raise ValueError(f"initial_cond must have length dim={self.dim}")
            object.__setattr__(self, 'initial_cond', arr)
        return self

    def calc_diff(self, state: np.ndarray) -> np.ndarray:
        x = state
        x_roll_p1 = np.roll(x, shift=-1, axis=0)  # x_{i+1}
        x_roll_m1 = np.roll(x, shift=1, axis=0)   # x_{i-1}
        x_roll_m2 = np.roll(x, shift=2, axis=0)   # x_{i-2} 
        return (x_roll_p1 - x_roll_m2) * x_roll_m1 - x + self.forcing

    def generate(self) -> np.ndarray:
        if self.shock_kind == 'switch':
            return run_ode_switch(self)
        return run_ode(self) 
 
def chua_nonlinearity(x: np.ndarray, m0: float, m1: float) -> np.ndarray:
    """
    Dimensionless Chua diode (3-segment PWL):
      h(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)
    Equivalent to the classic piecewise form with slopes m0 (outer) and m1 (inner).
    """
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1.0) - np.abs(x - 1.0))


class ChuaParams(ODEParams):
    # Canonical double-scroll region (dimensionless)
    alpha: float = Field(15.6, gt=0.0, description="α > 0")
    beta:  float = Field(28.0, gt=0.0, description="β > 0 (chaos around ~25–51 for classic m0,m1)")
    m0:    float = Field(-8.0/7.0, description="Outer-slope of PWL nonlinearity")
    m1:    float = Field(-5.0/7.0, description="Inner-slope of PWL nonlinearity") 
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([0.1, 0.0, 0.0], dtype=np.float64))
    
    alpha_after: Optional[float] = None
    beta_after:  Optional[float] = None
    m0_after:    Optional[float] = None
    m1_after:    Optional[float] = None
 
    @model_validator(mode='after')
    def _check(self):
        if self.initial_cond is not None and len(self.initial_cond) != 3:
            raise ValueError("initial_cond must be length 3 for Chua (x,y,z).")
        if self.dt <= 0.004:
            raise ValueError("dt must be greater than 0.004 for Chua (x,y,z).")
        return self 
    
    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_3d(cls, arr: np.ndarray):
        if arr.size != 3:
            raise ValueError("Chua requires 3D initial_cond: [x, y, z].")
        return arr

    def calc_diff(self, state: np.ndarray) -> np.ndarray:
        """
        Dimensionless Chua ODE:
          dx/dt = α * (y - x - h(x))
          dy/dt = x - y + z
          dz/dt = -β * y
        """
        x, y, z = state
        hx = chua_nonlinearity(x, self.m0, self.m1)
        dx = self.alpha * (y - x - hx)
        dy = x - y + z
        dz = -self.beta * y
        return np.array([dx, dy, dz], dtype=np.float64)
    
    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.shock_kind == 'state_eps':
            x = x.copy(); x[0] += self.shock_eps
        elif self.shock_kind in ('param'):
            if self.alpha_after is not None: self.alpha = self.alpha_after
            if self.beta_after  is not None: self.beta  = self.beta_after
            if self.m0_after    is not None: self.m0    = self.m0_after
            if self.m1_after    is not None: self.m1    = self.m1_after
        return x
    
    def generate(self, burn_in: int = 1000) -> np.ndarray:
        # run longer and drop the front
        total_steps = self.steps + burn_in
        tmp = self.model_copy(update={"steps": total_steps})
        if self.shock_kind == 'switch':
            return run_ode_switch(tmp)[burn_in:]
        return run_ode(tmp)[burn_in:]
    
    
class HenonParams(ODEParams):
    a: float = Field(1.4, description="Nonlinear coefficient; classic chaos at a = 1.4")
    b: float = Field(0.3, description="Linear coefficient; classic chaos at b = 0.3")
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float64), 
                                     description="Initial (x, y) values for the Hénon map")
    steps: PositiveInt = Field(..., description="Number of iterations")  # NEW

      
    def generate(self) -> np.ndarray:
        x, y = map(float, self.initial_cond)
        trajectory = np.empty((self.steps + 1, 2), dtype=np.float64)
        trajectory[0] = [x, y]
          
        for i in range(1, self.steps + 1):
            x_new = 1.0 - self.a * x**2 + y
            y_new = self.b * x 
            x, y = x_new, y_new  
            trajectory[i] = [x, y]

        return trajectory
  
class LogisticParams(ODEParams):
    r: float = Field(3.9, gt=0, description="Growth rate; chaos typically occurs when r ∈ [3.57, 4.0]")
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([0.5], dtype=np.float64), 
                                     description="Initial value x₀ ∈ [0, 1]") 
    steps: PositiveInt = Field(..., description="Number of iterations")  # NEW

    def generate(self) -> np.ndarray:
        x = float(self.initial_cond)
        trajectory = np.empty(self.steps + 1, dtype=np.float64)
        trajectory[0] = x
        for i in range(1, self.steps + 1):
            x = self.r * x * (1.0 - x) # Ensure float arithmetic
            trajectory[i] = x
        return trajectory.reshape(-1, 1)
    
 # --- Linear Swtiching 
class SLDSParams(ODEParams):
    """
    K=2 Switching LDS (1D):
      Regime z_t ∈ {1,2} Markov with P(1→1)=p11, P(2→2)=p22.
      x_{t} = A_{z_t} x_{t-1} + η_t,   η_t ~ N(0, Q_{z_t})
    Notes:
      - Discrete-time; we ignore `dt` and `method` (kept for API uniformity).
      - Shocks:
          'state_eps' : x <- x + shock_eps at step s
          'param'     : flip to *_after at step s and continue
          'switch'    : two independent runs (A base, B updated) then splice
    """
    # LDS params per regime
    A1: float = 0.95
    Q1: float = Field(0.8, ge=0.0)
    A2: float = 0.98
    Q2: float = Field(0.2, ge=0.0)

    # Markov self-transition probabilities
    p11: float = Field(0.97, ge=0.0, le=1.0)  # P(z=1→1)
    p22: float = Field(0.98, ge=0.0, le=1.0)  # P(z=2→2)

    # RNG seed
    seed: Optional[int] = None

    # Allow post-shock replacements for 'param' / 'switch'
    A1_after: Optional[float] = None
    Q1_after: Optional[float] = None
    A2_after: Optional[float] = None
    Q2_after: Optional[float] = None
    p11_after: Optional[float] = None
    p22_after: Optional[float] = None

    # 1D state for clarity
    initial_cond: np.ndarray = Field(
        default_factory=lambda: np.array([0.0], dtype=np.float64)
    )

    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_len1(cls, arr: np.ndarray):
        if arr.size != 1:
            raise ValueError("SLDS requires 1D initial_cond of length 1.")
        return arr

    def _param_flip(self):
        for name in ('A1','Q1','A2','Q2','p11','p22'):
            v = getattr(self, f"{name}_after")
            if v is not None:
                setattr(self, name, v)

    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.shock_kind == 'state_eps':
            x = x.copy(); x[0] += self.shock_eps
        elif self.shock_kind == 'param':
            self._param_flip()
        return x

    def generate(self) -> np.ndarray:
        if self.shock_kind == 'switch':
            return run_slds_switch(self)
        return run_slds(self)


def run_slds(params: SLDSParams) -> np.ndarray:
    """
    Single-run SLDS with optional single shock at step s.
    Returns: (steps+1, 1) float64 trajectory.
    """
    rng = np.random.default_rng(params.seed)
    steps = params.steps
    x = params.initial_cond.copy()   # shape (1,)
    traj = np.empty((steps + 1, 1), dtype=np.float64)
    traj[0, 0] = x[0]

    # start regime by stationary prior (optional): here pick 1 with prob 0.5
    z = 1 if rng.random() < 0.5 else 2

    # shock index (1..steps)
    s = params.shock_step
    if s is None and params.shock_frac is not None:
        s = int(round(params.shock_frac * steps))
        s = max(1, min(steps, s))
    shocked = False

    for t in range(1, steps + 1):
        # apply shock once
        if s is not None and t == s and not shocked:
            x = params.on_shock(x, t-1); shocked = True

        # transition regime
        if z == 1:
            z = 1 if rng.random() < params.p11 else 2
            A, Q = params.A1, params.Q1
        else:
            z = 2 if rng.random() < params.p22 else 1
            A, Q = params.A2, params.Q2

        # evolve x_t = A x_{t-1} + η, η~N(0,Q)
        x[0] = A * x[0] + rng.normal(0.0, np.sqrt(Q))
        traj[t, 0] = x[0]

        if not np.isfinite(x[0]):
            raise ValueError(f"Numerical instability at step {t}: state={x[0]}")

    return traj


def run_slds_switch(m: SLDSParams) -> np.ndarray:
    """
    'switch' = splice: prefix from base A, suffix from updated B.
    """
    if m.switch_update is None:
        raise ValueError("switch_update must be set for shock_kind='switch'.")
    s = m.shock_step
    if s is None and m.shock_frac is not None:
        s = int(round(m.shock_frac * m.steps))
    if s is None:
        raise ValueError("Provide shock_step or shock_frac for 'switch'.")
    s = max(1, min(m.steps, s))

    A = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None})
    B = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None, **m.switch_update})
    A = type(m).model_validate(A.model_dump())
    B = type(m).model_validate(B.model_dump())

    trajA = run_slds(A)
    trajB = run_slds(B)

    combo = trajB.copy()
    combo[:s+1] = trajA[:s+1]
    return combo



# Double well Langvevin

class DoubleWellParams(ODEParams):
    """
    Langevin SDE in double-well potential:
      U(x) = x^4/4 - (a/2) x^2   ⇒   dx_t = (-x^3 + a x) dt + σ dW_t
    Shocks:
      - 'state_eps': x <- x + eps at step s
      - 'param'    : a/σ <- *_after at step s (in-place) and continue
      - 'switch'   : two sims (A,B) and splice at s
    """
    a: float = Field(1.5, gt=0.0)
    sigma: float = Field(0.25, gt=0.0)
    a_after: Optional[float] = None
    sigma_after: Optional[float] = None
    seed: Optional[int] = None

    # 1D state (len=1) for clarity
    initial_cond: np.ndarray = Field(
        default_factory=lambda: np.array([0.1], dtype=np.float64)
    )
    method: Literal['euler', 'rk4'] = 'euler'  # force EM

    @field_validator('initial_cond', mode='after')
    @classmethod
    def _must_be_len1(cls, arr: np.ndarray):
        if arr.size != 1:
            raise ValueError("DoubleWell requires 1D initial_cond of length 1.")
        return arr

    def _param_flip(self):
        if self.a_after is not None: self.a = self.a_after
        if self.sigma_after is not None: self.sigma = self.sigma_after

    def on_shock(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.shock_kind == 'state_eps':
            x = x.copy(); x[0] += self.shock_eps
        elif self.shock_kind == 'param':
            self._param_flip()
        return x

    def generate(self) -> np.ndarray:
        if self.method != 'euler':
            raise ValueError("Double-well uses Euler–Maruyama; set method='euler'.")
        if self.shock_kind == 'switch':
            return run_doublewell_switch(self)
        return run_doublewell_em(self)

def run_doublewell_em(p: DoubleWellParams) -> np.ndarray:
    """Euler–Maruyama: x_{t+Δ} = x + (-x^3 + a x)Δ + σ√Δ ξ."""
    x = p.initial_cond.copy()   # shape (1,)
    dt, steps = p.dt, p.steps
    rng = np.random.default_rng(p.seed)

    s = p.shock_step
    if s is None and p.shock_frac is not None:
        s = int(round(p.shock_frac * steps)); s = max(1, min(steps, s))
    shocked = False

    traj = np.empty((steps + 1, 1), dtype=np.float64)
    traj[0, 0] = x[0]

    for i in range(1, steps + 1):
        if s is not None and i == s and not shocked:
            x = p.on_shock(x, t=(i-1)*dt); shocked = True

        drift = (-x[0]**3 + p.a * x[0]) * dt
        diffusion = p.sigma * np.sqrt(dt) * rng.standard_normal()
        x[0] = x[0] + drift + diffusion

        if not np.isfinite(x[0]):
            raise ValueError(f"Numerical instability at step {i}: state={x[0]}")
        traj[i, 0] = x[0]
    return traj

def run_doublewell_switch(m: DoubleWellParams) -> np.ndarray:
    """‘switch’: run A (base) and B (updated) then splice prefix from A up to s."""
    if m.switch_update is None:
        raise ValueError("switch_update must be set for shock_kind='switch'.")
    s = m.shock_step
    if s is None and m.shock_frac is not None:
        s = int(round(m.shock_frac * m.steps))
    if s is None:
        raise ValueError("Provide shock_step or shock_frac for 'switch'.")
    s = max(1, min(m.steps, s))

    A = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None})
    B = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None, **m.switch_update})
    A = type(m).model_validate(A.model_dump())
    B = type(m).model_validate(B.model_dump())

    trajA = run_doublewell_em(A)
    trajB = run_doublewell_em(B)

    combo = trajB.copy()
    combo[:s+1] = trajA[:s+1]
    return combo



class SeasonalARParams(BaseModel):
    """
    Seasonal AR with time-varying amplitude
    y_t = a(t) * sin(2π t / S) + φ y_{t-1} + ε_t,  ε_t ~ N(0, σ^2)
    a(t) = a0 + amp_drift_per_step * t   (simple linear drift)

    a_t is time-varying amplitude (e.g., linear trend or shock-flipped)
    """  
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # core
    steps: PositiveInt
    S: int = Field(24, ge=1)
    phi: float = Field(0.5, ge=-0.999, le=0.999)
    sigma: float = Field(0.2, gt=0.0)
    a0: float = 1.0
    amp_drift_per_step: float = 0.0
    initial_cond: np.ndarray = Field(default_factory=lambda: np.array([0.0], dtype=np.float64))
    seed: int = Field(1955, description="Random seed")

    # shocks
    shock_frac: Optional[float] = None
    shock_step: Optional[int] = None
    shock_kind: Optional[Literal['state_eps','param','switch']] = None
    shock_eps: float = 1e-2
    switch_update: Optional[Dict] = None

    # post-shock parameter replacements (for 'param'/'switch')
    a0_after: Optional[float] = None
    phi_after: Optional[float] = None
    sigma_after: Optional[float] = None
    S_after: Optional[int] = None
    amp_drift_after: Optional[float] = None

    @field_validator('initial_cond', mode='before')
    @classmethod
    def _as_f64_len1(cls, v):
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size != 1:
            raise ValueError("SeasonalAR requires 1D initial_cond of length 1.")
        return arr

    def _shock_index(self) -> Optional[int]:
        s = self.shock_step
        if s is None and self.shock_frac is not None:
            s = int(round(self.shock_frac * self.steps))
        return None if s is None else max(1, min(int(self.steps), int(s)))

    def _param_flip(self):
        if self.a0_after is not None: self.a0 = self.a0_after
        if self.phi_after is not None: self.phi = self.phi_after
        if self.sigma_after is not None: self.sigma = self.sigma_after
        if self.S_after is not None: self.S = int(self.S_after)
        if self.amp_drift_after is not None: self.amp_drift_per_step = self.amp_drift_after

    def on_shock(self, y_prev: np.ndarray, t_idx: int) -> np.ndarray:
        if self.shock_kind == 'state_eps':
            y_prev = y_prev.copy(); y_prev[0] += self.shock_eps
        elif self.shock_kind == 'param':
            self._param_flip()
        return y_prev

    def generate(self) -> np.ndarray:
        if self.shock_kind == 'switch':
            return seasonal_ar_switch(self)

        rng = np.random.default_rng(self.seed)
        y = np.empty((self.steps + 1, 1), dtype=np.float64)
        y[0, 0] = float(self.initial_cond[0])

        s = self._shock_index()
        shocked = False

        for t in range(1, self.steps + 1):
            if s is not None and t == s and not shocked:
                y[t-1, :] = self.on_shock(y[t-1, :], t-1); shocked = True

            a_t = self.a0 + self.amp_drift_per_step * t
            seasonal = a_t * np.sin(2.0 * np.pi * t / self.S)
            y[t, 0] = seasonal + self.phi * y[t-1, 0] + rng.normal(0.0, self.sigma)

            if not np.isfinite(y[t, 0]):
                raise ValueError(f"Numerical instability at step {t}: y={y[t,0]}")
        return y


def seasonal_ar_switch(m: SeasonalARParams) -> np.ndarray:
    if m.switch_update is None:
        raise ValueError("switch_update must be set for shock_kind='switch'.")
    s = m.shock_step
    if s is None and m.shock_frac is not None:
        s = int(round(m.shock_frac * m.steps))
    if s is None:
        raise ValueError("Provide shock_step or shock_frac for 'switch'.")
    s = max(1, min(m.steps, s))

    A = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None})
    B = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None, **m.switch_update})
    A = type(m).model_validate(A.model_dump())
    B = type(m).model_validate(B.model_dump())

    yA = A.generate()
    yB = B.generate()
    combo = yB.copy()
    combo[:s+1] = yA[:s+1]
    return combo



class GARCHParams(BaseModel): 
    """
    GARCH(1,1): r_t = σ_t ε_t,  ε_t~N(0,1);
    σ_t^2 = ω + α r_{t-1}^2 + β σ_{t-1}^2
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    steps: PositiveInt
    omega: float = Field(0.01, gt=0.0)
    alpha: float = Field(0.06, ge=0.0)
    beta:  float = Field(0.90, ge=0.0)
    r0: float = 0.0
    sigma2_0: Optional[float] = None
    seed: Optional[int] = None

    # shocks
    shock_frac: Optional[float] = None
    shock_step: Optional[int] = None
    shock_kind: Optional[Literal['state_eps','param','switch']] = None
    shock_eps: float = 1e-2
    switch_update: Optional[Dict] = None

    # post-shock parameter replacements
    omega_after: Optional[float] = None
    alpha_after: Optional[float] = None
    beta_after:  Optional[float] = None

    @model_validator(mode='after')
    def _check_params(self):
        # Not hard-failing on alpha+beta>=1 (non-stationary), but warn via comment.
        # You can hard-enforce by raising if desired.
        return self

    def _shock_index(self) -> Optional[int]:
        s = self.shock_step
        if s is None and self.shock_frac is not None:
            s = int(round(self.shock_frac * self.steps))
        return None if s is None else max(1, min(int(self.steps), int(s)))

    def _param_flip(self):
        if self.omega_after is not None: self.omega = self.omega_after
        if self.alpha_after is not None: self.alpha = self.alpha_after
        if self.beta_after  is not None: self.beta  = self.beta_after

    def generate(self) -> np.ndarray:
        if self.shock_kind == 'switch':
            return garch_switch(self)

        rng = np.random.default_rng(self.seed)
        r = np.empty(self.steps + 1, dtype=np.float64)
        r[0] = float(self.r0)

        # init variance
        if self.sigma2_0 is not None:
            sigma2 = float(self.sigma2_0)
        else:
            sigma2 = self.omega / max(1e-9, (1.0 - self.alpha - self.beta)) if (self.alpha + self.beta) < 1.0 else 1.0

        s = self._shock_index()
        shocked = False

        for t in range(1, self.steps + 1):
            if s is not None and t == s and not shocked:
                if self.shock_kind == 'state_eps':
                    r[t-1] += self.shock_eps
                elif self.shock_kind == 'param':
                    self._param_flip()
                shocked = True

            # update variance then draw return
            sigma2 = self.omega + self.alpha * (r[t-1]**2) + self.beta * sigma2
            sigma2 = max(sigma2, 0.0)  # numerical safety
            r[t] = np.sqrt(sigma2) * rng.standard_normal()

            if not np.isfinite(r[t]):
                raise ValueError(f"Numerical instability at step {t}: r={r[t]}")

        return r.reshape(-1, 1)


def garch_switch(m: GARCHParams) -> np.ndarray:
    if m.switch_update is None:
        raise ValueError("switch_update must be set for shock_kind='switch'.")
    s = m.shock_step
    if s is None and m.shock_frac is not None:
        s = int(round(m.shock_frac * m.steps))
    if s is None:
        raise ValueError("Provide shock_step or shock_frac for 'switch'.")
    s = max(1, min(m.steps, s))

    A = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None})
    B = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None, **m.switch_update})
    A = type(m).model_validate(A.model_dump())
    B = type(m).model_validate(B.model_dump())

    rA = A.generate()
    rB = B.generate()
    combo = rB.copy()
    combo[:s+1] = rA[:s+1]
    return combo

# region  
class KSParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    """
    # 1D Kuramoto–Sivashinsky (KS)
    # u_t = -u u_x - ν u_xx - u_xxxx,  x ∈ [0, Lx], periodic
    # Linear part in Fourier: L(k) = ν k^2 - k^4
    # Nonlinear part: N(u) = - 1/2 ∂x(u^2)  (computed pseudospectrally)
    # Time stepping: ETDRK4 (Kassam & Trefethen 2005)
    """
    # grid & time
    nx: PositiveInt = Field(64, description="Number of spatial points (power of 2 recommended) (e.g. 128 or 256)")
    Lx: float       = Field(60.0, gt=0.0, description="Domain length: [0, Lx]") # 22.0
    dt: float       = Field(0.25, gt=0.0, description="Time step")
    steps: PositiveInt = Field(..., description="Number of time steps")
    method: Literal['etdrk4'] = 'etdrk4'  # ETDRK4 only for KS

    # KS coefficients (standard is ν=1, coeff of u_xxxx is fixed to 1)
    nu: float = Field(0.5, gt=0.0, description="Diffusion coefficient (on -u_xx term)") # 1.0

    # initial condition; if None, we make small random noise (seeded)
    initial_cond: Optional[np.ndarray] = None
    seed: Optional[int] = None

    # dealiasing for the nonlinear term (2/3-rule)
    dealias: bool = True

    # shock controls (same semantics as your ODE/SDE API)
    shock_frac: Optional[float] = None
    shock_step: Optional[int]   = None
    shock_kind: Optional[Literal['state_eps','param','switch']] = None
    shock_eps: float = 1e-2
    switch_update: Optional[Dict] = None

    # post-shock parameter replacements (for 'param' / 'switch')
    nu_after: Optional[float] = None

    @model_validator(mode='after')
    def _fill_or_check_ic(self):
        if self.initial_cond is None:
            rng = np.random.default_rng(self.seed)
            arr = 0.1 * rng.standard_normal(self.nx)          # small random field
            object.__setattr__(self, 'initial_cond', arr.astype(np.float64))
        else:
            arr = np.asarray(self.initial_cond, dtype=np.float64).reshape(-1)
            if arr.size != self.nx:
                raise ValueError(f"initial_cond must have length nx={self.nx}")
            object.__setattr__(self, 'initial_cond', arr)
        return self
    
    # --- public API: generate trajectory (steps+1, nx) ---
    def generate(self) -> np.ndarray:
        if self.method != 'etdrk4':
            raise ValueError("KS uses ETDRK4; set method='etdrk4'.")
        if self.shock_kind == 'switch':
            return run_ks_switch(self)
        return run_ks_etdrk4(self)
    
    # --------- internals ---------

def _ks_fourier_grid(nx: int, Lx: float):
    dx = Lx / nx
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)    # rad/s units；
    # k are the wavenumbers (Fourier version of “how wiggly”)
    k2 = k * k
    L_lin = (k2) - (k2 * k2)        # for ν=1; we’ll scale by ν later
    # L is the effect of two terms in Fourier Space 
    return k, L_lin

def _etdrk4_coeffs(L: np.ndarray, dt: float):
    # Kassam & Trefethen contour integrals (robust near L≈0)
    E  = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)  # M roots on unit circle
    LR = dt * L[:, None] + r[None, :]
    Q  = dt * np.real(np.mean((np.exp(LR/2.0) - 1.0) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR*LR)) / (LR**3), axis=1))
    f2 = dt * np.real(np.mean(( 2 + LR + np.exp(LR) * (-2 + LR))              / (LR**3), axis=1))
    f3 = dt * np.real(np.mean((-4 - 3*LR - LR*LR + np.exp(LR) * (4 - LR))     / (LR**3), axis=1))
    return E, E2, Q, f1, f2, f3

def _nonlinear_hat(u: np.ndarray, k: np.ndarray, dealias_mask: Optional[np.ndarray]):
    # N(u) = -1/2 ∂x(u^2); compute via FFT
    u2_hat = np.fft.fft(u * u)
    if dealias_mask is not None:
        u2_hat = u2_hat * dealias_mask
    return -0.5j * k * u2_hat

def _dealias_mask(k: np.ndarray) -> np.ndarray:
    kmax = np.max(np.abs(k))
    cutoff = (2.0 / 3.0) * kmax
    return (np.abs(k) <= cutoff).astype(float)  # 1.0 or 0.0 for clean multiply

def _shock_index(steps: int, shock_step: Optional[int], shock_frac: Optional[float]) -> Optional[int]:
    s = shock_step
    if s is None and shock_frac is not None:
        s = int(round(shock_frac * steps))
    return None if s is None else max(1, min(int(steps), int(s)))

def run_ks_etdrk4(p: KSParams) -> np.ndarray:
    nx, Lx, dt, steps = p.nx, p.Lx, p.dt, p.steps
    k, L_base = _ks_fourier_grid(nx, Lx)
    dealias_mask = _dealias_mask(k) if p.dealias else None

    # Linear operator with ν: L = ν*k^2 - k^4 = ν*L_base + (1-1)*?  (L_base already k^2-k^4)
    L = p.nu * (k*k) - (k*k*k*k)
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(L, dt)

    u = p.initial_cond.copy()
    v = np.fft.fft(u) # Start with FFT of initial condition
    traj = np.empty((steps + 1, nx), dtype=np.float64)
    traj[0] = u

    s = _shock_index(steps, p.shock_step, p.shock_frac)
    shocked = False

    for i in range(1, steps + 1):
        if s is not None and i == s and not shocked:
            if p.shock_kind == 'state_eps':
                u = u + p.shock_eps
                v = np.fft.fft(u)
            elif p.shock_kind == 'param':
                # update ν → ν_after and rebuild ETDRK4 coefficients
                if p.nu_after is not None:
                    p.nu = p.nu_after
                    L = p.nu * (k*k) - (k*k*k*k)
                    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(L, dt)
            shocked = True
            
        # Linear part is done in Fourier, nonlinear part is done in real-space
        
        Nv = _nonlinear_hat(u, k, dealias_mask)
        a  = E2 * v + Q * Nv
        ua = np.fft.ifft(a).real # Inverse FFT to get back real-space
        Na = _nonlinear_hat(ua, k, dealias_mask)

        b  = E2 * v + Q * Na
        ub = np.fft.ifft(b).real
        Nb = _nonlinear_hat(ub, k, dealias_mask)

        c  = E2 * a + Q * (2*Nb - Nv)
        uc = np.fft.ifft(c).real
        Nc = _nonlinear_hat(uc, k, dealias_mask)

        v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        u = np.fft.ifft(v).real

        if not np.isfinite(u).all():
            raise ValueError(f"Numerical instability at step {i}")
        traj[i] = u

    return traj

def run_ks_switch(m: KSParams) -> np.ndarray:
    if m.switch_update is None:
        raise ValueError("switch_update must be set for shock_kind='switch'.")
    s = _shock_index(m.steps, m.shock_step, m.shock_frac)
    if s is None:
        raise ValueError("Provide shock_step or shock_frac for 'switch'.")
    # Disable inner shocks and build two runs (A base, B updated), then splice.
    A = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None})
    B = m.model_copy(update={'shock_kind': None, 'shock_step': None, 'shock_frac': None, **m.switch_update})
    A = type(m).model_validate(A.model_dump())
    B = type(m).model_validate(B.model_dump())

    trajA = run_ks_etdrk4(A)
    trajB = run_ks_etdrk4(B)
    combo = trajB.copy()
    combo[:s+1] = trajA[:s+1]
    return combo

# usage examples:
# # Baseline KS
# ks = KSParams(steps=6000, nx=128, Lx=22.0, dt=0.25, nu=1.0, seed=0)
# traj = ks.generate()  # (6001, 128)

# # Param shock (less diffusion at 35% of training)
# ks_param = KSParams(steps=6000, nx=128, Lx=22.0, dt=0.25, nu=1.0,
#                     shock_frac=0.35, shock_kind='param', nu_after=0.8, seed=1)
# traj_p = ks_param.generate()

# # Switch: splice to a different regime (and IC)
# ks_sw = KSParams(steps=6000, nx=128, Lx=22.0, dt=0.25, nu=1.0,
#                  shock_frac=0.35, shock_kind='switch',
#                  switch_update={'nu':0.7, 'seed':42})
# traj_sw = ks_sw.generate()