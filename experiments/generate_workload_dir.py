""" 
LLM-generated workload snippets for testing and evaluation. 
Any labels for physical meaning take with the largest grain of salt, I kept ones that generated interesting patterns.
Transforms for advection after initial conditions are bugged (only use first frame).
"""

from __future__ import annotations
import json, hashlib, shutil, time, os, sys, platform
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Dict, Any, Callable, Iterable, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any, Iterable
import json
import re 

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class GridSpec:
    nx: int
    ny: int
    T: int
    seed: int = 0
    def __post_init__(self):
        if self.nx <= 0 or self.ny <= 0 or self.T <= 0:
            raise ValueError("nx, ny, T must be positive")


@dataclass(frozen=True, slots=True)
class GenOpts:
    C0: float = 1.0
    A: float = 1.0
    noise_sigma: float = 0.0 


class Generator(Protocol):
    """A Generator maps (spec, opts, seed) -> series C of shape (T, nx, ny)."""
    name: str
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray: ...
    def default_params(self) -> Dict[str, Any]: ...


GEN_REGISTRY: Dict[str, Generator] = {}

def register_generator(gen: Generator) -> None:
    if gen.name in GEN_REGISTRY:
        raise KeyError(f"Duplicate generator name: {gen.name}")
    GEN_REGISTRY[gen.name] = gen


def split_seed(seed: int, k: int) -> List[int]:
    """Deterministically derive k sub-seeds from a series seed."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**31 - 1, size=k).tolist()


class Transform(Protocol):
    name: str
    def __call__(self, C: np.ndarray) -> np.ndarray: ...
    def to_meta(self) -> Dict[str, Any]: ...


def pipeline(transforms: Iterable[Transform]) -> Callable[[np.ndarray], Tuple[np.ndarray, List[Dict[str, Any]]]]:
    """Compose transforms into a single callable that returns (C_out, provenance)."""
    tlist = list(transforms)
    def run(C: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        for t in tlist:
            _assert_3d(C)
            C = t(C)
        return C, [t.to_meta() for t in tlist]
    return run

def _assert_3d(C: np.ndarray):
    if C.ndim != 3:
        raise ValueError(f"Expected C to have shape (T, nx, ny); got {C.shape}")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _sha256_of_array(array: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    h.update(array.tobytes(order="C"))
    return h.hexdigest()

def _gaussian_smooth_fft2(field: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return field
    import numpy.fft as fft
    nx, ny = field.shape
    kx = fft.fftfreq(nx)[:, None]
    ky = fft.fftfreq(ny)[None, :]
    G = np.exp(-2 * (np.pi ** 2) * sigma * sigma * (kx ** 2 + ky ** 2))
    F = fft.fft2(field)
    return np.real(fft.ifft2(F * G))

def _bilinear_periodic(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nx, ny = img.shape
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    dx = x - x0
    dy = y - y0
    x0 %= nx; y0 %= ny
    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny
    Ia = img[x0, y0]; Ib = img[x1, y0]
    Ic = img[x0, y1]; Id = img[x1, y1]
    out = Ia*(1-dx)*(1-dy) + Ib*dx*(1-dy) + Ic*(1-dx)*dy + Id*dx*dy
    return out.astype(np.float32)

def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k

def temporal_gaussian(C: np.ndarray, sigma_t: float) -> np.ndarray:
    if sigma_t <= 0:
        return C
    T, nx, ny = C.shape
    k = _gaussian_kernel1d(sigma_t)
    r = (len(k) - 1) // 2
    left  = C[:r][::-1]
    right = C[-r:][::-1] if r > 0 else C[:0]  # empty slice when r==0
    Cp = np.concatenate([left, C, right], axis=0)
    out = np.empty_like(C)
    for t in range(T):
        seg = Cp[t:t + len(k)]  # (len(k), nx, ny)
        out[t] = np.tensordot(k, seg, axes=(0, 0))
    return out

# Transforms

@dataclass(slots=True)
class Tag(Transform):
    """No-op transform that only writes metadata into provenance."""
    kv: Dict[str, Any]
    name: str = "tag"
    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        return C.astype(np.float32, copy=False)
    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.kv)}

@dataclass(slots=True)
class TemporalHysteresis(Transform):
    """Per-frame smoothing with asymmetric rise/decay rates."""
    rise: float = 0.6   # 0..1, larger -> faster rise
    decay: float = 0.1  # 0..1, smaller -> slower decay
    name: str = "temporal_hysteresis"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        out = np.empty_like(C, dtype=np.float32)
        out[0] = C[0]
        for t in range(1, T):
            prev = out[t-1]
            cur  = C[t]
            mask_rise = (cur >= prev)
            # faster blend when rising, slower when decaying
            out[t] = np.where(mask_rise,
                              (1.0 - self.rise) * prev + self.rise * cur,
                              (1.0 - self.decay) * prev + self.decay * cur).astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"rise": float(self.rise), "decay": float(self.decay)}}

@dataclass(slots=True)
class GradientBand(Transform):
    """Multiply by (1 + beta * normalized |grad|)^p to emphasize moving interfaces."""
    beta: float = 1.0
    power: float = 1.0
    eps: float = 1e-6
    name: str = "gradient_band"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        out = np.empty_like(C, dtype=np.float32)
        for t in range(T):
            frame = C[t]
            gx = np.roll(frame, -1, axis=0) - np.roll(frame, 1, axis=0)
            gy = np.roll(frame, -1, axis=1) - np.roll(frame, 1, axis=1)
            g = np.sqrt(gx*gx + gy*gy)
            s = np.median(g) + self.eps
            factor = (1.0 + self.beta * (g / s)) ** self.power
            out[t] = (frame * factor).astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"beta": self.beta, "power": self.power}}


@dataclass(slots=True)
class SolidBodyRotationAdvection(Transform):
    """Rotate the field a small angle each frame around (cx, cy). Periodic BC via wrap."""
    omega: float = 2*np.pi/480.0   # radians per frame
    cx: float | None = None
    cy: float | None = None
    name: str = "solid_body_rotation"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        cx = self.cx if self.cx is not None else (nx - 1)/2.0
        cy = self.cy if self.cy is not None else (ny - 1)/2.0
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        out = np.empty_like(C, dtype=np.float32)
        out[0] = C[0]

        for t in range(1, T):
            # backtrace by +omega (inverse rotation)
            th = self.omega
            cos, sin = np.cos(th), np.sin(th)
            xr =  cos*(X - cx) + sin*(Y - cy) + cx
            yr = -sin*(X - cx) + cos*(Y - cy) + cy
            xr %= nx; yr %= ny
            out[t] = _bilinear_periodic(out[t-1], xr, yr)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"omega": float(self.omega), "cx": self.cx, "cy": self.cy}}

@dataclass(slots=True)
class MeanderingWindAdvection(Transform):
    """Time-varying uniform wind: vx(t)=ax*sin(wx t+φx), vy(t)=ay*cos(wy t+φy)."""
    ax: float = 0.35
    ay: float = 0.25
    wx: float = 2*np.pi/220
    wy: float = 2*np.pi/180
    phx: float = 0.0
    phy: float = np.pi/3
    name: str = "meandering_wind"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        out = np.empty_like(C, dtype=np.float32)
        out[0] = C[0]

        for t in range(1, T):
            vx = self.ax * np.sin(self.wx * t + self.phx)
            vy = self.ay * np.cos(self.wy * t + self.phy)
            xs = (X - vx) % nx
            ys = (Y - vy) % ny
            out[t] = _bilinear_periodic(out[t-1], xs, ys)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"ax": self.ax, "ay": self.ay, "wx": self.wx, "wy": self.wy, "phx": self.phx, "phy": self.phy}}

@dataclass(slots=True)
class ClampTemporalDelta(Transform):
    """Limit per-pixel change per step to [-dmax, dmax]."""
    dmax: float = 0.05
    name: str = "clamp_temporal_delta"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        out = C.astype(np.float32, copy=True)
        for t in range(1, C.shape[0]):
            prev = out[t-1]
            cur  = out[t]
            diff = np.clip(cur - prev, -self.dmax, self.dmax)
            out[t] = prev + diff
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"dmax": self.dmax}}

@dataclass(slots=True)
class BFECCAdvection(Transform):
    """
    Low-diffusion advection via BFECC:
      1) backtrace t->t-1
      2) forward-trace that result t-1->t to estimate error
      3) correct and final backtrace
    Velocity field provided per frame as (vx, vy) arrays (pixels/frame).
    """
    vx: np.ndarray  # shape (T, nx, ny)
    vy: np.ndarray  # shape (T, nx, ny)
    clamp: bool = True
    name: str = "bfecc_advection"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        assert self.vx.shape == (T, nx, ny) and self.vy.shape == (T, nx, ny)
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        out = np.empty_like(C, dtype=np.float32)
        out[0] = C[0]
        for t in range(1, T):
            # 1) backtrace
            xs1 = (X - self.vx[t]) % nx
            ys1 = (Y - self.vy[t]) % ny
            f1 = _bilinear_periodic(out[t-1], xs1, ys1)
            # 2) forward trace estimate
            xf = (xs1 + self.vx[t]) % nx
            yf = (ys1 + self.vy[t]) % ny
            f2 = _bilinear_periodic(f1, xf, yf)
            # 3) error and correction
            e = out[t-1] - f2
            f_corr = f1 + 0.5 * e
            # final backtrace on corrected
            out[t] = _bilinear_periodic(f_corr, xs1, ys1)
            if self.clamp:
                # optional monotonic clamp to local extrema (Zalesak-style limiter)
                lo = np.minimum.reduce([np.roll(out[t-1], s, 0) for s in (-1,0,1)])
                lo = np.minimum.reduce([lo] + [np.roll(lo, s, 1) for s in (-1,1)])
                hi = np.maximum.reduce([np.roll(out[t-1], s, 0) for s in (-1,0,1)])
                hi = np.maximum.reduce([hi] + [np.roll(hi, s, 1) for s in (-1,1)])
                out[t] = np.clip(out[t], lo, hi).astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"clamp": self.clamp}}

@dataclass(slots=True)
class UnsharpMask(Transform):
    """High-boost filtering: out = frame + alpha*(frame - Gaussian(frame, sigma))."""
    sigma: float = 1.0
    alpha: float = 0.7
    name: str = "unsharp_mask"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        out = np.empty_like(C, dtype=np.float32)
        for t in range(C.shape[0]):
            base = C[t].astype(np.float32)
            blur = _gaussian_smooth_fft2(base, self.sigma).astype(np.float32)
            out[t] = (base + self.alpha * (base - blur)).astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"sigma": self.sigma, "alpha": self.alpha}}

@dataclass(slots=True)
class CurlNoiseAdvection(Transform):
    """
    Advect via divergence-free 'curl noise' flow:
      - build psi(t,x,y) by smoothing white noise spatially and temporally
      - u = (dpsi/dy, -dpsi/dx)
      - semi-Lagrangian backtrace from previous frame
    """
    amp: float = 0.6          # step size multiplier in pixels/frame
    ell: float = 6.0          # spatial correlation sigma (pixels)
    tau: float = 2.0          # temporal correlation sigma (frames)
    seed: int = 1357
    name: str = "curl_noise_advection"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        rng = np.random.default_rng(self.seed)
        # build psi: smooth spatially then temporally
        Psi = rng.normal(0.0, 1.0, size=(T, nx, ny)).astype(np.float32)
        for t in range(T):
            Psi[t] = _gaussian_smooth_fft2(Psi[t], self.ell).astype(np.float32)
        Psi = temporal_gaussian(Psi, self.tau).astype(np.float32)

        # finite differences for velocity (periodic)
        def grad_y(z):  # d/dy
            return 0.5*(np.roll(z, -1, 1) - np.roll(z, 1, 1))
        def grad_x(z):  # d/dx
            return 0.5*(np.roll(z, -1, 0) - np.roll(z, 1, 0))

        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        out = np.empty_like(C, dtype=np.float32); out[0] = C[0].astype(np.float32)

        def bilinear(img, x, y):
            x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
            x1 = (x0 + 1) % nx;          y1 = (y0 + 1) % ny
            dx = x - x0; dy = y - y0
            x0 %= nx; y0 %= ny
            Ia = img[x0, y0]; Ib = img[x1, y0]; Ic = img[x0, y1]; Id = img[x1, y1]
            return (Ia*(1-dx)*(1-dy) + Ib*dx*(1-dy) + Ic*(1-dx)*dy + Id*dx*dy).astype(np.float32)

        for t in range(1, T):
            psi = Psi[t]
            vx = grad_y(psi) * self.amp
            vy = -grad_x(psi) * self.amp
            xs = (X - vx) % nx
            ys = (Y - vy) % ny
            out[t] = bilinear(out[t-1], xs, ys)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"amp": self.amp, "ell": self.ell, "tau": self.tau, "seed": self.seed}}

@dataclass(slots=True)
class MeanVarNormalize(Transform):
    """Map each frame to target mean/std (optionally robust via median/MAD)."""
    target_mean: float = 1.0
    target_std: float = 0.25
    robust: bool = False
    eps: float = 1e-8
    name: str = "mean_var_normalize"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        out = np.empty_like(C, dtype=np.float32)
        for t in range(C.shape[0]):
            F = C[t].astype(np.float32)
            if self.robust:
                m = np.median(F)
                mad = np.median(np.abs(F - m)) + self.eps
                Z = (F - m) / (1.4826 * mad)
            else:
                m = float(F.mean())
                s = float(F.std()) + self.eps
                Z = (F - m) / s
            out[t] = self.target_mean + self.target_std * Z
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"target_mean": self.target_mean, "target_std": self.target_std, "robust": self.robust}}

@dataclass(slots=True)
class CorridorMask(Transform):
    """Apply a static, labyrinth-like mask from low-frequency noise."""
    cutoff: float = 0.0   # threshold in [-1,1] to define corridors
    smooth_sigma: float = 2.0
    name: str = "corridor_mask"
    seed: Optional[int] = None  # for reproducibility

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        s = 12345 if self.seed is None else int(self.seed)
        T, nx, ny = C.shape
        rng = np.random.default_rng(s ^ (nx * 73856093) ^ (ny * 19349663))
        # low-frequency noise via FFT shaping
        noise = rng.normal(0, 1, size=(nx, ny)).astype(np.float32)
        mask = _gaussian_smooth_fft2(noise, sigma=self.smooth_sigma)
        mask = mask / (np.max(np.abs(mask)) + 1e-8)
        corridors = (mask > self.cutoff).astype(np.float32)
        # Thicken slightly to avoid zero-area (optional)
        corridors = np.maximum(corridors,
                               _gaussian_smooth_fft2(corridors, sigma=0.8) > 0.3).astype(np.float32)
        out = (C * corridors[None, ...]).astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"cutoff": self.cutoff, "smooth_sigma": self.smooth_sigma}}

@dataclass(slots=True)
class MassNormalize(Transform):
    """Scale each frame so its sum equals target_sum (defaults to frame 0 sum)."""
    target_sum: Optional[float] = None
    eps: float = 1e-12
    name: str = "mass_normalize"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        out = C.astype(np.float32, copy=True)
        T = out.shape[0]
        ts = float(np.sum(out[0])) if self.target_sum is None else float(self.target_sum)
        for t in range(T):
            s = float(np.sum(out[t]))
            if s > self.eps:
                out[t] *= (ts / s)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"target_sum": self.target_sum, "eps": self.eps}}

@dataclass(slots=True)
class AdvectConstant(Transform):
    """Advect each frame from the previous with constant velocity (vx, vy). Periodic BCs."""
    vx: float  # pixels per frame, x-direction (rows)
    vy: float  # pixels per frame, y-direction (cols)
    name: str = "advect_constant"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        out = np.empty_like(C, dtype=np.float32)
        out[0] = C[0]
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

        for t in range(1, T):
            # Backtrace positions
            xs = (X - self.vx) % nx
            ys = (Y - self.vy) % ny
            out[t] = _bilinear_periodic(out[t-1], xs, ys)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"vx": self.vx, "vy": self.vy}}

@dataclass(slots=True)
class SpatialGaussian(Transform):
    sigma: float
    name: str = "spatial_gaussian"
    def __call__(self, C: np.ndarray) -> np.ndarray:
        T, nx, ny = C.shape
        out = np.empty_like(C)
        for t in range(T):
            out[t] = _gaussian_smooth_fft2(C[t], self.sigma)
        return out
    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"sigma": float(self.sigma)}}

@dataclass(slots=True)
class TemporalGaussian(Transform):
    sigma_t: float
    name: str = "temporal_gaussian"
    def __call__(self, C: np.ndarray) -> np.ndarray:
        return temporal_gaussian(C, self.sigma_t)
    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"sigma_t": float(self.sigma_t)}}

@dataclass(slots=True)
class MapToRange(Transform):
    low: float
    high: float
    mode: str = "global"   # "global" | "per_frame"
    name: str = "map_to_range"
    def __call__(self, C: np.ndarray) -> np.ndarray:
        low, high = float(self.low), float(self.high)
        if self.mode not in ("global", "per_frame"):
            raise ValueError("mode must be 'global' or 'per_frame'")
        if self.mode == "global":
            cmin = np.min(C); cmax = np.max(C)
            if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
                return np.full_like(C, (low + high) / 2.0)
            Y = (C - cmin) / (cmax - cmin)
            return low + Y * (high - low)
        else:
            out = np.empty_like(C)
            for t in range(C.shape[0]):
                cmin = np.min(C[t]); cmax = np.max(C[t])
                if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
                    out[t] = (low + high) / 2.0
                else:
                    Y = (C[t] - cmin) / (cmax - cmin)
                    out[t] = low + Y * (high - low)
            return out
    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"low": self.low, "high": self.high, "mode": self.mode}}

@dataclass(slots=True)
class NonLinearFrameMap(Transform):
    func: Callable[[np.ndarray], np.ndarray]
    low: float = 0.0
    high: float = 1.0
    clip: Optional[Tuple[float, float]] = None
    name: str = "nonlinear_frame_map"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        T, nx, ny = C.shape
        out = np.empty_like(C, dtype=np.float32)
        for t in range(T):
            f = self.func(C[t])
            if self.clip is not None:
                f = np.clip(f, self.clip[0], self.clip[1])
            # normalize to [low, high]
            fmin, fmax = f.min(), f.max()
            if fmax > fmin:
                f = (f - fmin) / (fmax - fmin)
                f = self.low + f * (self.high - self.low)
            else:
                f = np.full_like(f, (self.low + self.high) / 2)
            out[t] = f.astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "params": {
                "func": getattr(self.func, "__name__", str(self.func)),
                "low": self.low,
                "high": self.high,
                "clip": self.clip,
            },
        }

@dataclass(slots=True)
class TurbulentFlicker(Transform):
    sigma: float = 0.3        # log-normal std; ~0.2-0.5 is reasonable
    ell: float = 1.5          # spatial correlation (Gaussian sigma, pixels)
    tau: float = 2.0          # temporal correlation (Gaussian sigma, frames)
    seed: int = 1234
    name: str = "turbulent_flicker"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        rng = np.random.default_rng(self.seed)
        # base white noise
        N = rng.normal(0.0, 1.0, size=(T, nx, ny)).astype(np.float32)
        # spatially smooth each frame
        for t in range(T):
            N[t] = _gaussian_smooth_fft2(N[t], sigma=self.ell).astype(np.float32)
        # temporally smooth along time
        N = temporal_gaussian(N, self.tau)
        # log-normal factor; center median ~ 1
        F = np.exp(self.sigma * N).astype(np.float32)
        med = np.median(F)
        F /= (med + 1e-8)
        out = (C * F).astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"sigma": self.sigma, "ell": self.ell, "tau": self.tau, "seed": self.seed}}

# ---------- Strain-induced extinction: quench where |∇C| is large ----------
@dataclass(slots=True)
class StrainExtinction(Transform):
    k: float = 0.7       # quench strength (larger -> stronger extinction)
    p: float = 1.5       # nonlinearity on normalized gradient
    eps: float = 1e-6
    name: str = "strain_extinction"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        out = np.empty_like(C, dtype=np.float32)
        for t in range(T):
            frame = C[t]
            gx = np.roll(frame, -1, 0) - np.roll(frame, 1, 0)
            gy = np.roll(frame, -1, 1) - np.roll(frame, 1, 1)
            g  = np.sqrt(gx*gx + gy*gy)
            s  = np.median(g) + self.eps
            factor = 1.0 / (1.0 + self.k * (g / s) ** self.p)
            out[t] = (frame * factor).astype(np.float32)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"k": self.k, "p": self.p}}

@dataclass(slots=True)
class ShearAdvection(Transform):
    """ 
    # Shear advection: u(y) = vx0 + shear*(y - cy), v = vy 
    """
    shear: float = 0.004   # pixels/frame per pixel (row-velocity varies across columns)
    vx0: float  = 0.0      # base x-velocity (rows)
    vy: float   = 0.25     # upward velocity (cols); positive moves "up" if y increases upward in your convention
    name: str = "shear_advection"

    def __call__(self, C: np.ndarray) -> np.ndarray:
        _assert_3d(C)
        T, nx, ny = C.shape
        out = np.empty_like(C, dtype=np.float32)
        out[0] = C[0]
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        cy = (ny - 1) / 2.0

        for t in range(1, T):
            # per-column x-velocity from shear; constant v in y
            vx = self.vx0 + self.shear * (Y - cy)
            xs = (X - vx) % nx
            ys = (Y - self.vy) % ny
            out[t] = _bilinear_periodic(out[t-1], xs, ys)
        return out

    def to_meta(self) -> Dict[str, Any]:
        return {"name": self.name, "params": {"shear": self.shear, "vx0": self.vx0, "vy": self.vy}}


@dataclass(slots=True)
class TravelingWaveGen:
    name: str = "traveling_wave"
    kx: float = 2*np.pi/32
    ky: float = 0.0
    omega: float = 2*np.pi/64
    theta: float = 0.0
    def default_params(self) -> Dict[str, Any]:
        return {"kx": float(self.kx), "ky": float(self.ky), "omega": float(self.omega), "theta": float(self.theta)}
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        i = np.arange(spec.nx) - (spec.nx - 1)/2
        j = np.arange(spec.ny) - (spec.ny - 1)/2
        I, J = np.meshgrid(i, j, indexing='ij')
        X = I*np.cos(self.theta) + J*np.sin(self.theta)
        Y = -I*np.sin(self.theta) + J*np.cos(self.theta)
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        for t in range(spec.T):
            phi = self.kx * X + self.ky * Y - self.omega * t
            C[t] = opts.C0 * (1.0 + opts.A * np.maximum(0.0, np.sin(phi)))
        return C

register_generator(TravelingWaveGen())

@dataclass(slots=True)
class StickSlipFrontGen:
    name: str = "stick_slip_front"
    # direction unit vector (ux, uy)
    ux: float = 1.0
    uy: float = 0.0
    v: float = 0.2          # base speed (pixels/frame)
    period: int = 40        # frames per stick-slip cycle
    stick_ratio: float = 0.75  # fraction of period with near-zero extra motion
    jump: float = 8.0       # extra pixels per cycle during slip
    width: float = 2.0      # Gaussian band width
    def default_params(self) -> Dict[str, Any]:
        return {"ux": self.ux, "uy": self.uy, "v": self.v, "period": self.period,
                "stick_ratio": self.stick_ratio, "jump": self.jump, "width": self.width}
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        ux, uy = self.ux, self.uy
        # normalize direction
        nrm = np.sqrt(ux*ux + uy*uy) + 1e-8
        ux, uy = ux/nrm, uy/nrm
        xs = np.arange(spec.nx); ys = np.arange(spec.ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        # coordinate along direction
        S = ux*X + uy*Y
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        slip_len = max(1, int(round((1.0 - self.stick_ratio) * self.period)))
        for t in range(spec.T):
            phase = t % self.period
            # extra displacement in current cycle
            if phase < self.stick_ratio * self.period:
                extra = 0.0
            else:
                # linear ramp over brief slip window
                extra = self.jump * (phase - self.stick_ratio*self.period) / slip_len
            pos = self.v * t + extra
            d = S - pos
            band = np.exp(-0.5 * (d / self.width)**2)
            C[t] = (opts.C0 * (1.0 + opts.A * band)).astype(np.float32)
        return C

register_generator(StickSlipFrontGen())

@dataclass(slots=True)
class PulsatingFlameletsGen:
    name: str = "pulsating_flamelets"
    n_sources: int = 5
    pulse_period: int = 20
    spread_rate: float = 0.2
    r0: float = 1.0                  # reference width for mass conservation
    cap_sigma_frac: float = 0.45     # prevent r from engulfing the whole frame
    desync: bool = True              # optional: de-synchronize source phases

    def default_params(self) -> Dict[str, Any]:
        return {
            "n_sources": self.n_sources,
            "pulse_period": self.pulse_period,
            "spread_rate": float(self.spread_rate),
            "r0": float(self.r0),
            "cap_sigma_frac": float(self.cap_sigma_frac),
            "desync": self.desync,
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 9000)
        sources = [(rng.uniform(0, spec.nx-1), rng.uniform(0, spec.ny-1)) for _ in range(self.n_sources)]
        # optional per-source phase offsets to avoid synchronized pulsing
        phase_off = rng.uniform(0, 2*np.pi, size=self.n_sources) if self.desync else np.zeros(self.n_sources)

        xs = np.arange(spec.nx); ys = np.arange(spec.ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)

        r_cap = self.cap_sigma_frac * float(min(spec.nx, spec.ny))

        for t in range(spec.T):
            frame = np.zeros((spec.nx, spec.ny), dtype=np.float32)
            base_phase = 2*np.pi*(t % self.pulse_period)/self.pulse_period

            for k, (cx, cy) in enumerate(sources):
                r = self.r0 + self.spread_rate * t
                r = float(min(r, r_cap))  # cap the spread
                # in-[0,1] pulse with optional offset
                phase = np.sin(base_phase + phase_off[k]) * 0.5 + 0.5
                # L1-conserving amplitude ~ (r0/r)^2 in 2D
                amp = phase * (self.r0 / r)**2

                d2 = (X - cx)**2 + (Y - cy)**2
                frame += amp * np.exp(-d2 / (2 * r * r))

            C[t] = opts.C0 * (1.0 + opts.A * frame)
        return C

register_generator(PulsatingFlameletsGen())


@dataclass(slots=True)
class MovingInterfaceGen:
    name: str = "moving_interface"
    a: float = 16.0       # ellipse semi-axis x
    b: float = 10.0       # ellipse semi-axis y
    w_band: float = 2.0   # interface band half-width
    ax_path: float = 12.0 # center path amplitude x
    ay_path: float = 10.0 # center path amplitude y
    wx: float = 2*np.pi/120.0
    wy: float = 2*np.pi/90.0
    phi: float = np.pi/5
    def default_params(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "w_band": self.w_band,
                "ax_path": self.ax_path, "ay_path": self.ay_path,
                "wx": self.wx, "wy": self.wy, "phi": self.phi}
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        xs = np.arange(spec.nx); ys = np.arange(spec.ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        cx0, cy0 = (spec.nx-1)/2.0, (spec.ny-1)/2.0
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        for t in range(spec.T):
            cx = cx0 + self.ax_path * np.sin(self.wx * t)
            cy = cy0 + self.ay_path * np.sin(self.wy * t + self.phi)
            # signed distance to ellipse boundary ~ 0
            val = ((X - cx)/self.a)**2 + ((Y - cy)/self.b)**2
            d = np.abs(val - 1.0)  # |levelset - 1|
            band = np.exp(-0.5 * (d / (self.w_band / max(self.a, self.b)))**2)
            C[t] = (opts.C0 * (1.0 + opts.A * band)).astype(np.float32)
        return C

register_generator(MovingInterfaceGen())


@dataclass(slots=True)
class BirthDeathGradualGen:
    name: str = "birth_death_gradual"
    lambda_birth: float = 0.25
    lifetime_range: Tuple[int, int] = (20, 50)
    growth_time: int = 6
    decay_time: int = 6
    r_init: float = 0.02
    r_max: float = 0.3
    max_blobs: int = 50
    def default_params(self) -> Dict[str, Any]:
        return {
            "lambda_birth": float(self.lambda_birth),
            "lifetime_range": list(self.lifetime_range),
            "growth_time": self.growth_time,
            "decay_time": self.decay_time,
            "r_init": float(self.r_init),
            "r_max": float(self.r_max),
            "max_blobs": self.max_blobs
        }
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 5000)
        xs = np.arange(spec.nx); ys = np.arange(spec.ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        blobs: List[Tuple[float, float, int, int]] = []
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)

        r_init = self.r_init * float(min(spec.nx, spec.ny))
        r_max = self.r_max * float(min(spec.nx, spec.ny))

        blobs.append((float(rng.uniform(0, spec.nx-1)),
                      float(rng.uniform(0, spec.ny-1)),
                      0,
                      int(rng.integers(self.lifetime_range[0], self.lifetime_range[1]+1))))

        for t in range(spec.T):
            # age + cull
            blobs = [(cx, cy, age+1, L) for (cx, cy, age, L) in blobs if age+1 < L]
            # births
            n_birth = rng.poisson(self.lambda_birth)
            for _ in range(min(n_birth, self.max_blobs - len(blobs))):
                L = int(rng.integers(self.lifetime_range[0], self.lifetime_range[1]+1))
                blobs.append((float(rng.uniform(0, spec.nx-1)),
                              float(rng.uniform(0, spec.ny-1)),
                              0, L))
                
            # render
            frame = np.zeros((spec.nx, spec.ny), dtype=np.float32)
            for (cx, cy, age, L) in blobs:
                g = min(1.0, age / self.growth_time) if self.growth_time > 0 else 1.0
                time_left = L - age
                d = min(1.0, time_left / self.decay_time) if time_left < self.decay_time and self.decay_time > 0 else 1.0
                amp = opts.A * g * d
                r = r_init + (r_max - r_init) * min(1.0, age / max(L/2, 1))
                frame += amp * np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*r*r))
            C[t] = opts.C0 * (1.0 + frame)
        return C

register_generator(BirthDeathGradualGen())

@dataclass(slots=True)
class SplitMergeHotspotsGen:
    name: str = "split_merge_hotspots"
    period: int = 160
    sep_max: float = 20.0    # max separation in pixels
    width: float = 5.0
    angle: float = np.pi/6   # split axis angle
    def default_params(self) -> Dict[str, Any]:
        return {"period": self.period, "sep_max": self.sep_max,
                "width": self.width, "angle": self.angle}
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        xs = np.arange(spec.nx); ys = np.arange(spec.ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        cx0, cy0 = (spec.nx-1)/2.0, (spec.ny-1)/2.0
        ux, uy = np.cos(self.angle), np.sin(self.angle)
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        for t in range(spec.T):
            tau = (t % self.period) / self.period
            # separation profile: 0 -> sep_max -> 0 (smooth)
            sep = self.sep_max * np.sin(np.pi * tau)
            if tau < 0.5:
                # splitting: weight transitions from 1.0 to 0.5+0.5
                w1 = 1.0 - 2.0 * (0.5 - tau)
            else:
                w1 = 2.0 * (1.0 - tau)
            w1 = np.clip(w1, 0.0, 1.0)
            w2 = 1.0 - w1
            # centers along +/- (ux,uy)
            cx1, cy1 = cx0 - 0.5 * sep * ux, cy0 - 0.5 * sep * uy
            cx2, cy2 = cx0 + 0.5 * sep * ux, cy0 + 0.5 * sep * uy
            g1 = np.exp(-(((X-cx1)**2 + (Y-cy1)**2) / (2*self.width*self.width)))
            g2 = np.exp(-(((X-cx2)**2 + (Y-cy2)**2) / (2*self.width*self.width)))
            frame = w1 * g1 + w2 * g2
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)
        return C

register_generator(SplitMergeHotspotsGen())

@dataclass(slots=True)
class IgnitionExtinctionGen:
    name: str = "ignition_extinction"
    ignition_threshold: float = 0.6
    extinction_rate: float = 0.05
    noise_sigma: float = 0.1
    def default_params(self) -> Dict[str, Any]:
        return {
            "ignition_threshold": float(self.ignition_threshold),
            "extinction_rate": float(self.extinction_rate),
            "noise_sigma": float(self.noise_sigma),
        }
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 8000)
        field = rng.random((spec.nx, spec.ny))
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        for t in range(spec.T):
            ignited = field > self.ignition_threshold
            field = np.where(ignited, field - self.extinction_rate, field + self.extinction_rate * 0.5)
            field += rng.normal(0, self.noise_sigma, field.shape)
            field = np.clip(field, 0, 1)
            smooth = _gaussian_smooth_fft2(field, sigma=1.0)
            C[t] = opts.C0 * (1.0 + opts.A * smooth)
        return C

register_generator(IgnitionExtinctionGen())

@dataclass(slots=True)
class TwoPhaseAlternationGen:
    name: str = "two_phase_alternation"
    block: Tuple[int, int] = (8, 8)
    def default_params(self) -> Dict[str, Any]:
        return {"block": list(self.block)}
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        bx, by = self.block
        mask = np.zeros((spec.nx, spec.ny), dtype=np.float32); mask[:bx, :by] = 1.0
        mask2 = np.roll(mask, shift=(spec.nx//2, spec.ny//2), axis=(0,1))
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        for t in range(spec.T):
            M = mask if (t % 2 == 0) else mask2
            C[t] = opts.C0 * (1.0 + opts.A * M)
        return C

register_generator(TwoPhaseAlternationGen())

@dataclass(slots=True)
class RotatingVortexBandGen:
    name: str = "rotating_vortex_band"
    r0: float = 18.0
    sigma_r: float = 2.0
    k_theta: int = 2
    omega: float = 2*np.pi/140.0
    def default_params(self) -> Dict[str, Any]:
        return {"r0": self.r0, "sigma_r": self.sigma_r, "k_theta": self.k_theta, "omega": self.omega}
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        cx0, cy0 = (spec.nx-1)/2.0, (spec.ny-1)/2.0
        xs = np.arange(spec.nx); ys = np.arange(spec.ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        dx, dy = X - cx0, Y - cy0
        r = np.sqrt(dx*dx + dy*dy)
        theta = np.arctan2(dy, dx)
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        radial = np.exp(-0.5 * ((r - self.r0)/self.sigma_r)**2)
        for t in range(spec.T):
            ang = np.cos(self.k_theta * theta - self.omega * t)
            frame = radial * (1.0 + ang)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)
        return C

register_generator(RotatingVortexBandGen())

@dataclass(slots=True)
class QuasiPeriodicTilesGen:
    name: str = "quasiperiodic_tiles"
    tile: Tuple[int, int] = (8, 8)  # tile size
    periods: Tuple[int, ...] = (23, 29, 31, 37)  # coprime-ish
    def default_params(self) -> Dict[str, Any]:
        return {"tile": list(self.tile), "periods": list(self.periods)}
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 2222)
        px, py = self.tile
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        # assign a period and phase per tile
        nx_tiles = (spec.nx + px - 1) // px
        ny_tiles = (spec.ny + py - 1) // py
        tile_period = rng.choice(self.periods, size=(nx_tiles, ny_tiles))
        tile_phase  = rng.uniform(0, 2*np.pi, size=(nx_tiles, ny_tiles))
        for t in range(spec.T):
            frame = np.zeros((spec.nx, spec.ny), dtype=np.float32)
            for ix in range(nx_tiles):
                for iy in range(ny_tiles):
                    p = tile_period[ix, iy]; ph = tile_phase[ix, iy]
                    val = 0.5 * (1.0 + np.sin(2*np.pi * (t % p)/p + ph))
                    x0, x1 = ix*px, min(spec.nx, (ix+1)*px)
                    y0, y1 = iy*py, min(spec.ny, (iy+1)*py)
                    frame[x0:x1, y0:y1] = val
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)
        return C

register_generator(QuasiPeriodicTilesGen())

# ---------- Bistable reaction–diffusion with sparks (auto-ignition fronts) ----------
@dataclass(slots=True)
class ReactionDiffusionKPPGen:
    name: str = "reaction_diffusion_kpp"
    D: float = 0.6                 # diffusion coefficient
    r: float = 3.0                 # reaction rate
    alpha: float = 0.2             # ignition threshold in u*(1-u)*(u-alpha)
    dt: float = 0.15               # Euler time step per substep
    steps_per_frame: int = 2       # internal substeps per output frame
    n_init: int = 12               # initial hot spots
    spark_rate: float = 0.03       # expected sparks per frame
    spark_sigma: float = 1.6       # std of spark Gaussian (pixels)
    noise_std: float = 0.0         # additive noise per substep

    def default_params(self) -> Dict[str, Any]:
        return {
            "D": self.D, "r": self.r, "alpha": self.alpha, "dt": self.dt,
            "steps_per_frame": self.steps_per_frame, "n_init": self.n_init,
            "spark_rate": self.spark_rate, "spark_sigma": self.spark_sigma, "noise_std": self.noise_std
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 41000)
        nx, ny, T = spec.nx, spec.ny, spec.T
        u = np.zeros((nx, ny), dtype=np.float32)

        # seed initial hot spots
        xs = np.arange(nx); ys = np.arange(ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        for _ in range(self.n_init):
            cx = rng.uniform(0, nx-1); cy = rng.uniform(0, ny-1)
            u += np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*self.spark_sigma*self.spark_sigma)).astype(np.float32)
        u = np.clip(u, 0.0, 1.0)

        def laplace5(z):
            return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4.0*z)

        C = np.empty((T, nx, ny), dtype=np.float32)
        for t in range(T):
            for _ in range(self.steps_per_frame):
                # random sparks
                if rng.random() < self.spark_rate:
                    cx = rng.uniform(0, nx-1); cy = rng.uniform(0, ny-1)
                    u += np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*self.spark_sigma*self.spark_sigma)).astype(np.float32)
                    u = np.clip(u, 0.0, 1.0)
                # reaction–diffusion update
                lap = laplace5(u)
                R = self.r * u * (1.0 - u) * (u - self.alpha)
                u = u + self.dt * (self.D * lap + R)
                if self.noise_std > 0:
                    u = u + self.noise_std * rng.normal(0.0, 1.0, u.shape).astype(np.float32)
                u = np.clip(u, 0.0, 1.0)
            C[t] = (opts.C0 * (1.0 + opts.A * u)).astype(np.float32)
        return C

register_generator(ReactionDiffusionKPPGen())

@dataclass(slots=True)
class MovingGaussiansGen:
    name: str = "moving_gaussians"
    n_blobs: int = 6
    radius: float = 0.3
    speed_range: Tuple[float, float] = (0.2, 0.6)  # pixels per frame
    wrap: bool = True

    def default_params(self) -> Dict[str, Any]:
        return {
            "n_blobs": self.n_blobs,
            "radius": float(self.radius),
            "speed_range": list(self.speed_range),
            "wrap": self.wrap,
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 50200)
        nx, ny, T = spec.nx, spec.ny, spec.T

        self.radius = float(self.radius * min(nx, ny))  # scale radius with grid size

        # init centers and velocities
        cx = rng.uniform(0, nx - 1, size=self.n_blobs)
        cy = rng.uniform(0, ny - 1, size=self.n_blobs)
        theta = rng.uniform(0, 2 * np.pi, size=self.n_blobs)
        speed = rng.uniform(self.speed_range[0], self.speed_range[1], size=self.n_blobs)
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)

        xs = np.arange(nx); ys = np.arange(ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        r2 = self.radius * self.radius

        C = np.empty((T, nx, ny), dtype=np.float32)
        for t in range(T):
            frame = np.zeros((nx, ny), dtype=np.float32)
            for i in range(self.n_blobs):
                # evaluate exact Gaussian (no temporal interpolation)
                frame += np.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2) / (2 * r2)).astype(np.float32)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)
            # advance centers
            cx += vx; cy += vy
            if self.wrap:
                cx %= nx; cy %= ny
            else:
                # reflect at boundaries (optional branch)
                for i in range(self.n_blobs):
                    if cx[i] < 0 or cx[i] > nx - 1: vx[i] *= -1; cx[i] = np.clip(cx[i], 0, nx - 1)
                    if cy[i] < 0 or cy[i] > ny - 1: vy[i] *= -1; cy[i] = np.clip(cy[i], 0, ny - 1)
        return C

register_generator(MovingGaussiansGen())

@dataclass(slots=True)
class GrayScottRDGen:
    name: str = "gray_scott_rd"
    Du: float = 0.16
    Dv: float = 0.08
    F: float = 0.035
    k: float = 0.060
    dt: float = 1.0
    steps_per_frame: int = 8
    n_seed: int = 10
    seed_sigma: float = 2.0

    def default_params(self) -> Dict[str, Any]:
        return {"Du": self.Du, "Dv": self.Dv, "F": self.F, "k": self.k,
                "dt": self.dt, "steps_per_frame": self.steps_per_frame,
                "n_seed": self.n_seed, "seed_sigma": self.seed_sigma}

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 30001)
        nx, ny, T = spec.nx, spec.ny, spec.T
        u = np.ones((nx, ny), dtype=np.float32)
        v = np.zeros((nx, ny), dtype=np.float32)
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

        # local seeds in v
        for _ in range(self.n_seed):
            cx = rng.uniform(0, nx-1); cy = rng.uniform(0, ny-1)
            v += np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*self.seed_sigma**2)).astype(np.float32)
        v = np.clip(v, 0.0, 1.0)

        def lap(z):
            return (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4*z)

        C = np.empty((T, nx, ny), dtype=np.float32)
        for t in range(T):
            for _ in range(self.steps_per_frame):
                uvv = u * v * v
                u += self.dt * (self.Du * lap(u) - uvv + self.F * (1 - u))
                v += self.dt * (self.Dv * lap(v) + uvv - (self.F + self.k) * v)
                u = np.clip(u, 0.0, 1.0); v = np.clip(v, 0.0, 1.0)
            C[t] = (opts.C0 * (1.0 + opts.A * v)).astype(np.float32)
        return C

register_generator(GrayScottRDGen())

@dataclass(slots=True)
class CurlNoiseDyeGen:
    name: str = "curl_noise_dye"
    amp: float = 0.7
    ell: float = 5.0
    tau: float = 2.5
    decay: float = 0.01
    inj_rate: float = 0.3            # expected injections per frame
    inj_sigma: float = 2.0
    steps_per_frame: int = 1
    seed_stream: int = 4242
    seed_inj: int = 4243

    def default_params(self) -> Dict[str, Any]:
        return {"amp": self.amp, "ell": self.ell, "tau": self.tau, "decay": self.decay,
                "inj_rate": self.inj_rate, "inj_sigma": self.inj_sigma,
                "steps_per_frame": self.steps_per_frame}

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        dye = np.zeros((nx, ny), dtype=np.float32)
        C = np.empty((T, nx, ny), dtype=np.float32)

        # build streamfunction psi(t,x,y)
        rng_s = np.random.default_rng(seed + self.seed_stream)
        Psi = rng_s.normal(0.0, 1.0, size=(T, nx, ny)).astype(np.float32)
        for t in range(T):
            Psi[t] = _gaussian_smooth_fft2(Psi[t], self.ell).astype(np.float32)
        Psi = temporal_gaussian(Psi, self.tau).astype(np.float32)

        def grad_y(z): return 0.5*(np.roll(z, -1, 1) - np.roll(z, 1, 1))
        def grad_x(z): return 0.5*(np.roll(z, -1, 0) - np.roll(z, 1, 0))

        def bilinear(img, x, y):
            x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
            x1 = (x0 + 1) % nx;          y1 = (y0 + 1) % ny
            dx = x - x0; dy = y - y0
            x0 %= nx; y0 %= ny
            Ia = img[x0, y0]; Ib = img[x1, y0]; Ic = img[x0, y1]; Id = img[x1, y1]
            return (Ia*(1-dx)*(1-dy) + Ib*dx*(1-dy) + Ic*(1-dx)*dy + Id*dx*dy).astype(np.float32)

        rng_i = np.random.default_rng(seed + self.seed_inj)

        for t in range(T):
            # decay
            dye = (1.0 - self.decay) * dye

            # advect (can substep)
            for _ in range(self.steps_per_frame):
                psi = Psi[t]
                vx = grad_y(psi) * self.amp
                vy = -grad_x(psi) * self.amp
                xs = (X - vx) % nx
                ys = (Y - vy) % ny
                dye = bilinear(dye, xs, ys)

            # random injections
            n_inj = rng_i.poisson(self.inj_rate)
            for _ in range(n_inj):
                cx = rng_i.uniform(0, nx-1); cy = rng_i.uniform(0, ny-1)
                g = np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*self.inj_sigma**2)).astype(np.float32)
                dye += g

            C[t] = (opts.C0 * (1.0 + opts.A * np.clip(dye, 0.0, None))).astype(np.float32)
        return C

register_generator(CurlNoiseDyeGen())


@dataclass(slots=True)
class NBodyGaussianGen:
    """
    N point-masses on a 2D torus; gravitational-like attraction with softening.
    Positions are in pixel coordinates on [0..nx) × [0..ny).
    Rendering uses periodic Gaussians; total L¹ equals total mass.
    """
    name: str = "nbody_gaussian"

    # Dynamics
    n_bodies: int = 24
    G: float = 1.0                 # attraction strength
    softening: float = 0.8         # Plummer softening length (pixels)
    dt: float = 0.2
    steps_per_frame: int = 2
    damping: float = 0.0           # small linear drag per step, e.g. 0.01
    wrap: bool = True              # periodic BC (torus)

    # Mass & init
    total_mass: float = 1.0
    mass_lognorm_mu: float = -0.5  # log-normal for heterogeneity (before renorm)
    mass_lognorm_sigma: float = 0.6
    init_speed_frac: float = 0.02  # as fraction of min(nx,ny) per frame

    # Rendering
    sigma_pix: float = 1.5         # base Gaussian width (pixels)
    size_scales_with_mass: bool = True  # sigma ∝ m^(1/3)
    sigma_min: float = 0.8
    sigma_max: float = 4.0

    seed_offset: int = 81001

    def default_params(self) -> Dict[str, Any]:
        return {
            "n_bodies": self.n_bodies, "G": self.G, "softening": self.softening,
            "dt": self.dt, "steps_per_frame": self.steps_per_frame, "damping": self.damping,
            "wrap": self.wrap, "total_mass": self.total_mass,
            "mass_lognorm_mu": self.mass_lognorm_mu, "mass_lognorm_sigma": self.mass_lognorm_sigma,
            "init_speed_frac": self.init_speed_frac,
            "sigma_pix": self.sigma_pix, "size_scales_with_mass": self.size_scales_with_mass,
            "sigma_min": self.sigma_min, "sigma_max": self.sigma_max,
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T

        # --- init state ---
        pos = np.empty((self.n_bodies, 2), dtype=np.float64)
        pos[:, 0] = rng.uniform(0, nx - 1, size=self.n_bodies)
        pos[:, 1] = rng.uniform(0, ny - 1, size=self.n_bodies)

        # Masses (heterogeneous) -> normalize to total_mass
        m = rng.lognormal(self.mass_lognorm_mu, self.mass_lognorm_sigma, size=self.n_bodies)
        m = (self.total_mass * m / m.sum()).astype(np.float64)

        # Initial velocities: small random, zero net momentum
        v_scale = self.init_speed_frac * min(nx, ny)
        vel = rng.normal(0.0, v_scale, size=(self.n_bodies, 2))
        vel -= (m[:, None] * vel).sum(axis=0, keepdims=True) / m.sum()

        # Precompute grid for rendering
        X, Y = np.meshgrid(np.arange(nx, dtype=np.float64),
                           np.arange(ny, dtype=np.float64), indexing="ij")

        C = np.empty((T, nx, ny), dtype=np.float32)

        soft2 = float(self.softening**2)

        def min_image(dx: np.ndarray, L: float) -> np.ndarray:
            # map differences to [-L/2, L/2) for periodic torus distances
            return dx - np.round(dx / L) * L

        @np.errstate(divide="ignore", invalid="ignore")
        def accelerations(p: np.ndarray) -> np.ndarray:
            # Pairwise displacements with minimum image convention
            dx = p[None, :, 0] - p[:, None, 0]
            dy = p[None, :, 1] - p[:, None, 1]
            if self.wrap:
                dx = min_image(dx, nx)
                dy = min_image(dy, ny)
            r2 = dx*dx + dy*dy + soft2
            inv_r3 = 1.0 / (r2 * np.sqrt(r2))
            np.fill_diagonal(inv_r3, 0.0)                # no self-force
            # a_i = G * sum_j m_j * r_ij / |r_ij|^3
            ax = self.G * np.sum((dx * inv_r3) * m[None, :], axis=1)
            ay = self.G * np.sum((dy * inv_r3) * m[None, :], axis=1)
            return np.stack([ax, ay], axis=1)

        # --- velocity-Verlet integration across frames ---
        a = accelerations(pos)
        dt = float(self.dt)
        damp = float(self.damping)

        for t in range(T):
            # render current state to scalar field
            C[t] = self._render_frame(pos, m, X, Y, nx, ny).astype(np.float32)
            # advance (possibly with substeps)
            for _ in range(max(1, self.steps_per_frame)):
                # half-kick
                vel += 0.5 * dt * a
                if damp > 0:
                    vel *= (1.0 - damp)
                # drift
                pos += dt * vel
                if self.wrap:
                    pos[:, 0] %= nx
                    pos[:, 1] %= ny
                else:
                    pos[:, 0] = np.clip(pos[:, 0], 0, nx - 1)
                    pos[:, 1] = np.clip(pos[:, 1], 0, ny - 1)
                # new acceleration at updated positions
                a_new = accelerations(pos)
                # half-kick
                vel += 0.5 * dt * a_new
                if damp > 0:
                    vel *= (1.0 - damp)
                a = a_new

        # Map to visualization range
        return (opts.C0 * (1.0 + opts.A * C)).astype(np.float32)

    def _render_frame(self, pos, m, X, Y, nx, ny) -> np.ndarray:
        # Summed periodic Gaussians; L¹ ≈ total mass.
        field = np.zeros_like(X, dtype=np.float64)
        two_pi = 2.0 * np.pi
        for i in range(self.n_bodies):
            # mass-dependent sigma (optional)
            if self.size_scales_with_mass:
                sigma = self.sigma_pix * max( (m[i] / (self.total_mass / self.n_bodies)) ** (1/3), 0.5)
            else:
                sigma = self.sigma_pix
            sigma = float(np.clip(sigma, self.sigma_min, self.sigma_max))
            # periodic distances from grid to particle i
            dx = X - pos[i, 0];  dx = dx - np.round(dx / nx) * nx
            dy = Y - pos[i, 1];  dy = dy - np.round(dy / ny) * ny
            r2 = dx*dx + dy*dy
            g = np.exp(-0.5 * r2 / (sigma * sigma))
            # continuous normalization on the torus (good if sigma << domain)
            field += (m[i] / (two_pi * sigma * sigma)) * g
        return field
    
register_generator(NBodyGaussianGen())


@dataclass(slots=True)
class LevyFlightHotspotsGen:
    name: str = "levy_flight_hotspots"
    n_agents: int = 10
    alpha: float = 1.5              # Pareto tail exponent (1 < alpha < 3 typical)
    min_step: float = 0.2
    radius: float = 2.2
    growth_tau: int = 6
    decay_tau: int = 8
    wrap: bool = True
    seed_offset: int = 55001

    def default_params(self) -> Dict[str, Any]:
        return {"n_agents": self.n_agents, "alpha": self.alpha, "min_step": self.min_step,
                "radius": self.radius, "growth_tau": self.growth_tau, "decay_tau": self.decay_tau, "wrap": self.wrap}

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

        cx = rng.uniform(0, nx-1, size=self.n_agents)
        cy = rng.uniform(0, ny-1, size=self.n_agents)
        amp = np.zeros(self.n_agents, dtype=np.float32)  # smooth on/off per agent
        r2 = self.radius**2

        C = np.empty((T, nx, ny), dtype=np.float32)
        for t in range(T):
            # growth/decay envelope
            amp += (1.0 - amp) / max(1, self.growth_tau)
            if t % max(1, self.decay_tau) == 0:
                # randomly pick some to decay
                mask = rng.random(self.n_agents) < 0.2
                amp[mask] *= 0.5

            # render
            frame = np.zeros((nx, ny), dtype=np.float32)
            for i in range(self.n_agents):
                frame += amp[i] * np.exp(-((X - cx[i])**2 + (Y - cy[i])**2)/(2*r2)).astype(np.float32)

            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)

            # Lévy steps (Pareto with scale = min_step)
            theta = rng.uniform(0, 2*np.pi, size=self.n_agents)
            step = self.min_step * (1.0 + rng.pareto(self.alpha, size=self.n_agents))
            cx += step * np.cos(theta)
            cy += step * np.sin(theta)
            if self.wrap:
                cx %= nx; cy %= ny
            else:
                cx = np.clip(cx, 0, nx-1); cy = np.clip(cy, 0, ny-1)
        return C

register_generator(LevyFlightHotspotsGen())

@dataclass(slots=True)
class CrispBirthDeathGen:
    name: str = "crisp_birth_death"
    lambda_birth: float = 0.30
    lifetime_range: Tuple[int, int] = (20, 120)
    radius: float = 2.2
    max_blobs: int = 64

    def default_params(self) -> Dict[str, Any]:
        return {
            "lambda_birth": float(self.lambda_birth),
            "lifetime_range": list(self.lifetime_range),
            "radius": float(self.radius),
            "max_blobs": int(self.max_blobs),
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 50100)
        xs = np.arange(spec.nx); ys = np.arange(spec.ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        blobs: List[Tuple[float, float, int, int]] = []  # (cx, cy, age, L)
        C = np.empty((spec.T, spec.nx, spec.ny), dtype=np.float32)
        r2 = self.radius * self.radius
        for t in range(spec.T):
            # age and cull
            blobs = [(cx, cy, age + 1, L) for (cx, cy, age, L) in blobs if age + 1 < L]
            # births
            n_birth = rng.poisson(self.lambda_birth)
            for _ in range(min(n_birth, self.max_blobs - len(blobs))):
                L = int(rng.integers(self.lifetime_range[0], self.lifetime_range[1] + 1))
                blobs.append((
                    float(rng.uniform(0, spec.nx - 1)),
                    float(rng.uniform(0, spec.ny - 1)),
                    0, L
                ))
            # render (pure on/off amplitude; no temporal ramp)
            frame = np.zeros((spec.nx, spec.ny), dtype=np.float32)
            for (cx, cy, _, _) in blobs:
                frame += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r2)).astype(np.float32)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)
        return C

register_generator(CrispBirthDeathGen())

@dataclass(slots=True)
class IgnitionRefractoryCAGen:
    """
    States: 0 = fuel, 1..R = burning/refractory countdown.
    If any 8-neighbor is burning and a Bernoulli(p) succeeds, cell ignites.
    """
    name: str = "ignition_ca"
    refractory: int = 6
    ignite_prob: float = 0.35
    spontaneous_rate: float = 0.001  # per-cell spontaneous ignition per frame

    def default_params(self) -> Dict[str, Any]:
        return {"refractory": self.refractory, "ignite_prob": self.ignite_prob, "spontaneous_rate": self.spontaneous_rate}

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 50400)
        nx, ny, T = spec.nx, spec.ny, spec.T
        state = np.zeros((nx, ny), dtype=np.int16)  # 0..refractory
        C = np.empty((T, nx, ny), dtype=np.float32)

        def neighbors_burning(s):
            # 8-neighborhood burning mask (state==refractory initially on ignition)
            nb = (
                (np.roll(s, 1, 0) == self.refractory) | (np.roll(s, -1, 0) == self.refractory) |
                (np.roll(s, 1, 1) == self.refractory) | (np.roll(s, -1, 1) == self.refractory) |
                (np.roll(np.roll(s, 1, 0), 1, 1) == self.refractory) |
                (np.roll(np.roll(s, 1, 0), -1, 1) == self.refractory) |
                (np.roll(np.roll(s, -1, 0), 1, 1) == self.refractory) |
                (np.roll(np.roll(s, -1, 0), -1, 1) == self.refractory)
            )
            return nb

        for t in range(T):
            # ignition candidates
            nb = neighbors_burning(state)
            fuel = (state == 0)
            ignite_nb = fuel & nb & (rng.random((nx, ny)) < self.ignite_prob)
            ignite_sp = fuel & (rng.random((nx, ny)) < self.spontaneous_rate)
            ignite = ignite_nb | ignite_sp

            # update states (crisp, integer countdown)
            state[ignite] = self.refractory
            active = (state > 0)
            state[active] -= 1

            # map to intensity (fuel low; burning high; refractory moderate)
            frame = np.zeros((nx, ny), dtype=np.float32)
            frame[state == 0] = 0.05
            frame[state == self.refractory] = 1.0
            # linearly fade refractory 1..(refractory-1)
            mask_ref = (state > 0) & (state < self.refractory)
            frame[mask_ref] = 0.2 + 0.6 * (state[mask_ref] / float(self.refractory))
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)
        return C

register_generator(IgnitionRefractoryCAGen())

@dataclass(slots=True)
class LiftedJetAutoIgnitionGen:
    name: str = "lifted_jet_autoignition"
    n_jets: int = 3
    jet_halfwidth: float = 3.0
    period: int = 60
    duty: float = 0.6                # fraction on
    delay_range: Tuple[int, int] = (0, 80)
    theta: float = 0.45              # ignition threshold
    k: float = 10.0                  # logistic steepness
    spread_sigma: float = 1.2        # local diffusion kernel
    decay: float = 0.02              # per-frame decay before ignition nonlinearity
    noise_std: float = 0.01

    def default_params(self) -> Dict[str, Any]:
        return {
            "n_jets": self.n_jets, "jet_halfwidth": self.jet_halfwidth, "period": self.period,
            "duty": self.duty, "delay_range": list(self.delay_range), "theta": self.theta, "k": self.k,
            "spread_sigma": self.spread_sigma, "decay": self.decay, "noise_std": self.noise_std
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + 42000)
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

        # jet x-positions
        xs = np.linspace(nx*0.15, nx*0.85, self.n_jets)
        delays = rng.integers(self.delay_range[0], self.delay_range[1]+1, size=self.n_jets)
        phases = rng.uniform(0, self.period, size=self.n_jets)

        state = np.zeros((nx, ny), dtype=np.float32)
        C = np.empty((T, nx, ny), dtype=np.float32)

        def inj_frame(t):
            frame = np.zeros_like(state)
            for j, cx in enumerate(xs):
                on = (t >= delays[j]) and (((t - delays[j] + phases[j]) % self.period) < self.duty * self.period)
                if on:
                    # bottom injection band (first few rows)
                    cy = 1.5
                    frame += np.exp(-(((X-cx)**2 + (Y-cy)**2) / (2*self.jet_halfwidth*self.jet_halfwidth))).astype(np.float32)
            return frame

        for t in range(T):
            # decay + local spread
            state = (1.0 - self.decay) * state
            state = _gaussian_smooth_fft2(state, sigma=self.spread_sigma).astype(np.float32)
            # inject new fuel/heat at the bottom
            state += inj_frame(t)
            # noise
            if self.noise_std > 0:
                state += self.noise_std * rng.normal(0.0, 1.0, size=state.shape).astype(np.float32)
            state = np.clip(state, 0.0, 1.0)
            # ignition nonlinearity (S-shaped)
            ign = 1.0 / (1.0 + np.exp(-self.k * (state - self.theta)))
            C[t] = (opts.C0 * (1.0 + opts.A * ign)).astype(np.float32)
        return C

register_generator(LiftedJetAutoIgnitionGen())

@dataclass
class _Bouncer:
    """
    Stochastic 2D kinematics with specular reflection:
      - velocity jitter (Gaussian acceleration noise each frame)
      - occasional random turn (random heading change)
      - speed clamp [v_min, v_max]
      - reflection on [margin_x, nx-1-margin_x] × [margin_y, ny-1-margin_y]
    """
    nx: int
    ny: int
    v_min: float = 0.25
    v_max: float = 0.90
    accel_sigma: float = 0.05       # per-frame velocity noise
    p_turn: float = 0.03            # per-frame prob of random heading change
    turn_sigma_deg: float = 45.0    # std of random turn (degrees)
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        theta = float(self.rng.uniform(0.0, 2*np.pi))
        speed = float(self.rng.uniform(self.v_min, self.v_max))
        self.vx = speed * np.cos(theta)
        self.vy = speed * np.sin(theta)

    def step(self, cx: float, cy: float, margin_x: float, margin_y: float) -> Tuple[float, float]:
        # Clamp margins so x_lo <= x_hi and y_lo <= y_hi
        max_mx = 0.49 * (self.nx - 1.0)
        max_my = 0.49 * (self.ny - 1.0)
        margin_x = float(np.clip(margin_x, 0.0, max_mx))
        margin_y = float(np.clip(margin_y, 0.0, max_my))

        # Random acceleration (velocity jitter)
        self.vx += self.rng.normal(0.0, self.accel_sigma)
        self.vy += self.rng.normal(0.0, self.accel_sigma)

        # Occasional random heading change
        if self.rng.random() < self.p_turn:
            ang = np.deg2rad(self.turn_sigma_deg) * self.rng.normal(0.0, 1.0)
            c, s = np.cos(ang), np.sin(ang)
            vx, vy = self.vx, self.vy
            self.vx, self.vy = vx*c - vy*s, vx*s + vy*c

        # Speed clamp
        sp = (self.vx*self.vx + self.vy*self.vy) ** 0.5
        if sp < 1e-8:
            theta = float(self.rng.uniform(0.0, 2*np.pi))
            self.vx = self.v_min * np.cos(theta)
            self.vy = self.v_min * np.sin(theta)
        else:
            if sp < self.v_min:
                s = self.v_min / sp
                self.vx *= s; self.vy *= s
            elif sp > self.v_max:
                s = self.v_max / sp
                self.vx *= s; self.vy *= s

        # Advance position
        cx += self.vx; cy += self.vy

        # Reflect with margins (keep full object visible)
        x_lo, x_hi = float(margin_x), float(self.nx - 1 - margin_x)
        y_lo, y_hi = float(margin_y), float(self.ny - 1 - margin_y)

        if cx < x_lo:
            cx = x_lo + (x_lo - cx); self.vx = -self.vx
        elif cx > x_hi:
            cx = x_hi - (cx - x_hi); self.vx = -self.vx

        if cy < y_lo:
            cy = y_lo + (y_lo - cy); self.vy = -self.vy
        elif cy > y_hi:
            cy = y_hi - (cy - y_hi); self.vy = -self.vy

        return cx, cy


# ---------- Gaussian blob with OU width; accelerates/turns; margin-safe ----------
@dataclass(slots=True)
class DVDBounceGaussianGen:
    name: str = "dvd_bounce_gaussian"
    # width (sigma) as fraction of min(nx,ny)
    sigma_min_frac: float = 0.12
    sigma_max_frac: float = 0.28
    sigma_ou_theta: float = 0.06
    sigma_ou_noise: float = 0.02
    # kinematics
    v_min: float = 0.25
    v_max: float = 0.85
    accel_sigma: float = 0.05
    p_turn: float = 0.03
    turn_sigma_deg: float = 35.0
    sigma_margin_mult: float = 0.2   # keep ~99% mass in view
    seed_offset: int = 79001

    def default_params(self) -> Dict[str, Any]:
        return {
            "sigma_min_frac": self.sigma_min_frac,
            "sigma_max_frac": self.sigma_max_frac,
            "sigma_ou_theta": self.sigma_ou_theta,
            "sigma_ou_noise": self.sigma_ou_noise,
            "v_min": self.v_min, "v_max": self.v_max,
            "accel_sigma": self.accel_sigma,
            "p_turn": self.p_turn, "turn_sigma_deg": self.turn_sigma_deg,
            "sigma_margin_mult": self.sigma_margin_mult,
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        C = np.empty((T, nx, ny), dtype=np.float32)

        # width bounds (pixels), clamped so k*sigma <= ~0.49*(n-1)
        dmin = float(min(nx, ny))
        smin0 = max(0.5, self.sigma_min_frac * dmin)
        smax0 = max(smin0 + 1e-6, self.sigma_max_frac * dmin)
        max_sigma_by_margin = 0.49 * (min(nx, ny) - 1.0) / max(self.sigma_margin_mult, 1e-6)
        smax = min(smax0, max_sigma_by_margin)
        smin = min(smin0, smax - 1e-6)
        if not (np.isfinite(smin) and np.isfinite(smax) and smin > 0 and smin < smax):
            smin, smax = 0.5, max(1.0, 0.25 * min(nx, ny))
        s_mu = 0.5 * (smin + smax)
        s = float(rng.uniform(smin, smax))

        # initial center with clamped margin
        margin0 = float(self.sigma_margin_mult * s)
        margin0 = float(min(margin0, 0.49 * (nx - 1.0), 0.49 * (ny - 1.0)))
        cx_low, cx_high = margin0, (nx - 1.0) - margin0
        cy_low, cy_high = margin0, (ny - 1.0) - margin0
        if not (cx_low < cx_high and cy_low < cy_high):
            raise ValueError(f"Grid too small for Gaussian with margins: nx={nx}, ny={ny}, margin={margin0:.3f}")
        cx = float(rng.uniform(cx_low, cx_high))
        cy = float(rng.uniform(cy_low, cy_high))

        b = _Bouncer(nx, ny, self.v_min, self.v_max, self.accel_sigma,
                     self.p_turn, self.turn_sigma_deg, seed=seed + 13)

        for t in range(T):
            r2 = (X - cx)**2 + (Y - cy)**2
            frame = np.exp(-0.5 * r2 / (s * s)).astype(np.float32)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)

            # OU update of width (clamped)
            s = s + self.sigma_ou_theta * (s_mu - s) + self.sigma_ou_noise * float(rng.normal())
            s = float(np.clip(s, smin, smax))

            # move & bounce with clamped margin
            margin = float(self.sigma_margin_mult * s)
            margin = float(min(margin, 0.49 * (nx - 1.0), 0.49 * (ny - 1.0)))
            cx, cy = b.step(cx, cy, margin, margin)

        return C

register_generator(DVDBounceGaussianGen())

# ---------- Smooth DVD Gaussian with random split/merge (torus-aware, cross-faded) ----------
@dataclass(slots=True)
class DVDSplitMergeGaussianSmoothGen:
    """
    One bouncing Gaussian that can split into two and later merge when they touch.
    - Periodic rendering; L1 mass is conserved exactly.
    - Merges use torus-aware (minimum-image) mass-weighted means.
    - Split/Merge are cross-faded over a few frames to avoid visual jumps.
    """
    name: str = "dvd_gaussian_split_merge_smooth"

    # Size / mass
    sigma_frac: float = 0.20          # initial σ as fraction of min(nx, ny)
    split_sigma_scale: float = 2**-0.5
    sigma_min: float = 0.8
    sigma_max_frac: float = 0.35
    sigma_margin_mult: float = 0.1
    total_mass: float = 1.0

    # Kinematics (DVD bounce + mild jitter)
    v_min: float = 0.25
    v_max: float = 0.85
    accel_sigma: float = 0.04         # small per-step velocity jitter
    p_turn: float = 0.02
    turn_sigma_deg: float = 25.0

    # Split / merge logic
    p_split_per_frame: float = 0.01
    min_frames_between_splits: int = 30
    split_kick: float = 0.30
    merge_alpha: float = 1.20         # contact threshold multiplier
    blend_frames_split: int = 4       # frames to cross-fade single -> double
    blend_frames_merge: int = 4       # frames to cross-fade double -> single

    seed_offset: int = 91001

    # ----------------- helpers -----------------
    @staticmethod
    def _min_image(dx: np.ndarray, L: float) -> np.ndarray:
        return dx - np.round(dx / L) * L

    def _bounce(self, c, v, nx, ny, margin_x, margin_y, rng):
        # jitter + occasional turn
        v = v + rng.normal(0.0, self.accel_sigma, size=2)
        if rng.random() < self.p_turn:
            ang = np.deg2rad(self.turn_sigma_deg) * rng.normal(0.0, 1.0)
            cang, sang = np.cos(ang), np.sin(ang)
            vx, vy = v[0], v[1]
            v = np.array([vx*cang - vy*sang, vx*sang + vy*cang], dtype=np.float64)

        # clamp speed
        sp = float(np.hypot(v[0], v[1]))
        if sp < 1e-8:
            th = rng.uniform(0, 2*np.pi); v[:] = self.v_min * np.array([np.cos(th), np.sin(th)])
        elif sp < self.v_min:
            v *= (self.v_min / sp)
        elif sp > self.v_max:
            v *= (self.v_max / sp)

        # advance + specular reflect against inner box [mx..nx-1-mx]×[my..ny-1-my]
        c = c + v
        mx = float(min(margin_x, 0.49*(nx - 1.0)))
        my = float(min(margin_y, 0.49*(ny - 1.0)))
        x_lo, x_hi = mx, (nx - 1.0) - mx
        y_lo, y_hi = my, (ny - 1.0) - my

        if c[0] < x_lo:
            c[0] = x_lo + (x_lo - c[0]); v[0] = -v[0]
        elif c[0] > x_hi:
            c[0] = x_hi - (c[0] - x_hi); v[0] = -v[0]
        if c[1] < y_lo:
            c[1] = y_lo + (y_lo - c[1]); v[1] = -v[1]
        elif c[1] > y_hi:
            c[1] = y_hi - (c[1] - y_hi); v[1] = -v[1]

        return c, v

    @staticmethod
    def _render_gaussian_periodic(X, Y, cx, cy, sigma, mass, nx, ny):
        two_pi = 2.0 * np.pi
        dx = X - cx; dx = dx - np.round(dx / nx) * nx
        dy = Y - cy; dy = dy - np.round(dy / ny) * ny
        r2 = dx*dx + dy*dy
        return (mass / (two_pi * sigma * sigma)) * np.exp(-0.5 * r2 / (sigma*sigma))

    def _moment_match_single(self, c1, s1, m1, c2, s2, m2, nx, ny):
        """
        Given two blobs (centers c1,c2 in R^2, sigmas s1,s2, masses m1,m2),
        compute torus-aware mass-weighted mean μ and variance σ^2 of the mixture.
        """
        # unwrap c2 relative to c1
        dx = self._min_image(c2[0] - c1[0], nx); dy = self._min_image(c2[1] - c1[1], ny)
        c2u = np.array([c1[0] + dx, c1[1] + dy], dtype=np.float64)

        M = m1 + m2
        mu_u = (m1*c1 + m2*c2u) / M                          # unwrapped mean near c1
        # centered offsets
        d1 = c1 - mu_u
        d2 = c2u - mu_u
        var = (m1*(s1*s1 + d1[0]*d1[0] + d1[1]*d1[1]) +
               m2*(s2*s2 + d2[0]*d2[0] + d2[1]*d2[1])) / M
        sigma = float(np.sqrt(max(self.sigma_min*self.sigma_min, var)))
        # wrap mean into the domain
        mu = np.array([mu_u[0] % nx, mu_u[1] % ny], dtype=np.float64)
        return mu, sigma

    # ----------------- main -----------------
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T
        dmin = float(min(nx, ny))

        # initial σ and caps
        sigma0 = float(min(max(self.sigma_min, self.sigma_frac * dmin), self.sigma_max_frac * dmin))

        # initial state: single blob
        margin0 = float(min(self.sigma_margin_mult * sigma0, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
        if not (margin0 < (nx - 1.0) - margin0 and margin0 < (ny - 1.0) - margin0):
            raise ValueError(f"Grid too small for sigma={sigma0:.2f} and margins.")

        c = np.array([rng.uniform(margin0, (nx - 1.0) - margin0),
                      rng.uniform(margin0, (ny - 1.0) - margin0)], dtype=np.float64)
        th = rng.uniform(0, 2*np.pi)
        v = np.array([self.v_min*np.cos(th), self.v_min*np.sin(th)], dtype=np.float64)
        sigma = sigma0
        mass = float(self.total_mass)

        # children (when split)
        c1 = np.zeros(2, dtype=np.float64); v1 = np.zeros(2, dtype=np.float64); s1 = 0.0; m1 = 0.0
        c2 = np.zeros(2, dtype=np.float64); v2 = np.zeros(2, dtype=np.float64); s2 = 0.0; m2 = 0.0

        # mode & blending
        mode = "single"                  # "single" | "double" | "blend_split" | "blend_merge"
        alpha = 0.0                      # blend weight for double (0=single, 1=double)
        split_cooldown = 0

        # grid for rendering
        X, Y = np.meshgrid(np.arange(nx, dtype=np.float64),
                           np.arange(ny, dtype=np.float64), indexing="ij")
        C = np.empty((T, nx, ny), dtype=np.float32)
        sigma_cap = self.sigma_max_frac * dmin

        for t in range(T):
            # ---------- render ----------
            if mode in ("single", "blend_merge"):
                single_field = self._render_gaussian_periodic(X, Y, c[0], c[1], sigma, mass, nx, ny)
            else:
                # single surrogate from the two children (moment-matched) for smooth split
                mu_s, sig_s = self._moment_match_single(c1, s1, m1, c2, s2, m2, nx, ny)
                single_field = self._render_gaussian_periodic(X, Y, mu_s[0], mu_s[1], sig_s, m1+m2, nx, ny)

            if mode in ("double", "blend_split", "blend_merge"):
                f1 = self._render_gaussian_periodic(X, Y, c1[0], c1[1], s1, m1, nx, ny)
                f2 = self._render_gaussian_periodic(X, Y, c2[0], c2[1], s2, m2, nx, ny)
                double_field = f1 + f2
            else:
                double_field = np.zeros_like(single_field)

            if mode == "single":
                field = single_field
            elif mode == "double":
                field = double_field
            elif mode == "blend_split":
                field = (1.0 - alpha) * single_field + alpha * double_field
            else:  # "blend_merge"
                field = (1.0 - alpha) * single_field + alpha * double_field

            C[t] = (opts.C0 * (1.0 + opts.A * field.astype(np.float32))).astype(np.float32)

            # ---------- dynamics ----------
            if mode == "single":
                # maybe split?
                if split_cooldown <= 0 and (rng.random() < self.p_split_per_frame):
                    # create children along current heading (small random yaw)
                    yaw = rng.normal(0.0, np.deg2rad(10.0))
                    dirv = np.array([np.cos(np.arctan2(v[1], v[0]) + yaw),
                                     np.sin(np.arctan2(v[1], v[0]) + yaw)], dtype=np.float64)
                    child_sigma = float(min(max(self.split_sigma_scale * sigma, self.sigma_min), sigma_cap))
                    child_mass  = 0.5 * mass
                    sep = 1.2 * child_sigma

                    c1 = (c + 0.5*sep*dirv) % [nx, ny]
                    c2 = (c - 0.5*sep*dirv) % [nx, ny]
                    v1 = v + self.split_kick * dirv
                    v2 = v - self.split_kick * dirv
                    s1 = child_sigma; s2 = child_sigma
                    m1 = child_mass;  m2 = child_mass

                    mode = "blend_split"; alpha = 0.0
                    split_cooldown = self.min_frames_between_splits
                else:
                    # continue single bounce
                    margin = float(min(self.sigma_margin_mult * sigma, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
                    c, v = self._bounce(c, v, nx, ny, margin, margin, rng)
                    split_cooldown = max(0, split_cooldown - 1)

            elif mode == "blend_split":
                # children move; single surrogate tracks their current COM (via rendering step above)
                m1 = max(1e-12, m1); m2 = max(1e-12, m2)
                msum = m1 + m2
                # bounce children
                m1g = float(min(self.sigma_margin_mult * s1, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
                m2g = float(min(self.sigma_margin_mult * s2, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
                c1, v1 = self._bounce(c1, v1, nx, ny, m1g, m1g, rng)
                c2, v2 = self._bounce(c2, v2, nx, ny, m2g, m2g, rng)
                # advance blend
                alpha += 1.0 / max(1, self.blend_frames_split)
                if alpha >= 1.0:
                    mode = "double"; alpha = 1.0

            elif mode == "double":
                # bounce both; check merge
                m1g = float(min(self.sigma_margin_mult * s1, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
                m2g = float(min(self.sigma_margin_mult * s2, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
                c1, v1 = self._bounce(c1, v1, nx, ny, m1g, m1g, rng)
                c2, v2 = self._bounce(c2, v2, nx, ny, m2g, m2g, rng)

                # contact test (periodic)
                dx = self._min_image(c1[0] - c2[0], nx); dy = self._min_image(c1[1] - c2[1], ny)
                dist = float(np.hypot(dx, dy))
                touch = self.merge_alpha * np.sqrt(s1*s1 + s2*s2)
                if dist <= touch:
                    # compute merged single state (torus-aware)
                    mu, sig_m = self._moment_match_single(c1, s1, m1, c2, s2, m2, nx, ny)
                    M = m1 + m2
                    # unwrap c2 to c1 to compute momentum properly
                    dxu = self._min_image(c2[0] - c1[0], nx); dyu = self._min_image(c2[1] - c1[1], ny)
                    c2u = np.array([c1[0] + dxu, c1[1] + dyu])
                    # velocities are Euclidean (no wrap), so momentum add is direct
                    v_m = (m1*v1 + m2*v2) / M
                    c = mu.copy(); v = v_m.copy(); sigma = float(min(sig_m, sigma_cap)); mass = M
                    mode = "blend_merge"; alpha = 1.0  # start fading double -> single

            else:  # "blend_merge"
                # children continue moving a bit while we fade out
                m1g = float(min(self.sigma_margin_mult * s1, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
                m2g = float(min(self.sigma_margin_mult * s2, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
                c1, v1 = self._bounce(c1, v1, nx, ny, m1g, m1g, rng)
                c2, v2 = self._bounce(c2, v2, nx, ny, m2g, m2g, rng)
                # fade parameter
                alpha -= 1.0 / max(1, self.blend_frames_merge)
                if alpha <= 0.0:
                    mode = "single"; alpha = 0.0
                    # enforce cooldown after recombination
                    split_cooldown = self.min_frames_between_splits

        return C

# Register
register_generator(DVDSplitMergeGaussianSmoothGen())



# ---------- Ring/annulus; radius & thickness vary (OU); accelerates/turns; margin-safe ----------
@dataclass(slots=True)
class DVDBounceRingGen:
    name: str = "dvd_bounce_ring"
    # mean radius & thickness as fractions of min(nx,ny)
    r0_min_frac: float = 0.05
    r0_max_frac: float = 0.8
    sigma_r_min_frac: float = 0.04
    sigma_r_max_frac: float = 0.8
    r0_ou_theta: float = 0.05
    r0_ou_noise: float = 0.02
    sr_ou_theta: float = 0.08
    sr_ou_noise: float = 0.02
    # kinematics
    v_min: float = 0.20
    v_max: float = 0.70
    accel_sigma: float = 0.04
    p_turn: float = 0.025
    turn_sigma_deg: float = 30.0
    # bounce margin multiplier (outer radius approx r0 + k*sigma_r)
    margin_k: float = 0.01
    seed_offset: int = 79011

    def default_params(self) -> Dict[str, Any]:
        return {
            "r0_min_frac": self.r0_min_frac, "r0_max_frac": self.r0_max_frac,
            "sigma_r_min_frac": self.sigma_r_min_frac, "sigma_r_max_frac": self.sigma_r_max_frac,
            "r0_ou_theta": self.r0_ou_theta, "r0_ou_noise": self.r0_ou_noise,
            "sr_ou_theta": self.sr_ou_theta, "sr_ou_noise": self.sr_ou_noise,
            "v_min": self.v_min, "v_max": self.v_max,
            "accel_sigma": self.accel_sigma, "p_turn": self.p_turn,
            "turn_sigma_deg": self.turn_sigma_deg, "margin_k": self.margin_k,
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        C = np.empty((T, nx, ny), dtype=np.float32)

        dmin = float(min(nx, ny))
        r0min = self.r0_min_frac * dmin
        r0max = self.r0_max_frac * dmin
        srmin = self.sigma_r_min_frac * dmin
        srmax = self.sigma_r_max_frac * dmin

        r0_mu = 0.5 * (r0min + r0max)
        sr_mu = 0.5 * (srmin + srmax)
        r0 = float(rng.uniform(r0min, r0max))
        sr = float(rng.uniform(srmin, srmax))

        outer = float(max(1.0, r0 + self.margin_k * sr))
        max_marg_x = 0.49 * (nx - 1.0)
        max_marg_y = 0.49 * (ny - 1.0)
        outer_x = float(min(outer, max_marg_x))
        outer_y = float(min(outer, max_marg_y))
        if not (outer_x < (nx - 1.0) - outer_x and outer_y < (ny - 1.0) - outer_y):
            raise ValueError(f"Grid too small for ring with margins: nx={nx}, ny={ny}, outer≈{outer:.3f}")
        cx = float(rng.uniform(outer_x, (nx - 1.0) - outer_x))
        cy = float(rng.uniform(outer_y, (ny - 1.0) - outer_y))

        b = _Bouncer(nx, ny, self.v_min, self.v_max, self.accel_sigma,
                     self.p_turn, self.turn_sigma_deg, seed=seed + 17)

        for t in range(T):
            r = np.sqrt((X - cx)**2 + (Y - cy)**2)
            frame = np.exp(-0.5 * ((r - r0) / (sr + 1e-8))**2).astype(np.float32)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)

            # OU updates (clamped)
            r0 = r0 + self.r0_ou_theta * (r0_mu - r0) + self.r0_ou_noise * float(rng.normal())
            sr = sr + self.sr_ou_theta * (sr_mu - sr) + self.sr_ou_noise * float(rng.normal())
            r0 = float(np.clip(r0, r0min, r0max))
            sr = float(np.clip(sr, srmin, srmax))

            margin = float(r0 + self.margin_k * sr)
            margin_x = float(min(margin, 0.49 * (nx - 1.0)))
            margin_y = float(min(margin, 0.49 * (ny - 1.0)))
            cx, cy = b.step(cx, cy, margin_x, margin_y)

        return C

register_generator(DVDBounceRingGen())


# ---------- Soft-edged rectangle; size varies (OU); accelerates/turns; margin-safe ----------
@dataclass(slots=True)
class DVDBounceBoxGen:
    name: str = "dvd_bounce_box"
    # size fracs of domain
    w_min_frac: float = 0.18
    w_max_frac: float = 0.35
    h_min_frac: float = 0.12
    h_max_frac: float = 0.28
    size_ou_theta: float = 0.07
    size_ou_noise: float = 0.025
    edge_soft_frac: float = 0.02   # softness as fraction of min(nx,ny) for logistic edge
    # kinematics
    v_min: float = 0.22
    v_max: float = 0.80
    accel_sigma: float = 0.05
    p_turn: float = 0.04
    turn_sigma_deg: float = 40.0
    seed_offset: int = 79021

    def default_params(self) -> Dict[str, Any]:
        return {
            "w_min_frac": self.w_min_frac, "w_max_frac": self.w_max_frac,
            "h_min_frac": self.h_min_frac, "h_max_frac": self.h_max_frac,
            "size_ou_theta": self.size_ou_theta, "size_ou_noise": self.size_ou_noise,
            "edge_soft_frac": self.edge_soft_frac,
            "v_min": self.v_min, "v_max": self.v_max, "accel_sigma": self.accel_sigma,
            "p_turn": self.p_turn, "turn_sigma_deg": self.turn_sigma_deg,
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        C = np.empty((T, nx, ny), dtype=np.float32)

        dmin = float(min(nx, ny))
        wmin, wmax = self.w_min_frac * nx, self.w_max_frac * nx
        hmin, hmax = self.h_min_frac * ny, self.h_max_frac * ny
        w_mu, h_mu = 0.5*(wmin+wmax), 0.5*(hmin+hmax)
        w = float(rng.uniform(wmin, wmax))
        h = float(rng.uniform(hmin, hmax))
        soft = max(1.0, self.edge_soft_frac * dmin)  # logistic edge softness (pixels)

        # initial center with clamped margins (half-sizes)
        half_w = float(min(w/2.0, 0.49 * (nx - 1.0)))
        half_h = float(min(h/2.0, 0.49 * (ny - 1.0)))
        if not (half_w < (nx - 1.0) - half_w and half_h < (ny - 1.0) - half_h):
            raise ValueError(f"Grid too small for box: nx={nx}, ny={ny}, w={w:.2f}, h={h:.2f}")
        cx = float(rng.uniform(half_w, (nx - 1.0) - half_w))
        cy = float(rng.uniform(half_h, (ny - 1.0) - half_h))

        b = _Bouncer(nx, ny, self.v_min, self.v_max, self.accel_sigma,
                     self.p_turn, self.turn_sigma_deg, seed=seed + 23)

        def sigmoid(z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

        for t in range(T):
            # Separable soft rectangle via logistic edges
            plate_x = sigmoid((X - (cx - w/2)) / soft) * sigmoid(((cx + w/2) - X) / soft)
            plate_y = sigmoid((Y - (cy - h/2)) / soft) * sigmoid(((cy + h/2) - Y) / soft)
            frame = (plate_x * plate_y).astype(np.float32)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)

            # OU update sizes (clamped)
            w = w + self.size_ou_theta * (w_mu - w) + self.size_ou_noise * float(rng.normal())
            h = h + self.size_ou_theta * (h_mu - h) + self.size_ou_noise * float(rng.normal())
            w = float(np.clip(w, wmin, min(wmax, nx - 2.0)))
            h = float(np.clip(h, hmin, min(hmax, ny - 2.0)))

            # move & bounce with clamped half-size margins
            half_w = float(min(w/2.0, 0.49 * (nx - 1.0)))
            half_h = float(min(h/2.0, 0.49 * (ny - 1.0)))
            cx, cy = b.step(cx, cy, half_w, half_h)

        return C

register_generator(DVDBounceBoxGen())


@dataclass(slots=True)
class Recipe:
    name: str
    make_generator: Callable[[], Generator]
    make_pipeline: Callable[[], Callable[[np.ndarray], Tuple[np.ndarray, List[Dict[str, Any]]]]]
    level: int = 1

def default_dvd_suite_recipes() -> List[Recipe]:
    return [
        Recipe(
            name="dvd_gaussian_accel",
            make_generator=lambda: DVDBounceGaussianGen(
                sigma_min_frac=0.14, sigma_max_frac=0.28,
                sigma_ou_theta=0.06, sigma_ou_noise=0.025,
                v_min=0.25, v_max=0.85, accel_sigma=0.05, p_turn=0.035, turn_sigma_deg=35.0
            ),
            make_pipeline=lambda: pipeline([
                
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
        Recipe(
            name="dvd_ring_accel",
            make_generator=lambda: DVDBounceRingGen(
                r0_min_frac=0.20, r0_max_frac=0.30,
                sigma_r_min_frac=0.045, sigma_r_max_frac=0.08,
                r0_ou_theta=0.05, r0_ou_noise=0.02,
                sr_ou_theta=0.08, sr_ou_noise=0.02,
                v_min=0.22, v_max=0.75, accel_sigma=0.045, p_turn=0.03, turn_sigma_deg=30.0,
                margin_k=3.0
            ),
            make_pipeline=lambda: pipeline([
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
        Recipe(
            name="dvd_box_accel",
            make_generator=lambda: DVDBounceBoxGen(
                w_min_frac=0.20, w_max_frac=0.36,
                h_min_frac=0.14, h_max_frac=0.30,
                size_ou_theta=0.07, size_ou_noise=0.025,
                edge_soft_frac=0.02,
                v_min=0.24, v_max=0.80, accel_sigma=0.05, p_turn=0.04, turn_sigma_deg=40.0
            ),
            make_pipeline=lambda: pipeline([
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
    ]
@dataclass(slots=True)
class StraightBounceDotGen:
    name: str = "straight_bounce_dot"
    sigma: float = 2.0            # fraction (<=1) of min(nx,ny) OR legacy pixels (>1)
    speed: float = 0.45           # pixels per frame
    sigma_margin_mult: float = 0.2
    seed_offset: int = 56001

    def default_params(self) -> Dict[str, Any]:
        mode = "fraction" if self.sigma <= 1.0 else "pixels_legacy"
        return {
            "sigma": self.sigma,
            "sigma_mode": mode,
            "speed": self.speed,
            "sigma_margin_mult": self.sigma_margin_mult
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

        # Resolve sigma (width in pixels)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.sigma <= 1.0:
            s = float(self.sigma * min(nx, ny))
        else:  # legacy behavior (value given in pixels)
            s = float(self.sigma)

        s = float(max(0.8, s))
        margin = float(min(self.sigma_margin_mult * s, 0.49*(nx - 1.0), 0.49*(ny - 1.0)))
        if not (margin < (nx - 1.0) - margin and margin < (ny - 1.0) - margin):
            raise ValueError(f"Grid too small for sigma/margin: nx={nx}, ny={ny}, sigma_px={s:.3f}")

        # init center and direction
        theta = float(rng.uniform(0.0, 2*np.pi))
        vx = self.speed * np.cos(theta)
        vy = self.speed * np.sin(theta)
        cx = float(rng.uniform(margin, (nx - 1.0) - margin))
        cy = float(rng.uniform(margin, (ny - 1.0) - margin))

        two_s2 = 2.0 * s * s
        C = np.empty((T, nx, ny), dtype=np.float32)
        for t in range(T):
            r2 = (X - cx)**2 + (Y - cy)**2
            frame = np.exp(-r2 / two_s2).astype(np.float32)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)

            cx += vx; cy += vy
            x_lo, x_hi = margin, (nx - 1.0) - margin
            y_lo, y_hi = margin, (ny - 1.0) - margin
            if cx < x_lo: cx = x_lo + (x_lo - cx); vx = -vx
            elif cx > x_hi: cx = x_hi - (cx - x_hi); vx = -vx
            if cy < y_lo: cy = y_lo + (y_lo - cy); vy = -vy
            elif cy > y_hi: cy = y_hi - (cy - y_hi); vy = -vy

        return C

register_generator(StraightBounceDotGen())
# ---------- Circular orbits (one or more objects) ----------
@dataclass(slots=True)
class CircularOrbitGen:
    name: str = "circular_orbit"
    n_objects: int = 1
    radius_frac: float = 0.08        # fraction of min(nx, ny) (0..0.5 typical)
    omega: float = 2*np.pi/160.0     # radians/frame
    width: float = 0.08              # Gaussian width (fraction of min(nx, ny))
    center_jitter: float = 0.0       # optional jitter of orbit center (pixels)
    seed_offset: int = 56011

    def default_params(self) -> Dict[str, Any]:
        return {
            "n_objects": self.n_objects,
            "radius_frac": self.radius_frac,
            "omega": self.omega,
            "width": self.width,
            "center_jitter": self.center_jitter
        }

    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed + self.seed_offset)
        nx, ny, T = spec.nx, spec.ny, spec.T
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        cx0, cy0 = (nx - 1)/2.0, (ny - 1)/2.0

        # Orbit radius in pixels (fraction of smaller dimension)
        radius_px = float(self.radius_frac * min(nx, ny))

        phases = rng.uniform(0, 2*np.pi, size=self.n_objects)
        if self.center_jitter > 0:
            centers = np.stack([
                cx0 + rng.normal(0, self.center_jitter, size=self.n_objects),
                cy0 + rng.normal(0, self.center_jitter, size=self.n_objects),
            ], axis=1)
        else:
            centers = np.tile(np.array([cx0, cy0], dtype=np.float64), (self.n_objects, 1))

        width = float(self.width * min(nx, ny))
        two_s2 = 2.0 * width * width

        C = np.empty((T, nx, ny), dtype=np.float32)
        for t in range(T):
            frame = np.zeros((nx, ny), dtype=np.float32)
            ang = self.omega * t
            for k in range(self.n_objects):
                cx, cy = centers[k]
                xk = cx + radius_px * np.cos(ang + phases[k])
                yk = cy + radius_px * np.sin(ang + phases[k])
                r2 = (X - xk)**2 + (Y - yk)**2
                frame += np.exp(-r2 / two_s2).astype(np.float32)
            C[t] = (opts.C0 * (1.0 + opts.A * frame)).astype(np.float32)
        return C

register_generator(CircularOrbitGen())


def default_curriculum_recipes() -> List[Recipe]:
    return [
        Recipe(
            level=1,
            name="bounding_gaussian",
            make_generator=lambda: DVDBounceGaussianGen(sigma_min_frac=0.14, sigma_max_frac=0.28,
                                                        sigma_ou_theta=0.08, sigma_ou_noise=0.025,
                                                        v_min=0.25, v_max=2, accel_sigma=0.07, p_turn=0.035, turn_sigma_deg=35.0, sigma_margin_mult=0.02),
            make_pipeline=lambda: pipeline([
                Tag({"difficulty": 1, "task": "tracking"}),
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
        Recipe(
            level=1,
            name="slow_birth_decay",
            make_generator=lambda: BirthDeathGradualGen(
                lambda_birth=0.06,             # slower arrivals
                lifetime_range=(60, 180),      # long-lived
                growth_time=20, decay_time=28, # slow ramps
                r_init=0.01, r_max=0.3,        # broader Gaussians
                max_blobs=72
            ),
            make_pipeline=lambda: pipeline([
                Tag({"difficulty": 1, "task": "detection"}),
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
        Recipe(
            level=2,
            name="bounding_gaussian_noise",
            make_generator=lambda: DVDBounceGaussianGen(sigma_min_frac=0.14, sigma_max_frac=0.6,
                                                        sigma_ou_theta=0.08, sigma_ou_noise=0.1,
                                                        v_min=0.25, v_max=0.85, accel_sigma=0.05, p_turn=0.035, turn_sigma_deg=35.0, sigma_margin_mult=0.02),
            make_pipeline=lambda: pipeline([
                Tag({"difficulty": 2, "task": "tracking"}),
                TurbulentFlicker(sigma=0.18, ell=1.2, tau=1.4, seed=20240819),
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
         Recipe(
            level=2,
            name="bounding_ring_noise",
            make_generator=lambda: DVDBounceRingGen(r0_min_frac=0.02, r0_max_frac=0.30),
            make_pipeline=lambda: pipeline([
                Tag({"difficulty": 1, "task": "tracking"}),
                TurbulentFlicker(sigma=0.18, ell=1.2, tau=1.4, seed=20240819),
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
        Recipe(
            level=2,
            name="slow_birth_decay_noise",
            make_generator=lambda: BirthDeathGradualGen(
                lambda_birth=0.06,             # slower arrivals
                lifetime_range=(60, 180),      # long-lived
                growth_time=20, decay_time=28, # slow ramps
                r_init=0.01, r_max=0.3,        # broader Gaussians
                max_blobs=72
            ),
            make_pipeline=lambda: pipeline([
                Tag({"difficulty": 2, "task": "detection"}),
                TurbulentFlicker(sigma=0.18, ell=1.2, tau=1.4, seed=1234),
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
        Recipe(
            level=4,
            name="combined",
            make_generator=lambda: WeightedSumGen(
                name="mix",
                parts=[(
                    BirthDeathGradualGen(
                        lambda_birth=0.06,             # slower arrivals
                        lifetime_range=(20, 180),      # long-lived
                        growth_time=20, decay_time=28, # slow ramps
                        r_init=0.01, r_max=0.3,        # broader Gaussians
                        max_blobs=72
                    ), 0.9),
                    (
                        DVDBounceGaussianGen(sigma_min_frac=0.14, sigma_max_frac=0.28,
                                                        sigma_ou_theta=0.2, sigma_ou_noise=0.025,
                                                        v_min=0.25, v_max=2, accel_sigma=0.11, p_turn=0.05, turn_sigma_deg=35.0, sigma_margin_mult=0.02)
                    , 0.8)]),
            make_pipeline=lambda: pipeline([
                Tag({"difficulty": 4, "task": "tracking"}),
                TurbulentFlicker(sigma=0.18, ell=1.2, tau=1.4, seed=20240819),
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
    ]

def build_bundle(
    recipes: List[Recipe],
    out_root: str,
    bundle_tag: str,
    seeds: Iterable[int],
    spec: GridSpec,
    opts: GenOpts,
    *,
    select_levels: Optional[Iterable[int]] = None,  # choose subset of difficulty levels
    fps: int = 12,
    compress: str = "zip",
    overwrite: bool = False,
    dtype: str = "float32",
) -> Tuple[Path, Path]:
    """
    Build a curriculum bundle of workloads across difficulty levels and seeds.
    Item tags look like: lvl_<L>__<name>__seed_<s>
    """
    if select_levels is not None:
        wanted = set(int(x) for x in select_levels)
        recipes = [r for r in recipes if r.level in wanted]
        if not recipes:
            raise ValueError(f"No matching levels in select_levels={list(select_levels)}")

    # Ensure deterministic ordering by level then name
    recipes = sorted(recipes, key=lambda r: (r.level, r.name))

    items: List[Dict[str, Any]] = []
    for s in seeds:
        spec_local = GridSpec(nx=spec.nx, ny=spec.ny, T=spec.T, seed=int(s))
        for r in recipes:
            gen = r.make_generator()
            proc = r.make_pipeline()
            C_raw = gen(spec_local, opts, seed=int(s))
            C_out, tmeta = proc(C_raw)

            tag = f"lvl_{r.level:02d}__{r.name}__seed_{int(s)}"
            add_item(
                items,
                tag=tag,
                C=C_out.astype(dtype, copy=False),
                generator=getattr(gen, "name", "callable"),
                generator_params={**getattr(gen, "default_params", lambda: {})(), "difficulty_level": r.level},
                transforms=tmeta,
                dtype=dtype,
            )

    return save_workload_bundle(
        out_root=out_root,
        bundle_tag=bundle_tag,
        items=items,
        fps=fps,
        compress=compress,
        overwrite=overwrite,
    )


@dataclass(slots=True)
class WeightedSumGen:
    """Sum of sub-generators with given weights. Uses split_seed to give each sub-gen a stable sub-seed."""
    name: str
    parts: List[Tuple[Generator, float]]
    def default_params(self) -> Dict[str, Any]:
        return {
            "parts": [{"name": g.name, "weight": float(w), "params": g.default_params()} for (g, w) in self.parts]
        }
    def __call__(self, spec: GridSpec, opts: GenOpts, seed: int) -> np.ndarray:
        sub_seeds = split_seed(seed, len(self.parts))
        Csum = None
        for (i, (g, w)) in enumerate(self.parts):
            Cg = g(spec, opts, sub_seeds[i])
            Csum = (w * Cg) if Csum is None else (Csum + w * Cg)
        return Csum.astype(np.float32)


def save_animation(C_series: np.ndarray, path: str, fps: int = 10):
    """Save a GIF with fixed vmin/vmax for consistent visualization."""
    _assert_3d(C_series)
    from matplotlib.animation import FuncAnimation, PillowWriter
    T, nx, ny = C_series.shape
    vmin, vmax = float(np.min(C_series)), float(np.max(C_series))
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(C_series[0], interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    def update(frame):
        im.set_data(C_series[frame])
        ax.set_title(f"t={frame}")
        return (im,)
    anim = FuncAnimation(fig, update, frames=T, interval=1000.0/fps, blit=True)
    writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer)
    plt.close(fig)


def add_item(
    items: list,
    tag: str,
    C: np.ndarray,
    generator: str,
    generator_params: Dict[str, Any],
    transforms: List[Dict[str, Any]],
    dtype: str = "float32",
):
    """Register one pattern item to be saved."""
    _assert_3d(C)
    tag_sanitized = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
    Cc = C.astype(dtype, copy=False)
    items.append({
        "tag": tag_sanitized,
        "C": Cc,
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generator": generator,
            "generator_params": generator_params,
            "transforms": transforms,
            "shape": tuple(Cc.shape),
            "dtype": str(Cc.dtype),
            "min": float(np.min(Cc)),
            "max": float(np.max(Cc)),
            "sha256": _sha256_of_array(Cc),
        }
    })

def save_workload_bundle(
    out_root: str,
    bundle_tag: str,
    items: list,
    fps: int = 10,
    compress: str = "zip",   # "zip" | "tar.gz"
    overwrite: bool = False
) -> Tuple[Path, Path]:
    """Save items into a folder with NPZ+GIF+meta, then compress the folder."""
    assert compress in ("zip", "tar.gz")
    out_root = Path(out_root); _ensure_dir(out_root)

    ts = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"workload_{bundle_tag}_{ts}"
    bundle_dir = out_root / folder_name
    if bundle_dir.exists():
        if overwrite:
            shutil.rmtree(bundle_dir)
        else:
            raise FileExistsError(f"{bundle_dir} exists. Use overwrite=True to replace.")
    _ensure_dir(bundle_dir)

    index = {
        "schema_version": SCHEMA_VERSION,
        "bundle_tag": bundle_tag,
        "created_utc": ts,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "count": len(items),
        "items": []
    }

    for it in items:
        tag = it["tag"]
        pat_dir = bundle_dir / tag
        _ensure_dir(pat_dir)

        C = it["C"]
        # NPZ
        npz_path = pat_dir / "C.npz"
        np.savez_compressed(npz_path, C=C)

        # GIF
        gif_path = pat_dir / "preview.gif"
        save_animation(C, str(gif_path), fps=fps)

        # Meta
        meta = dict(it["meta"])
        meta.update({
            "tag": tag,
            "files": {"npz": npz_path.name, "gif": gif_path.name}
        })
        with open(pat_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        index["items"].append({
            "tag": tag,
            "shape": meta["shape"],
            "dtype": meta["dtype"],
            "min": meta["min"], "max": meta["max"],
            "sha256": meta["sha256"],
            "paths": {
                "dir": str(tag),
                "npz": f"{tag}/C.npz",
                "gif": f"{tag}/preview.gif",
                "meta": f"{tag}/meta.json"
            },
            "generator": meta.get("generator", "unknown"),
            "generator_params": meta.get("generator_params", {}),
            "transforms": meta.get("transforms", [])
        })

    with open(bundle_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    # Compress
    base = out_root / folder_name
    if compress == "zip":
        archive = shutil.make_archive(str(base), "zip", root_dir=bundle_dir)
    else:
        archive = shutil.make_archive(str(base), "gztar", root_dir=bundle_dir)

    return bundle_dir, Path(archive)


def build_bundle_from_callable(
    out_root: str,
    bundle_tag: str,
    seeds: Iterable[int],
    make_series: Callable[[int], Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]],
    *,
    tag_fn: Optional[Callable[[int], str]] = None,
    fps: int = 10,
    compress: str = "zip",
    overwrite: bool = False
) -> Tuple[Path, Path]:
    """
    Call `make_series(seed)` for each seed to produce:
        C, generator_meta, generator_params, transforms_meta
    - generator_meta: {"name": <str>}
    - generator_params: dict recorded in meta
    - transforms_meta: list of {"name":..., "params":...}
    """
    items = []
    for s in seeds:
        C, gen_meta, gen_params, tmeta = make_series(int(s))
        tag = tag_fn(s) if tag_fn else f"series_seed_{s}"
        add_item(items, tag=tag, C=C,
                 generator=gen_meta.get("name", "callable"),
                 generator_params=gen_params,
                 transforms=tmeta,
                 dtype=str(C.dtype))
    return save_workload_bundle(out_root, bundle_tag, items, fps=fps, compress=compress, overwrite=overwrite)


def _mk_traveling_wave(theta: float, wavelength: float, omega: float) -> TravelingWaveGen:
    """Construct a wave by direction (theta), wavelength, and temporal frequency omega."""
    k = 2*np.pi / wavelength
    return TravelingWaveGen(kx=k*np.cos(theta), ky=k*np.sin(theta), omega=omega, theta=0.0)

def default_prototypical_recipes() -> List[Recipe]:
    return [
        Recipe(
            name="big_moving_gaussian_realistic",
            make_generator=lambda: MovingGaussiansGen(
                n_blobs=1,
                # radius in pixels; pick ~18% of min(nx,ny) at runtime (set in wrapper below)
                radius=8.0,                   # placeholder; will be overridden in builder
                speed_range=(0.18, 0.36),     # pixels / frame
                wrap=True
            ),
            make_pipeline=lambda: pipeline([
                SpatialGaussian(sigma=1.0),
                MeanderingWindAdvection(ax=0.35, ay=0.25, wx=2*np.pi/220, wy=2*np.pi/180, phx=0.0, phy=np.pi/3),
                TurbulentFlicker(sigma=0.18, ell=1.2, tau=1.4, seed=20240819),
                TemporalHysteresis(rise=0.55, decay=0.06),
                MapToRange(low=0.2, high=1.0, mode="global"),
            ])
        ),
        Recipe(
            name="slow_birth_decay_realistic",
            make_generator=lambda: BirthDeathGradualGen(
                lambda_birth=0.06,             # slower arrivals
                lifetime_range=(60, 180),      # long-lived
                growth_time=20, decay_time=28, # slow ramps
                r_init=2.0, r_max=10.0,        # broader Gaussians
                max_blobs=72
            ),
            make_pipeline=lambda: pipeline([
                SpatialGaussian(sigma=0.8),
                TemporalGaussian(sigma_t=1.6),
                TemporalHysteresis(rise=0.45, decay=0.08),
                MapToRange(low=0.2, high=1.0, mode="per_frame"),
            ])
        ),
    ]

def _read_index(bundle_path: str) -> Tuple[Dict[str, Any], str]:
    """
    Reads index.json from a bundle directory or archive.
    Returns (index_dict, storage_kind) where storage_kind in {"dir","zip","tar"}.
    """
    p = Path(bundle_path)
    if p.is_dir():
        with open(p / "index.json", "r") as f:
            return json.load(f), "dir"
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            with zf.open("index.json") as f:
                return json.loads(f.read().decode("utf-8")), "zip"
    if p.suffixes[-2:] == [".tar", ".gz"] or p.suffix.lower() == ".tgz":
        with tarfile.open(p, "r:gz") as tf:
            member = tf.getmember("index.json")
            with tf.extractfile(member) as f:
                return json.loads(f.read().decode("utf-8")), "tar"
    raise FileNotFoundError(f"Could not find index.json in: {bundle_path}")

def _infer_difficulty(item: Dict[str, Any]) -> Optional[int]:
    """
    Try multiple sources to infer difficulty:
      1) Tag transform: {'name': 'tag', 'params': {'difficulty': <int>, ...}}
      2) generator_params['difficulty_level']
      3) tag/name prefix: lvl_XX__...
    """
    # 1) Tag transform
    for t in item.get("transforms", []):
        if t.get("name") == "tag":
            params = t.get("params", {})
            if "difficulty" in params:
                try:
                    return int(params["difficulty"])
                except Exception:
                    pass
    # 2) generator_params
    gp = item.get("generator_params", {})
    if "difficulty_level" in gp:
        try:
            return int(gp["difficulty_level"])
        except Exception:
            pass
    # 3) lvl_XX__ prefix
    tag = item.get("tag") or ""
    m = re.match(r"^lvl_(\d+)", tag)
    if m:
        return int(m.group(1))
    return None

def _open_binary(bundle_path: str, rel_path: str, storage_kind: str):
    """
    Returns a file-like object (BytesIO) containing the binary content for rel_path.
    """
    p = Path(bundle_path)
    if storage_kind == "dir":
        return open(p / rel_path, "rb")
    if storage_kind == "zip":
        zf = zipfile.ZipFile(p, "r")
        data = zf.read(rel_path)  # bytes
        zf.close()
        return io.BytesIO(data)
    if storage_kind == "tar":
        tf = tarfile.open(p, "r:gz")
        try:
            member = tf.getmember(rel_path)
        except KeyError:
            # Some tar implementations store paths without leading './'
            # Try a fallback search
            member = next(m for m in tf.getmembers() if m.name.endswith(rel_path))
        f = tf.extractfile(member)
        data = f.read()
        tf.close()
        return io.BytesIO(data)
    raise RuntimeError(f"Unknown storage_kind {storage_kind}")

def _read_text(bundle_path: str, rel_path: str, storage_kind: str) -> str:
    p = Path(bundle_path)
    if storage_kind == "dir":
        return (p / rel_path).read_text()
    bio = _open_binary(bundle_path, rel_path, storage_kind)
    return bio.read().decode("utf-8")

def load_random_series_by_difficulty(
    bundle_path: str,
    max_difficulty: int,
    *,
    rng_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, str]]:
    """
    Select a random item with difficulty <= max_difficulty and load its NPZ array 'C'.

    Args:
      bundle_path: Path to workload folder OR the .zip/.tar.gz archive.
      max_difficulty: upper bound (inclusive).
      rng_seed: optional seed for deterministic sampling.

    Returns:
      C: np.ndarray of shape (T, nx, ny)
      meta: dict parsed from that item's meta.json
      paths: dict with keys {'dir','npz','gif','meta'} for that item (relative to bundle root).
    """
    idx, storage_kind = _read_index(bundle_path)
    items: List[Dict[str, Any]] = idx.get("items", [])
    if not items:
        raise RuntimeError("Index has no items.")

    # Filter by difficulty (with robust inference)
    eligible = []
    for it in items:
        d = _infer_difficulty(it)
        if d is None:
            # If difficulty missing, exclude by default to keep curriculum clean.
            continue
        if d <= int(max_difficulty):
            eligible.append(it)

    if not eligible:
        raise ValueError(f"No items with difficulty <= {max_difficulty} found in bundle: {bundle_path}")

    # Deterministic pick if rng_seed provided
    rng = np.random.default_rng(rng_seed)
    choice = eligible[int(rng.integers(0, len(eligible)))]

    rel_npz = choice["paths"]["npz"]
    rel_meta = choice["paths"]["meta"]

    # Load NPZ->C
    with _open_binary(bundle_path, rel_npz, storage_kind) as fobj:
        with np.load(fobj) as npz:
            if "C" not in npz:
                raise KeyError(f"'C' not found in {rel_npz}")
            C = np.array(npz["C"], copy=False)

    # Load meta.json for full provenance
    meta_text = _read_text(bundle_path, rel_meta, storage_kind)
    meta = json.loads(meta_text)

    # Optional sanity checks (shape/dtype)
    if C.ndim != 3:
        raise ValueError(f"Expected C to be 3D (T,nx,ny), got shape {C.shape}")
    if not np.issubdtype(C.dtype, np.floating):
        raise TypeError(f"Expected floating dtype, got {C.dtype}")

    return C, meta, choice["paths"]


if __name__ == "__main__":
    spec = GridSpec(nx=256, ny=256, T=180, seed=0)
    opts = GenOpts(C0=1.0, A=1.0)
    seeds = range(10)

    out_dir = "./out"
    bundle_tag = "eval_test"

    bundle_dir, archive = build_bundle(
        recipes=default_curriculum_recipes(),
        out_root=out_dir,
        bundle_tag=bundle_tag,
        seeds=seeds,
        spec=spec,
        opts=opts,
        fps=12,
        compress="zip",
        overwrite=True,
    )
    print("Saved to:", bundle_dir, "Archive:", archive)

    # #Test random load 
    # C, meta, path = load_random_series_by_difficulty(
    #     bundle_path="./out/workload_curriculum_v1_20250820-141813",
    #     max_difficulty=2,
    # )

    # print(meta)
    # print("Loaded C from:", path)
    # print("C shape:", C.shape, "dtype:", C.dtype, "min:", np.min(C), "max:", np.max(C))

    # from task4feedback.graphs.dynamic_jacobi import DynamicJacobiGraph, DynamicJacobiConfig
    # from archived_workloads import ArchivedFineGridWorkload
    # from task4feedback.graphs.mesh import *
    # from task4feedback.graphs.base import *
    # from matplotlib.animation import FuncAnimation, PillowWriter
    # import time 


    # mesh = generate_quad_mesh(L=1, n=5)
    # geom = build_geometry(mesh)
    # config = DynamicJacobiConfig(
    #     n=4,
    # )
    
    # w = WorkloadInterpolator(geom)
    # wl_list = []

    # for i in range(C.shape[0]):
    #     time_s = time.perf_counter()
    #     wl = w.workload_to_cells(C[i])
    #     time_e = time.perf_counter()
    #     print(f"Time to convert workload to cells at frame {i}: {time_e - time_s:.6f} seconds")
    #     wl_list.append(wl)

    

# if __name__ == "__main__":
#     spec = GridSpec(nx=64, ny=64, T=320, seed=32)
#     opts = GenOpts(C0=1.0, A=1.0)

#     # Build a composite generator: traveling wave + flamelets + birth-death + ignition
#     front = ArrheniusTwoFieldGen()
#     flam  = ArrheniusTwoFieldGen()
#     bd    = BirthDeathGradualGen(lambda_birth=0.05, lifetime_range=(24, 200),
#                                   growth_time=7, decay_time=9, r_init=1.2, r_max=2.2)
#     ign   = IgnitionExtinctionGen(ignition_threshold=0.6, extinction_rate=0.04, noise_sigma=0.08)
#     #bd = DVDSplitMergeGaussianSmoothGen()
#     ign = ArrheniusTwoFieldGen()

#     combo = WeightedSumGen(
#         name="mix",
#         parts=[(front, 0.9), (flam, 0.8), (bd, 1.0), (ign, 0.6)]
#     )

#     # Transform pipeline: spatial -> temporal -> map to [low, high]
#     LOW, HIGH = 0.2, 1.0
#     proc = pipeline([
#         #SpatialGaussian(sigma=1.0),
#         #TemporalGaussian(sigma_t=2.0),
#         #TurbulentFlicker(),
#         MassNormalize(),
#         UnsharpMask(),
#         #ShearAdvection(shear=0.002, vx0=0.02, vy=0.18),  # gentle shear
#         #MapToRange(low=LOW, high=HIGH, mode="per_frame"),
#         NonLinearFrameMap(func=np.tanh, low=LOW, high=HIGH, clip=(0.0, 1.0)),
#     ])

#     # Generate a single series with the composite generator
#     C_raw = combo(spec, opts, seed=32)  # use the main seed
#     C_out, tmeta = proc(C_raw)

#     # make gif 
#     save_animation(C_out,
#         "test.gif",
#         fps=12)


# #     # # A callable that produces (C, generator_meta, generator_params, transforms_meta)
# #     # def make_series(seed: int):
# #     #     # Use main seed for series; sub-seeds are handled by combo internally
# #     #     spec_local = GridSpec(nx=spec.nx, ny=spec.ny, T=spec.T, seed=seed)
# #     #     C_raw = combo(spec_local, opts, seed)
# #     #     C_out, tmeta = proc(C_raw)
# #     #     gen_meta = {"name": combo.name}
# #     #     gen_params = combo.default_params()
# #     #     # enforce float32 for storage efficiency
# #     #     return C_out.astype(np.float32), gen_meta, gen_params, tmeta

# #     # # Build a bundle across several seeds
# #     # seeds = [11, 12, 13, 14, 15]
# #     # out_dir, archive = build_bundle_from_callable(
# #     #     out_root="workloads",
# #     #     bundle_tag="gaussian_advect",
# #     #     seeds=seeds,
# #     #     make_series=make_series,
# #     #     tag_fn=lambda s: f"mix_fixed_range_seed_{s}",
# #     #     fps=10,
# #     #     compress="zip",
# #     #     overwrite=False,
# #     # )

# #     # print(f"[OK] Saved workload bundle folder: {out_dir}")
# #     # print(f"[OK] Compressed archive:          {archive}")