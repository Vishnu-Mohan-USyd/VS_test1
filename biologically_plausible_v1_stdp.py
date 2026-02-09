#!/usr/bin/env python3
"""biologically_plausible_v1_stdp.py

RGC -> LGN -> V1(L4) spiking network with STDP that learns orientation selectivity.

BIOLOGICAL PLAUSIBILITY IMPROVEMENTS:
1. Izhikevich neurons replace LIF neurons (proper parameters for each cell type)
2. Center–surround RGC front-end (DoG) replaces global DC-removal proxies
3. Local PV/SOM interneuron circuits replace global inhibition
4. Thalamocortical short-term depression (STP) provides fast gain control
5. Optional slow synaptic scaling (disabled by default; no global normalization)
6. Triplet STDP rule for more realistic plasticity
7. Lateral connectivity between ensembles (excitatory and inhibitory)

Neuron types and their Izhikevich parameters (from Izhikevich 2003, 2007):
- Thalamocortical (TC) LGN: a=0.02, b=0.25, c=-65, d=0.05 (rebound bursting)
- Regular Spiking (RS) V1 excitatory: a=0.02, b=0.2, c=-65, d=8
- Fast Spiking (FS) PV interneurons: a=0.1, b=0.2, c=-65, d=2
- Low-threshold spiking (LTS) SOM interneurons: a=0.02, b=0.25, c=-65, d=2

References:
- Izhikevich (2003) "Simple model of spiking neurons"
- Izhikevich (2007) "Dynamical Systems in Neuroscience"
- Turrigiano (2008) "Homeostatic synaptic plasticity"
- Pfister & Gerstner (2006) "Triplets of spikes in STDP"

License: MIT
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Tuple, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    gaussian_filter = None


def _gaussian_filter_fallback(img: np.ndarray, sigma: float) -> np.ndarray:
    """Small, dependency-free Gaussian blur fallback (separable, reflect padding).

    Only used for visualization when SciPy isn't available.
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return img
    radius = int(max(1, math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= float(k.sum() + 1e-12)

    arr = img.astype(np.float64, copy=False)

    pad = radius
    a = np.pad(arr, ((0, 0), (pad, pad)), mode="reflect")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, a)

    a = np.pad(a, ((pad, pad), (0, 0)), mode="reflect")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, a)
    return a.astype(np.float32, copy=False)


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def circ_mean_resultant_180(angles_deg: np.ndarray) -> Tuple[float, float]:
    """Return (resultant_length, mean_angle_deg) for orientation angles (period 180°)."""
    ang = np.deg2rad(angles_deg.astype(np.float64))
    vec = np.mean(np.exp(1j * 2.0 * ang))
    r = float(np.abs(vec))
    mu = float((0.5 * np.angle(vec)) % np.pi)
    return r, float(np.rad2deg(mu))

def max_circ_gap_180(angles_deg: np.ndarray) -> float:
    """Max circular gap (deg) on [0,180) for a set of orientation angles."""
    if angles_deg.size <= 1:
        return 180.0
    a = np.sort(angles_deg.astype(np.float64) % 180.0)
    gaps = np.diff(np.concatenate([a, a[:1] + 180.0]))
    return float(gaps.max())

def _projection_kernel(N: int, X_src: np.ndarray, Y_src: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian "scatter" kernel from irregular source samples -> regular N×N grid.

    Returns K with shape (N*N, N*N) such that:
        field_grid_flat = weights @ K.T

    where `weights` has shape (M, N*N) corresponding to source samples at (X_src, Y_src).
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0 for projection kernel")

    xs = (np.arange(int(N), dtype=np.float64) - (int(N) - 1) / 2.0).astype(np.float64)
    ys = (np.arange(int(N), dtype=np.float64) - (int(N) - 1) / 2.0).astype(np.float64)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1)  # (G,2)

    src = np.stack([X_src.astype(np.float64).ravel(), Y_src.astype(np.float64).ravel()], axis=1)  # (P,2)
    d2 = np.square(grid[:, None, :] - src[None, :, :]).sum(axis=2)  # (G,P)
    K = np.exp(-d2 / (2.0 * sigma * sigma)).astype(np.float32, copy=False)
    return K


def compute_osi(rates_hz: np.ndarray, thetas_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Classic doubled-angle OSI.

    OSI = |sum r(theta) e^{i2*theta}| / sum r(theta)
    pref = 0.5 * arg(sum r(theta) e^{i2*theta}) in [0,180)
    """
    th = np.deg2rad(thetas_deg)
    vec = (rates_hz * np.exp(1j * 2 * th)[None, :]).sum(axis=1)
    denom = rates_hz.sum(axis=1) + 1e-9
    osi = np.abs(vec) / denom
    pref = (0.5 * np.angle(vec)) % np.pi
    return osi, np.rad2deg(pref)

def onoff_weight_corr(
    W: np.ndarray,
    N: int,
    *,
    on_to_off: np.ndarray | None = None,
    X_on: np.ndarray | None = None,
    Y_on: np.ndarray | None = None,
    X_off: np.ndarray | None = None,
    Y_off: np.ndarray | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """Per-neuron correlation between ON and OFF thalamocortical weights (mean-removed).

    If `on_to_off` is provided, OFF weights are re-indexed onto the ON lattice using the mapping
    (nearest-neighbor ON↔OFF matching), which makes this metric meaningful when ON/OFF mosaics are
    not perfectly co-registered.

    Positive values indicate ON/OFF weights are spatially similar (bad for push–pull).
    Negative values indicate phase-opponent ON/OFF structure (push–pull-like).
    """
    n_pix = int(N) * int(N)
    W_on = W[:, :n_pix].astype(np.float64, copy=False)
    W_off = W[:, n_pix:].astype(np.float64, copy=False)

    if (X_on is not None) and (Y_on is not None) and (X_off is not None) and (Y_off is not None):
        sig = 0.5 if sigma is None else float(sigma)
        K_on = _projection_kernel(int(N), X_on, Y_on, sig)   # (G,P)
        K_off = _projection_kernel(int(N), X_off, Y_off, sig)
        W_on_g = (W_on @ K_on.T).astype(np.float64, copy=False)   # (M,G)
        W_off_g = (W_off @ K_off.T).astype(np.float64, copy=False)
        W_on_g = W_on_g - W_on_g.mean(axis=1, keepdims=True)
        W_off_g = W_off_g - W_off_g.mean(axis=1, keepdims=True)
        denom = (np.linalg.norm(W_on_g, axis=1) * np.linalg.norm(W_off_g, axis=1)) + 1e-12
        return (W_on_g * W_off_g).sum(axis=1) / denom

    if on_to_off is not None:
        W_off = W_off[:, on_to_off.astype(np.int32, copy=False)]

    W_on = W_on - W_on.mean(axis=1, keepdims=True)
    W_off = W_off - W_off.mean(axis=1, keepdims=True)
    denom = (np.linalg.norm(W_on, axis=1) * np.linalg.norm(W_off, axis=1)) + 1e-12
    return (W_on * W_off).sum(axis=1) / denom

def rf_fft_orientation_metrics(
    W: np.ndarray,
    N: int,
    *,
    on_to_off: np.ndarray | None = None,
    X_on: np.ndarray | None = None,
    Y_on: np.ndarray | None = None,
    X_off: np.ndarray | None = None,
    Y_off: np.ndarray | None = None,
    sigma: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orientedness and preferred orientation from the signed RF Wdiff = Won - Woff.

    Uses the doubled-angle vector sum of Fourier power:
        orientedness = |Σ P(k) e^{i2φ_k}| / Σ P(k)
        pref = 0.5 * arg(Σ P(k) e^{i2φ_k})  in [0,180)

    This is a weight-based diagnostic (not spike-based) to avoid "spike-sparse" OSI artifacts.
    """
    n_pix = int(N) * int(N)
    M = int(W.shape[0])

    W_on = W[:, :n_pix].astype(np.float64, copy=False)
    W_off = W[:, n_pix:].astype(np.float64, copy=False)

    if (X_on is not None) and (Y_on is not None) and (X_off is not None) and (Y_off is not None):
        sig = 0.5 if sigma is None else float(sigma)
        K_on = _projection_kernel(int(N), X_on, Y_on, sig)
        K_off = _projection_kernel(int(N), X_off, Y_off, sig)
        W_on_g = (W_on @ K_on.T).astype(np.float64, copy=False)
        W_off_g = (W_off @ K_off.T).astype(np.float64, copy=False)
        Wdiff = (W_on_g - W_off_g).reshape(M, N, N).astype(np.float64, copy=False)
    else:
        Won = W_on.reshape(M, N, N).astype(np.float64, copy=False)
        if on_to_off is not None:
            W_off = W_off[:, on_to_off.astype(np.int32, copy=False)]
        Woff = W_off.reshape(M, N, N).astype(np.float64, copy=False)
        Wdiff = Won - Woff

    Wdiff = Wdiff - Wdiff.mean(axis=(1, 2), keepdims=True)

    F = np.fft.fftshift(np.fft.fft2(Wdiff, axes=(1, 2)), axes=(1, 2))
    P = (F.real * F.real + F.imag * F.imag)  # power
    cx = int(N // 2)
    cy = int(N // 2)
    P[:, cx, cy] = 0.0

    ys, xs = np.indices((N, N))
    dx = (xs - cy).astype(np.float64)
    dy = (ys - cx).astype(np.float64)
    r = np.sqrt(dx * dx + dy * dy)
    rmin = 1.0
    rmax = max(2.0, float(N) / 2.0 - 1.0)
    mask = (r >= rmin) & (r <= rmax)
    phi = np.arctan2(dy, dx)[mask]  # spatial-frequency angle

    w = P[:, mask]
    vec = (w * np.exp(1j * 2.0 * phi)[None, :]).sum(axis=1)
    denom = w.sum(axis=1) + 1e-12
    orientedness = np.abs(vec) / denom
    pref = (0.5 * np.angle(vec)) % np.pi
    return orientedness.astype(np.float32), np.rad2deg(pref).astype(np.float32)

def rf_grating_match_tuning(
    W: np.ndarray,
    N: int,
    spatial_freq: float,
    thetas_deg: np.ndarray,
    *,
    on_to_off: np.ndarray | None = None,
    X_on: np.ndarray | None = None,
    Y_on: np.ndarray | None = None,
    X_off: np.ndarray | None = None,
    Y_off: np.ndarray | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """Weight-based orientation tuning: project Wdiff onto sinusoidal gratings at `spatial_freq`.

    For each orientation θ, compute the maximum (over phase) dot-product amplitude between the
    signed RF Wdiff = Won - Woff and a sinusoidal grating at θ:

        amp(θ) = sqrt( (Wdiff·cos(gθ))^2 + (Wdiff·sin(gθ))^2 )

    Returns: (M, K) amplitude matrix for K orientations.
    """
    n_pix = int(N) * int(N)
    M = int(W.shape[0])
    K = int(len(thetas_deg))

    W_on = W[:, :n_pix].astype(np.float64, copy=False)
    W_off = W[:, n_pix:].astype(np.float64, copy=False)

    if (X_on is not None) and (Y_on is not None) and (X_off is not None) and (Y_off is not None):
        sig = 0.5 if sigma is None else float(sigma)
        K_on = _projection_kernel(int(N), X_on, Y_on, sig)
        K_off = _projection_kernel(int(N), X_off, Y_off, sig)
        W_on_g = (W_on @ K_on.T).astype(np.float64, copy=False)
        W_off_g = (W_off @ K_off.T).astype(np.float64, copy=False)
        Wdiff = (W_on_g - W_off_g).reshape(M, N, N).astype(np.float64, copy=False)
    else:
        if on_to_off is not None:
            W_off = W_off[:, on_to_off.astype(np.int32, copy=False)]
        Wdiff = (W_on - W_off).reshape(M, N, N).astype(np.float64, copy=False)
    Wdiff = Wdiff - Wdiff.mean(axis=(1, 2), keepdims=True)

    xs = (np.arange(N, dtype=np.float64) - (N - 1) / 2.0)
    ys = (np.arange(N, dtype=np.float64) - (N - 1) / 2.0)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    amps = np.zeros((M, K), dtype=np.float64)
    sf = float(spatial_freq)
    for j, th_deg in enumerate(thetas_deg.astype(np.float64)):
        th = float(np.deg2rad(th_deg))
        proj = X * math.cos(th) + Y * math.sin(th)
        gcos = np.cos(2.0 * math.pi * sf * proj)
        gsin = np.sin(2.0 * math.pi * sf * proj)
        gcos -= float(gcos.mean())
        gsin -= float(gsin.mean())
        a = (Wdiff * gcos[None, :, :]).sum(axis=(1, 2))
        b = (Wdiff * gsin[None, :, :]).sum(axis=(1, 2))
        amps[:, j] = np.sqrt(a * a + b * b)

    return amps.astype(np.float32)


def fit_von_mises_180(y: np.ndarray, thetas_deg: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Fit a 180-deg periodic von Mises tuning curve:

        y(theta) ~= b + a * exp(kappa * cos(2*(theta - theta0)))

    Returns: (kappa, theta0_deg, a, b, y_fit)
    """
    th = np.deg2rad(thetas_deg.astype(np.float64))
    y = y.astype(np.float64)

    # Grids chosen to be lightweight but stable for self-tests.
    theta0_grid = np.deg2rad(np.linspace(0.0, 180.0, 181, endpoint=False))
    kappa_grid = np.concatenate([
        np.linspace(0.1, 2.0, 20, endpoint=True),
        np.linspace(2.0, 20.0, 30, endpoint=True),
    ])

    best = (float("inf"), 1.0, 0.0, 0.0, 0.0, None)  # (sse, kappa, theta0, a, b, y_fit)
    for theta0 in theta0_grid:
        cos_term = np.cos(2.0 * (th - theta0))
        for kappa in kappa_grid:
            f = np.exp(kappa * cos_term)
            # Linear least squares for b + a*f
            A = np.stack([np.ones_like(f), f], axis=1)  # (K,2)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            b, a = float(coef[0]), float(coef[1])
            if a < 0:
                continue
            y_fit = b + a * f
            sse = float(np.square(y_fit - y).sum())
            if sse < best[0]:
                best = (sse, float(kappa), float(theta0), a, b, y_fit)

    _, kappa, theta0, a, b, y_fit = best
    theta0_deg = float(np.rad2deg(theta0) % 180.0)
    return kappa, theta0_deg, float(a), float(b), y_fit.astype(np.float32)


def von_mises_hwhh_deg(kappa: float) -> float:
    """Half-width at half-height (degrees) for the 180-deg von Mises term exp(kappa*cos(2Δ))."""
    if kappa <= math.log(2.0):
        return 90.0
    return float(np.rad2deg(0.5 * np.arccos(1.0 - math.log(2.0) / kappa)))


def tuning_hwhh_deg(rates_hz: np.ndarray, thetas_deg: np.ndarray) -> np.ndarray:
    """Compute HWHH (deg) for each neuron's tuning curve using a von Mises fit."""
    hwhh = np.zeros(rates_hz.shape[0], dtype=np.float32)
    for i in range(rates_hz.shape[0]):
        kappa, _, _, _, _ = fit_von_mises_180(rates_hz[i], thetas_deg)
        hwhh[i] = von_mises_hwhh_deg(kappa)
    return hwhh

# =============================================================================
# Izhikevich Neuron Parameters (from literature)
# =============================================================================

@dataclass
class IzhikevichParams:
    """Izhikevich neuron parameters for different cell types."""
    a: float  # Recovery time scale (smaller = slower)
    b: float  # Sensitivity of recovery to subthreshold fluctuations
    c: float  # After-spike reset value of v (mV)
    d: float  # After-spike increment of u
    v_peak: float = 30.0  # Spike cutoff (mV)
    v_init: float = -65.0  # Initial membrane potential


# Literature-based parameters
TC_PARAMS = IzhikevichParams(a=0.02, b=0.25, c=-65.0, d=0.05)  # Thalamocortical
RS_PARAMS = IzhikevichParams(a=0.02, b=0.2, c=-65.0, d=8.0)    # Regular spiking
FS_PARAMS = IzhikevichParams(a=0.1, b=0.2, c=-65.0, d=2.0)     # Fast spiking (PV)
LTS_PARAMS = IzhikevichParams(a=0.02, b=0.25, c=-65.0, d=2.0)  # Low-threshold spiking (SOM)


@dataclass
class Params:
    """Network and simulation parameters."""
    # Species / interpretation (metadata only; used to keep assumptions explicit when scaling).
    species: str = "generic"  # e.g., {"generic","cat","ferret","primate","mouse"}

    # Geometry
    N: int = 8  # Patch size (NxN)
    M: int = 8  # Number of V1 ensembles (like a hypercolumn)
    cortex_shape: Tuple[int, int] | None = None  # (H,W) for 2D sheet; None => (1,M)
    cortex_wrap: bool = True  # periodic boundary for lateral distance computations
    dt_ms: float = 0.5  # Time step (smaller for Izhikevich stability)

    # Training
    segment_ms: int = 300
    train_segments: int = 200
    seed: int = 1

    # Developmental stimulus during training (evaluation still uses gratings for OSI).
    # - "grating": drifting gratings (classic OS emergence toy)
    # - "sparse_spots": flickering sparse spot movie (e.g., Ohshiro et al., 2011)
    # - "white_noise": dense spatiotemporal noise (Linsker-style)
    train_stimulus: str = "grating"  # {"grating","sparse_spots","white_noise"}
    train_contrast: float = 1.0

    # Drifting gratings
    # NOTE: With very small patches (e.g., N=8), too-low spatial_freq can yield <~1 cycle across the
    # receptive field and tends to produce coarse "edge-like" solutions with lattice-dependent biases.
    # If you see mostly-diagonal receptive fields, try increasing `--spatial-freq` (e.g., 0.16–0.24) or N.
    spatial_freq: float = 0.18
    temporal_freq: float = 8.0
    base_rate: float = 1.0
    gain_rate: float = 205.0

    # Sparse spot movie (flash-like stimulus).
    # Implemented as a random sparse set of bright/dark pixels that refresh every `spots_frame_ms`.
    spots_density: float = 0.02   # fraction of pixels active per frame (0..1)
    # Ohshiro et al. (2011) used 375 ms refresh with randomly positioned bright/dark spots.
    spots_frame_ms: float = 375.0  # refresh period (ms)
    spots_amp: float = 3.0         # luminance amplitude of each spot (+/-amp)
    spots_sigma: float = 1.2       # pixels; <=0 => single-pixel spots, >0 => Gaussian blobs

    # Dense random noise stimulus (spatiotemporal white noise).
    # NOTE: Defaults aim to produce robust thalamic/cortical spiking during training.
    noise_sigma: float = 1.0      # std of pixel luminance noise
    noise_clip: float = 2.5       # clip luminance noise to [-clip, +clip]
    noise_frame_ms: float = 16.7  # refresh period (ms) ~60 Hz

    # RGC center–surround front-end (Difference-of-Gaussians, DoG).
    # This replaces any global DC-removal "proxy" and better matches retinal/LGN contrast encoding.
    rgc_center_surround: bool = True
    rgc_center_sigma: float = 0.6     # pixels (center)
    rgc_surround_sigma: float = 1.8   # pixels (surround)
    rgc_surround_balance: bool = True  # choose surround gain per RGC so each kernel sums ~0
    rgc_dog_norm: str = "l1"          # {"none","l1","l2"} normalize kernel rows for stable gain
    # Implementation details for DoG filtering. "padded_fft" avoids edge-induced orientation bias
    # by filtering on a larger padded field and sampling the central patch.
    rgc_dog_impl: str = "padded_fft"  # {"matrix","padded_fft"}
    rgc_dog_pad: int = 0              # padding (pixels); 0 => auto based on surround sigma

    rgc_pos_jitter: float = 0.15  # Break lattice artifacts (fraction of pixel spacing)
    # ON/OFF mosaics are distinct in real retina/LGN (not perfectly co-registered).
    # When enabled, ON and OFF RGCs sample the stimulus at slightly different positions.
    rgc_separate_onoff_mosaics: bool = False
    rgc_onoff_offset: float = 0.5  # pixels (magnitude of ON↔OFF lattice offset)
    rgc_onoff_offset_angle_deg: float | None = None  # None => choose a seeded random angle (avoids baked-in axis bias)

    # RGC temporal dynamics (optional).
    # Real RGC/LGN channels have temporal filtering and refractory effects; these are
    # turned off by default to preserve prior behavior, but can be enabled for
    # spot/noise rearing experiments.
    rgc_temporal_filter: bool = False
    rgc_tau_fast: float = 10.0   # ms
    rgc_tau_slow: float = 50.0   # ms
    rgc_temporal_gain: float = 1.0
    rgc_refractory_ms: float = 0.0  # absolute refractory (ms); 0 disables

    # RGC->LGN synaptic weight (scaled for Izhikevich pA currents)
    # Izhikevich model uses currents ~0-40 pA for typical spiking
    w_rgc_lgn: float = 5.0
    # Retinogeniculate pooling (RGC->LGN).
    # Relay-cell center drive pools nearby same-sign RGCs, while weaker opposite-sign
    # pooling provides an antagonistic surround-like contribution.
    lgn_pooling: bool = False
    lgn_pool_sigma_center: float = 0.9
    lgn_pool_sigma_surround: float = 1.8
    lgn_pool_same_gain: float = 1.0
    lgn_pool_opponent_gain: float = 0.18
    # Optional temporal smoothing of pooled RGC drive before LGN relay spiking.
    lgn_rgc_tau_ms: float = 0.0

    # LGN->V1 weights & delays
    delay_max: int = 12
    w_init_mean: float = 0.25  # Scaled for Izhikevich (total input ~15-30 pA)
    w_init_std: float = 0.08
    w_max: float = 1.0

    # Thalamocortical short-term synaptic depression (STP) at LGN->V1 synapses.
    # This is a local, fast gain-control mechanism that complements slower homeostatic plasticity.
    tc_stp_enabled: bool = True
    tc_stp_u: float = 0.05          # per-spike depletion fraction (0..1)
    tc_stp_tau_rec: float = 50.0    # ms recovery time constant
    # Thalamocortical STP at LGN->PV synapses (feedforward inhibition pathway).
    # Kept separate because depression dynamics can differ between PV/FS and pyramidal targets.
    tc_stp_pv_enabled: bool = True
    tc_stp_pv_u: float = 0.05
    tc_stp_pv_tau_rec: float = 50.0

    # Retinotopic locality (thalamocortical arbor). Implemented as a fixed spatial envelope that
    # caps the maximum synaptic weight for each LGN pixel (same envelope for ON/OFF channels).
    lgn_sigma_e: float = 2.0   # pixels (E receptive field radius)
    lgn_sigma_pv: float = 3.0  # pixels (PV tends to pool more broadly)
    lgn_sigma_pp: float = 2.0  # pixels (push–pull pathway locality)

    # Anatomical sparsity prior for thalamocortical connectivity (LGN->E).
    # Implemented as a fixed structural mask: absent synapses never appear and do not undergo plasticity.
    # `tc_conn_fraction_e=1.0` recovers the original dense-with-cap initialization.
    tc_conn_fraction_e: float = 0.75  # fraction of LGN afferents present per E neuron (0..1]
    # Sparse thalamic drive to local interneurons (PV, PP). Defaults keep E/I roughly balanced under sparsity.
    tc_conn_fraction_pv: float = 0.8
    tc_conn_fraction_pp: float = 0.5
    tc_conn_balance_onoff: bool = True  # when sparse, sample similar counts from ON and OFF channels

    # Homeostatic synaptic scaling (replaces global normalization)
    target_rate_hz: float = 8.0  # Target firing rate for homeostasis
    # NOTE: Canonical synaptic scaling is slow (hours–days). By default, scaling is disabled for
    # short simulations; stability is instead provided by local STDP bounds, heterosynaptic effects,
    # inhibitory plasticity, and short-term depression.
    tau_homeostasis: float = 3_600_000.0  # ms (~1 hour; affects firing-rate averaging)
    homeostasis_rate: float = 0.0  # Learning rate for synaptic scaling (0 disables)
    homeostasis_clip: float = 0.02  # Per-application multiplicative clamp (e.g., 0.02 => [0.98,1.02])

    # Developmental ON/OFF "split constraint" (local synaptic resource conservation).
    # Inspired by correlation-based RF development models used for spot/noise rearing analyses:
    # maintain separate ON and OFF synaptic resource pools per postsynaptic neuron, implemented
    # as slow, local scaling at segment boundaries (not a fast global normalization).
    split_constraint_rate: float = 0.2   # 0 disables; >0 applies per-segment ON/OFF pool scaling
    split_constraint_clip: float = 0.02  # clamp per-application multiplicative factor
    split_constraint_equalize_onoff: bool = True  # target ON and OFF pools to equal total strength

    # Intrinsic homeostasis (bias current) for V1 excitatory neurons
    v1_bias_init: float = 0.0
    v1_bias_eta: float = 0.0
    v1_bias_clip: float = 20.0

    # STDP parameters (pair-based with triplet enhancement)
    # Time constants matched to original working code
    tau_plus: float = 20.0   # Pre-before-post time constant
    tau_minus: float = 20.0  # Post-before-pre time constant
    tau_x: float = 101.0     # Slow pre trace for triplet
    tau_y: float = 125.0     # Slow post trace for triplet
    A2_plus: float = 0.008   # Pair LTP amplitude (matches original)
    A3_plus: float = 0.002   # Triplet LTP enhancement
    A2_minus: float = 0.010  # Pair LTD amplitude (matches original)
    A3_minus: float = 0.0    # Triplet LTD amplitude (often 0)

    # Heterosynaptic (resource-like) depression on postsynaptic spikes
    A_het: float = 0.032
    # ON/OFF split competition (developmental constraint).
    # A local heterosynaptic depression term that discourages co-located ON and OFF subfields
    # from strengthening together under non-oriented stimuli (a spiking analog of the
    # "split constraint" used in correlation-based RF development models).
    A_split: float = 0.2
    # Adaptive gain for split competition based on each neuron's ON/OFF overlap.
    # High ON/OFF overlap -> stronger split competition; phase-opponent weights -> weaker competition.
    split_overlap_adaptive: bool = False
    split_overlap_min: float = 0.6
    split_overlap_max: float = 1.4

    # Weight decay (biologically: synaptic turnover)
    w_decay: float = 0.00000001  # Per-timestep weight decay (slow turnover; ~hours time scale)

    # Local inhibitory circuit parameters
    n_pv_per_ensemble: int = 1  # PV interneurons per ensemble
    n_som_per_ensemble: int = 1  # SOM interneurons per ensemble for lateral inhibition
    # PV connectivity realism: allow PV to couple to multiple nearby ensembles instead of acting as a
    # private "shadow" interneuron. Units are in cortical-distance coordinates (same as lateral kernels).
    # Set to 0 to recover the legacy private PV<->E wiring.
    pv_in_sigma: float = 0.0   # E -> PV spread
    pv_out_sigma: float = 0.0  # PV -> E spread
    # PV<->PV mutual inhibition (optional; current-based inhibitory input to PV).
    pv_pv_sigma: float = 0.0   # 0 disables
    w_pv_pv: float = 0.0       # inhibitory current increment onto PV per PV spike

    # LGN->PV feedforward inhibition (thalamocortical drive to FS interneurons)
    w_lgn_pv_gain: float = 0.5
    w_lgn_pv_init_mean: float = 0.20
    w_lgn_pv_init_std: float = 0.05

    # Push–pull inhibition: LGN-driven phase-opponent inhibitory conductance onto E
    # Implemented as an explicit interneuron pathway: LGN (ON/OFF swapped) -> PP interneuron -> E (GABA conductance).
    n_pp_per_ensemble: int = 1
    tau_ampa_pp: float = 2.0  # ms (thalamic drive to PP can be faster than to E)
    w_lgn_pp_gain: float = 5.0
    w_pushpull: float = 0.011  # PP->E inhibitory conductance increment
    # Hypothesis knob: whether PP receives ON/OFF-swapped thalamic input (phase opposition).
    # If False, PP still provides LGN-driven inhibition but is not explicitly "push–pull" by construction.
    pp_onoff_swap: bool = False
    pp_plastic: bool = True
    pp_w_init_mean: float = 0.05
    pp_w_init_std: float = 0.02
    pp_w_match: float = 0.0  # Initial correlation of LGN->PP weights with LGN->E weights (0..1)
    pp_w_max: float = 1.0
    # Local thalamic plasticity onto PP interneurons (pair-based STDP).
    # Uses the same tau_plus/tau_minus time constants as LGN->E by default.
    pp_A_plus: float = 0.02
    pp_A_minus: float = 0.025
    pp_decay: float = 0.00001

    # E->PV (feedforward inhibition) - scaled for Izhikevich
    w_e_pv: float = 5.0
    # PV->E (feedback inhibition, local)
    # NOTE: Treated as a GABA conductance increment (not subtractive current).
    w_pv_e: float = 0.10

    # PV->E inhibitory plasticity (homeostatic iSTDP-style)
    pv_inhib_plastic: bool = True
    tau_pv_istdp: float = 20.0
    eta_pv_istdp: float = 0.0001
    w_pv_e_max: float = 0.5
    # E->SOM (lateral inhibition drive from this ensemble)
    w_e_som: float = 6.0
    # SOM->E (lateral inhibition TO OTHER ensembles - NOT self)
    # NOTE: Treated as a GABA conductance increment (not subtractive current).
    w_som_e: float = 0.05

    # SOM lateral circuit spatial scales (in "ensemble index" distance; circular)
    som_in_sigma: float = 2.0   # E->SOM spread (can be longer-range)
    som_out_sigma: float = 0.75  # SOM->E spread (more local)
    som_self_inhibit: bool = True

    # VIP interneurons (disinhibitory motif): VIP -> SOM -> E
    # Set n_vip_per_ensemble=0 to disable (default preserves legacy behavior).
    n_vip_per_ensemble: int = 0
    # Local E->VIP recruitment (current-based, delayed by one step like E->PV).
    w_e_vip: float = 0.0
    # VIP->SOM inhibition (current-based).
    w_vip_som: float = 0.0
    # Optional tonic bias current to VIP (models state/top-down drive in a crude way).
    vip_bias_current: float = 0.0

    # Lateral excitatory connections (between nearby ensembles)
    w_e_e_lateral: float = 0.01
    lateral_sigma: float = 1.5  # Gaussian spread for lateral connections

    # Lateral/recurrent E->E plasticity (slow STDP to promote like-to-like coupling)
    ee_plastic: bool = False
    ee_tau_plus: float = 20.0
    ee_tau_minus: float = 20.0
    ee_A_plus: float = 0.0005
    ee_A_minus: float = 0.0002
    ee_w_max: float = 0.2
    ee_decay: float = 0.000001

    # Synaptic time constants
    tau_ampa: float = 5.0   # AMPA receptor
    tau_gaba: float = 10.0  # GABA receptor
    tau_gaba_rise_pv: float = 1.0  # ms (PV->E synaptic rise; makes inhibition slightly delayed)
    tau_apical: float = 20.0  # ms (apical/feedback-like excitatory conductance)

    # Reversal potentials (for conductance-based synapses)
    E_exc: float = 0.0  # mV (AMPA/NMDA; simplified)
    E_inh: float = -70.0  # mV (GABA_A)

    # Conductance scaling: convert weight-sums into effective synaptic conductances.
    # Roughly, g * (E_exc - V) should be in the same range as the previous current-based drive.
    w_exc_gain: float = 0.015  # ~1/65 for E_exc=0mV and V_rest≈-65mV

    # Apical modulation (minimal two-stream scaffold for future feedback/expectation modeling).
    # When apical_gain=0, apical drive has no effect (default preserves legacy behavior).
    apical_gain: float = 0.0
    apical_threshold: float = 0.0
    apical_slope: float = 0.1

    # Minimal laminar scaffold: an optional L2/3 excitatory population driven by L4.
    # This makes "feedback/apical" inputs anatomically interpretable (apical -> L2/3) without
    # rewriting the existing L4 thalamocortical learning block.
    laminar_enabled: bool = False
    # Basal drive from L4 E spikes to L2/3 E conductance (in the same "current weight" units as W_e_e).
    w_l4_l23: float = 10.0
    # Spread of L4->L2/3 projections over cortex_dist2 (0 => same-ensemble only).
    l4_l23_sigma: float = 0.0


class IzhikevichPopulation:
    """Population of Izhikevich neurons."""

    def __init__(self, n: int, params: IzhikevichParams, dt_ms: float, rng: np.random.Generator):
        self.n = n
        self.p = params
        self.dt = dt_ms
        self.rng = rng

        # State variables
        self.v = np.full(n, params.v_init, dtype=np.float32)
        self.u = params.b * self.v.copy()

        # Small random perturbation to break symmetry
        self._apply_symmetry_breaking_jitter()

    def _apply_symmetry_breaking_jitter(self) -> None:
        """Add small random perturbations (models background fluctuations; breaks symmetry)."""
        self.v += self.rng.uniform(-5, 5, self.n).astype(np.float32)
        self.u += self.rng.uniform(-2, 2, self.n).astype(np.float32)

    def reset(self):
        """Reset to initial state."""
        self.v.fill(self.p.v_init)
        self.u = self.p.b * self.v.copy()

    def step(self, I_ext: np.ndarray) -> np.ndarray:
        """
        Advance one time step with external current I_ext.
        Returns binary spike array.

        Uses the standard Izhikevich equations:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)

        if v >= v_peak: v <- c, u <- u + d
        """
        p = self.p
        dt = self.dt

        # Euler integration (with sub-stepping for stability)
        # Using 2 sub-steps per dt for better numerical stability
        dt_sub = dt / 2.0

        for _ in range(2):
            # Clamp v to prevent numerical blowup
            v_clamped = np.clip(self.v, -100, p.v_peak)

            dv = (0.04 * v_clamped * v_clamped + 5.0 * v_clamped + 140.0 - self.u + I_ext) * dt_sub
            du = p.a * (p.b * v_clamped - self.u) * dt_sub

            self.v += dv
            self.u += du

        # Detect spikes
        spikes = (self.v >= p.v_peak).astype(np.uint8)

        # Reset spiking neurons
        spike_idx = spikes.astype(bool)
        self.v[spike_idx] = p.c
        self.u[spike_idx] += p.d

        return spikes


class TripletSTDP:
    """
    Triplet STDP rule from Pfister & Gerstner (2006).

    This implementation properly handles per-synapse traces with delays.
    Each synapse from pre neuron j to post neuron i has its own trace,
    because the spike arrival times depend on axonal delays.

    Maintains traces per synapse (M, n_pre):
    - x_pre: fast pre trace (incremented when pre spike arrives at synapse)
    - x_pre_slow: slow pre trace for triplet (same)
    And traces per post neuron (M,):
    - x_post: fast post trace
    - x_post_slow: slow post trace for triplet
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        p: Params,
        rng: np.random.Generator,
        *,
        split_on_to_off: np.ndarray | None = None,
        split_off_to_on: np.ndarray | None = None,
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.p = p

        # Pre traces - per synapse (n_post, n_pre)
        # These track the arrival of pre-synaptic spikes at each synapse
        self.x_pre = np.zeros((n_post, n_pre), dtype=np.float32)
        self.x_pre_slow = np.zeros((n_post, n_pre), dtype=np.float32)

        # Post traces - per neuron (n_post,)
        self.x_post = np.zeros(n_post, dtype=np.float32)
        self.x_post_slow = np.zeros(n_post, dtype=np.float32)

        # Decay factors
        self.decay_pre = math.exp(-p.dt_ms / p.tau_plus)
        self.decay_pre_slow = math.exp(-p.dt_ms / p.tau_x)
        self.decay_post = math.exp(-p.dt_ms / p.tau_minus)
        self.decay_post_slow = math.exp(-p.dt_ms / p.tau_y)

        self.split_on_to_off: np.ndarray | None = None
        self.split_off_to_on: np.ndarray | None = None
        if (split_on_to_off is not None) and (split_off_to_on is not None):
            self.split_on_to_off = split_on_to_off.astype(np.int32, copy=True)
            self.split_off_to_on = split_off_to_on.astype(np.int32, copy=True)

    def reset(self):
        self.x_pre.fill(0)
        self.x_pre_slow.fill(0)
        self.x_post.fill(0)
        self.x_post_slow.fill(0)

    def update(self, arrivals: np.ndarray, post_spikes: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Update traces and compute MULTIPLICATIVE weight changes.

        CRITICAL: Order of operations matches original working code:
        1. Decay traces
        2. LTD: when pre arrives, depress based on OLD post trace
        3. Update pre traces (so current arrivals are included)
        4. LTP: when post fires, potentiate based on NEW pre trace (includes current arrivals)
        5. Update post traces

        This order ensures that coincident pre-post activity within the same timestep
        contributes to LTP, which is essential for proper orientation selectivity learning.

        arrivals: (n_post, n_pre) - which pre-spikes arrived at each synapse this timestep
        post_spikes: (n_post,) binary
        W: (n_post, n_pre) current weights

        Returns: dW weight change matrix (already includes multiplicative factors)
        """
        p = self.p

        # Decay all traces
        self.x_pre *= self.decay_pre
        self.x_pre_slow *= self.decay_pre_slow
        self.x_post *= self.decay_post
        self.x_post_slow *= self.decay_post_slow

        dW = np.zeros_like(W)

        # LTD: When pre spike arrives, depress based on post trace (OLD, before this spike)
        # Multiplicative: dW- proportional to W (stronger synapses lose more)
        if arrivals.any():
            dW -= p.A2_minus * arrivals * self.x_post[:, None] * W

        # Update pre traces BEFORE computing LTP
        # This ensures current arrivals contribute to LTP if post fires this timestep
        self.x_pre += arrivals
        self.x_pre_slow += arrivals

        # LTP: When post fires, potentiate based on pre trace (NEW, includes current arrivals)
        # Multiplicative: dW+ proportional to (w_max - W) (room to grow)
        # Triplet enhancement: stronger LTP when there's recent post activity
        if post_spikes.any():
            post_mask = post_spikes.astype(np.float32)
            triplet_boost = 1.0 + p.A3_plus * self.x_post_slow[:, None] / p.A2_plus
            dW += p.A2_plus * post_mask[:, None] * self.x_pre * (p.w_max - W) * triplet_boost

            # Heterosynaptic depression: postsynaptic spiking induces depression of *inactive*
            # synapses (those without a presynaptic arrival at that moment), implementing
            # competition without hard normalization.
            if p.A_het > 0:
                inactive = (1.0 - arrivals).astype(np.float32)
                dW -= p.A_het * post_mask[:, None] * inactive * W

            # ON/OFF split competition: when a postsynaptic spike occurs, recently active ON inputs
            # weaken their OFF counterparts (and vice versa). This encourages development of phase-
            # opponent ON/OFF subfields under non-oriented developmental stimuli (spots/noise),
            # without requiring any global weight normalization.
            if p.A_split > 0 and self.n_pre == 2 * p.N * p.N:
                n_pix = int(p.N) * int(p.N)
                on_trace = self.x_pre[:, :n_pix]
                off_trace = self.x_pre[:, n_pix:]
                split_gain = np.ones_like(post_mask, dtype=np.float32)
                if p.split_overlap_adaptive:
                    W_on = W[:, :n_pix]
                    W_off = W[:, n_pix:]
                    if self.split_on_to_off is not None:
                        W_off = W_off[:, self.split_on_to_off]
                    W_on_c = W_on - W_on.mean(axis=1, keepdims=True)
                    W_off_c = W_off - W_off.mean(axis=1, keepdims=True)
                    denom = (np.linalg.norm(W_on_c, axis=1) * np.linalg.norm(W_off_c, axis=1)) + 1e-12
                    overlap = (W_on_c * W_off_c).sum(axis=1) / denom
                    overlap = np.clip(0.5 * (overlap + 1.0), 0.0, 1.0)
                    split_gain = p.split_overlap_min + (p.split_overlap_max - p.split_overlap_min) * overlap
                if (self.split_on_to_off is None) or (self.split_off_to_on is None):
                    on_at_off = on_trace
                    off_at_on = off_trace
                else:
                    on_at_off = on_trace[:, self.split_off_to_on]
                    off_at_on = off_trace[:, self.split_on_to_off]
                dW[:, n_pix:] -= p.A_split * split_gain[:, None] * post_mask[:, None] * on_at_off * W[:, n_pix:]
                dW[:, :n_pix] -= p.A_split * split_gain[:, None] * post_mask[:, None] * off_at_on * W[:, :n_pix]

        # Update post traces AFTER computing plasticity
        self.x_post += post_spikes.astype(np.float32)
        self.x_post_slow += post_spikes.astype(np.float32)

        return dW


class HomeostaticScaling:
    """
    Biologically plausible homeostatic synaptic scaling.

    Based on Turrigiano (2008): neurons slowly adjust their synaptic
    strengths to maintain a target firing rate. This is a LOCAL mechanism
    that operates on each neuron independently.

    The scaling is multiplicative: w <- w * (1 + eta * (r_target - r_actual))
    """

    def __init__(self, n_post: int, p: Params):
        self.n_post = n_post
        self.p = p

        # Running average of firing rate (exponential moving average)
        self.rate_avg = np.full(n_post, p.target_rate_hz, dtype=np.float32)

        # Decay for rate averaging
        self.decay = math.exp(-p.dt_ms / p.tau_homeostasis)

    def reset(self):
        self.rate_avg.fill(self.p.target_rate_hz)

    def update_rate(self, spikes: np.ndarray, dt_ms: float):
        """Update running rate estimate."""
        instant_rate = spikes.astype(np.float32) * (1000.0 / dt_ms)  # Convert to Hz
        self.rate_avg = self.decay * self.rate_avg + (1 - self.decay) * instant_rate

    def get_scaling_factors(self) -> np.ndarray:
        """
        Get multiplicative scaling factors for each neuron's input weights.

        Returns: (n_post,) array of scaling factors
        """
        p = self.p
        # Error signal: positive if firing too slow, negative if too fast
        error = p.target_rate_hz - self.rate_avg
        # Multiplicative scaling factor
        scale = 1.0 + p.homeostasis_rate * error
        lo = 1.0 - p.homeostasis_clip
        hi = 1.0 + p.homeostasis_clip
        return np.clip(scale, lo, hi)  # Limit rate of change


class PairSTDP:
    """Pair-based STDP with per-synapse pre traces (handles axonal delays via arrivals)."""

    def __init__(self, n_pre: int, n_post: int, *, dt_ms: float, tau_plus: float, tau_minus: float,
                 A_plus: float, A_minus: float, w_max: float):
        self.n_pre = n_pre
        self.n_post = n_post
        self.dt_ms = float(dt_ms)
        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.A_plus = float(A_plus)
        self.A_minus = float(A_minus)
        self.w_max = float(w_max)

        self.x_pre = np.zeros((n_post, n_pre), dtype=np.float32)
        self.x_post = np.zeros(n_post, dtype=np.float32)

        self.decay_pre = math.exp(-self.dt_ms / max(1e-6, self.tau_plus))
        self.decay_post = math.exp(-self.dt_ms / max(1e-6, self.tau_minus))

    def reset(self) -> None:
        self.x_pre.fill(0)
        self.x_post.fill(0)

    def update(self, arrivals: np.ndarray, post_spikes: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Return multiplicative-bounded dW (same shape as W)."""
        self.x_pre *= self.decay_pre
        self.x_post *= self.decay_post

        dW = np.zeros_like(W)

        # LTD on presynaptic arrivals (uses OLD post trace).
        if arrivals.any():
            dW -= self.A_minus * arrivals * self.x_post[:, None] * W

        # Update pre traces so current arrivals can contribute to LTP.
        self.x_pre += arrivals

        # LTP on postsynaptic spikes (uses NEW pre trace).
        if post_spikes.any():
            post_mask = post_spikes.astype(np.float32)
            dW += self.A_plus * post_mask[:, None] * self.x_pre * (self.w_max - W)

        # Update post trace after computing plasticity.
        self.x_post += post_spikes.astype(np.float32)

        return dW


class LateralEESynapticPlasticity:
    """Pair-based STDP for recurrent/lateral E->E connections (slow)."""

    def __init__(self, n: int, p: Params):
        self.n = n
        self.p = p

        self.x_pre = np.zeros(n, dtype=np.float32)
        self.x_post = np.zeros(n, dtype=np.float32)

        self.decay_pre = math.exp(-p.dt_ms / p.ee_tau_plus)
        self.decay_post = math.exp(-p.dt_ms / p.ee_tau_minus)

    def reset(self):
        self.x_pre.fill(0)
        self.x_post.fill(0)

    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, W: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return dW for E->E weights (same shape as W)."""
        p = self.p

        pre = pre_spikes.astype(np.float32)
        post = post_spikes.astype(np.float32)

        self.x_pre *= self.decay_pre
        self.x_post *= self.decay_post

        dW = np.zeros_like(W)

        if pre.any():
            dW -= p.ee_A_minus * (self.x_post[:, None] * pre[None, :]) * W

        self.x_pre += pre

        if post.any():
            dW += p.ee_A_plus * (post[:, None] * self.x_pre[None, :]) * (p.ee_w_max - W)

        self.x_post += post

        dW *= mask
        return dW


class PVInhibitoryPlasticity:
    """
    Homeostatic inhibitory plasticity for PV->E synapses.

    A minimal iSTDP-inspired rule (Vogels et al., 2011 style):
    - Maintain a postsynaptic (E) trace x_post (decays with tau_pv_istdp).
    - On PV presynaptic spikes, update inhibitory weights:
        w_ij += eta * (x_post_i - rho)

    Where rho is the target mean of x_post corresponding to the desired firing rate.
    """

    def __init__(self, n_post: int, n_pre: int, p: Params):
        self.n_post = n_post
        self.n_pre = n_pre
        self.p = p

        self.x_post = np.zeros(n_post, dtype=np.float32)
        self.decay = math.exp(-p.dt_ms / p.tau_pv_istdp)
        # Target trace value for target firing rate (Hz): E[x_post] ~= r*(tau/1000)
        self.rho = float(p.target_rate_hz * (p.tau_pv_istdp / 1000.0))

    def reset(self):
        self.x_post.fill(0)

    def update(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, W: np.ndarray, mask: np.ndarray) -> None:
        """
        Update postsynaptic trace and PV->E weights in-place.

        pre_spikes: (n_pre,) binary
        post_spikes: (n_post,) binary
        W: (n_post, n_pre) inhibitory weights (conductance increments)
        mask: (n_post, n_pre) bool mask for existing connections
        """
        p = self.p

        # Update postsynaptic trace
        self.x_post *= self.decay
        if post_spikes.any():
            self.x_post += post_spikes.astype(np.float32)

        if not pre_spikes.any():
            return

        # Apply iSTDP update on presynaptic (PV) spikes
        delta_post = p.eta_pv_istdp * (self.x_post - self.rho)  # (n_post,)
        W += (delta_post[:, None] * pre_spikes.astype(np.float32)[None, :]) * mask
        np.clip(W, 0.0, p.w_pv_e_max, out=W)


class RgcLgnV1Network:
    """
    Biologically plausible RGC -> LGN -> V1 network.

    Key biological features:
    1. Izhikevich neurons (TC for LGN, RS for V1 excitatory, FS for PV, LTS for SOM)
    2. Local inhibitory circuits (PV for feedforward, SOM for lateral)
    3. Triplet STDP for plasticity
    4. Optional slow synaptic scaling (disabled by default)
    5. Lateral excitatory connections
    """

    def __init__(self, p: Params, *, init_mode: str = "random"):
        self.p = p
        self.rng = np.random.default_rng(p.seed)

        self.N = p.N
        self.n_lgn = 2 * p.N * p.N  # ON + OFF channels
        self.M = p.M  # Number of V1 ensembles
        self.L = p.delay_max + 1  # Delay buffer length

        # Cortical geometry for lateral connectivity (defaults to a 1×M ring).
        if p.cortex_shape is None:
            cortex_h, cortex_w = 1, int(p.M)
        else:
            cortex_h, cortex_w = int(p.cortex_shape[0]), int(p.cortex_shape[1])
            if cortex_h <= 0 or cortex_w <= 0 or (cortex_h * cortex_w) != int(p.M):
                raise ValueError("cortex_shape must be (H,W) with H*W == M and H,W>0")
        self.cortex_h = cortex_h
        self.cortex_w = cortex_w
        idxs = np.arange(p.M, dtype=np.int32)
        self.cortex_x = (idxs % cortex_w).astype(np.int32)
        self.cortex_y = (idxs // cortex_w).astype(np.int32)
        # Squared distances between ensembles (used by Gaussian lateral kernels).
        dx = np.abs(self.cortex_x[:, None] - self.cortex_x[None, :]).astype(np.int32)
        dy = np.abs(self.cortex_y[:, None] - self.cortex_y[None, :]).astype(np.int32)
        if p.cortex_wrap:
            dx = np.minimum(dx, cortex_w - dx)
            dy = np.minimum(dy, cortex_h - dy)
        self.cortex_dist2 = (dx * dx + dy * dy).astype(np.float32)

        # Spatial coordinates for RGC mosaics (used to sample stimuli and build retinotopic priors).
        xs = np.arange(p.N, dtype=np.float32) - (p.N - 1) / 2.0
        ys = np.arange(p.N, dtype=np.float32) - (p.N - 1) / 2.0
        X0, Y0 = np.meshgrid(xs, ys, indexing="xy")
        X0 = X0.astype(np.float32, copy=False)
        Y0 = Y0.astype(np.float32, copy=False)

        # Real ON and OFF mosaics are distinct lattices (not perfectly co-registered).
        # When enabled, we offset ON and OFF sampling positions and jitter them independently.
        self.rgc_onoff_offset_angle_deg: float | None = None
        if p.rgc_separate_onoff_mosaics:
            mosaic_rng = np.random.default_rng(np.random.SeedSequence([p.seed, 11111, 0]))
            if p.rgc_onoff_offset_angle_deg is None:
                ang = float(mosaic_rng.uniform(0.0, 2.0 * math.pi))
            else:
                ang = float(math.radians(float(p.rgc_onoff_offset_angle_deg)))
            self.rgc_onoff_offset_angle_deg = float((math.degrees(ang)) % 360.0)
            dx = float(p.rgc_onoff_offset) * math.cos(ang)
            dy = float(p.rgc_onoff_offset) * math.sin(ang)
            X_on = (X0 - 0.5 * dx).astype(np.float32, copy=True)
            Y_on = (Y0 - 0.5 * dy).astype(np.float32, copy=True)
            X_off = (X0 + 0.5 * dx).astype(np.float32, copy=True)
            Y_off = (Y0 + 0.5 * dy).astype(np.float32, copy=True)

            if p.rgc_pos_jitter > 0:
                # Important: naive random jitter on a *small* patch can introduce a net shear/dipole,
                # producing systematic orientation biases across the whole network. Enforce 180° rotational
                # antisymmetry so each mosaic remains globally centered while still breaking the lattice.
                j = float(p.rgc_pos_jitter)
                jx = mosaic_rng.uniform(-j, j, size=X_on.shape).astype(np.float32)
                jy = mosaic_rng.uniform(-j, j, size=Y_on.shape).astype(np.float32)
                jx = 0.5 * (jx - jx[::-1, ::-1])
                jy = 0.5 * (jy - jy[::-1, ::-1])
                X_on += jx
                Y_on += jy

                jx = mosaic_rng.uniform(-j, j, size=X_off.shape).astype(np.float32)
                jy = mosaic_rng.uniform(-j, j, size=Y_off.shape).astype(np.float32)
                jx = 0.5 * (jx - jx[::-1, ::-1])
                jy = 0.5 * (jy - jy[::-1, ::-1])
                X_off += jx
                Y_off += jy

            self.X_on, self.Y_on = X_on, Y_on
            self.X_off, self.Y_off = X_off, Y_off
        else:
            X = X0.astype(np.float32, copy=True)
            Y = Y0.astype(np.float32, copy=True)
            # Real RGC mosaics are not perfect grids; small positional jitter reduces lattice biases.
            if p.rgc_pos_jitter > 0:
                # Important: enforce 180° rotational antisymmetry so the mosaic remains globally centered.
                j = float(p.rgc_pos_jitter)
                jx = self.rng.uniform(-j, j, size=X.shape).astype(np.float32)
                jy = self.rng.uniform(-j, j, size=Y.shape).astype(np.float32)
                jx = 0.5 * (jx - jx[::-1, ::-1])
                jy = 0.5 * (jy - jy[::-1, ::-1])
                X += jx
                Y += jy
            self.X_on, self.Y_on = X, Y
            self.X_off, self.Y_off = X.copy(), Y.copy()

        # Backwards-compat aliases: many helper functions use `self.X/self.Y` to mean "the RGC sampling lattice".
        # In separated-mosaic mode, these refer to the ON mosaic.
        self.X, self.Y = self.X_on, self.Y_on

        # RGC center–surround DoG front-end.
        # Important for biological plausibility AND for avoiding orientation-biased learning: small patches
        # with truncated DoG kernels can introduce systematic oblique biases. The default "padded_fft"
        # implementation filters on a padded field so the central patch sees an approximately translation-
        # invariant DoG response.
        self.rgc_dog_on = None   # (N^2,N^2) matrix for legacy "matrix" mode (ON mosaic)
        self.rgc_dog_off = None  # (N^2,N^2) matrix for legacy "matrix" mode (OFF mosaic)
        self._rgc_pad = 0
        self._X_pad = None
        self._Y_pad = None
        self._rgc_dog_fft = None  # rfft2 kernel for padded DoG
        # Bilinear samplers from padded fields -> RGC mosaics.
        self._rgc_on_sample_idx00 = None
        self._rgc_on_sample_idx10 = None
        self._rgc_on_sample_idx01 = None
        self._rgc_on_sample_idx11 = None
        self._rgc_on_sample_wx = None
        self._rgc_on_sample_wy = None
        self._rgc_off_sample_idx00 = None
        self._rgc_off_sample_idx10 = None
        self._rgc_off_sample_idx01 = None
        self._rgc_off_sample_idx11 = None
        self._rgc_off_sample_wx = None
        self._rgc_off_sample_wy = None
        self._init_rgc_frontend()

        # RGC->LGN pooling matrix and optional temporal smoothing of retinogeniculate drive.
        self.W_rgc_lgn = self._build_rgc_lgn_pool_matrix()
        self._lgn_rgc_drive = np.zeros(self.n_lgn, dtype=np.float32)
        self._lgn_rgc_alpha = 0.0
        if p.lgn_rgc_tau_ms > 0:
            self._lgn_rgc_alpha = float(1.0 - math.exp(-p.dt_ms / max(1e-6, float(p.lgn_rgc_tau_ms))))

        # Optional RGC temporal dynamics (local, per-pixel).
        # Implemented as a simple biphasic temporal filter (fast - slow) and an optional
        # absolute refractory period. Disabled by default to preserve prior behavior.
        self._rgc_drive_fast_on = None
        self._rgc_drive_slow_on = None
        self._rgc_drive_fast_off = None
        self._rgc_drive_slow_off = None
        self._rgc_alpha_fast = 0.0
        self._rgc_alpha_slow = 0.0
        if p.rgc_temporal_filter:
            self._rgc_drive_fast_on = np.zeros((p.N, p.N), dtype=np.float32)
            self._rgc_drive_slow_on = np.zeros((p.N, p.N), dtype=np.float32)
            self._rgc_drive_fast_off = np.zeros((p.N, p.N), dtype=np.float32)
            self._rgc_drive_slow_off = np.zeros((p.N, p.N), dtype=np.float32)
            self._rgc_alpha_fast = float(1.0 - math.exp(-p.dt_ms / max(1e-6, float(p.rgc_tau_fast))))
            self._rgc_alpha_slow = float(1.0 - math.exp(-p.dt_ms / max(1e-6, float(p.rgc_tau_slow))))

        self._rgc_refr_steps = 0
        self._rgc_refr_on = None
        self._rgc_refr_off = None
        if float(p.rgc_refractory_ms) > 0.0:
            self._rgc_refr_steps = int(math.ceil(float(p.rgc_refractory_ms) / max(1e-6, float(p.dt_ms))))
            self._rgc_refr_steps = int(max(1, self._rgc_refr_steps))
            self._rgc_refr_on = np.zeros((p.N, p.N), dtype=np.int16)
            self._rgc_refr_off = np.zeros((p.N, p.N), dtype=np.int16)

        # Retinotopic envelopes (fixed structural locality) for thalamocortical projections.
        # Implemented as a spatially varying *cap* on synaptic weights (far inputs cannot become strong).
        d2_on = (self.X_on.astype(np.float32) ** 2 + self.Y_on.astype(np.float32) ** 2).astype(np.float32)
        d2_off = (self.X_off.astype(np.float32) ** 2 + self.Y_off.astype(np.float32) ** 2).astype(np.float32)

        def lgn_mask_vec(sigma: float) -> np.ndarray:
            if sigma <= 0:
                return np.ones(self.n_lgn, dtype=np.float32)
            pix_on = np.exp(-d2_on / (2.0 * float(sigma) * float(sigma))).astype(np.float32).ravel()
            pix_off = np.exp(-d2_off / (2.0 * float(sigma) * float(sigma))).astype(np.float32).ravel()
            vec = np.concatenate([pix_on, pix_off]).astype(np.float32)
            vec /= float(vec.max() + 1e-12)
            return vec

        self._lgn_mask_e_vec = lgn_mask_vec(p.lgn_sigma_e)
        self._lgn_mask_pv_vec = lgn_mask_vec(p.lgn_sigma_pv)
        self._lgn_mask_pp_vec = lgn_mask_vec(p.lgn_sigma_pp)

        # --- LGN Layer (Thalamocortical neurons) ---
        self.lgn = IzhikevichPopulation(self.n_lgn, TC_PARAMS, p.dt_ms, self.rng)

        # --- V1 Excitatory Layer (Regular spiking) ---
        self.v1_exc = IzhikevichPopulation(p.M, RS_PARAMS, p.dt_ms, self.rng)

        # --- Optional L2/3 Excitatory Layer (Regular spiking) ---
        # Enabled via `Params.laminar_enabled`. This population is driven by L4 and is where
        # apical/feedback-like modulation is applied (see step()).
        self.v1_l23 = None
        if p.laminar_enabled:
            l23_rng = np.random.default_rng(np.random.SeedSequence([p.seed, 33333, 0]))
            self.v1_l23 = IzhikevichPopulation(p.M, RS_PARAMS, p.dt_ms, l23_rng)

        # --- Local PV Interneurons (Fast spiking) ---
        # One PV per ensemble for local feedforward inhibition
        self.n_pv = p.M * p.n_pv_per_ensemble
        self.pv = IzhikevichPopulation(self.n_pv, FS_PARAMS, p.dt_ms, self.rng)

        # --- Push–pull interneurons (Fast spiking) ---
        # A dedicated inhibitory pathway driven by ON/OFF-swapped thalamic input.
        self.n_pp = p.M * p.n_pp_per_ensemble
        # Use a separate RNG so adding this population does not change the initialization
        # of other components (delays/weights) for a fixed seed.
        pp_rng = np.random.default_rng(np.random.SeedSequence([p.seed, 12345, 0]))
        self.pp = IzhikevichPopulation(self.n_pp, FS_PARAMS, p.dt_ms, pp_rng)

        # --- SOM Interneurons (Low-threshold spiking) ---
        # Each ensemble has its own SOM neuron for lateral inhibition
        self.n_som = p.M * p.n_som_per_ensemble
        self.som = IzhikevichPopulation(self.n_som, LTS_PARAMS, p.dt_ms, self.rng)

        # --- VIP Interneurons (disinhibitory: VIP -> SOM -> E), optional ---
        self.n_vip = p.M * p.n_vip_per_ensemble
        self.vip = None
        if self.n_vip > 0:
            vip_rng = np.random.default_rng(np.random.SeedSequence([p.seed, 22222, 0]))
            self.vip = IzhikevichPopulation(self.n_vip, RS_PARAMS, p.dt_ms, vip_rng)

        # --- Synaptic currents / conductances ---
        self.I_lgn = np.zeros(self.n_lgn, dtype=np.float32)
        self.g_v1_exc = np.zeros(p.M, dtype=np.float32)  # basal excitatory AMPA conductance onto V1 E
        self.g_v1_apical = np.zeros(p.M, dtype=np.float32)  # apical/feedback-like excitatory conductance
        # L2/3 excitatory (optional, laminar mode). These are inert when `v1_l23 is None`.
        self.g_l23_exc = np.zeros(p.M, dtype=np.float32)
        self.g_l23_apical = np.zeros(p.M, dtype=np.float32)
        self.g_l23_inh_som = np.zeros(p.M, dtype=np.float32)
        self.I_l23_bias = np.zeros(p.M, dtype=np.float32)
        self.I_pv = np.zeros(self.n_pv, dtype=np.float32)
        self.I_pp = np.zeros(self.n_pp, dtype=np.float32)
        self.I_som = np.zeros(self.n_som, dtype=np.float32)
        self.I_som_inh = np.zeros(self.n_som, dtype=np.float32)  # VIP->SOM inhibition (current-based)
        self.I_vip = np.zeros(self.n_vip, dtype=np.float32)

        # Intrinsic excitability homeostasis (bias current) for V1 excitatory neurons
        self.I_v1_bias = np.full(p.M, p.v1_bias_init, dtype=np.float32)

        # Thalamocortical STP state (LGN->E): available resources per synapse (1 = fully recovered).
        self.tc_stp_x = None
        self.tc_stp_rec_alpha = 0.0
        if p.tc_stp_enabled and p.tc_stp_tau_rec > 0:
            self.tc_stp_x = np.ones((p.M, self.n_lgn), dtype=np.float32)
            self.tc_stp_rec_alpha = float(1.0 - math.exp(-p.dt_ms / float(p.tc_stp_tau_rec)))

        # Thalamocortical STP state (LGN->PV): available resources per synapse (1 = fully recovered).
        self.tc_stp_x_pv = None
        self.tc_stp_rec_alpha_pv = 0.0
        if p.tc_stp_pv_enabled and p.tc_stp_pv_tau_rec > 0:
            self.tc_stp_x_pv = np.ones((self.n_pv, self.n_lgn), dtype=np.float32)
            self.tc_stp_rec_alpha_pv = float(1.0 - math.exp(-p.dt_ms / float(p.tc_stp_pv_tau_rec)))

        # Synaptic decays
        self.decay_ampa = math.exp(-p.dt_ms / p.tau_ampa)
        self.decay_ampa_pp = math.exp(-p.dt_ms / max(1e-3, p.tau_ampa_pp))
        self.decay_gaba = math.exp(-p.dt_ms / p.tau_gaba)
        self.decay_gaba_rise_pv = math.exp(-p.dt_ms / max(1e-3, p.tau_gaba_rise_pv))
        self.decay_apical = math.exp(-p.dt_ms / max(1e-3, p.tau_apical))

        # Inhibitory conductances onto V1 excitatory neurons.
        # PV inhibition uses a difference-of-exponentials (rise + decay) to avoid unrealistically
        # zero-lag inhibition in a discrete-time update.
        self.g_v1_inh_pv_rise = np.zeros(p.M, dtype=np.float32)
        self.g_v1_inh_pv_decay = np.zeros(p.M, dtype=np.float32)
        self.g_v1_inh_som = np.zeros(p.M, dtype=np.float32)
        self.g_v1_inh_pp = np.zeros(p.M, dtype=np.float32)  # push–pull (LGN-driven) inhibition

        # Previous-step spikes (for delayed recurrent effects)
        self.prev_v1_spk = np.zeros(p.M, dtype=np.uint8)
        self.prev_v1_l23_spk = np.zeros(p.M, dtype=np.uint8)

        # --- Delay buffer for LGN->V1 ---
        self.delay_buf = np.zeros((self.L, self.n_lgn), dtype=np.uint8)
        self.ptr = 0
        self.lgn_ids = np.arange(self.n_lgn)[None, :]

        # Random delays (no orientation bias)
        self.D = self.rng.integers(0, self.L, size=(p.M, self.n_lgn),
                                   endpoint=False, dtype=np.int16)

        # --- LGN->V1 weights (unbiased initialization) ---
        if init_mode == "random":
            W = self.rng.normal(p.w_init_mean, p.w_init_std,
                               size=(p.M, self.n_lgn)).astype(np.float32)
        elif init_mode == "near_uniform":
            W = (p.w_init_mean + self.rng.normal(0, p.w_init_std * 0.05,
                                                  size=(p.M, self.n_lgn))).astype(np.float32)
        else:
            raise ValueError("init_mode must be 'random' or 'near_uniform'")

        self.W = np.clip(W, 0.0, p.w_max)

        # Structural retinotopic caps for thalamocortical weights.
        self.lgn_mask_e = np.tile(self._lgn_mask_e_vec[None, :], (p.M, 1)).astype(np.float32)
        self.lgn_mask_pv = np.tile(self._lgn_mask_pv_vec[None, :], (self.n_pv, 1)).astype(np.float32)
        self.lgn_mask_pp = np.tile(self._lgn_mask_pp_vec[None, :], (self.n_pp, 1)).astype(np.float32)

        def sample_tc_mask(n_post: int, frac: float, mask_vec: np.ndarray, seed_tag: int) -> np.ndarray:
            frac = float(frac)
            if not (0.0 < frac <= 1.0):
                raise ValueError("tc_conn_fraction_* must be in (0, 1]")
            if frac >= 1.0:
                return np.ones((n_post, self.n_lgn), dtype=bool)
            tc_rng = np.random.default_rng(np.random.SeedSequence([p.seed, 54321, seed_tag]))
            mask = np.zeros((n_post, self.n_lgn), dtype=bool)
            n_keep = int(round(frac * float(self.n_lgn)))
            n_keep = int(max(1, min(self.n_lgn, n_keep)))
            n_pix = p.N * p.N
            if p.tc_conn_balance_onoff:
                n_on = int(min(n_pix, n_keep // 2))
                n_off = int(min(n_pix, n_keep - n_on))
                prob_pix = mask_vec[:n_pix].astype(np.float64, copy=True)
                prob_pix /= float(prob_pix.sum() + 1e-12)
                for i in range(n_post):
                    if n_on > 0:
                        on_idx = tc_rng.choice(n_pix, size=n_on, replace=False, p=prob_pix)
                        mask[i, on_idx] = True
                    if n_off > 0:
                        off_idx = tc_rng.choice(n_pix, size=n_off, replace=False, p=prob_pix)
                        mask[i, n_pix + off_idx] = True
            else:
                prob = mask_vec.astype(np.float64, copy=True)
                prob /= float(prob.sum() + 1e-12)
                for i in range(n_post):
                    idxs = tc_rng.choice(self.n_lgn, size=n_keep, replace=False, p=prob)
                    mask[i, idxs] = True
            return mask

        # Structural sparsity masks for thalamocortical connectivity (anatomical priors).
        self.tc_mask_e = sample_tc_mask(p.M, p.tc_conn_fraction_e, self._lgn_mask_e_vec, seed_tag=1)
        self.tc_mask_pv = sample_tc_mask(self.n_pv, p.tc_conn_fraction_pv, self._lgn_mask_pv_vec, seed_tag=2)
        self.tc_mask_pp = sample_tc_mask(self.n_pp, p.tc_conn_fraction_pp, self._lgn_mask_pp_vec, seed_tag=3)
        self.tc_mask_e_f32 = self.tc_mask_e.astype(np.float32)
        self.tc_mask_pv_f32 = self.tc_mask_pv.astype(np.float32)
        self.tc_mask_pp_f32 = self.tc_mask_pp.astype(np.float32)
        np.minimum(self.W, p.w_max * self.lgn_mask_e, out=self.W)
        self.W *= self.tc_mask_e_f32

        n_pix = p.N * p.N
        # Targets for ON/OFF "split constraint" scaling (local per-neuron resource pools).
        self.split_target_on = self.W[:, :n_pix].sum(axis=1).astype(np.float32)
        self.split_target_off = self.W[:, n_pix:].sum(axis=1).astype(np.float32)
        if p.split_constraint_equalize_onoff:
            tgt = 0.5 * (self.split_target_on + self.split_target_off)
            self.split_target_on = tgt.astype(np.float32, copy=False)
            self.split_target_off = tgt.astype(np.float32, copy=False)

        # Nearest-neighbor ON↔OFF matching based on mosaic coordinates.
        # Used by developmental ON/OFF competition and by the optional PP ON/OFF swap.
        on_pos = np.stack([self.X_on.ravel(), self.Y_on.ravel()], axis=1).astype(np.float32, copy=False)
        off_pos = np.stack([self.X_off.ravel(), self.Y_off.ravel()], axis=1).astype(np.float32, copy=False)
        d2_onoff = np.square(on_pos[:, None, :] - off_pos[None, :, :]).sum(axis=2)
        self.on_to_off = np.argmin(d2_onoff, axis=1).astype(np.int32, copy=False)
        self.off_to_on = np.argmin(d2_onoff, axis=0).astype(np.int32, copy=False)

        self.pp_swap_idx = np.empty(2 * n_pix, dtype=np.int32)
        self.pp_swap_idx[:n_pix] = n_pix + self.on_to_off
        self.pp_swap_idx[n_pix:] = self.off_to_on

        # --- Push–pull thalamic weights (LGN -> PP interneurons) ---
        # Bias initial LGN->PP weights to be correlated with LGN->E weights (same retinotopic pool),
        # while still allowing independent variability. Push–pull phase opposition is implemented by
        # swapping ON/OFF *inputs* into PP (see step()).
        W_pp_rand = self.rng.normal(p.pp_w_init_mean, p.pp_w_init_std,
                                    size=(self.n_pp, self.n_lgn)).astype(np.float32)
        parent_pp = np.repeat(np.arange(p.M, dtype=np.int32), p.n_pp_per_ensemble)
        W_pp_base = self.W[parent_pp].astype(np.float32)
        scale = float(p.pp_w_init_mean / max(1e-6, p.w_init_mean))
        W_pp = (1.0 - p.pp_w_match) * W_pp_rand + p.pp_w_match * (scale * W_pp_base)
        self.W_pp = np.clip(W_pp, 0.0, p.pp_w_max)
        np.minimum(self.W_pp, p.pp_w_max * self.lgn_mask_pp, out=self.W_pp)
        self.W_pp *= self.tc_mask_pp_f32

        # --- LGN->PV feedforward weights (thalamocortical drive to FS interneurons) ---
        # PV thalamic drive is initialized broad/dense (can be weakly tuned via learning elsewhere).
        W_lgn_pv = self.rng.normal(p.w_lgn_pv_init_mean, p.w_lgn_pv_init_std,
                                   size=(self.n_pv, self.n_lgn)).astype(np.float32)
        self.W_lgn_pv = np.clip(W_lgn_pv, 0.0, p.w_max)
        np.minimum(self.W_lgn_pv, p.w_max * self.lgn_mask_pv, out=self.W_lgn_pv)
        self.W_lgn_pv *= self.tc_mask_pv_f32

        # Delays for LGN->PV. By default, inherit the parent ensemble's delays (keeps timing aligned).
        self.D_pv = np.zeros((self.n_pv, self.n_lgn), dtype=np.int16)
        for pv_idx in range(self.n_pv):
            parent = pv_idx // p.n_pv_per_ensemble
            self.D_pv[pv_idx, :] = self.D[parent, :]

        # Delays for LGN->PP. By default, inherit the parent ensemble's delays (keeps timing aligned).
        self.D_pp = np.zeros((self.n_pp, self.n_lgn), dtype=np.int16)
        for pp_idx in range(self.n_pp):
            parent = pp_idx // p.n_pp_per_ensemble
            self.D_pp[pp_idx, :] = self.D[parent, :]

        # --- Local inhibitory connectivity ---
        pv_parent = (np.arange(self.n_pv, dtype=np.int32) // max(1, int(p.n_pv_per_ensemble))).astype(np.int32, copy=False)

        # E->PV connectivity (local-to-nearby by default; sigma=0 recovers legacy private wiring).
        if float(p.pv_in_sigma) <= 0.0:
            self.W_e_pv = np.zeros((self.n_pv, p.M), dtype=np.float32)
            for m in range(p.M):
                pv_start = m * p.n_pv_per_ensemble
                pv_end = pv_start + p.n_pv_per_ensemble
                self.W_e_pv[pv_start:pv_end, m] = p.w_e_pv
        else:
            sig = float(p.pv_in_sigma)
            d2_pv_e = self.cortex_dist2[pv_parent, :].astype(np.float32, copy=False)  # (n_pv, M)
            k = np.exp(-d2_pv_e / (2.0 * sig * sig)).astype(np.float32)
            k_sum = k.sum(axis=1, keepdims=True) + 1e-12
            self.W_e_pv = (float(p.w_e_pv) * (k / k_sum)).astype(np.float32, copy=False)

        # PP->E connectivity (local push–pull inhibition)
        self.W_pp_e = np.zeros((p.M, self.n_pp), dtype=np.float32)
        for m in range(p.M):
            pp_start = m * p.n_pp_per_ensemble
            pp_end = pp_start + p.n_pp_per_ensemble
            # Divide by n_pp_per_ensemble so total inhibition per ensemble stays comparable.
            self.W_pp_e[m, pp_start:pp_end] = p.w_pushpull / max(1, p.n_pp_per_ensemble)

        # PV->E connectivity (local-to-nearby by default; sigma=0 recovers legacy private wiring).
        if float(p.pv_out_sigma) <= 0.0:
            self.W_pv_e = np.zeros((p.M, self.n_pv), dtype=np.float32)
            for m in range(p.M):
                pv_start = m * p.n_pv_per_ensemble
                pv_end = pv_start + p.n_pv_per_ensemble
                self.W_pv_e[m, pv_start:pv_end] = p.w_pv_e
        else:
            sig = float(p.pv_out_sigma)
            d2_e_pv = self.cortex_dist2[:, pv_parent].astype(np.float32, copy=False)  # (M, n_pv)
            k = np.exp(-d2_e_pv / (2.0 * sig * sig)).astype(np.float32)
            k_sum = k.sum(axis=1, keepdims=True) + 1e-12
            target_total = float(p.w_pv_e) * float(max(1, int(p.n_pv_per_ensemble)))
            self.W_pv_e = (target_total * (k / k_sum)).astype(np.float32, copy=False)
        self.mask_pv_e = (self.W_pv_e > 0)

        # PV<->PV coupling (optional).
        self.W_pv_pv = None
        self.I_pv_inh = np.zeros(self.n_pv, dtype=np.float32)
        if (float(p.pv_pv_sigma) > 0.0) and (float(p.w_pv_pv) > 0.0):
            sig = float(p.pv_pv_sigma)
            d2_pv_pv = self.cortex_dist2[pv_parent[:, None], pv_parent[None, :]].astype(np.float32, copy=False)
            k = np.exp(-d2_pv_pv / (2.0 * sig * sig)).astype(np.float32)
            np.fill_diagonal(k, 0.0)
            k_sum = k.sum(axis=1, keepdims=True) + 1e-12
            self.W_pv_pv = (float(p.w_pv_pv) * (k / k_sum)).astype(np.float32, copy=False)

        # E->SOM connectivity (can be long-range): E activity recruits SOM near the target site,
        # producing disynaptic long-range suppression without literal long-range inhibitory axons.
        self.W_e_som = np.zeros((self.n_som, p.M), dtype=np.float32)
        for som_idx in range(self.n_som):
            m = som_idx // p.n_som_per_ensemble
            kernel = np.zeros(p.M, dtype=np.float32)
            for pre in range(p.M):
                d2 = float(self.cortex_dist2[pre, m])
                kernel[pre] = math.exp(-d2 / (2.0 * (p.som_in_sigma ** 2)))
            kernel /= float(kernel.sum() + 1e-12)
            self.W_e_som[som_idx, :] = p.w_e_som * kernel

        # SOM->E connectivity (local): SOM inhibits nearby excitatory neurons.
        self.W_som_e = np.zeros((p.M, self.n_som), dtype=np.float32)
        for som_idx in range(self.n_som):
            m = som_idx // p.n_som_per_ensemble
            kernel = np.zeros(p.M, dtype=np.float32)
            for post in range(p.M):
                d2 = float(self.cortex_dist2[post, m])
                kernel[post] = math.exp(-d2 / (2.0 * (p.som_out_sigma ** 2)))
            if not p.som_self_inhibit:
                kernel[m] = 0.0
            kernel /= float(kernel.sum() + 1e-12)
            self.W_som_e[:, som_idx] = p.w_som_e * kernel

        # VIP connectivity (local disinhibition): E -> VIP -> SOM.
        self.W_e_vip = np.zeros((self.n_vip, p.M), dtype=np.float32)
        self.W_vip_som = np.zeros((self.n_som, self.n_vip), dtype=np.float32)
        if self.n_vip > 0:
            for m in range(p.M):
                vip_start = m * p.n_vip_per_ensemble
                vip_end = vip_start + p.n_vip_per_ensemble
                self.W_e_vip[vip_start:vip_end, m] = p.w_e_vip
                som_start = m * p.n_som_per_ensemble
                som_end = som_start + p.n_som_per_ensemble
                if p.w_vip_som != 0.0:
                    self.W_vip_som[som_start:som_end, vip_start:vip_end] = float(p.w_vip_som) / max(
                        1, int(p.n_vip_per_ensemble)
                    )

        # --- Lateral excitatory connectivity ---
        # Gaussian connectivity based on cortical distance (1×M ring by default).
        self.W_e_e = np.zeros((p.M, p.M), dtype=np.float32)
        for i in range(p.M):
            for j in range(p.M):
                if i != j:
                    d2 = float(self.cortex_dist2[i, j])
                    self.W_e_e[i, j] = p.w_e_e_lateral * math.exp(-d2 / (2.0 * p.lateral_sigma**2))
        # Allow plasticity on all off-diagonal connections (structural plasticity can grow weights from 0).
        self.mask_e_e = np.ones((p.M, p.M), dtype=bool)
        np.fill_diagonal(self.mask_e_e, False)

        # --- Laminar (L4 -> L2/3) connectivity (optional) ---
        # Implemented as a fixed Gaussian kernel on the same cortical geometry used for lateral E->E.
        self.W_l4_l23 = None
        if p.laminar_enabled:
            self.W_l4_l23 = np.zeros((p.M, p.M), dtype=np.float32)
            sig = float(p.l4_l23_sigma)
            if sig <= 0.0:
                np.fill_diagonal(self.W_l4_l23, float(p.w_l4_l23))
            else:
                for post in range(p.M):
                    kernel = np.exp(-self.cortex_dist2[:, post] / (2.0 * sig * sig)).astype(np.float32)
                    kernel /= float(kernel.sum() + 1e-12)
                    self.W_l4_l23[post, :] = float(p.w_l4_l23) * kernel

        # --- Plasticity mechanisms ---
        self.stdp = TripletSTDP(
            self.n_lgn,
            p.M,
            p,
            self.rng,
            split_on_to_off=self.on_to_off,
            split_off_to_on=self.off_to_on,
        )
        self.pp_stdp = PairSTDP(
            self.n_lgn,
            self.n_pp,
            dt_ms=p.dt_ms,
            tau_plus=p.tau_plus,
            tau_minus=p.tau_minus,
            A_plus=p.pp_A_plus,
            A_minus=p.pp_A_minus,
            w_max=p.pp_w_max,
        )
        self.homeostasis = HomeostaticScaling(p.M, p)
        self.pv_istdp = PVInhibitoryPlasticity(p.M, self.n_pv, p)
        self.ee_stdp = LateralEESynapticPlasticity(p.M, p)

    def _init_rgc_frontend(self) -> None:
        """Initialize the RGC center–surround front-end (DoG)."""
        p = self.p
        if not p.rgc_center_surround:
            return
        impl = str(p.rgc_dog_impl).lower()
        if impl == "matrix":
            self.rgc_dog_on = self._build_rgc_dog_filter_matrix(self.X_on, self.Y_on)
            self.rgc_dog_off = self._build_rgc_dog_filter_matrix(self.X_off, self.Y_off)
            return
        if impl == "padded_fft":
            self._setup_rgc_dog_padded_fft()
            return
        raise ValueError("rgc_dog_impl must be one of: 'matrix', 'padded_fft'")

    def _build_rgc_dog_filter_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Build an (N^2, N^2) DoG filter using the provided RGC coordinates (legacy mode)."""
        p = self.p
        if p.rgc_center_sigma <= 0.0 or p.rgc_surround_sigma <= 0.0:
            raise ValueError("rgc_center_sigma and rgc_surround_sigma must be > 0 for DoG filtering")
        if p.rgc_surround_sigma <= p.rgc_center_sigma:
            raise ValueError("rgc_surround_sigma must be > rgc_center_sigma for a center–surround DoG")

        x = X.astype(np.float32).ravel()
        y = Y.astype(np.float32).ravel()
        n_pix = int(x.size)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d2 = dx * dx + dy * dy

        sig_c = float(p.rgc_center_sigma)
        sig_s = float(p.rgc_surround_sigma)
        Kc = np.exp(-d2 / (2.0 * sig_c * sig_c)).astype(np.float32)
        Ks = np.exp(-d2 / (2.0 * sig_s * sig_s)).astype(np.float32)

        if p.rgc_surround_balance:
            gain = (Kc.sum(axis=1) / (Ks.sum(axis=1) + 1e-12)).astype(np.float32)
        else:
            gain = np.ones(n_pix, dtype=np.float32)

        dog = (Kc - gain[:, None] * Ks).astype(np.float32)

        norm = str(p.rgc_dog_norm).lower()
        if norm == "none":
            pass
        elif norm == "l1":
            dog /= (np.sum(np.abs(dog), axis=1, keepdims=True).astype(np.float32) + 1e-12)
        elif norm == "l2":
            dog /= (np.sqrt(np.sum(dog * dog, axis=1, keepdims=True)).astype(np.float32) + 1e-12)
        else:
            raise ValueError("rgc_dog_norm must be one of: 'none', 'l1', 'l2'")

        return dog

    def _setup_rgc_dog_padded_fft(self) -> None:
        """Precompute a padded FFT DoG kernel and a sampler for the central RGC mosaic."""
        p = self.p
        if p.rgc_center_sigma <= 0.0 or p.rgc_surround_sigma <= 0.0:
            raise ValueError("rgc_center_sigma and rgc_surround_sigma must be > 0 for DoG filtering")
        if p.rgc_surround_sigma <= p.rgc_center_sigma:
            raise ValueError("rgc_surround_sigma must be > rgc_center_sigma for a center–surround DoG")

        pad = int(p.rgc_dog_pad)
        if pad <= 0:
            # 3σ surround captures >99% of mass; keep a minimum so even small sigmas are stable.
            pad = max(4, int(math.ceil(3.0 * float(p.rgc_surround_sigma))))
        self._rgc_pad = pad

        n = int(p.N + 2 * pad)
        xs = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0).astype(np.float32)
        ys = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0).astype(np.float32)
        self._X_pad, self._Y_pad = np.meshgrid(xs, ys, indexing="xy")

        # Periodic (circular) kernels on the padded grid. With sufficient padding, wrap-around terms
        # are negligible for the central crop, approximating an infinite-plane convolution.
        ax = np.arange(n, dtype=np.float32)
        d = np.minimum(ax, n - ax).astype(np.float32)
        DX, DY = np.meshgrid(d, d, indexing="xy")
        d2 = (DX * DX + DY * DY).astype(np.float32)

        sig_c = float(p.rgc_center_sigma)
        sig_s = float(p.rgc_surround_sigma)
        Kc = np.exp(-d2 / (2.0 * sig_c * sig_c)).astype(np.float32)
        Ks = np.exp(-d2 / (2.0 * sig_s * sig_s)).astype(np.float32)
        Kc /= float(Kc.sum() + 1e-12)
        Ks /= float(Ks.sum() + 1e-12)

        if p.rgc_surround_balance:
            gain = float(Kc.sum() / (Ks.sum() + 1e-12))  # ~=1.0 after normalization
        else:
            gain = 1.0

        dog = (Kc - gain * Ks).astype(np.float32)

        norm = str(p.rgc_dog_norm).lower()
        if norm == "none":
            pass
        elif norm == "l1":
            dog /= float(np.sum(np.abs(dog)) + 1e-12)
        elif norm == "l2":
            dog /= float(np.sqrt(np.sum(dog * dog)) + 1e-12)
        else:
            raise ValueError("rgc_dog_norm must be one of: 'none', 'l1', 'l2'")

        self._rgc_dog_fft = np.fft.rfft2(dog.astype(np.float32, copy=False))

        def build_sampler(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            fx = X.astype(np.float32).ravel() + (n - 1) / 2.0
            fy = Y.astype(np.float32).ravel() + (n - 1) / 2.0

            x0 = np.floor(fx).astype(np.int32)
            y0 = np.floor(fy).astype(np.int32)
            wx = (fx - x0).astype(np.float32)
            wy = (fy - y0).astype(np.float32)
            x1 = x0 + 1
            y1 = y0 + 1

            x0 = np.clip(x0, 0, n - 1)
            y0 = np.clip(y0, 0, n - 1)
            x1 = np.clip(x1, 0, n - 1)
            y1 = np.clip(y1, 0, n - 1)

            idx00 = (y0 * n + x0).astype(np.int32)
            idx10 = (y0 * n + x1).astype(np.int32)
            idx01 = (y1 * n + x0).astype(np.int32)
            idx11 = (y1 * n + x1).astype(np.int32)
            return idx00, idx10, idx01, idx11, wx, wy

        (
            self._rgc_on_sample_idx00,
            self._rgc_on_sample_idx10,
            self._rgc_on_sample_idx01,
            self._rgc_on_sample_idx11,
            self._rgc_on_sample_wx,
            self._rgc_on_sample_wy,
        ) = build_sampler(self.X_on, self.Y_on)

        (
            self._rgc_off_sample_idx00,
            self._rgc_off_sample_idx10,
            self._rgc_off_sample_idx01,
            self._rgc_off_sample_idx11,
            self._rgc_off_sample_wx,
            self._rgc_off_sample_wy,
        ) = build_sampler(self.X_off, self.Y_off)

    def _build_rgc_lgn_pool_matrix(self) -> np.ndarray:
        """Build retinogeniculate pooling matrix (same-sign center + opponent surround)."""
        p = self.p
        n_pix = int(p.N) * int(p.N)
        n_lgn = 2 * n_pix

        if not p.lgn_pooling:
            return np.eye(n_lgn, dtype=np.float32)

        X_on = self.X_on.astype(np.float32).ravel()
        Y_on = self.Y_on.astype(np.float32).ravel()
        X_off = self.X_off.astype(np.float32).ravel()
        Y_off = self.Y_off.astype(np.float32).ravel()

        dx = X_on[:, None] - X_on[None, :]
        dy = Y_on[:, None] - Y_on[None, :]
        d2_on = (dx * dx + dy * dy).astype(np.float32)

        dx = X_off[:, None] - X_off[None, :]
        dy = Y_off[:, None] - Y_off[None, :]
        d2_off = (dx * dx + dy * dy).astype(np.float32)

        dx = X_on[:, None] - X_off[None, :]
        dy = Y_on[:, None] - Y_off[None, :]
        d2_onoff = (dx * dx + dy * dy).astype(np.float32)

        dx = X_off[:, None] - X_on[None, :]
        dy = Y_off[:, None] - Y_on[None, :]
        d2_offon = (dx * dx + dy * dy).astype(np.float32)

        sig_c = max(1e-6, float(p.lgn_pool_sigma_center))
        sig_s = max(1e-6, float(p.lgn_pool_sigma_surround))
        same_on = np.exp(-d2_on / (2.0 * sig_c * sig_c)).astype(np.float32)
        same_off = np.exp(-d2_off / (2.0 * sig_c * sig_c)).astype(np.float32)
        opp_onoff = np.exp(-d2_onoff / (2.0 * sig_s * sig_s)).astype(np.float32)
        opp_offon = np.exp(-d2_offon / (2.0 * sig_s * sig_s)).astype(np.float32)

        same_on /= (same_on.sum(axis=1, keepdims=True) + 1e-12)
        same_off /= (same_off.sum(axis=1, keepdims=True) + 1e-12)
        opp_onoff /= (opp_onoff.sum(axis=1, keepdims=True) + 1e-12)
        opp_offon /= (opp_offon.sum(axis=1, keepdims=True) + 1e-12)

        w_same = float(p.lgn_pool_same_gain)
        w_opp = float(p.lgn_pool_opponent_gain)
        mat = np.zeros((n_lgn, n_lgn), dtype=np.float32)
        # ON relay: ON-center pool, OFF opponent surround
        mat[:n_pix, :n_pix] = w_same * same_on
        mat[:n_pix, n_pix:] = -w_opp * opp_onoff
        # OFF relay: OFF-center pool, ON opponent surround
        mat[n_pix:, n_pix:] = w_same * same_off
        mat[n_pix:, :n_pix] = -w_opp * opp_offon
        return mat

    def reset_state(self) -> None:
        """Reset all dynamic state (but not weights)."""
        self.lgn.reset()
        self.v1_exc.reset()
        if self.v1_l23 is not None:
            self.v1_l23.reset()
        self.pv.reset()
        self.pp.reset()
        self.som.reset()
        if self.vip is not None:
            self.vip.reset()

        if self._rgc_drive_fast_on is not None:
            self._rgc_drive_fast_on.fill(0.0)
        if self._rgc_drive_slow_on is not None:
            self._rgc_drive_slow_on.fill(0.0)
        if self._rgc_drive_fast_off is not None:
            self._rgc_drive_fast_off.fill(0.0)
        if self._rgc_drive_slow_off is not None:
            self._rgc_drive_slow_off.fill(0.0)
        if self._rgc_refr_on is not None:
            self._rgc_refr_on.fill(0)
        if self._rgc_refr_off is not None:
            self._rgc_refr_off.fill(0)
        self._lgn_rgc_drive.fill(0.0)

        self.I_lgn.fill(0)
        self.g_v1_exc.fill(0)
        self.g_v1_apical.fill(0)
        self.g_l23_exc.fill(0)
        self.g_l23_apical.fill(0)
        self.g_l23_inh_som.fill(0)
        self.I_l23_bias.fill(0)
        self.I_pv.fill(0)
        self.I_pv_inh.fill(0)
        self.I_pp.fill(0)
        self.I_som.fill(0)
        self.I_som_inh.fill(0)
        if self.I_vip.size:
            self.I_vip.fill(0)
        self.g_v1_inh_pv_rise.fill(0)
        self.g_v1_inh_pv_decay.fill(0)
        self.g_v1_inh_som.fill(0)
        self.g_v1_inh_pp.fill(0)
        self.prev_v1_spk.fill(0)
        self.prev_v1_l23_spk.fill(0)

        self.delay_buf.fill(0)
        self.ptr = 0

        if self.tc_stp_x is not None:
            self.tc_stp_x.fill(1.0)
        if self.tc_stp_x_pv is not None:
            self.tc_stp_x_pv.fill(1.0)

        self.stdp.reset()
        self.pp_stdp.reset()
        self.pv_istdp.reset()
        self.ee_stdp.reset()
        # Note: we don't reset homeostasis to preserve rate estimates

    def grating_on_coords(self, theta_deg: float, t_ms: float, phase: float,
                          X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Generate a drifting grating sampled at arbitrary coordinate arrays X,Y."""
        p = self.p
        th = math.radians(theta_deg)
        coord = X * math.cos(th) + Y * math.sin(th)
        return np.sin(
            2.0 * math.pi * (p.spatial_freq * coord - p.temporal_freq * (t_ms / 1000.0)) + phase
        ).astype(np.float32)

    def grating(self, theta_deg: float, t_ms: float, phase: float) -> np.ndarray:
        """Generate drifting grating stimulus on the RGC mosaic coordinates."""
        return self.grating_on_coords(theta_deg, t_ms, phase, self.X, self.Y)

    def _rgc_sample_from_pad_field(self, field_pad: np.ndarray, *, mosaic: str) -> np.ndarray:
        """Bilinearly sample a padded field to the ON or OFF RGC mosaic."""
        if mosaic == "on":
            idx00 = self._rgc_on_sample_idx00
            idx10 = self._rgc_on_sample_idx10
            idx01 = self._rgc_on_sample_idx01
            idx11 = self._rgc_on_sample_idx11
            wx = self._rgc_on_sample_wx
            wy = self._rgc_on_sample_wy
        elif mosaic == "off":
            idx00 = self._rgc_off_sample_idx00
            idx10 = self._rgc_off_sample_idx10
            idx01 = self._rgc_off_sample_idx01
            idx11 = self._rgc_off_sample_idx11
            wx = self._rgc_off_sample_wx
            wy = self._rgc_off_sample_wy
        else:
            raise ValueError("mosaic must be one of: 'on', 'off'")

        if (
            (idx00 is None)
            or (idx10 is None)
            or (idx01 is None)
            or (idx11 is None)
            or (wx is None)
            or (wy is None)
        ):
            raise RuntimeError("padded sampler not initialized (need rgc_dog_impl='padded_fft')")

        flat = field_pad.ravel()
        v00 = flat[idx00]
        v10 = flat[idx10]
        v01 = flat[idx01]
        v11 = flat[idx11]

        omx = (1.0 - wx).astype(np.float32, copy=False)
        omy = (1.0 - wy).astype(np.float32, copy=False)
        out = (omx * omy) * v00 + (wx * omy) * v10 + (omx * wy) * v01 + (wx * wy) * v11
        return out.reshape(self.N, self.N).astype(np.float32, copy=False)

    def _rgc_drives_from_pad_stimulus(self, stim_pad: np.ndarray, *, contrast: float) -> tuple[np.ndarray, np.ndarray]:
        """Apply padded DoG filtering to a stimulus on the padded grid, then sample to (ON,OFF) mosaics."""
        if self._rgc_dog_fft is None:
            raise RuntimeError("padded_fft DoG front-end not initialized")
        stim_pad = (contrast * stim_pad).astype(np.float32, copy=False)
        dog_pad = np.fft.irfft2(
            np.fft.rfft2(stim_pad) * self._rgc_dog_fft,
            s=stim_pad.shape,
        ).astype(np.float32, copy=False)

        return (
            self._rgc_sample_from_pad_field(dog_pad, mosaic="on"),
            self._rgc_sample_from_pad_field(dog_pad, mosaic="off"),
        )

    def rgc_drives_grating(
        self,
        theta_deg: float,
        t_ms: float,
        phase: float,
        *,
        contrast: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the (ON, OFF) RGC drive fields for a drifting grating."""
        p = self.p

        if not p.rgc_center_surround:
            drive_on = (contrast * self.grating_on_coords(theta_deg, t_ms, phase, self.X_on, self.Y_on)).astype(
                np.float32, copy=False
            )
            drive_off = (contrast * self.grating_on_coords(theta_deg, t_ms, phase, self.X_off, self.Y_off)).astype(
                np.float32, copy=False
            )
            return drive_on, drive_off

        impl = str(p.rgc_dog_impl).lower()
        if impl == "matrix":
            stim_on = self.grating_on_coords(theta_deg, t_ms, phase, self.X_on, self.Y_on)
            stim_off = self.grating_on_coords(theta_deg, t_ms, phase, self.X_off, self.Y_off)
            stim_on = (contrast * stim_on).astype(np.float32, copy=False)
            stim_off = (contrast * stim_off).astype(np.float32, copy=False)
            if self.rgc_dog_on is None or self.rgc_dog_off is None:
                raise RuntimeError("matrix DoG front-end not initialized")
            drive_on = (self.rgc_dog_on @ stim_on.ravel()).reshape(stim_on.shape).astype(np.float32, copy=False)
            drive_off = (self.rgc_dog_off @ stim_off.ravel()).reshape(stim_off.shape).astype(np.float32, copy=False)
            return drive_on, drive_off

        if impl != "padded_fft":
            raise ValueError("rgc_dog_impl must be one of: 'matrix', 'padded_fft'")
        if self._X_pad is None or self._Y_pad is None or self._rgc_dog_fft is None:
            raise RuntimeError("padded_fft DoG front-end not initialized")

        stim_pad = self.grating_on_coords(theta_deg, t_ms, phase, self._X_pad, self._Y_pad)
        return self._rgc_drives_from_pad_stimulus(stim_pad, contrast=contrast)

    def rgc_drive_grating(self, theta_deg: float, t_ms: float, phase: float, *, contrast: float = 1.0) -> np.ndarray:
        """Backwards-compatible single drive accessor (returns ON drive)."""
        drive_on, _ = self.rgc_drives_grating(theta_deg, t_ms, phase, contrast=contrast)
        return drive_on

    def rgc_spikes_from_drives(self, drive_on: np.ndarray, drive_off: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ON and OFF RGC spikes from separate (ON,OFF) drive fields."""
        p = self.p

        drive_on_f = drive_on
        drive_off_f = drive_off
        if p.rgc_temporal_filter:
            # Simple biphasic temporal filtering: fast - slow (per mosaic).
            if (
                (self._rgc_drive_fast_on is None)
                or (self._rgc_drive_slow_on is None)
                or (self._rgc_drive_fast_off is None)
                or (self._rgc_drive_slow_off is None)
            ):
                self._rgc_drive_fast_on = np.zeros_like(drive_on, dtype=np.float32)
                self._rgc_drive_slow_on = np.zeros_like(drive_on, dtype=np.float32)
                self._rgc_drive_fast_off = np.zeros_like(drive_off, dtype=np.float32)
                self._rgc_drive_slow_off = np.zeros_like(drive_off, dtype=np.float32)
                self._rgc_alpha_fast = float(1.0 - math.exp(-p.dt_ms / max(1e-6, float(p.rgc_tau_fast))))
                self._rgc_alpha_slow = float(1.0 - math.exp(-p.dt_ms / max(1e-6, float(p.rgc_tau_slow))))

            self._rgc_drive_fast_on += self._rgc_alpha_fast * (drive_on - self._rgc_drive_fast_on)
            self._rgc_drive_slow_on += self._rgc_alpha_slow * (drive_on - self._rgc_drive_slow_on)
            drive_on_f = float(p.rgc_temporal_gain) * (self._rgc_drive_fast_on - self._rgc_drive_slow_on)

            self._rgc_drive_fast_off += self._rgc_alpha_fast * (drive_off - self._rgc_drive_fast_off)
            self._rgc_drive_slow_off += self._rgc_alpha_slow * (drive_off - self._rgc_drive_slow_off)
            drive_off_f = float(p.rgc_temporal_gain) * (self._rgc_drive_fast_off - self._rgc_drive_slow_off)

        on_rate = p.base_rate + p.gain_rate * np.clip(drive_on_f, 0, None)
        off_rate = p.base_rate + p.gain_rate * np.clip(-drive_off_f, 0, None)
        dt_s = p.dt_ms / 1000.0
        if self._rgc_refr_on is None or self._rgc_refr_off is None:
            on_spk = (self.rng.random(drive_on.shape) < (on_rate * dt_s)).astype(np.uint8)
            off_spk = (self.rng.random(drive_off.shape) < (off_rate * dt_s)).astype(np.uint8)
            return on_spk, off_spk

        # Absolute refractory: suppress spiking for a fixed number of timesteps after a spike.
        np.maximum(self._rgc_refr_on - 1, 0, out=self._rgc_refr_on)
        np.maximum(self._rgc_refr_off - 1, 0, out=self._rgc_refr_off)
        can_on = (self._rgc_refr_on == 0)
        can_off = (self._rgc_refr_off == 0)
        on_spk_b = (self.rng.random(drive_on.shape) < (on_rate * dt_s)) & can_on
        off_spk_b = (self.rng.random(drive_off.shape) < (off_rate * dt_s)) & can_off
        self._rgc_refr_on[on_spk_b] = int(self._rgc_refr_steps)
        self._rgc_refr_off[off_spk_b] = int(self._rgc_refr_steps)
        return on_spk_b.astype(np.uint8), off_spk_b.astype(np.uint8)

    def rgc_spikes_from_drive(self, drive: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backwards-compatible ON/OFF spikes from a single shared drive field."""
        return self.rgc_spikes_from_drives(drive, drive)

    def rgc_spikes_grating(self, theta_deg: float, t_ms: float, phase: float, *,
                           contrast: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ON and OFF RGC spikes for a drifting grating (preferred code path)."""
        drive_on, drive_off = self.rgc_drives_grating(theta_deg, t_ms, phase, contrast=contrast)
        return self.rgc_spikes_from_drives(drive_on, drive_off)

    def rgc_spikes(
        self,
        stim_on: np.ndarray,
        *,
        contrast: float = 1.0,
        stim_off: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ON and OFF RGC spikes from explicit stimulus fields sampled at ON/OFF RGC positions.

        Note: When `rgc_dog_impl='padded_fft'`, gratings should be passed via `rgc_spikes_grating(...)`
        so the DoG can be computed on a padded field (avoids edge-induced orientation bias).
        """
        p = self.p
        if stim_off is None:
            stim_off = stim_on
        stim_on = stim_on.astype(np.float32, copy=False)
        stim_off = stim_off.astype(np.float32, copy=False)

        if not p.rgc_center_surround:
            drive_on = (contrast * stim_on).astype(np.float32, copy=False)
            drive_off = (contrast * stim_off).astype(np.float32, copy=False)
            return self.rgc_spikes_from_drives(drive_on, drive_off)

        impl = str(p.rgc_dog_impl).lower()
        if impl == "matrix":
            if self.rgc_dog_on is None or self.rgc_dog_off is None:
                raise RuntimeError("matrix DoG front-end not initialized")
            drive_on = (self.rgc_dog_on @ (contrast * stim_on).ravel()).reshape(stim_on.shape).astype(np.float32, copy=False)
            drive_off = (self.rgc_dog_off @ (contrast * stim_off).ravel()).reshape(stim_off.shape).astype(np.float32, copy=False)
            return self.rgc_spikes_from_drives(drive_on, drive_off)
        if impl == "padded_fft":
            raise ValueError(
                "rgc_spikes(stim) is ambiguous in padded_fft mode; use rgc_spikes_grating(...) "
                "or the padded stimulus path (_rgc_drives_from_pad_stimulus)"
            )
        raise ValueError("rgc_dog_impl must be one of: 'matrix', 'padded_fft'")

    def step(
        self,
        on_spk: np.ndarray,
        off_spk: np.ndarray,
        plastic: bool,
        *,
        vip_td: float = 0.0,
        apical_drive: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Advance network by one timestep.

        Returns V1 excitatory spikes.
        """
        p = self.p

        # Combine ON/OFF RGC spikes
        rgc = np.concatenate([on_spk.ravel(), off_spk.ravel()]).astype(np.float32)
        rgc_lgn = self.W_rgc_lgn @ rgc
        if self._lgn_rgc_alpha > 0.0:
            self._lgn_rgc_drive += self._lgn_rgc_alpha * (rgc_lgn - self._lgn_rgc_drive)
            rgc_lgn = self._lgn_rgc_drive

        # --- LGN layer ---
        self.I_lgn *= self.decay_ampa
        self.I_lgn += p.w_rgc_lgn * rgc_lgn
        lgn_spk = self.lgn.step(self.I_lgn)
        self.last_lgn_spk = lgn_spk

        # Store LGN spikes in delay buffer
        self.delay_buf[self.ptr, :] = lgn_spk

        # Get delayed LGN spikes arriving at V1
        idx = (self.ptr - self.D) % self.L
        arrivals = self.delay_buf[idx, self.lgn_ids].astype(np.float32)  # (M, n_lgn)
        arrivals_tc = arrivals * self.tc_mask_e_f32

        # --- V1 feedforward input ---
        if self.tc_stp_x is None:
            I_ff = (self.W * arrivals_tc).sum(axis=1)
        else:
            # Recover resources.
            self.tc_stp_x += (1.0 - self.tc_stp_x) * self.tc_stp_rec_alpha
            # Use available resources to scale efficacy for *this* spike arrival.
            arrivals_eff = arrivals_tc * self.tc_stp_x
            I_ff = (self.W * arrivals_eff).sum(axis=1)
            # Deplete after release (local to each synapse).
            if p.tc_stp_u > 0 and arrivals_eff.any():
                self.tc_stp_x -= float(p.tc_stp_u) * arrivals_eff
                np.clip(self.tc_stp_x, 0.0, 1.0, out=self.tc_stp_x)
        self.last_I_ff = I_ff

        # --- V1 excitatory layer (integrate excitatory synaptic conductance) ---
        self.g_v1_exc *= self.decay_ampa
        self.g_v1_exc += p.w_exc_gain * I_ff
        self.g_v1_apical *= self.decay_apical
        # In laminar mode, apical/feedback drive targets L2/3 (handled below).
        if (apical_drive is not None) and (self.v1_l23 is None):
            ap = np.asarray(apical_drive, dtype=np.float32)
            if ap.ndim == 0:
                self.g_v1_apical += p.w_exc_gain * float(ap)
            else:
                if ap.shape != (self.M,):
                    raise ValueError(f"apical_drive must have shape (M,), got {tuple(ap.shape)}")
                self.g_v1_apical += p.w_exc_gain * ap

        # Inhibitory conductances (GABA decay)
        self.g_v1_inh_pv_rise *= self.decay_gaba_rise_pv
        self.g_v1_inh_pv_decay *= self.decay_gaba
        self.g_v1_inh_som *= self.decay_gaba
        self.g_v1_inh_pp *= self.decay_gaba
        self.g_l23_inh_som *= self.decay_gaba

        # --- PV interneurons (feedforward inhibition; must run BEFORE E to be feedforward-in-time) ---
        self.I_pv *= self.decay_ampa
        self.I_pv_inh *= self.decay_gaba
        # Thalamocortical drive to PV (feedforward inhibition)
        idx_pv = (self.ptr - self.D_pv) % self.L
        arrivals_pv = self.delay_buf[idx_pv, self.lgn_ids].astype(np.float32)  # (n_pv, n_lgn)
        arrivals_pv_tc = arrivals_pv * self.tc_mask_pv_f32
        if self.tc_stp_x_pv is None:
            self.I_pv += p.w_lgn_pv_gain * (self.W_lgn_pv * arrivals_pv_tc).sum(axis=1)
        else:
            # Recover resources.
            self.tc_stp_x_pv += (1.0 - self.tc_stp_x_pv) * self.tc_stp_rec_alpha_pv
            # Use available resources to scale efficacy for *this* spike arrival.
            arrivals_pv_eff = arrivals_pv_tc * self.tc_stp_x_pv
            self.I_pv += p.w_lgn_pv_gain * (self.W_lgn_pv * arrivals_pv_eff).sum(axis=1)
            # Deplete after release (local to each synapse).
            if p.tc_stp_pv_u > 0 and arrivals_pv_eff.any():
                self.tc_stp_x_pv -= float(p.tc_stp_pv_u) * arrivals_pv_eff
                np.clip(self.tc_stp_x_pv, 0.0, 1.0, out=self.tc_stp_x_pv)

        # Local recurrent drive from E to PV (feedback component, delayed by one step)
        self.I_pv += self.W_e_pv @ self.prev_v1_spk.astype(np.float32)
        pv_spk = self.pv.step(self.I_pv - self.I_pv_inh)
        self.last_pv_spk = pv_spk

        # PV->PV mutual inhibition (affects next step).
        if self.W_pv_pv is not None:
            self.I_pv_inh += self.W_pv_pv @ pv_spk.astype(np.float32)

        # PV->E inhibition (GABA conductance increment with rise time)
        g_pv_inc = self.W_pv_e @ pv_spk.astype(np.float32)
        self.g_v1_inh_pv_rise += g_pv_inc
        self.g_v1_inh_pv_decay += g_pv_inc

        # --- Push–pull pathway: LGN (ON/OFF swapped) -> PP interneuron -> E inhibition ---
        pp_spk = np.zeros(self.n_pp, dtype=np.uint8)
        arrivals_pp = None
        if (p.w_pushpull > 0) or p.pp_plastic:
            self.I_pp *= self.decay_ampa_pp
            idx_pp = (self.ptr - self.D_pp) % self.L
            arrivals_pp_raw = self.delay_buf[idx_pp, self.lgn_ids].astype(np.float32)  # (n_pp, n_lgn)
            arrivals_pp = arrivals_pp_raw[:, self.pp_swap_idx] if p.pp_onoff_swap else arrivals_pp_raw
            arrivals_pp = arrivals_pp * self.tc_mask_pp_f32
            pp_drive = (self.W_pp * arrivals_pp).sum(axis=1)
            self.I_pp += p.w_lgn_pp_gain * pp_drive
            pp_spk = self.pp.step(self.I_pp)
            self.last_pp_spk = pp_spk
            # Store per-ensemble push–pull drive for diagnostics/tests.
            if p.n_pp_per_ensemble == 1:
                self.last_g_pp_input = pp_drive.astype(np.float32)
            else:
                self.last_g_pp_input = pp_drive.reshape(self.M, p.n_pp_per_ensemble).mean(axis=1).astype(np.float32)
            # PP->E inhibition (GABA conductance increment)
            if p.w_pushpull > 0:
                self.g_v1_inh_pp += self.W_pp_e @ pp_spk.astype(np.float32)
        else:
            self.last_g_pp_input = np.zeros(self.M, dtype=np.float32)
            self.last_pp_spk = pp_spk

        # Total current to V1 excitatory (conductance-based inhibition)
        g_pv = np.clip(self.g_v1_inh_pv_decay - self.g_v1_inh_pv_rise, 0.0, None)
        g_inh = g_pv + self.g_v1_inh_som + self.g_v1_inh_pp
        I_exc_basal = self.g_v1_exc * (p.E_exc - self.v1_exc.v)
        if (float(p.apical_gain) > 0.0) and (self.v1_l23 is None):
            x = (self.g_v1_apical - float(p.apical_threshold)) / max(1e-6, float(p.apical_slope))
            gate = 1.0 + float(p.apical_gain) * (1.0 / (1.0 + np.exp(-x)))
            I_exc = I_exc_basal * gate.astype(np.float32, copy=False)
        else:
            I_exc = I_exc_basal
        I_v1_total = I_exc + g_inh * (p.E_inh - self.v1_exc.v)
        I_v1_total = I_v1_total + self.I_v1_bias
        v1_spk = self.v1_exc.step(I_v1_total)
        self.last_v1_spk = v1_spk

        # --- Optional L2/3 excitatory layer (receives basal L4 drive + apical modulation) ---
        v1_l23_spk = np.zeros(self.M, dtype=np.uint8)
        if self.v1_l23 is not None:
            self.g_l23_exc *= self.decay_ampa
            if self.W_l4_l23 is not None:
                self.g_l23_exc += p.w_exc_gain * (self.W_l4_l23 @ v1_spk.astype(np.float32))

            self.g_l23_apical *= self.decay_apical
            if apical_drive is not None:
                ap = np.asarray(apical_drive, dtype=np.float32)
                if ap.ndim == 0:
                    self.g_l23_apical += p.w_exc_gain * float(ap)
                else:
                    if ap.shape != (self.M,):
                        raise ValueError(f"apical_drive must have shape (M,), got {tuple(ap.shape)}")
                    self.g_l23_apical += p.w_exc_gain * ap

            I_l23_exc_basal = self.g_l23_exc * (p.E_exc - self.v1_l23.v)
            if float(p.apical_gain) > 0.0:
                x = (self.g_l23_apical - float(p.apical_threshold)) / max(1e-6, float(p.apical_slope))
                gate = 1.0 + float(p.apical_gain) * (1.0 / (1.0 + np.exp(-x)))
                I_l23_exc = I_l23_exc_basal * gate.astype(np.float32, copy=False)
            else:
                I_l23_exc = I_l23_exc_basal

            I_l23_total = I_l23_exc + self.g_l23_inh_som * (p.E_inh - self.v1_l23.v) + self.I_l23_bias
            v1_l23_spk = self.v1_l23.step(I_l23_total)
        self.last_v1_l23_spk = v1_l23_spk

        # --- VIP interneurons (disinhibitory; updated AFTER E, affects next step) ---
        vip_spk = np.zeros(self.n_vip, dtype=np.uint8)
        if self.vip is not None:
            self.I_vip *= self.decay_ampa
            if self.W_e_vip.size:
                drive_spk = self.prev_v1_l23_spk if (self.v1_l23 is not None) else self.prev_v1_spk
                self.I_vip += self.W_e_vip @ drive_spk.astype(np.float32)
            self.I_vip += float(p.vip_bias_current) + float(vip_td)
            vip_spk = self.vip.step(self.I_vip)
        self.last_vip_spk = vip_spk

        # --- SOM interneurons (lateral / dendritic inhibition; updated AFTER E, affects next step) ---
        self.I_som *= self.decay_ampa
        self.I_som_inh *= self.decay_gaba
        som_drive = v1_l23_spk if (self.v1_l23 is not None) else v1_spk
        self.I_som += self.W_e_som @ som_drive.astype(np.float32)
        if (self.vip is not None) and (self.W_vip_som.size) and (float(p.w_vip_som) != 0.0):
            self.I_som_inh += self.W_vip_som @ vip_spk.astype(np.float32)
        som_spk = self.som.step(self.I_som - self.I_som_inh)
        self.last_som_spk = som_spk

        # SOM->E lateral inhibition (GABA conductance increment; affects next step).
        # In laminar mode, SOM targets L2/3; L4 remains purely feedforward/local-inhibition.
        if self.v1_l23 is not None:
            self.g_l23_inh_som += self.W_som_e @ som_spk.astype(np.float32)
        else:
            self.g_v1_inh_som += self.W_som_e @ som_spk.astype(np.float32)

        # --- Lateral excitation (recurrent; applied after E, affects next step) ---
        self.g_v1_exc += p.w_exc_gain * (self.W_e_e @ v1_spk.astype(np.float32))

        # --- Plasticity ---
        if plastic:
            # Triplet STDP with per-synapse arrivals
            # arrivals is (M, n_lgn) - different for each post neuron due to delays
            # dW already includes multiplicative bounds
            dW = self.stdp.update(arrivals_tc, v1_spk, self.W)

            # Apply weight changes directly (multiplicative bounds already in dW)
            self.W += dW

            # Weight decay (models synaptic turnover/protein degradation)
            self.W *= (1.0 - p.w_decay)

            # Clip to valid range
            np.clip(self.W, 0.0, p.w_max, out=self.W)
            # Retinotopic cap (structural locality)
            np.minimum(self.W, p.w_max * self.lgn_mask_e, out=self.W)
            # Structural sparsity mask (absent synapses remain absent).
            self.W *= self.tc_mask_e_f32

            # Push–pull thalamic plasticity (LGN->PP interneurons).
            # Local plasticity at LGN->PP synapses: pre (LGN arrivals onto PP) x post (PP spikes).
            if p.pp_plastic:
                if arrivals_pp is not None:
                    dW_pp = self.pp_stdp.update(arrivals_pp, pp_spk, self.W_pp)
                    self.W_pp += dW_pp
                    if p.pp_decay > 0:
                        self.W_pp *= (1.0 - p.pp_decay)
                    np.clip(self.W_pp, 0.0, p.pp_w_max, out=self.W_pp)
                    np.minimum(self.W_pp, p.pp_w_max * self.lgn_mask_pp, out=self.W_pp)
                    self.W_pp *= self.tc_mask_pp_f32

            # Homeostatic inhibitory plasticity on PV->E (keeps E firing stable without hard normalization)
            if p.pv_inhib_plastic:
                self.pv_istdp.update(pv_spk, v1_spk, self.W_pv_e, self.mask_pv_e)

            # Slow lateral E->E plasticity (like-to-like coupling)
            if p.ee_plastic:
                # Use a 1-step pre→post lag to approximate finite axonal/dendritic delays and avoid
                # discrete-time zero-lag artifacts in recurrent STDP.
                dW_ee = self.ee_stdp.update(self.prev_v1_spk, v1_spk, self.W_e_e, self.mask_e_e)
                self.W_e_e += dW_ee
                self.W_e_e *= (1.0 - p.ee_decay)
                np.clip(self.W_e_e, 0.0, p.ee_w_max, out=self.W_e_e)

            # Update homeostatic rate estimate
            self.homeostasis.update_rate(v1_spk, p.dt_ms)

        # Update delay buffer pointer
        self.ptr = (self.ptr + 1) % self.L
        self.prev_v1_spk = v1_spk
        self.prev_v1_l23_spk = v1_l23_spk

        return v1_spk

    def apply_homeostasis(self):
        """Apply homeostatic scaling to weights (call periodically, not every step)."""
        if self.p.homeostasis_rate <= 0.0:
            return
        scale = self.homeostasis.get_scaling_factors()
        self.W *= scale[:, None]
        np.clip(self.W, 0.0, self.p.w_max, out=self.W)
        # Retinotopic cap (structural locality) must remain enforced even under synaptic scaling.
        np.minimum(self.W, self.p.w_max * self.lgn_mask_e, out=self.W)
        # Structural sparsity mask must remain enforced under slow scaling.
        self.W *= self.tc_mask_e_f32

    def apply_split_constraint(self) -> None:
        """Apply an ON/OFF split-constraint scaling to LGN->E weights (local per neuron)."""
        p = self.p
        if p.split_constraint_rate <= 0.0:
            return
        n_pix = int(p.N) * int(p.N)

        sum_on = self.W[:, :n_pix].sum(axis=1).astype(np.float32, copy=False)
        sum_off = self.W[:, n_pix:].sum(axis=1).astype(np.float32, copy=False)

        tgt_on = self.split_target_on.astype(np.float32, copy=False)
        tgt_off = self.split_target_off.astype(np.float32, copy=False)
        err_on = (tgt_on - sum_on) / (tgt_on + 1e-12)
        err_off = (tgt_off - sum_off) / (tgt_off + 1e-12)

        scale_on = 1.0 + p.split_constraint_rate * err_on
        scale_off = 1.0 + p.split_constraint_rate * err_off
        lo = 1.0 - float(p.split_constraint_clip)
        hi = 1.0 + float(p.split_constraint_clip)
        scale_on = np.clip(scale_on, lo, hi).astype(np.float32, copy=False)
        scale_off = np.clip(scale_off, lo, hi).astype(np.float32, copy=False)

        self.W[:, :n_pix] *= scale_on[:, None]
        self.W[:, n_pix:] *= scale_off[:, None]

        np.clip(self.W, 0.0, p.w_max, out=self.W)
        np.minimum(self.W, p.w_max * self.lgn_mask_e, out=self.W)
        self.W *= self.tc_mask_e_f32

    def _segment_boundary_updates(self, v1_counts: np.ndarray) -> None:
        """Update slow plasticity/homeostasis terms at a segment boundary (local, per-neuron)."""
        p = self.p

        # Optional slow synaptic scaling (no hard normalization).
        self.apply_homeostasis()
        # Optional ON/OFF split constraint (local resource pools; no global normalization).
        self.apply_split_constraint()

        # Intrinsic homeostasis (bias current) to keep firing near target.
        seg_rate_hz = v1_counts.astype(np.float32) / (p.segment_ms / 1000.0)
        self.I_v1_bias += p.v1_bias_eta * (p.target_rate_hz - seg_rate_hz)
        np.clip(self.I_v1_bias, -p.v1_bias_clip, p.v1_bias_clip, out=self.I_v1_bias)

    def run_segment(self, theta_deg: float, plastic: bool, *, contrast: float = 1.0) -> np.ndarray:
        """Run one stimulus segment and return V1 spike counts."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        phase = float(self.rng.uniform(0, 2 * math.pi))
        v1_counts = np.zeros(self.M, dtype=np.int32)

        for k in range(steps):
            on_spk, off_spk = self.rgc_spikes_grating(theta_deg, t_ms=k * p.dt_ms, phase=phase, contrast=contrast)
            v1_counts += self.step(on_spk, off_spk, plastic=plastic)

        # Slow processes updated at segment boundaries
        if plastic:
            self._segment_boundary_updates(v1_counts)

        return v1_counts

    def run_segment_sparse_spots(self, plastic: bool, *, contrast: float = 1.0) -> np.ndarray:
        """Run one segment of flickering sparse spots (flash-like developmental stimulus)."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        frame_steps = max(1, int(round(float(p.spots_frame_ms) / float(p.dt_ms))))
        v1_counts = np.zeros(self.M, dtype=np.int32)

        impl = str(p.rgc_dog_impl).lower()
        use_pad = p.rgc_center_surround and (impl == "padded_fft")
        if use_pad and (self._X_pad is None or self._Y_pad is None):
            raise RuntimeError("padded_fft DoG front-end not initialized")

        stim_pad = None
        stim = None
        for k in range(steps):
            if (k % frame_steps) == 0:
                if use_pad:
                    n = int(self._X_pad.shape[0])
                    X = self._X_pad
                    Y = self._Y_pad
                else:
                    n = int(p.N)
                    X = self.X
                    Y = self.Y

                stim_frame = np.zeros((n, n), dtype=np.float32)
                density = float(p.spots_density)
                if density > 0:
                    n_spots = int(round(density * float(n * n)))
                    n_spots = int(max(1, min(n * n, n_spots)))
                    idx = self.rng.choice(n * n, size=n_spots, replace=False)
                    pol = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_spots, replace=True)

                    sigma = float(p.spots_sigma)
                    if sigma <= 0:
                        stim_frame.ravel()[idx] = pol * float(p.spots_amp)
                    else:
                        cx = X.ravel()[idx].astype(np.float32, copy=False)
                        cy = Y.ravel()[idx].astype(np.float32, copy=False)
                        # Subpixel jitter keeps flashes from locking to the sampling lattice.
                        cx = (cx + self.rng.uniform(-0.5, 0.5, size=cx.shape).astype(np.float32))
                        cy = (cy + self.rng.uniform(-0.5, 0.5, size=cy.shape).astype(np.float32))
                        inv2s2 = float(1.0 / (2.0 * sigma * sigma))
                        for j in range(n_spots):
                            stim_frame += (pol[j] * float(p.spots_amp)) * np.exp(
                                -(((X - cx[j]) ** 2 + (Y - cy[j]) ** 2) * inv2s2)
                            ).astype(np.float32)

                if use_pad:
                    stim_pad = stim_frame
                else:
                    stim = stim_frame

            if use_pad:
                drive_on, drive_off = self._rgc_drives_from_pad_stimulus(stim_pad, contrast=contrast)
                on_spk, off_spk = self.rgc_spikes_from_drives(drive_on, drive_off)
            else:
                on_spk, off_spk = self.rgc_spikes(stim, contrast=contrast)

            v1_counts += self.step(on_spk, off_spk, plastic=plastic)

        if plastic:
            self._segment_boundary_updates(v1_counts)

        return v1_counts

    def run_segment_white_noise(self, plastic: bool, *, contrast: float = 1.0) -> np.ndarray:
        """Run one segment of dense spatiotemporal white noise (Linsker-style)."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        frame_steps = max(1, int(round(float(p.noise_frame_ms) / float(p.dt_ms))))
        v1_counts = np.zeros(self.M, dtype=np.int32)

        impl = str(p.rgc_dog_impl).lower()
        use_pad = p.rgc_center_surround and (impl == "padded_fft")
        if use_pad and (self._X_pad is None or self._Y_pad is None):
            raise RuntimeError("padded_fft DoG front-end not initialized")

        stim_pad = None
        stim = None
        for k in range(steps):
            if (k % frame_steps) == 0:
                if use_pad:
                    n = int(self._X_pad.shape[0])
                    stim_pad = self.rng.normal(0.0, float(p.noise_sigma), size=(n, n)).astype(np.float32)
                    if p.noise_clip > 0:
                        np.clip(stim_pad, -float(p.noise_clip), float(p.noise_clip), out=stim_pad)
                else:
                    stim = self.rng.normal(0.0, float(p.noise_sigma), size=(p.N, p.N)).astype(np.float32)
                    if p.noise_clip > 0:
                        np.clip(stim, -float(p.noise_clip), float(p.noise_clip), out=stim)

            if use_pad:
                drive_on, drive_off = self._rgc_drives_from_pad_stimulus(stim_pad, contrast=contrast)
                on_spk, off_spk = self.rgc_spikes_from_drives(drive_on, drive_off)
            else:
                on_spk, off_spk = self.rgc_spikes(stim, contrast=contrast)

            v1_counts += self.step(on_spk, off_spk, plastic=plastic)

        if plastic:
            self._segment_boundary_updates(v1_counts)

        return v1_counts

    def run_segment_sparse_spots_counts(self, plastic: bool, *, contrast: float = 1.0) -> dict:
        """Run one sparse_spots segment and return spike counts for E/PV/PP/SOM/LGN (diagnostics/tests)."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        frame_steps = max(1, int(round(float(p.spots_frame_ms) / float(p.dt_ms))))

        v1_counts = np.zeros(self.M, dtype=np.int32)
        l23_counts = np.zeros(self.M, dtype=np.int32)
        pv_counts = np.zeros(self.n_pv, dtype=np.int32)
        pp_counts = np.zeros(self.n_pp, dtype=np.int32)
        som_counts = np.zeros(self.n_som, dtype=np.int32)
        vip_counts = np.zeros(self.n_vip, dtype=np.int32)
        lgn_counts = np.zeros(self.n_lgn, dtype=np.int32)

        impl = str(p.rgc_dog_impl).lower()
        use_pad = p.rgc_center_surround and (impl == "padded_fft")
        if use_pad and (self._X_pad is None or self._Y_pad is None):
            raise RuntimeError("padded_fft DoG front-end not initialized")

        stim_pad = None
        stim = None
        for k in range(steps):
            if (k % frame_steps) == 0:
                if use_pad:
                    n = int(self._X_pad.shape[0])
                    X = self._X_pad
                    Y = self._Y_pad
                else:
                    n = int(p.N)
                    X = self.X
                    Y = self.Y

                stim_frame = np.zeros((n, n), dtype=np.float32)
                density = float(p.spots_density)
                if density > 0:
                    n_spots = int(round(density * float(n * n)))
                    n_spots = int(max(1, min(n * n, n_spots)))
                    idx = self.rng.choice(n * n, size=n_spots, replace=False)
                    pol = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_spots, replace=True)

                    sigma = float(p.spots_sigma)
                    if sigma <= 0:
                        stim_frame.ravel()[idx] = pol * float(p.spots_amp)
                    else:
                        cx = X.ravel()[idx].astype(np.float32, copy=False)
                        cy = Y.ravel()[idx].astype(np.float32, copy=False)
                        cx = (cx + self.rng.uniform(-0.5, 0.5, size=cx.shape).astype(np.float32))
                        cy = (cy + self.rng.uniform(-0.5, 0.5, size=cy.shape).astype(np.float32))
                        inv2s2 = float(1.0 / (2.0 * sigma * sigma))
                        for j in range(n_spots):
                            stim_frame += (pol[j] * float(p.spots_amp)) * np.exp(
                                -(((X - cx[j]) ** 2 + (Y - cy[j]) ** 2) * inv2s2)
                            ).astype(np.float32)

                if use_pad:
                    stim_pad = stim_frame
                else:
                    stim = stim_frame

            if use_pad:
                drive_on, drive_off = self._rgc_drives_from_pad_stimulus(stim_pad, contrast=contrast)
                on_spk, off_spk = self.rgc_spikes_from_drives(drive_on, drive_off)
            else:
                on_spk, off_spk = self.rgc_spikes(stim, contrast=contrast)

            v1_counts += self.step(on_spk, off_spk, plastic=plastic)
            l23_counts += self.last_v1_l23_spk
            pv_counts += self.last_pv_spk
            pp_counts += self.last_pp_spk
            som_counts += self.last_som_spk
            vip_counts += self.last_vip_spk
            lgn_counts += self.last_lgn_spk

        if plastic:
            self._segment_boundary_updates(v1_counts)

        return {
            "v1_counts": v1_counts,
            "l23_counts": l23_counts,
            "pv_counts": pv_counts,
            "pp_counts": pp_counts,
            "som_counts": som_counts,
            "vip_counts": vip_counts,
            "lgn_counts": lgn_counts,
        }

    def run_segment_white_noise_counts(self, plastic: bool, *, contrast: float = 1.0) -> dict:
        """Run one white_noise segment and return spike counts for E/PV/PP/SOM/LGN (diagnostics/tests)."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        frame_steps = max(1, int(round(float(p.noise_frame_ms) / float(p.dt_ms))))

        v1_counts = np.zeros(self.M, dtype=np.int32)
        l23_counts = np.zeros(self.M, dtype=np.int32)
        pv_counts = np.zeros(self.n_pv, dtype=np.int32)
        pp_counts = np.zeros(self.n_pp, dtype=np.int32)
        som_counts = np.zeros(self.n_som, dtype=np.int32)
        vip_counts = np.zeros(self.n_vip, dtype=np.int32)
        lgn_counts = np.zeros(self.n_lgn, dtype=np.int32)

        impl = str(p.rgc_dog_impl).lower()
        use_pad = p.rgc_center_surround and (impl == "padded_fft")
        if use_pad and (self._X_pad is None or self._Y_pad is None):
            raise RuntimeError("padded_fft DoG front-end not initialized")

        stim_pad = None
        stim = None
        for k in range(steps):
            if (k % frame_steps) == 0:
                if use_pad:
                    n = int(self._X_pad.shape[0])
                    stim_pad = self.rng.normal(0.0, float(p.noise_sigma), size=(n, n)).astype(np.float32)
                    if p.noise_clip > 0:
                        np.clip(stim_pad, -float(p.noise_clip), float(p.noise_clip), out=stim_pad)
                else:
                    stim = self.rng.normal(0.0, float(p.noise_sigma), size=(p.N, p.N)).astype(np.float32)
                    if p.noise_clip > 0:
                        np.clip(stim, -float(p.noise_clip), float(p.noise_clip), out=stim)

            if use_pad:
                drive_on, drive_off = self._rgc_drives_from_pad_stimulus(stim_pad, contrast=contrast)
                on_spk, off_spk = self.rgc_spikes_from_drives(drive_on, drive_off)
            else:
                on_spk, off_spk = self.rgc_spikes(stim, contrast=contrast)

            v1_counts += self.step(on_spk, off_spk, plastic=plastic)
            l23_counts += self.last_v1_l23_spk
            pv_counts += self.last_pv_spk
            pp_counts += self.last_pp_spk
            som_counts += self.last_som_spk
            vip_counts += self.last_vip_spk
            lgn_counts += self.last_lgn_spk

        if plastic:
            self._segment_boundary_updates(v1_counts)

        return {
            "v1_counts": v1_counts,
            "l23_counts": l23_counts,
            "pv_counts": pv_counts,
            "pp_counts": pp_counts,
            "som_counts": som_counts,
            "vip_counts": vip_counts,
            "lgn_counts": lgn_counts,
        }

    def run_segment_counts(self, theta_deg: float, plastic: bool, *, contrast: float = 1.0) -> dict:
        """Run one segment and return spike counts for E/PV/SOM/LGN (for diagnostics/tests)."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        phase = float(self.rng.uniform(0, 2 * math.pi))

        v1_counts = np.zeros(self.M, dtype=np.int32)
        l23_counts = np.zeros(self.M, dtype=np.int32)
        pv_counts = np.zeros(self.n_pv, dtype=np.int32)
        pp_counts = np.zeros(self.n_pp, dtype=np.int32)
        som_counts = np.zeros(self.n_som, dtype=np.int32)
        vip_counts = np.zeros(self.n_vip, dtype=np.int32)
        lgn_counts = np.zeros(self.n_lgn, dtype=np.int32)

        for k in range(steps):
            on_spk, off_spk = self.rgc_spikes_grating(theta_deg, t_ms=k * p.dt_ms, phase=phase, contrast=contrast)
            v1_counts += self.step(on_spk, off_spk, plastic=plastic)
            l23_counts += self.last_v1_l23_spk
            pv_counts += self.last_pv_spk
            pp_counts += self.last_pp_spk
            som_counts += self.last_som_spk
            vip_counts += self.last_vip_spk
            lgn_counts += self.last_lgn_spk

        return {
            "v1_counts": v1_counts,
            "l23_counts": l23_counts,
            "pv_counts": pv_counts,
            "pp_counts": pp_counts,
            "som_counts": som_counts,
            "vip_counts": vip_counts,
            "lgn_counts": lgn_counts,
        }

    def run_recording(self, theta_deg: float, duration_ms: float, *, contrast: float = 1.0,
                      phase: float = 0.0, reset: bool = True) -> dict:
        """Run for duration_ms and record key time series for diagnostics/tests."""
        p = self.p
        steps = int(duration_ms / p.dt_ms)
        if reset:
            self.reset_state()

        I_ff_ts = np.zeros((steps, self.M), dtype=np.float32)
        g_pp_ts = np.zeros((steps, self.M), dtype=np.float32)
        g_pp_inh_ts = np.zeros((steps, self.M), dtype=np.float32)
        v_ts = np.zeros((steps, self.M), dtype=np.float32)
        v1_spk_ts = np.zeros((steps, self.M), dtype=np.uint8)
        v_l23_ts = None
        v1_l23_spk_ts = None
        if self.v1_l23 is not None:
            v_l23_ts = np.zeros((steps, self.M), dtype=np.float32)
            v1_l23_spk_ts = np.zeros((steps, self.M), dtype=np.uint8)

        for k in range(steps):
            on_spk, off_spk = self.rgc_spikes_grating(theta_deg, t_ms=k * p.dt_ms, phase=phase, contrast=contrast)
            v1_spk = self.step(on_spk, off_spk, plastic=False)
            I_ff_ts[k] = self.last_I_ff
            g_pp_ts[k] = self.last_g_pp_input
            g_pp_inh_ts[k] = self.g_v1_inh_pp
            v_ts[k] = self.v1_exc.v
            v1_spk_ts[k] = v1_spk
            if v_l23_ts is not None and v1_l23_spk_ts is not None:
                v_l23_ts[k] = self.v1_l23.v
                v1_l23_spk_ts[k] = self.last_v1_l23_spk

        t_ms = (np.arange(steps, dtype=np.float32) * p.dt_ms).astype(np.float32)
        return {
            "t_ms": t_ms,
            "I_ff": I_ff_ts,
            "g_pp_input": g_pp_ts,
            "g_pp_inh": g_pp_inh_ts,
            "v": v_ts,
            "v1_spk": v1_spk_ts,
            "v_l23": v_l23_ts,
            "v1_l23_spk": v1_l23_spk_ts,
        }

    def evaluate_tuning(self, thetas_deg: np.ndarray, repeats: int, *, contrast: float = 1.0) -> np.ndarray:
        """
        Evaluate orientation tuning.

        Returns rates (Hz) per ensemble per orientation.
        """
        p = self.p
        rates = np.zeros((self.M, len(thetas_deg)), dtype=np.float32)

        # Save state
        rng_state = self.rng.bit_generator.state
        saved_lgn_v = self.lgn.v.copy()
        saved_lgn_u = self.lgn.u.copy()
        saved_v1_v = self.v1_exc.v.copy()
        saved_v1_u = self.v1_exc.u.copy()
        saved_l23_v = None if self.v1_l23 is None else self.v1_l23.v.copy()
        saved_l23_u = None if self.v1_l23 is None else self.v1_l23.u.copy()
        saved_pv_v = self.pv.v.copy()
        saved_pv_u = self.pv.u.copy()
        saved_pp_v = self.pp.v.copy()
        saved_pp_u = self.pp.u.copy()
        saved_som_v = self.som.v.copy()
        saved_som_u = self.som.u.copy()
        saved_vip_v = None if self.vip is None else self.vip.v.copy()
        saved_vip_u = None if self.vip is None else self.vip.u.copy()
        saved_I_lgn = self.I_lgn.copy()
        saved_lgn_rgc_drive = self._lgn_rgc_drive.copy()
        saved_g_v1_exc = self.g_v1_exc.copy()
        saved_g_v1_apical = self.g_v1_apical.copy()
        saved_g_l23_exc = self.g_l23_exc.copy()
        saved_g_l23_apical = self.g_l23_apical.copy()
        saved_g_l23_inh_som = self.g_l23_inh_som.copy()
        saved_I_l23_bias = self.I_l23_bias.copy()
        saved_I_v1_bias = self.I_v1_bias.copy()
        saved_g_v1_inh_pv_rise = self.g_v1_inh_pv_rise.copy()
        saved_g_v1_inh_pv_decay = self.g_v1_inh_pv_decay.copy()
        saved_g_v1_inh_som = self.g_v1_inh_som.copy()
        saved_g_v1_inh_pp = self.g_v1_inh_pp.copy()
        saved_I_pv = self.I_pv.copy()
        saved_I_pv_inh = self.I_pv_inh.copy()
        saved_I_pp = self.I_pp.copy()
        saved_I_som = self.I_som.copy()
        saved_I_som_inh = self.I_som_inh.copy()
        saved_I_vip = self.I_vip.copy()
        saved_buf = self.delay_buf.copy()
        saved_ptr = self.ptr
        saved_tc_stp_x = None if self.tc_stp_x is None else self.tc_stp_x.copy()
        saved_tc_stp_x_pv = None if self.tc_stp_x_pv is None else self.tc_stp_x_pv.copy()
        saved_stdp_x_pre = self.stdp.x_pre.copy()
        saved_stdp_x_pre_slow = self.stdp.x_pre_slow.copy()
        saved_stdp_x_post = self.stdp.x_post.copy()
        saved_stdp_x_post_slow = self.stdp.x_post_slow.copy()
        saved_pv_istdp_x_post = self.pv_istdp.x_post.copy()
        saved_ee_x_pre = self.ee_stdp.x_pre.copy()
        saved_ee_x_post = self.ee_stdp.x_post.copy()
        saved_prev_v1_spk = self.prev_v1_spk.copy()
        saved_prev_v1_l23_spk = self.prev_v1_l23_spk.copy()
        saved_rgc_drive_fast_on = None if self._rgc_drive_fast_on is None else self._rgc_drive_fast_on.copy()
        saved_rgc_drive_slow_on = None if self._rgc_drive_slow_on is None else self._rgc_drive_slow_on.copy()
        saved_rgc_drive_fast_off = None if self._rgc_drive_fast_off is None else self._rgc_drive_fast_off.copy()
        saved_rgc_drive_slow_off = None if self._rgc_drive_slow_off is None else self._rgc_drive_slow_off.copy()
        saved_rgc_refr_on = None if self._rgc_refr_on is None else self._rgc_refr_on.copy()
        saved_rgc_refr_off = None if self._rgc_refr_off is None else self._rgc_refr_off.copy()

        for j, th in enumerate(thetas_deg):
            cnt = np.zeros(self.M, dtype=np.float32)
            for _ in range(repeats):
                self.reset_state()
                cnt += self.run_segment(float(th), plastic=False, contrast=contrast)
            rates[:, j] = cnt / (repeats * (p.segment_ms / 1000.0))

        # Restore state
        self.lgn.v = saved_lgn_v
        self.lgn.u = saved_lgn_u
        self.v1_exc.v = saved_v1_v
        self.v1_exc.u = saved_v1_u
        if (self.v1_l23 is not None) and (saved_l23_v is not None) and (saved_l23_u is not None):
            self.v1_l23.v = saved_l23_v
            self.v1_l23.u = saved_l23_u
        self.pv.v = saved_pv_v
        self.pv.u = saved_pv_u
        self.pp.v = saved_pp_v
        self.pp.u = saved_pp_u
        self.som.v = saved_som_v
        self.som.u = saved_som_u
        if (self.vip is not None) and (saved_vip_v is not None) and (saved_vip_u is not None):
            self.vip.v = saved_vip_v
            self.vip.u = saved_vip_u
        self.I_lgn = saved_I_lgn
        self._lgn_rgc_drive = saved_lgn_rgc_drive
        self.g_v1_exc = saved_g_v1_exc
        self.g_v1_apical = saved_g_v1_apical
        self.g_l23_exc = saved_g_l23_exc
        self.g_l23_apical = saved_g_l23_apical
        self.g_l23_inh_som = saved_g_l23_inh_som
        self.I_l23_bias = saved_I_l23_bias
        self.I_v1_bias = saved_I_v1_bias
        self.g_v1_inh_pv_rise = saved_g_v1_inh_pv_rise
        self.g_v1_inh_pv_decay = saved_g_v1_inh_pv_decay
        self.g_v1_inh_som = saved_g_v1_inh_som
        self.g_v1_inh_pp = saved_g_v1_inh_pp
        self.I_pv = saved_I_pv
        self.I_pv_inh = saved_I_pv_inh
        self.I_pp = saved_I_pp
        self.I_som = saved_I_som
        self.I_som_inh = saved_I_som_inh
        if self.I_vip.size:
            self.I_vip = saved_I_vip
        self.delay_buf = saved_buf
        self.ptr = saved_ptr
        if (self.tc_stp_x is not None) and (saved_tc_stp_x is not None):
            self.tc_stp_x[...] = saved_tc_stp_x
        if (self.tc_stp_x_pv is not None) and (saved_tc_stp_x_pv is not None):
            self.tc_stp_x_pv[...] = saved_tc_stp_x_pv
        self.stdp.x_pre = saved_stdp_x_pre
        self.stdp.x_pre_slow = saved_stdp_x_pre_slow
        self.stdp.x_post = saved_stdp_x_post
        self.stdp.x_post_slow = saved_stdp_x_post_slow
        self.pv_istdp.x_post = saved_pv_istdp_x_post
        self.ee_stdp.x_pre = saved_ee_x_pre
        self.ee_stdp.x_post = saved_ee_x_post
        self.prev_v1_spk = saved_prev_v1_spk
        self.prev_v1_l23_spk = saved_prev_v1_l23_spk
        if saved_rgc_drive_fast_on is not None:
            if self._rgc_drive_fast_on is None:
                self._rgc_drive_fast_on = saved_rgc_drive_fast_on
            else:
                self._rgc_drive_fast_on[...] = saved_rgc_drive_fast_on
        if saved_rgc_drive_slow_on is not None:
            if self._rgc_drive_slow_on is None:
                self._rgc_drive_slow_on = saved_rgc_drive_slow_on
            else:
                self._rgc_drive_slow_on[...] = saved_rgc_drive_slow_on
        if saved_rgc_drive_fast_off is not None:
            if self._rgc_drive_fast_off is None:
                self._rgc_drive_fast_off = saved_rgc_drive_fast_off
            else:
                self._rgc_drive_fast_off[...] = saved_rgc_drive_fast_off
        if saved_rgc_drive_slow_off is not None:
            if self._rgc_drive_slow_off is None:
                self._rgc_drive_slow_off = saved_rgc_drive_slow_off
            else:
                self._rgc_drive_slow_off[...] = saved_rgc_drive_slow_off
        if saved_rgc_refr_on is not None:
            if self._rgc_refr_on is None:
                self._rgc_refr_on = saved_rgc_refr_on
            else:
                self._rgc_refr_on[...] = saved_rgc_refr_on
        if saved_rgc_refr_off is not None:
            if self._rgc_refr_off is None:
                self._rgc_refr_off = saved_rgc_refr_off
            else:
                self._rgc_refr_off[...] = saved_rgc_refr_off
        self.rng.bit_generator.state = rng_state

        return rates


# =============================================================================
# Visualization functions
# =============================================================================

def plot_weight_maps(W: np.ndarray, N: int, outpath: str, title: str, *, smooth_sigma: float = 0.0) -> None:
    """Plot ON, OFF, and ON-OFF weight maps for each ensemble."""
    M = W.shape[0]
    W_on = W[:, :N * N].reshape(M, N, N)
    W_off = W[:, N * N:].reshape(M, N, N)
    W_diff = W_on - W_off

    sigma = float(smooth_sigma)
    if sigma > 0.0:
        for m in range(M):
            if gaussian_filter is None:
                W_on[m] = _gaussian_filter_fallback(W_on[m], sigma=sigma)
                W_off[m] = _gaussian_filter_fallback(W_off[m], sigma=sigma)
                W_diff[m] = _gaussian_filter_fallback(W_diff[m], sigma=sigma)
            else:
                W_on[m] = gaussian_filter(W_on[m], sigma=sigma, mode="nearest")
                W_off[m] = gaussian_filter(W_off[m], sigma=sigma, mode="nearest")
                W_diff[m] = gaussian_filter(W_diff[m], sigma=sigma, mode="nearest")

    fig, axes = plt.subplots(M, 3, figsize=(9, 2.1 * M))
    if M == 1:
        axes = np.array([axes])

    for m in range(M):
        for j, (arr, coltitle) in enumerate([(W_on[m], "ON"), (W_off[m], "OFF"),
                                              (W_diff[m], "ON-OFF")]):
            ax = axes[m, j]
            im = ax.imshow(arr, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if m == 0:
                ax.set_title(coltitle)
            if j == 0:
                ax.set_ylabel(f"E{m}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_weight_maps_before_after(W_before: np.ndarray, W_after: np.ndarray, N: int,
                                  outpath: str, title: str, *, smooth_sigma: float = 0.0) -> None:
    """Plot initial vs final ON/OFF/ON-OFF weights with matched color scales."""
    M = int(W_before.shape[0])
    n_pix = int(N) * int(N)

    b_on = W_before[:, :n_pix].reshape(M, N, N).astype(np.float32, copy=True)
    b_off = W_before[:, n_pix:].reshape(M, N, N).astype(np.float32, copy=True)
    b_diff = b_on - b_off

    a_on = W_after[:, :n_pix].reshape(M, N, N).astype(np.float32, copy=True)
    a_off = W_after[:, n_pix:].reshape(M, N, N).astype(np.float32, copy=True)
    a_diff = a_on - a_off

    sigma = float(smooth_sigma)
    if sigma > 0.0:
        for m in range(M):
            if gaussian_filter is None:
                b_on[m] = _gaussian_filter_fallback(b_on[m], sigma=sigma)
                b_off[m] = _gaussian_filter_fallback(b_off[m], sigma=sigma)
                b_diff[m] = _gaussian_filter_fallback(b_diff[m], sigma=sigma)
                a_on[m] = _gaussian_filter_fallback(a_on[m], sigma=sigma)
                a_off[m] = _gaussian_filter_fallback(a_off[m], sigma=sigma)
                a_diff[m] = _gaussian_filter_fallback(a_diff[m], sigma=sigma)
            else:
                b_on[m] = gaussian_filter(b_on[m], sigma=sigma, mode="nearest")
                b_off[m] = gaussian_filter(b_off[m], sigma=sigma, mode="nearest")
                b_diff[m] = gaussian_filter(b_diff[m], sigma=sigma, mode="nearest")
                a_on[m] = gaussian_filter(a_on[m], sigma=sigma, mode="nearest")
                a_off[m] = gaussian_filter(a_off[m], sigma=sigma, mode="nearest")
                a_diff[m] = gaussian_filter(a_diff[m], sigma=sigma, mode="nearest")

    vmax_on = float(max(np.abs(b_on).max(), np.abs(a_on).max(), 1e-9))
    vmax_off = float(max(np.abs(b_off).max(), np.abs(a_off).max(), 1e-9))
    vmax_diff = float(max(np.abs(b_diff).max(), np.abs(a_diff).max(), 1e-9))

    fig, axes = plt.subplots(M, 6, figsize=(16, 2.0 * M))
    if M == 1:
        axes = np.array([axes])

    colspec = [
        ("Init ON", b_on, -vmax_on, vmax_on),
        ("Init OFF", b_off, -vmax_off, vmax_off),
        ("Init ON-OFF", b_diff, -vmax_diff, vmax_diff),
        ("Final ON", a_on, -vmax_on, vmax_on),
        ("Final OFF", a_off, -vmax_off, vmax_off),
        ("Final ON-OFF", a_diff, -vmax_diff, vmax_diff),
    ]
    for m in range(M):
        for j, (coltitle, arrs, vmin, vmax) in enumerate(colspec):
            ax = axes[m, j]
            im = ax.imshow(arrs[m], interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if m == 0:
                ax.set_title(coltitle)
            if j == 0:
                ax.set_ylabel(f"E{m}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_tuning(rates: np.ndarray, thetas_deg: np.ndarray, osi: np.ndarray,
                pref_deg: np.ndarray, outpath: str, title: str) -> None:
    """Plot orientation tuning curves."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in range(rates.shape[0]):
        ax.plot(thetas_deg, rates[m], marker="o",
                label=f"E{m} OSI={osi[m]:.2f} pref={pref_deg[m]:.0f}")
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_scalar_over_time(xs: np.ndarray, ys: np.ndarray, outpath: str,
                          ylabel: str, title: str) -> None:
    """Plot scalar metric over training."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Training segment")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_interneuron_activity(pv_rates: List[float], som_rates: List[float],
                              segments: List[int], outpath: str) -> None:
    """Plot interneuron activity over training."""
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(segments, pv_rates, marker="o", label="PV (FS)")
    ax.plot(segments, som_rates, marker="s", label="SOM (LTS)")
    ax.set_xlabel("Training segment")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_title("Interneuron activity over training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_pref_hist(pref_deg: np.ndarray, osi: np.ndarray, outpath: str,
                   title: str, *, osi_thresh: float = 0.3, bin_deg: int = 15) -> None:
    """Plot histogram of preferred orientations for tuned ensembles."""
    tuned = (osi >= osi_thresh)
    prefs = pref_deg[tuned]
    bins = np.arange(0.0, 180.0 + float(bin_deg), float(bin_deg))

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.hist(prefs, bins=bins, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Preferred orientation (deg)")
    ax.set_ylabel(f"Count (OSI≥{osi_thresh:.1f})")
    ax.set_title(title)
    ax.set_xlim(0.0, 180.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def save_eval_npz(outpath: str, *, thetas_deg: np.ndarray, rates_hz: np.ndarray,
                  osi: np.ndarray, pref_deg: np.ndarray, net: "RgcLgnV1Network") -> None:
    """Save numeric evaluation artifacts (no plotting) for reproducibility/debugging."""
    on_to_off = getattr(net, "on_to_off", None)
    proj_kwargs: dict = {}
    if bool(getattr(net.p, "rgc_separate_onoff_mosaics", False)):
        proj_kwargs = dict(
            X_on=net.X_on,
            Y_on=net.Y_on,
            X_off=net.X_off,
            Y_off=net.Y_off,
            sigma=float(getattr(net.p, "rgc_center_sigma", 0.5)),
        )

    rf_ori, rf_pref = rf_fft_orientation_metrics(net.W, net.N, on_to_off=on_to_off, **proj_kwargs)
    rf_grating_amp = rf_grating_match_tuning(
        net.W,
        net.N,
        float(net.p.spatial_freq),
        thetas_deg,
        on_to_off=on_to_off,
        **proj_kwargs,
    )
    rf_grating_osi, rf_grating_pref = compute_osi(rf_grating_amp, thetas_deg)
    w_onoff_corr = onoff_weight_corr(net.W, net.N, on_to_off=on_to_off, **proj_kwargs).astype(np.float32)
    np.savez_compressed(
        outpath,
        thetas_deg=thetas_deg.astype(np.float32),
        rates_hz=rates_hz.astype(np.float32),
        osi=osi.astype(np.float32),
        pref_deg=pref_deg.astype(np.float32),
        rf_orientedness=rf_ori.astype(np.float32),
        rf_pref_deg=rf_pref.astype(np.float32),
        rf_grating_amp=rf_grating_amp.astype(np.float32),
        rf_grating_osi=rf_grating_osi.astype(np.float32),
        rf_grating_pref_deg=rf_grating_pref.astype(np.float32),
        w_onoff_corr=w_onoff_corr.astype(np.float32),
        W=net.W.astype(np.float32),
        W_pp=net.W_pp.astype(np.float32),
        W_e_e=net.W_e_e.astype(np.float32),
        W_pv_e=net.W_pv_e.astype(np.float32),
        W_som_e=net.W_som_e.astype(np.float32),
        W_e_som=net.W_e_som.astype(np.float32),
    )


# =============================================================================
# Main training loop
# =============================================================================

def circ_diff_180(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Circular difference for orientation angles in degrees (period 180)."""
    d = np.abs(a_deg - b_deg) % 180.0
    return np.minimum(d, 180.0 - d)


def run_self_tests(out_dir: str) -> None:
    """Run a deterministic self-test suite and raise on failure."""
    safe_mkdir(out_dir)

    print("\n[tests] Running self-tests...")
    report: list[str] = []
    report.append("Self-test report")
    report.append("================")

    thetas = np.linspace(0, 180 - 180 / 12, 12)

    # --- Test 1: OSI improves with learning ---
    p = Params(N=8, M=8, seed=1, v1_bias_eta=0.0)
    net = RgcLgnV1Network(p)
    W_e_e0 = net.W_e_e.copy()
    train_segments = 300
    for _ in range(train_segments):
        th = float(net.rng.uniform(0.0, 180.0))
        net.run_segment(th, plastic=True)

    rates = net.evaluate_tuning(thetas, repeats=7, contrast=1.0)
    osi, pref = compute_osi(rates, thetas)
    mean_osi = float(osi.mean())
    if mean_osi < 0.30:
        raise AssertionError(f"OSI learning failed: mean OSI={mean_osi:.3f} (expected >= 0.30)")
    report.append(f"Test 1 (OSI learning): mean OSI={mean_osi:.3f}, max OSI={float(osi.max()):.3f}, mean rate={float(rates.mean()):.3f} Hz")

    plot_tuning(rates, thetas, osi, pref,
                os.path.join(out_dir, "selftest_tuning.png"),
                title=f"Self-test tuning (mean OSI={mean_osi:.3f})")
    plot_weight_maps(net.W, p.N, os.path.join(out_dir, "selftest_weights.png"),
                     title="Self-test LGN->V1 weights (after learning)")
    plot_pref_hist(pref, osi, os.path.join(out_dir, "selftest_pref_hist.png"),
                   title="Self-test preferred orientation histogram")
    save_eval_npz(os.path.join(out_dir, "selftest_eval.npz"),
                  thetas_deg=thetas, rates_hz=rates, osi=osi, pref_deg=pref, net=net)
    with open(os.path.join(out_dir, "selftest_params.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(p), f, indent=2, sort_keys=True)

    # --- Test 2: Preferred orientation stability across contrast ---
    rates_hi = net.evaluate_tuning(thetas, repeats=7, contrast=1.0)
    _, pref_hi = compute_osi(rates_hi, thetas)
    # Low-contrast responses can be spike-sparse; use more repeats to reduce finite-sample noise.
    rates_lo = net.evaluate_tuning(thetas, repeats=21, contrast=0.85)
    _, pref_lo = compute_osi(rates_lo, thetas)

    # Use PEAK response rather than mean: tuned cells can have low mean rates but still show a clear peak.
    peak_hi = rates_hi.max(axis=1)
    peak_lo = rates_lo.max(axis=1)
    # If low-contrast responses are still extremely sparse, estimate with even more repeats.
    if float(peak_lo.max()) <= 0.0:
        rates_lo = net.evaluate_tuning(thetas, repeats=49, contrast=0.85)
        _, pref_lo = compute_osi(rates_lo, thetas)
        peak_lo = rates_lo.max(axis=1)
    active = (peak_hi > 0.5) & (peak_lo > 0.15)
    if not active.any():
        raise AssertionError(
            "Contrast test: no ensembles active at both contrasts "
            f"(peak_hi max={float(peak_hi.max()):.3f} Hz, peak_lo max={float(peak_lo.max()):.3f} Hz)"
        )
    d_pref = circ_diff_180(pref_hi[active], pref_lo[active])
    if float(d_pref.mean()) > 15.0 or float(d_pref.max()) > 30.0:
        raise AssertionError(
            f"Contrast test failed: pref shift mean={float(d_pref.mean()):.1f}°, max={float(d_pref.max()):.1f}°"
        )
    report.append(
        f"Test 2 (Contrast pref stability): mean shift={float(d_pref.mean()):.1f}°, max shift={float(d_pref.max()):.1f}° (n={int(active.sum())})"
    )

    # --- Test 3: Push–pull phase opposition at temporal frequency ---
    # Not every tuned cell must be perfectly push–pull; require that *at least one* tuned ensemble
    # exhibits a clear counterphase signature.
    fs = 1000.0 / p.dt_ms
    tuned = (osi >= 0.2)
    if not tuned.any():
        raise AssertionError("Push–pull phase test failed: no tuned ensembles (OSI>=0.2) to test")

    best_err = float("inf")
    best_m = -1
    best_mod = 0.0
    for m in np.where(tuned)[0]:
        rec = net.run_recording(float(pref[m]), duration_ms=2000.0, contrast=1.0, phase=0.0, reset=True)
        I = rec["I_ff"][:, m].astype(np.float64)
        # Compare feedforward excitation with the *push–pull drive* (opposite-polarity thalamic drive into PP).
        # This is smoother than spike-based conductances and corresponds to the classic "counterphase" signature.
        G = rec["g_pp_input"][:, m].astype(np.float64)
        I -= I.mean()
        G -= G.mean()
        freqs = np.fft.rfftfreq(len(I), d=1.0 / fs)
        FI = np.fft.rfft(I)
        FG = np.fft.rfft(G)
        k = int(np.argmin(np.abs(freqs - p.temporal_freq)))
        if abs(FG[k]) < 1e-6 or abs(FI[k]) < 1e-6:
            continue
        diff = float(np.angle(np.exp(1j * (np.angle(FG[k]) - np.angle(FI[k])))))
        err_deg = float(np.degrees(abs(abs(diff) - math.pi)))
        mod = float(abs(FG[k]))
        if err_deg < best_err:
            best_err = err_deg
            best_m = int(m)
            best_mod = mod

    if best_m < 0:
        raise AssertionError("Push–pull phase test failed: insufficient modulation at temporal frequency")
    if best_err > 75.0:
        raise AssertionError(f"Push–pull phase test failed: best |Δphase-π|={best_err:.1f}° (expected <= 75°)")
    report.append(
        f"Test 3 (Push–pull phase): best |Δphase-π|={best_err:.1f}° at ~{p.temporal_freq:.1f} Hz (ensemble {best_m}, |FG|={best_mod:.2e})"
    )

    # --- Test 4: PV is thalamically recruitable even without E->PV ---
    p_ff = Params(N=8, M=8, seed=1, w_e_pv=0.0)
    net_ff = RgcLgnV1Network(p_ff)
    counts = net_ff.run_segment_counts(90.0, plastic=False, contrast=2.0)
    if int(counts["pv_counts"].sum()) <= 0:
        raise AssertionError("PV feedforward test failed: PV did not spike with LGN drive (contrast=2.0)")
    report.append(f"Test 4 (PV feedforward): PV spikes={int(counts['pv_counts'].sum())} (contrast=2.0, w_e_pv=0)")

    # --- Test 5: SOM circuit geometry (E->SOM broader than SOM->E) ---
    som_idx = 0
    in_kernel = net.W_e_som[som_idx].astype(np.float64)
    out_kernel = net.W_som_e[:, som_idx].astype(np.float64)
    ds = np.arange(p.M)
    ds = np.minimum(ds, p.M - ds).astype(np.float64)
    in_var = float((in_kernel * (ds ** 2)).sum() / (in_kernel.sum() + 1e-12))
    out_var = float((out_kernel * (ds ** 2)).sum() / (out_kernel.sum() + 1e-12))
    if not (in_var > out_var):
        raise AssertionError(f"SOM geometry test failed: in_var={in_var:.3f} not > out_var={out_var:.3f}")
    report.append(f"Test 5 (SOM geometry): in_var={in_var:.3f}, out_var={out_var:.3f}")

    # --- Test 6a: PV connectivity spread (optional realism) ---
    p_pv = Params(N=8, M=9, seed=1, pv_in_sigma=1.0, pv_out_sigma=1.0)
    net_pv = RgcLgnV1Network(p_pv)
    frac_max_in = float((net_pv.W_pv_e.max(axis=1) / (net_pv.W_pv_e.sum(axis=1) + 1e-12)).max())
    frac_max_out = float((net_pv.W_e_pv.max(axis=1) / (net_pv.W_e_pv.sum(axis=1) + 1e-12)).max())
    if frac_max_in > 0.95 or frac_max_out > 0.95:
        raise AssertionError(
            f"PV spread test failed: max frac weight too concentrated (in={frac_max_in:.3f}, out={frac_max_out:.3f})"
        )
    report.append(f"Test 6a (PV spread): max frac(in)={frac_max_in:.3f}, max frac(out)={frac_max_out:.3f}")

    # --- Test 6b: PV<->PV coupling matrix well-formed ---
    p_pvpv = Params(N=8, M=9, seed=1, pv_pv_sigma=1.0, w_pv_pv=1.0)
    net_pvpv = RgcLgnV1Network(p_pvpv)
    if net_pvpv.W_pv_pv is None:
        raise AssertionError("PV↔PV coupling test failed: W_pv_pv is None when enabled")
    if float(np.abs(np.diag(net_pvpv.W_pv_pv)).max()) > 1e-12:
        raise AssertionError("PV↔PV coupling test failed: diagonal not zero")
    report.append("Test 6b (PV↔PV coupling): W_pv_pv present, diag=0")

    # --- Test 6c: VIP disinhibition suppresses SOM spiking (smoke test) ---
    p_no_vip = Params(N=8, M=4, seed=1, segment_ms=120)
    net_no_vip = RgcLgnV1Network(p_no_vip)
    c0 = net_no_vip.run_segment_counts(90.0, plastic=False, contrast=2.0)
    p_vip = Params(
        N=8,
        M=4,
        seed=1,
        segment_ms=120,
        n_vip_per_ensemble=1,
        w_vip_som=25.0,
        vip_bias_current=30.0,
    )
    net_vip = RgcLgnV1Network(p_vip)
    c1 = net_vip.run_segment_counts(90.0, plastic=False, contrast=2.0)
    if int(c1["vip_counts"].sum()) <= 0:
        raise AssertionError("VIP disinhibition test failed: VIP produced 0 spikes")
    if int(c1["som_counts"].sum()) >= int(c0["som_counts"].sum()):
        raise AssertionError(
            "VIP disinhibition test failed: SOM not suppressed "
            f"(no_vip={int(c0['som_counts'].sum())}, vip={int(c1['som_counts'].sum())})"
        )
    report.append(
        f"Test 6c (VIP disinhibition): SOM spikes {int(c0['som_counts'].sum())} -> {int(c1['som_counts'].sum())}, "
        f"VIP spikes={int(c1['vip_counts'].sum())}"
    )

    # --- Test 6d: Apical scaffold is inert when gain=0 and boosts when enabled ---
    p_ap = Params(N=8, M=1, seed=1, w_pv_e=0.0, w_som_e=0.0, w_pushpull=0.0, w_e_pv=0.0, w_e_som=0.0)
    net_ap = RgcLgnV1Network(p_ap)
    on_spk = np.ones((p_ap.N, p_ap.N), dtype=np.uint8)
    off_spk = np.zeros_like(on_spk)
    net_ap.reset_state()
    cnt_a = 0
    for _ in range(120):
        cnt_a += int(net_ap.step(on_spk, off_spk, plastic=False, apical_drive=np.array([25.0], dtype=np.float32)).sum())
    net_ap.reset_state()
    cnt_b = 0
    for _ in range(120):
        cnt_b += int(net_ap.step(on_spk, off_spk, plastic=False).sum())
    if cnt_a != cnt_b:
        raise AssertionError(f"Apical inertness test failed: gain=0 but spikes differed ({cnt_a} vs {cnt_b})")
    net_ap.p.apical_gain = 1.0
    net_ap.p.apical_threshold = 0.0
    net_ap.p.apical_slope = 0.1
    net_ap.reset_state()
    cnt_c = 0
    for _ in range(120):
        cnt_c += int(net_ap.step(on_spk, off_spk, plastic=False, apical_drive=np.array([25.0], dtype=np.float32)).sum())
    if cnt_c < cnt_b:
        raise AssertionError(f"Apical gain test failed: enabled apical reduced spikes ({cnt_b} -> {cnt_c})")
    report.append(f"Test 6d (Apical scaffold): gain=0 spikes={cnt_b}, gain=1 spikes={cnt_c}")

    # --- Test 6: Lateral E->E plasticity produces like-to-like coupling ---
    if p.ee_plastic:
        w_change = float(np.mean(np.abs(net.W_e_e - W_e_e0)))
        if w_change < 1e-4:
            raise AssertionError(f"E->E plasticity test failed: mean |ΔW_e_e|={w_change:.3e} (expected > 1e-4)")

        # Stronger recurrent weights should preferentially link similarly tuned ensembles.
        diff = circ_diff_180(pref[:, None], pref[None, :])
        sim = np.cos(np.deg2rad(2.0 * diff))  # 1 for same, -1 for orthogonal
        w = net.W_e_e[net.mask_e_e].astype(np.float64).ravel()
        s = sim[net.mask_e_e].astype(np.float64).ravel()
        order = np.argsort(w)
        k = max(1, len(w) // 5)
        s_bot = float(s[order[:k]].mean())
        s_top = float(s[order[-k:]].mean())
        if not (s_top > s_bot + 0.05):
            raise AssertionError(
                f"E->E like-to-like test failed: s_top={s_top:.3f}, s_bot={s_bot:.3f}, diff={s_top - s_bot:.3f}"
            )
        report.append(
            f"Test 6 (E→E like-to-like): mean|ΔW|={w_change:.3e}, s_top={s_top:.3f}, s_bot={s_bot:.3f}, diff={s_top - s_bot:.3f}"
        )

    # --- Test 7: Retinotopic caps remain enforced (including after synaptic scaling) ---
    max_violation_e = float((net.W - p.w_max * net.lgn_mask_e).max())
    if max_violation_e > 1e-4:
        raise AssertionError(f"Retinotopic cap test failed (W): max violation={max_violation_e:.3e}")
    max_violation_pp = float((net.W_pp - p.pp_w_max * net.lgn_mask_pp).max())
    if max_violation_pp > 1e-4:
        raise AssertionError(f"Retinotopic cap test failed (W_pp): max violation={max_violation_pp:.3e}")
    report.append(
        f"Test 7 (Retinotopy caps): max(W-cap)={max_violation_e:.2e}, max(W_pp-cap)={max_violation_pp:.2e}"
    )

    # --- Test 8: LGN->PP plasticity is PP-local (no cross-population teaching signal) ---
    # If PP receives no drive and has no intrinsic decay, W_pp should not change.
    p_pp = Params(N=8, M=8, seed=1, w_lgn_pp_gain=0.0, w_pushpull=0.0, pp_decay=0.0, pp_plastic=True)
    net_pp = RgcLgnV1Network(p_pp)
    W_pp0 = net_pp.W_pp.copy()
    for _ in range(20):
        th = float(net_pp.rng.uniform(0.0, 180.0))
        net_pp.run_segment(th, plastic=True)
    d_pp = float(np.mean(np.abs(net_pp.W_pp - W_pp0)))
    if d_pp > 1e-7:
        raise AssertionError(f"PP locality test failed: mean |ΔW_pp|={d_pp:.3e} (expected ~0 when PP is silent)")
    report.append(f"Test 8 (PP locality): mean|ΔW_pp|={d_pp:.3e} with w_lgn_pp_gain=0, pp_decay=0")

    # --- Test 9: 2D cortical geometry distance metric is well-formed ---
    p_2d = Params(N=8, M=9, seed=1, cortex_shape=(3, 3), cortex_wrap=True)
    net_2d = RgcLgnV1Network(p_2d)
    if not (net_2d.cortex_h == 3 and net_2d.cortex_w == 3):
        raise AssertionError("2D cortex shape test failed: cortex_h/w mismatch")
    # Wrap-around: (0,0) to (2,0) is dy=1 on a 3-high torus => dist2=1.
    if abs(float(net_2d.cortex_dist2[0, 6]) - 1.0) > 1e-6:
        raise AssertionError(
            f"2D cortex wrap test failed: dist2(0,6)={float(net_2d.cortex_dist2[0, 6]):.3f} (expected 1)"
        )
    # Diagonal neighbor: (0,0) to (1,1) => dist2=2.
    if abs(float(net_2d.cortex_dist2[0, 4]) - 2.0) > 1e-6:
        raise AssertionError(
            f"2D cortex metric test failed: dist2(0,4)={float(net_2d.cortex_dist2[0, 4]):.3f} (expected 2)"
        )
    report.append("Test 9 (2D cortex geometry): dist2 wrap/metric OK (3×3, torus)")

    # --- Test 10: RGC DoG isotropy (avoid edge-induced oblique bias) ---
    # For a fixed-frequency drifting grating, the RGC front-end should not introduce large
    # orientation-dependent energy differences, otherwise STDP is systematically biased.
    p_iso = Params(N=8, M=1, seed=1)
    net_iso = RgcLgnV1Network(p_iso)
    thetas_iso = np.linspace(0, 180 - 180 / 12, 12).astype(np.float32)
    rng_iso = np.random.default_rng(0)
    phases = rng_iso.uniform(0.0, 2.0 * math.pi, size=250).astype(np.float32)
    energies = np.zeros_like(thetas_iso, dtype=np.float64)
    for i, th in enumerate(thetas_iso):
        acc = 0.0
        for ph in phases:
            drive = net_iso.rgc_drive_grating(float(th), t_ms=0.0, phase=float(ph), contrast=1.0)
            acc += float(np.mean(drive.astype(np.float64) ** 2))
        energies[i] = acc / float(len(phases))
    e_ratio = float(energies.max() / (energies.min() + 1e-12))
    if e_ratio > 1.01:
        raise AssertionError(f"RGC isotropy test failed: energy max/min={e_ratio:.4f} (expected <= 1.01)")
    report.append(f"Test 10 (RGC DoG isotropy): energy max/min={e_ratio:.4f} (<=1.01)")

    # --- Test 11: Preferred-orientation diversity (hypercolumn should cover orientation space) ---
    # Use a low-discrepancy orientation schedule to avoid finite-sample skews dominating learning.
    p_div = Params(N=8, M=32, seed=1, segment_ms=300, v1_bias_eta=0.0)
    net_div = RgcLgnV1Network(p_div)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta_step = 180.0 / phi
    theta0 = 0.0
    train_segments_div = 250
    for s in range(train_segments_div):
        th = float((theta0 + s * theta_step) % 180.0)
        net_div.run_segment(th, plastic=True)

    rates_div = net_div.evaluate_tuning(thetas, repeats=3, contrast=1.0)
    osi_div, pref_div = compute_osi(rates_div, thetas)
    tuned_div = (osi_div >= 0.3)
    prefs = pref_div[tuned_div]
    if prefs.size < 20:
        raise AssertionError(f"Preference diversity test failed: tuned={int(prefs.size)}/32 (expected >= 20)")
    r_pref, mu_pref = circ_mean_resultant_180(prefs)
    gap = max_circ_gap_180(prefs)
    if r_pref > 0.45 or gap > 75.0:
        raise AssertionError(
            f"Preference diversity test failed: resultant={r_pref:.3f} (<=0.45), max_gap={gap:.1f}° (<=75°)"
        )
    report.append(
        f"Test 11 (Pref diversity): tuned={int(prefs.size)}/32, resultant={r_pref:.3f}, max_gap={gap:.1f}°, mean={mu_pref:.1f}°"
    )

    # --- Test 12: Alternative developmental stimuli (smoke test) ---
    p_alt = Params(N=8, M=4, seed=1, segment_ms=60, v1_bias_eta=0.0)
    net_alt = RgcLgnV1Network(p_alt)
    c1 = net_alt.run_segment_sparse_spots(plastic=False, contrast=1.0)
    c2 = net_alt.run_segment_white_noise(plastic=False, contrast=1.0)
    if not (c1.shape == (p_alt.M,) and c2.shape == (p_alt.M,)):
        raise AssertionError("Alt-stimulus smoke test failed: bad count shapes")
    if int(c1.sum()) < 0 or int(c2.sum()) < 0:
        raise AssertionError("Alt-stimulus smoke test failed: negative counts")
    report.append("Test 12 (Alt stimuli smoke): sparse_spots + white_noise ran")

    # --- Test 13: Alternative stimuli actually drive spikes (avoid silent-training regimes) ---
    p_alt2 = Params(N=8, M=8, seed=1, segment_ms=300, v1_bias_eta=0.0)
    net_alt2 = RgcLgnV1Network(p_alt2)
    cnt_noise = net_alt2.run_segment_white_noise_counts(plastic=False, contrast=1.0)
    if int(cnt_noise["v1_counts"].sum()) <= 0:
        raise AssertionError(
            "Alt-stimulus drive test failed (white_noise): V1 produced 0 spikes in one segment "
            f"(lgn_spikes={int(cnt_noise['lgn_counts'].sum())})"
        )
    net_alt2 = RgcLgnV1Network(p_alt2)
    cnt_spots = net_alt2.run_segment_sparse_spots_counts(plastic=False, contrast=1.0)
    if int(cnt_spots["v1_counts"].sum()) <= 0:
        raise AssertionError(
            "Alt-stimulus drive test failed (sparse_spots): V1 produced 0 spikes in one segment "
            f"(lgn_spikes={int(cnt_spots['lgn_counts'].sum())})"
        )
    report.append(
        "Test 13 (Alt stimuli drive): "
        f"white_noise V1={int(cnt_noise['v1_counts'].sum())}, "
        f"sparse_spots V1={int(cnt_spots['v1_counts'].sum())}"
    )

    # --- Test 14: White-noise training yields nontrivial OSI when probed with gratings ---
    p_wn = Params(N=8, M=8, seed=1, segment_ms=300, v1_bias_eta=0.0)
    net_wn = RgcLgnV1Network(p_wn)
    for _ in range(120):
        net_wn.run_segment_white_noise(plastic=True, contrast=1.0)
    rates_wn = net_wn.evaluate_tuning(thetas, repeats=5, contrast=1.0)
    osi_wn, _ = compute_osi(rates_wn, thetas)
    mean_osi_wn = float(osi_wn.mean())
    if mean_osi_wn < 0.15:
        raise AssertionError(f"Alt-stimulus OSI test failed (white_noise): mean OSI={mean_osi_wn:.3f} (expected >= 0.15)")
    if float(rates_wn.mean()) <= 0.05:
        raise AssertionError(f"Alt-stimulus OSI test failed (white_noise): mean rate={float(rates_wn.mean()):.3f} Hz (too spike-sparse)")
    report.append(f"Test 14 (White-noise OSI): mean OSI={mean_osi_wn:.3f}, max OSI={float(osi_wn.max()):.3f}, mean rate={float(rates_wn.mean()):.3f} Hz")
    proj_kwargs_wn: dict = {}
    if bool(getattr(net_wn.p, "rgc_separate_onoff_mosaics", False)):
        proj_kwargs_wn = dict(
            X_on=net_wn.X_on,
            Y_on=net_wn.Y_on,
            X_off=net_wn.X_off,
            Y_off=net_wn.Y_off,
            sigma=float(getattr(net_wn.p, "rgc_center_sigma", 0.5)),
        )
    rf_ori_wn, _ = rf_fft_orientation_metrics(net_wn.W, p_wn.N, on_to_off=net_wn.on_to_off, **proj_kwargs_wn)
    rf_amp_wn = rf_grating_match_tuning(
        net_wn.W,
        p_wn.N,
        float(p_wn.spatial_freq),
        thetas,
        on_to_off=net_wn.on_to_off,
        **proj_kwargs_wn,
    )
    rf_osi_wn, _ = compute_osi(rf_amp_wn, thetas)
    w_corr_wn = onoff_weight_corr(net_wn.W, p_wn.N, on_to_off=net_wn.on_to_off, **proj_kwargs_wn)
    if float(rf_osi_wn.mean()) < 0.18:
        raise AssertionError(
            f"Alt-stimulus RF test failed (white_noise): weight grating-match mean OSI={float(rf_osi_wn.mean()):.3f} (expected >= 0.18)"
        )
    if float(w_corr_wn.mean()) > -0.05:
        raise AssertionError(
            f"Alt-stimulus RF test failed (white_noise): ON/OFF weight corr mean={float(w_corr_wn.mean()):+.3f} (expected <= -0.05)"
        )
    report.append(
        f"      (weights) rf_orientedness mean={float(rf_ori_wn.mean()):.3f}, "
        f"grating-match OSI mean={float(rf_osi_wn.mean()):.3f}, "
        f"ON/OFF corr mean={float(w_corr_wn.mean()):+.3f}"
    )

    # --- Test 15: Sparse-spot training yields nontrivial OSI when probed with gratings ---
    p_ss = Params(N=8, M=8, seed=1, segment_ms=300, v1_bias_eta=0.0)
    net_ss = RgcLgnV1Network(p_ss)
    for _ in range(120):
        net_ss.run_segment_sparse_spots(plastic=True, contrast=1.0)
    rates_ss = net_ss.evaluate_tuning(thetas, repeats=5, contrast=1.0)
    osi_ss, _ = compute_osi(rates_ss, thetas)
    mean_osi_ss = float(osi_ss.mean())
    if mean_osi_ss < 0.15:
        raise AssertionError(f"Alt-stimulus OSI test failed (sparse_spots): mean OSI={mean_osi_ss:.3f} (expected >= 0.15)")
    if float(rates_ss.mean()) <= 0.05:
        raise AssertionError(f"Alt-stimulus OSI test failed (sparse_spots): mean rate={float(rates_ss.mean()):.3f} Hz (too spike-sparse)")
    report.append(f"Test 15 (Sparse-spots OSI): mean OSI={mean_osi_ss:.3f}, max OSI={float(osi_ss.max()):.3f}, mean rate={float(rates_ss.mean()):.3f} Hz")
    proj_kwargs_ss: dict = {}
    if bool(getattr(net_ss.p, "rgc_separate_onoff_mosaics", False)):
        proj_kwargs_ss = dict(
            X_on=net_ss.X_on,
            Y_on=net_ss.Y_on,
            X_off=net_ss.X_off,
            Y_off=net_ss.Y_off,
            sigma=float(getattr(net_ss.p, "rgc_center_sigma", 0.5)),
        )
    rf_ori_ss, _ = rf_fft_orientation_metrics(net_ss.W, p_ss.N, on_to_off=net_ss.on_to_off, **proj_kwargs_ss)
    rf_amp_ss = rf_grating_match_tuning(
        net_ss.W,
        p_ss.N,
        float(p_ss.spatial_freq),
        thetas,
        on_to_off=net_ss.on_to_off,
        **proj_kwargs_ss,
    )
    rf_osi_ss, _ = compute_osi(rf_amp_ss, thetas)
    w_corr_ss = onoff_weight_corr(net_ss.W, p_ss.N, on_to_off=net_ss.on_to_off, **proj_kwargs_ss)
    if float(rf_osi_ss.mean()) < 0.20:
        raise AssertionError(
            f"Alt-stimulus RF test failed (sparse_spots): weight grating-match mean OSI={float(rf_osi_ss.mean()):.3f} (expected >= 0.20)"
        )
    if float(w_corr_ss.mean()) > -0.05:
        raise AssertionError(
            f"Alt-stimulus RF test failed (sparse_spots): ON/OFF weight corr mean={float(w_corr_ss.mean()):+.3f} (expected <= -0.05)"
        )
    report.append(
        f"      (weights) rf_orientedness mean={float(rf_ori_ss.mean()):.3f}, "
        f"grating-match OSI mean={float(rf_osi_ss.mean()):.3f}, "
        f"ON/OFF corr mean={float(w_corr_ss.mean()):+.3f}"
    )

    # --- Test 16: Separate ON/OFF mosaics mode (functional + OSI) ---
    p_mos = Params(
        N=8,
        M=4,
        seed=1,
        segment_ms=240,
        v1_bias_eta=0.0,
        rgc_separate_onoff_mosaics=True,
    )
    net_mos = RgcLgnV1Network(p_mos)
    for _ in range(200):
        th = float(net_mos.rng.uniform(0.0, 180.0))
        net_mos.run_segment(th, plastic=True)
    rates_mos = net_mos.evaluate_tuning(thetas, repeats=3, contrast=1.0)
    osi_mos, _ = compute_osi(rates_mos, thetas)
    mean_osi_mos = float(osi_mos.mean())
    if mean_osi_mos < 0.25:
        raise AssertionError(
            f"Separated-mosaic OSI test failed: mean OSI={mean_osi_mos:.3f} (expected >= 0.25)"
        )
    ang = getattr(net_mos, "rgc_onoff_offset_angle_deg", None)
    ang_s = "None" if ang is None else f"{float(ang):.1f}°"
    report.append(
        f"Test 16 (ON/OFF mosaics): mean OSI={mean_osi_mos:.3f}, angle={ang_s}"
    )

    # --- Test 17: Laminar L2/3 scaffold (apical gating affects L2/3 spiking) ---
    p_lam = Params(
        N=8,
        M=1,
        seed=1,
        segment_ms=200,
        v1_bias_eta=0.0,
        laminar_enabled=True,
        w_l4_l23=6.0,
        l4_l23_sigma=0.0,
        apical_gain=2.0,
        apical_threshold=0.5,
        apical_slope=0.1,
        n_som_per_ensemble=0,
        n_vip_per_ensemble=0,
        w_e_som=0.0,
        w_som_e=0.0,
        w_e_vip=0.0,
        w_vip_som=0.0,
    )
    net_lam = RgcLgnV1Network(p_lam)
    if net_lam.v1_l23 is None:
        raise AssertionError("Laminar scaffold test failed: v1_l23 not initialized")

    steps = int(p_lam.segment_ms / p_lam.dt_ms)
    theta = 90.0
    phase = 0.0
    rng_state = net_lam.rng.bit_generator.state

    def run_l23_spikes(apical: float) -> int:
        net_lam.reset_state()
        net_lam.rng.bit_generator.state = rng_state
        s = 0
        for k in range(steps):
            on_spk, off_spk = net_lam.rgc_spikes_grating(theta, t_ms=k * p_lam.dt_ms, phase=phase, contrast=2.0)
            net_lam.step(on_spk, off_spk, plastic=False, apical_drive=apical)
            s += int(net_lam.last_v1_l23_spk.sum())
        return int(s)

    l23_off = run_l23_spikes(0.0)
    l23_on = run_l23_spikes(100.0)
    if l23_on <= l23_off:
        raise AssertionError(
            f"Laminar apical-gating test failed: L2/3 spikes off={l23_off}, on={l23_on} (expected on>off)"
        )
    report.append(f"Test 17 (Laminar apical): L2/3 spikes off={l23_off}, on={l23_on} (contrast=2.0)")

    # Save a small numeric bundle + a human-readable report.
    np.savez_compressed(os.path.join(out_dir, "selftest_metrics.npz"),
                        thetas_deg=thetas.astype(np.float32),
                        rates_hz=rates.astype(np.float32),
                        osi=osi.astype(np.float32),
                        pref_deg=pref.astype(np.float32))
    report_path = os.path.join(out_dir, "selftest_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"[tests] Wrote: {report_path}")
    print("[tests] Summary:")
    for line in report:
        print(f"[tests]   {line}")

    print("[tests] All self-tests passed.")

def main() -> None:
    ap = argparse.ArgumentParser(description="Biologically plausible V1 STDP network")
    ap.add_argument("--out", type=str, default="runs/bio_plausible",
                    help="output directory")
    ap.add_argument("--train-segments", type=int, default=1000)
    ap.add_argument("--segment-ms", type=int, default=300)
    ap.add_argument("--N", type=int, default=8, help="Patch size NxN")
    ap.add_argument("--M", type=int, default=32, help="Number of V1 ensembles")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--species", type=str, default="generic",
                    help="metadata label for intended species/interpretation (no effect on dynamics)")
    ap.add_argument("--viz-every", type=int, default=50)
    ap.add_argument("--eval-K", type=int, default=12, help="Number of orientations to test")
    ap.add_argument("--eval-repeats", type=int, default=3)
    ap.add_argument("--baseline-repeats", type=int, default=7)
    ap.add_argument("--weight-smooth-sigma", type=float, default=1.0,
                    help="sigma for Gaussian smoothing in filter visualizations (0 disables)")
    ap.add_argument("--train-theta-schedule", type=str, default="low_discrepancy",
                    choices=["random", "low_discrepancy"],
                    help="orientation schedule during training (low_discrepancy reduces finite-sample skews)")
    ap.add_argument("--train-stimulus", type=str, default="grating",
                    choices=["grating", "sparse_spots", "white_noise"],
                    help="developmental stimulus during training (evaluation still uses gratings)")
    ap.add_argument("--train-contrast", type=float, default=1.0,
                    help="stimulus contrast scalar during training (applied to chosen train-stimulus)")
    ap.add_argument("--spots-density", type=float, default=None,
                    help="sparse_spots: fraction of pixels active per frame (0..1)")
    ap.add_argument("--spots-frame-ms", type=float, default=None,
                    help="sparse_spots: refresh period (ms)")
    ap.add_argument("--spots-amp", type=float, default=None,
                    help="sparse_spots: luminance amplitude for each spot (+/-amp)")
    ap.add_argument("--spots-sigma", type=float, default=None,
                    help="sparse_spots: spot size (pixels); <=0 => single-pixel spots, >0 => Gaussian blobs")
    ap.add_argument("--noise-sigma", type=float, default=None,
                    help="white_noise: per-pixel luminance std (Gaussian)")
    ap.add_argument("--noise-clip", type=float, default=None,
                    help="white_noise: clip luminance to [-clip,+clip]; 0 disables")
    ap.add_argument("--noise-frame-ms", type=float, default=None,
                    help="white_noise: refresh period (ms)")
    ap.add_argument("--init-mode", type=str, default="random",
                    choices=["random", "near_uniform"])
    ap.add_argument("--spatial-freq", type=float, default=None,
                    help="grating spatial frequency (cycles per pixel unit)")
    ap.add_argument("--temporal-freq", type=float, default=None,
                    help="grating temporal frequency (Hz)")
    ap.add_argument("--base-rate", type=float, default=None,
                    help="RGC baseline Poisson rate (Hz)")
    ap.add_argument("--gain-rate", type=float, default=None,
                    help="RGC gain (Hz per unit drive)")
    ap.add_argument("--no-rgc-center-surround", action="store_true",
                    help="disable RGC center–surround DoG filtering (less biological; mostly for debugging)")
    ap.add_argument("--rgc-center-sigma", type=float, default=None,
                    help="RGC DoG center sigma (pixels)")
    ap.add_argument("--rgc-surround-sigma", type=float, default=None,
                    help="RGC DoG surround sigma (pixels)")
    ap.add_argument("--rgc-dog-norm", type=str, default=None,
                    choices=["none", "l1", "l2"],
                    help="RGC DoG kernel row normalization")
    ap.add_argument("--rgc-dog-impl", type=str, default=None,
                    choices=["matrix", "padded_fft"],
                    help="RGC DoG implementation (padded_fft avoids edge-induced orientation bias)")
    ap.add_argument("--rgc-dog-pad", type=int, default=None,
                    help="padding (pixels) for padded_fft DoG; 0/None => auto")
    ap.add_argument("--rgc-pos-jitter", type=float, default=None,
                    help="RGC mosaic position jitter (fraction of pixel spacing; 0 disables)")
    ap.add_argument("--separate-onoff-mosaics", action="store_true",
                    help="use distinct ON and OFF RGC mosaics (biological; avoids perfectly co-registered ON/OFF pairs)")
    ap.add_argument("--onoff-offset", type=float, default=None,
                    help="ON/OFF mosaic offset magnitude (pixels) when --separate-onoff-mosaics is enabled")
    ap.add_argument("--onoff-offset-angle-deg", type=float, default=None,
                    help="ON/OFF mosaic offset angle (deg); if omitted, choose a seeded random angle")
    ap.add_argument("--rgc-temporal-filter", action="store_true",
                    help="enable simple biphasic temporal filtering in RGC drive (fast - slow)")
    ap.add_argument("--rgc-tau-fast", type=float, default=None,
                    help="RGC temporal filter fast time constant (ms)")
    ap.add_argument("--rgc-tau-slow", type=float, default=None,
                    help="RGC temporal filter slow time constant (ms)")
    ap.add_argument("--rgc-temporal-gain", type=float, default=None,
                    help="RGC temporal filter gain multiplier")
    ap.add_argument("--rgc-refractory-ms", type=float, default=None,
                    help="RGC absolute refractory period (ms); 0 disables")
    ap.add_argument("--no-lgn-pooling", action="store_true",
                    help="disable explicit retinogeniculate pooling (fallback to one-to-one RGC->LGN)")
    ap.add_argument("--lgn-pool-sigma-center", type=float, default=None,
                    help="RGC->LGN same-sign center pooling sigma (pixels)")
    ap.add_argument("--lgn-pool-sigma-surround", type=float, default=None,
                    help="RGC->LGN opposite-sign surround pooling sigma (pixels)")
    ap.add_argument("--lgn-pool-same-gain", type=float, default=None,
                    help="gain for same-sign retinogeniculate pooling")
    ap.add_argument("--lgn-pool-opponent-gain", type=float, default=None,
                    help="gain for opposite-sign retinogeniculate pooling (antagonistic)")
    ap.add_argument("--lgn-rgc-tau-ms", type=float, default=None,
                    help="LGN pre-integration time constant for pooled RGC drive (ms)")
    ap.add_argument("--no-tc-stp", action="store_true",
                    help="disable thalamocortical short-term depression (LGN->V1 E)")
    ap.add_argument("--tc-stp-u", type=float, default=None,
                    help="thalamocortical STP depletion fraction per spike (0..1)")
    ap.add_argument("--tc-stp-tau-rec", type=float, default=None,
                    help="thalamocortical STP recovery time constant (ms)")
    ap.add_argument("--no-tc-stp-pv", action="store_true",
                    help="disable thalamocortical short-term depression (LGN->PV feedforward inhibition)")
    ap.add_argument("--tc-stp-pv-u", type=float, default=None,
                    help="LGN->PV STP depletion fraction per spike (0..1)")
    ap.add_argument("--tc-stp-pv-tau-rec", type=float, default=None,
                    help="LGN->PV STP recovery time constant (ms)")
    ap.add_argument("--lgn-sigma-e", type=float, default=None,
                    help="retinotopic weight-cap sigma for E (pixels; 0 disables)")
    ap.add_argument("--lgn-sigma-pv", type=float, default=None,
                    help="retinotopic weight-cap sigma for PV (pixels; 0 disables)")
    ap.add_argument("--lgn-sigma-pp", type=float, default=None,
                    help="retinotopic weight-cap sigma for push–pull (pixels; 0 disables)")
    ap.add_argument("--pv-in-sigma", type=float, default=None,
                    help="E->PV connectivity sigma in cortical-distance units (0 => private PV)")
    ap.add_argument("--pv-out-sigma", type=float, default=None,
                    help="PV->E connectivity sigma in cortical-distance units (0 => private PV)")
    ap.add_argument("--pv-pv-sigma", type=float, default=None,
                    help="PV->PV coupling sigma in cortical-distance units (0 disables)")
    ap.add_argument("--w-pv-pv", type=float, default=None,
                    help="PV->PV inhibitory current increment (0 disables)")
    ap.add_argument("--n-vip-per-ensemble", type=int, default=None,
                    help="VIP interneurons per ensemble (0 disables)")
    ap.add_argument("--w-e-vip", type=float, default=None,
                    help="E->VIP excitatory current increment")
    ap.add_argument("--w-vip-som", type=float, default=None,
                    help="VIP->SOM inhibitory current increment")
    ap.add_argument("--vip-bias-current", type=float, default=None,
                    help="VIP tonic bias current (models state/top-down)")
    ap.add_argument("--tau-apical", type=float, default=None,
                    help="apical/feedback-like excitatory conductance time constant (ms)")
    ap.add_argument("--apical-gain", type=float, default=None,
                    help="apical multiplicative gain (0 disables apical modulation)")
    ap.add_argument("--apical-threshold", type=float, default=None,
                    help="apical gating threshold (conductance units)")
    ap.add_argument("--apical-slope", type=float, default=None,
                    help="apical gating slope (conductance units)")
    ap.add_argument("--laminar", action="store_true",
                    help="enable minimal laminar L4->L2/3 scaffold (apical targets L2/3 when enabled)")
    ap.add_argument("--w-l4-l23", type=float, default=None,
                    help="laminar: L4->L2/3 basal current weight (same units as W_e_e)")
    ap.add_argument("--l4-l23-sigma", type=float, default=None,
                    help="laminar: spread of L4->L2/3 projections over cortex_dist2 (0 => same-ensemble only)")
    ap.add_argument("--tc-conn-fraction-e", type=float, default=None,
                    help="fraction of LGN afferents present per excitatory neuron (0..1]; <1 enforces sparse anatomical mask)")
    ap.add_argument("--tc-conn-fraction-pv", type=float, default=None,
                    help="fraction of LGN afferents present per PV interneuron (0..1]")
    ap.add_argument("--tc-conn-fraction-pp", type=float, default=None,
                    help="fraction of LGN afferents present per PP interneuron (0..1]")
    ap.add_argument("--tc-no-balance-onoff", action="store_true",
                    help="when using sparse thalamocortical connectivity, do not enforce balanced ON/OFF sampling")
    ap.add_argument("--pp-onoff-swap", action=argparse.BooleanOptionalAction, default=None,
                    help="enable explicit ON/OFF swap into PP interneurons (adds built-in phase opposition)")
    ap.add_argument("--a-split", type=float, default=None,
                    help="ON/OFF split-competition strength (0 disables; developmental constraint)")
    ap.add_argument("--split-constraint-rate", type=float, default=None,
                    help="ON/OFF split-constraint scaling rate (0 disables; local per-neuron)")
    ap.add_argument("--split-constraint-clip", type=float, default=None,
                    help="ON/OFF split-constraint multiplicative clip per application (e.g., 0.02 => [0.98,1.02])")
    ap.add_argument("--no-split-equalize-onoff", action="store_true",
                    help="do not equalize ON/OFF resource targets for split constraint (use initial sums)")
    ap.add_argument("--no-split-constraint", action="store_true",
                    help="disable ON/OFF split-constraint scaling (equivalent to --split-constraint-rate 0)")
    ap.add_argument("--split-overlap-adaptive", action="store_true",
                    help="enable adaptive gain of ON/OFF split competition based on ON/OFF overlap")
    ap.add_argument("--no-split-overlap-adaptive", action="store_true",
                    help="disable adaptive gain of ON/OFF split competition based on overlap")
    ap.add_argument("--split-overlap-min", type=float, default=None,
                    help="minimum multiplier for overlap-adaptive split competition")
    ap.add_argument("--split-overlap-max", type=float, default=None,
                    help="maximum multiplier for overlap-adaptive split competition")
    ap.add_argument("--run-tests", action="store_true",
                    help="run built-in self-tests and exit")

    args = ap.parse_args()
    safe_mkdir(args.out)

    if args.run_tests:
        run_self_tests(os.path.join(args.out, "self_tests"))
        return

    # Create network
    p_kwargs: dict = dict(
        N=args.N,
        M=args.M,
        seed=args.seed,
        species=str(args.species),
        train_segments=args.train_segments,
        segment_ms=args.segment_ms,
        train_stimulus=str(args.train_stimulus),
        train_contrast=float(args.train_contrast),
    )
    if args.spatial_freq is not None:
        p_kwargs["spatial_freq"] = float(args.spatial_freq)
    if args.temporal_freq is not None:
        p_kwargs["temporal_freq"] = float(args.temporal_freq)
    if args.base_rate is not None:
        p_kwargs["base_rate"] = float(args.base_rate)
    if args.gain_rate is not None:
        p_kwargs["gain_rate"] = float(args.gain_rate)
    if args.spots_density is not None:
        p_kwargs["spots_density"] = float(args.spots_density)
    if args.spots_frame_ms is not None:
        p_kwargs["spots_frame_ms"] = float(args.spots_frame_ms)
    if args.spots_amp is not None:
        p_kwargs["spots_amp"] = float(args.spots_amp)
    if args.spots_sigma is not None:
        p_kwargs["spots_sigma"] = float(args.spots_sigma)
    if args.noise_sigma is not None:
        p_kwargs["noise_sigma"] = float(args.noise_sigma)
    if args.noise_clip is not None:
        p_kwargs["noise_clip"] = float(args.noise_clip)
    if args.noise_frame_ms is not None:
        p_kwargs["noise_frame_ms"] = float(args.noise_frame_ms)
    if args.no_rgc_center_surround:
        p_kwargs["rgc_center_surround"] = False
    if args.rgc_center_sigma is not None:
        p_kwargs["rgc_center_sigma"] = float(args.rgc_center_sigma)
    if args.rgc_surround_sigma is not None:
        p_kwargs["rgc_surround_sigma"] = float(args.rgc_surround_sigma)
    if args.rgc_dog_norm is not None:
        p_kwargs["rgc_dog_norm"] = str(args.rgc_dog_norm)
    if args.rgc_dog_impl is not None:
        p_kwargs["rgc_dog_impl"] = str(args.rgc_dog_impl)
    if args.rgc_dog_pad is not None:
        p_kwargs["rgc_dog_pad"] = int(args.rgc_dog_pad)
    if args.rgc_pos_jitter is not None:
        p_kwargs["rgc_pos_jitter"] = float(args.rgc_pos_jitter)
    if args.separate_onoff_mosaics:
        p_kwargs["rgc_separate_onoff_mosaics"] = True
    if args.onoff_offset is not None:
        p_kwargs["rgc_onoff_offset"] = float(args.onoff_offset)
    if args.onoff_offset_angle_deg is not None:
        p_kwargs["rgc_onoff_offset_angle_deg"] = float(args.onoff_offset_angle_deg)
    if args.rgc_temporal_filter:
        p_kwargs["rgc_temporal_filter"] = True
    if args.rgc_tau_fast is not None:
        p_kwargs["rgc_tau_fast"] = float(args.rgc_tau_fast)
    if args.rgc_tau_slow is not None:
        p_kwargs["rgc_tau_slow"] = float(args.rgc_tau_slow)
    if args.rgc_temporal_gain is not None:
        p_kwargs["rgc_temporal_gain"] = float(args.rgc_temporal_gain)
    if args.rgc_refractory_ms is not None:
        p_kwargs["rgc_refractory_ms"] = float(args.rgc_refractory_ms)
    if args.no_lgn_pooling:
        p_kwargs["lgn_pooling"] = False
    if args.lgn_pool_sigma_center is not None:
        p_kwargs["lgn_pool_sigma_center"] = float(args.lgn_pool_sigma_center)
    if args.lgn_pool_sigma_surround is not None:
        p_kwargs["lgn_pool_sigma_surround"] = float(args.lgn_pool_sigma_surround)
    if args.lgn_pool_same_gain is not None:
        p_kwargs["lgn_pool_same_gain"] = float(args.lgn_pool_same_gain)
    if args.lgn_pool_opponent_gain is not None:
        p_kwargs["lgn_pool_opponent_gain"] = float(args.lgn_pool_opponent_gain)
    if args.lgn_rgc_tau_ms is not None:
        p_kwargs["lgn_rgc_tau_ms"] = float(args.lgn_rgc_tau_ms)
    if args.no_tc_stp:
        p_kwargs["tc_stp_enabled"] = False
    if args.tc_stp_u is not None:
        p_kwargs["tc_stp_u"] = float(args.tc_stp_u)
    if args.tc_stp_tau_rec is not None:
        p_kwargs["tc_stp_tau_rec"] = float(args.tc_stp_tau_rec)
    if args.no_tc_stp_pv:
        p_kwargs["tc_stp_pv_enabled"] = False
    if args.tc_stp_pv_u is not None:
        p_kwargs["tc_stp_pv_u"] = float(args.tc_stp_pv_u)
    if args.tc_stp_pv_tau_rec is not None:
        p_kwargs["tc_stp_pv_tau_rec"] = float(args.tc_stp_pv_tau_rec)
    if args.lgn_sigma_e is not None:
        p_kwargs["lgn_sigma_e"] = float(args.lgn_sigma_e)
    if args.lgn_sigma_pv is not None:
        p_kwargs["lgn_sigma_pv"] = float(args.lgn_sigma_pv)
    if args.lgn_sigma_pp is not None:
        p_kwargs["lgn_sigma_pp"] = float(args.lgn_sigma_pp)
    if args.pv_in_sigma is not None:
        p_kwargs["pv_in_sigma"] = float(args.pv_in_sigma)
    if args.pv_out_sigma is not None:
        p_kwargs["pv_out_sigma"] = float(args.pv_out_sigma)
    if args.pv_pv_sigma is not None:
        p_kwargs["pv_pv_sigma"] = float(args.pv_pv_sigma)
    if args.w_pv_pv is not None:
        p_kwargs["w_pv_pv"] = float(args.w_pv_pv)
    if args.n_vip_per_ensemble is not None:
        p_kwargs["n_vip_per_ensemble"] = int(args.n_vip_per_ensemble)
    if args.w_e_vip is not None:
        p_kwargs["w_e_vip"] = float(args.w_e_vip)
    if args.w_vip_som is not None:
        p_kwargs["w_vip_som"] = float(args.w_vip_som)
    if args.vip_bias_current is not None:
        p_kwargs["vip_bias_current"] = float(args.vip_bias_current)
    if args.tau_apical is not None:
        p_kwargs["tau_apical"] = float(args.tau_apical)
    if args.apical_gain is not None:
        p_kwargs["apical_gain"] = float(args.apical_gain)
    if args.apical_threshold is not None:
        p_kwargs["apical_threshold"] = float(args.apical_threshold)
    if args.apical_slope is not None:
        p_kwargs["apical_slope"] = float(args.apical_slope)
    if args.laminar:
        p_kwargs["laminar_enabled"] = True
    if args.w_l4_l23 is not None:
        p_kwargs["w_l4_l23"] = float(args.w_l4_l23)
    if args.l4_l23_sigma is not None:
        p_kwargs["l4_l23_sigma"] = float(args.l4_l23_sigma)
    if args.tc_conn_fraction_e is not None:
        p_kwargs["tc_conn_fraction_e"] = float(args.tc_conn_fraction_e)
    if args.tc_conn_fraction_pv is not None:
        p_kwargs["tc_conn_fraction_pv"] = float(args.tc_conn_fraction_pv)
    if args.tc_conn_fraction_pp is not None:
        p_kwargs["tc_conn_fraction_pp"] = float(args.tc_conn_fraction_pp)
    if args.tc_no_balance_onoff:
        p_kwargs["tc_conn_balance_onoff"] = False
    if args.pp_onoff_swap is not None:
        p_kwargs["pp_onoff_swap"] = bool(args.pp_onoff_swap)
    if args.a_split is not None:
        p_kwargs["A_split"] = float(args.a_split)
    if args.split_constraint_rate is not None:
        p_kwargs["split_constraint_rate"] = float(args.split_constraint_rate)
    if args.split_constraint_clip is not None:
        p_kwargs["split_constraint_clip"] = float(args.split_constraint_clip)
    if args.no_split_equalize_onoff:
        p_kwargs["split_constraint_equalize_onoff"] = False
    if args.no_split_constraint:
        p_kwargs["split_constraint_rate"] = 0.0
    if args.split_overlap_adaptive:
        p_kwargs["split_overlap_adaptive"] = True
    if args.no_split_overlap_adaptive:
        p_kwargs["split_overlap_adaptive"] = False
    if args.split_overlap_min is not None:
        p_kwargs["split_overlap_min"] = float(args.split_overlap_min)
    if args.split_overlap_max is not None:
        p_kwargs["split_overlap_max"] = float(args.split_overlap_max)

    p = Params(**p_kwargs)
    net = RgcLgnV1Network(p, init_mode=args.init_mode)

    print(f"[init] Biologically plausible RGC->LGN->V1 network")
    print(f"[init] N={p.N} (patch), M={p.M} (ensembles), n_lgn={net.n_lgn}")
    print(f"[init] Neuron types: LGN=TC(Izhikevich), V1=RS, PV=FS, SOM=LTS")
    print(f"[init] Plasticity: Triplet STDP + iSTDP (PV) + STP (TC) + optional synaptic scaling={'ON' if p.homeostasis_rate>0 else 'OFF'}")
    print(f"[init] Inhibition: PV (LGN-driven + local E-driven) + SOM (lateral)")
    if (float(p.pv_in_sigma) > 0.0) or (float(p.pv_out_sigma) > 0.0):
        print(f"[init] PV connectivity: spread (in_sigma={float(p.pv_in_sigma):.2f}, out_sigma={float(p.pv_out_sigma):.2f})")
    if (float(p.pv_pv_sigma) > 0.0) and (float(p.w_pv_pv) > 0.0):
        print(f"[init] PV↔PV coupling: ON (sigma={float(p.pv_pv_sigma):.2f}, w_pv_pv={float(p.w_pv_pv):.3f})")
    if int(p.n_vip_per_ensemble) > 0:
        print(
            f"[init] VIP disinhibition: ON (n_vip/ens={int(p.n_vip_per_ensemble)}, "
            f"w_e_vip={float(p.w_e_vip):.3f}, w_vip_som={float(p.w_vip_som):.3f}, "
            f"vip_bias={float(p.vip_bias_current):.3f})"
        )
    if float(p.apical_gain) > 0.0:
        print(
            f"[init] Apical modulation: ON (tau_apical={float(p.tau_apical):.1f} ms, "
            f"gain={float(p.apical_gain):.3f}, thr={float(p.apical_threshold):.3f}, slope={float(p.apical_slope):.3f})"
        )
    if p.rgc_center_surround:
        if str(p.rgc_dog_impl).lower() == "padded_fft":
            print(f"[init] RGC DoG: padded_fft (pad={net._rgc_pad}, norm={p.rgc_dog_norm}, jitter={p.rgc_pos_jitter:.3f})")
        else:
            print(f"[init] RGC DoG: matrix (norm={p.rgc_dog_norm}, jitter={p.rgc_pos_jitter:.3f})")
    else:
        print(f"[init] RGC DoG: OFF (raw grating -> ON/OFF Poisson)")
    if p.rgc_separate_onoff_mosaics:
        ang = net.rgc_onoff_offset_angle_deg
        ang_s = "seeded-random" if ang is None else f"{float(ang):.1f}°"
        print(f"[init] RGC mosaics: separate ON/OFF (offset={float(p.rgc_onoff_offset):.2f} px, angle={ang_s})")
    else:
        print("[init] RGC mosaics: co-registered ON/OFF")
    if p.rgc_temporal_filter or float(p.rgc_refractory_ms) > 0.0:
        print(
            f"[init] RGC temporal: filter={'ON' if p.rgc_temporal_filter else 'OFF'} "
            f"(tau_fast={float(p.rgc_tau_fast):.1f} ms, tau_slow={float(p.rgc_tau_slow):.1f} ms, gain={float(p.rgc_temporal_gain):.2f}) "
            f"| refractory={float(p.rgc_refractory_ms):.1f} ms"
        )
    if p.lgn_pooling:
        print(
            f"[init] Retinogeniculate pooling: ON "
            f"(center sigma={float(p.lgn_pool_sigma_center):.2f}, surround sigma={float(p.lgn_pool_sigma_surround):.2f}, "
            f"same_gain={float(p.lgn_pool_same_gain):.2f}, opp_gain={float(p.lgn_pool_opponent_gain):.2f}, "
            f"tau={float(p.lgn_rgc_tau_ms):.1f} ms)"
        )
    else:
        print("[init] Retinogeniculate pooling: OFF (one-to-one RGC->LGN)")
    print(f"[init] Train stimulus: {p.train_stimulus} (contrast={p.train_contrast:.3f})")
    if p.train_stimulus == "grating":
        print(f"[init] Training θ schedule: {args.train_theta_schedule}")
    else:
        print(f"[init] Training θ schedule: {args.train_theta_schedule} (ignored for train-stimulus='{p.train_stimulus}')")
    print(f"[init] init-mode = {args.init_mode}")
    cycles_across = float(p.spatial_freq) * float(p.N - 1)
    if cycles_across < 1.0:
        print(
            f"[warn] spatial_freq={p.spatial_freq:.3f} gives only ~{cycles_across:.2f} cycles across the N={p.N} patch; "
            "this can bias learned RFs toward coarse/diagonal gradients. Consider increasing --spatial-freq or N."
        )
    params_path = os.path.join(args.out, "params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(asdict(p), f, indent=2, sort_keys=True)
    print(f"[init] wrote params: {params_path}")

    thetas = np.linspace(0, 180 - 180 / args.eval_K, args.eval_K)

    proj_kwargs: dict = {}
    if bool(getattr(p, "rgc_separate_onoff_mosaics", False)):
        proj_kwargs = dict(
            X_on=net.X_on,
            Y_on=net.Y_on,
            X_off=net.X_off,
            Y_off=net.Y_off,
            sigma=float(getattr(p, "rgc_center_sigma", 0.5)),
        )

    # --- Baseline evaluation ---
    print("\n[baseline] Evaluating tuning at initialization...")
    rates0 = net.evaluate_tuning(thetas, repeats=args.baseline_repeats)
    osi0, pref0 = compute_osi(rates0, thetas)
    rf_ori0, rf_pref0 = rf_fft_orientation_metrics(net.W, p.N, on_to_off=net.on_to_off, **proj_kwargs)
    rf_amp0 = rf_grating_match_tuning(
        net.W,
        p.N,
        float(p.spatial_freq),
        thetas,
        on_to_off=net.on_to_off,
        **proj_kwargs,
    )
    rf_osi0, _ = compute_osi(rf_amp0, thetas)
    w_corr0 = onoff_weight_corr(net.W, p.N, on_to_off=net.on_to_off, **proj_kwargs)

    print(f"[seg {0:4d}] mean rate={rates0.mean():.3f} Hz | mean OSI={osi0.mean():.3f} | max OSI={osi0.max():.3f}")
    print(
        f"          RF(weight): orientedness mean={float(rf_ori0.mean()):.3f} "
        f"| grating-match OSI mean={float(rf_osi0.mean()):.3f} "
        f"(frac>0.45={float((rf_ori0>0.45).mean()):.2f}) | "
        f"ON/OFF weight corr mean={float(w_corr0.mean()):+.3f}"
    )
    print(f"          prefs(deg) = {np.round(pref0, 1)}")
    print("          NOTE: Nonzero OSI at init is expected from random RF structure")
    tuned0 = (osi0 >= 0.3)
    near90_0 = (circ_diff_180(pref0, 90.0) <= 10.0) & tuned0
    if tuned0.any():
        print(f"          tuned(OSI≥0.3) = {int(tuned0.sum())}/{p.M} | near 90° (±10°) = {int(near90_0.sum())}/{int(tuned0.sum())}")
        r0, mu0 = circ_mean_resultant_180(pref0[tuned0])
        gap0 = max_circ_gap_180(pref0[tuned0])
        print(f"          pref diversity: resultant={r0:.3f}, max_gap={gap0:.1f}° (mean={mu0:.1f}°)")

    W_init = net.W.copy()
    plot_weight_maps(net.W, p.N, os.path.join(args.out, "weights_seg0000.png"),
                     title="LGN->V1 weights at init (segment 0)")
    plot_weight_maps(net.W, p.N, os.path.join(args.out, "weights_seg0000_smoothed.png"),
                     title=f"LGN->V1 weights at init (segment 0, Gaussian sigma={float(args.weight_smooth_sigma):.2f})",
                     smooth_sigma=float(args.weight_smooth_sigma))
    plot_tuning(rates0, thetas, osi0, pref0,
                os.path.join(args.out, "tuning_seg0000.png"),
                title="Baseline tuning (segment 0, before learning)")
    plot_pref_hist(pref0, osi0,
                   os.path.join(args.out, "pref_hist_seg0000.png"),
                   title="Preferred orientation histogram (segment 0)")
    save_eval_npz(os.path.join(args.out, "eval_seg0000.npz"),
                  thetas_deg=thetas, rates_hz=rates0, osi=osi0, pref_deg=pref0, net=net)

    # Tracking
    seg_hist = [0]
    osi_hist = [float(osi0.mean())]
    rate_hist = [float(rates0.mean())]
    pv_rate_hist = []
    som_rate_hist = []

    if p.train_segments == 0:
        print("[final] train-segments=0, no learning occurred")
        print(f"[done] outputs written to: {args.out}")
        return

    # --- Training ---
    print("\n[training] Starting STDP training...")
    theta_offset = 0.0
    theta_step = 0.0
    if p.train_stimulus == "grating" and args.train_theta_schedule == "low_discrepancy":
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        theta_step = 180.0 / phi
        theta_offset = float(net.rng.uniform(0.0, 180.0))

    for s in range(1, p.train_segments + 1):
        if p.train_stimulus == "grating":
            if args.train_theta_schedule == "random":
                th = float(net.rng.uniform(0.0, 180.0))
            else:
                th = float((theta_offset + (s - 1) * theta_step) % 180.0)
            net.run_segment(th, plastic=True, contrast=p.train_contrast)
        elif p.train_stimulus == "sparse_spots":
            net.run_segment_sparse_spots(plastic=True, contrast=p.train_contrast)
        elif p.train_stimulus == "white_noise":
            net.run_segment_white_noise(plastic=True, contrast=p.train_contrast)
        else:
            raise ValueError(f"Unknown train_stimulus: {p.train_stimulus!r}")

        if (s % args.viz_every) == 0 or s == p.train_segments:
            rates = net.evaluate_tuning(thetas, repeats=args.eval_repeats)
            osi, pref = compute_osi(rates, thetas)
            rf_ori, rf_pref = rf_fft_orientation_metrics(net.W, p.N, on_to_off=net.on_to_off, **proj_kwargs)
            rf_amp = rf_grating_match_tuning(
                net.W,
                p.N,
                float(p.spatial_freq),
                thetas,
                on_to_off=net.on_to_off,
                **proj_kwargs,
            )
            rf_osi, _ = compute_osi(rf_amp, thetas)
            w_corr = onoff_weight_corr(net.W, p.N, on_to_off=net.on_to_off, **proj_kwargs)

            print(f"[seg {s:4d}] mean rate={rates.mean():.3f} Hz | mean OSI={osi.mean():.3f} | max OSI={osi.max():.3f}")
            print(
                f"          RF(weight): orientedness mean={float(rf_ori.mean()):.3f} "
                f"| grating-match OSI mean={float(rf_osi.mean()):.3f} "
                f"(frac>0.45={float((rf_ori>0.45).mean()):.2f}) | "
                f"ON/OFF weight corr mean={float(w_corr.mean()):+.3f}"
            )
            print(f"          prefs(deg) = {np.round(pref, 1)}")
            tuned = (osi >= 0.3)
            near90 = (circ_diff_180(pref, 90.0) <= 10.0) & tuned
            if tuned.any():
                print(f"          tuned(OSI≥0.3) = {int(tuned.sum())}/{p.M} | near 90° (±10°) = {int(near90.sum())}/{int(tuned.sum())}")
                r, mu = circ_mean_resultant_180(pref[tuned])
                gap = max_circ_gap_180(pref[tuned])
                print(f"          pref diversity: resultant={r:.3f}, max_gap={gap:.1f}° (mean={mu:.1f}°)")

            plot_weight_maps(net.W, p.N,
                           os.path.join(args.out, f"weights_seg{s:04d}.png"),
                           title=f"LGN->V1 weights (segment {s})")
            plot_tuning(rates, thetas, osi, pref,
                       os.path.join(args.out, f"tuning_seg{s:04d}.png"),
                       title=f"Tuning during training (segment {s})")
            plot_pref_hist(pref, osi,
                           os.path.join(args.out, f"pref_hist_seg{s:04d}.png"),
                           title=f"Preferred orientation histogram (segment {s})")
            save_eval_npz(os.path.join(args.out, f"eval_seg{s:04d}.npz"),
                          thetas_deg=thetas, rates_hz=rates, osi=osi, pref_deg=pref, net=net)

            seg_hist.append(int(s))
            osi_hist.append(float(osi.mean()))
            rate_hist.append(float(rates.mean()))

    # --- Final evaluation ---
    print("\n[final] Final evaluation with robust repeats...")
    final_repeats = max(args.baseline_repeats, args.eval_repeats, 7)
    rates1 = net.evaluate_tuning(thetas, repeats=final_repeats)
    osi1, pref1 = compute_osi(rates1, thetas)
    rf_ori1, rf_pref1 = rf_fft_orientation_metrics(net.W, p.N, on_to_off=net.on_to_off, **proj_kwargs)
    rf_amp1 = rf_grating_match_tuning(
        net.W,
        p.N,
        float(p.spatial_freq),
        thetas,
        on_to_off=net.on_to_off,
        **proj_kwargs,
    )
    rf_osi1, _ = compute_osi(rf_amp1, thetas)
    w_corr1 = onoff_weight_corr(net.W, p.N, on_to_off=net.on_to_off, **proj_kwargs)

    d_osi = osi1 - osi0
    print(f"[final] baseline mean OSI={osi0.mean():.3f} -> final mean OSI={osi1.mean():.3f} (delta={d_osi.mean():+.3f})")
    print(
        f"[final] RF(weight): orientedness mean={float(rf_ori1.mean()):.3f} "
        f"| grating-match OSI mean={float(rf_osi1.mean()):.3f} "
        f"(frac>0.45={float((rf_ori1>0.45).mean()):.2f}) | "
        f"ON/OFF weight corr mean={float(w_corr1.mean()):+.3f}"
    )
    print(f"[final] fraction ensembles with OSI>0.3: {(osi1>0.3).mean()*100:.1f}%")
    print(f"[final] fraction ensembles with OSI>0.5: {(osi1>0.5).mean()*100:.1f}%")
    tuned1 = (osi1 >= 0.3)
    if tuned1.any():
        r1, mu1 = circ_mean_resultant_180(pref1[tuned1])
        gap1 = max_circ_gap_180(pref1[tuned1])
        print(f"[final] pref diversity (OSI≥0.3): resultant={r1:.3f}, max_gap={gap1:.1f}°, mean={mu1:.1f}°")

    # Final plots
    plot_weight_maps(net.W, p.N,
                    os.path.join(args.out, "weights_final.png"),
                    title="LGN->V1 weights (final)")
    plot_weight_maps(net.W, p.N,
                    os.path.join(args.out, "weights_final_smoothed.png"),
                    title=f"LGN->V1 weights (final, Gaussian sigma={float(args.weight_smooth_sigma):.2f})",
                    smooth_sigma=float(args.weight_smooth_sigma))
    plot_weight_maps_before_after(
        W_init, net.W, p.N,
        os.path.join(args.out, "weights_before_vs_after_smoothed.png"),
        title=f"LGN->V1 filters before vs after training (Gaussian sigma={float(args.weight_smooth_sigma):.2f})",
        smooth_sigma=float(args.weight_smooth_sigma),
    )
    plot_tuning(rates1, thetas, osi1, pref1,
               os.path.join(args.out, "tuning_final.png"),
               title="Final tuning (after learning)")
    plot_pref_hist(pref1, osi1,
                   os.path.join(args.out, "pref_hist_final.png"),
                   title="Preferred orientation histogram (final)")
    save_eval_npz(os.path.join(args.out, "eval_final.npz"),
                  thetas_deg=thetas, rates_hz=rates1, osi=osi1, pref_deg=pref1, net=net)
    plot_scalar_over_time(np.array(seg_hist), np.array(osi_hist),
                         os.path.join(args.out, "mean_osi_over_time.png"),
                         ylabel="mean OSI", title="Mean OSI over training")
    plot_scalar_over_time(np.array(seg_hist), np.array(rate_hist),
                         os.path.join(args.out, "mean_rate_over_time.png"),
                         ylabel="mean rate (Hz)", title="Mean firing rate over training")

    print(f"\n[done] Outputs written to: {args.out}")


if __name__ == "__main__":
    main()
