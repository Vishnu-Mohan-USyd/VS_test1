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
    base_rate: float = 5.0
    gain_rate: float = 205.0

    # Sparse spot movie (flash-like stimulus).
    # Implemented as a random sparse set of bright/dark pixels that refresh every `spots_frame_ms`.
    spots_density: float = 0.02   # fraction of pixels active per frame (0..1)
    spots_frame_ms: float = 33.3  # refresh period (ms) ~30 Hz
    spots_amp: float = 1.0        # luminance amplitude of each spot (+/-amp)

    # Dense random noise stimulus (spatiotemporal white noise).
    noise_sigma: float = 0.35     # std of pixel luminance noise
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

    # RGC->LGN synaptic weight (scaled for Izhikevich pA currents)
    # Izhikevich model uses currents ~0-40 pA for typical spiking
    w_rgc_lgn: float = 5.0

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

    # Weight decay (biologically: synaptic turnover)
    w_decay: float = 0.00000001  # Per-timestep weight decay (slow turnover; ~hours time scale)

    # Local inhibitory circuit parameters
    n_pv_per_ensemble: int = 1  # PV interneurons per ensemble
    n_som_per_ensemble: int = 1  # SOM interneurons per ensemble for lateral inhibition

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
    pp_onoff_swap: bool = True
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

    # Reversal potentials (for conductance-based synapses)
    E_exc: float = 0.0  # mV (AMPA/NMDA; simplified)
    E_inh: float = -70.0  # mV (GABA_A)

    # Conductance scaling: convert weight-sums into effective synaptic conductances.
    # Roughly, g * (E_exc - V) should be in the same range as the previous current-based drive.
    w_exc_gain: float = 0.015  # ~1/65 for E_exc=0mV and V_rest≈-65mV


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
        self.v += rng.uniform(-5, 5, n).astype(np.float32)
        self.u += rng.uniform(-2, 2, n).astype(np.float32)

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

    def __init__(self, n_pre: int, n_post: int, p: Params, rng: np.random.Generator):
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

        # Spatial coordinates for stimulus
        xs = np.arange(p.N) - (p.N - 1) / 2.0
        ys = np.arange(p.N) - (p.N - 1) / 2.0
        self.X, self.Y = np.meshgrid(xs, ys, indexing="xy")
        # Real RGC mosaics are not perfect grids; small positional jitter reduces lattice biases.
        if p.rgc_pos_jitter > 0:
            # Important: naive random jitter on a *small* patch can introduce a net shear/dipole,
            # producing systematic orientation biases across the whole network. Enforce 180° rotational
            # antisymmetry so the mosaic remains globally centered while still breaking the lattice.
            j = float(p.rgc_pos_jitter)
            jx = self.rng.uniform(-j, j, size=self.X.shape).astype(np.float32)
            jy = self.rng.uniform(-j, j, size=self.Y.shape).astype(np.float32)
            jx = 0.5 * (jx - jx[::-1, ::-1])
            jy = 0.5 * (jy - jy[::-1, ::-1])
            self.X = (self.X + jx).astype(np.float32)
            self.Y = (self.Y + jy).astype(np.float32)

        # RGC center–surround DoG front-end.
        # Important for biological plausibility AND for avoiding orientation-biased learning: small patches
        # with truncated DoG kernels can introduce systematic oblique biases. The default "padded_fft"
        # implementation filters on a padded field so the central patch sees an approximately translation-
        # invariant DoG response.
        self.rgc_dog = None  # (N^2,N^2) matrix for legacy "matrix" mode
        self._rgc_pad = 0
        self._X_pad = None
        self._Y_pad = None
        self._rgc_dog_fft = None  # rfft2 kernel for padded DoG
        self._rgc_sample_idx00 = None
        self._rgc_sample_idx10 = None
        self._rgc_sample_idx01 = None
        self._rgc_sample_idx11 = None
        self._rgc_sample_wx = None
        self._rgc_sample_wy = None
        self._init_rgc_frontend()

        # Retinotopic envelopes (fixed structural locality) for thalamocortical projections.
        # Implemented as a spatially varying *cap* on synaptic weights (far inputs cannot become strong).
        d2 = (self.X.astype(np.float32) ** 2 + self.Y.astype(np.float32) ** 2)

        def lgn_mask_vec(sigma: float) -> np.ndarray:
            if sigma <= 0:
                return np.ones(self.n_lgn, dtype=np.float32)
            pix = np.exp(-d2 / (2.0 * float(sigma) * float(sigma))).astype(np.float32).ravel()
            vec = np.concatenate([pix, pix]).astype(np.float32)  # ON and OFF share the same envelope
            vec /= float(vec.max() + 1e-12)
            return vec

        self._lgn_mask_e_vec = lgn_mask_vec(p.lgn_sigma_e)
        self._lgn_mask_pv_vec = lgn_mask_vec(p.lgn_sigma_pv)
        self._lgn_mask_pp_vec = lgn_mask_vec(p.lgn_sigma_pp)

        # --- LGN Layer (Thalamocortical neurons) ---
        self.lgn = IzhikevichPopulation(self.n_lgn, TC_PARAMS, p.dt_ms, self.rng)

        # --- V1 Excitatory Layer (Regular spiking) ---
        self.v1_exc = IzhikevichPopulation(p.M, RS_PARAMS, p.dt_ms, self.rng)

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

        # --- Synaptic currents / conductances ---
        self.I_lgn = np.zeros(self.n_lgn, dtype=np.float32)
        self.g_v1_exc = np.zeros(p.M, dtype=np.float32)  # excitatory AMPA conductance onto V1 E
        self.I_pv = np.zeros(self.n_pv, dtype=np.float32)
        self.I_pp = np.zeros(self.n_pp, dtype=np.float32)
        self.I_som = np.zeros(self.n_som, dtype=np.float32)

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

        # Inhibitory conductances onto V1 excitatory neurons.
        # PV inhibition uses a difference-of-exponentials (rise + decay) to avoid unrealistically
        # zero-lag inhibition in a discrete-time update.
        self.g_v1_inh_pv_rise = np.zeros(p.M, dtype=np.float32)
        self.g_v1_inh_pv_decay = np.zeros(p.M, dtype=np.float32)
        self.g_v1_inh_som = np.zeros(p.M, dtype=np.float32)
        self.g_v1_inh_pp = np.zeros(p.M, dtype=np.float32)  # push–pull (LGN-driven) inhibition

        # Previous-step spikes (for delayed recurrent effects)
        self.prev_v1_spk = np.zeros(p.M, dtype=np.uint8)

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
        self.pp_swap_idx = np.concatenate([np.arange(n_pix, 2 * n_pix), np.arange(0, n_pix)]).astype(np.int32)

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
        # E->PV connectivity (each E connects to its local PV)
        self.W_e_pv = np.zeros((self.n_pv, p.M), dtype=np.float32)
        for m in range(p.M):
            pv_start = m * p.n_pv_per_ensemble
            pv_end = pv_start + p.n_pv_per_ensemble
            self.W_e_pv[pv_start:pv_end, m] = p.w_e_pv

        # PP->E connectivity (local push–pull inhibition)
        self.W_pp_e = np.zeros((p.M, self.n_pp), dtype=np.float32)
        for m in range(p.M):
            pp_start = m * p.n_pp_per_ensemble
            pp_end = pp_start + p.n_pp_per_ensemble
            # Divide by n_pp_per_ensemble so total inhibition per ensemble stays comparable.
            self.W_pp_e[m, pp_start:pp_end] = p.w_pushpull / max(1, p.n_pp_per_ensemble)

        # PV->E connectivity (local inhibition only)
        self.W_pv_e = np.zeros((p.M, self.n_pv), dtype=np.float32)
        for m in range(p.M):
            pv_start = m * p.n_pv_per_ensemble
            pv_end = pv_start + p.n_pv_per_ensemble
            self.W_pv_e[m, pv_start:pv_end] = p.w_pv_e
        self.mask_pv_e = (self.W_pv_e > 0)

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

        # --- Plasticity mechanisms ---
        self.stdp = TripletSTDP(self.n_lgn, p.M, p, self.rng)
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
            self.rgc_dog = self._build_rgc_dog_filter_matrix()
            return
        if impl == "padded_fft":
            self._setup_rgc_dog_padded_fft()
            return
        raise ValueError("rgc_dog_impl must be one of: 'matrix', 'padded_fft'")

    def _build_rgc_dog_filter_matrix(self) -> np.ndarray:
        """Build an (N^2, N^2) DoG filter using the (jittered) RGC coordinates (legacy mode)."""
        p = self.p
        if p.rgc_center_sigma <= 0.0 or p.rgc_surround_sigma <= 0.0:
            raise ValueError("rgc_center_sigma and rgc_surround_sigma must be > 0 for DoG filtering")
        if p.rgc_surround_sigma <= p.rgc_center_sigma:
            raise ValueError("rgc_surround_sigma must be > rgc_center_sigma for a center–surround DoG")

        x = self.X.astype(np.float32).ravel()
        y = self.Y.astype(np.float32).ravel()
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

        # Precompute bilinear sampling indices/weights from the padded field to the (possibly jittered)
        # RGC mosaic locations in the central patch.
        fx = self.X.astype(np.float32).ravel() + (n - 1) / 2.0
        fy = self.Y.astype(np.float32).ravel() + (n - 1) / 2.0

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

        self._rgc_sample_idx00 = (y0 * n + x0).astype(np.int32)
        self._rgc_sample_idx10 = (y0 * n + x1).astype(np.int32)
        self._rgc_sample_idx01 = (y1 * n + x0).astype(np.int32)
        self._rgc_sample_idx11 = (y1 * n + x1).astype(np.int32)
        self._rgc_sample_wx = wx
        self._rgc_sample_wy = wy

    def reset_state(self) -> None:
        """Reset all dynamic state (but not weights)."""
        self.lgn.reset()
        self.v1_exc.reset()
        self.pv.reset()
        self.pp.reset()
        self.som.reset()

        self.I_lgn.fill(0)
        self.g_v1_exc.fill(0)
        self.I_pv.fill(0)
        self.I_pp.fill(0)
        self.I_som.fill(0)
        self.g_v1_inh_pv_rise.fill(0)
        self.g_v1_inh_pv_decay.fill(0)
        self.g_v1_inh_som.fill(0)
        self.g_v1_inh_pp.fill(0)
        self.prev_v1_spk.fill(0)

        self.delay_buf.fill(0)
        self.ptr = 0

        if self.tc_stp_x is not None:
            self.tc_stp_x.fill(1.0)

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

    def _rgc_drive_from_pad_stimulus(self, stim_pad: np.ndarray, *, contrast: float) -> np.ndarray:
        """Apply padded DoG filtering to an arbitrary stimulus on the padded grid, then sample to the mosaic."""
        if self._rgc_dog_fft is None:
            raise RuntimeError("padded_fft DoG front-end not initialized")
        stim_pad = (contrast * stim_pad).astype(np.float32, copy=False)
        dog_pad = np.fft.irfft2(
            np.fft.rfft2(stim_pad) * self._rgc_dog_fft,
            s=stim_pad.shape,
        ).astype(np.float32, copy=False)

        flat = dog_pad.ravel()
        v00 = flat[self._rgc_sample_idx00]
        v10 = flat[self._rgc_sample_idx10]
        v01 = flat[self._rgc_sample_idx01]
        v11 = flat[self._rgc_sample_idx11]

        wx = self._rgc_sample_wx
        wy = self._rgc_sample_wy
        omx = (1.0 - wx).astype(np.float32, copy=False)
        omy = (1.0 - wy).astype(np.float32, copy=False)

        out = (omx * omy) * v00 + (wx * omy) * v10 + (omx * wy) * v01 + (wx * wy) * v11
        return out.reshape(self.N, self.N).astype(np.float32, copy=False)

    def rgc_drive_grating(self, theta_deg: float, t_ms: float, phase: float, *,
                          contrast: float = 1.0) -> np.ndarray:
        """Compute the RGC 'drive' field (after optional DoG filtering) for a drifting grating."""
        p = self.p

        if not p.rgc_center_surround:
            return (contrast * self.grating(theta_deg, t_ms, phase)).astype(np.float32, copy=False)

        impl = str(p.rgc_dog_impl).lower()
        if impl == "matrix":
            stim = self.grating(theta_deg, t_ms, phase)
            stim_c = (contrast * stim).astype(np.float32, copy=False)
            stim_vec = stim_c.ravel()
            return (self.rgc_dog @ stim_vec).reshape(stim.shape).astype(np.float32, copy=False)

        if impl != "padded_fft":
            raise ValueError("rgc_dog_impl must be one of: 'matrix', 'padded_fft'")
        if self._X_pad is None or self._Y_pad is None or self._rgc_dog_fft is None:
            raise RuntimeError("padded_fft DoG front-end not initialized")

        stim_pad = self.grating_on_coords(theta_deg, t_ms, phase, self._X_pad, self._Y_pad)
        return self._rgc_drive_from_pad_stimulus(stim_pad, contrast=contrast)

    def rgc_spikes_from_drive(self, drive: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ON and OFF RGC spikes from a contrast-like drive field."""
        p = self.p
        on_rate = p.base_rate + p.gain_rate * np.clip(drive, 0, None)
        off_rate = p.base_rate + p.gain_rate * np.clip(-drive, 0, None)
        dt_s = p.dt_ms / 1000.0
        on_spk = (self.rng.random(drive.shape) < (on_rate * dt_s)).astype(np.uint8)
        off_spk = (self.rng.random(drive.shape) < (off_rate * dt_s)).astype(np.uint8)
        return on_spk, off_spk

    def rgc_spikes_grating(self, theta_deg: float, t_ms: float, phase: float, *,
                           contrast: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ON and OFF RGC spikes for a drifting grating (preferred code path)."""
        drive = self.rgc_drive_grating(theta_deg, t_ms, phase, contrast=contrast)
        return self.rgc_spikes_from_drive(drive)

    def rgc_spikes(self, stim: np.ndarray, *, contrast: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate ON and OFF RGC spikes from an explicit stimulus field sampled at RGC positions.

        Note: When `rgc_dog_impl='padded_fft'`, gratings should be passed via `rgc_spikes_grating(...)`
        so the DoG can be computed on a padded field (avoids edge-induced orientation bias).
        """
        p = self.p
        stim_c = (contrast * stim).astype(np.float32, copy=False)
        if p.rgc_center_surround:
            impl = str(p.rgc_dog_impl).lower()
            if impl == "matrix":
                stim_vec = stim_c.ravel()
                stim_c = (self.rgc_dog @ stim_vec).reshape(stim.shape).astype(np.float32, copy=False)
            elif impl == "padded_fft":
                raise ValueError("rgc_spikes(stim) is ambiguous in padded_fft mode; use rgc_spikes_grating(...)")
            else:
                raise ValueError("rgc_dog_impl must be one of: 'matrix', 'padded_fft'")
        return self.rgc_spikes_from_drive(stim_c)

    def step(self, on_spk: np.ndarray, off_spk: np.ndarray, plastic: bool) -> np.ndarray:
        """
        Advance network by one timestep.

        Returns V1 excitatory spikes.
        """
        p = self.p

        # Combine ON/OFF RGC spikes
        rgc = np.concatenate([on_spk.ravel(), off_spk.ravel()]).astype(np.float32)

        # --- LGN layer ---
        self.I_lgn *= self.decay_ampa
        self.I_lgn += p.w_rgc_lgn * rgc
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

        # Inhibitory conductances (GABA decay)
        self.g_v1_inh_pv_rise *= self.decay_gaba_rise_pv
        self.g_v1_inh_pv_decay *= self.decay_gaba
        self.g_v1_inh_som *= self.decay_gaba
        self.g_v1_inh_pp *= self.decay_gaba

        # --- PV interneurons (feedforward inhibition; must run BEFORE E to be feedforward-in-time) ---
        self.I_pv *= self.decay_ampa
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
        pv_spk = self.pv.step(self.I_pv)
        self.last_pv_spk = pv_spk

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
        I_exc = self.g_v1_exc * (p.E_exc - self.v1_exc.v)
        I_v1_total = I_exc + g_inh * (p.E_inh - self.v1_exc.v)
        I_v1_total = I_v1_total + self.I_v1_bias
        v1_spk = self.v1_exc.step(I_v1_total)
        self.last_v1_spk = v1_spk

        # --- SOM interneurons (lateral / dendritic inhibition; updated AFTER E, affects next step) ---
        self.I_som *= self.decay_ampa
        self.I_som += self.W_e_som @ v1_spk.astype(np.float32)
        som_spk = self.som.step(self.I_som)
        self.last_som_spk = som_spk

        # SOM->E lateral inhibition (GABA conductance increment; affects next step)
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

    def _segment_boundary_updates(self, v1_counts: np.ndarray) -> None:
        """Update slow plasticity/homeostasis terms at a segment boundary (local, per-neuron)."""
        p = self.p

        # Optional slow synaptic scaling (no hard normalization).
        self.apply_homeostasis()

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
                    stim_pad = np.zeros((n, n), dtype=np.float32)
                    density = float(p.spots_density)
                    if density > 0:
                        n_spots = int(round(density * float(n * n)))
                        n_spots = int(max(1, min(n * n, n_spots)))
                        idx = self.rng.choice(n * n, size=n_spots, replace=False)
                        pol = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_spots, replace=True)
                        stim_pad.ravel()[idx] = pol * float(p.spots_amp)
                else:
                    stim = np.zeros((p.N, p.N), dtype=np.float32)
                    density = float(p.spots_density)
                    if density > 0:
                        n_spots = int(round(density * float(p.N * p.N)))
                        n_spots = int(max(1, min(p.N * p.N, n_spots)))
                        idx = self.rng.choice(p.N * p.N, size=n_spots, replace=False)
                        pol = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_spots, replace=True)
                        stim.ravel()[idx] = pol * float(p.spots_amp)

            if use_pad:
                drive = self._rgc_drive_from_pad_stimulus(stim_pad, contrast=contrast)
                on_spk, off_spk = self.rgc_spikes_from_drive(drive)
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
                drive = self._rgc_drive_from_pad_stimulus(stim_pad, contrast=contrast)
                on_spk, off_spk = self.rgc_spikes_from_drive(drive)
            else:
                on_spk, off_spk = self.rgc_spikes(stim, contrast=contrast)

            v1_counts += self.step(on_spk, off_spk, plastic=plastic)

        if plastic:
            self._segment_boundary_updates(v1_counts)

        return v1_counts

    def run_segment_counts(self, theta_deg: float, plastic: bool, *, contrast: float = 1.0) -> dict:
        """Run one segment and return spike counts for E/PV/SOM/LGN (for diagnostics/tests)."""
        p = self.p
        steps = int(p.segment_ms / p.dt_ms)
        phase = float(self.rng.uniform(0, 2 * math.pi))

        v1_counts = np.zeros(self.M, dtype=np.int32)
        pv_counts = np.zeros(self.n_pv, dtype=np.int32)
        pp_counts = np.zeros(self.n_pp, dtype=np.int32)
        som_counts = np.zeros(self.n_som, dtype=np.int32)
        lgn_counts = np.zeros(self.n_lgn, dtype=np.int32)

        for k in range(steps):
            on_spk, off_spk = self.rgc_spikes_grating(theta_deg, t_ms=k * p.dt_ms, phase=phase, contrast=contrast)
            v1_counts += self.step(on_spk, off_spk, plastic=plastic)
            pv_counts += self.last_pv_spk
            pp_counts += self.last_pp_spk
            som_counts += self.last_som_spk
            lgn_counts += self.last_lgn_spk

        return {
            "v1_counts": v1_counts,
            "pv_counts": pv_counts,
            "pp_counts": pp_counts,
            "som_counts": som_counts,
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

        for k in range(steps):
            on_spk, off_spk = self.rgc_spikes_grating(theta_deg, t_ms=k * p.dt_ms, phase=phase, contrast=contrast)
            v1_spk = self.step(on_spk, off_spk, plastic=False)
            I_ff_ts[k] = self.last_I_ff
            g_pp_ts[k] = self.last_g_pp_input
            g_pp_inh_ts[k] = self.g_v1_inh_pp
            v_ts[k] = self.v1_exc.v
            v1_spk_ts[k] = v1_spk

        t_ms = (np.arange(steps, dtype=np.float32) * p.dt_ms).astype(np.float32)
        return {
            "t_ms": t_ms,
            "I_ff": I_ff_ts,
            "g_pp_input": g_pp_ts,
            "g_pp_inh": g_pp_inh_ts,
            "v": v_ts,
            "v1_spk": v1_spk_ts,
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
        saved_pv_v = self.pv.v.copy()
        saved_pv_u = self.pv.u.copy()
        saved_pp_v = self.pp.v.copy()
        saved_pp_u = self.pp.u.copy()
        saved_som_v = self.som.v.copy()
        saved_som_u = self.som.u.copy()
        saved_I_lgn = self.I_lgn.copy()
        saved_g_v1_exc = self.g_v1_exc.copy()
        saved_I_v1_bias = self.I_v1_bias.copy()
        saved_g_v1_inh_pv_rise = self.g_v1_inh_pv_rise.copy()
        saved_g_v1_inh_pv_decay = self.g_v1_inh_pv_decay.copy()
        saved_g_v1_inh_som = self.g_v1_inh_som.copy()
        saved_g_v1_inh_pp = self.g_v1_inh_pp.copy()
        saved_I_pv = self.I_pv.copy()
        saved_I_pp = self.I_pp.copy()
        saved_I_som = self.I_som.copy()
        saved_buf = self.delay_buf.copy()
        saved_ptr = self.ptr
        saved_stdp_x_pre = self.stdp.x_pre.copy()
        saved_stdp_x_pre_slow = self.stdp.x_pre_slow.copy()
        saved_stdp_x_post = self.stdp.x_post.copy()
        saved_stdp_x_post_slow = self.stdp.x_post_slow.copy()
        saved_pv_istdp_x_post = self.pv_istdp.x_post.copy()
        saved_ee_x_pre = self.ee_stdp.x_pre.copy()
        saved_ee_x_post = self.ee_stdp.x_post.copy()
        saved_prev_v1_spk = self.prev_v1_spk.copy()

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
        self.pv.v = saved_pv_v
        self.pv.u = saved_pv_u
        self.pp.v = saved_pp_v
        self.pp.u = saved_pp_u
        self.som.v = saved_som_v
        self.som.u = saved_som_u
        self.I_lgn = saved_I_lgn
        self.g_v1_exc = saved_g_v1_exc
        self.I_v1_bias = saved_I_v1_bias
        self.g_v1_inh_pv_rise = saved_g_v1_inh_pv_rise
        self.g_v1_inh_pv_decay = saved_g_v1_inh_pv_decay
        self.g_v1_inh_som = saved_g_v1_inh_som
        self.g_v1_inh_pp = saved_g_v1_inh_pp
        self.I_pv = saved_I_pv
        self.I_pp = saved_I_pp
        self.I_som = saved_I_som
        self.delay_buf = saved_buf
        self.ptr = saved_ptr
        self.stdp.x_pre = saved_stdp_x_pre
        self.stdp.x_pre_slow = saved_stdp_x_pre_slow
        self.stdp.x_post = saved_stdp_x_post
        self.stdp.x_post_slow = saved_stdp_x_post_slow
        self.pv_istdp.x_post = saved_pv_istdp_x_post
        self.ee_stdp.x_pre = saved_ee_x_pre
        self.ee_stdp.x_post = saved_ee_x_post
        self.prev_v1_spk = saved_prev_v1_spk
        self.rng.bit_generator.state = rng_state

        return rates


# =============================================================================
# Visualization functions
# =============================================================================

def plot_weight_maps(W: np.ndarray, N: int, outpath: str, title: str) -> None:
    """Plot ON, OFF, and ON-OFF weight maps for each ensemble."""
    M = W.shape[0]
    W_on = W[:, :N * N].reshape(M, N, N)
    W_off = W[:, N * N:].reshape(M, N, N)
    W_diff = W_on - W_off

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
    np.savez_compressed(
        outpath,
        thetas_deg=thetas_deg.astype(np.float32),
        rates_hz=rates_hz.astype(np.float32),
        osi=osi.astype(np.float32),
        pref_deg=pref_deg.astype(np.float32),
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
    ap.add_argument("--tc-conn-fraction-e", type=float, default=None,
                    help="fraction of LGN afferents present per excitatory neuron (0..1]; <1 enforces sparse anatomical mask)")
    ap.add_argument("--tc-conn-fraction-pv", type=float, default=None,
                    help="fraction of LGN afferents present per PV interneuron (0..1]")
    ap.add_argument("--tc-conn-fraction-pp", type=float, default=None,
                    help="fraction of LGN afferents present per PP interneuron (0..1]")
    ap.add_argument("--tc-no-balance-onoff", action="store_true",
                    help="when using sparse thalamocortical connectivity, do not enforce balanced ON/OFF sampling")
    ap.add_argument("--no-pp-onoff-swap", action="store_true",
                    help="disable ON/OFF swap into PP interneurons (removes built-in phase opposition)")
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
    if args.spots_density is not None:
        p_kwargs["spots_density"] = float(args.spots_density)
    if args.spots_frame_ms is not None:
        p_kwargs["spots_frame_ms"] = float(args.spots_frame_ms)
    if args.spots_amp is not None:
        p_kwargs["spots_amp"] = float(args.spots_amp)
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
    if args.tc_conn_fraction_e is not None:
        p_kwargs["tc_conn_fraction_e"] = float(args.tc_conn_fraction_e)
    if args.tc_conn_fraction_pv is not None:
        p_kwargs["tc_conn_fraction_pv"] = float(args.tc_conn_fraction_pv)
    if args.tc_conn_fraction_pp is not None:
        p_kwargs["tc_conn_fraction_pp"] = float(args.tc_conn_fraction_pp)
    if args.tc_no_balance_onoff:
        p_kwargs["tc_conn_balance_onoff"] = False
    if args.no_pp_onoff_swap:
        p_kwargs["pp_onoff_swap"] = False

    p = Params(**p_kwargs)
    net = RgcLgnV1Network(p, init_mode=args.init_mode)

    print(f"[init] Biologically plausible RGC->LGN->V1 network")
    print(f"[init] N={p.N} (patch), M={p.M} (ensembles), n_lgn={net.n_lgn}")
    print(f"[init] Neuron types: LGN=TC(Izhikevich), V1=RS, PV=FS, SOM=LTS")
    print(f"[init] Plasticity: Triplet STDP + iSTDP (PV) + STP (TC) + optional synaptic scaling={'ON' if p.homeostasis_rate>0 else 'OFF'}")
    print(f"[init] Inhibition: PV (LGN-driven + local E-driven) + SOM (lateral)")
    if p.rgc_center_surround:
        if str(p.rgc_dog_impl).lower() == "padded_fft":
            print(f"[init] RGC DoG: padded_fft (pad={net._rgc_pad}, norm={p.rgc_dog_norm}, jitter={p.rgc_pos_jitter:.3f})")
        else:
            print(f"[init] RGC DoG: matrix (norm={p.rgc_dog_norm}, jitter={p.rgc_pos_jitter:.3f})")
    else:
        print(f"[init] RGC DoG: OFF (raw grating -> ON/OFF Poisson)")
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

    # --- Baseline evaluation ---
    print("\n[baseline] Evaluating tuning at initialization...")
    rates0 = net.evaluate_tuning(thetas, repeats=args.baseline_repeats)
    osi0, pref0 = compute_osi(rates0, thetas)

    print(f"[seg {0:4d}] mean rate={rates0.mean():.3f} Hz | mean OSI={osi0.mean():.3f} | max OSI={osi0.max():.3f}")
    print(f"          prefs(deg) = {np.round(pref0, 1)}")
    print("          NOTE: Nonzero OSI at init is expected from random RF structure")
    tuned0 = (osi0 >= 0.3)
    near90_0 = (circ_diff_180(pref0, 90.0) <= 10.0) & tuned0
    if tuned0.any():
        print(f"          tuned(OSI≥0.3) = {int(tuned0.sum())}/{p.M} | near 90° (±10°) = {int(near90_0.sum())}/{int(tuned0.sum())}")
        r0, mu0 = circ_mean_resultant_180(pref0[tuned0])
        gap0 = max_circ_gap_180(pref0[tuned0])
        print(f"          pref diversity: resultant={r0:.3f}, max_gap={gap0:.1f}° (mean={mu0:.1f}°)")

    plot_weight_maps(net.W, p.N, os.path.join(args.out, "weights_seg0000.png"),
                     title="LGN->V1 weights at init (segment 0)")
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

            print(f"[seg {s:4d}] mean rate={rates.mean():.3f} Hz | mean OSI={osi.mean():.3f} | max OSI={osi.max():.3f}")
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

    d_osi = osi1 - osi0
    print(f"[final] baseline mean OSI={osi0.mean():.3f} -> final mean OSI={osi1.mean():.3f} (delta={d_osi.mean():+.3f})")
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
