#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utility_2D.py  (PyTorch port)
# Author: ported from NumPy version by ChatGPT

from __future__ import annotations
import math
from typing import Tuple

import torch

device_default = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
_COMPLEX = torch.complex64


# ---------- small helpers ----------
def _to_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, dtype=dtype, device=device or device_default)

def _as_complex2d(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        x = x.to(torch.float32).to(_COMPLEX)
    assert x.ndim == 2, "expected a 2D tensor"
    return x

def _fro(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(x)


# ---------- API parity with your NumPy utility_2D ----------
def align_objects(obj_tr, obj, mask):
    """Align global phase of obj to obj_tr on the support mask."""
    obj_tr = _as_complex2d(_to_tensor(obj_tr))
    obj    = _as_complex2d(_to_tensor(obj))
    mask   = _to_tensor(mask, dtype=torch.float32)

    prod = (obj.conj() * obj_tr) * mask
    alpha = torch.exp(1j * torch.angle(prod.sum()))
    obj_r = alpha * obj
    return obj_r


def eliminate_linear_ambiguity(obj, win, threshold=1e-10):
    """Remove linear phase ramp ambiguity using window statistics (original version)."""
    obj = _as_complex2d(_to_tensor(obj))
    win = _as_complex2d(_to_tensor(win))

    # normalize energy exchange
    obj_n, win_n = normalize_object_and_window(obj.clone(), win.clone())

    idx = (torch.abs(win_n) > threshold)
    ph = torch.ones_like(win_n, dtype=_COMPLEX)
    ph[idx] = win_n[idx]

    # estimate ramp along both axes (finite differences of phase)
    fr1 = torch.angle(ph[1:, :] / ph[:-1, :])
    usable1 = torch.logical_or(idx[1:, :], idx[:-1, :])
    if torch.count_nonzero(usable1) > 0:
        b1 = fr1[usable1].mean()
    else:
        b1 = torch.tensor(0.0, device=win.device)

    fr2 = torch.angle(ph[:, 1:] / ph[:, :-1])
    usable2 = torch.logical_or(idx[:, 1:], idx[:, :-1])
    if torch.count_nonzero(usable2) > 0:
        b2 = fr2[usable2].mean()
    else:
        b2 = torch.tensor(0.0, device=win.device)

    gy, gx = torch.meshgrid(
        torch.arange(win_n.shape[0], device=win.device, dtype=torch.float32),
        torch.arange(win_n.shape[1], device=win.device, dtype=torch.float32),
        indexing='ij'
    )
    win_mode = torch.exp(-1j * (gx * b1 + gy * b2))
    win_r = win * win_mode

    gy, gx = torch.meshgrid(
        torch.arange(obj.shape[0], device=obj.device, dtype=torch.float32),
        torch.arange(obj.shape[1], device=obj.device, dtype=torch.float32),
        indexing='ij'
    )
    obj_mode = torch.exp(1j * (gx * b1 + gy * b2))
    obj_r = obj * obj_mode
    return obj_r, win_r


def eliminate_linear_ambiguity2(obj, win, threshold=1e-10):
    """Remove linear phase ramp ambiguity using object statistics (alternate)."""
    obj = _as_complex2d(_to_tensor(obj))
    win = _as_complex2d(_to_tensor(win))
    obj, win = normalize_object_and_window(obj, win)

    idx = (torch.abs(obj) > threshold)
    ph = torch.ones_like(obj, dtype=_COMPLEX)
    ph[idx] = obj[idx]

    fr1 = torch.angle(ph[1:, :] / ph[:-1, :])
    usable1 = torch.logical_or(idx[1:, :], idx[:-1, :])
    b1 = fr1[usable1].mean() if torch.count_nonzero(usable1) > 0 else torch.tensor(0.0, device=obj.device)

    fr2 = torch.angle(ph[:, 1:] / ph[:, :-1])
    usable2 = torch.logical_or(idx[:, 1:], idx[:, :-1])
    b2 = fr2[usable2].mean() if torch.count_nonzero(usable2) > 0 else torch.tensor(0.0, device=obj.device)

    gy, gx = torch.meshgrid(
        torch.arange(win.shape[0], device=win.device, dtype=torch.float32),
        torch.arange(win.shape[1], device=win.device, dtype=torch.float32),
        indexing='ij'
    )
    win_mode = torch.exp(1j * (gx * b1 + gy * b2))
    win_r = win * win_mode

    gy, gx = torch.meshgrid(
        torch.arange(obj.shape[0], device=obj.device, dtype=torch.float32),
        torch.arange(obj.shape[1], device=obj.device, dtype=torch.float32),
        indexing='ij'
    )
    obj_mode = torch.exp(-1j * (gx * b1 + gy * b2))
    obj_r = obj * obj_mode
    return obj_r, win_r


def eliminate_grid_ambiguity_l1(obj, win, s, start0=0, end0=0, start1=0, end1=0,
                                threshold=1e-6, maxit=100, eps=1e-8):
    """
    IRLS approach (L1-like) to estimate periodic lambda that minimizes TV.
    Port uses dense Hermitian eigensolve via torch.linalg.eigh.
    """
    obj = _as_complex2d(_to_tensor(obj))
    win = _as_complex2d(_to_tensor(win))
    H, W = obj.shape

    if end0 == 0:
        end0 = H
    if ((end0 - start0) % s) != 0:
        end0 = start0 + ((end0 - start0) // s) * s

    if end1 == 0:
        end1 = W
    if ((end1 - start1) % s) != 0:
        end1 = start1 + ((end1 - start1) // s) * s

    obj_trunc = obj[start0:end0, start1:end1]

    # init lambda
    lambd = (torch.randn((s, s), device=obj.device) + 1j * torch.randn((s, s), device=obj.device)).to(_COMPLEX)
    lambd = lambd / _fro(lambd).clamp_min(1e-20)

    for t in range(maxit):
        Z = torch.zeros((s, s, s, s), dtype=_COMPLEX, device=obj.device)

        # vertical differences
        for k1 in range(obj_trunc.shape[1]):
            idx2 = 0
            z2 = obj_trunc[idx2, k1]
            s1 = k1 % s
            for k0 in range(obj_trunc.shape[0] - 1):
                idx1 = idx2
                idx2 = (k0 + 1) % s
                z1 = z2
                z2 = obj_trunc[k0 + 1, k1]
                diff = 1.0 / torch.maximum(torch.abs(z1 * lambd[idx1, s1] - z2 * lambd[idx2, s1]),
                                           torch.tensor(eps, device=obj.device))
                Z[idx1, s1, idx1, s1] += torch.abs(z1) ** 2 * diff
                Z[idx2, s1, idx2, s1] += torch.abs(z2) ** 2 * diff
                Z[idx1, s1, idx2, s1] -= z1.conj() * z2 * diff
                Z[idx2, s1, idx1, s1] -= z2.conj() * z1 * diff

        # horizontal differences
        for k0 in range(obj_trunc.shape[0]):
            idy2 = 0
            z2 = obj_trunc[k0, idy2]
            s0 = k0 % s
            for k1 in range(obj_trunc.shape[1] - 1):
                idy1 = idy2
                idy2 = (k1 + 1) % s
                z1 = z2
                z2 = obj_trunc[k0, k1 + 1]
                diff = 1.0 / torch.maximum(torch.abs(z1 * lambd[s0, idy1] - z2 * lambd[s0, idy2]),
                                           torch.tensor(eps, device=obj.device))
                Z[s0, idy1, s0, idy1] += torch.abs(z1) ** 2 * diff
                Z[s0, idy2, s0, idy2] += torch.abs(z2) ** 2 * diff
                Z[s0, idy1, s0, idy2] -= z1.conj() * z2 * diff
                Z[s0, idy2, s0, idy1] -= z2.conj() * z1 * diff

        Z = Z.reshape(s * s, s * s)
        # Hermitian guarantee (numerical safety)
        Zh = (Z + Z.conj().T) * 0.5

        # eigenvector for smallest eigenvalue
        evals, evecs = torch.linalg.eigh(Zh)   # ascending order
        lambd_new = evecs[:, 0]

        # distance metric
        lambd_long = lambd.reshape(s * s)
        # 2 - 2*|<x,y>|
        dist = 2 - 2 * torch.abs(torch.dot(lambd_new, lambd_long.conj()))
        lambd = lambd_new.reshape(s, s)

        print(t, evals[0].item(), float(dist.real.item()))
        if float(dist.real.item()) < threshold:
            break

    # normalize lambda (average magnitude ~ 1)
    lambd_avg = torch.mean(torch.abs(lambd))
    lambd = lambd / lambd_avg.clamp_min(1e-20)
    lambd[torch.abs(lambd) < threshold] = 1 + 0j

    # roll to starting offsets
    offset0 = start0 % s
    offset1 = start1 % s
    lambd = torch.roll(lambd, shifts=(offset0, offset1), dims=(0, 1))

    # tile over object/window and crop
    rep0 = obj.shape[0] // s + 1
    rep1 = obj.shape[1] // s + 1
    lambd_obj = lambd.repeat(rep0, rep1)[:obj.shape[0], :obj.shape[1]]

    rep0w = win.shape[0] // s + 1
    rep1w = win.shape[1] // s + 1
    lambd_win = lambd.repeat(rep0w, rep1w)[:win.shape[0], :win.shape[1]]

    obj_r = obj * lambd_obj
    win_r = win / lambd_win
    return obj_r, win_r, lambd


def eliminate_grid_ambiguity_l2(obj, win, s, start0=0, end0=0, start1=0, end1=0, threshold=1e-6):
    """
    L2 variant: chooses lambda by minimizing squared TV (dense eigensolve).
    """
    obj = _as_complex2d(_to_tensor(obj))
    win = _as_complex2d(_to_tensor(win))
    H, W = obj.shape

    if end0 == 0:
        end0 = H
    if ((end0 - start0) % s) != 0:
        end0 = start0 + ((end0 - start0) // s) * s

    if end1 == 0:
        end1 = W
    if ((end1 - start1) % s) != 0:
        end1 = start1 + ((end1 - start1) // s) * s

    obj_trunc = obj[start0:end0, start1:end1]

    Z = torch.zeros((s, s, s, s), dtype=_COMPLEX, device=obj.device)

    # vertical
    for k1 in range(obj_trunc.shape[1]):
        idx2 = 0
        z2 = obj_trunc[idx2, k1]
        s1 = k1 % s
        for k0 in range(obj_trunc.shape[0] - 1):
            idx1 = idx2
            idx2 = (k0 + 1) % s
            z1 = z2
            z2 = obj_trunc[k0 + 1, k1]
            Z[idx1, s1, idx1, s1] += torch.abs(z1) ** 2
            Z[idx2, s1, idx2, s1] += torch.abs(z2) ** 2
            Z[idx1, s1, idx2, s1] -= z1.conj() * z2
            Z[idx2, s1, idx1, s1] -= z2.conj() * z1

    # horizontal
    for k0 in range(obj_trunc.shape[0]):
        idy2 = 0
        z2 = obj_trunc[k0, idy2]
        s0 = k0 % s
        for k1 in range(obj_trunc.shape[1] - 1):
            idy1 = idy2
            idy2 = (k1 + 1) % s
            z1 = z2
            z2 = obj_trunc[k0, k1 + 1]
            Z[s0, idy1, s0, idy1] += torch.abs(z1) ** 2
            Z[s0, idy2, s0, idy2] += torch.abs(z2) ** 2
            Z[s0, idy1, s0, idy2] -= z1.conj() * z2
            Z[s0, idy2, s0, idy1] -= z2.conj() * z1

    Z = Z.reshape(s * s, s * s)
    Zh = (Z + Z.conj().T) * 0.5
    evals, evecs = torch.linalg.eigh(Zh)
    lambd = evecs[:, 0]  # smallest eigenvalue
    lambd = lambd.reshape(s, s)

    offset0 = start0 % s
    offset1 = start1 % s
    lambd = torch.roll(lambd, shifts=(offset0, offset1), dims=(0, 1))

    lambd[torch.abs(lambd) < threshold] = 1 + 0j

    lambd_obj = lambd.repeat(obj.shape[0] // s + 1, obj.shape[1] // s + 1)[:obj.shape[0], :obj.shape[1]]
    lambd_win = lambd.repeat(win.shape[0] // s + 1, win.shape[1] // s + 1)[:win.shape[0], :win.shape[1]]

    obj_r = obj * lambd_obj
    win_r = win / lambd_win
    return obj_r, win_r, lambd


def relative_error(obj_tr, obj, mask):
    obj_tr = _as_complex2d(_to_tensor(obj_tr))
    obj    = _as_complex2d(_to_tensor(obj))
    mask   = _to_tensor(mask, dtype=torch.float32)
    obj_r = align_objects(obj_tr, obj, mask)
    num = torch.sum(torch.abs(obj_r - obj_tr) ** 2 * mask)
    den = torch.sum(torch.abs(obj_tr) ** 2 * mask).clamp_min(1e-20)
    return torch.sqrt(num / den).item()


def relative_measurement_error(b, meas_obj):
    b = _to_tensor(b, dtype=torch.float32).reshape(-1)
    meas_obj = _to_tensor(meas_obj, dtype=torch.float32).reshape(-1)
    b = torch.sqrt(torch.clamp(b, min=0.0))
    meas_obj = torch.sqrt(torch.clamp(meas_obj, min=0.0))
    return (torch.linalg.norm(b - meas_obj) / torch.linalg.norm(b).clamp_min(1e-20)).item()


def relative_sq_measurement_error(b, meas_obj):
    # Convert both to CPU tensors for calculation
    b_cpu = _to_tensor(b, dtype=torch.float32).reshape(-1).cpu()
    meas_obj_cpu = _to_tensor(meas_obj, dtype=torch.float32).reshape(-1).cpu()
    
    print(b_cpu.device, meas_obj_cpu.device)  # should both show 'cpu'
    
    # Compute on CPU
    err = (torch.linalg.norm(b_cpu - meas_obj_cpu) /
           torch.linalg.norm(b_cpu).clamp_min(1e-20)).item()
    
    return err


def log10_measurement_error(b, meas_obj):
    b = _to_tensor(b, dtype=torch.float32).reshape(-1)
    meas_obj = _to_tensor(meas_obj, dtype=torch.float32).reshape(-1)
    b = torch.sqrt(torch.clamp(b, min=0.0))
    meas_obj = torch.sqrt(torch.clamp(meas_obj, min=0.0))
    return torch.log10(torch.linalg.norm(b - meas_obj).clamp_min(1e-20)).item()


def relative_intensity_error(b, meas_obj):
    b = _to_tensor(b, dtype=torch.float32).reshape(-1)
    meas_obj = _to_tensor(meas_obj, dtype=torch.float32).reshape(-1)
    return (torch.linalg.norm(b - meas_obj) / torch.linalg.norm(b).clamp_min(1e-20)).item()


def normalize_window(window):
    window = _as_complex2d(_to_tensor(window))
    window = window / _fro(window).clamp_min(1e-20)
    return window


def normalize_object_and_window(obj, window):
    obj = _as_complex2d(_to_tensor(obj))
    window = _as_complex2d(_to_tensor(window))
    norm = _fro(window).clamp_min(1e-20)
    window = window / norm
    obj = obj * norm
    return obj, window

