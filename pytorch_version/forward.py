#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Pure PyTorch translation of the provided NumPy-based ptycho utilities.

Author: translated to PyTorch by ChatGPT
Original author (NumPy version): oleh.melnyk
"""

from __future__ import annotations
import math
import copy
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------
device_default = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)  # default real dtype
_COMPLEX_DTYPE = torch.complex64


# ---------------------------------------------------------------------
# Helper creators / converters
# ---------------------------------------------------------------------
def _as_tensor(x, dtype=None, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device if device is not None else x.device,
                    dtype=dtype if dtype is not None else x.dtype)
    return torch.as_tensor(x, dtype=dtype, device=device if device is not None else device_default)


def _ensure_2d_complex(t: torch.Tensor) -> torch.Tensor:
    assert t.ndim == 2, "Expected a 2D tensor"
    if not torch.is_complex(t):
        t = t.to(torch.float32).to(_COMPLEX_DTYPE)
    return t


# ---------------------------------------------------------------------
# Location generators (Torch)
# ---------------------------------------------------------------------
def loc_grid_noncirc(obj_s: Tuple[int,int], win_s: Tuple[int,int],
                     shift_s: Tuple[float,float], float_shift: bool,
                     origin_s: Tuple[float,float] = (0.0, 0.0)):
    o0,o1 = int(obj_s[0]), int(obj_s[1])
    w0,w1 = int(win_s[0]), int(win_s[1])
    s0,s1 = float(shift_s[0]), float(shift_s[1])
    y0,x0 = float(origin_s[0]), float(origin_s[1])

    # cover the full footprint: from origin to origin + (o-w)
    # +1e-6 avoids dropping the last step due to float rounding
    l1 = torch.arange(y0, y0 + (o0 - w0) + 1e-6, s0, device=device_default)
    l2 = torch.arange(x0, x0 + (o1 - w1) + 1e-6, s1, device=device_default)

    n1, n2 = l1.numel(), l2.numel()
    dt = torch.float32 if float_shift else torch.int64
    locations = torch.zeros((n1*n2, 2), dtype=dt, device=device_default)
    locations[:,0] = l1.repeat_interleave(n2)
    locations[:,1] = l2.repeat(n1)
    if not float_shift:  # integer grid
        locations = locations.to(torch.int64)
    return locations

def loc_grid_circ(obj_s: Tuple[int,int], shift_s: Tuple[float,float], float_shift: bool,
                  origin_s: Tuple[float,float] = (0.0, 0.0)):
    o0,o1 = int(obj_s[0]), int(obj_s[1])
    s0,s1 = float(shift_s[0]), float(shift_s[1])
    y0,x0 = float(origin_s[0]), float(origin_s[1])

    l1 = y0 + torch.arange(0, o0, s0, device=device_default)
    l2 = x0 + torch.arange(0, o1, s1, device=device_default)

    n1, n2 = l1.numel(), l2.numel()
    dt = torch.float32 if float_shift else torch.int64
    locations = torch.zeros((n1*n2, 2), dtype=dt, device=device_default)
    locations[:,0] = l1.repeat_interleave(n2)
    locations[:,1] = l2.repeat(n1)
    if not float_shift:
        locations = locations.to(torch.int64)
    return locations


def loc_Fermat_spiral(obj_s: Tuple[int, int], win_s: Tuple[int, int], seed_size: float):
    """Fermat spiral sampling (Torch). Returns (R,2) int32 locations."""
    o0, o1 = int(obj_s[0]), int(obj_s[1])
    w0, w1 = int(win_s[0]), int(win_s[1])
    s1 = o0 - w0
    s2 = o1 - w1

    N = int(math.ceil(2 * (0.5 * max(s1, s2) / float(seed_size)) ** 2))
    phi_0 = 8 * math.pi / (1 + math.sqrt(5)) ** 2

    locs = []
    for n in range(N):
        r = seed_size * math.sqrt(n)
        phi = n * phi_0
        x = int(round(r * math.cos(phi) + 0.5 * s1))
        y = int(round(r * math.sin(phi) + 0.5 * s2))
        if x < 0 or x >= s1 or y < 0 or y >= s2:
            continue
        locs.append([x, y])

    if len(locs) == 0:
        return torch.zeros((0, 2), dtype=torch.int32, device=device_default)
    return torch.tensor(locs, dtype=torch.int32, device=device_default)


# ---------------------------------------------------------------------
# Ptycho class (Torch)
# ---------------------------------------------------------------------
class ptycho:
    def __init__(self, **kwargs):
        # ---------------- Object shape ----------------
        assert 'object_shape' in kwargs.keys(), "Object shape is not specified"
        assert isinstance(kwargs['object_shape'], tuple), "Object shape is not a tuple"
        assert len(kwargs['object_shape']) == 2, "Object shape is not 2D"
        assert kwargs['object_shape'][0] > 0 and kwargs['object_shape'][1] > 0, "Object shape should be positive"
        self.object_shape = kwargs['object_shape']
        self.obj = torch.zeros(self.object_shape, dtype=_COMPLEX_DTYPE, device=device_default)

        # ---------------- Window (Probe) ----------------
        if 'window' in kwargs.keys():
            self.set_window(kwargs['window'])
        else:
            assert 'window_shape' in kwargs.keys(), "Neither window nor window_shape is specified"
            assert isinstance(kwargs['window_shape'], tuple), "Window shape is not a tuple"
            assert len(kwargs['window_shape']) == 2, "Window shape is not 2D"
            assert kwargs['window_shape'][0] > 0 and kwargs['window_shape'][1] > 0, "Window shape should be positive"
            dummy = torch.zeros(kwargs['window_shape'], dtype=_COMPLEX_DTYPE, device=device_default)
            self.set_window(dummy)

        # ---------------- Circular & float shifts ----------------
        self.circular = bool(kwargs.get('circular', False))
        if 'circular' not in kwargs:
            print('Warning: Parameter circular is not specified. Set to False')

        self.float_shift = bool(kwargs.get('float_shift', False))
        if 'float_shift' not in kwargs:
            print('Warning: Parameter float_shift is not specified. Set to False')

        # ---------------- Origin (base offset) ----------------
        self.origin = kwargs.get('origin', (0.0, 0.0))
        if isinstance(self.origin, (int, float)):
            self.origin = (float(self.origin), float(self.origin))
        else:
            self.origin = (float(self.origin[0]), float(self.origin[1]))

        # If you demand integer pixel origin when float shifts are OFF:
        if not self.float_shift:
            assert all(abs(v - round(v)) < 1e-6 for v in self.origin), \
                "Non-float shifts require integer origin."


        # ---------------- Scan locations ----------------
        if 'locations' in kwargs.keys():
            locs = _as_tensor(kwargs['locations'])
            assert locs.ndim == 2 and locs.shape[1] == 2, "Locations should be an array of 2d coordinates"
            self.locations = locs
            self.check_outbound()
        elif 'loc_type' in kwargs.keys():
            loc_type = kwargs['loc_type']
            if loc_type == 'grid':
                assert 'shift' in kwargs.keys(), "Shift size is not given for loc_type = grid"
                shift_kw = kwargs['shift']
                if self.float_shift:
                    assert isinstance(shift_kw, (tuple, int, float)), "Shift must be tuple/int/float for float_shift=True"
                else:
                    assert isinstance(shift_kw, (tuple, int)), "Shift must be tuple/int (or set float_shift=True)"
                if isinstance(shift_kw, tuple):
                    shift = (shift_kw[0], shift_kw[1])
                elif isinstance(shift_kw, int):
                    shift = (shift_kw, shift_kw)
                    if self.float_shift:
                        self.float_shift = False
                else:  # float and float_shift=True
                    shift = (float(shift_kw), float(shift_kw))

                assert shift[0] < self.object_shape[0] and shift[1] < self.object_shape[1], "Shift is larger or equal to object shape"
                if shift[0] > self.window_shape[0] or shift[1] > self.window_shape[1]:
                    print('Warning: Shift is larger than window shape')
                if not self.float_shift:
                    if (self.object_shape[0] % int(shift[0]) != 0 or self.window_shape[0] % int(shift[0]) != 0 or
                        self.object_shape[1] % int(shift[1]) != 0 or self.window_shape[1] % int(shift[1]) != 0):
                        print('Warning: Shift is not divisor of object shape or window shape')

                assert shift[0] > 0 and shift[1] > 0, "Shift is negative"

                self.shift = shift
                if self.circular:
                    self.locations = loc_grid_circ(self.object_shape, shift, self.float_shift, self.origin)
                else:
                    self.locations = loc_grid_noncirc(self.object_shape, self.window_shape, shift, self.float_shift, self.origin)

            elif loc_type == 'spiral':
                assert 'fermat_seed_size' in kwargs.keys(), "Fermat spiral seed size (fermat_seed_size) is not given for loc_type = spiral"
                fss = kwargs['fermat_seed_size']
                assert isinstance(fss, (float, int)), "Fermat spiral seed size must be float or int"
                assert fss > 0, "Fermat spiral seed size should be positive"
                self.locations = loc_Fermat_spiral(self.object_shape, self.window_shape, float(fss))
            else:
                raise AssertionError("loc_type is not in [grid, spiral]")
        else:
            raise AssertionError("Neither locations nor location type generation is given")

        self.R = int(self.locations.shape[0])

        # ---------------- Object mask from locations ----------------
        if self.float_shift:
            self.mask = torch.ones(self.object_shape, dtype=torch.float32, device=device_default)
        else:
            self.mask = torch.zeros(self.object_shape, dtype=torch.float32, device=device_default)
            for k in range(self.R):
                loc0 = int(self.locations[k, 0].item())
                loc1 = int(self.locations[k, 1].item())
                self.mask[loc0:(loc0 + self.window_shape[0]), loc1:(loc1 + self.window_shape[1])] = 1.0

        # ---------------- Fourier dimension ----------------
        if 'fourier_dimension' in kwargs.keys():
            fd = kwargs['fourier_dimension']
            assert isinstance(fd, tuple) and len(fd) == 2, "Fourier dimension must be a 2D tuple"
            assert fd[0] > 0 and fd[1] > 0, "Fourier dimension should be positive"
            self.fourier_dimension = [int(fd[0]), int(fd[1])]
            if self.fourier_dimension[0] < self.window_shape[0]:
                print('Warning: Fourier dimension[0] is smaller than window shape[0]. Increased.')
                self.fourier_dimension[0] = self.window_shape[0]
            if self.fourier_dimension[1] < self.window_shape[1]:
                print('Warning: Fourier dimension[1] is smaller than window shape[1]. Increased.')
                self.fourier_dimension[1] = self.window_shape[1]
        else:
            print('Warning: Fourier dimension is not specified, set to object_shape')
            self.fourier_dimension = [int(self.object_shape[0]), int(self.object_shape[1])]

        # ---------------- Detector mask ----------------
        if 'detector_mask' in kwargs.keys():
            dm = kwargs['detector_mask']
            if isinstance(dm, torch.Tensor):
                assert dm.ndim == 2, "Detector mask is not 2D"
                assert dm.shape[0] == self.fourier_dimension[0] and dm.shape[1] == self.fourier_dimension[1], \
                    "Detector mask shape is different from Fourier dimension"
                # Expect 0/1 values; convert to complex
                self.detector_mask = dm.to(torch.float32).clamp(0, 1).to(_COMPLEX_DTYPE).to(device_default)
            else:
                if isinstance(dm, tuple):
                    det = [int(dm[0]), int(dm[1])]
                else:
                    det = [int(dm), int(dm)]
                assert det[0] > 0 and det[1] > 0, "Detector mask size is negative"
                det[0] = min(det[0], self.fourier_dimension[0])
                det[1] = min(det[1], self.fourier_dimension[1])

                self.detector_mask = torch.zeros(self.fourier_dimension, dtype=_COMPLEX_DTYPE, device=device_default)
                dr1 = (self.fourier_dimension[0] - det[0]) // 2
                dr2 = (self.fourier_dimension[1] - det[1]) // 2
                self.detector_mask[dr1:(dr1 + det[0]), dr2:(dr2 + det[1])] = 1 + 0j
        else:
            self.detector_mask = torch.ones(self.fourier_dimension, dtype=_COMPLEX_DTYPE, device=device_default)

        # ---------------- Threads (kept for API compatibility) ----------------
        self.num_threads = int(kwargs.get('num_threads', 1))

    # ---------------- Copy ----------------
    def copy(self):
        pty = ptycho(
            object_shape=copy.deepcopy(tuple(self.object_shape)),
            window=copy.deepcopy(self.window.detach().clone()),
            circular=copy.deepcopy(self.circular),
            float_shift=copy.deepcopy(self.float_shift),
            locations=copy.deepcopy(self.locations.detach().clone()),
            fourier_dimension=copy.deepcopy(tuple(self.fourier_dimension)),
            detector_mask=copy.deepcopy(self.detector_mask.detach().clone()),
            num_threads=copy.deepcopy(self.num_threads),
            origin=copy.deepcopy(tuple(self.origin)),
        )
        pty.set_object(self.obj.detach().clone())
    
        # preserve multimode if present
        if self.has_multimode():
            pty.set_windows(self.window_modes.detach().clone())
    
        if hasattr(self, 'shift'):
            pty.shift = copy.deepcopy(self.shift)
        if hasattr(self, 'loc_type'):
            pty.loc_type = copy.deepcopy(self.loc_type)
        return pty


    # ---------------- Validations ----------------
    def check_outbound(self):
        if not hasattr(self, 'locations') or not hasattr(self, 'circular') or not hasattr(self, 'window_shape'):
            return
        if not self.circular and not self.float_shift:
            min_loc = torch.min(self.locations, dim=0).values
            max_loc = torch.max(self.locations, dim=0).values
            outbound = (
                (min_loc[0] < 0) or (min_loc[1] < 0) or
                (max_loc[0] + self.window_shape[0] > self.object_shape[0]) or
                (max_loc[1] + self.window_shape[1] > self.object_shape[1])
            )
            assert not bool(outbound), "Location out of bound"

    # ---------------- Setters ----------------
    def set_object(self, obj):
        t = _as_tensor(obj, device=device_default)
        assert t.ndim == 2, "Object is not 2D"
        t = _ensure_2d_complex(t)
        assert self.window_shape[0] <= t.shape[0] and self.window_shape[1] <= t.shape[1], "Window is larger than object"

        if hasattr(self, 'obj'):
            old = self.obj
            if old.shape != t.shape:
                print('Warning: Object shape has changed')
                self.object_shape = t.shape
                self.check_outbound()

        self.obj = t
        self.object_shape = t.shape

    def set_window(self, window):
        t = _as_tensor(window, device=device_default)
        assert t.ndim == 2, "Window is not 2D"
        t = _ensure_2d_complex(t)
        assert t.shape[0] <= self.object_shape[0] and t.shape[1] <= self.object_shape[1], "Window is larger than object"

        if hasattr(self, 'window'):
            old = self.window
            if old.shape != t.shape:
                print('Warning: Window shape has changed')
                self.window_shape = t.shape
                self.check_outbound()

        self.window = t
        self.window_shape = t.shape
        # ---------------- Multimode probe support ----------------
    def set_windows(self, windows: torch.Tensor):
        """
        windows: (K, H, W) complex64 tensor of K probe modes.
        All modes must have same spatial shape as the (single) window.
        """
        W = _as_tensor(windows, device=device_default)
        assert W.ndim == 3 and torch.is_complex(W), "windows must be complex (K,H,W)"
        K, H, Wd = int(W.shape[0]), int(W.shape[1]), int(W.shape[2])
        if not hasattr(self, "window_shape"):
            raise AssertionError("set a single window first so window_shape is defined")
        assert (H, Wd) == self.window_shape, "Each mode must match window shape"
        self.window_modes = W.to(_COMPLEX_DTYPE)
        self.K = K

    def clear_windows(self):
        """Disable multimode (return to single-mode)."""
        if hasattr(self, "window_modes"):
            del self.window_modes
        self.K = 0

    def has_multimode(self) -> bool:
        return hasattr(self, "window_modes") and isinstance(self.window_modes, torch.Tensor) and (self.window_modes.ndim == 3)

    def set_window_mode(self, idx: int, w: torch.Tensor):
        """Replace a single mode in-place (keeps shapes consistent)."""
        assert self.has_multimode(), "call set_windows first"
        t = _ensure_2d_complex(_as_tensor(w, device=device_default))
        assert t.shape == self.window_shape, "mode window shape mismatch"
        self.window_modes[idx, ...] = t

    def get_window_mode(self, idx: int) -> torch.Tensor:
        assert self.has_multimode(), "no window_modes set"
        return self.window_modes[idx, ...].clone()

    def num_modes(self) -> int:
        return int(self.window_modes.shape[0]) if self.has_multimode() else 1

    # ---------------- Core FFT helpers ----------------
    @staticmethod
    def _fft2c(x: torch.Tensor) -> torch.Tensor:
        # centered FFT2 (no normalization)
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))

    @staticmethod
    def _ifft2c(X: torch.Tensor) -> torch.Tensor:
        # centered IFFT2 (Torch matches NumPy's normalization: 1/N; we follow original by multiplying later when needed)
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1))), dim=(-2, -1))

    # ---------------- Oversampled forward/adjoint ----------------
    def forward_2D_os(self, z: torch.Tensor) -> torch.Tensor:
        """
        Oversampled forward (Torch). Mimics NumPy version:
        - place window*z into top-left of padded canvas
        - roll so the window center lands at (0,0)
        - FFT2
        - roll to center DC
        - apply detector mask
        """
        z = _ensure_2d_complex(z)
        h1 = self.fourier_dimension[0] // 2
        h2 = self.fourier_dimension[1] // 2
        r1 = self.window_shape[0] // 2
        r2 = self.window_shape[1] // 2

        out_padded = torch.zeros(self.fourier_dimension, dtype=_COMPLEX_DTYPE, device=device_default)
        out_padded[:self.window_shape[0], :self.window_shape[1]] = self.window * z

        # place middle of illuminated area to zero
        out_padded = torch.roll(out_padded, shifts=(-r1, -r2), dims=(0, 1))

        out_fft = torch.fft.fft2(out_padded)

        out_fft = torch.roll(out_fft, shifts=(h1, h2), dims=(0, 1))

        forw = out_fft * self.detector_mask
        return forw

    def forward_adj_2D_os(self, forw: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of oversampled forward (Torch). Mirrors NumPy logic:
        - apply detector mask
        - uncenter (roll back)
        - IFFT2 then multiply by prod(fourier_dimension) to match original scaling
        - roll back the window center
        - multiply by conj(window) and crop to window size
        """
        forw = _ensure_2d_complex(forw)
        d1, d2 = self.window_shape
        h1 = self.fourier_dimension[0] // 2
        h2 = self.fourier_dimension[1] // 2

        unzoomed_padded = forw * self.detector_mask
        unzoomed_padded = torch.roll(unzoomed_padded, shifts=(-h1, -h2), dims=(0, 1))
        out_spatial = torch.fft.ifft2(unzoomed_padded) * (self.fourier_dimension[0] * self.fourier_dimension[1])

        r1 = d1 // 2
        r2 = d2 // 2

        out_spatial = torch.roll(out_spatial, shifts=(r1, r2), dims=(0, 1))
        result = torch.conj(self.window) * out_spatial[:d1, :d2]
        return result

    # ---------------- Measurements mapping ----------------
    @staticmethod
    def forward_to_meas_2D(forw: torch.Tensor) -> torch.Tensor:
        return torch.abs(forw) ** 2

    def forward_to_meas_2D_pty(self, forw: torch.Tensor) -> torch.Tensor:
        return self.forward_to_meas_2D(forw)

    # ---------------- Shifts ----------------
    def shift_vec(self, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Shift object z by vector r (length-2), supporting:
        - float_shift: subpixel shift via Fourier phase ramps
        - circular: integer wrap-around shift
        - else: handled elsewhere by cropping
        """
        z = _ensure_2d_complex(z)
        if self.float_shift:
            # Fourier shift theorem (note the sign convention matching original)
            H, W = self.object_shape
            # r is 2-vector; ensure tensor on device
            r = _as_tensor(r, device=z.device).to(torch.float32)
            # FFT
            ZF = torch.fft.fft2(z)
            # multiply by exp(2j*pi*-r[axis]*k/N) along each axis
            ky = torch.arange(H, device=z.device, dtype=torch.float32).view(H, 1)
            kx = torch.arange(W, device=z.device, dtype=torch.float32).view(1, W)
            phase_y = torch.exp(2j * math.pi * (-r[0]) * ky / H)
            phase_x = torch.exp(2j * math.pi * (-r[1]) * kx / W)
            ZF = (ZF * phase_y) * phase_x
            z_r = torch.fft.ifft2(ZF)
        else:
            # integer circular roll
            r0 = int(r[0].item()) if isinstance(r, torch.Tensor) else int(r[0])
            r1 = int(r[1].item()) if isinstance(r, torch.Tensor) else int(r[1])
            z_r = torch.roll(z, shifts=(r0, r1), dims=(0, 1))
        return z_r

    # ---------------- Full ptychographic forward/adjoint ----------------
    def forward_2D_pty(self, z: torch.Tensor) -> torch.Tensor:
        """
        Build a (F0, F1, R) stack of Fourier-domain fields for each location.
        Mirrors NumPy version:
          - if float_shift or circular: shift via shift_vec then crop to window
          - else: take spatial crop from object
          - feed through oversampled forward
        """
        z = _ensure_2d_complex(z)
        d1, d2 = self.window_shape
        F0, F1 = self.fourier_dimension
        forw = torch.zeros((F0, F1, self.R), dtype=_COMPLEX_DTYPE, device=device_default)

        for r in range(self.R):
            loc = self.locations[r, :]
            if self.float_shift or self.circular:
                z_r = self.shift_vec(z, -loc)
                z_r = z_r[:d1, :d2]
            else:
                i0 = int(loc[0].item()); i1 = int(loc[1].item())
                z_r = z[i0:(i0 + d1), i1:(i1 + d2)]
            forw[:, :, r] = self.forward_2D_os(z_r)

        return forw
        
    def forward_2D_pty_multimode(self, z: torch.Tensor):
        """
        Return list of per-mode Fourier fields [Phi_m] with shape (F0,F1,R) each.
        Uses existing single-mode machinery by temporarily swapping self.window.
        """
        assert self.has_multimode(), "no window_modes present"
        z = _ensure_2d_complex(z)
    
        out_list = []
        old_win = self.window
        try:
            for m in range(self.num_modes()):
                self.window = self.window_modes[m, ...]  # swap-in mode m
                out_list.append(self.forward_2D_pty(z))  # (F0,F1,R)
        finally:
            self.window = old_win  # always restore
        return out_list

    def forward_adj_2D_pty(self, forw: torch.Tensor) -> torch.Tensor:
        """
        Adjoint of the above. Accumulates contributions back to object domain.
        """
        forw = _as_tensor(forw, dtype=_COMPLEX_DTYPE, device=device_default)
        d1, d2 = self.window_shape
        z = torch.zeros(self.object_shape, dtype=_COMPLEX_DTYPE, device=device_default)

        for r in range(self.R):
            loc = self.locations[r, :]
            if self.float_shift or self.circular:
                z_r = torch.zeros(self.object_shape, dtype=_COMPLEX_DTYPE, device=device_default)
                z_r[:d1, :d2] = self.forward_adj_2D_os(forw[:, :, r])
                z_r = self.shift_vec(z_r, loc)
                z = z + z_r
            else:
                i0 = int(loc[0].item()); i1 = int(loc[1].item())
                z[i0:(i0 + d1), i1:(i1 + d2)] += self.forward_adj_2D_os(forw[:, :, r])

        return z

