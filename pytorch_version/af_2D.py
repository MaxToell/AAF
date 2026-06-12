#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# af_2D.py  (PyTorch port)
# Author: ported from NumPy version by ChatGPT
# Requires: forward.py (PyTorch version of your ptycho model)

from __future__ import annotations
import copy
import types
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Globals / defaults
# ---------------------------------------------------------------------
device_default = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)
_COMPLEX = torch.complex64


# ---------------------------------------------------------------------
# Try to import your Torch 'forward' and 'utility_2D'; provide shims if absent
# ---------------------------------------------------------------------
import importlib

_forward = importlib.import_module("forward")  # must exist (your Torch port)

try:
    util = importlib.import_module("utility_2D")  # optional
    has_util = True
except Exception:
    has_util = False

    class _UtilShim:
        @staticmethod
        def normalize_object_and_window(z: torch.Tensor, w: torch.Tensor):
            zn = torch.linalg.norm(z)
            wn = torch.linalg.norm(w)
            if zn > 0 and wn > 0:
                # scale so product ~ 1 (simple choice)
                s = torch.sqrt(zn * wn)
                return z / s, w * s
            return z, w

        @staticmethod
        def align_objects(ref: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
            # crude: remove global phase to align to ref (where mask==1)
            if ref.numel() == 0 or mask.numel() == 0:
                return x
            idx = mask.bool()
            if torch.count_nonzero(idx) == 0:
                return x
            a = ref[idx]
            b = x[idx]
            phase = torch.angle((a.conj() * b).sum())
            return x * torch.exp(-1j * phase)

    util = _UtilShim()
# --- Phase-regularization helpers (branch-cut aware) ---
def _circ_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # wrapped phase difference in (-pi, pi]
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))

def _grad_phase_TV(z: torch.Tensor, lam: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Gradient of anisotropic TV on the phase field, using circular differences.
    Returns a complex tensor shaped like z.
    """
    amp = torch.abs(z).clamp_min(eps)
    phi = torch.angle(z)
    # forward diffs
    dx = _circ_diff(torch.roll(phi, -1, dims=0), phi)
    dy = _circ_diff(torch.roll(phi, -1, dims=1), phi)
    # divergence (backward diffs)
    div = (dx - torch.roll(dx, 1, dims=0)) + (dy - torch.roll(dy, 1, dims=1))
    # map back to complex domain
    return 1j * (z / amp) * lam * div


# ---------------------------------------------------------------------
# Minimal in-file replacement for gradient_descent_solver.grad_desc
# ---------------------------------------------------------------------
class _GradDesc:
    """
    Internal replacement for gradient_descent_solver.grad_desc used by af_2D.
    Expects:
      grad_f(x, forw, sqrt_meas) -> grad_x
      obj_f(x) -> (objective_value, l2_term, forw, sqrt_meas)
    Performs simple AG (Nesterov-like) if enabled.
    """
    def __init__(self,
                 grad_f,
                 obj_f,
                 x0: torch.Tensor,
                 number_of_iterations: int,
                 grad_threshold: float,
                 learning_rate: float,
                 AG_params: types.SimpleNamespace,
                 learn_rate_decay: float,
                 proximal_TV: bool,
                 alpha_TV: float,
                 verbose: int,
                 track: bool,
                 obj_tr: Optional[torch.Tensor],
                 track_it: Optional[torch.Tensor],
                 mask: Optional[torch.Tensor] = None):
        self.grad_f = grad_f
        self.obj_f = obj_f
        self.x = x0.detach().clone().to(device_default)
        self.niter = int(number_of_iterations)
        self.grad_thr = float(grad_threshold)
        self.lr = float(learning_rate)
        self.AG = AG_params or types.SimpleNamespace(enable_AG=False, control=0.5, tau=0.3, AG_iterations=2)
        self.decay = float(learn_rate_decay)
        self.proxTV = bool(proximal_TV)
        self.alphaTV = float(alpha_TV)
        self.verbose = int(verbose)
        self.track = bool(track)
        self.obj_tr = obj_tr if isinstance(obj_tr, torch.Tensor) else None
        self.track_it = track_it if isinstance(track_it, torch.Tensor) else None
        self.mask = mask if isinstance(mask, torch.Tensor) else None

        self.measurement_error = torch.zeros(0, device=device_default)
        self.objective = torch.zeros(0, device=device_default)
        self.object_error = torch.zeros(0, device=device_default)
        self.tracked_objects = None


    def _append_track(self, meas_err, obj_val, x):
        if not self.track:
            return
        self.measurement_error = torch.cat([self.measurement_error, torch.tensor([meas_err], device=device_default)])
        self.objective = torch.cat([self.objective, torch.tensor([obj_val], device=device_default)])
        if self.obj_tr is not None:
            # object error as relative Fro norm
            num = torch.linalg.norm(self.obj_tr - x)
            den = torch.linalg.norm(self.obj_tr).clamp_min(1e-20)
            self.object_error = torch.cat([self.object_error, (num / den).reshape(1)])
        else:
            self.object_error = torch.cat([self.object_error, torch.tensor([0.0], device=device_default)])

    def run(self):
        y = self.x.clone()
        t_prev = 1.0
        tracked = []
        
        # Handle zero-iteration calls safely
        if self.niter <= 0:
            obj_val, l2_term, forw, sqrt_meas = self.obj_f(y)
            if self.track:
                meas_err = l2_term.item() if isinstance(l2_term, torch.Tensor) else float(l2_term)
                self._append_track(
                    meas_err,
                    obj_val.item() if isinstance(obj_val, torch.Tensor) else float(obj_val),
                    y
                )
            return y, obj_val   # <-- return INSIDE the if
        
        
        
        
        for it in range(1, self.niter + 1):
            obj_val, l2_term, forw, sqrt_meas = self.obj_f(y)
            grad = self.grad_f(y, forw, sqrt_meas)

            gnorm = torch.linalg.norm(grad).item()
            # measurement error = l2_term / ||meas||^2 approx (keep raw here)
            meas_err = l2_term.item() if isinstance(l2_term, torch.Tensor) else float(l2_term)
            self._append_track(meas_err, obj_val.item() if isinstance(obj_val, torch.Tensor) else float(obj_val), y)

            if self.verbose and (it == 1 or it % 5 == 0 or it == self.niter):
                print(f"[GD] iter {it:3d}  obj={obj_val:.4e}  ||g||={gnorm:.4e}  lr={self.lr:.3e}")

            if gnorm <= self.grad_thr:
                break

            # take step
            y = y - self.lr * grad

            # optional proximal-TV (very light; anisotropic TV on |y|)
            if self.proxTV and self.alphaTV > 0:
                amp = torch.abs(y)
                # ROF-like shrink on finite differences (simple heuristic)
                dx = F.pad(amp[:, 1:] - amp[:, :-1], (1, 0))
                dy = F.pad(amp[1:, :] - amp[:-1, :], (0, 0, 1, 0))
                shrink = torch.clamp(1 - self.lr * self.alphaTV, min=0.0)
                dx = dx * shrink
                dy = dy * shrink
                # reconstruct amplitude approximately
                amp2 = amp  # (keep simple; proper prox TV requires solver)
                y = amp2 * torch.exp(1j * torch.angle(y))

            # AG (Nesterov-like)
            if getattr(self.AG, "enable_AG", False):
                t = (1 + torch.sqrt(torch.tensor(1 + 4 * t_prev * t_prev))) / 2
                beta = (t_prev - 1) / t
                x_new = y + beta * (y - self.x)
                self.x = y
                y = x_new
                t_prev = float(t.item())
            else:
                self.x = y

            # decay
            if self.decay != 0.0:
                dec = ((it) / (it + 1)) ** self.decay
                self.lr *= float(dec)

            # tracking snapshots
            if self.track and self.track_it is not None and (it in self.track_it.tolist()):
                tracked.append(self.x.detach().clone())

        if self.track and tracked:
            self.tracked_objects = torch.stack(tracked, dim=0)

        return self.x, obj_val


# ---------------------------------------------------------------------
# AF classes (ported to Torch)
# ---------------------------------------------------------------------
class amplitude_flow:
    """
    Single-variable amplitude flow (optimize the object OR the window).
    Torch port of your NumPy class. Uses the same signatures.
    """
    def __init__(self, **kwargs):
        # ----- epsilon (meas floor) -----
        if 'epsilon' in kwargs:
            self.epsilon = float(kwargs['epsilon']) if isinstance(kwargs['epsilon'], (float, int)) else 0.0
        else:
            print('epsilon is not given. Set to 0.')
            self.epsilon = 0.0

        # ----- measurements / sqb -----
        assert ('measurements' in kwargs) or ('sqb' in kwargs), "Neither measurements nor their square root is given"
        if 'measurements' in kwargs:
            meas = torch.as_tensor(kwargs['measurements'], device=device_default)
            self.sqb = torch.sqrt(torch.clamp(meas, min=0.0) + self.epsilon)
        else:
            self.sqb = torch.as_tensor(kwargs['sqb'], device=device_default)

        # ----- poly / ptycho wiring -----
        self.poly = kwargs.get('poly', None)
        if self.poly is None:
            # Need a ptycho when no poly
            assert 'ptycho' in kwargs, "Forward model of ptycho class is not given."
            assert isinstance(kwargs['ptycho'], _forward.ptycho), "ptycho is not an instance of class ptycho."
            self.par = kwargs['ptycho'].copy()
        else:
            # If a ptycho is provided, use it, else take template from the first poly channel
            if 'ptycho' in kwargs and isinstance(kwargs['ptycho'], _forward.ptycho):
                self.par = kwargs['ptycho'].copy()
            else:
                # poly.pars is a list of ptycho clones; use the first as a template
                self.par = self.poly.pars[0].copy()

        self.R = int(self.par.locations.shape[0])
        self.delta1 = self.par.window.shape[0]
        self.delta2 = self.par.window.shape[1]

        # ----- AG params -----
        self.AG_params = copy.deepcopy(kwargs.get('AG_params', types.SimpleNamespace(
            enable_AG=False, control=0.5, tau=0.3, AG_iterations=2
        )))

        # ----- Core hyperparams -----
        self.number_of_iterations = int(kwargs['number_of_iterations'])
        self.grad_threshold = float(kwargs.get('grad_threshold', 0.0))
        self.learning_rate_type = kwargs.get('learning_rate_type', 'optimal')
        self.learning_rate = float(kwargs.get('learning_rate', 1.0))
        self.learn_rate_decay = float(kwargs.get('learn_rate_decay', 0.0))

        # ----- Penalties -----
        self.alpha_T = float(kwargs.get('alpha_T', 0.0))
        self.alpha_R = float(kwargs.get('alpha_R', 0.0))
        self.alpha_ST = float(kwargs.get('alpha_ST', 0.0))
        self.alpha_TV = float(kwargs.get('alpha_TV', 0.0))
        self.TV_param = float(kwargs.get('TV_param', 0.0))
        self.proximal_TV = bool(kwargs.get('proximal_TV', False))

        # ----- Verbose/track -----
        self.verbose = int(kwargs.get('verbose', 0))
        self.track = bool(kwargs.get('track', False))
        self.obj_tr = kwargs.get('obj_tr', None)
        if isinstance(self.obj_tr, torch.Tensor):
            self.obj_tr = self.obj_tr.to(device_default)

        # ----- Tracking arrays -----
        self.measurement_error = torch.zeros(0, device=device_default)
        self.objective = torch.zeros(0, device=device_default)
        self.object_error = torch.zeros(0, device=device_default)
        self.track_it = torch.as_tensor(kwargs.get('track_it', []), device=device_default) if self.track else None

        # Cache for shift refinement (used later)
        self._tile_freq_cache = {}


    # ---------- gradients (Torch) ----------
    def fast_grad_AF_2D_os(self, forw: torch.Tensor, sqrt_meas: torch.Tensor, sqb: torch.Tensor):
        diff = 1 - sqb / sqrt_meas
        t = forw * diff
        grad2 = self.par.forward_adj_2D_os(t)
        return grad2

    def fast_grad_L2_2D_pty(self, forw: torch.Tensor, sqrt_meas: torch.Tensor):
        grad = torch.zeros(self.par.object_shape, dtype=_COMPLEX, device=device_default)
        for r in range(self.R):
            loc = self.par.locations[r, :]
            grad_r = torch.zeros(self.par.object_shape, dtype=_COMPLEX, device=device_default)
            grad_r[:self.delta1, :self.delta2] = self.fast_grad_AF_2D_os(forw[:, :, r], sqrt_meas[:, :, r], self.sqb[:, :, r])
            grad_r = self.par.shift_vec(grad_r, loc)  # inverse shift back to object grid
            grad = grad + grad_r
        return grad

    def fast_grad_L2_win_2D_pty(self, forw: torch.Tensor, sqrt_meas: torch.Tensor):
        grad = torch.zeros((self.delta1, self.delta2), dtype=_COMPLEX, device=device_default)
        z = self.par.obj.detach().clone()
        for r in range(self.R):
            loc = self.par.locations[r, :]
            z_r = self.par.shift_vec(z, -loc)
            self.par.set_window(z_r[:self.delta1, :self.delta2])
            grad_r = self.fast_grad_AF_2D_os(forw[:, :, r], sqrt_meas[:, :, r], self.sqb[:, :, r])
            grad = grad + grad_r
        return grad

    # ---------- penalties (Torch) ----------
    def grad_smoothness_Tikhonov(self, z: torch.Tensor):
        if self.par.circular:
            gr = (4 * z
                  - torch.roll(z, 1, dims=0) - torch.roll(z, -1, dims=0)
                  - torch.roll(z, 1, dims=1) - torch.roll(z, -1, dims=1))
        else:
            gr = torch.zeros_like(z)
            gr[:-1, :] += z[:-1, :] - z[1:, :]
            gr[1:, :]  += z[1:, :] - z[:-1, :]
            gr[:, :-1] += z[:, :-1] - z[:, 1:]
            gr[:, 1:]  += z[:, 1:] - z[:, :-1]
        return gr

    def grad_TV_smoothed(self, z: torch.Tensor):
        # Use |df| rather than df**2 for complex inputs
        if self.par.circular:
            df = z - torch.roll(z, 1, dims=0); gr = df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
            df = z - torch.roll(z, -1, dims=0); gr += df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
            df = z - torch.roll(z, 1, dims=1); gr += df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
            df = z - torch.roll(z, -1, dims=1); gr += df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
        else:
            gr = torch.zeros_like(z)
            df = z[:-1, :] - z[1:, :];            gr[:-1, :] += df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
            df = z[1:, :] - z[:-1, :];            gr[1:,  :] += df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
            df = z[:, :-1] - z[:, 1:];            gr[:, :-1] += df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
            df = z[:, 1:] - z[:, :-1];            gr[:, 1:]  += df / torch.sqrt(torch.abs(df) ** 2 + self.TV_param)
        return 0.5 * gr

    def grad_AF_2D_pen(self, z: torch.Tensor):
        grad = torch.zeros_like(z)

        # Simple Tikhonov term
        if self.alpha_T != 0:
            grad = grad + self.alpha_T * z

        # Avoid using alpha_R > 0 for complex phase objects (forces real phase)
        if self.alpha_R != 0:
            grad = grad + self.alpha_R * (1j * z.imag.to(_COMPLEX))

        # Smoothness (Laplacian-type)
        if self.alpha_ST != 0:
            grad = grad + self.alpha_ST * self.grad_smoothness_Tikhonov(z)

        # Regular TV on the complex amplitude
        if self.alpha_TV != 0 and not self.proximal_TV:
            grad = grad + self.alpha_TV * self.grad_TV_smoothed(z)

        # NEW: circular (branch-cut–aware) TV on the phase
        if getattr(self, "alpha_TV_phase", 0.0) != 0.0:
            grad = grad + _grad_phase_TV(z, self.alpha_TV_phase, eps=max(1e-8, self.TV_param))

        return grad



    def fast_grad_AF_2D_pty(self, z, forw, sqrt_meas):
        if self.poly is None:
            grad = self.fast_grad_L2_2D_pty(forw, sqrt_meas)
            grad = grad + self.grad_AF_2D_pen(z)
            return grad
        else:
            forw_list = forw  # here 'forw' is a list in poly mode
            diff = 1 - self.sqb / sqrt_meas
            grad = self.poly.adjoint_object(forw_list, diff)
            grad = grad + self.grad_AF_2D_pen(z)
            return grad

    def fast_grad_AF_2D_pty_win(self, w, forw, sqrt_meas, idx=0):
        if self.poly is None:
            grad = self.fast_grad_L2_win_2D_pty(forw, sqrt_meas)
            grad = grad + self.grad_AF_2D_pen(w)
            return grad
        else:
            forw_list = forw
            diff = 1 - self.sqb / sqrt_meas
            grad = self.poly.adjoint_window(idx, forw_list, diff)
            grad = grad + self.grad_AF_2D_pen(w)
            return grad


    # ---------- LR helpers (Torch) ----------
    def learn_rate_penalties(self):
        lr = self.alpha_T + self.alpha_R
        if self.par.circular:
            lr += self.alpha_ST * 8
        else:
            lr += self.alpha_ST * (8 + 2 * (torch.sqrt(torch.tensor(self.par.object_shape[0], dtype=torch.float32)) +
                                            torch.sqrt(torch.tensor(self.par.object_shape[1], dtype=torch.float32))))
        if self.alpha_TV != 0 and not self.proximal_TV:
            root = torch.sqrt(torch.tensor(self.TV_param if self.TV_param > 0 else 1.0, dtype=torch.float32))
            if self.par.circular:
                lr += self.alpha_TV * 0.5 * 8 / root
            else:
                lr += self.alpha_TV * 0.5 * (8 + 2 * (torch.sqrt(torch.tensor(self.par.object_shape[0], dtype=torch.float32)) +
                                                      torch.sqrt(torch.tensor(self.par.object_shape[1], dtype=torch.float32)))) / root
        return float(lr)

    def learn_rate_penalties_win(self):
        lr = self.alpha_T + self.alpha_R
        if self.par.circular:
            lr += self.alpha_ST * 8
        else:
            lr += self.alpha_ST * (8 + 2 * torch.sqrt(torch.tensor(self.delta1, dtype=torch.float32))
                                     + 2 * torch.sqrt(torch.tensor(self.delta2, dtype=torch.float32)))
        if self.alpha_TV != 0 and not self.proximal_TV:
            root = torch.sqrt(torch.tensor(self.TV_param if self.TV_param > 0 else 1.0, dtype=torch.float32))
            if self.par.circular:
                lr += self.alpha_TV * 0.5 * 8 / root
            else:
                lr += self.alpha_TV * 0.5 * (8 + 2 * torch.sqrt(torch.tensor(self.delta1, dtype=torch.float32))
                                               + 2 * torch.sqrt(torch.tensor(self.delta2, dtype=torch.float32))) / root
        return float(lr)

    def learn_rate_2D_pty_optimal(self):
        abs_win = torch.abs(self.par.window) ** 2
        temp = torch.zeros(self.par.object_shape, device=device_default)
        for r in range(self.R):
            loc = self.par.locations[r, :]
            win_t = torch.zeros(self.par.object_shape, device=device_default)
            win_t[:self.delta1, :self.delta2] = abs_win
            win_t = torch.abs(self.par.shift_vec(win_t, loc))
            temp = temp + win_t
        max_win = torch.max(temp)
        learn_rate = max_win * (self.par.fourier_dimension[0] * self.par.fourier_dimension[1])
        learn_rate = learn_rate + self.learn_rate_penalties()
        learn_rate = self.learning_rate * 2 * (1 - self.AG_params.control) / float(learn_rate)
        return float(learn_rate)

    def learn_rate_2D_pty_sub_optimal(self):
        max_win = self.R * torch.max(torch.abs(self.par.window) ** 2)
        learn_rate = max_win * (self.par.fourier_dimension[0] * self.par.fourier_dimension[1])
        learn_rate = learn_rate + self.learn_rate_penalties()
        learn_rate = self.learning_rate * 2 * (1 - self.AG_params.control) / float(learn_rate)
        return float(learn_rate)

    def learn_rate_2D_pty_win_optimal(self):
        z = self.par.obj
        temp = torch.zeros(self.par.window_shape, device=device_default)
        for r in range(self.R):
            loc = self.par.locations[r, :]
            z_r = self.par.shift_vec(z, -loc)
            temp = temp + (torch.abs(z_r[:self.delta1, :self.delta2]) ** 2)
        max_win = torch.max(temp)
        learn_rate = max_win * (self.par.fourier_dimension[0] * self.par.fourier_dimension[1])
        learn_rate = learn_rate + self.learn_rate_penalties_win()
        learn_rate = self.learning_rate * 2 * (1 - self.AG_params.control) / float(learn_rate)
        return float(learn_rate)

    def learn_rate_2D_pty_win_sub_optimal(self):
        z = self.par.obj
        max_win = self.R * torch.max(torch.abs(z))
        learn_rate = max_win * (self.par.fourier_dimension[0] * self.par.fourier_dimension[1])
        learn_rate = learn_rate + self.learn_rate_penalties_win()
        learn_rate = self.learning_rate * 2 * (1 - self.AG_params.control) / float(learn_rate)
        return float(learn_rate)

    # ---------- objectives (Torch) ----------

    def objective_L2_2D(self, z: torch.Tensor):
        if self.poly is None:
            fz = self.par.forward_2D_pty(z)
            sqbx = torch.sqrt(torch.abs(fz) ** 2 + self.epsilon)
            objective = torch.sum(torch.abs(sqbx - self.sqb) ** 2)
            return objective, fz, sqbx
        else:
            forw_list = self.poly.forward_2D_pty_poly(z)
            I = self.poly.intensity_sum(forw_list)
            sqbx = torch.sqrt(I + self.epsilon)
            objective = torch.sum(torch.abs(sqbx - self.sqb) ** 2)
            return objective, forw_list, sqbx

    def objective_smoothness_Tikhonov(self, z: torch.Tensor):
        if self.par.circular:
            obj = torch.linalg.norm(z - torch.roll(z, 1, dims=0), 'fro') ** 2
            obj += torch.linalg.norm(z - torch.roll(z, -1, dims=0), 'fro') ** 2
            obj += torch.linalg.norm(z - torch.roll(z, 1, dims=1), 'fro') ** 2
            obj += torch.linalg.norm(z - torch.roll(z, -1, dims=1), 'fro') ** 2
        else:
            obj = torch.linalg.norm(z[:-1, :] - z[1:, :], 'fro') ** 2
            obj += torch.linalg.norm(z[1:, :] - z[:-1, :], 'fro') ** 2
            obj += torch.linalg.norm(z[:, :-1] - z[:, 1:], 'fro') ** 2
            obj += torch.linalg.norm(z[:, 1:] - z[:, :-1], 'fro') ** 2
        return obj

    def objective_TV_smoothed(self, z: torch.Tensor):
        if self.par.circular:
            obj = torch.sum(torch.sqrt(torch.abs(z - torch.roll(z, 1, dims=0)) ** 2 + self.TV_param))
            obj += torch.sum(torch.sqrt(torch.abs(z - torch.roll(z, -1, dims=0)) ** 2 + self.TV_param))
            obj += torch.sum(torch.sqrt(torch.abs(z - torch.roll(z, 1, dims=1)) ** 2 + self.TV_param))
            obj += torch.sum(torch.sqrt(torch.abs(z - torch.roll(z, -1, dims=1)) ** 2 + self.TV_param))
        else:
            obj = torch.sum(torch.sqrt(torch.abs(z[:-1, :] - z[1:, :]) ** 2 + self.TV_param))
            obj += torch.sum(torch.sqrt(torch.abs(z[1:, :] - z[:-1, :]) ** 2 + self.TV_param))
            obj += torch.sum(torch.sqrt(torch.abs(z[:, :-1] - z[:, 1:]) ** 2 + self.TV_param))
            obj += torch.sum(torch.sqrt(torch.abs(z[:, 1:] - z[:, :-1]) ** 2 + self.TV_param))
        return obj

    def objective_pen_2D(self, z: torch.Tensor):
        objective = torch.tensor(0.0, device=device_default)
        if self.alpha_T != 0:
            objective = objective + self.alpha_T * torch.sum(torch.abs(z) ** 2)
        if self.alpha_R != 0:
            objective = objective + self.alpha_R * torch.sum((z.imag) ** 2)
        if self.alpha_ST != 0:
            objective = objective + self.objective_smoothness_Tikhonov(z)
        if self.alpha_TV != 0:
            objective = objective + self.objective_TV_smoothed(z)
        return objective

    def obj_pty(self, z: torch.Tensor, obj_offset: float):
        l2_z, f, sqbx = self.objective_L2_2D(z)
        obj = l2_z + self.objective_pen_2D(z) + obj_offset
        return obj, l2_z, f, sqbx

    def obj_pty_win(self, w: torch.Tensor, obj_offset: float, idx: int = 0):
        if self.poly is None:
            self.par.set_window(w)
            l2_z, f, sqbx = self.objective_L2_2D(self.par.obj)
            obj = l2_z + self.objective_pen_2D(w) + obj_offset
            return obj, l2_z, f, sqbx
        else:
            # update ONLY the idx-th channel window inside the poly model
            if hasattr(self.poly, "set_window"):
                self.poly.set_window(idx, w)
            elif hasattr(self.poly, "windows"):
                self.poly.windows[idx] = w
            elif hasattr(self.poly, "pars"):  # fallback: each channel keeps its own ptycho clone
                self.poly.pars[idx].set_window(w)
            # objective uses poly forward internally
            l2_z, f, sqbx = self.objective_L2_2D(self.par.obj)
            obj = l2_z + self.objective_pen_2D(w) + obj_offset
            return obj, l2_z, f, sqbx

    # ---------- runners (Torch) ----------
    def run(self, z_0: torch.Tensor, obj_shift: float = 0.0):
        grad_f = lambda x, forw, sqrt_meas: self.fast_grad_AF_2D_pty(x, forw, sqrt_meas)
        obj_f = lambda x: self.obj_pty(x, obj_shift)

        if self.learning_rate_type == 'optimal':
            lr = self.learn_rate_2D_pty_optimal()
        elif self.learning_rate_type == 'suboptimal':
            lr = self.learn_rate_2D_pty_sub_optimal()
        else:
            lr = self.learning_rate

        gd = _GradDesc(
            grad_f, obj_f, z_0, self.number_of_iterations, self.grad_threshold, lr,
            self.AG_params, self.learn_rate_decay, self.proximal_TV, self.alpha_TV,
            self.verbose, self.track, self.obj_tr, self.track_it, self.par.mask if hasattr(self.par, "mask") else None
        )
        obj, val = gd.run()

        # take tracking buffers back (Torch tensors)
        self.measurement_error = gd.measurement_error
        self.objective = gd.objective
        self.object_error = gd.object_error
        self.tracked_objects = gd.tracked_objects

        return obj, val

    def run_win(self, w_0: torch.Tensor, obj_shift: float = 0.0, idx: int = 0):
        
        if self.number_of_iterations <= 0:
            obj_val, _, _, _ = self.obj_pty_win(w_0, obj_shift, idx)
            return w_0, obj_val
        grad_f = lambda w, forw, sqrt_meas: self.fast_grad_AF_2D_pty_win(w, forw, sqrt_meas, idx)
        obj_f  = lambda w: self.obj_pty_win(w, obj_shift, idx)

        if self.learning_rate_type == 'optimal':
            lr = self.learn_rate_2D_pty_win_optimal()
        elif self.learning_rate_type == 'suboptimal':
            lr = self.learn_rate_2D_pty_win_sub_optimal()
        else:
            lr = self.learning_rate

        gd = _GradDesc(
            grad_f, obj_f, w_0, self.number_of_iterations, self.grad_threshold, lr,
            self.AG_params, self.learn_rate_decay, self.proximal_TV, self.alpha_TV,
            self.verbose, self.track, None, None
        )
        win, val = gd.run()

        self.measurement_error = gd.measurement_error
        self.objective = gd.objective
        self.object_error = gd.object_error
        self.tracked_objects = gd.tracked_objects

        return win, val

    # ---------- shift refinement (Torch port) ----------
    #XXX: helper to cache frequency vectors for given sizes
    def _freq_vecs(self, H: int, W: int, device: torch.device):
        key = (H, W, device)
        if key in self._tile_freq_cache:
            return self._tile_freq_cache[key]
        ky = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
        kx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
        self._tile_freq_cache[key] = (ky, kx)
        return ky, kx

    #XXX: fast fractional crop on the WINDOW tile (avoids full-image FFT)
    def _fractional_crop_tile(self, z: torch.Tensor, loc: torch.Tensor, d1: int, d2: int, circular: bool):
        H, W = z.shape
        base0 = torch.floor(loc[0]).to(torch.int64)
        base1 = torch.floor(loc[1]).to(torch.int64)
        dy = (loc[0] - base0.to(loc.dtype)).to(torch.float32)
        dx = (loc[1] - base1.to(loc.dtype)).to(torch.float32)

        if circular:
            rows = (torch.arange(d1, device=z.device) + base0) % H
            cols = (torch.arange(d2, device=z.device) + base1) % W
            tile = z.index_select(0, rows).index_select(1, cols)
        else:
            # clamp for non-circular (keeps previous implicit-wrap behavior from full-image DFT)
            i0 = torch.clamp(base0, 0, H - d1)
            i1 = torch.clamp(base1, 0, W - d2)
            tile = z[i0:(i0 + d1), i1:(i1 + d2)]

        # subpixel shift on tile via Fourier phase ramps
        ky, kx = self._freq_vecs(d1, d2, z.device)
        T = torch.fft.fft2(tile)
        T = T * torch.exp(-2j * torch.pi * dy * ky / d1) * torch.exp(-2j * torch.pi * dx * kx / d2)
        tile_shifted = torch.fft.ifft2(T)
        return tile_shifted

    def obj_shift(self, loc: torch.Tensor, z: torch.Tensor, sqb_loc: torch.Tensor):
        #XXX: use tile-based fractional crop instead of full-image shift
        z_r = self._fractional_crop_tile(
        z, loc, self.delta1, self.delta2, circular=True if self.par.circular else False   # <-- remove "or self.par.float_shift"
        )#XXX
        fz = self.par.forward_2D_os(z_r)  # unchanged physics
        sqbx = torch.sqrt(torch.abs(fz) ** 2 + self.epsilon)
        obj = torch.sum(torch.abs(sqbx - sqb_loc) ** 2)
        return obj, obj, fz, sqbx

    def grad_shift(self, loc: torch.Tensor, forw: torch.Tensor, sqrt_meas: torch.Tensor,
                   sqb_loc: torch.Tensor, z: torch.Tensor, fft_z_c: torch.Tensor,
                   ky_full: torch.Tensor, kx_full: torch.Tensor):  #XXX added cached ky/kx
        # simple gradient w.r.t. loc using Fourier shift theorem derivatives
        grad = torch.zeros_like(z)
        grad[:self.delta1, :self.delta2] = self.fast_grad_AF_2D_os(forw, sqrt_meas, sqb_loc)
        grad = torch.fft.fft2(grad) / (z.shape[0] * z.shape[1])
        grad = grad * fft_z_c

        H, W = z.shape
        #XXX: use cached frequency grids (avoid re-alloc each call)
        gr_1 = (grad * (-2j * torch.pi * ky_full / H)).real.sum()  # d/dy
        gr_2 = (grad * (-2j * torch.pi * kx_full / W)).real.sum()  # d/dx
        return torch.stack([gr_1, gr_2])

    def run_shifts(self, z: torch.Tensor):
        #XXX removed per-position prints; keep function silent and fast
        norm_z = torch.linalg.norm(z)
        z = z / norm_z.clamp_min(1e-20)

        fft_z_c = torch.fft.fft2(z).conj()
        A_norm_sq = torch.max(torch.abs(fft_z_c)) ** 2 * torch.max(torch.abs(self.par.window)) ** 2
        A_norm_sq *= (self.par.fourier_dimension[0] * self.par.fourier_dimension[1]) / (z.shape[0] * z.shape[1])

        #XXX: cache full-image ky/kx once
        H, W = z.shape  #XXX
        ky_full, kx_full = self._freq_vecs(H, W, z.device)  #XXX

        for r in range(int(self.par.R)):
            loc = self.par.locations[r, :].clone()
            sqb_loc = self.sqb[:, :, r]
            sqb_loc = torch.sqrt((sqb_loc ** 2 - self.epsilon).clamp_min(0.0) / (norm_z ** 2) + self.epsilon)

            if self.learning_rate_type == 'optimal':
                lr = (z.shape[0] * z.shape[1]) * 32 * (torch.pi ** 2) * A_norm_sq
                lr = lr + 16 * (torch.pi ** 2) * torch.sqrt((z.shape[0] * z.shape[1]) * A_norm_sq) * torch.linalg.norm(sqb_loc)
                lr = 1.0 / float(lr)
            else:
                lr = float(self.learning_rate)

            #XXX: pass cached ky/kx to grad for speed
            grad_f = lambda x, forw, sqrt_meas: self.grad_shift(x, forw, sqrt_meas, sqb_loc, z, fft_z_c, ky_full, kx_full)  #XXX
            obj_f  = lambda x: self.obj_shift(x, z, sqb_loc)  #XXX uses tile-based crop

            gd = _GradDesc(
                grad_f, obj_f, loc, self.number_of_iterations, 0.0, lr,
                self.AG_params, 0.0, False, 0.0, 0, False, None, None, self.par.mask if hasattr(self.par, "mask") else None
            )
            loc_new, _ = gd.run()

            # wrap / clamp into image range
            loc_new = loc_new.detach().clone()

            #XXX: If scan is circular or uses float subpixel shifts, wrap modulo (old behavior).
            #XXX: Otherwise (non-circular integer scan), CLAMP so that a full window fits.
            H, W = z.shape  #XXX
            # in run_shifts(), after computing loc_new
            if self.par.circular:
                loc_new[0] = loc_new[0] % H
                loc_new[1] = loc_new[1] % W
            else:
                if not hasattr(self, "_base_locations"):
                    self._base_locations = self.par.locations.detach().clone()
                y_min = float(self._base_locations[:,0].min()); y_max = float(self._base_locations[:,0].max())
                x_min = float(self._base_locations[:,1].min()); x_max = float(self._base_locations[:,1].max())
                loc_new[0] = torch.clamp(loc_new[0], y_min, y_max)
                loc_new[1] = torch.clamp(loc_new[1], x_min, x_max)

            # after the for-loop, kill any constant offset:
            if not hasattr(self, "_base_locations"):
                self._base_locations = self.par.locations.detach().clone()
            delta = self.par.locations.mean(0) - self._base_locations.mean(0)
            self.par.locations -= delta


            #XXX: removed per-position printing of old/new locations
            self.par.locations[r, :] = loc_new

        if not hasattr(self, "_base_locations"):
            self._base_locations = self.par.locations.detach().clone()
            delta = self.par.locations.mean(0) - self._base_locations.mean(0)
            self.par.locations -= delta
        return self.par.locations


class blind_amplitude_flow:
    """
    Two-variable blind amplitude flow (optimize object and window alternately).
    Torch port mirroring your NumPy API and behavior.
    """
    def __init__(self, **kwargs):
        # epsilon
        if 'epsilon' in kwargs:
            self.epsilon = float(kwargs['epsilon']) if isinstance(kwargs['epsilon'], (float, int)) else 0.0
        else:
            print('epsilon is not given. Set to 0.')
            self.epsilon = 0.0

        # sqrt of measurements
        assert ('measurements' in kwargs) or ('sqb' in kwargs), "Neither measurements nor their square root is given"
        if 'measurements' in kwargs:
            meas = torch.as_tensor(kwargs['measurements'], device=device_default)
            self.sqb = torch.sqrt(torch.clamp(meas, min=0.0) + self.epsilon)
        else:
            self.sqb = torch.as_tensor(kwargs['sqb'], device=device_default)

        # ptycho
        assert 'ptycho' in kwargs, "Forward model of ptycho class is not given."
        assert isinstance(kwargs['ptycho'], _forward.ptycho), "ptycho is not an instance of class ptycho."
        self.par = kwargs['ptycho'].copy()

        # AG params
        self.AGP_object = copy.deepcopy(kwargs.get('AGP_object', types.SimpleNamespace(enable_AG=False, control=0.5, tau=0.3, AG_iterations=2)))
        self.AGP_window = copy.deepcopy(kwargs.get('AGP_window', types.SimpleNamespace(enable_AG=False, control=0.5, tau=0.3, AG_iterations=2)))
        self.AGP_shift  = copy.deepcopy(kwargs.get('AGP_shift',  types.SimpleNamespace(enable_AG=True,  control=0.5, tau=0.1, AG_iterations=20)))

        # iterations / subiters
        self.number_of_iterations = int(kwargs['number_of_iterations'])
        self.object_subiterations = int(kwargs['object_subiterations'])
        self.window_subiterations = int(kwargs['window_subiterations'])
        self.shift_subiterations  = int(kwargs.get('shift_subiterations', 100))
        #polychromatic
        self.poly = kwargs.get('poly', None)

        # LRs
        self.learning_rate = float(kwargs.get('learning_rate', 1.0))
        self.learning_rate_type = kwargs.get('learning_rate_type', 'optimal')
        self.shift_learning_rate = float(kwargs.get('shift_learning_rate', 1.0))
        self.shift_learning_rate_type = kwargs.get('shift_learning_rate_type', 'optimal')
        self.learn_rate_decay = float(kwargs.get('learn_rate_decay', 0.0))

        # thresholds
        self.grad_thr_object = float(kwargs.get('grad_thr_object', 0.0))
        self.grad_thr_window = float(kwargs.get('grad_thr_window', 0.0))
        self.grad_thr_shift  = float(kwargs.get('grad_thr_shift', 0.0))

        # penalties (object)
        self.alpha_T  = float(kwargs.get('alpha_T', 0.0))
        self.alpha_R  = float(kwargs.get('alpha_R', 0.0))
        self.alpha_ST = float(kwargs.get('alpha_ST', 0.0))
        self.alpha_TV = float(kwargs.get('alpha_TV', 0.0))
        self.TV_param_obj = float(kwargs.get('TV_param_obj', 0.0))
        self.proximal_TV_obj = bool(kwargs.get('proximal_TV_obj', False))
        self.alpha_TV_phase = float(kwargs.get('alpha_TV_phase', 0.0))  # NEW: circular phase TV weight

        # penalties (window)
        self.beta_T  = float(kwargs.get('beta_T', 0.0))
        self.beta_R  = float(kwargs.get('beta_R', 0.0))
        self.beta_ST = float(kwargs.get('beta_ST', 0.0))
        self.beta_TV = float(kwargs.get('beta_TV', 0.0))
        self.TV_param_win = float(kwargs.get('TV_param_win', 0.0))
        self.proximal_TV_win = bool(kwargs.get('proximal_TV_win', False))

        # schedule / options
        self.skip_win_it = int(kwargs.get('skip_win_it', 0))
        self.update_shifts = bool(kwargs.get('update_shifts', False))
        self.update_shifts_period = int(kwargs.get('update_shifts_period', 10))
        self.skip_shift_it = int(kwargs.get('skip_shift_it', 0))
        self.reweight = bool(kwargs.get('reweight', False))
        
        # io
        self.verbose = int(kwargs.get('verbose', 0))
        self.track = bool(kwargs.get('track', False))
        self.obj_tr = kwargs.get('obj_tr', None)
        self.win_tr = kwargs.get('win_tr', None)
        if isinstance(self.obj_tr, torch.Tensor): self.obj_tr = self.obj_tr.to(device_default)
        if isinstance(self.win_tr, torch.Tensor): self.win_tr = self.win_tr.to(device_default)
        self.track_it = torch.as_tensor(kwargs.get('track_it', []), device=device_default) if self.track else None

        # tracking buffers
        self.measurement_error = torch.zeros(0, device=device_default)
        self.objective = torch.zeros(0, device=device_default)
        self.object_error = torch.zeros(0, device=device_default)
        self.window_error = torch.zeros(0, device=device_default)

    def prepare_algorithms(self):
        if getattr(self, "poly", None) is not None:
            af_o = amplitude_flow(
                sqb=self.sqb, poly=self.poly, AG_params=self.AGP_object,
                number_of_iterations=self.object_subiterations,
                grad_threshold=self.grad_thr_object,
                learning_rate_type=self.learning_rate_type,
                learning_rate=self.learning_rate,
                epsilon=self.epsilon,
                alpha_T=self.alpha_T, alpha_R=self.alpha_R, alpha_ST=self.alpha_ST,
                alpha_TV=self.alpha_TV, TV_param=self.TV_param_obj,
                proximal_TV=self.proximal_TV_obj,
                verbose=self.verbose, track=self.track, obj_tr=self.obj_tr
            )
            af_ws = []
            for l in range(self.poly.L):
                af_w = amplitude_flow(
                    sqb=self.sqb, poly=self.poly, AG_params=self.AGP_window,
                    number_of_iterations=self.window_subiterations,
                    grad_threshold=self.grad_thr_window,
                    learning_rate_type=self.learning_rate_type,
                    learning_rate=self.learning_rate,
                    epsilon=self.epsilon,
                    alpha_T=self.beta_T, alpha_R=self.beta_R, alpha_ST=self.beta_ST,
                    alpha_TV=self.beta_TV, TV_param=self.TV_param_win,
                    proximal_TV=self.proximal_TV_win,
                    verbose=self.verbose, track=self.track, obj_tr=self.win_tr
                )
                af_ws.append(af_w)
            return af_o, af_ws
        else:
            # MONO: single object and single window optimizer
            af_o = amplitude_flow(
                sqb=self.sqb, ptycho=self.par.copy(), AG_params=self.AGP_object,
                number_of_iterations=self.object_subiterations,
                grad_threshold=self.grad_thr_object,
                learning_rate_type=self.learning_rate_type,
                learning_rate=self.learning_rate,
                epsilon=self.epsilon,
                alpha_T=self.alpha_T, alpha_R=self.alpha_R, alpha_ST=self.alpha_ST,
                alpha_TV=self.alpha_TV, TV_param=self.TV_param_obj,
                proximal_TV=self.proximal_TV_obj,
                verbose=self.verbose, track=self.track, obj_tr=self.obj_tr
            )
            af_w = amplitude_flow(
                sqb=self.sqb, ptycho=self.par.copy(), AG_params=self.AGP_window,
                number_of_iterations=self.window_subiterations,
                grad_threshold=self.grad_thr_window,
                learning_rate_type=self.learning_rate_type,
                learning_rate=self.learning_rate,
                epsilon=self.epsilon,
                alpha_T=self.beta_T, alpha_R=self.beta_R, alpha_ST=self.beta_ST,
                alpha_TV=self.beta_TV, TV_param=self.TV_param_win,
                proximal_TV=self.proximal_TV_win,
                verbose=self.verbose, track=self.track, obj_tr=self.win_tr
            )
            return af_o, af_w

    def run(self, z_0: torch.Tensor, w_0: torch.Tensor):
        z = z_0.to(device_default)
        w = w_0.to(device_default)

        if self.reweight:
            z, w = util.normalize_object_and_window(z, w)

        af_o, af_w = self.prepare_algorithms()
        count = 0

        if self.track and isinstance(self.track_it, torch.Tensor) and self.track_it.numel() > 0:
            self.tracked_objects = torch.zeros((0, z.shape[0], z.shape[1]), dtype=_COMPLEX, device=device_default)
            self.tracked_windows = torch.zeros((0, w.shape[0], w.shape[1]), dtype=_COMPLEX, device=device_default)

        is_poly = getattr(self, "poly", None) is not None

        for lp in range(self.number_of_iterations):
            offset = 0 if lp == 0 else 1

            if self.reweight:
                z, w = util.normalize_object_and_window(z, w)

            if self.verbose:
                print('Iteration: ', lp)

            # ---- OBJECT UPDATE ----
            if is_poly:
                # compute penalty from all λ windows currently in poly
                penalty_w = torch.tensor(0.0, device=device_default)
                # best-effort: pull windows out (support a few common layouts)
                if hasattr(self.poly, "windows"):
                    win_list = self.poly.windows
                elif hasattr(self.poly, "pars"):
                    win_list = [p.window for p in self.poly.pars]
                else:
                    win_list = []

                for wl in win_list:
                    penalty_w = penalty_w + af_w[0].objective_pen_2D(wl)  # use same penalty fn

                # af_o forward path uses poly internally; its ptycho window is irrelevant
                z, obj = af_o.run(z, penalty_w)
            else:
                penalty_w = af_w.objective_pen_2D(w)
                af_o.par.set_window(w)
                z, obj = af_o.run(z, penalty_w)
            # --- Gauge fix: set background phase ~ 0 to keep away from ±π branch ---
            with torch.no_grad():
                if hasattr(af_o.par, "mask") and af_o.par.mask is not None and (af_o.par.mask == 0).any():
                    bg = (af_o.par.mask == 0)
                    shift = torch.median(torch.angle(z[bg]))
                else:
                    shift = torch.median(torch.angle(z))
                z *= torch.exp(-1j * shift)

            if self.track:
                if is_poly:
                    self.measurement_error = torch.cat([self.measurement_error, af_o.measurement_error[offset:]])
                    self.objective         = torch.cat([self.objective,         af_o.objective[offset:]])
                    self.object_error      = torch.cat([self.object_error,      af_o.object_error[offset:]])
                else:
                    self.measurement_error = torch.cat([self.measurement_error, af_o.measurement_error[offset:]])
                    self.objective         = torch.cat([self.objective,         af_o.objective[offset:]])
                    self.object_error      = torch.cat([self.object_error,      af_o.object_error[offset:]])

            # ---- SHIFT UPDATE (optional) ----
            if self.update_shifts and lp >= self.skip_shift_it and (lp % self.update_shifts_period == 0):
                af_s = amplitude_flow(
                    sqb=self.sqb, ptycho=af_o.par.copy(), AG_params=self.AGP_shift,
                    number_of_iterations=self.shift_subiterations,
                    grad_threshold=self.grad_thr_shift,
                    learning_rate_type=self.shift_learning_rate_type,
                    learning_rate=self.shift_learning_rate,
                    learn_rate_decay=0.0, epsilon=self.epsilon,
                    verbose=1, track=False
                )
                self.par.locations = af_s.run_shifts(z)
                af_o.par.locations = self.par.locations
                if is_poly:
                    for aw in af_w: aw.par.locations = self.par.locations
                else:
                    af_w.par.locations = self.par.locations

            # ---- WINDOW UPDATE ----
            if self.reweight and not is_poly:
                w, z = util.normalize_object_and_window(w, z)

            if lp >= self.skip_win_it:
                penalty_z = af_o.objective_pen_2D(z)

                if is_poly:
                    # update each λ window independently
                    if hasattr(self.poly, "windows"):
                        win_list = self.poly.windows
                    elif hasattr(self.poly, "pars"):
                        win_list = [p.window for p in self.poly.pars]
                    else:
                        win_list = []

              
                    for l, aw in enumerate(af_w):
                        w_l0 = win_list[l]

                        aw.par.set_object(z)  # ensure LR sees the current object for this λ

                        w_l, _ = aw.run_win(w_l0, penalty_z, idx=l)
                        # write back the updated window
                        if hasattr(self.poly, "set_window"):
                            self.poly.set_window(l, w_l)
                        elif hasattr(self.poly, "windows"):
                            self.poly.windows[l] = w_l
                        elif hasattr(self.poly, "pars"):
                            self.poly.pars[l].set_window(w_l)
                    # tracking buffers for windows (aggregate length-matched zeros if needed)
                    if self.track:
                        # concatenate measurement/objective the same way as mono (using object AF cadence)
                        pass  # optional: collect per-window traces if you want
                else:
                    af_w.par.set_object(z)
                    w, obj = af_w.run_win(w, penalty_z)
                    if self.track:
                        self.measurement_error = torch.cat([self.measurement_error, af_w.measurement_error[1:]])
                        self.objective         = torch.cat([self.objective,         af_w.objective[1:]])
                        self.window_error      = torch.cat([self.window_error,      af_w.object_error[1:]])
            else:
                if self.track:
                    if self.measurement_error.numel() > 0:
                        last_me = self.measurement_error[-1]
                        last_obj = self.objective[-1]
                    else:
                        last_me = torch.tensor(0.0, device=device_default)
                        last_obj = torch.tensor(0.0, device=device_default)
                    # fill the skipped window-iters so traces remain aligned
                    if is_poly:
                        fill_len = af_w[0].number_of_iterations
                    else:
                        fill_len = af_w.number_of_iterations
                    self.measurement_error = torch.cat([self.measurement_error, last_me.repeat(fill_len)])
                    self.objective         = torch.cat([self.objective,         last_obj.repeat(fill_len)])
                    self.window_error      = torch.cat([self.window_error,      torch.zeros(fill_len, device=device_default)])

            # decay LRs (object + window optimizers)
            if self.learn_rate_decay != 0.0:
                decay = ((lp + 1) / (lp + 2)) ** self.learn_rate_decay
                af_o.learning_rate *= float(decay)
                if is_poly:
                    for aw in af_w: aw.learning_rate *= float(decay)
                else:
                    af_w.learning_rate *= float(decay)

            # snapshot tracking (unchanged; optional to extend for per-λ windows)
            if self.track and isinstance(self.track_it, torch.Tensor) and count < self.track_it.numel():
                if lp == int(self.track_it[count].item()):
                    if isinstance(self.obj_tr, torch.Tensor):
                        self.tracked_objects = torch.cat([
                            self.tracked_objects,
                            util.align_objects(self.obj_tr, z, self.par.mask if hasattr(self.par, "mask") else torch.ones_like(z)).unsqueeze(0)
                        ], dim=0)
                    if isinstance(self.win_tr, torch.Tensor) and not is_poly:
                        self.tracked_windows = torch.cat([
                            self.tracked_windows,
                            util.align_objects(self.win_tr, w, torch.ones_like(w)).unsqueeze(0)
                        ], dim=0)
                    count += 1

        if self.reweight and not is_poly:
            z, w = util.normalize_object_and_window(z, w)

        return z, w, obj
