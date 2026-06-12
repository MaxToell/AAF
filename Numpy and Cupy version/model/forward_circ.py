# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 13:02:41 2020

@author: oleh.melnyk
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 17:12:54 2020

@author: oleh.melnyk
"""

import numpy as np
import numpy.random
import math

from recordtype import recordtype

def pty_params():
    pty_params = recordtype("pty_params", 
                            "window, locations, detector_shape, object_shape, window_shape, fourier_dimension, mask")

    return pty_params


def loc_mesh_grid(d, delta, shift):
    locations_1d = np.array(range(0,d, shift))
    locations_2d = np.zeros((len(locations_1d)**2,2),dtype=int)
    locations_2d[:,0] = np.repeat(locations_1d,len(locations_1d))
    locations_2d[:,1] = np.tile(locations_1d,len(locations_1d))
    
    mask = np.zeros((d,d))
    mask_t = np.zeros((d,d))
    mask_t[:delta,:delta] = 1
    for loc in locations_2d:
        mask_r = np.roll(mask_t,loc[0],axis = 0)
        mask_r = np.roll(mask_r,loc[1],axis = 1)
        mask += mask_r
    
    return locations_2d,mask

def loc_Fermat_spiral(d, delta, c):
    
    locations = []
    
    N = np.ceil(2*(0.5*(d - delta)/ c)**2).astype(np.int32)  
    phi_0 = 8 * math.pi / (1 + math.sqrt(5))**2 
    for n in range(N):
        r = c * math.sqrt(n)
        phi = n * phi_0
        x = np.round(r * math.cos(phi) + 0.5*(d - delta)).astype(np.int32)  
        y = np.round(r * math.sin(phi) + 0.5*(d - delta)).astype(np.int32)
        
        if (x < 0 or x >= d-delta or y < 0 or y >= d-delta):
            continue
        
        locations.append([x, y])
    
    locations = np.array(locations)
    mask = np.zeros((d,d))
    for loc in locations:
        mask[loc[0]:(loc[0] +delta),loc[1]:(loc[1] +delta)]=1
    
    return locations,mask

def forward_2D_os(z, par):
    
    out_padded = np.zeros(par.fourier_dimension,dtype=complex)
    out_padded[:par.window_shape[0],:par.window_shape[1]] = par.window * z[:par.window_shape[0],:par.window_shape[1]]
    
    out_fft = np.fft.fft2(out_padded)
    
    h1 = par.fourier_dimension[0] //2
    h2 = par.fourier_dimension[1] //2
    
    out_fft = np.roll(out_fft, h1, axis=0)
    out_fft = np.roll(out_fft, h2, axis=1)
    
    return out_fft


def forward_to_meas_2D(forw):
    return np.abs(forw)**2


def forward_to_meas_2D_pty(forw):
    return forward_to_meas_2D(forw)
    
def forward_2D_pty(z,par):
    # window is vector of length delta
    # obj is a matrix of size d x d
    R = par.locations.shape[0]
    
    forw = np.zeros((par.detector_shape[0],par.detector_shape[1],R),dtype = complex)
    
    for r in range(R):
        loc = par.locations[r,:] 
        z_r = np.roll(z,-loc[0],0)
        z_r = np.roll(z_r,-loc[1],1)
        forw[:,:,r] = forward_2D_os(z_r,par)
    
    return forw

def forward_adj_2D_os(forw,par):  
    h1 = par.fourier_dimension[0] //2
    h2 = par.fourier_dimension[1] //2
    
    unzoomed_padded = np.roll(forw, -h2, axis=1)
    unzoomed_padded = np.roll(unzoomed_padded, -h1, axis=0)
    out_fft = np.fft.ifft2(unzoomed_padded)*np.prod(par.fourier_dimension)    
    
    # result = np.zeros((par.object_shape[0],par.object_shape[1]),dtype = complex)
    result = par.window.conj() * out_fft[:par.window_shape[0],:par.window_shape[1]]
        
    return result

# def forward_adj_2D_pty(forw,par):
#     # window is vector of length delta
#     # forw is a array of size delta1 X delta2 X R
#     R = par.locations.shape[0]
#     delta1 = par.window_shape[0]
#     delta2 = par.window_shape[1]
    
#     z = np.zeros((par.object_shape[0],par.object_shape[1]),dtype = complex)
    
#     for r in range(R):
#         loc = par.locations[r,:] 
        
#         z_r = forward_adj_2D_os(forw[:,:,r],par)
        
#         z_r = np.roll(z_r,+loc[0],0)
#         z_r = np.roll(z_r,+loc[1],1)
#         z +=z_r
        
#     return z