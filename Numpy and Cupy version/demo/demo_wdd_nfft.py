# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:58:30 2024

@author: oleh.melnyk
"""

import numpy as np
import sys
sys.path.insert(1, '../model')
sys.path.insert(1, '../algorithms')

# for testing

# import forward_circ as forward
import forward as forward
# import forward_2D as forward

import wigner_2D_nfft as wdd_nfft
import wigner_2D as wdd
import utility_2D as util
from scipy.ndimage import zoom


import cmath
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage import data
import matplotlib.pyplot as plt

import time


#from recordtype import recordtype

def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (satur_parser(im_hsv[:,:,0],im_hsv[:,:,1]) * 2 -1)* cmath.pi
    obj = modulus * np.exp(1.0j * phase)
    obj_mod = satur_parser(obj,im_hsv[:,:,1])
    
    return obj_mod

def show_object(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    
    modulus = np.repeat(np.round(np.abs(obj)).astype(np.float64)[:, :,np.newaxis], 3, axis=2)
    modulus *= 255.0 / np.max(modulus) 
    modulus = np.round(modulus).astype(np.uint64)
    
    phase = np.ones((obj.shape[0],obj.shape[1],3))
    phase[:,:,0] = (np.angle(obj) + cmath.pi)/(2*cmath.pi)
    phase_rgb = hsv2rgb(phase)
                
    ax[0].imshow(modulus)
    ax[0].axis('off')
    ax[1].imshow(phase_rgb)
    ax[1].axis('off')
        
    plt.show()
    
    
def show_measurements(b,locations,shift):
    R = locations.shape[0]
    R_1D = np.sqrt(R).astype(int)
#    delta = b.shape[0]
#    half = np.round(delta*0.5).astype(np.int)
    
    b_sc = np.log10(b)
#    b_sc = np.roll(b_sc, half, axis=0)
#    b_sc = np.roll(b_sc, half, axis=1)
    
    fig, ax = plt.subplots(nrows = R_1D, ncols = R_1D)
    for r in range(R):
        loc = locations[r,:]
        x = (loc[0]/shift).astype(int)
        y = (loc[1]/shift).astype(int)
        im = ax[x][y].imshow(b_sc[:,:,r],cmap='hot')
        ax[x][y].axis('off')
    
    fig.colorbar(im,ax=ax.ravel().tolist(), label='log_10(Intensity)')
    plt.show()

im_big = data.coffee()/255.0
#im = im[300:,:]
start = np.round(0.25*im_big.shape[0]).astype(int)
im_big = im_big[:,start:(start + im_big.shape[0]),:]

outd = int(im_big.shape[0])//4
factor = outd*1.0/im_big.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_big[:,:,0],factor)
im[:,:,1] = zoom(im_big[:,:,1],factor)
im[:,:,2] = zoom(im_big[:,:,2],factor)

# outd = 100
# im = im_big[:outd,:outd,:]
d = im.shape[0]

obj = image_to_object(im,lambda x,v: x)


d = 30
band = 4
obj = np.zeros((d,d),dtype = complex)
obj[:band,:band] = np.random.normal(size=(band,band)) + 1j*np.random.normal(size=(band,band))
# obj[:(2*band-1),:(2*band-1)] = np.random.normal(size=(2*band-1,2*band-1)) + 1j*np.random.normal(size=(2*band-1,2*band-1))
# obj = np.roll(np.roll(obj,-band + 1,0), -band + 1,1)       
# obj[:band,:band] = np.array([[1, 1j], [1, -1j]],dtype=  complex)
obj = np.fft.ifft2(obj)

obj /= np.max(np.abs(obj))

delta = 12
shift = 1
# delta SHOULD be at least 2*shift

f_dim = (d,d)
# dsize = (d,d)

# f_dim = (2*delta -1,2*delta -1)
dsize = f_dim


# locations_1d = np.array(range(0,d-delta+1, shift))
# locations_2d = np.zeros((len(locations_1d)**2,2),dtype=int)
# locations_2d[:,0] = np.repeat(locations_1d,len(locations_1d))
# locations_2d[:,1] = np.tile(locations_1d,len(locations_1d))
# locations_2d, mask = forward.loc_mesh_grid(d,delta,shift)
# locations_2d = forward.loc_Fermat_spiral(d, delta, 4.9)

show_object(255*obj)

cov_mat = np.eye(2,dtype = complex)/0.1
mu = np.array([0.5 + delta*0.5, 0.5 + delta*0.5])
gauss = lambda x: np.exp(-0.5* (( x - mu).conj().T).dot((cov_mat/delta**2).dot(x - mu)))
window = np.zeros((delta,delta), dtype = complex);
for ix in range(delta):
    for iy in range(delta):
        # window[ix,iy] = gauss( np.array([ix+1, iy+1]))
        window[ix,iy] = gauss( np.array([ix+1, iy+1])) #* np.exp(1j*np.pi*( (ix + 1 - mu[0])**2+(iy + 1 - mu[1])**2)/delta)

window = window *np.exp(2j*np.pi*np.random.rand(delta,delta))

show_object(255*window)
        
# window = np.array([[1, 1j], [1, -1j]],dtype=  complex)


# pty_params = forward.pty_params()

par = forward.ptycho(
            object_shape = obj.shape,
            window = window, 
            # circular = True,
            # loc_type = 'spiral',
            # fermat_seed_size = 0.5,#0.5,
            float_shift = True,
            loc_type = 'grid',
            shift = shift, 
            fourier_dimension = f_dim)


plt.scatter(par.locations[:,0],par.locations[:,1])
plt.show()

# par = pty_params(window = window, 
#                  locations = locations_2d,
#                  detector_shape = dsize,
#                  object_shape = obj.shape,
#                  window_shape = window.shape,
#                  fourier_dimension = f_dim,
#                  mask = mask)

print('Computing forward model')
f = par.forward_2D_pty(obj)

print('Computing measurements')
b = par.forward_to_meas_2D_pty(f)

# show_measurements(b,locations_2d,shift)

# obj_0 = 255*np.sqrt(0.5)*( np.random.normal(size=(d,d)) + 1.0j * np.random.normal(size=(d,d)))
# show_object(obj_0)

nop = 10**2
scaling = nop / np.linalg.norm(window,'fro')**2 / np.prod(dsize)
b_scaled =  scaling * b;
b_n = np.random.poisson(b_scaled, b_scaled.shape) / scaling

b_n = b

print('Noise level: ', util.relative_measurement_error(b,b_n))

# wdd_params = wdd.wdd_params_object()

# wigner = wdd.wdd(b_n,
#                  ptycho = par,
#                  reg_type = 'percent',
#                  reg_threshold = 0.2,#10**-1,#10**-10,
#                  mg_type = 'log',
#                  mg_diagonals_type = 'percent',
#                  mg_diagonals_param = 0.8,
#                  as_wtype = 'weighted',
#                  as_threshold = 10**-10,                                  
#                  add_dummy = False,
#                  subspace_completion = False,
#                  sbc_threshold = 0.1)

# print('Reconstructing...')

# start_time = time.time()
# obj_r = wigner.run()
# end_time = time.time()

# # obj_r, pl_value = af.run_AF_2D_pty(obj_0,sqb,afpar)

# obj_r = util.align_objects(obj,obj_r,par.mask)
# show_object(obj_r)

# print('Time: ', end_time - start_time)

# f_r = par.forward_2D_pty(obj_r)
# b_r = par.forward_to_meas_2D_pty(f_r)
# # print('Mesurements from reconstructed object:')
# # # show_measurements(b_r,locations_2d,shift)
# # # print('Absolute pixelwise error:')
# # # show_measurements(np.abs(b-b_r),locations_2d,shift)

# print('Reconstruction:')
# print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
# print( 'Relative measurement error: ', util.relative_measurement_error(b_n,b_r))

# wigner = wdd.wdd(b_n,
#                  ptycho = par,
#                  reg_type = 'value',#'percent',
#                  reg_threshold = -1,#-1.0,#0.2,#10**-1,#10**-10,
#                  nfft_diags = True,
#                  mg_type = 'diag',
#                  mg_diagonals_type = 'all',#'percent',
#                  mg_diagonals_param = 0.8,
#                  as_wtype = 'weighted',
#                  as_threshold = 10**-10,                                  
#                  add_dummy = False,
#                  subspace_completion = False,
#                  sbc_threshold = 0.1,
#                  memory_saving = True,
#                  xt = obj)

wigner2 = wdd_nfft.wigner_2D_nfft(b_n,
                 ptycho = par,
                 reg_type = 'value',#'value',#'percent',
                 reg_threshold = 1e-5,#0.2,#10**-1,#10**-10,
                 nfft_diags = True,
                 mg_type = 'diag',
                 mg_diagonals_type = 'all',#'percent',
                 mg_diagonals_param = 0.75,
                 as_wtype = 'weighted',
                 as_threshold = 10**-10,  
                 add_dummy = False,
                 subspace_completion = False,
                 sbc_threshold = 0.1,
                 nfft_rel_err_thr = 1e-12,
                 nfft_dfft_count = 10000,
                 memory_saving = True,
                 xt = obj)


print('Reconstructing...')

start_time = time.time()
obj_r = wigner2.run()
end_time = time.time()

# obj_r, pl_value = af.run_AF_2D_pty(obj_0,sqb,afpar)

obj_r = util.align_objects(obj,obj_r,par.mask)
show_object(255*obj_r)

print('Time: ', end_time - start_time)

f_r = par.forward_2D_pty(obj_r)
b_r = par.forward_to_meas_2D_pty(f_r)
# print('Mesurements from reconstructed object:')
# # show_measurements(b_r,locations_2d,shift)
# # print('Absolute pixelwise error:')
# # show_measurements(np.abs(b-b_r),locations_2d,shift)

print('Reconstruction:')
print( 'Relative error: ', util.relative_error(obj,obj_r,par.mask) )
print( 'Relative measurement error: ', util.relative_measurement_error(b_n,b_r))
