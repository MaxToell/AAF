# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:36:08 2022

@author: oleh.melnyk
"""


import numpy as np
import sys
sys.path.insert(1, '../model')
sys.path.insert(1, '../algorithms')

import forward as forward

import wigner_2D as wdd
import af_2D as af

import utility_2D as util
import cmath
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt

import time
import copy

from scipy import ndimage, misc

import hdf5plugin
import h5py
from pathlib import Path

def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (satur_parser(im_hsv[:,:,0],im_hsv[:,:,1]) * 2 -1)* cmath.pi
    obj = modulus * np.exp(1.0j * phase)
    obj_mod = satur_parser(obj,im_hsv[:,:,1])
    
    return obj_mod

def show_object(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    obj_norm = 255 * obj / np.max(np.abs(obj))
    
    modulus = np.repeat(np.round(np.abs(obj_norm)).astype(np.uint8)[:, :,np.newaxis], 3, axis=2)
    
    phase = np.ones((obj_norm.shape[0],obj.shape[1],3))
    phase[:,:,0] = (np.angle(obj_norm) + cmath.pi)/(2*cmath.pi)
    phase_rgb = hsv2rgb(phase)
                
    ax[0].imshow(modulus)
    ax[0].axis('off')
    ax[1].imshow(phase_rgb)
    ax[1].axis('off')
        
    plt.show()
    
    
def show_measurements(b,locations,shift):
    R = locations.shape[0]
    R_1D = np.sqrt(R).astype(np.int)
#    delta = b.shape[0]
#    half = np.round(delta*0.5).astype(np.int)
    
    b_sc = np.log10(b+ 10**-5)
#    b_sc = np.roll(b_sc, half, axis=0)
#    b_sc = np.roll(b_sc, half, axis=1)
    
    fig, ax = plt.subplots(nrows = R_1D, ncols = R_1D)
    for r in range(R):
        loc = locations[r,:]
        x = (loc[0]/shift).astype(np.int)
        y = (loc[1]/shift).astype(np.int)
        im = ax[x][y].imshow(b_sc[:,:,r],cmap='hot')
        ax[x][y].axis('off')
    
    fig.colorbar(im,ax=ax.ravel().tolist(), label='log_10(Intensity)')
    plt.show()

# load and prepare data

p = Path("D:\Studies\dataset.h5")

# s = h5py.File(p)
# s = s['entry/data/data']

# b = s[:,:,:]
# b = np.moveaxis(b, 0, 2)

# new_m = 64
# new_det = 64

# b2 = np.zeros( (new_det,new_det,b.shape[2]) )
# for idx in range(b.shape[2]):
#     b2[:,:,idx] = ndimage.zoom(b[:,:,idx], 1.0*new_det/b.shape[0])

# b = np.load('data4_x64.npy')
b = np.load('dataset.npy')

d_orig = 128

# show brightfield image

bright = np.sum(b, axis = (0,1))
bright = np.reshape(bright,(d_orig,d_orig))#,order='F')

plt.imshow(bright[2:,:])
plt.show()

# show the aggregated diffraction pattern

pic = np.sum(b,axis = 2)
plt.imshow(np.log10(pic))
plt.show()

# center the bright spot in the middle of the detector by cutting it out

c = (33,34)
# w = 10
w = np.min( [c[0],b.shape[0] - c[0],c[1],b.shape[1] - c[1]])
b2 = b[(c[0]-w):(c[0]+w),(c[1]-w):(c[1]+w),:]
pic = np.sum(b2,axis = 2)
plt.imshow(np.log10(pic))  
plt.show()

# show the fourier transform of the aggregated diffraction pattern,
# which is rough approximation of the probe/window. Use it to estimate 
# support parameter delta

pic2 = np.fft.fft2(pic)
pic2 = np.fft.fftshift(pic2)
plt.imshow(np.log10(np.abs(pic2))) 
plt.show()

# cut out part of the dataset to reduce computation time 

loc_num = 60

# set selected delta
delta = 20

# shift in the dataset is = 1
shift = 1

# cut = 16
# f_dim = (b2.shape[0],b2.shape[1])
# dsize = (f_dim[0] - cut, f_dim[1] -cut)

# print(b2.shape)

# cut_h = cut // 2
# b2 = b2[cut_h:(b2.shape[0]-cut_h),cut_h:(b2.shape[1]-cut_h),:]

pic = np.sum(b2,axis = 2)
plt.imshow(np.log10(pic))  
plt.show()

detector_mask = np.ones_like(pic)
# detector_mask[17,57:59] = 0
# detector_mask[18,56:59] = 0
# detector_mask[19,57] = 0
# detector_mask[32,52:54] = 0
# detector_mask[34,52:54] = 0
# detector_mask[37,27] = 0

plt.imshow(np.log10(pic * detector_mask + 10**-1))  
plt.show()

pic2 = np.fft.fft2(pic * detector_mask)
pic2 = np.fft.fftshift(pic2)
plt.imshow(np.log10(np.abs(pic2 + 10**-1))) 
plt.show()

# adjust dataset for cropped part of the object
d = loc_num + delta -shift #(100 + 5 - 1) (dim + delta  - shift)

b2 = np.reshape(b2, (b2.shape[0],b2.shape[1],d_orig,d_orig))
b2 = b2[:,:,(d_orig-loc_num):,(d_orig-loc_num):] 
bright2 = np.sum(b2, axis = (0,1))
b2 = np.reshape(b2, (b2.shape[0],b2.shape[1],loc_num**2))

plt.imshow(bright2)
plt.colorbar()
plt.show()

# b3 = np.zeros((b2.shape[0]//2,b2.shape[1]//2,b2.shape[2])) 
# for idx in range(b2.shape[2]):
    # b3[:,:,idx] = ndimage.zoom(b2[:,:,idx], 0.5)
    # ADD ROTOATION IF NEEDED 
    # b3[:,:,idx] = ndimage.rotate(b3[:,:,idx],angle = -90, reshape= False)

# f_dim = (b2.shape[0],b2.shape[1])
# dsize = f_dim



# locations_2d, mask = forward.loc_mesh_grid(d,delta,shift)

# generate window

cov_mat = np.eye(2,dtype = complex)/0.1
mu = np.array([0.5 + delta*0.5, 0.5 + delta*0.5])
gauss = lambda x: np.exp(-0.5* (( x - mu).conj().T).dot((cov_mat/delta**2).dot(x - mu)))
window = np.zeros((delta,delta), dtype = complex);
for ix in range(delta):
    for iy in range(delta):
        # window[ix,iy] = gauss( np.array([ix+1, iy+1]))
        window[ix,iy] = gauss( np.array([ix+1, iy+1])) #* np.exp(1j*np.pi*( (ix + 1 - mu[0])**2+(iy + 1 - mu[1])**2)/delta)


h1 = delta //2
h2 = pic2.shape[0] // 2
if h1 <= h2:
    window = np.sqrt(np.abs(pic2[(h2-h1):(h2+h1),(h2-h1):(h2+h1)]))
else:
    window = np.ones((delta,delta), dtype = complex)
    window = window * np.min(np.sqrt(np.abs(pic2)))
    window[(h1-h2):(h1+h2),(h1-h2):(h1+h2)] = np.sqrt(np.abs(pic2))
    
window =window / np.max(window)
    
# window = np.load('window.npy')


show_object(window)

par = forward.ptycho(
            object_shape = (d,d),
            window = window, 
            circular = False,
            loc_type = 'grid',
            shift = 1, 
            fourier_dimension = (b2.shape[0],b2.shape[1]),
            detector_mask = detector_mask)

par_wdd = copy.deepcopy(par)

# par_r = copy.deepcopy(par)
# par_r.detector_shape = par_r.fourier_dimension

# wdd_params = wdd.wdd_params_object()
# wdd_par = wdd_params(ptycho_params = par_wdd,
#                         shift_size = (shift,shift),
#                         as_wtype = 'weighted',
#                         as_threshold = 10**-10,
#                         reg_threshold = 10**-1,#10**-10,
#                         mg_type = 'log',
#                         mg_factor = 1.5,
#                         circ_shifts = False,
#                         add_dummy = False,
#                         subspace_completion = False,
#                         sbc_threshold = 0.1)

# print('Reconstructing...')

wdd_solver = wdd.wdd(b2,
                    ptycho = par_wdd,
                    reg_type = 'percent',
                    reg_threshold = 0.35,#10**-1,#10**-10,
                    mg_type = 'log',
                    as_wtype = 'weighted',
                    as_threshold = 10**-10,
                    add_dummy = False,
                    subspace_completion = False)
start_time = time.time()
obj_wdd = wdd_solver.run() #wdd.Wigner_2D(b, wdd_par)
end_time = time.time()
show_object(obj_wdd)

print('Time: ', end_time - start_time)


f_r = par_wdd.forward_2D_pty(obj_wdd)
b_r = par_wdd.forward_to_meas_2D_pty(f_r)
print( 'Relative measurement error: ', util.relative_sq_measurement_error(b2,b_r))

delta = 60
d = loc_num + delta -shift

h1 = delta //2
h2 = pic2.shape[0] // 2
if h1 <= h2:
    window = np.sqrt(np.abs(pic2[(h2-h1):(h2+h1),(h2-h1):(h2+h1)]))
else:
    window = np.ones((delta,delta), dtype = complex)
    window = window * np.min(np.sqrt(np.abs(pic2)))
    window[(h1-h2):(h1+h2),(h1-h2):(h1+h2)] = np.sqrt(np.abs(pic2))
    
window =window / np.max(window)

d_old = obj_wdd.shape[0]
h_old = (d - d_old)//2
obj_r = np.ones((d,d), dtype = complex)
obj_r[h_old:(h_old + d_old),h_old:(h_old + d_old)] = obj_wdd  


par = forward.ptycho(
            object_shape = (d,d),
            window = window, 
            circular = False,
            loc_type = 'grid',
            shift = 1, 
            fourier_dimension = (b2.shape[0],b2.shape[1]),
            detector_mask = detector_mask)

baf_par, AG_par = af.baf_params_object()

# # Armijo-Goldstein condition (learning rate parameters)
# # enable_AG=False implies that iterative algorithm uses constant learning rate
# # enable_AG=True searches for largerlearning rate, but slower
 
AG_pars = AG_par(enable_AG=False,
                    control=0.5,
                    tau = 0.3,
                    AG_iterations = 2)

bafpar = baf_par(ptycho_params = par,
                  number_of_iterations = 50, 
                  object_subiterations = 5,
                  window_subiterations = 5,
                  grad_thr_object = 10**-3,
                  grad_thr_window = 10**-5,
                  AGP_object = AG_pars,
                  AGP_window = AG_pars,
                  epsilon = 10**-12, # smoothing parameter of object
                  alpha_T = 0,#10**2.5, # Tikhonov regularization for object
                  beta_T = 0,#10**-4, # Tikhonov regularization for probe
                  alpha_R = 0, # Tikhonov regularization for imaginary part of the object
                  beta_R = 0, # Tikhonov regularization for imaginary part of the probe
                  verbose =1)

baf = af.blind_amplitude_flow(
                        measurements= b2,
                        bafpar= bafpar, 
                        reweight = True,
                        # circ = False,
                        track = False)



obj_r = obj_r / np.linalg.norm(obj_r,'fro') * np.sqrt(np.sum(b2,axis = None))/np.linalg.norm(window,'fro')

win_r = window

start_time = time.time()
obj_r, win_r, pl_value = baf.run_AF_2D_blind_pty(obj_r,win_r)
# obj_r, win_r = util.eliminate_linear_ambiguity(obj_r, win_r, threshold= 10**-1)
end_time = time.time()

print('Time: ', end_time - start_time)

show_object(obj_r)
show_object(win_r)

par_r = par.copy()
par_r.set_window(win_r)

f_r2 = par_r.forward_2D_pty(obj_r)
b_r2 = par_r.forward_to_meas_2D_pty(f_r2)
print( 'Relative measurement error: ', util.relative_sq_measurement_error(b2,b_r2))

bright3 = np.reshape(np.sum(b_r2, axis = (0,1)), (loc_num, loc_num))





#