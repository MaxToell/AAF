
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 09:59:10 2022

@author: oleh.melnyk
"""
import numpy as np
import cupy as cp
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
import types

from scipy import ndimage, misc

# import hdf5plugin
# import h5py
from pathlib import Path


def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (satur_parser(im_hsv[:,:,0],im_hsv[:,:,1]) * 2 -1)* cmath.pi
    obj = modulus * cp.exp(1.0j * phase)
    obj_mod = satur_parser(obj,im_hsv[:,:,1])
    
    return obj_mod

def show_object(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    obj_norm = 255 * obj / np.max(cp.abs(obj))
    
    modulus = np.repeat(np.round(cp.abs(obj_norm)).astype(np.uint8)[:, :,np.newaxis], 3, axis=2)
    
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
    R_1D = cp.sqrt(R).astype(cp.int)
#    delta = b.shape[0]
#    half = np.round(delta*0.5).astype(np.int)
    
    b_sc = cp.log10(b+ 10**-5)
#    b_sc = np.roll(b_sc, half, axis=0)
#    b_sc = np.roll(b_sc, half, axis=1)
    
    fig, ax = plt.subplots(nrows = R_1D, ncols = R_1D)
    for r in range(R):
        loc = locations[r,:]
        x = (loc[0]/shift).astype(cp.int)
        y = (loc[1]/shift).astype(cp.int)
        im = ax[x][y].imshow(b_sc[:,:,r],cmap='hot')
        ax[x][y].axis('off')
    
    fig.colorbar(im,ax=ax.ravel().tolist(), label='log_10(Intensity)')
    plt.show()



# b = np.load('data4_x64.npy')
b = cp.load('your_dataset.npy')

d_orig = 64

# show brightfield image

bright = cp.sum(b, axis = (0,1))
bright = cp.reshape(bright,(d_orig,d_orig))#,order='F')

# plt.imshow(bright[1:,:])
# plt.show()

# show the aggregated diffraction pattern

pic = cp.sum(b,axis = 2)
# plt.imshow(np.log10(pic))
# plt.show()

# center the bright spot in the middle of the detector by cutting it out

c = (32,32)
# w = 10
w = 32 #np.min( [c[0],b.shape[0] - c[0],c[1],b.shape[1] - c[1]])
b2 = b[(c[0]-w):(c[0]+w),(c[1]-w):(c[1]+w),:]
pic = cp.sum(b2,axis = 2)
# plt.imshow(np.log10(pic))  
# plt.show()

# show the fourier transform of the aggregated diffraction pattern,
# which is rough approximation of the probe/window. Use it to estimate 
# support parameter delta

pic2 = cp.fft.fft2(pic)
pic2 = cp.fft.fftshift(pic2)
# plt.imshow(np.log10(np.abs(pic2))) 
# plt.show()

# # cut out part of the dataset to reduce computation time 

loc_num = 64#64#128

# # set selected delta
delta = 1024#80#64#40

# # shift in the dataset is = 1
shift = 2#1#2

# # cut = 16
# # f_dim = (b2.shape[0],b2.shape[1])
# # dsize = (f_dim[0] - cut, f_dim[1] -cut)

# # print(b2.shape)

# # cut_h = cut // 2
# # b2 = b2[cut_h:(b2.shape[0]-cut_h),cut_h:(b2.shape[1]-cut_h),:]

pic = cp.sum(b2,axis = 2)
# plt.imshow(np.log10(pic))  
# plt.show()

detector_mask = cp.ones_like(pic)
detector_mask[0,:] = 0
detector_mask[:,0] = 0

x =cp.log10(pic * detector_mask + 10**-1)
plt.imshow(x.get())  
plt.show()

pic2 = cp.fft.fft2(pic * detector_mask)
pic2 = cp.fft.fftshift(pic2)
# plt.imshow(np.log10(np.abs(pic2 + 10**-1))) 
# plt.show()

# # adjust dataset for cropped part of the object
d = loc_num + delta -shift #(100 + 5 - 1) (dim + delta  - shift)

b2 = cp.reshape(b2, (b2.shape[0],b2.shape[1],d_orig,d_orig))
# b2 = b2[:,:,(d_orig-loc_num):,(d_orig-loc_num):] 
bright2 = cp.sum(b2, axis = (0,1))
b2 = cp.reshape(b2, (b2.shape[0],b2.shape[1],d_orig**2))

plt.imshow(bright2.get())
plt.colorbar()
plt.show()

window = cp.load('win_after40iterations.npy')
# window =ndimage.zoom(window,0.5)

# x, y = np.meshgrid(np.linspace(-1,1,b2.shape[0]), np.linspace(-1,1,b2.shape[1]))
# gauss_pre = np.sqrt(x*x+y*y)
# sigma, mu = np.array(0.1),np.array(0.0)
# g = np.exp(-( (gauss_pre-mu)**2 / ( np.array(2.0) * sigma**np.array(2 ) ) )) 
# g_Matrix = g

# window= window * g_Matrix
# plt.imshow(np.abs(window))

# window = np.fft.fft2(window)
# window = np.fft.fftshift(window)

prepared_object = cp.load('1088x1088_object_BF128x128.npy')
#show_object(np.abs(window.get()))

#show_object(prepared_object.get())

# window = np.load('d4_win_TV_1.npy')

obj_wdd = prepared_object

obj_wdd = ndimage.zoom(obj_wdd.get(),0.9865)
# show_object(np.abs(obj_wdd))
h0 = (window.shape[0] - delta) // 2
h1 = (window.shape[1] - delta) // 2

# window = window[h0:(h0 + delta),h1:(h1+delta)]
# window = np.sqrt(np.abs(window))
# window = window / np.max(window)

# show_object(window)

f_dim = (b2.shape[0], b2.shape[1])

# if delta > np.max([b2.shape[0], b2.shape[1]]):
#     h2 = (delta - b2.shape[0]) // 2
#     h3 = (delta - b2.shape[1]) // 2
#     b2_long = np.zeros( (delta,delta, b2.shape[2]))
#     b2_long[h2:(h2 + b2.shape[0]),h3:(h3+b2.shape[1]),:] = b2
    
#     detector_big = np.zeros( (delta, delta))
#     detector_big[h2:(h2 + b2.shape[0]),h3:(h3+b2.shape[1])] = detector_mask
#     f_dim = (delta,delta)
#     detector_mask = detector_big
#     b2=  b2_long

par = forward.ptycho(
            object_shape = (d,d),
            window = window, 
            circular = False,
            loc_type = 'grid',
            shift = shift, 
            fourier_dimension = f_dim,)
            #detector_mask = cp.array(detector_mask))


# # baf_par, AG_par = af.baf_params_object()

# # # Armijo-Goldstein condition (learning rate parameters)
# # # enable_AG=False implies that iterative algorithm uses constant learning rate
# # # enable_AG=True searches for largerlearning rate, but slower
 
AG_pars = types.SimpleNamespace(enable_AG=True,
                    control=0.5,
                    tau = 0.3,
                    AG_iterations = 5)

AG_pars_win = types.SimpleNamespace(enable_AG=True,
                    control=0.5,
                    tau = 0.5,
                    AG_iterations = 5)


baf = af.blind_amplitude_flow(
                        measurements= b2,
                        reweight = False,
                        ptycho = par,
                        number_of_iterations = 40, 
                        object_subiterations = 1,
                        window_subiterations = 1,
                        grad_thr_object = 10**-3,
                        grad_thr_window = 10**-5,
                        AGP_object = AG_pars,
                        AGP_window = AG_pars_win,
                        epsilon = 10**-20, # smoothing parameter of object
                        alpha_T = 10**-1, # Tikhonov regularization for object
                        beta_T = 10**-2, # Tikhonov regularization for probe
                        # alpha_R = 0, # Tikhonov regularization for imaginary part of the object
                        # beta_R = 0, # Tikhonov regularization for imaginary part of the probe
                        alpha_ST = 10**4, # Smoothness regularization || (obj_{x,y}-obj_{x+1,y}) + (obj_{x,y}-obj_{x,y+1}) ||_2^2
                        beta_ST = 10**6,
                        # alpha_TV = 10**3, # Total variation regularization || (obj_{x,y}-obj_{x+1,y}) + (obj_{x,y}-obj_{x,y+1}) ||_1
                        # beta_TV = 10**4,
                        TV_param = 1,
                        verbose =1,
                        track = True)

# obj_old = np.load('obj_r_S35.npy')
# obj_0 = obj_old[22:86, 22:86]
# obj_0 = np.repeat(np.repeat(obj_0, shift, axis =0), shift, axis = 1) / shift

# obj_r = np.zeros((d,d),dtype= complex)
# h = (d - loc_num) // 2
# obj_r[h:(h+loc_num),h:(h+loc_num)] = obj_0

obj_r = obj_wdd#np.zeros((d,d),dtype= complex) 
# # obj_r = (np.random.randn(d,d) + 1.0j*np.random.randn(d,d)) 
# # obj_r = obj_r /np.abs(obj_r)

h = (d - loc_num) // 2

#bright_rep = np.repeat(np.repeat(bright2, shift, axis =0), shift, axis = 1) / shift

#obj_r[h:(h+loc_num),h:(h+loc_num)] = np.sqrt(bright_rep) / np.linalg.norm(window,'fro') / d

#show_object(obj_r)

par_r = par.copy()
par_r.set_window(window)

f_r2 = par_r.forward_2D_pty(obj_r)
b_r2 = par_r.forward_to_meas_2D_pty(f_r2)
print( 'Relative measurement error: ', util.relative_sq_measurement_error(b2,b_r2))

# obj_r *= loc_num / d

# f_r2 = par_r.forward_2D_pty(obj_r)
# b_r2 = par_r.forward_to_meas_2D_pty(f_r2)
# print( 'Relative measurement error: ', util.relative_sq_measurement_error(b2,b_r2))

# # obj_0 = obj_0 * obj_r
# # obj_0 = obj_0 / np.linalg.norm(obj_0,'fro') * np.sqrt(np.sum(b3,axis = None))/np.linalg.norm(window,'fro')

# # obj_r = obj_0

# # obj_r = (np.random.randn(d,d) + 1.0j*np.random.randn(d,d)) 
# # obj_r = np.ones((d,d),dtype= complex)
# # obj_r = obj_r / np.linalg.norm(obj_r,'fro') * np.sqrt(np.sum(b2,axis = None))/np.linalg.norm(window,'fro')

# # obj_r = np.ones((d,d),dtype= complex)
# # h3 = d // 2
# # h4 = loc_num // 2

# # obj_r[(h3 - h4):(h3 + h4),(h3 - h4):(h3 + h4)] = np.sqrt(bright2) 

# obj_r = obj_r / np.linalg.norm(obj_r,'fro') * np.sqrt(np.sum(b2,axis = None))/np.linalg.norm(window,'fro')

win_r = window

start_time = time.time()
obj_r, win_r, pl_value = baf.run(obj_r,win_r)
# obj_r, win_r = util.eliminate_linear_ambiguity(obj_r, win_r, threshold= 10**-1)
end_time = time.time()

print('Time: ', end_time - start_time)

#show_object(obj_r)
#show_object(win_r)

par_r = par.copy()
par_r.set_window(win_r)

f_r2 = par_r.forward_2D_pty(obj_r)
b_r2 = par_r.forward_to_meas_2D_pty(f_r2)
print( 'Relative measurement error: ', util.relative_sq_measurement_error(b2,b_r2))
cp.save('test_series_obj_512x512.npy', obj_r)
cp.save('test_series_win_512x512.npy', win_r)
obj_z=obj_r[31:1055,31:1055]
cp.save('test_series_obj_z_512x512.npy', obj_z)
#show_object(obj_z)
import winsound
frequency = 400  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
