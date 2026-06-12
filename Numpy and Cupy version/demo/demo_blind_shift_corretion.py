# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:20:57 2024

@author: oleh.melnyk
"""

import numpy as np
import sys
sys.path.insert(1, '../model')
sys.path.insert(1, '../algorithms')
import forward as forward
import af_2D as af
import utility_2D as util
from scipy.ndimage import zoom

import cmath
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage import data
import matplotlib.pyplot as plt
import time
import types


def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (satur_parser(im_hsv[:,:,0],im_hsv[:,:,1]) * 2 -1)* cmath.pi
    obj = modulus * np.exp(1.0j * phase)
    obj_mod = satur_parser(obj,im_hsv[:,:,1])
    
    return obj_mod

def show_object(obj):
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2)
    
    modulus = np.repeat(np.round(np.abs(obj)).astype(np.uint8)[:, :,np.newaxis], 3, axis=2)
    
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
im_big = im_big[:,:im_big.shape[0],:]
outd = np.round(im_big.shape[0]*0.125).astype(int)
factor = outd*1.0/im_big.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_big[:,:,0],factor)
im[:,:,1] = zoom(im_big[:,:,1],factor)
im[:,:,2] = zoom(im_big[:,:,2],factor)

d = im.shape[0]

obj = image_to_object(im,lambda x,v: x )

delta = 20
shift = 5.0

dsize = (d,d)

# locations_2d = forward.loc_Fermat_spiral(d, delta, 4.9)
# locations_2d, mask = forward.loc_mesh_grid(d,delta,shift)

show_object(obj)

cov_mat = np.eye(2,dtype = complex)/0.05
mu = np.array([0.5 + delta*0.5, 0.5 + delta*0.5])
gauss = lambda x: np.exp(-0.5* (( x - mu).conj().T).dot((cov_mat/delta**2).dot(x - mu)))
window = np.zeros((delta,delta), dtype = complex);
for ix in range(delta):
    for iy in range(delta):
        window[ix,iy] = gauss( np.array([ix+1, iy+1])) #* np.exp(1j*np.pi*( (ix + 1 - mu[0])**2+(iy + 1 - mu[1])**2)/20)
        
window = util.normalize_window(window)

show_object(255.0*window / np.max(window, (0,1)) )
        
par = forward.ptycho(
            object_shape = obj.shape,
            window = window, 
            float_shift = True,
            circular = False,
            loc_type = 'grid',
            shift = shift, 
            fourier_dimension = (d,d))

print('Computing forward model')
f = par.forward_2D_pty(obj)

print('Computing measurements')
b = par.forward_to_meas_2D_pty(f)

factor = par.locations // shift
par.locations += factor *0.5


# show_measurements(b,par.locations,shift)

obj_0 = 255*np.sqrt(0.5)*( np.random.normal(size=(d,d)) + 1.0j * np.random.normal(size=(d,d)))
win_0 =np.ones_like(window)
win_0 = util.normalize_window(win_0)

show_object(obj_0)
show_object(win_0)

# baf_par, AG_par = af.baf_params_object()

AG_pars = types.SimpleNamespace(enable_AG=True,
                   control=0.5,
                   tau = 0.3,
                   AG_iterations = 2)

AG_pars_shift = types.SimpleNamespace(enable_AG=False,
                    control=0.5,
                    tau = 0.1,
                    AG_iterations = 10)

# |F[Sx o w]|^2
# F diag(w) S x = Ax
# F diag(Sx) w = Bw

# L(x) = |\sqrt y - |F[Sx o w]| |^2 + alpha_o ||x||^2 + alpha_w ||w||^2

# L(x) = sum_ scanning points L(x,s)
# L(x_0)/T

# bafpar = types.SimpleNamespace(ptycho_params = par,
#                   number_of_iterations = 30, 
#                   object_subiterations = 20,
#                   window_subiterations = 20,
#                   grad_thr_object = 10**-1,
#                   grad_thr_window = 10**-5,
#                   AGP_object = AG_pars,
#                   AGP_window = AG_pars,
#                   epsilon = 10**-12,
#                   alpha_T = 10**-2,
#                   beta_T = 10**-2,
#                   alpha_R = 0,
#                   beta_R = 0,
#                   alpha_ST = 10**0,
#                   beta_ST = 10**0,
#                   verbose =1)

baf = af.blind_amplitude_flow(
                        measurements= b,
                        ptycho = par,
                        number_of_iterations = 30, 
                        object_subiterations = 20,
                        window_subiterations = 20,
                        shift_subiterations=300,
                        skip_win_it = 5,
                        update_shifts = True,
                        skip_shift_it = 15,
                        shift_learning_rate_type = 'custom',
                        shift_learning_rate = 1.0,
                        update_shifts_period = 5,
                        grad_thr_object = 10**-1,
                        grad_thr_window = 10**-5,
                        AGP_object = AG_pars,
                        AGP_window = AG_pars,
                        AGP_shift = AG_pars_shift,
                        epsilon = 10**-12,
                        alpha_T = 10**-2,
                        beta_T = 10**-2,
                        alpha_R = 0,
                        beta_R = 0,
                        # alpha_ST = 10**0,
                        beta_ST = 10**0,
                        alpha_TV = 10**1,
                        # beta_TV = 10**1,
                        TV_param = 10**-1,
                        # reweight = True,
                        verbose =1,
                        track = True, 
                        obj_tr=  obj, 
                        win_tr= window,
                        track_it= np.array([10,20,50]))

start_time = time.time()
obj_r, win_r, pl_value = baf.run(obj_0,win_0)
end_time = time.time()

print('Runtime: ', end_time - start_time)

obj_r, win_r = util.eliminate_linear_ambiguity(obj_r, win_r)
obj_r = util.align_objects(obj,obj_r,par.mask)
win_r = util.align_objects(window,win_r,np.ones( (window.shape[0],window.shape[1])))
show_object(obj_r)
show_object(255.0*win_r / np.max( np.abs(win_r), (0,1)) )

par_r = par.copy()
par_r.set_window(win_r)
f_r = par_r.forward_2D_pty(obj_r)
b_r = par_r.forward_to_meas_2D_pty(f_r)
# print('Mesurements from reconstructed object:')
# show_measurements(b_r,locations_2d,shift)
# print('Absolute pixelwise error:')
# show_measurements(np.abs(b-b_r),par.locations,shift)
# print('Relative pixelwise error:')
# show_measurements(np.abs(b-b_r)/b,par.locations,shift)

print('Reconstruction:')
print( 'Relative error (object): ', util.relative_error(obj,obj_r, par.mask) )
print( 'Relative error (window): ', util.relative_error(window,win_r, np.ones( (window.shape[0],window.shape[1])) ))
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))

