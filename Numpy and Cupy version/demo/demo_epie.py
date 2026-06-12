# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 17:36:45 2022

@author: oleh.melnyk
"""

import numpy as np
import sys
sys.path.insert(1, '../model')
sys.path.insert(1, '../algorithms')
import forward as forward
# import af_2D as af
import epie_2D as epie
import utility_2D as util
from scipy.ndimage import zoom


import cmath
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage import data
import matplotlib.pyplot as plt


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
outd = im_big.shape[0] // 8
factor = outd*1.0/im_big.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_big[:,:,0],factor)
im[:,:,1] = zoom(im_big[:,:,1],factor)
im[:,:,2] = zoom(im_big[:,:,2],factor)

d = im.shape[0]

obj = image_to_object(im,lambda x,v: x)

delta = 20
shift = 5

dsize = (d,d)

# locations_1d = np.array(range(0,d-delta+1, shift))
# locations_2d = np.zeros((len(locations_1d)**2,2),dtype=int)
# locations_2d[:,0] = np.repeat(locations_1d,len(locations_1d))
# locations_2d[:,1] = np.tile(locations_1d,len(locations_1d))
# locations_2d, mask = forward.loc_mesh_grid(d,delta,shift)
# locations_2d = forward.loc_Fermat_spiral(d, delta, 4.9)

show_object(obj)

cov_mat = np.eye(2,dtype = complex)/0.05
mu = np.array([0.5 + delta*0.5, 0.5 + delta*0.5])
gauss = lambda x: np.exp(-0.5* (( x - mu).conj().T).dot((cov_mat/delta**2).dot(x - mu)))
window = np.zeros((delta,delta), dtype = complex);
for ix in range(delta):
    for iy in range(delta):
        window[ix,iy] = gauss( np.array([ix+1, iy+1]));
        
par = forward.ptycho(
            object_shape = obj.shape,
            window = window, 
            circular = False,
            loc_type = 'grid',
            shift = shift, 
            fourier_dimension = (d,d))

print('Computing forward model')
f = par.forward_2D_pty(obj)

print('Computing measurements')
b = par.forward_to_meas_2D_pty(f)

show_measurements(b,par.locations,shift)

obj_0 = 255 * np.sqrt(0.5) * (np.random.normal(size=(d,d)) + 1.0j * np.random.normal(size=(d,d)))
show_object(obj_0)

win_0 = np.ones_like(window)
win_0 = util.normalize_window(win_0)

# af_params,AG_params = epie.af_params_object()


# AG_par = AG_params(enable_AG = True,
                    # control = 0.5,
                    # tau = 0.3,
                    # AG_iterations = 2)

# afpar = af_params(ptycho_params = par.copy(),
#                   AG_params = AG_par,
#                   number_of_iterations = 100000, 
#                   grad_threshold = 10**-3,
#                   learning_rate_type = 'optimal',
#                   learning_rate = 1.0,
#                   epsilon = 10**-8,
#                   alpha_T = 0,
#                   alpha_R = 0,
#                   verbose = 1)

# sqb = np.sqrt(np.maximum(b,0) + afpar.epsilon)

# obj_r, pl_value = af.run_AF_2D_pty(obj_0,sqb,afpar)
pie_alg = epie.ePIE(
    measurements= b,
    ptycho = par.copy(),
    # AG_params = AG_par,
    number_of_iterations = 100000, 
    grad_threshold = 10**-3,
    # learning_rate_type = 'optimal',
    # learning_rate = 1.0,
    epsilon = 10**-8,
    alpha_T = 0,
    alpha_R = 0,
    verbose = 1,
    track = False, 
    alpha = 0.005,
    beta = 0.005)

# obj_r, pl_value = af3.run_AF_2D_pty(obj_0,sqb,afpar)
obj_r, win_r = pie_alg.run(obj_0,win_0)

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
print('Absolute pixelwise error:')
show_measurements(np.abs(b-b_r),par.locations,shift)
print('Relative pixelwise error:')
show_measurements(np.abs(b-b_r)/b,par.locations,shift)

print('Reconstruction:')
print( 'Relative error (object): ', util.relative_error(obj,obj_r, par.mask) )
print( 'Relative error (window): ', util.relative_error(window,win_r, np.ones( (window.shape[0],window.shape[1])) ))
print( 'Relative measurement error: ', util.relative_measurement_error(b,b_r))
