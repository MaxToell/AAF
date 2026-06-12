# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:19:45 2022

@author: oleh.melnyk
"""


import numpy as np
import sys
sys.path.insert(1, '../model')
sys.path.insert(1, '../algorithms')

# for testing

import forward_circ as forward
# import forward_2D as forward

import wigner_2D as wdd
import utility_2D as util
from scipy.ndimage import zoom


import cmath
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb
from skimage import data
import matplotlib.pyplot as plt


#from recordtype import recordtype

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

def image_to_object(im,satur_parser):
    im_hsv = rgb2hsv(im)
       
    modulus = im_hsv[:,:,2] * 255 
    phase = (satur_parser(im_hsv[:,:,0],im_hsv[:,:,1]) * 2 -1)* cmath.pi
    obj = modulus * np.exp(1.0j * phase)
    obj_mod = satur_parser(obj,im_hsv[:,:,1])
    
    return obj_mod

print('Preparing image')

im_big = data.coffee()/255.0
#im = im[300:,:]
start = np.round(0.25*im_big.shape[0]).astype(np.int)
im_big = im_big[:,start:(start + im_big.shape[0]),:]

outd = 1024#im_big.shape[0]//8
factor = outd*1.0/im_big.shape[0]
im = np.zeros((outd,outd,3))
im[:,:,0] = zoom(im_big[:,:,0],factor)
im[:,:,1] = zoom(im_big[:,:,1],factor)
im[:,:,2] = zoom(im_big[:,:,2],factor)

print('Setting up parameters')
# outd = 100
# im = im_big[:outd,:outd,:]
d = im.shape[0]

obj = image_to_object(im,lambda x,v: x)

delta = 128
shift = 64
# delta SHOULD be at least 2*shift

print('Grid')
locations_2d, mask = forward.loc_mesh_grid(d,delta,shift)

# locations_2d = np.ndarray((1,2))
# locations_2d[0,:] = [20,25]
mask = []

# f_dim = (d,d)
# dsize = (d,d)

f_dim = (2*delta,2*delta)
dsize = f_dim


# locations_1d = np.array(range(0,d-delta+1, shift))
# locations_2d = np.zeros((len(locations_1d)**2,2),dtype=int)
# locations_2d[:,0] = np.repeat(locations_1d,len(locations_1d))
# locations_2d[:,1] = np.tile(locations_1d,len(locations_1d))

# locations_2d = forward.loc_Fermat_spiral(d, delta, 4.9)

# show_object(obj)

print('Window')

cov_mat = np.eye(2,dtype = complex)/0.005
mu = np.array([0.5 + delta*0.5, 0.5 + delta*0.5])
gauss = lambda x: np.exp(-0.5* (( x - mu).conj().T).dot((cov_mat/delta**2).dot(x - mu)))
window = np.zeros((delta,delta), dtype = complex);
for ix in range(delta):
    for iy in range(delta):
        # window[ix,iy] = gauss( np.array([ix+1, iy+1]))
        window[ix,iy] = gauss( np.array([ix+1, iy+1])) * np.exp(1j*np.pi*( (ix + 1 - mu[0])**2+(iy + 1 - mu[1])**2)/delta)

# window = window *np.exp(2j*np.pi*np.random.rand(delta,delta))

show_object(255*window)
        

pty_params = forward.pty_params()

par = pty_params(window = window, 
                 locations = locations_2d,
                 detector_shape = dsize,
                 object_shape = obj.shape,
                 window_shape = window.shape,
                 fourier_dimension = f_dim,
                 mask = mask)

print('Computing forward model')
f = forward.forward_2D_pty(obj,par)

print('Computing measurements')
b = forward.forward_to_meas_2D_pty(f)

f_dim = (d,d)
dsize = (d,d)

par.fourier_dimension = f_dim 
par.detector_shape = dsize

print('Computing forward model')
f2 = forward.forward_2D_pty(obj,par)

print('Computing measurements')
b2 = forward.forward_to_meas_2D_pty(f2)

fig, ax = plt.subplots(nrows = 1, ncols = 2)
    
b1_log = np.log10(b[:,:,150] + 10**-5)
b2_log = np.log10(b2[:,:,150] + 10**-5)

# plt.figure(figsize=(256, 256), dpi=1)
plt.imshow(b1_log,cmap='hot')
plt.axis('off')
plt.savefig('example_subsampling_sub.png',pad_inches = 0.0,bbox_inches='tight')

# fig = plt.figure(frameon=False)
# fig.set_size_inches(b1_log.shape[0],b1_log.shape[1])
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
# ax.imshow(b1_log)
# fig.savefig('example_subsampling_sub.png', dpi=1)


# plt.figure(figsize=(1024, 1024), dpi=300)
plt.imshow(b2_log,cmap='hot')
plt.axis('off')
plt.savefig('example_subsampling_comp.png',pad_inches = 0.0,bbox_inches='tight')

# im = ax[0].imshow(b1_log,cmap='hot')
# ax[0].axis('off')
    
# im = ax[1].imshow(b2_log,cmap='hot')
# ax[1].axis('off')

# # fig.colorbar(im,ax=ax.ravel().tolist(), label='log_10(Intensity)')
# fig.show()

