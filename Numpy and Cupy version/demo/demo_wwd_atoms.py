# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:47:52 2020

@author: oleh.melnyk
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import json

import sys
sys.path.insert(1, '../model')
sys.path.insert(1, '../algorithms')

from forward_circ import pty_params
from forward_circ import loc_mesh_grid
import wigner_2D as wdd

#%matplotlib nbagg

## Set path
path = 'D:\\Studies\\PhD\\Ptychography4.0\\pyslice\\OOP\\multislice\\Cu_100_rel.asc\\06-12-2020_14.29.29'
#directory = os.listdir(path)
            
## Get random specimen
#idx_atom = np.random.randint(len(directory))
            
## Set path
#path_choose_specimen = os.path.join(os.path.join(path,directory[idx_atom]))
#time_stamp = os.listdir(path_choose_specimen)
#path_specimen = os.path.join(path_choose_specimen, time_stamp[0])
path_specimen = path
## Print Specimen's name
print(path_specimen)
## Load Parameters simulation
jsonfile =  os.path.join(path_specimen, 'parameters.json')
f = open(jsonfile)
simulation_params = json.load(f)
dim_uncut = (simulation_params['detector']['dim_un'])
scan_points = (np.int(dim_uncut[2]/simulation_params['microparam']['Npix']),
                           simulation_params['microparam']['Npix'])
f.close()
             
## Load slice 1
idx = 1
OTF = os.path.join(path_specimen,sorted(os.listdir(path_specimen))[idx], 'object.raw')
Probes = os.path.join(path_specimen,sorted(os.listdir(path_specimen))[idx], 'probes.raw')
           
## Object
object_tf  = np.memmap(OTF, dtype=np.complex, mode='r', 
                                 shape = (dim_uncut[0],dim_uncut[1]))
probe3d  = np.memmap(Probes, dtype=np.complex, mode='r', 
                                 shape = (dim_uncut[0],dim_uncut[1],dim_uncut[2]))
probe3d = probe3d/(np.max(np.abs(probe3d), axis = (0,1))) 

probe4d = np.moveaxis(probe3d.reshape(dim_uncut[0],dim_uncut[1],
                                                    scan_points[0],scan_points[1]), [0, 1], [2, 3])
            
## Suppose we know the probe, take at the center
scan_dim = probe4d.shape[0:2]
det_dim = probe4d.shape[2:4]
center =  (np.array(scan_dim)//2)
print(center)

## Multiplication probe and object         
mlt = probe4d*object_tf[np.newaxis,np.newaxis,:,:]

## Diffraction Pattern
cdp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mlt), axes=(2, 3)))
idp = np.abs(cdp)**2

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(np.abs(object_tf))
ax2.imshow(np.angle(object_tf))
plt.show()

probe_init = probe4d[30, 10]
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(np.abs(probe_init))
ax2.imshow(np.angle(probe_init))
plt.show()

delta = 40
shift = 2

starting_point = ((center[0] + delta//(2*shift) ) % scan_dim[0], (delta//(2*shift))% scan_dim[1])

cutout_probe = probe4d[starting_point[0], starting_point[1]][:delta,:delta]

# Rearrange measurements, so that scan dimensions are a single dim 

## First step, prepare the scan position ([0,1] is the diffraction,[2,3] is the coordinate)
idp_scan = np.moveaxis(idp,[0,1],[2,3]) 

idp_shifted = np.roll(np.roll(idp_scan, -starting_point[0], axis = 2), -starting_point[1], axis = 3)
idp_reshaped = np.reshape(idp_shifted, (idp_shifted.shape[0], idp_shifted.shape[1], idp_shifted.shape[2]*idp_shifted.shape[3]))

locations_2d = loc_mesh_grid(dim_uncut[0],delta,shift)

pty_params_cons = pty_params()

par = pty_params_cons(window = cutout_probe, 
                 locations = locations_2d,
                 detector_shape = (dim_uncut[0],dim_uncut[1]),
                 object_shape = (dim_uncut[0],dim_uncut[1]),
                 window_shape = cutout_probe.shape,
                 fourier_dimension = (dim_uncut[0],dim_uncut[1]))

wdd_params = wdd.wdd_params_object()

wdd_par = wdd_params(ptycho_params = par,
                     shift_size = (shift,shift),
                     as_wtype = 'weighted',
                     as_threshold = 10**-10,
                     reg_threshold = 10**-0,
                     mg_type = 'diag', # try using 'diag' to see stability problems comming from magnitude estimation
                     mg_factor = 1.5,
                     circ_shifts = True,
                     add_dummy = False)

obj_r = wdd.Wigner_2D(idp_reshaped, wdd_par)

vmin = np.min([np.min(np.abs(obj_r)), np.min(np.abs(object_tf) )])
vmax = np.max([np.max(np.abs(obj_r)), np.max(np.abs(object_tf) )])

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(np.abs(obj_r))
ax2.imshow(np.angle(obj_r))
plt.show()


fig, axes = plt.subplots(1,2)
im = axes[0].imshow(np.abs(obj_r), vmin=vmin, vmax=vmax)
im = axes[1].imshow(np.abs(object_tf), vmin=vmin, vmax=vmax)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()


zer = np.repeat(0,100)
x = np.linspace(0, 80, 100)
x1 = np.linspace(-40, 40, 100)
probe_plot = probe4d[0, 20]
# fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.imshow(np.abs(probe_plot),extent=[-40,40,-40,40])
ax1.plot(zer+1,x1,color = "red")
ax1.plot(x1,zer,color = "red")
ax2.imshow(np.abs(probe_plot),extent=[0,80,80,0])
ax2.plot(zer+1,x,color = "red")
ax2.plot(x,zer,color = "red")
ax2.invert_yaxis()
# ax3.imshow(np.abs(probe4d[25, 5]),extent=[0,80,80,0])
# ax3.invert_yaxis()
plt.show()