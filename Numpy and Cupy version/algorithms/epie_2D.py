# -*- coding: utf-8 -*-
"""
Created on Mon May 16 10:11:44 2022

@author: oleh.melnyk
"""

import numpy as np
# import forward_circ as forward_circ
# import forward_2D as forward
from af_2D import amplitude_flow
# import gradient_descent_solver as gds
# from af_2D import AG_params_object
# from af_2D import af_params_object 
# from af_2D import baf_params_object

import copy

import sys
sys.path.insert(1, '../model')
import utility_2D as util

class ePIE(amplitude_flow):
    def __init__(self, 
                 **kwargs):
                 # measurements, 
                 # afpar,
                 # sqb=np.array([]),
                 # # circ = False,
                 # alpha = 0.05,
                 # beta = 0.05,
                 # probabilities = 'uniform',
                 # track = False, 
                 # obj_tr = [],
                 # track_it = []):
        super().__init__(**kwargs)
        
        if 'alpha' in kwargs.keys():
            self.alpha = kwargs['alpha']
        else:
            self.alpha = 0.05
        
        if 'beta' in kwargs.keys():
            self.beta = kwargs['beta']
        else:
            self.beta = 0.05 
        
        self.AG_params.enable_AG = False
        self.learning_rate_type = 'value'
        self.learning_rate = 1.0
        
        if 'probabilities' in kwargs.keys():
            self.probabilities = kwargs['probabilities']
        else:
            self.probabilities = 'uniform'
            
        if self.probabilities == 'distance':
            self.r_last = np.random.randint(0,self.R)
        
    def fast_grad_AF_2D_pty(self,zv,forw,sqrt_meas):
        z = zv[:self.par.object_shape[0],:]
        v = zv[self.par.object_shape[0]:,:self.delta2]
        
        if self.probabilities == 'uniform': 
            r_sel = np.random.randint(0,self.R)
            mult = self.R
        elif self.probabilities == 'distance':
            dist = np.sum(np.abs(self.par.locations - self.par.locations[self.r_last,:])**2, axis = 1)
            dist[self.r_last] = np.max(dist)
            prob = np.exp(-dist/(2*np.sqrt(self.delta1*self.delta2)))
            prob = prob / np.sum(prob)
            r_sel = np.random.choice(a = self.R,size = 1,p = prob)[0]
            mult = 1.0 / prob[r_sel]
            # print(np.sum( np.abs(self.par.locations[r_sel] - self.par.locations[self.r_last])**2))
            self.r_last = r_sel
            
        loc = self.par.locations[r_sel]
        self.par.set_window(v)
        
        # z_r = np.roll(np.roll(z,-loc[0],axis =0), - loc[1],axis = 1)
        z_r = self.par.shift_vec(z,-loc)
        z_r = z_r[:self.delta1,:self.delta2]
        f_r = self.par.forward_2D_os(z_r)
        m_r = np.sqrt(np.abs(f_r)**2 + self.epsilon)
        # obj_r = np.sum(np.abs(m_r - self.sqb[:,r_sel])**2, axis = None)
         
        grad_z = self.fast_grad_AF_2D_os(f_r,m_r,self.sqb[:,:,r_sel])
        grad_z = mult * self.alpha *grad_z / (np.prod(self.par.fourier_dimension) * np.max( np.abs(v)**2, axis = None))
        
        self.par.set_window(z_r)
        grad_v = self.fast_grad_AF_2D_os(f_r,m_r,self.sqb[:,:,r_sel])
        grad_v = mult * self.beta * grad_v / (np.prod(self.par.fourier_dimension) * np.max( np.abs(self.par.window)**2, axis = None ))
        
        grad_full = np.zeros((self.par.object_shape[0] + self.delta1, self.par.object_shape[1]),dtype = complex)
        grad = np.zeros((self.par.object_shape[0],self.par.object_shape[1]),dtype = complex)
        grad[:self.delta1,:self.delta2] =  grad_z
        # grad = np.roll(np.roll(grad, loc[0], axis =0), loc[1], axis = 1)
        grad = self.par.shift_vec(grad,loc)
        grad_full[:self.par.object_shape[0],:] = grad
        grad_full[self.par.object_shape[0]:,:self.delta2] = grad_v
            
        return grad_full

    def obj_pty(self,zv, obj_offset):
        # z = zv[:self.par.object_shape]
        # v = zv[self.par.object_shape:]
        # self.par.window = v
        # l2_z, f, sqbx = self.objective_L2_PIE(z)
        # obj = 0 + l2_z
        # # obj += self.objective_pen_1D(z)
        # obj += obj_offset
        return 1, 1, [], []
    
    def run(self, z0, v0):
        zv0 = np.zeros((self.par.object_shape[0] + self.delta1,self.par.object_shape[1]),dtype = complex)
        zv0[:self.par.object_shape[0],:] = z0
        zv0[self.par.object_shape[0]:,:self.delta2] = v0
        
        zv, val =  super().run(zv0)
        
        return zv[:self.par.object_shape[0],:], zv[self.par.object_shape[0]:,:self.delta2]