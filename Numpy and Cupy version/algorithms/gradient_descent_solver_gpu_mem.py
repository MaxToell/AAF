# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:03:39 2022

@author: oleh.melnyk
"""

import cupy as cp
# import numpy as np
import sys
sys.path.insert(1, '../model')
import utility_2D_gpu as util

from cupyx.scipy.sparse.linalg import LinearOperator
from cupyx.scipy.sparse.linalg import lsqr

class grad_desc:
    def __init__(self,
                 grad_f,
                 obj_f,
                 z_0, 
                 maxit, 
                 threshold, 
                 learn_rate,
                 AG_par,
                 learn_rate_decay_rate = 0.0,
                 TV_proximal = False,
                 TV_param = 0,
                 verbose = 0,
                 track = False,
                 obj_tr = [],
                 track_it = [],
                 mask = []):
        self.grad_f = grad_f
        self.obj_f = obj_f
        self.z_0 = z_0
        self.maxit = maxit
        self.threshold = threshold
        self.learn_rate = learn_rate
        self.AG_par = AG_par
        self.learn_rate_decay_rate = learn_rate_decay_rate
        self.verbose = verbose
        self.track = track
        self.obj_tr = obj_tr
        self.track_it = cp.sort(track_it)
        self.tracked_objects = []
        self.mask = mask
        self.TV_proximal = TV_proximal
        self.TV_param = TV_param
        if len(mask) == 0:
            self.mask = cp.ones((z_0.shape[0]))
            
        
    def const_rate(self,z,grad):
        zt = z - self.learn_rate*grad
        obj,l2_z = self.obj_f(zt)
        return zt,self.learn_rate, obj,l2_z

    def AG_condition(self,z,grad,objz_val):
        mu = self.learn_rate * self.AG_par.tau**-self.AG_par.AG_iterations
        ngrad = cp.sum(cp.abs(grad)**2, axis = None)
                
        for n in range(self.AG_par.AG_iterations):
            zt = z - mu*grad 
            obj_zt,l2_z = self.obj_f(zt) 
            if (obj_zt - objz_val <= -2*self.AG_par.control*mu*ngrad):
                # print('AG',np.log10(objz_val - obj_zt), np.log10(2*AG_par.control*mu*ngrad))
                # print('AG',np.log10(objz_val),np.log10(obj_zt))
                return zt,mu,obj_zt,l2_z
            mu *= self.AG_par.tau
        
        # zt, mu, obj_zt = const_rate(z,sqrt_meas,grad,learn_rate,fast_obj_f)
        
        # print('C',np.log10(obj_f(zt)), np.log10(fast_obj_f(zt,sqrt_meas)))
        
        return self.const_rate(z,grad)

    def Av(self,v,wx,wy,shape):
        v_sq = cp.reshape(v, shape)
        
        # pr = np.zeros((3*v.shape[0] - z.shape[0] - z.shape[1]),dtype = complex)
        pr = cp.zeros((3*v.shape[0]),dtype = complex)
        pr[0:v.shape[0]] = v
        
        # diff_zx = np.maximum(np.abs(z[1:,:] - z[:(z.shape[0]-1),:]),eps)
        diff_vx = v_sq - cp.roll(v_sq,1,axis = 0)
        # diff_vx = v_sq[1:,:] - v_sq[:(z.shape[0]-1),:]
        diff_vx *= wx
        # pr[v.shape[0]:(2*v.shape[0] - z.shape[1])] = diff_vx
        pr[v.shape[0]:(2*v.shape[0])] = cp.reshape(diff_vx, v.shape)
        
        
        diff_vy = v_sq - cp.roll(v_sq,1,axis = 1)
        # diff_vy = v_sq[:,1:] - v_sq[:,:(z.shape[0]-1)]
        diff_vy *= wy
        # pr[(2*v.shape[0] - z.shape[1]):] = diff_vy
        pr[(2*v.shape[0]):] = cp.reshape(diff_vy, v.shape)
        
        return pr

    def AHv(self,v,wx,wy,shape):
        dim = cp.prod(shape)
        pr = v[0:dim]
        
        v2 = v[dim:(2*dim)]
        v2_sq = cp.reshape(v2, shape)
        v2_sq *= wx
        diff_vx = v2_sq - cp.roll(v2_sq,-1,axis = 0)
        pr += cp.reshape(diff_vx, (dim))
        
        v3 = v[(2*dim):]
        v3_sq = cp.reshape(v3, shape)
        v3_sq *= wy
        diff_vy = v3_sq - cp.roll(v3_sq,-1,axis = 1)
        pr += cp.reshape(diff_vy, (dim))
        
        return pr

    def TV_proximal_step_l1(self, z0, lr, maxit = 100, tol = 10**-5, eps = 10**-8):
        dim  = cp.prod(z0.shape)
        z_flat = cp.reshape(z0,dim)
        b = cp.zeros((3*dim), dtype = complex)
        b[:dim] = z_flat
        
        for it in range(maxit):
            z = cp.reshape(z_flat, z0.shape)
            diff_zx = cp.maximum(cp.abs(z - cp.roll(z,1,axis = 0)),eps)
            wx = cp.sqrt(self.TV_param)/cp.sqrt(diff_zx)
            
            diff_zy = cp.maximum(cp.abs(z - cp.roll(z, 1, axis = 1)),eps)
            wy = cp.sqrt(self.TV_param)/cp.sqrt(diff_zy)
            
            A_f = lambda v: self.Av(v,wx,wy,z0.shape)
            AH_f= lambda v: self.AHv(v,wx,wy,z0.shape) 
            
            A = LinearOperator((3*dim,dim), matvec=A_f, rmatvec = AH_f)
            mod = lsqr(A, b, x0 = z_flat)
            
            z_new = mod[0]
            dist = cp.linalg.norm(z_new - z_flat)/cp.linalg.norm(z_flat)
            if (dist < tol):
                z_flat = z_new
                
                break
            
            # print(it,np.linalg.norm(z_new - z_flat))
            z_flat = z_new
        
        if self.verbose:
            print('Proximal mapping:', it, dist)
        z_new = cp.reshape(z_flat, z0.shape)
        obj_zt,l2_z, f, sqrt_meas = self.obj_f(z_new)
        
        return z_new, obj_zt,l2_z, f, sqrt_meas
    
    def TV_proximal_step_l2(self, z0, lr, maxit = 100, tol = 10**-5, eps = 10**-8):
        dim  = cp.prod(z0.shape)
        z_flat = cp.reshape(z0,dim)
        b = cp.zeros((3*dim), dtype = complex)
        b[:dim] = z_flat
        
        for it in range(maxit):
            z = cp.reshape(z_flat, z0.shape)
            diff_zx = cp.linalg.norm(z - cp.roll(z,1,axis = 0),'fro')**2
            diff_zy = cp.linalg.norm(z - cp.roll(z, 1, axis = 1),'fro')**2
            diff = cp.maximum(cp.sqrt(diff_zx + diff_zy),eps)
            w = cp.sqrt(self.TV_param)/cp.sqrt(diff)
            
            A_f = lambda v: self.Av(v,w,w,z0.shape)
            AH_f= lambda v: self.AHv(v,w,w,z0.shape) 
            
            A = LinearOperator((3*dim,dim), matvec=A_f, rmatvec = AH_f)
            mod = lsqr(A, b, x0 = z_flat)
            
            z_new = mod[0]
            dist = cp.linalg.norm(z_new - z_flat)/cp.linalg.norm(z_flat)
            if (dist < tol):
                z_flat = z_new
                
                break
            
            # print(it,np.linalg.norm(z_new - z_flat))
            z_flat = z_new
        
        if self.verbose:
            print('Proximal mapping:', it, dist)
        z_new = cp.reshape(z_flat, z0.shape)
        obj_zt,l2_z, f, sqrt_meas = self.obj_f(z_new)
        
        return z_new, obj_zt,l2_z, f, sqrt_meas

    def run(self): 
        z = self.z_0
        obj_z, l2_z = self.obj_f(z)
        # print(sqrt_meas[0:10])
        
        if self.track:
            self.measurement_error = cp.array(l2_z)
            self.objective = cp.array(obj_z)
            self.object_error = cp.array(util.relative_error(self.obj_tr,z,self.mask))
        
        if len(self.track_it) > 0:
            self.tracked_objects = cp.zeros((0,z.shape[0],z.shape[1]),dtype = complex)
        
        count = 0
        
        if self.AG_par.enable_AG:
            lr_f = lambda _z, _objz_val, _grad: self.AG_condition(_z,_grad,_objz_val)
        else:
            lr_f = lambda _z, _objz_val, _grad: self.const_rate(_z,_grad)
                
        for it in range(self.maxit):
            grad = self.grad_f(z)
            # print(np.sqrt(np.einsum('ijk->',np.abs(grad)**2)))
            
            if self.learn_rate_decay_rate != 0.0 and it != 0:
                self.learn_rate *= (it / (it + 1))**self.learn_rate_decay_rate
            
            z_old = z
            z, lr, obj_z,l2_z = lr_f(z,obj_z,grad)
            
            
            if self.TV_proximal:
                z, obj_z,l2_z = self.TV_proximal_step_l2(z,lr)
                step_size = cp.linalg.norm(z_old - z, 'fro')
            else:
                step_size = cp.sqrt(cp.sum(cp.abs(lr*grad)**2,axis = None))
                
            # beta = (1 + it)/(3 + it)
            # mem = z_new + beta*(z_new - mem)
            
            
            if cp.isnan(step_size):
                return
            
            # z = z_new
            if self.verbose==1 :
                print(it,': LO:', cp.log10(obj_z), 'ST:', step_size)
    #            print(it,': LO:', np.log10(obj_f(z_new)), 'LP:', np.log10(10**8*np.einsum('ijk->',np.imag(z)**2)), 'ST:', step_size)
            
            if self.track:
                self.measurement_error = cp.append(self.measurement_error,l2_z)
                self.objective = cp.append(self.objective,obj_z)
                self.object_error = cp.append(self.object_error,util.relative_error(self.obj_tr,z,self.mask))
                
            if count < len(self.track_it):
                if it == self.track_it[count]:
                    self.tracked_objects = cp.append(self.tracked_objects,[util.align_objects(self.obj_tr,z,self.mask)],axis= 0)        
                    count += 1
                    
            if step_size < self.threshold:
                break;
    
        return z, obj_z