# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 12:05:54 2022

@author: oleh.melnyk
"""

import numpy as np
import sys
sys.path.insert(1, '../model')
import utility_2D as util

import scipy 

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
        self.track_it = np.sort(track_it)
        self.tracked_objects = []
        self.mask = mask
        self.TV_proximal = TV_proximal
        self.TV_param = TV_param
        if len(mask) == 0:
            self.mask = np.ones((z_0.shape[0]))
            
        
    def const_rate(self,z,grad):
        zt = z - self.learn_rate*grad
        obj,l2_z,f, sqrt_meas = self.obj_f(zt)
        return zt,self.learn_rate, obj,l2_z, f, sqrt_meas

    def AG_condition(self,z,grad,objz_val):
        mu = self.learn_rate * self.AG_par.tau**-self.AG_par.AG_iterations
        ngrad = np.sum(np.abs(grad)**2, axis = None)
                
        for n in range(self.AG_par.AG_iterations):
            zt = z - mu*grad 
            obj_zt,l2_z, f, sqrt_meas = self.obj_f(zt) 
            if (obj_zt - objz_val <= -2*self.AG_par.control*mu*ngrad):
                # print('AG',np.log10(objz_val - obj_zt), np.log10(2*AG_par.control*mu*ngrad))
                # print('AG',np.log10(objz_val),np.log10(obj_zt))
                return zt,mu,obj_zt,l2_z, f, sqrt_meas
            mu *= self.AG_par.tau
        
        # zt, mu, obj_zt = const_rate(z,sqrt_meas,grad,learn_rate,fast_obj_f)
        
        # print('C',np.log10(obj_f(zt)), np.log10(fast_obj_f(zt,sqrt_meas)))
        
        return self.const_rate(z,grad)

    def Av(self,v,wx,wx_norm,wy,wy_norm,shape):
        scaling = 1.0/np.max([1,wx_norm,wy_norm])#(np.sqrt(v.shape[0]) * np.max([1,wx_norm,wy_norm]))
        v_sq = np.reshape(v, shape)
        
        # pr = np.zeros((3*v.shape[0] - z.shape[0] - z.shape[1]),dtype = complex)
        pr = np.zeros((3*v.shape[0]),dtype = complex)
        pr[0:v.shape[0]] = v
        
        # diff_zx = np.maximum(np.abs(z[1:,:] - z[:(z.shape[0]-1),:]),eps)
        diff_vx = v_sq - np.roll(v_sq,1,axis = 0)
        # diff_vx = v_sq[1:,:] - v_sq[:(z.shape[0]-1),:]
        diff_vx *= wx
        # pr[v.shape[0]:(2*v.shape[0] - z.shape[1])] = diff_vx
        pr[v.shape[0]:(2*v.shape[0])] = np.reshape(diff_vx, v.shape)
        
        
        diff_vy = v_sq - np.roll(v_sq,1,axis = 1)
        # diff_vy = v_sq[:,1:] - v_sq[:,:(z.shape[0]-1)]
        diff_vy *= wy
        # pr[(2*v.shape[0] - z.shape[1]):] = diff_vy
        pr[(2*v.shape[0]):] = np.reshape(diff_vy, v.shape)
        
        return pr*scaling

    def AHv(self,v,wx,wx_norm,wy,wy_norm,shape):
        scaling = 1.0/np.max([1,wx_norm,wy_norm])#(np.sqrt(v.shape[0]) * np.max([1,wx_norm,wy_norm]))
        dim = np.prod(shape)
        pr = v[0:dim]
        
        v2 = v[dim:(2*dim)]
        v2_sq = np.reshape(v2, shape)
        v2_sq *= wx
        diff_vx = v2_sq - np.roll(v2_sq,-1,axis = 0)
        pr += np.reshape(diff_vx, (dim))
        
        v3 = v[(2*dim):]
        v3_sq = np.reshape(v3, shape)
        v3_sq *= wy
        diff_vy = v3_sq - np.roll(v3_sq,-1,axis = 1)
        pr += np.reshape(diff_vy, (dim))
        
        return pr*scaling

    def TV_proximal_step_l1(self, z0, lr, maxit = 100, tol = 10**-5, eps = 10**-8):
        dim  = np.prod(z0.shape)
        z_flat = np.reshape(z0,dim)
        b = np.zeros((3*dim), dtype = complex)
        b[:dim] = z_flat/np.sqrt(dim)
        
        for it in range(maxit):
            z = np.reshape(z_flat, z0.shape)
            diff_zx = np.maximum(np.abs(z - np.roll(z,1,axis = 0)),eps)
            wx = np.sqrt(self.TV_param)/np.sqrt(diff_zx)
            
            diff_zy = np.maximum(np.abs(z - np.roll(z, 1, axis = 1)),eps)
            wy = np.sqrt(self.TV_param)/np.sqrt(diff_zy)
            
            A_f = lambda v: self.Av(v,wx,wy,z0.shape)
            AH_f= lambda v: self.AHv(v,wx,wy,z0.shape) 
            
            A = scipy.sparse.linalg.LinearOperator((3*dim,dim), matvec=A_f, rmatvec = AH_f)
            mod = scipy.sparse.linalg.lsqr(A, b, atol=0, btol=0, conlim=0, iter_lim=100, x0 = z_flat)
            
            z_new = mod[0]
            dist = np.linalg.norm(z_new - z_flat)/np.linalg.norm(z_flat)
            if (dist < tol*self.TV_param**-1):
                z_flat = z_new
                
                break
            
            # print(it,np.linalg.norm(z_new - z_flat))
            z_flat = z_new
        
        if self.verbose:
            print('Proximal mapping:', it, dist)
        z_new = np.reshape(z_flat, z0.shape)
        obj_zt,l2_z, f, sqrt_meas = self.obj_f(z_new)
        
        return z_new, obj_zt,l2_z, f, sqrt_meas
    
    def TV_proximal_step_l2(self, z0, lr, maxit = 100, tol = 10**-5, eps = 10**-8):
        dim  = np.prod(z0.shape)
        z_flat = np.reshape(z0,dim)
        b = np.zeros((3*dim), dtype = complex)
        b[:dim] = z_flat#/np.sqrt(dim)
        
        for it in range(maxit):
            z = np.reshape(z_flat, z0.shape)
            # diff_zx = np.linalg.norm(z - np.roll(z,1,axis = 0),'fro')**2
            # diff_zy = np.linalg.norm(z - np.roll(z, 1, axis = 1),'fro')**2
            diff_zx = np.abs(z - np.roll(z,1,axis = 0))**2
            diff_zy = np.abs(z - np.roll(z, 1, axis = 1))**2
            diff = np.maximum(np.sqrt(diff_zx + diff_zy),eps)
            w = np.sqrt(self.TV_param)/np.sqrt(diff)
            w_norm = np.linalg.norm(w,'fro')
            
            A_f = lambda v: self.Av(v,w,w_norm,w,w_norm,z0.shape)
            AH_f= lambda v: self.AHv(v,w,w_norm,w,w_norm,z0.shape) 
            
            scaling = 1.0/np.max([1,w_norm]) # (np.sqrt(z_flat.shape[0]))
            b_sc = b * scaling
            
            A = scipy.sparse.linalg.LinearOperator((3*dim,dim), matvec=A_f, rmatvec = AH_f)
            mod = scipy.sparse.linalg.lsqr(A, 
                                           b_sc, 
                                           atol=0, 
                                           btol=0, 
                                           conlim=10**20, 
                                           iter_lim=1000, 
                                           x0 = z_flat)
            
            z_new = mod[0]
            dist = np.linalg.norm(z_new - z_flat)/np.linalg.norm(z_flat)
            if (dist < tol):#*self.TV_param**-1):
                z_flat = z_new
                
                break
            
            # print(it,np.linalg.norm(z_new - z_flat))
            z_flat = z_new
        
        if self.verbose:
            print('Proximal mapping:', it, dist)
        z_new = np.reshape(z_flat, z0.shape)
        obj_zt,l2_z, f, sqrt_meas = self.obj_f(z_new)
        
        return z_new, obj_zt,l2_z, f, sqrt_meas

    def run(self): 
        z = self.z_0
        obj_z, l2_z ,f,sqrt_meas = self.obj_f(z)
        # print(sqrt_meas[0:10])
        
        if self.track:
            self.measurement_error = np.array(l2_z)
            self.objective = np.array(obj_z)
            self.object_error = np.array(util.relative_error(self.obj_tr,z,self.mask))
        
        if len(self.track_it) > 0:
            self.tracked_objects = np.zeros((0,z.shape[0],z.shape[1]),dtype = complex)
        
        count = 0
        
        if self.AG_par.enable_AG:
            lr_f = lambda _z, _objz_val, _grad: self.AG_condition(_z,_grad,_objz_val)
        else:
            lr_f = lambda _z, _objz_val, _grad: self.const_rate(_z,_grad)
                
        for it in range(self.maxit):
            grad = self.grad_f(z,f,sqrt_meas)
            # print(np.sqrt(np.einsum('ijk->',np.abs(grad)**2)))
            
            if self.learn_rate_decay_rate != 0.0 and it != 0:
                self.learn_rate *= (it / (it + 1))**self.learn_rate_decay_rate
            
            z_old = z
            z, lr, obj_z,l2_z,f,sqrt_meas = lr_f(z,obj_z,grad)
            
            
            if self.TV_proximal:
                z, obj_z,l2_z,f,sqrt_meas = self.TV_proximal_step_l2(z,lr)
                # z, obj_z,l2_z,f,sqrt_meas = self.TV_proximal_step_l1(z,lr)
                step_size = np.linalg.norm(z_old - z, 'fro')
            else:
                step_size = np.sqrt(np.sum(np.abs(lr*grad)**2,axis = None))
                
            # beta = (1 + it)/(3 + it)
            # mem = z_new + beta*(z_new - mem)
            
            
            if np.isnan(step_size):
                return
            
            # z = z_new
            if self.verbose==1 :
                print(it,': LO:', np.log10(obj_z), 'ST:', step_size)
    #            print(it,': LO:', np.log10(obj_f(z_new)), 'LP:', np.log10(10**8*np.einsum('ijk->',np.imag(z)**2)), 'ST:', step_size)
            
            if self.track:
                self.measurement_error = np.append(self.measurement_error,l2_z)
                self.objective = np.append(self.objective,obj_z)
                self.object_error = np.append(self.object_error,util.relative_error(self.obj_tr,z,self.mask))
                
            if count < len(self.track_it):
                if it == self.track_it[count]:
                    self.tracked_objects = np.append(self.tracked_objects,[util.align_objects(self.obj_tr,z,self.mask)],axis= 0)        
                    count += 1
                    
            if step_size < self.threshold:
                break;
    
        return z, obj_z