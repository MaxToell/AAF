# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 15:14:33 2022

@author: oleh.melnyk
"""

import cupy as cp
from scipy.sparse.linalg import eigsh

def align_objects(obj_tr,obj,mask):
    alpha = cp.exp( 1j *cp.angle(cp.sum(obj[:,:].conj()*obj_tr[:,:] * mask, axis = None)))
    obj_r = alpha * obj
    return obj_r

def eliminate_linear_ambiguity(obj,win, threshold = 10**-10):
    obj, win =  normalize_object_and_window(obj,win)
    
    idx = cp.abs(win) > threshold
    ph = cp.ones_like(win)
    ph[idx] = win[idx]
    
    fr1 = cp.angle(ph[1:,:] / ph[:(ph.shape[0]-1),:])
    usable1 = cp.maximum(idx[1:,:],idx[:(idx.shape[0]-1),:])
    b1 = cp.mean(fr1[usable1],axis = None)
    
    if cp.isnan(b1):
        b1 = 0
    
    fr2 = cp.angle(ph[:,1:] / ph[:,:(ph.shape[0]-1)])
    usable2 = cp.maximum(idx[:,1:],idx[:,:(idx.shape[0]-1)])
    b2 = cp.mean(fr2[usable2],axis = None)
    
    if cp.isnan(b2):
        b2 = 0
    
    gy,gx = cp.meshgrid(range(win.shape[0]), range(win.shape[1]))
    win_mode = cp.exp(-1j*(gx*b1 + gy*b2))
    win_r = win * win_mode 
    
    gy,gx = cp.meshgrid(range(obj.shape[0]), range(obj.shape[1]))
    obj_mode = cp.exp(1j*(gx*b1 + gy*b2))
    obj_r = obj * obj_mode
    
    return obj_r, win_r
    
def eliminate_linear_ambiguity2(obj,win, threshold = 10**-10):
    obj, win =  normalize_object_and_window(obj,win)
    
    idx = cp.abs(obj) > threshold
    ph = cp.ones_like(obj)
    ph[idx] = obj[idx]
    
    fr1 = cp.angle(ph[1:,:] / ph[:(ph.shape[0]-1),:])
    usable1 = cp.maximum(idx[1:,:],idx[:(idx.shape[0]-1),:])
    b1 = cp.mean(fr1[usable1],axis = None)
    
    if cp.isnan(b1):
        b1 = 0
    
    fr2 = cp.angle(ph[:,1:] / ph[:,:(ph.shape[0]-1)])
    usable2 = cp.maximum(idx[:,1:],idx[:,:(idx.shape[0]-1)])
    b2 = cp.mean(fr2[usable2],axis = None)
    
    if cp.isnan(b2):
        b2 = 0
    
    gy,gx = cp.meshgrid(range(win.shape[0]), range(win.shape[1]))
    win_mode = cp.exp(1j*(gx*b1 + gy*b2))
    win_r = win * win_mode 
    
    gy,gx = cp.meshgrid(range(obj.shape[0]), range(obj.shape[1]))
    obj_mode = cp.exp(-1j*(gx*b1 + gy*b2))
    obj_r = obj * obj_mode
    
    return obj_r, win_r

def eliminate_grid_ambiguity_l1(obj, win, s, threshold = 10**-6, maxit= 100, eps = 10**-8):
    # mtype 'ls1' corresponds to constrained minimization of TV norm
    # selects lambda as a minimizaer of the total variation
    # via IRLS
    
    # lambd = np.ones((s,s),dtype = complex)/s**2
    lambd = cp.random.normal(size=(s,s)) + 1j * cp.random.normal(size=(s,s))
    lambd /= cp.linalg.norm(lambd,'fro')
    for t in range(maxit):    
        Z = cp.zeros((s,s,s,s), dtype = complex)
    
        for k1 in range(obj.shape[1]):
            idx2 = 0
            z2 = obj[idx2, k1]
            s1 = k1 % s
            for k0 in range(obj.shape[0]-1):
                idx1 = idx2
                idx2 = (k0+1) % s
                z1 = z2
                z2 = obj[k0+1,k1]        
                diff = 1.0/cp.maximum(cp.abs(z1 * lambd[idx1,s1] - z2 * lambd[idx2,s1]),eps)
                #print(diff)
                Z[idx1, s1, idx1, s1] += cp.abs(z1)**2 * diff
                Z[idx2, s1, idx2, s1] += cp.abs(z2)**2 * diff
                Z[idx1, s1, idx2, s1] -= z1.conj() * z2 * diff
                Z[idx2, s1, idx1, s1] -= z2.conj() * z1 * diff
            
        for k0 in range(obj.shape[0]):
            idy2 = 0
            z2 = obj[k0, idy2]
            s0 = k0 % s
            for k1 in range(obj.shape[1]-1):
                idy1 = idy2
                idy2 = (k1+1) % s
                z1 = z2
                z2 = obj[k0,k1+1]       
                diff = 1.0/cp.maximum(cp.abs(z1 * lambd[s0,idy1] - z2 * lambd[s0,idy2]),eps)
                Z[s0, idy1, s0, idy1] += cp.abs(z1)**2 * diff
                Z[s0, idy2, s0, idy2] += cp.abs(z2)**2 * diff
                Z[s0, idy1, s0, idy2] -= z1.conj() * z2 * diff
                Z[s0, idy2, s0, idy1] -= z2.conj() * z1 * diff

        Z = cp.reshape(Z, (s**2, s**2))
     
        lambd_long = cp.reshape(lambd, s**2)   
     
        try:
            sig,v = eigsh(Z,1,which = 'SM',v0 = lambd_long)
            lambd_new = v[:,0] 
            
        except cp.linalg.LinAlgError as err:
            if 'SVD did not converge in Linear Least Squares' in str(err):
                lambd_new = cp.ones(s**2,dtype = complex)/s**2
            else:
                raise
        
        
        dist = 2 - 2*cp.abs(lambd_new.dot(lambd_long.conj()))
        lambd_new = cp.reshape(lambd_new, (s,s))
        
        print(t,sig,dist)
        
        if (dist<threshold):
            lambd = lambd_new
            break
        lambd = lambd_new

    lambd_avg = cp.mean(cp.abs(lambd))
    lambd /= lambd_avg
    lambd[ cp.abs(lambd) < threshold] = 1
    
        
    lambd_obj = cp.tile(lambd, (obj.shape[0] // s, obj.shape[1] // s) )
    lambd_win = cp.tile(lambd, (win.shape[0] // s, win.shape[1] // s) )
    
    obj_r = obj * lambd_obj
    win_r = win / lambd_win
    
    return obj_r,win_r, lambd

def eliminate_grid_ambiguity_l2(obj, win, s, threshold = 10**-6):
    # mtype 'ls2' corresponds to constrained minimization of l2 norm of differences
    # selects lambda as a minimizaer of the squared total variation
    
    Z = cp.zeros((s,s,s,s), dtype = complex)
    
    for k1 in range(obj.shape[1]):
        idx2 = 0
        z2 = obj[idx2, k1]
        s1 = k1 % s
        for k0 in range(obj.shape[0]-1):
            idx1 = idx2
            idx2 = (k0+1) % s
            z1 = z2
            z2 = obj[k0+1,k1]        
            Z[idx1, s1, idx1, s1] += cp.abs(z1)**2
            Z[idx2, s1, idx2, s1] += cp.abs(z2)**2
            Z[idx1, s1, idx2, s1] -= z1.conj() * z2
            Z[idx2, s1, idx1, s1] -= z2.conj() * z1
            
    for k0 in range(obj.shape[0]):
        idy2 = 0
        z2 = obj[k0, idy2]
        s0 = k0 % s
        for k1 in range(obj.shape[1]-1):
            idy1 = idy2
            idy2 = (k1+1) % s
            z1 = z2
            z2 = obj[k0,k1+1]        
            Z[s0, idy1, s0, idy1] += cp.abs(z1)**2
            Z[s0, idy2, s0, idy2] += cp.abs(z2)**2
            Z[s0, idy1, s0, idy2] -= z1.conj() * z2
            Z[s0, idy2, s0, idy1] -= z2.conj() * z1
    
    Z = cp.reshape(Z, (s**2, s**2))
    
    lambd = cp.ones(s**2,dtype = complex) 
    try:
        sig,v = eigsh(Z,1,which = 'SM',v0 = lambd/s)
        lambd = v[:,0] 
        
        lambd[ cp.abs(lambd) < threshold] = 1
        
    except cp.linalg.LinAlgError as err:
        if 'SVD did not converge in Linear Least Squares' in str(err):
            lambd = cp.ones(s**2,dtype = complex)
        else:
            raise
            
    lambd = cp.reshape(lambd, (s,s))
            
    
    lambd[ cp.abs(lambd) < threshold] = 1
    lambd_obj = cp.tile(lambd, (obj.shape[0] // s, obj.shape[1] // s) )
    lambd_win = cp.tile(lambd, (win.shape[0] // s, win.shape[1] // s) )
    
    obj_r = obj * lambd_obj
    win_r = win / lambd_win
    
    return obj_r,win_r, lambd


def relative_error(obj_tr,obj,mask):
    obj_r = align_objects(obj_tr,obj,mask)
    return cp.sqrt(cp.sum(cp.abs(obj_r - obj_tr)**2 * mask, axis = None))/cp.sqrt(cp.sum(cp.abs(obj_tr)**2 * mask, axis = None))

# def relative_measurement_error(b, meas_obj):
#     b = np.reshape(b,(np.prod(b.shape)))
#     meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
#     b = np.sqrt(np.maximum(b,0))
#     meas_obj = np.sqrt(meas_obj)
#     return np.linalg.norm(b-meas_obj)/np.linalg.norm(b)

# def relative_sq_measurement_error(b, meas_obj):
#     b = np.reshape(b,(np.prod(b.shape)))
#     meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
#     # b = np.sqrt(np.maximum(b,0))
#     # meas_obj = np.sqrt(meas_obj)
#     return np.linalg.norm(b-meas_obj)/np.linalg.norm(b)

# def log10_measurement_error(b, meas_obj):
#     b = np.reshape(b,(np.prod(b.shape)))
#     meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
#     b = np.sqrt(np.maximum(b,0))
#     meas_obj = np.sqrt(meas_obj)
#     return np.log10(np.linalg.norm(b-meas_obj))

# def relative_intensity_error(b, meas_obj):
#     b = np.reshape(b,(np.prod(b.shape)))
#     meas_obj = np.reshape(meas_obj,(np.prod(meas_obj.shape)))
#     return np.linalg.norm(b-meas_obj)/np.linalg.norm(b)

def normalize_window(window):
    window /= cp.linalg.norm(window,'fro')
        
    return window

def normalize_object_and_window(obj, window):
    norm = cp.linalg.norm(window,'fro')
    window /= norm
    obj *= norm
    
    return obj, window

