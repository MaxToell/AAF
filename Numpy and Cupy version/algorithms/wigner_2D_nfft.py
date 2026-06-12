# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:55:27 2024

@author: oleh.melnyk
"""
import numpy as np
# from scipy.sparse.linalg import eigsh

from scipy.sparse.linalg import lsqr


import sys
sys.path.insert(1, '../model')
# import forward as forward
# import utility_2D as util

# from recordtype import recordtype
import copy
import wigner_2D
import pynfft

class wigner_2D_nfft(wigner_2D.wdd):
    
    def __init__(self, 
                 measurements, 
                 **kwargs
                 ):
        super().__init__(measurements,**kwargs)
        
        self.nfft_rel_err_thr = 10**-6
        if 'nfft_rel_err_thr' in kwargs.keys():
            if not isinstance(kwargs['nfft_rel_err_thr'], float):
                print(' is given, but not a float. Set to default value 10**-3.')
            else:
                self.nfft_rel_err_thr = kwargs['nfft_rel_err_thr']
        
        self.nfft_max_it = 500
        if 'nfft_max_it' in kwargs.keys():
            if not isinstance(kwargs[''], int):
                print('nfft_max_it is given, but not a int. Set to default value 500.')
            else:
                self.nfft_max_it = kwargs['nfft_max_it']
        
        self.nfft_dfft_count = 400
        if 'nfft_dfft_count' in kwargs.keys():
            if not isinstance(kwargs['nfft_dfft_count'], int):
                print('nfft_dfft_count is given, but not a int. Set to default value  400.')
            else:
                self.nfft_dfft_count = kwargs['nfft_dfft_count']
        
        self.band_limit = np.minimum( (0.5*(np.floor(np.sqrt(len(self.par.locations))-2))).astype(int), self.par.window_shape)
        self.band_limit = self.band_limit // 2
        print('Band Limit: ', self.band_limit)
        # self.band_limit = self.band_limit // 2
        # self.band_limit = (4,4)
        self.dim_c_ext = self.par.object_shape
        self.dim_c = self.par.object_shape
        self.wnd_size_c = self.par.window_shape
        self.shift = (1,1)
        
    def compute_singular_values(self):
        win_band = (2*self.wnd_size_c[0]-1,2*self.wnd_size_c[1]-1)
        
        self.singular_values = np.zeros((self.dim_c_ext[0],self.dim_c_ext[1],win_band[0],win_band[1]),dtype = complex)
        
        for k0 in range(win_band[0]):
            # positive second index 
            for k1 in range(win_band[1]):
                w = np.zeros(self.dim_c_ext,dtype = complex)
                w1 = np.zeros((2*self.par.window_shape[0],2*self.par.window_shape[0]),dtype = complex)
                w1[:self.par.window_shape[0], :self.par.window_shape[1]] = self.par.window
                w1 = np.roll(np.roll(w1,k0 - self.wnd_size_c[0] + 1,axis = 0),k1- self.wnd_size_c[1] + 1,axis = 1)
                w[:self.par.window_shape[0], :self.par.window_shape[1]] = w1[:self.par.window_shape[0], :self.par.window_shape[1]]*self.par.window.conj()
                # w2 = np.zeros_like(self.par.window,dtype = complex)
                # w[:self.par.window_shape[0], :self.par.window_shape[1]] = w2
                
                w_fft = np.fft.fft2(w) #np.fft.fft2(w2.conj())
                w_fft = w_fft.conj()    
                
                self.singular_values[:,:,k0, k1] = w_fft
        
        # Tikhonov regularization
        # idx = np.abs(self.singular_values) > 1e-12
        # self.singular_values[idx] += 1e-8 * self.singular_values[idx] /np.abs(self.singular_values[idx])**2  
        # self.singular_values[~idx] += 1e-3
        
             
        if self.reg_type == 'value':
             self.reg_param = self.reg_threshold
             # print( np.sum(np.abs(self.singular_values) < self.reg_param )/np.prod(self.singular_values.shape))
        elif self.reg_type == 'percent':
            self.reg_param = np.quantile(np.abs(self.singular_values),self.reg_threshold, axis = None)     
       
        self.above_threshold = np.abs(self.singular_values) > self.reg_param
        # self.above_threshold_neg = np.abs(self.singular_values_neg) > self.reg_param
        
    def construct_diagonals_fft(self):
        h1 = self.par.fourier_dimension[0] // 2 
        h2 = self.par.fourier_dimension[1] // 2
        
        print('Inversion...')
        b_shifted = np.zeros_like(self.b,dtype = complex)
        b_shifted = np.roll(self.b, -h2, axis=1)    
        b_shifted = np.roll(b_shifted, -h1, axis=0)
        
        totalshifts = len(self.par.locations)
        b_ifft = np.zeros_like(self.b,dtype = complex)
        for s in range(totalshifts):
            b_ifft[:,:,s] = np.fft.ifft2(b_shifted[:,:,s])
            # m[:,:,s] = np.fft.fft2(m_iift[:,:,s])
        
        # For each frequency divisible by shift recover compressed diagonal
        
        band = (2*self.band_limit[0]-1, 2*self.band_limit[1]-1)
        win_band = (2*self.wnd_size_c[0]-1,2*self.wnd_size_c[1]-1)
        
        self.diags_fft = np.zeros( (win_band[0],win_band[1],band[0],band[1]), dtype = complex)
        above_threshold_actual = np.zeros( (win_band[0],win_band[1],band[0],band[1]))
        # We can reconstruct more diagonals!
        # self.diags_fft = np.zeros( (2*self.wnd_size_c[0]-1,2*self.wnd_size_c[1]-1, band[0],band[1]), dtype = complex) 
        
        b_ifft = np.roll(np.roll(b_ifft, self.wnd_size_c[0] -1, 0), self.wnd_size_c[1]-1,1) 
        
        for k0 in range(win_band[0]):
            print('Progress: ', k0+1, ' out of ',win_band[0])
            for k1 in range(win_band[1]):
                     
                # Apply location-wise Fourier transform
                b_cur = b_ifft[k0,k1,:]
                # b_cur_r = np.reshape(b_cur, self.xt.shape)
                
                # # SANITY CHECK 1
                xq = self.xt * np.roll(np.roll(self.xt,k0- self.wnd_size_c[0] +1,0),k1- self.wnd_size_c[1] +1,1).conj()
                xq = np.fft.fft2(xq)
                # axq = np.abs(xq)
                
        
                pr = xq * self.singular_values[:,:,k0,k1] * np.prod(self.par.object_shape)
                # # idx = np.ones(pr.shape[0])
                # # idx[14:17] = 0
                # # idx = idx == 1
                # # pr_short = pr[idx,:]
                # # pr_short = pr_short[:,idx]
                pr_f = np.fft.ifft2(pr) 
                
                
                # print(np.sum(np.abs(pr_f.flatten() - b_cur*np.prod(self.par.object_shape))))
                # xqp = copy.deepcopy(xq)
                
                # SANITY CHECK 2
                # xf = np.fft.fft2(self.xt)
                # xvec = np.zeros_like(xf)
                # xvec_premod = np.zeros_like(xf)
                # for j0 in range(xf.shape[0]):
                #     for j1 in range(xf.shape[1]):
                #         xfp = xf * np.roll(np.roll(xf,j0,0),j1,1).conj()
                #         xfpf = np.fft.fft2(xfp)
                #         xfpf = np.roll(np.roll(xfpf, self.wnd_size_c[0] -1, 0), self.wnd_size_c[1]-1,1) 
                #         xvec_premod[j0,j1] = xfpf[win_band[0]-1 - k0,win_band[1]-1- k1]
                #         xvec[j0,j1] = xfpf[win_band[0]-1 - k0,win_band[1]-1- k1] * np.exp(-2j*np.pi*( j0*(k0- self.wnd_size_c[0] +1)/xf.shape[0] + j1*(k1- self.wnd_size_c[1] +1)/xf.shape[1] ) )
                #         # xqp[j0,j1] *= np.exp(2j*np.pi*( j0*(k0- self.band_limit[0] +1)/xf.shape[0] + j1*(k1- self.band_limit[1] +1)/xf.shape[1] ) )
                        
                # xvec /= np.prod(xf.shape)
                # axvec = np.abs(xvec)
                # print(np.sum(np.abs(xq - xvec)))
                
                x = copy.deepcopy(self.par.locations)
                # x[:,0] = (x[:,0] - (k0- self.band_limit[0] +1))* band[0] % self.par.object_shape[0]
                # x[:,1] = (x[:,1] - (k1- self.band_limit[0] +1)) * band[1] % self.par.object_shape[1]
                x = x.astype('float64')
                x[:,0] = - x[:,0] / self.dim_c_ext[0]
                x[:,1] = - x[:,1] / self.dim_c_ext[1]
                
                # x[:,0] *= -1.0/self.par.object_shape[0]
                # x[:,1] *= -1.0/self.par.object_shape[1]
                x[x > 0.5] -= 1 
                x[x < -0.5] += 1
                
                if np.prod(band) <= self.nfft_dfft_count:
                    # double_band = (4*self.band_limit[0]-3, 4*self.band_limit[1]-3)
                    # h = (2*self.band_limit[0]-2, 2*self.band_limit[1]-2)
                    # idx = np.zeros((np.prod(double_band),2))
                    # idx[:,0] = np.repeat(np.arange(double_band[0]) - h[0],double_band[1])
                    # idx[:,1] = np.tile(np.arange(double_band[1]) - h[1],double_band[0])
                    
                    
                    # Partially working old 
                    idx = np.zeros((np.prod(band),2))
                    idx[:,0] = np.repeat(np.arange(band[0]) - band[0]//2,band[1])
                    idx[:,1] = np.tile(np.arange(band[1]) - band[1]//2,band[0])
                    
                    # matrix = np.zeros((x.shape[0], np.prod(size)),dtype = complex)
                    matrix = np.exp( - 2j * np.pi * np.einsum('ik,jk -> ji',idx,x))
                    
                    # h = (size[0] // 2, size[1] // 2)
                    
                    
                    model = np.linalg.lstsq(matrix, b_cur * np.prod(self.dim_c_ext)**2)
                    # model = lsqr(matrix, b_cur * np.prod(self.dim_c_ext)**2, btol = self.nfft_rel_err_thr)
                    adj = model[0]
                    adj = np.reshape(adj,band)
                    
                    # diag = np.zeros(self.dim_c_ext,dtype = complex)
                    # diag[:sol.shape[0],:sol.shape[1]] = sol
                    # h = (size[0] // 2, size[1] // 2)
                    # diag = np.roll(np.roll(diag,-h[0],0),-h[1] + offset,1)
                    
                    # Partially working old 
                    h = ((band[0]) // 2,(band[1]) // 2)
                    # h = (band[0],band[1])
                
                    xfp_s = np.roll(np.roll(pr,h[0],0),h[1],1)
                    xfp_ss = xfp_s[:band[0],:band[1]]
                    # Partially working old 
                    # xfp_s = xfp_s[:band[0],:band[1]]
                    xfp_sf = xfp_ss.flatten()
                    test = matrix.dot(xfp_sf) 
                    t = b_cur * np.prod(self.dim_c_ext)**2
                    # diag_fft_check = np.reshape(test,diag_fft.shape)
                    
                    print('Diag fft: ', np.sqrt(np.sum(np.abs(b_cur * np.prod(self.dim_c_ext)**2 - test)**2) / np.sum(np.abs(test)**2) ))
                    print('Diag: ', np.sqrt(np.sum(np.abs(xfp_ss - adj)**2) / np.sum(np.abs(xfp_ss)**2) ))
                    
                    pr_comp = np.roll(np.roll(adj, -h[0],0),-h[1],1)
                    # t = 1
                    
                else:
                    # band is odd and nfft needs even dimension
                    plan = pynfft.nfft.NFFT([band[0]+1, band[1]+1],x.shape[0])
                    # plan = pynfft.nfft.NFFT([self.par.object_shape[0], self.par.object_shape[1]],x.shape[0])
                    plan.x = x
                    plan.precompute()
                    plan.f = b_cur * np.prod(self.dim_c_ext)
                    adj = plan.adjoint()
                    # RUN TRUE INFFT HERE
                    plan.f_hat = adj
                    diag_fft_check = plan.trafo()
                    # diag_fft_check = np.reshape(diag_fft_check,diag_fft.shape)
                    
                    # print(np.sum(np.abs(b_cur* np.prod(self.dim_c_ext)**2 - diag_fft_check)))
                    
                    infft = pynfft.solver.Solver(plan, flags=['CGNR'])
                    # w_hat = np.ones([band[0]+1, band[1]+1])
                    # w_hat[-1,:] = 10**-3
                    # w_hat[:,-1] = 10**-3
                    # infft.w_hat = w_hat
                    infft.y = b_cur * np.prod(self.dim_c_ext)**2
                    infft.f_hat_iter = adj
                    infft.before_loop()
                    print('Solving nfft: ',k0, ' ', k1)
                    
                    for iiter in range(self.nfft_max_it):
                        infft.loop_one_step()
                        
                        if iiter % 10 == 0:
                            sol = infft.f_hat_iter
                            plan.f_hat = sol
                            diag_fft_check = plan.trafo()
                            rel_err = np.linalg.norm(diag_fft_check - infft.y) / np.linalg.norm(infft.y)
                            
                            print('Relative err: ', rel_err)
                            
                            if rel_err < self.nfft_rel_err_thr:
                                break
                            
                            
                    adj = infft.f_hat_iter
                    
                    plan.f_hat = adj
                    diag_fft_check = plan.trafo()
                    # diag_fft_check = np.reshape(diag_fft_check,diag_fft.shape)
                    
                    # print(np.sum(np.abs(b_cur* np.prod(self.dim_c_ext)**2 - diag_fft_check)))
                    
                    h = ((band[0]+1) // 2,(band[1]+1) // 2)
                    pr_comp = np.roll(np.roll(adj, -h[0],0),-h[1],1)
                    pr_comp = np.delete(np.delete(pr_comp,self.band_limit[0],0),self.band_limit[1],1)
                    
                idx0 = np.arange(self.band_limit[0],self.dim_c_ext[0] - self.band_limit[0]+1)
                idx1 = np.arange(self.band_limit[1],self.dim_c_ext[1] - self.band_limit[1]+1)
                
                above_threshold = self.above_threshold[:,:,k0,k1]
                above_threshold = np.delete(np.delete(above_threshold, idx0,0), idx1,1)
                sing = self.singular_values[:,:,k0,k1]
                sing = np.delete(np.delete(sing, idx0,0), idx1,1)
                
                diags_fft = np.zeros(band,dtype = complex)
                diags_fft[above_threshold] = pr_comp[above_threshold] / sing[above_threshold]
                
                idx_f0 = np.concatenate((np.arange(self.band_limit[0]),np.arange(-self.band_limit[0]+1,0)))
                idx_f1 = np.concatenate((np.arange(self.band_limit[1]),np.arange(-self.band_limit[1]+1,0)))
                gx,gy = np.meshgrid(idx_f0, idx_f1,indexing='ij')
                
                mod_fac = np.exp(2j*np.pi*( gx*(k0- self.wnd_size_c[0] +1)/self.dim_c_ext[0] + gy*(k1- self.wnd_size_c[1] +1)/self.dim_c_ext[1] ) )
                diags_fft *= mod_fac
                
                # xvec_premod = np.delete(np.delete(xvec_premod, idx0,0), idx1,1)
                # print(np.sum(np.abs(diags_fft - xvec_premod)))
                
                self.diags_fft[win_band[0]-1 -k0,win_band[0]-1 -k1,:,:] = diags_fft
                above_threshold_actual[win_band[0]-1 -k0,win_band[0]-1 -k1,:,:] = above_threshold
                
                # diags_fft currently correspond to conxcS-kx
                # we adjust to make xcSkconx
                # gy,gx = np.meshgrid(range(self.dim_c_ext[0]), range(self.dim_c_ext[1]))
                # f_mode = np.exp(-2j*np.pi*(gx*k0*1.0/self.dim_c_ext[0] + gy*k1*1.0/self.dim_c_ext[1]))
                # self.diags_fft[:,:,k0,k1] *= f_mode
        
        self.diags_fft = np.roll(np.roll(self.diags_fft, self.band_limit[0] -1, 2),self.band_limit[1] -1,3)
        above_threshold_actual = np.roll(np.roll(above_threshold_actual, self.band_limit[0] -1, 2),self.band_limit[1] -1,3)
        self.diags = np.zeros( (self.dim_c_ext[0], self.dim_c_ext[1],band[0],band[1]), dtype = complex)
        
        for k0 in range(self.band_limit[0]):
        # for k0 in range(band[0]):
            print('Progress: ', k0+1, ' out of ',self.band_limit[0])
            for k1 in range(band[1]):        
                
                if (k0 == self.band_limit[0] - 1) and (k1 >= self.band_limit[1]):
                    continue
                
                diag_fft = self.diags_fft[:,:,k0,k1]
                diag_fft_flat = diag_fft.flatten()
                x_len = len(diag_fft_flat)
                
                diag_fft_2 = self.diags_fft[:,:,band[0]-1-k0,band[1]-1-k1]
                diag_fft_2 = np.flip(diag_fft_2,axis=(0,1))
                diag_fft_2_flat = diag_fft_2.flatten()
                
                # SANITY CHECK 3
                
                xf = np.fft.fft2(self.xt)
                xfp = xf * np.roll(np.roll(xf,k0- self.band_limit[0] + 1,0),k1- self.band_limit[1] + 1,1).conj()
                # xfp = xf * np.roll(np.roll(xf,k0,0),k1,1).conj()
                xfpf = np.fft.fft2(xfp)
                xfpf = np.roll(np.roll(xfpf, self.wnd_size_c[0] -1, 0), self.wnd_size_c[1]-1,1)
                xfpf_small = xfpf[:win_band[0],:win_band[1]]
                
                above_threshold = above_threshold_actual[:,:,k0,k1] == 1 
                above_threshold_2 = above_threshold_actual[:,:,band[0]-1-k0,band[1]-1-k1] == 1 
                
                print(np.sum(np.abs(diag_fft - xfpf_small)))
                print(np.sum(np.abs(diag_fft[above_threshold] - xfpf_small[above_threshold])))
                
                
                
                x = np.zeros((x_len,2))
                x[:,0] = np.repeat(np.arange(win_band[0]) - self.wnd_size_c[0] + 1,win_band[1])
                x[:,1] = np.tile(np.arange(win_band[1])- self.wnd_size_c[1] + 1,win_band[0])
                # x[:,0] = (x[:,0] - (k0- self.band_limit[0] +1))* band[0] % self.par.object_shape[0]
                # x[:,1] = (x[:,1] - (k1- self.band_limit[0] +1)) * band[1] % self.par.object_shape[1]
                x = x.astype('float64')
                x[:,0] = x[:,0] / self.dim_c_ext[0]
                x[:,1] = x[:,1] / self.dim_c_ext[1]
                
                # x[:,0] *= -1.0/self.par.object_shape[0]
                # x[:,1] *= -1.0/self.par.object_shape[1]
                x[x > 0.5] -= 1 
                x[x < -0.5] += 1
                
                mod_fac = np.exp(-2j*np.pi*( x[:,0]*(self.band_limit[0]-1-k0) + x[:,1]*(self.band_limit[1]-1-k1)) )
                diag_fft_2_flat  = (diag_fft_2_flat * mod_fac).conj()
                
                at_final = (above_threshold & above_threshold_2).flatten()
                x = x[at_final,:]
                
                # print('1 vs 2: ',np.linalg.norm(diag_fft_flat - diag_fft_2_flat))
                
                size = np.array([2*k0+1, 2*k1+1])
                if k1 >= self.band_limit[1]:
                    size[1] = 2*(band[1] - 1 - k1) + 1
                # size[size % 2 == 1] +=1
                # size = np.maximum(size, [7, 7])
                
                if np.prod(size) <= self.nfft_dfft_count:
                    if k1 >= self.band_limit[1]:
                        offset = k1 - self.band_limit[1] +1
                    else:
                        offset = 0  
                    
                    idx = np.zeros((np.prod(size),2))
                    idx[:,0] = np.repeat(np.arange(size[0]) - size[0]//2,size[1])
                    idx[:,1] = np.tile(np.arange(size[1]) - size[1]//2 + offset,size[0])
                    
                    # matrix = np.zeros((x.shape[0], np.prod(size)),dtype = complex)
                    matrix = np.exp( - 2j * np.pi * np.einsum('ik,jk -> ji',idx,x))
                    obs = 0.5*(diag_fft_flat + diag_fft_2_flat)[at_final]
                     
                    
                    
                    # h = (size[0] // 2, size[1] // 2)
                    
                    
                    model = np.linalg.lstsq(matrix, obs)
                    # model = lsqr(matrix, 0.5*(diag_fft_flat + diag_fft_2_flat), btol = self.nfft_rel_err_thr)
                    sol = model[0]
                    sol = np.reshape(sol,size)
                                       
                    diag = np.zeros(self.dim_c_ext,dtype = complex)
                    diag[:sol.shape[0],:sol.shape[1]] = sol
                    h = (size[0] // 2, size[1] // 2)
                    diag = np.roll(np.roll(diag,-h[0],0),-h[1] + offset,1)
                    
                    # h = ((band[0]+1) // 2,(band[1]+1) // 2)
                    
                      
                    
                    xfp_s = np.roll(np.roll(xfp,h[0],0),h[1]-offset,1)
                    xfp_s = xfp_s[:size[0],:size[1]]
                    xfp_sf = xfp_s.flatten()
                    diag_fft_check = matrix.dot(xfp_sf) 
                    # diag_fft_check = np.reshape(test,diag_fft.shape)
                    
                    
                    norm_fft = np.sqrt(np.sum(np.abs(diag_fft_check)**2))
                    if norm_fft > 1e-6:
                        print('Diag fft: ', np.sqrt(np.sum(np.abs(diag_fft_check - obs)**2))/norm_fft)
                    else:
                        print('Diag fft: (abs) ', np.sqrt(np.sum(np.abs(diag_fft_check - obs)**2)))
                        
                    norm_diag = np.sqrt(np.sum(np.abs(xfp)**2))
                    if norm_diag > 1e-6:
                        print('Diag: ', np.sqrt(np.sum(np.abs(xfp - diag)**2))/norm_diag)
                    else:
                        print('Diag: (abs) ', np.sqrt(np.sum(np.abs(xfp - diag)**2)))   
                    
                    # tttt = 1
                    
                else:
                    plan = pynfft.nfft.NFFT([band[0]+1, band[1]+1],x.shape[0])
                    # plan = pynfft.nfft.NFFT(size,x.shape[0])
                    plan.x = x
                    plan.precompute()
                    # plan.f = diag_fft_flat/ np.prod(self.dim_c_ext)
                    plan.f = 0.5*(diag_fft_flat + diag_fft_2_flat)
                    adj = plan.adjoint()
                    
                    infft = pynfft.solver.Solver(plan, flags=['CGNR'])
                    # w_hat = np.ones([band[0]+1, band[1]+1])
                    # w_hat[-1,:] = 10**-3
                    # w_hat[:,-1] = 10**-3
                    # infft.w_hat = w_hat
                    infft.y = 0.5*(diag_fft_flat + diag_fft_2_flat)
                    infft.f_hat_iter = adj
                    infft.before_loop()
                    print('Solving nfft: ',k0, ' ', k1)
                    for iiter in range(self.nfft_max_it):
                        infft.loop_one_step()
                        
                        if iiter % 10 == 0:
                            sol = infft.f_hat_iter
                            plan.f_hat = sol
                            diag_fft_check = plan.trafo()
                            rel_err = np.linalg.norm(diag_fft_check - infft.y) / np.linalg.norm(infft.y)
                            
                            # print('Relative err: ', rel_err)
                            
                            if rel_err < self.nfft_rel_err_thr:
                                break
                            
                    # RUN TRUE INFFT HERE
                    
                    sol = infft.f_hat_iter
                    
                    plan.f_hat = sol
                    diag_fft_check = plan.trafo()
                    diag_fft_check = np.reshape(diag_fft_check,diag_fft.shape)
                    
                    print(np.sum(np.abs(diag_fft - diag_fft_check)))
                    
                    diag = np.zeros(self.dim_c_ext,dtype = complex)
                    diag[:sol.shape[0],:sol.shape[1]] = sol
                    h = ((band[0]+1) // 2,(band[1]+1) // 2)
                    # h = (size[0] // 2, size[1] // 2)
                    diag = np.roll(np.roll(diag,-h[0],0),-h[1],1)
                    # print(np.sum(np.abs(diag - xfp)))
                    
                    xfp_s = np.roll(np.roll(xfp,h[0],0),h[1],1)
                    xfp_s = xfp_s[:sol.shape[0],:sol.shape[1]]
                    plan.f_hat = xfp_s
                    diag_fft_check = plan.trafo()
                    diag_fft_check = np.reshape(diag_fft_check,diag_fft.shape)
                    print('Diag fft: ', np.sqrt(np.sum(np.abs(diag_fft - diag_fft_check)**2)/np.sum(np.abs(diag_fft_check)**2)) )
                    print('Diag: ', np.sqrt(np.sum(np.abs(xfp - diag)**2) / np.sum(np.abs(diag)**2) ) )
                
                
                # diag_fft = np.zeros(self.dim_c_ext,dtype = complex)
                # diag_fft[:band[0],:band[1]] = self.diags_fft[:,:,k0,k1]
                # diag_fft = np.roll(np.roll(diag_fft,- self.band_limit[0] + 1,0),k1- self.band_limit[1] + 1,1)
                # diag = np.fft.ifft2(diag_fft)
                
                self.diags[:,:,k0,k1] = diag
                # self.diags[:,:,band[0]- k0-1,band[1]-1-k1] = diag
                self.diags[:,:,band[0]- k0-1,band[1]-1-k1] = np.roll(np.roll(diag,self.band_limit[0]-1-k0,axis = 0),self.band_limit[1]-1-k1,axis = 1).conj()
                # self.diags[:,:,k0,k1] = np.roll(np.roll(diag,self.band_limit[0]-1-k0,axis = 0),self.band_limit[1]-1-k1,axis = 1).conj()
                
        # self.diags2 = np.zeros( (self.dim_c_ext[0], self.dim_c_ext[1],band[0],band[1]), dtype = complex)
        # for k0 in range(self.band_limit[0]):
        # # for k0 in range(band[0]):
        #     print('Progress: ', k0+1, ' out of ',self.band_limit[0])
        #     for k1 in range(band[1]):        
                
        #         if (k0 == self.band_limit[0] - 1) and (k1 >= self.band_limit[1]):
        #             continue
                
        #         diag_fft = self.diags_fft[:,:,k0,k1]
        #         diag_fft_flat = diag_fft.flatten()
        #         x_len = len(diag_fft_flat)
                
        #         diag_fft_2 = self.diags_fft[:,:,band[0]-1-k0,band[1]-1-k1]
        #         diag_fft_2 = np.flip(diag_fft_2,axis=(0,1))
        #         diag_fft_2_flat = diag_fft_2.flatten()
                
        #         x = np.zeros((x_len,2))
        #         x[:,0] = np.repeat(np.arange(win_band[0]) - self.wnd_size_c[0] + 1,win_band[1])
        #         x[:,1] = np.tile(np.arange(win_band[1])- self.wnd_size_c[1] + 1,win_band[0])
        #         x = x.astype('float64')
        #         x[:,0] = x[:,0] / self.dim_c_ext[0]
        #         x[:,1] = x[:,1] / self.dim_c_ext[1]
                
        #         x[x > 0.5] -= 1 
        #         x[x < -0.5] += 1
                
        #         mod_fac = np.exp(-2j*np.pi*( x[:,0]*(self.band_limit[0]-1-k0) + x[:,1]*(self.band_limit[1]-1-k1)) )
        #         diag_fft_2_flat  = (diag_fft_2_flat * mod_fac).conj()
                
        #         plan = pynfft.nfft.NFFT([band[0]+1, band[1]+1],x.shape[0])
        #         plan.x = x
        #         plan.precompute()
        #         plan.f = 0.5*(diag_fft_flat + diag_fft_2_flat) / np.prod(self.dim_c_ext)
        #         adj = plan.adjoint()
                
        #         infft = pynfft.solver.Solver(plan, flags=['CGNR'])
        #         infft.y = 0.5*(diag_fft_flat + diag_fft_2_flat)
        #         infft.f_hat_iter = adj
        #         infft.before_loop()
        #         for iiter in range(100):
        #             infft.loop_one_step()
                
        #         sol = infft.f_hat_iter
                
        #         plan.f_hat = sol
        #         diag_fft_check = plan.trafo()
        #         diag_fft_check = np.reshape(diag_fft_check,diag_fft.shape)
                
        #         print(np.sum(np.abs(diag_fft - diag_fft_check)))
                
        #         diag = np.zeros(self.dim_c_ext,dtype = complex)
        #         diag[:sol.shape[0],:sol.shape[1]] = sol
        #         h = ((band[0]+1) // 2,(band[1]+1) // 2)
        #         diag = np.roll(np.roll(diag,-h[0],0),-h[1],1)
        #         # self.diags[:,:,k0,k1] = diag
        #         self.diags2[:,:,band[0]- k0-1,band[1]-1-k1] = diag
        #         self.diags2[:,:,k0,k1] = np.roll(np.roll(diag,self.band_limit[0]-1-k0,axis = 0),self.band_limit[1]-1-k1,axis = 1).conj()
                
        self.diags = np.flip(self.diags,axis=(2,3))        
        # self.obj_size_c = self.band_limit        
        self.wnd_size_c = self.band_limit
        
        # SANITY CHECK 4
        # t1 = self.diags[:,:,8,8]
        # t2 = self.diags[:,:,6,6]
        # np.sum(np.abs(np.roll(np.roll(t1,-1,axis = 1),-1,0) - t2.conj()))
        
        # self.diags_fft = np.roll(np.roll(self.diags_fft, self.band_limit[0] -1, 2),self.band_limit[1] -1,3)
        
        
        # self.diags_fft = np.zeros( (self.dim_c_ext[0], self.dim_c_ext[1],self.wnd_size_c[0],self.wnd_size_c[1]), dtype = complex)
        # self.diags_fft_neg = np.zeros( (self.dim_c_ext[0], self.dim_c_ext[1],self.wnd_size_c[0],self.wnd_size_c[1]-1), dtype = complex)
        # self.diags_fft = np.roll(np.roll(self.diags_fft, -self.band_limit[0] +1, 0), -self.band_limit[1]+1,1)
        
        
        # self.diags = np.fft.ifft2(self.diags_fft, axes = (0,1))
        
        
        # self.diags_fft_neg = self.diags_fft[self.band_limit[0]:,:(self.band_limit[1]-1)]
        # self.diags_fft_neg = np.flip(self.diags_fft_neg, axis = 3) 
        # self.diags_fft = self.diags_fft[self.band_limit[0]:,self.band_limit[1]:]
        
         
        
        # above_threshold_idx = np.abs(self.singular_values) > self.reg_param
        # self.diags_fft[~self.above_threshold] = 0
        # self.diags_fft_neg[~self.above_threshold_neg] = 0
        
    def construct_object(self):
        x = super().construct_object()

        return np.fft.ifft2(x)
        
    def run(self):
        self.compute_singular_values()
        self.construct_diagonals_fft()
        # self.truncate_diag_fft()
        
        # if self.subspace_completion:
            # self.subspace_completion()
            
        if self.memory_saving:
            # self.construct_diags()
            self.reconstruct_magnitudes_from_diags()
            self.angular_sync_from_diags()
        else:
            self.construct_lifted_matrix()
            self.reconstruct_magnitudes_from_lifted_matrix()
            self.angular_sync_from_lifted_matrix()
                

        # u_vec = np.arange(np.prod(self.dim_c_ext))
        # u = np.reshape(u_vec,self.dim_c_ext)
        
        # u = np.random.rand(self.dim_c_ext[0],self.dim_c_ext[1])
        # u_vec = np.reshape(u,(np.prod(self.dim_c_ext)))
        
        # e1 = self.x_lifted_comp[0,:] * u_vec
        
        # res1 = self.x_lifted_comp.dot(u_vec)
        # res2 = self.multiply_via_diag(self.diags, u)
        
        # self.reconstruct_magnitudes()
        
        return self.construct_object()