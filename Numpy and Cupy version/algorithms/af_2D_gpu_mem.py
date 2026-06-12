# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 14:44:41 2022

@author: oleh.melnyk
"""

import cupy as cp

# from scipy.sparse.linalg import eigsh
# import multiprocessing as mp
# from multiprocessing.pool import ThreadPool

import sys
sys.path.insert(1, '../model')
import forward_gpu_mem as forward
import gradient_descent_solver_gpu_mem as gds
import utility_2D_gpu as util

# from recordtype import recordtype
import types
import copy

######################## MULTIPROCESING/MULTITHREADING TRIAL 2
# def fast_grad_AF_2D_os(tup):
#     par,forw,sqrt_meas,sqb,loc = tup
#     grad_r = np.zeros(par.object_shape,dtype = complex)
       
#     # print(loc)
    
#     diff = 1 - sqb/sqrt_meas 
#     t = forw * diff 
#     grad_r[:par.window_shape[0],:par.window_shape[1]] = par.forward_adj_2D_os(t)
#     grad_r = np.roll(grad_r, loc[0], axis =0)
#     grad_r = np.roll(grad_r, loc[1], axis =1)
    
#     return grad_r


######################## MULTIPROCESING TRIAL 1

# def fast_grad_AF_2D_os(par,forw,sqrt_meas,sqb):    
#     diff = 1 - sqb/sqrt_meas 
#     t = forw * diff 
#     grad2 = par.forward_adj_2D_os(t)
    
#     return grad2

# def fast_grad_AF_2D_os_loc(tup):
#     # tup = (par,forw,sqrt_meas,sqb,r_start, r_end)
#     par,forw,sqrt_meas,sqb,r_start, r_end = tup
    
#     grad = np.zeros(par.object_shape,dtype = complex)
#     for r in np.arange(r_start,r_end):
#         # print ('Start: ', r, ' ')
        
#         # sys.stdout.flush()

#         loc = par.locations[r,:]
#         grad_r = np.zeros(par.object_shape,dtype = complex)
        
#         # diff = 1 - sqb[:,:,r-r_start]/sqrt_meas[:,:,r-r_start] 
#         # t = forw[:,:,r-r_start] * diff 
#         # grad_r[:par.window_shape[0],:par.window_shape[1]] = par.forward_adj_2D_os(t)
        
#         grad_r[:par.window_shape[0],:par.window_shape[1]] = fast_grad_AF_2D_os(par,forw[:,:,r-r_start],sqrt_meas[:,:,r-r_start],sqb[:,:,r-r_start])
#         grad_r = np.roll(grad_r, loc[0], axis =0)
#         grad_r = np.roll(grad_r, loc[1], axis =1)
#         grad += grad_r
#         # print ('End: ', r, ' ')

#     return grad

# def AG_params_object():
#     return recordtype("AG_params", "enable_AG, control, tau, AG_iterations")

# def af_params_object():
#     AG_par = AG_params_object()
#     af_params = recordtype("af_params", "ptycho_params, AG_params, number_of_iterations, grad_threshold, learning_rate_type, learning_rate, epsilon, alpha_T, alpha_R, verbose")
#     return af_params,AG_par

# def baf_params_object():
#     AG_par = AG_params_object()
#     baf_params = recordtype("baf_params", "ptycho_params, number_of_iterations, object_subiterations, window_subiterations, grad_thr_object, grad_thr_window, AGP_object, AGP_window, epsilon, alpha_T, beta_T, alpha_R, beta_R, verbose")
#     return baf_params,AG_par

class amplitude_flow:
    def __init__(self,
                 **kwargs):
                 # measurements, 
                 # afpar,
                 # sqb=np.array([]),
                 # track = False, 
                 # obj_tr = [],
                 # track_it = []):
        
        if 'epsilon' in kwargs.keys():
            if isinstance(kwargs['epsilon'], float) or isinstance(kwargs['epsilon'], int): 
                self.epsilon = kwargs['epsilon']
            else:
                print('epsilon is given but neither float or int. Set to 0.')
                self.epsilon = 0
        else:
            print('epsilon is not given. Set to 0.')
            self.epsilon = 0
        
        
        assert 'measurements' in kwargs.keys() or 'sqb' in kwargs.keys(), "Neither measurements nor their square root is given"   
        
        if 'measurements' in kwargs.keys():
            self.sqb = cp.sqrt(cp.maximum(kwargs['measurements'],0) + self.epsilon)    
        
        if 'sqb' in kwargs.keys():
            self.sqb = kwargs['sqb']
            
        assert 'ptycho' in kwargs.keys(), "Forward model of ptycho class is not given."
        assert isinstance(kwargs['ptycho'], forward.ptycho), "ptycho is not an instance of class ptycho."
        self.par = kwargs['ptycho'].copy()    
        self.R = len(self.par.locations)
        self.delta1 = self.par.window.shape[0]
        self.delta2 = self.par.window.shape[1]
        
        if 'AG_params' in kwargs.keys():
            self.AG_params = copy.deepcopy(kwargs['AG_params'])
        else:
            self.AG_params = types.SimpleNamespace(enable_AG=False,
                                control=0.5,
                                tau = 0.3,
                                AG_iterations = 2)

        assert 'number_of_iterations' in kwargs.keys(), "number_of_iterations is not specified."
        assert isinstance(kwargs['number_of_iterations'], int), "number_of_iterations is not an integer"        
        self.number_of_iterations = kwargs['number_of_iterations']
        
        if 'grad_threshold' in kwargs.keys():
            self.grad_threshold = kwargs['grad_threshold']
        else:
            self.grad_threshold = 0
            
        if 'learning_rate_type' in kwargs.keys():
            self.learning_rate_type = kwargs['learning_rate_type']
        else:
            self.learning_rate_type = 'optimal'
            
        if 'learning_rate' in kwargs.keys():
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 1.0
            
        if 'learn_rate_decay' in kwargs.keys():
            self.learn_rate_decay = kwargs['learn_rate_decay']
        else:
            self.learn_rate_decay = 0.0
            
        if 'alpha_T' in kwargs.keys():
            self.alpha_T = kwargs['alpha_T']
        else:
            self.alpha_T = 0
        
        if 'alpha_R' in kwargs.keys():
            self.alpha_R = kwargs['alpha_R']
        else:
            self.alpha_R = 0
            
        if 'alpha_ST' in kwargs.keys():
            self.alpha_ST = kwargs['alpha_ST']
        else:
            self.alpha_ST = 0
            
        if 'proximal_TV' in kwargs.keys() and not ('alpha_TV' in kwargs.keys()):
            print('proximal_TV is given, but alpha_TV is not. Set to False')
            self.proximal_TV = False
        else:
            self.proximal_TV = False
            
        if 'alpha_TV' in kwargs.keys():
            self.alpha_TV = kwargs['alpha_TV']
            
            if 'proximal_TV' in kwargs.keys():
                self.proximal_TV = kwargs['proximal_TV']            
            else:
                self.proximal_TV = False
            
            if 'TV_param' in kwargs.keys():
                self.TV_param = kwargs['TV_param']
            else:
                if self.proximal_TV:
                    self.TV_param = 0
                else:
                    self.TV_param = 1  
        else:
            self.alpha_TV = 0
            self.TV_param = 0
        
        if 'verbose' in kwargs.keys():
            self.verbose = kwargs['verbose']
        else:
            self.verbose = 0
        
        if 'track' in kwargs.keys():
            self.track = kwargs['track']
        else:
            self.track = False
            
        if 'obj_tr' in kwargs.keys():
            # assert kwargs['obj_tr'].shape[0] == self.par.object_shape[0] and kwargs['obj_tr'].shape[1] == self.par.object_shape[1], "Shape of true object is different from par.object_shape."
            
            self.obj_tr = kwargs['obj_tr']  
        else:
            self.obj_tr = []
        
        if self.track:
            self.measurement_error = cp.zeros((0))
            self.objective = cp.zeros((0))
            self.object_error = cp.zeros((0))
            
            if 'track_it' in kwargs.keys():
                self.track_it = cp.sort(kwargs['track_it'])
            else:
                self.track_it = []
        else:
            self.track_it = []
            self.obj_tr = []
        
        
# class amplitude_flow:
#     def __init__(self,
#                  **kwargs):
#                  # measurements, 
#                  # afpar,
#                  # sqb=np.array([]),
#                  # track = False, 
#                  # obj_tr = [],
#                  # track_it = []):
        
#         assert 'measurements' in kwargs.keys() or 'sqb' in kwargs.keys(), 'Neither measurements nor their square root is given'        
        
#         if 'measurements' in kwargs.keys():
            
        
        
#         self.sqb = np.sqrt(np.maximum(measurements,0) + afpar.epsilon)
#         if (sqb.shape[0] != 0):
#             self.sqb = sqb
        
#         self.afpar = copy.deepcopy(afpar) 
#         self.par = self.afpar.ptycho_params.copy()
#         self.R = len(self.par.locations)
#         self.delta1 = self.par.window.shape[0]
#         self.delta2 = self.par.window.shape[1]
        
#         # if circ:
#         #     self.forward = lambda z,par: forward_circ.forward_2D_pty(z,par)
#         #     self.forward_adj = lambda t,par: forward_circ.forward_adj_2D_os(t,par)
#         # else:
#         #     self.forward = lambda z,par: forward.forward_2D_pty(z,par)
#         #     self.forward_adj = lambda t,par: forward.forward_adj_2D_os(t,par)
        
#         self.track = track
#         self.obj_tr = obj_tr
#         self.measurement_error = np.zeros((0))
#         self.objective = np.zeros((0))
#         self.object_error = np.zeros((0))
#         self.track_it = np.sort(track_it)        

        # if self.par.num_threads !=1:
            # self.pool = mp.Pool(self.par.num_threads)
    
    
    ######################## MULTIPROCESING/MULTITHREADING TRIAL 2
    # def fast_grad_L2_2D_pty(self,forw, sqrt_meas):
        
    #     # if self.par.num_threads == 1:
    #         # return fast_grad_AF_2D_os_loc((self.par.object_shape,forw,sqrt_meas,self.sqb,0, self.R))
            
    #     grad = np.zeros(self.par.object_shape,dtype = complex)
        
    #     # if __name__ == "__main__":
        
        
    #     # pool = mp.Pool(self.par.num_threads)
    #     pool = ThreadPool(self.par.num_threads)
    #     itr = []
    #     for r in range(self.R):
    #         itr.append(
    #             (self.par.copy(),
    #               copy.deepcopy(forw[:,:,r]),
    #               copy.deepcopy(sqrt_meas[:,:,r]),
    #               copy.deepcopy(self.sqb[:,:,r]),
    #               copy.deepcopy(self.par.locations[r,:])))
    #         # print (itr[t][0])
        
    #     res = pool.map(fast_grad_AF_2D_os,itr,chunksize=self.R // (10*self.par.num_threads))
        
    #     for rit in res:
    #         grad  += rit
            
    #     pool.close()
    #     pool.join()
            
    #     return grad
    
    ######################## MULTIPROCESING TRIAL 1
    
    # def fast_grad_L2_2D_pty(self,forw, sqrt_meas):
        
    #     if self.par.num_threads == 1:
    #         return fast_grad_AF_2D_os_loc((self.par,forw,sqrt_meas,self.sqb,0, self.R))
            
    #     grad = np.zeros(self.par.object_shape,dtype = complex)
        
    #     # if __name__ == "__main__":
        
            
    #     step = self.R // self.par.num_threads
        
    #     if step * self.par.num_threads < self.R:
    #         step +=1
        
        
    #     pool = mp.Pool(self.par.num_threads)
    #     # pool = ThreadPool(self.par.num_threads)
    #     itr = []
    #     for t in range(self.par.num_threads):
    #         r_start = t*step
    #         r_end = np.minimum((t+1)*step,self.R)
    #         # print(r_start,r_end)
            
    #         itr.append(
    #             (self.par.copy(),
    #               copy.deepcopy(forw[:,:,r_start:r_end]),
    #               copy.deepcopy(sqrt_meas[:,:,r_start:r_end]),
    #               copy.deepcopy(self.sqb[:,:,r_start:r_end]),
    #               r_start,
    #               r_end))
            
    #         # print (itr[t][0])
        
    #     res = pool.map(fast_grad_AF_2D_os_loc,itr)
        
    #     for rit in res:
    #         grad  += rit#.get()
            
    #     pool.close()
    #     pool.join()
            
    #     return grad
    
    ################ NO MULTIPROCESING
    
    def fast_grad_AF_2D_os(self,forw,sqrt_meas,sqb):    
        diff = 1 - sqb/sqrt_meas 
        t = forw * diff 
        grad2 = self.par.forward_adj_2D_os(t)
        
        return grad2
    
    def fast_grad_L2_2D_pty(self,z):
        grad = cp.zeros(self.par.object_shape,dtype = complex)
        
        for r in range(self.R):
            loc = self.par.locations[r,:]
            
            if self.par.circular:
                z_r = cp.roll(z,-loc[0],0)
                z_r = cp.roll(z_r,-loc[1],1)
                z_r = z_r[:self.delta1,:self.delta2]
            else:
                z_r = z[loc[0]:(loc[0] + self.delta1),loc[1]:(loc[1] + self.delta2)]
            
            forw = self.par.forward_2D_os(z_r)
            sqrt_meas = cp.sqrt(cp.abs(forw)**2 + self.epsilon)
            grad_r = cp.zeros(self.par.object_shape,dtype = complex)
            grad_r[:self.delta1,:self.delta2] = self.fast_grad_AF_2D_os(forw,sqrt_meas,self.sqb[:,:,r])
            grad_r = cp.roll(grad_r, loc[0], axis =0)
            grad_r = cp.roll(grad_r, loc[1], axis =1)
            grad += grad_r
                
        return grad 

    def fast_grad_L2_win_2D_pty(self,w):
        grad = cp.zeros((self.delta1,self.delta2),dtype = complex)
            
        z = self.par.obj.copy()
        for r in range(self.R):
            loc = self.par.locations[r,:]
            z_r = z.copy()
            z_r = cp.roll(z_r, -loc[0], axis =0)
            z_r = cp.roll(z_r, -loc[1], axis =1)
            
            self.par.set_window(z_r[:self.delta1,:self.delta2])
            forw = self.par.forward_2D_os(w)
            sqrt_meas = cp.sqrt(cp.abs(forw)**2 + self.epsilon)
            grad_r = self.fast_grad_AF_2D_os(forw,sqrt_meas,self.sqb[:,:,r])
            grad += grad_r
                
        # self.par.set_object(z)    
        
        return grad

    def grad_smoothness_Tikhonov(self,z):
        if self.par.circular:
                gr = (4*z - cp.roll(z,1,axis =0) - cp.roll(z,-1,axis =0) - cp.roll(z,1,axis =1) - cp.roll(z,-1,axis =0))
        else:
            gr = cp.zeros_like(z)
            gr[:z.shape[0]-1,:] += z[:z.shape[0]-1,:] - z[1:,:]
            gr[1:,:] += z[1:,:] - z[:z.shape[0]-1,:]
            gr[:,:z.shape[1]-1] += z[:,:z.shape[1]-1] - z[:,1:]
            gr[:,1:] += z[:,1:] - z[:,:z.shape[1]-1]
        
        return gr
    
    def grad_TV_smoothed(self,z):
        if self.par.circular:
            df  = z - cp.roll(z,1,axis = 0)
            gr = df / cp.sqrt(df**2 + self.TV_param)
            df  = z - cp.roll(z,-1,axis = 0)
            gr += df / cp.sqrt(df**2 + self.TV_param)
            df  = z - cp.roll(z,1,axis = 1)
            gr += df / cp.sqrt(df**2 + self.TV_param)
            df  = z - cp.roll(z,-1,axis = 1)
            gr += df / cp.sqrt(df**2 + self.TV_param)
        else:
            gr = cp.zeros_like(z)
            df = z[:z.shape[0]-1,:] - z[1:,:]
            gr[:z.shape[0]-1,:] += df / cp.sqrt(df**2 + self.TV_param)
            df = z[1:,:] - z[:z.shape[0]-1,:]
            gr[1:,:] += df / cp.sqrt(df**2 + self.TV_param)
            df = z[:,:z.shape[1]-1] - z[:,1:]
            gr[:,:z.shape[1]-1] += df / cp.sqrt(df**2 + self.TV_param)
            df = z[:,1:] - z[:,:z.shape[1]-1]
            gr[:,1:] += df / cp.sqrt(df**2 + self.TV_param)
        
        return 0.5*gr

    def grad_AF_2D_pen(self,z):
        grad = cp.zeros_like(z)
        
        if self.alpha_T != 0:
            grad += self.alpha_T*z
        
        if self.alpha_R != 0:
            grad += self.alpha_R* 1.0j* cp.imag(z)
            
        if self.alpha_ST != 0:
            grad += self.alpha_ST* self.grad_smoothness_Tikhonov(z)
            
        if self.alpha_TV != 0 and not self.proximal_TV:
            grad += self.alpha_TV*self.grad_TV_smoothed(z)
                
        return grad

    def fast_grad_AF_2D_pty(self,z):
        grad = self.fast_grad_L2_2D_pty(z)
        
        grad += self.grad_AF_2D_pen(z)
        return grad

    def fast_grad_AF_2D_pty_win(self,w):
        self.par.set_window(w)
        grad = self.fast_grad_L2_win_2D_pty(w)
        
        grad += self.grad_AF_2D_pen(w)
        return grad

    def learn_rate_penalties(self):
        # Tikhonov penalty
        learn_rate = self.alpha_T
        # imaginary-value penalty
        learn_rate += self.alpha_R
        
        # smoothness Tikhonov penalty
        # check non-circular case
        if self.par.circular:    
            learn_rate += self.alpha_ST * 8
        else:
            learn_rate += self.alpha_ST * (8 + 2* cp.sum(cp.sqrt(self.par.object_shape)))
        
        # smoothed TV penalty
        if self.alpha_TV != 0 and not self.proximal_TV:
            if self.par.circular:    
                learn_rate += self.alpha_TV * 0.5*8 / cp.sqrt(self.TV_param)
            else:
                learn_rate += self.alpha_TV * 0.5*(8 + 2* cp.sum(cp.sqrt(self.par.object_shape))) / cp.sqrt(self.TV_param) 

        return learn_rate
    
    def learn_rate_penalties_win(self):
        # Tikhonov penalty
        learn_rate = self.alpha_T
        # imaginary-value penalty
        learn_rate += self.alpha_R
        
        # smoothness Tikhonov penalty
        # check non-circular case
        if self.par.circular:    
            learn_rate += self.alpha_ST * 8
        else:
            learn_rate += self.alpha_ST * (8 + 2*cp.sqrt(self.delta1) + 2* cp.sqrt(self.delta2))
        # learn_rate = 2*(1-self.afpar.AG_params.control)*learn_rate**-1
        
        # smoothed TV penalty
        if self.alpha_TV != 0 and not self.proximal_TV:
            if self.par.circular:    
                learn_rate += self.alpha_TV * 0.5*8 / cp.sqrt(self.TV_param)
            else:
                learn_rate += self.alpha_TV * 0.5*(8 + 2* cp.sqrt(self.delta1) + 2* cp.sqrt(self.delta2)) / cp.sqrt(self.TV_param) 
                
        return learn_rate

    def learn_rate_2D_pty_optimal(self):
        abs_win = cp.abs(self.par.window)**2
        temp = cp.zeros(self.par.object_shape)
        for r in range(self.R):
            loc = self.par.locations[r,:]
            # temp[loc[0]:(loc[0] + delta1),loc[1]:(loc[1] + delta2)] += abs_win
            win_t = cp.zeros(self.par.object_shape)
            win_t[:self.delta1,:self.delta2] = abs_win
            win_t = cp.roll(win_t, loc[0], axis=0)
            win_t = cp.roll(win_t, loc[1], axis=1)
            temp += win_t
        
        max_win = cp.max(temp)
        
        learn_rate = max_win*cp.prod(self.par.fourier_dimension)
        learn_rate += self.learn_rate_penalties()
        learn_rate = self.learning_rate * 2*(1-self.AG_params.control)*learn_rate**-1
        
        return learn_rate

    def learn_rate_2D_pty_sub_optimal(self):
        
        max_win = self.R*cp.max(cp.abs(self.par.window)**2)
        
        learn_rate = max_win*cp.prod(self.par.fourier_dimension)
        learn_rate += self.learn_rate_penalties()
        learn_rate = self.learning_rate * 2*(1-self.AG_params.control)*learn_rate**-1
        
        return learn_rate

    def learn_rate_2D_pty_win_optimal(self):
        z = self.par.obj
        
        temp = cp.zeros(self.par.window_shape)
        for r in range(self.R):
            loc = self.par.locations[r,:]
            z_r = z.copy()
            z_r = cp.roll(z_r, -loc[0], axis=0)
            z_r = cp.roll(z_r, -loc[1], axis=1)
            temp += cp.abs(z_r[:self.delta1,:self.delta2])**2     
        
        max_win = cp.max(temp)
        
        learn_rate = max_win*cp.prod(self.par.fourier_dimension)
        learn_rate += self.learn_rate_penalties_win()    
        learn_rate = self.learning_rate *2*(1-self.AG_params.control)*learn_rate**-1
            
        return learn_rate
    
    def learn_rate_2D_pty_win_sub_optimal(self):
        z = self.par.obj   
        max_win = self.R * cp.max(cp.abs(z))
        
        learn_rate = max_win*cp.prod(self.par.fourier_dimension)        
        learn_rate += self.learn_rate_penalties_win()        
        learn_rate = self.learning_rate * 2*(1-self.AG_params.control)*learn_rate**-1
            
        return learn_rate

    def objective_L2_2D(self,z):
        l2_obj = 0
        
        for r in range(self.R):
            loc = self.locations[r,:] 
            
            if self.circular:
                z_r = cp.roll(z,-loc[0],0)
                z_r = cp.roll(z_r,-loc[1],1)
                z_r = z_r[:self.delta1,:self.delta2]
            else:
                z_r = z[loc[0]:(loc[0] + self.delta1),loc[1]:(loc[1] + self.delta2)]
            
            forw = self.par.forward_2D_os(z_r)
            meas = cp.sqrt( cp.abs(forw)**2 + self.epsilon)
            l2_obj += cp.sum(cp.abs(meas - self.sqb[:,:,r])**2)
        
        return l2_obj
    
    def relative_measurement_error(self,z):
        # \sqrt(sum_j ||Ax_j|^2 - y_j|^2) / sqrt(sum_j y_j^2) 
        l2_obj = 0
        sum_b = 0
        
        for r in range(self.R):
            loc = self.locations[r,:] 
            
            if self.circular:
                z_r = cp.roll(z,-loc[0],0)
                z_r = cp.roll(z_r,-loc[1],1)
                z_r = z_r[:self.delta1,:self.delta2]
            else:
                z_r = z[loc[0]:(loc[0] + self.delta1),loc[1]:(loc[1] + self.delta2)]
            
            forw = self.par.forward_2D_os(z_r)
            meas = cp.abs(forw)**2
            sqb_part =self.sqb[:,:,r]**2 - self.epsilon
            sum_b =cp.sum(sqb_part)
            l2_obj += cp.sum(cp.abs(meas - sqb_part)**2)
            
        return cp.sqrt(l2_obj/sum_b)
    
    # || (I - S) z||_2^2
    def objective_smoothness_Tikhonov(self,z):
        if self.par.circular:
            obj = cp.linalg.norm(z - cp.roll(z,1,axis =0),'fro')**2
            obj += cp.linalg.norm(z - cp.roll(z,-1,axis =0),'fro')**2
            obj += cp.linalg.norm(z - cp.roll(z,1,axis =1),'fro')**2
            obj += cp.linalg.norm(z - cp.roll(z,-1,axis =1),'fro')**2
        else:
            obj = cp.linalg.norm(z[:z.shape[0]-1,:] - z[1:,:],'fro')**2
            obj += cp.linalg.norm(z[1:,:] - z[:z.shape[0]-1,:],'fro')**2
            obj += cp.linalg.norm(z[:,:z.shape[1]-1] - z[:,1:],'fro')**2
            obj += cp.linalg.norm(z[:,1:] - z[:,:z.shape[1]-1],'fro')**2
            
        return obj
    
    def objective_TV_smoothed(self,z):
        if self.par.circular:
            obj = cp.sum(cp.sqrt(cp.abs(z - cp.roll(z,1,axis =0))**2 + self.TV_param))
            obj += cp.sum( cp.sqrt(cp.abs(z - cp.roll(z,-1,axis =0))**2 + self.TV_param))
            obj += cp.sum( cp.sqrt(cp.abs(z - cp.roll(z,1,axis =1))**2 + self.TV_param))
            obj += cp.sum( cp.sqrt(cp.abs(z - cp.roll(z,-1,axis =1))**2 + self.TV_param))
        else:
            obj = cp.sum(cp.sqrt(cp.abs(z[:z.shape[0]-1,:] - z[1:,:])**2 + self.TV_param))
            obj += cp.sum(cp.sqrt(cp.abs(z[1:,:] - z[:z.shape[0]-1,:])**2 + self.TV_param))
            obj += cp.sum(cp.sqrt(cp.abs(z[:,:z.shape[1]-1] - z[:,1:])**2 + self.TV_param))
            obj += cp.sum(cp.sqrt(cp.abs(z[:,1:] - z[:,:z.shape[1]-1])**2 + self.TV_param))
            
        return obj
    
    def objective_pen_2D(self,z):
        objective = 0
        
        if self.alpha_T != 0:
            objective += self.alpha_T*cp.sum(cp.abs(z)**2, axis = None)
        
        if self.alpha_R != 0:
            objective += self.alpha_R*cp.sum(cp.imag(z)**2, axis = None)
        
        if self.alpha_ST != 0:
            objective += self.alpha_ST*self.objective_smoothness_Tikhonov(z)
        
        if self.alpha_TV != 0:
            objective += self.alpha_TV*self.objective_TV_smoothed(z)
        
        return objective

    def obj_pty(self,z, obj_offset):
        l2_z = self.objective_L2_2D(z)
        obj  = l2_z + 0
        obj += self.objective_pen_2D(z)
        obj += obj_offset
        return obj, l2_z

    def obj_pty_win(self,w, obj_offset):
        # z = self.par.obj.copy()
        self.par.set_window(w)
        l2_z =  self.objective_L2_2D(self.par.obj)
        # self.par.set_window(z)
        obj = l2_z + 0
        obj += self.objective_pen_2D(w)
        obj += obj_offset
        return obj, l2_z

    def run(self, z_0, obj_shift=0.0):
        grad_f = lambda x,forw,sqrt_meas: self.fast_grad_AF_2D_pty(x)
        
        obj_f = lambda x : self.obj_pty(x, obj_shift)
        
        if self.learning_rate_type == 'optimal':
            lr =  self.learn_rate_2D_pty_optimal()
        elif self.learning_rate_type == 'suboptimal':
            lr =  self.learn_rate_2D_pty_sub_optimal()
        else:
            lr = self.learning_rate
        
        gd = gds.grad_desc(grad_f,
                           obj_f, 
                           z_0, 
                           self.number_of_iterations,
                           self.grad_threshold,
                           lr,
                           self.AG_params,
                           self.learn_rate_decay,
                           self.proximal_TV,
                           self.alpha_TV,
                           self.verbose,
                           self.track,
                           self.obj_tr,
                           self.track_it,
                           self.par.mask)
            
        obj, val = gd.run()
        if self.track:
            self.measurement_error = gd.measurement_error
            self.objective = gd.objective
            self.object_error = gd.object_error      
            self.tracked_objects = gd.tracked_objects
            
        return obj, val              

    def run_win(self,w_0, obj_shift=0.0):
        grad_f = lambda w, forw,sqrt_meas : self.fast_grad_AF_2D_pty_win(w)
            
        obj_f = lambda w : self.obj_pty_win(w, obj_shift)
        
        if self.learning_rate_type == 'optimal':
            lr = self.learn_rate_2D_pty_win_optimal()
        elif self.learning_rate_type == 'suboptimal':
            lr = self.learn_rate_2D_pty_win_sub_optimal()
        else:
            lr = self.learning_rate
        
        gd = gds.grad_desc(grad_f,
                           obj_f, 
                           w_0, 
                           self.number_of_iterations,
                           self.grad_threshold,
                           lr,
                           self.AG_params,
                           self.learn_rate_decay,
                           self.proximal_TV,
                           self.alpha_TV,
                           self.verbose,
                           self.track,
                           self.obj_tr,
                           self.track_it)
        
        win, val = gd.run()
        if self.track:
            self.measurement_error = gd.measurement_error
            self.objective = gd.objective
            self.object_error = gd.object_error      
            self.tracked_objects = gd.tracked_objects
            
        return win, val          

class blind_amplitude_flow:
    def __init__(self, 
                 **kwargs
                 # measurements, 
                 # bafpar, 
                 # reweight = False, 
                 # track = False,
                 # obj_tr = np.array([]),
                 # win_tr = np.array([]),
                 # track_it = []
                 ):
        if 'epsilon' in kwargs.keys():
            if isinstance(kwargs['epsilon'], float) or isinstance(kwargs['epsilon'], int): 
                self.epsilon = kwargs['epsilon']
            else:
                print('epsilon is given but neither float or int. Set to 0.')
                self.epsilon = 0
        else:
            print('epsilon is not given. Set to 0.')
            self.epsilon = 0
        
        
        assert 'measurements' in kwargs.keys() or 'sqb' in kwargs.keys(), "Neither measurements nor their square root is given"   
        
        if 'measurements' in kwargs.keys():
            self.sqb = cp.sqrt(cp.maximum(kwargs['measurements'],0) + self.epsilon)    
        
        if 'sqb' in kwargs.keys():
            self.sqb = kwargs['sqb']
            
        assert 'ptycho' in kwargs.keys(), "Forward model of ptycho class is not given."
        assert isinstance(kwargs['ptycho'], forward.ptycho), "ptycho is not an instance of class ptycho."
        self.par = kwargs['ptycho'].copy()    
        # self.R = len(self.par.locations)
        # self.delta1 = self.par.window.shape[0]
        # self.delta2 = self.par.window.shape[1]
        
        if 'AGP_object' in kwargs.keys():
            self.AGP_object = copy.deepcopy(kwargs['AGP_object'])
        else:
            self.AGP_object = types.SimpleNamespace(enable_AG=False,
                                control=0.5,
                                tau = 0.3,
                                AG_iterations = 2)
            
        if 'AGP_window' in kwargs.keys():
            self.AGP_window = copy.deepcopy(kwargs['AGP_window'])
        else:
            self.AGP_window = types.SimpleNamespace(enable_AG=False,
                                control=0.5,
                                tau = 0.3,
                                AG_iterations = 2) 
            
        assert 'number_of_iterations' in kwargs.keys(), "number_of_iterations is not specified."
        assert isinstance(kwargs['number_of_iterations'], int), "number_of_iterations is not an integer"        
        self.number_of_iterations = kwargs['number_of_iterations']
        
        assert 'object_subiterations' in kwargs.keys(), "object_subiterations is not specified."
        assert isinstance(kwargs['object_subiterations'], int), "object_subiterations is not an integer"        
        self.object_subiterations = kwargs['object_subiterations']
        
        assert 'window_subiterations' in kwargs.keys(), "window_subiterations is not specified."
        assert isinstance(kwargs['window_subiterations'], int), "window_subiterations is not an integer"        
        self.window_subiterations = kwargs['window_subiterations']
        
        if 'learning_rate' in kwargs.keys():
            self.learning_rate = kwargs['learning_rate']
        else:
            self.learning_rate = 1.0
            
        if 'learning_rate_type' in kwargs.keys():
            self.learning_rate_type = kwargs['learning_rate_type']
        else:
            self.learning_rate_type = 'optimal'
        
        if 'learn_rate_decay' in kwargs.keys():
            self.learn_rate_decay = kwargs['learn_rate_decay']
        else:
            self.learn_rate_decay = 0.0
        
        if 'grad_thr_object' in kwargs.keys():
            self.grad_thr_object = kwargs['grad_thr_object']
        else:
            self.grad_thr_object = 0
        
        if 'grad_thr_window' in kwargs.keys():
            self.grad_thr_window = kwargs['grad_thr_window']
        else:
            self.grad_thr_window = 0
            
        if 'alpha_T' in kwargs.keys():
            self.alpha_T = kwargs['alpha_T']
        else:
            self.alpha_T = 0
        
        if 'alpha_R' in kwargs.keys():
            self.alpha_R = kwargs['alpha_R']
        else:
            self.alpha_R = 0
            
        if 'alpha_ST' in kwargs.keys():
            self.alpha_ST = kwargs['alpha_ST']
        else:
            self.alpha_ST = 0
            
        if 'proximal_TV_obj' in kwargs.keys() and not ('alpha_TV' in kwargs.keys()):
            print('proximal_TV_obj is given, but alpha_TV is not. Set to False')
            self.proximal_TV_obj = False
        else:
            self.proximal_TV_obj = False    
        
        if 'alpha_TV' in kwargs.keys():
            self.alpha_TV = kwargs['alpha_TV']
            
            if 'proximal_TV_obj' in kwargs.keys():
                self.proximal_TV_obj = kwargs['proximal_TV_obj']            
            else:
                self.proximal_TV_obj = False
            
            if 'TV_param_obj' in kwargs.keys():
                self.TV_param_obj = kwargs['TV_param_obj']
            else:
                if self.proximal_TV_obj:
                    self.TV_param_obj = 0
                else:
                    self.TV_param_obj = 1    
        else:
            self.alpha_TV = 0
            self.TV_param_obj = 0.0
            
        if 'beta_T' in kwargs.keys():
            self.beta_T = kwargs['beta_T']
        else:
            self.beta_T = 0
        
        if 'beta_R' in kwargs.keys():
            self.beta_R = kwargs['beta_R']
        else:
            self.beta_R = 0
            
        if 'beta_ST' in kwargs.keys():
            self.beta_ST = kwargs['beta_ST']
        else:
            self.beta_ST = 0
            
        if 'proximal_TV_win' in kwargs.keys() and not ('beta_TV' in kwargs.keys()):
            print('proximal_TV_win is given, but beta_TV is not. Set to False')
            self.proximal_TV_win = False
        else:
            self.proximal_TV_win = False    
        
        if 'beta_TV' in kwargs.keys():
            self.beta_TV = kwargs['beta_TV']
            
            if 'proximal_TV_win' in kwargs.keys():
                self.proximal_TV_win = kwargs['proximal_TV_win']            
            else:
                self.proximal_TV_win = False
            
            if 'TV_param_win' in kwargs.keys():
                self.TV_param_win = kwargs['TV_param_win']
            else:
                if self.proximal_TV_win:
                    self.TV_param_win = 0
                else:
                    self.TV_param_win = 1    
        else:
            self.beta_TV = 0
            self.TV_param_win = 0.0
        
        if 'skip_win_it' in kwargs.keys():
            self.skip_win_it = kwargs['skip_win_it']
        else:
            self.skip_win_it = 0
        
        if 'reweight' in kwargs.keys():
            self.reweight = kwargs['reweight']
        else:
            self.reweight = False
        
        if 'verbose' in kwargs.keys():
            self.verbose = kwargs['verbose']
        else:
            self.verbose = 0
        
        if 'track' in kwargs.keys():
            self.track = kwargs['track']
        else:
            self.track = False
            
        if self.track:
            
            if 'obj_tr' in kwargs.keys():
                assert kwargs['obj_tr'].shape[0] == self.par.object_shape[0] and kwargs['obj_tr'].shape[1] == self.par.object_shape[1], "Shape of true object is different from par.object_shape."
                
                self.obj_tr = kwargs['obj_tr']
                
            if 'win_tr' in kwargs.keys():
                assert kwargs['win_tr'].shape[0] == self.par.window_shape[0] and kwargs['win_tr'].shape[1] == self.par.window_shape[1], "Shape of true window is different from par.window_shape."
                
                self.win_tr = kwargs['win_tr']
            
            if self.reweight:
                obj_norm = cp.sum(cp.abs(self.obj_tr)**2,axis = None)
                win_norm = cp.sum(cp.abs(self.win_tr)**2,axis = None)
                self.obj_tr = self.obj_tr * obj_norm 
                self.win_tr = self.win_tr * win_norm 
                 
            self.measurement_error = cp.zeros((0))
            self.objective = cp.zeros((0))
            self.object_error = cp.zeros((0))
            self.window_error = cp.zeros((0))
            
            if 'track_it' in kwargs.keys():
                self.track_it = cp.sort(kwargs['track_it']) 
        else:
            self.obj_tr = [] 
            self.win_tr = []
            self.track_it = []
            
            
            self.measurement_error = cp.zeros((0))
            self.objective = cp.zeros((0))
            self.object_error = cp.zeros((0))

    def prepare_algorithms(self):
        af_o = amplitude_flow(sqb = self.sqb,
                              ptycho = self.par.copy(),
                              AG_params = self.AGP_object,
                              number_of_iterations = self.object_subiterations, 
                              grad_threshold = self.grad_thr_object,
                              learning_rate_type = self.learning_rate_type,
                              learning_rate = self.learning_rate,
                              learn_rate_decay = 0.0,
                              epsilon = self.epsilon,
                              alpha_T = self.alpha_T,
                              alpha_R = self.alpha_R,
                              alpha_ST = self.alpha_ST,
                              alpha_TV = self.alpha_TV,
                              TV_param = self.TV_param_obj,
                              proximal_TV= self.proximal_TV_obj,
                              verbose = self.verbose,
                              track = self.track,
                              obj_tr = self.obj_tr)
        
        af_w = amplitude_flow(sqb = self.sqb,
                              ptycho = self.par.copy(),
                              AG_params = self.AGP_window,
                              number_of_iterations = self.window_subiterations, 
                              grad_threshold = self.grad_thr_window,
                              learning_rate_type = self.learning_rate_type,
                              learning_rate = self.learning_rate,
                              learn_rate_decay = 0.0,
                              epsilon = self.epsilon,
                              alpha_T = self.beta_T,
                              alpha_R = self.beta_R,
                              alpha_ST = self.beta_ST,
                              alpha_TV = self.beta_TV,
                              TV_param = self.TV_param_win,
                              proximal_TV= self.proximal_TV_win,
                              verbose = self.verbose,
                              track = self.track,
                              obj_tr = self.win_tr)
        
        return af_o, af_w

    def run(self,
            z_0,
            w_0):
        
        z = z_0
        w = w_0
        
        if self.reweight:
            z,w = util.normalize_object_and_window(z,w)
        
        af_o, af_w = self.prepare_algorithms()
        
        count = 0
        
        if len(self.track_it) > 0:
            self.tracked_objects = cp.zeros((0,z.shape[0],z.shape[1]),dtype = complex)
            self.tracked_windows = cp.zeros((0,w.shape[0],w.shape[1]),dtype = complex)
        
        for lp in range(self.number_of_iterations):
            if lp == 0:
                offset = 0
            else:
                offset = 1
            
            if self.reweight:
                z,w = util.normalize_object_and_window(z,w)
            
            if self.verbose:
                print('Iteration: ',lp)
            
            penalty_w = af_w.objective_pen_2D(w)
            af_o.par.set_window(w)        
            z,obj = af_o.run(z,penalty_w)
            
            if self.track:
                self.measurement_error = cp.append(self.measurement_error,af_o.measurement_error[offset:])
                self.objective = cp.append(self.objective,af_o.objective[offset:])
                self.object_error = cp.append(self.object_error,af_o.object_error[offset:])
                
            if self.reweight:
                w,z = util.normalize_object_and_window(w,z)
            
            if lp >= self.skip_win_it:
                penalty_z = af_o.objective_pen_2D(z)
                af_w.par.set_object(z)  
                w,obj = af_w.run_win(w, penalty_z)  
            
                if self.track:
                    self.measurement_error = cp.append(self.measurement_error,af_w.measurement_error[1:])
                    self.objective = cp.append(self.objective,af_w.objective[1:])                
                    self.window_error = cp.append(self.window_error,af_w.object_error[1:])
            else:
                if self.track:
                    self.measurement_error = cp.append(self.measurement_error,cp.repeat(self.measurement_error[-1], self.window_subiterations))
                    self.objective = cp.append(self.objective,cp.repeat(self.objective[-1], self.window_subiterations))                
                    self.window_error = cp.append(self.window_error,cp.repeat(0, self.window_subiterations))                     
  
            if self.learn_rate_decay != 0.0:
                decay = ( (lp +1)/(lp+2) )**self.learn_rate_decay
                af_o.learning_rate *= decay
                af_w.learning_rate *= decay
            
            if count < len(self.track_it):
                if lp == self.track_it[count]:
                    self.tracked_objects = cp.append(self.tracked_objects,[util.align_objects(self.obj_tr,z,self.par.mask)],axis= 0)
                    self.tracked_windows = cp.append(self.tracked_windows,[util.align_objects(self.win_tr,w,cp.ones((w.shape[0],w.shape[1])))],axis= 0)
                    count += 1
            
        if self.reweight:
            z,w = util.normalize_object_and_window(z,w)
        
        return z,w,obj
