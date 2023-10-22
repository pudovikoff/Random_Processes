import math
import numpy as np 
import scipy.stats as ss
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.notebook import tqdm


class Observing_Mean_Reversion:
    def __init__(self, a, b, x0, T, x_mean, H, B, h, p_w):
        '''
        a, b, x_mean, x0 - constants for mean-reversion
        T - [0, T] - interval for mean-reversion simulation
        h - step of mean-reversion simulation
        H - step of observing mean-reversion
        '''
        self.a, self.b, self.x0, self.x_mean, self.T = a, b, x0, x_mean, T
        self.h, self.H = h, H
        self.B = B
        self.time = [0]
        self.p_w = p_w
    
    
    def mean_reversion_simulation(self):
        '''
        making one simulation of mean_reversion process
        using Euler–Maruyama method
        a, b - fixes constants
        x0 - starting point at t=0
        x_mean - fixed mean value
        T - total time of simulation
        h - step for simulation    
        '''
        steps = int(self.T/self.h)
        mean_reversion = np.zeros(steps + 1)
        mean_reversion[0] = self.x0
        for i in range(1, steps + 1):
            prev = mean_reversion[i-1]
            mean_reversion[i] = prev + self.a * (self.x_mean - prev)* self.h + self.b * self.h ** 0.5 * np.random.normal(size=1)
            self.time.append(i * self.h)
        self.mean_rev = mean_reversion
        return mean_reversion
    
    
    def mean_reversion_mean(self):
        '''
        making mean value of mean_reversion process
        a, b - fixes constants
        x0 - starting point at t=0
        x_mean - fixed mean value
        T - total time of simulation
        h - step for simulation    
        '''
        steps = int(self.T/self.h)
        mean = np.zeros(steps + 1)
        mean[0] = self.x0
        for i,tau in enumerate(self.time):
            mean[i] = self.x0 * np.exp(- self.a * tau) + self.x_mean * (1 - np.exp(- self.a * tau))
        return mean
    
    def observe(self):
        '''
        oberve our mean-reversion
        in suppose H = k * h
        k - some integer
        '''
        k = int(H/h)
        
        tmp = self.mean_rev[[i for i in range(0, len(self.mean_rev), k)]]
        y = tmp + self.B * H ** 0.5 * self.p_w.rvs(size = len(tmp))[0]
        
        self.observed = y
        self.k = k
        return y
    
    def estimate(self):
        '''
        kalman filter - read above for more comments
        '''
        self.est_x_list = []
        self.est_cov_list = []
        
        # translating our params to Kalman filter terms
        a = 1 - self.a * self.h
        c = self.a * self.x_mean * self.h
        b = self.b * h ** 0.5 
        A = 1
        C = 0
        B = self.B * H ** 0.5
        
        for i in tqdm(range(len(self.mean_rev))):
            if i == 0:
                # start
                X_hat = self.x0
                self.est_x_list.append(X_hat)
                
                k0 = 1
                self.est_cov_list.append(k0)
            else:
                #prediction
                
                X_wide = a * self.est_x_list[i - 1] + c
                k_wide = a * self.est_cov_list[i - 1] * a + b * b
                
                #correction if and only if we have observed value at this time
                if not i % self.k:
                    X_hat = X_wide + k_wide * A * (A * k_wide * A + B * B)**(-1) * (self.observed[int(i/self.k)] - A * X_wide - C)
                    k = k_wide - k_wide * A * (A * k_wide * A + B * B)**(-1) * A * k_wide
                else:
                    X_hat = X_wide
                    k = k_wide
                
                self.est_x_list.append(X_hat)
                self.est_cov_list.append(k)
        
        return self.est_x_list, self.est_cov_list
        
                