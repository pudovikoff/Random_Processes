import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import math
from scipy import integrate
from tqdm.notebook import tqdm

class LinearSystem:
        def __init__(self, a, b, c, A, B, C):
            self.a = a
            self.b = b
            self.c = c
            self.A = A
            self.B = B
            self.C = C

            self.time_list = []
            self.x_list = []
            self.y_list = []
            self.est_x_list = []

            self.k_t = []
            self.a_r = []
            self.b_r = []
            self.est_x_r_list = []
            self.est_cov_r_list = []
            
            self.smooth_x_list = []
            self.smooth_cov_list = []
        
        def fit(self, x: np.ndarray, y: np.ndarray, mu, D0):
            self.x_list = x
            self.y_list = y
            self.m0 = mu
            self.D_0 = D0

        def estimate_x(self):
            self.est_x_list = []
            self.est_cov_list = []
            for i in tqdm(range(self.x_list.shape[0])):
                if i == 0:
                    # start
                    X_hat = self.m0 + self.D_0 @ self.A.T @ np.linalg.pinv((self.A @ self.D_0 @ self.A.T + self.B @ self.B.T)) @ (self.y_list[0] - self.A @ self.m0 - self.C)
                    self.est_x_list.append(X_hat)
                    
                    k0 =  self.D_0 -  self.D_0 @ self.A.T @ np.linalg.pinv(self.A @  self.D_0 @ self.A.T + self.B @ self.B.T) @ self.A @  self.D_0
                    self.est_cov_list.append(k0)
                    
                    self.k_t.append(self.D_0) 
                else:
                    #prediction
                    X_wide = self.a @ self.est_x_list[i - 1] + self.c
                    k_wide = self.a @ self.est_cov_list[i - 1] @ self.a.T + self.b @ self.b.T

                    #correction
                    X_hat = X_wide + k_wide @ self.A.T @ np.linalg.pinv(self.A @ k_wide @ self.A.T + self.B @ self.B.T) @ (self.y_list[i] - self.A @ X_wide - self.C)
                    k = k_wide - k_wide @ self.A.T @ np.linalg.pinv(self.A @ k_wide @ self.A.T + self.B @ self.B.T) @ self.A @ k_wide
                    
                    self.est_x_list.append(X_hat)
                    self.est_cov_list.append(k)

                    #self.k_t.append(self.a * self.k_t[-1] * self.a + self.b * self.b)
            #Additional element 
            #self.k_t.append(self.a * self.k_t[-1] * self.a + self.b * self.b)
            
            return self.est_x_list