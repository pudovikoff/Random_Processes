import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import math
from tqdm.notebook import tqdm
import seaborn as sns

class Woman:
    def __init__(self, days = 365, P = np.array([[0.7, 0.3],[0.4, 0.6]]),
                 p0 = np.array([0.6, 0.4]), p_action = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])):
        if np.sum(p0) != 1:
            raise Exception(f"Wrong start distibution p0 {p0}, must an np.array with sum of elements equal 1")
        if P.shape[0] != P.shape[1]:
            raise Exception(f"Wrong probability transition matrix")
        if np.array_equal(np.sum(p_action, axis=1), np.ones(p_action.shape[0])):
            raise Exception(f"Wrong action function")
        self.P = P
        self.p0 = p0
        self.p_action = p_action
        self.days = days
        self.weather = []
        self.d_weather = {0: 'Rain', 1: 'Sunny'}
        # 0 position - rain
        # 1 position - sunny
        self.actions = []
        self.d_actions = {0: 'Walk', 1: 'Shop', 2:'Clean'}
        # 0 position - walk
        # 1 postion - shop
        # 2 position - clean
    
    def simulate_weather(self):
        '''
        clear the history, simulate only weather for self.days times
        '''
        weather = []
        cur_p = self.p0
        for i in range(self.days):
            today = np.random.choice([0,1], size=1, p = cur_p)
            weather.append(int(today))
            cur_p = self.P.T @ cur_p
        self.weather = weather
        return self.weather

    def simulate_action(self):
        '''
        clear action history, simulate only action by weather
        '''
        actions = []
        if len(self.weather) != self.days:
            raise Exception(f"Simulate weather first")
        for i in range(self.days):
            today = np.random.choice([0,1,2], size=1, p = self.p_action[self.weather[i]] )
            actions.append(int(today))
        self.actions = actions
        return self.actions
    
    def get_weather(self, words=False):
        '''if words=True, return result in words))'''
        if words:
            return [self.d_weather[x] for x in self.weather]
        return self.weather

    def get_actions(self, words=False):
        '''if words=True, return result in words))'''
        if words:
            return [self.d_actions[x] for x in self.actions]
        return self.actions
    
    def optimal_estimation(self):
        '''make optimal estimation'''
        if len(self.actions) != self.days:
            raise Exception("Nothing to estimate")
        x_est = []
        for i in range(self.days):
            Y = np.zeros(self.p_action.shape[1])
            Y[self.actions[i]] = 1
            if i == 0:
                tmp = np.diag(self.p_action @ Y) @ self.p0
                x_hat = tmp / np.sum(tmp)
            else:
                x_wide = self.P.T @ x_est[-1]
                tmp = np.diag(self.p_action @ Y) @ x_wide
                x_hat = tmp / np.sum(tmp)
            x_est.append(x_hat)
        self.x_est = x_est
        return self.x_est
    
    def statistic(self):
        if len(self.actions) != self.days:
            raise Exception("Nothing to estimate")
        matrix = np.zeros(self.p_action.shape)
        for i in range(self.days):
            weather = self.weather[i]
            act = self.actions[i]
            matrix[weather, act] += 1
        return matrix
    
    def plot_statistic(self, ret = False):
        true_weather = self.get_weather(words=True)
        act = self.get_actions(words=True)
        df = pd.DataFrame(np.array([true_weather, act]).T, columns=['weather', 'act'])
        sns.countplot(x = 'weather', hue = 'act', data = df)
        if ret:
            return df


