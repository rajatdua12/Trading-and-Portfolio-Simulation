#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:25:46 2020

@author: rajatdua
"""

"""
In this problem, I have estimated 99th percentile of portfolio loss distribution 
using Monte-Carlo simulation and the Euler scheme under the assumption of correlated 
driving Geometric Brownian Motion. I have assumed parameters as given in the question. 
There are 2 equally weighted stocks in the portfolio. After computationally running the 
algorithm for 252-time steps and 10,000 paths, I have found out that the actual 99th percentile 
is 0.01862 using the empirical mean-variance formula [2.326 standard deviations from the mean 
for 99th percentile]. Using Monte-Carlo simulation, the 99th percentile estimate came out to be 0.01849. 

I have further evaluated the convergence properties of Monte-Carlo simulation technique 
by plotting multiple histograms for number of paths corresponding to 10, 100, 1000, 10000, 
and 100000. I have found that as we increase the number of paths the difference between 
the actual 99th percentile and estimated 99th percentile using Monte-Carlo simulation decreases. 
This result can be shown using the convergence of histogram curves towards the Actual 99th percentile 
of portfolio-loss distribution, as we increase the number of paths from 10 to 100000. 
"""

import numpy as np
import matplotlib.pyplot as plt

S0 = [100, 100]
timesteps = 252
dt =1/timesteps
covMat = np.ones([2,2])
rho = -0.1
covMat[0,0] = dt
covMat[0,1] = rho*dt
covMat[1,0] = rho*dt
covMat[1,1] = dt
mat = 1
sigma = 0.1
mu = 0.1
num_stock = len(S0)
numPaths = 10000
init_stock_price = np.repeat(np.array([S0]), numPaths, axis = 0)

L = np.linalg.cholesky(covMat)
z = np.random.randn(2, 10000)
y = np.dot(L,z)
rets = mu * dt + sigma * y
losses = - (np.sum(rets, axis=0))
percentile_99 = np.percentile(losses, 99)

print('Estimated 99th Percentile using Monte-Carlo Simulation: ', percentile_99)

mu_p = np.mean(losses)
std_p = np.std(losses)
percentile_99_actual = mu_p + 2.326 * std_p
print('Actual 99th Percentile: ', percentile_99_actual)

paths = [10, 100, 1000, 10000, 100000]

results = {}

for path in paths:
    stock1_rets = np.zeros([path, timesteps])
    stock2_rets = np.zeros([path, timesteps])
    portfolio_rets = np.zeros([path, timesteps])
    
    for t in np.arange(timesteps):
        
        z = np.random.randn(2, path)
        rets = (mu * 1/timesteps + sigma * (np.dot(L, z))).T
        stock1_rets[:, t] = rets[:, 0]
        stock2_rets[:, t] = rets[:, 1]
        portfolio_rets[:, t] = (stock1_rets[:, t] + stock2_rets[:, t])/2
        
    stock1_ret = (1 + stock1_rets).prod(axis=1) - 1
    stock2_ret = (1 + stock2_rets).prod(axis=1) - 1 
    port_ret = (1 + portfolio_rets).prod(axis=1) - 1
    ex_port = port_ret.mean()
    
    perce = np.percentile(port_ret, 1)
    
    results[path] = {'port_ret': port_ret, 'pct1': perce}
    
    fig = plt.figure()
    plt.xlabel('Portfolio Return')
    plt.title('Distribution of portfolio return for N: {} and NumPath: {}'.format(num_stock, path))
    plt.hist(port_ret)
    fig.savefig('Plot {}'.format(path), dpi = 100)