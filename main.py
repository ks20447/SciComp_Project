"""
Created: 23/01/2023

Author: ks20447 (Adam Morris)

Initial main.py file for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


def eurler_step(f, x, t, h):
    """Single Euler step for solving ODE's

    Args:
        f (function): ODE function being solved
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize

    Returns:
        float: approximation for next x 
    """
    x_new = x + h*f(t, x)
    return x_new


def runga_kutta():
    x = 0
    
    
def solve_to(f, x0, t, h, method):
    """Solves given ODE from t0 to t1

    Args:
        f (function): Set of ODE's being solved
        x0 (array): Array of initial conditions [x_1(0) x_2(0) ...]
        t (array): Array of timesteps 
        h (float): stepsize
        method (string): specifies the type of solver (Euler, RK)

    Returns:
        array: Array of approximated x values 
    """
        
    x = np.zeros(len(t))
    x[0] = x0
    
    match method:
        case "Euler":
            for i in range(len(t) - 1):
                x[i + 1] = eurler_step(f, x[i], t[i], h)
        case "RK":
            x = runga_kutta()
    
    return x
        
        
def error_plot(steps, error):
    fig, ax = plt.subplots()
    ax.plot(steps, error)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    plt.show()
        

def func(x, t):
    dxdt = x
    return dxdt
    

def run():
    x0 = 1
    h = 0.01
    t1, t2 = 0, 1 
    t = np.arange(t1, t2 + h, h)
    x = solve_to(func, x0, t, h, 'Euler')
  
    
run()