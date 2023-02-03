"""
Created: 23/01/2023

Author: ks20447 (Adam Morris)

Initial main.py file for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch
"""

import numpy as np
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
    
  
def solve_to(f, x0, t, h, method, h_max=None):
    """Solves given ODE from t0 to t1

    Args:
        f (np.array): Set of ODE's being solved
        x0 (np.array): Array of initial conditions [x_1(0) x_2(0) ...]
        t (np.array): Array of timesteps 
        h (float): stepsize
        method (string): specifies the type of solver (Euler, RK)

    Returns:
        array: Array of approximated ODE solutions 
    """   
    
    x = np.zeros((len(x0), len(t)))
    for i in range(len(x)):
        x[i][0] = x0[i] 
    

    match method:
        case "Euler":
            for i in range(len(t) - 1):
                for j in range(len(x0)):
                    x[j][i + 1] = eurler_step(f[j], x[j][i], t[i], h)      
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
        



def run():
    
    def func(x, t):
        dxdt = x
        return dxdt

    x0 = np.array([1, 1])
    ode = np.array([func])
    h = 0.01
    t1, t2 = 0, 1
    t = np.arange(t1, t2 + h, h)
    x = solve_to(ode, x0, t, h, 'Euler')
    print(x)
  
    
run()