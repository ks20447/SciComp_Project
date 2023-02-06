"""
Created: 23/01/2023

Author: ks20447 (Adam Morris)

Initial main.py file for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

Notes:
Needs adapting to systems of 1st Order ODE's, currently can only solve sets of independant ODE's, rather than sets forming higher order ODE's
Need a system of producing the error for all technqiues, and adapting h as required
"""

import numpy as np
import matplotlib.pyplot as plt


def midpoint_method(f, x, t, h):
    """Explicit midpoint method for ODE approximations

    Args:
        f (function): ODE function being solved
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize

    Returns:
        float: approximation for next x 
    """
    
    t_new = t + h
    x_new = x + h*f(t + h/2, x + (h/2)*f(t, x))
    
    return x_new, t_new


def eurler_step(f, x, t, h):
    """Single Euler step for ODE approximations

    Args:
        f (function): ODE function being solved
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize

    Returns:
        float: approximation for next x 
    """
    
    t_new = t + h
    x_new = x + h*f(t, x)
    
    return x_new, t_new


def runga_kutta(f, x, t, h):
    """Runga-Kutta 4th order method for ODE approximations

    Args:
        f (function): ODE function being solved
        x (float): current x approximation
        t (float): current timestep
        h (float): stepsize

    Returns:
        float: approximation for next x
    """
    
    t_new = t + h
    
    # Runga-Kutta coefficients
    k1 = f(t, x)
    k2 = f(t + h/2, x + h*k1/2)
    k3 = f(t + h/2, x + h*k2/2)
    k4 = f(t + h, x + h*k3)
    
    x_new = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return x_new, t_new
    
  
def solve_to(f, x0, t1, t2, h, method, tol=10e-6):
    """Solves set of given ODE's over time t using specified method

    Args:
        f (array): Set of 1st Order ODE's being solved [f(x_1) f(x_2) ...]
        x0 (array): Array of initial conditions [x_1(0) x_2(0) ...]
        t1 (float): Start time
        t2 (float) : End time
        h (float): stepsize
        method (string): specifies the type of solver (Euler, RK4, Midpoint)
        tol (float), optional: desired error tolerance (default = 10e-6) 

    Returns:
        array: Array of approximated ODE solutions
        array: Array of timesteps 
    """   
    
    if not type(f) is np.ndarray:
        raise TypeError("f should be a numpy array of functions")
    
    no_steps = int((t1 + t2)/h)
    x = np.zeros((len(f), no_steps + 1))
    t = np.zeros((len(f), no_steps + 1))
    
    for i in range(len(f)):
        x[i][0] = x0[i]

    match method:
        case "Euler":
            for j in range(len(f)):
                for k in range(len(t[0]) - 1):
                    x[j][k + 1], t[j][k + 1] = eurler_step(f[j], x[j][k], t[j][k], h)
                    
   
        case "RK4":
            for j in range(len(f)):
                for k in range(len(t[0]) - 1):
                    x[j][k + 1], t[j][k + 1] = runga_kutta(f[j], x[j][k], t[j][k], h)
                                     
        case "Midpoint":
            for j in range(len(f)):
                for k in range(len(t[0]) - 1):
                    x[j][k + 1], t[j][k + 1] = midpoint_method(f[j], x[j][k], t[j][k], h)
                    
    return x, t

    
def error_plot(steps, error):
    fig, ax = plt.subplots()
    ax.plot(steps, error)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    plt.show()
        

def run():
    
    # Example 1st order ODE function to be solved
    def func(t, x):
        dxdt = x
        return dxdt

    x0 = np.array([1])
    ode = np.array([func])
    h = 0.01
    t1, t2 = 0, 1
    tol = 10e-6
    x, t = solve_to(ode, x0, t1, t2, h, "Euler", tol=tol)
    plt.plot(t[0], x[0], label="Approx 1")
    plt.legend()
    plt.show()

    
run()
