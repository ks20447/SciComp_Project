"""
Created: 23/01/2023

Author: ks20447 (Adam Morris)

numerical_methods.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
4) (Optional) Add another numerical integration method
"""

import numpy as np


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


def eurler_step_second(f, x1, x2, t, h):
    """SIngle Euler setp for 2nd order ODE approximation

    Args:
        f (function): 2nd order ODE funcction being approximated
        x1 (float): current x1 approximation
        x2 (float): current x2 approximation
        t (float): current timestep
        h (float): stepsize

    Returns:
        float: approximation for next x1
        float: approximation for next x2
        float: next timestep
    """
    t_new = t + h
    x1_new = x1 + h*f(t, x1, x2)[0]
    x2_new = x2 + h*f(t, x1, x2)[1]
    
    return x1_new, x2_new, t_new


def eurler_step(f, x, t, h):
    """Single Euler step for 1st odrer ODE approximations

    Args:
        f (function): 1st order ODE function being approximated
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize

    Returns:
        float: approximation for next x
        float: next timestep 
    """
    
    t_new = t + h
    x_new = x + h*f(t, x)
    
    return x_new, t_new


def runge_kutta_second(f, x1, x2, t, h):
    """Runga-Kutta 4th order method for ODE approximations

    Args:
        f (function): ODE function being solved
        x (float): current x approximation
        t (float): current timestep
        h (float): stepsize

    Returns:
        float: approximation for next x
        float: next timestep
    """
    
    t_new = t + h
    
    # Runge-Kutta coefficients
    k11 = f(t, x1, x2)[0]
    k21 = f(t, x1, x2)[1]
    k12 = f(t + (h/2), x1 + h*(k11/2), x2 + h*(k21/2))[0]
    k22 = f(t + (h/2), x1 + h*(k11/2), x2 + h*(k21/2))[1]
    k13 = f(t + (h/2), x1 + h*(k12/2), x2 + h*(k22/2))[0]
    k23 = f(t + (h/2), x1 + h*(k12/2), x2 + h*(k22/2))[1]
    k14 = f(t + h, x1 + h*k13, x2 + h*k23)[0]
    k24 = f(t + h, x1 + h*k13, x2 + h*k23)[1]
    
    x1_new = x1 + (h/6)*(k11 + 2*k12 + 2*k13 + k14)
    x2_new = x2 + (h/6)*(k21 + 2*k22 + 2*k23 + k24)
    
    
    return x1_new, x2_new, t_new


def runge_kutta(f, x, t, h):
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
    
    # Runge-Kutta coefficients
    k1 = f(t, x)
    k2 = f(t + h/2, x + h*k1/2)
    k3 = f(t + h/2, x + h*k2/2)
    k4 = f(t + h, x + h*k3)
    
    x_new = x + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return x_new, t_new
    
  
def solve_to(f, x0, t1, t2, h, method, deltat_max=0.5):
    """_summary_

    Args:
        f (function): 1st or 2nd order ODE to be solved
        x0 (float, array): Initial condition x0 = a, or vector x = [a1, a2] 
        t1 (float): Start time
        t2 (float): End time
        h (float): Step-size
        method (string): {'EulerFirst', 'EulerSecond', 'RK4First', 'RK4Second', 'Midpoint'} Method of solving ODE

    Returns:
        array: approximation of ODE solution
        array: timestpes of ODE solution
    """
    
    if h > deltat_max:
        raise ValueError("Given step-size exceeds maximum step-size")
    
    no_steps = int((t1 + t2)/h)
    t = np.zeros(no_steps + 1)

    match method:
        case "EulerFirst":
            x = np.zeros(len(t))
            x[0] = x0
            for i in range(len(t) - 1):
                x[i + 1], t[i + 1] = eurler_step(f, x[i], t[i], h)

        case "EulerSecond":   
            x = np.zeros((2, len(t)))
            x[0][0], x[1][0] = x0[0], x0[1]
            for i in range(len(t) - 1):
                x[0][i + 1], x[1][i + 1], t[i + 1] = eurler_step_second(f, x[0][i], x[1][i], t[i], h)                     
   
        case "RK4First":
            x = np.zeros(len(t))
            x[0] = x0
            for i in range(len(t) - 1):
                x[i + 1], t[i + 1] = runge_kutta(f, x[i], t[i], h)

        case "RK4Second":
            x = np.zeros((2, len(t)))
            x[0][0], x[1][0] = x0[0], x0[1]
            for i in range(len(t) - 1):
                x[0][i + 1], x[1][i + 1], t[i + 1] = runge_kutta_second(f, x[0][i], x[1][i], t[i], h)  
                                     
        case "Midpoint":
            x = np.zeros(len(t))
            x[0] = x0
            for i in range(len(t) - 1):
                x[i + 1], t[i + 1] = midpoint_method(f, x[i], t[i], h)

                    
    return x, t

