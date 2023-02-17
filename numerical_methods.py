"""
Created: 23/01/2023

Author: ks20447 (Adam Morris)

numerical_methods.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
4) (Optional) Add another numerical integration method

Notes:
For log error graph, use log space for time
"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots


def graph_format(x_label, y_label, title, filename):
    """Matplotlib.pyplot plot formatting

    Args:
        x_label (string): x-axis label
        y_label (string): y-axis label
        title (string): title of plot
        filename (string): name given to saved .png plot
    """
    
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"results/{filename}") 
    

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
    if type(x) == np.float64:
        x = [x]
    
    dim = len(x)
    x = [t] + x 
    x_new = np.zeros(dim)
    
    t_new = t + h

    if len(x) > 2:
        for i in range(dim):
            x_new[i] = x[i + 1] + h*f(*x)[i]
    else:
        for i in range(dim):
            x_new[i] = x[i + 1] + h*f(*x)
    
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
    """Numerically solves given ODE(s) from t1 to t2, in step-size h, with intitial condition(s) x0

    Args:
        f (function): ODE system to be solved. lambda t, x_1, ..., x_n : f(x_1, ..., x_n)
        x0 (float, array-like): Initial condition x_0 = a, or vector x = [a_1, ..., a_n] 
        t1 (float): Start time
        t2 (float): End time
        h (float): Step-size
        method (string): {'Euler', 'RK4First', 'RK4Second', 'Midpoint'} Method of solving ODE
        deltat_max (float, optional): Maximum step-size allowed for solution. Defaults to 0.5.

    Raises:
        ValueError: h value is larger than deltat_max
        TypeError: x0 should be given as an integer/float or array-like
        SyntaxError: method type did not match predefined methods

    Returns:
        array: approximation of ODE solution
        array: timestpes of ODE solution
    """
    
    
    if h > deltat_max:
        raise ValueError("Given step-size exceeds maximum step-size")
    
    
    if isinstance(x0, (int, float)):
        x0 = [x0]
    elif (x0, (list, np.array)):
        x0 = x0
    else:
        raise TypeError("x0 is incorrect data type")
    
    
    try:
        f(t1, *x0)
    except TypeError:
        print(f"Function and initial condition dimesions do not match")
        quit()
      
        
    no_steps = int((t2 - t1)/h)
    t = np.zeros(no_steps)
    t[0], t[-1] = t1, t2
    
    x = np.zeros((len(x0), len(t)))
    for ind, iv in enumerate(x0):
        x[ind][0] = iv
    
    
    match method:
        
        case "Euler":
            for i in range(len(t) - 1):
                args = []
                for j in range(len(x0)):
                    args.append(x[j][i])
                for j in range(len(x0)):
                    x_new, t_new = eurler_step(f, args, t[i], h)
                    for k in range(len(x_new)):
                        x[k][i+1] = x_new[k]
                    t[i+1] = t_new 
                          
        case "RK4First":
            x = np.zeros(len(t))
            x[0] = x0[0]
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
                
        case _:
            raise SyntaxError("Incorrect method type specified")
            

                    
    return x, t

