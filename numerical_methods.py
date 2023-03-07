"""
NUMERCIAL METHODS
-----------------

Created: 23/01/2023

Author: ks20447@bristol.ac.uk (Adam Morris)

numerical_methods.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
Update documentation;
Update testing;
Implement numerical continuation, including natural paramter and pseudo-arclength;
(Optional) Add another numerical integration method

Notes:
Perhaps can move if final_h statement to shorten code further;
Add examples to documentation;
Comment
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def error_handle(f, x0, t, h, deltat_max):
    """Error handling used across several functions"""
    
    if h > deltat_max:
        raise ValueError("Given step-size exceeds maximum step-size")
    
    if isinstance(x0, (int, float)):
        x0 = [x0]
    elif isinstance(x0, (list, tuple, np.ndarray)):
        x0 = x0
    else:
        raise TypeError("x is incorrect data type")
    
    try:
        f(t, x0)
    except ValueError:
        raise TypeError(f"Function and initial condition dimesions do not match")
        
    return x0


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
    """Explicit midpoint method for any order ODE approximations

    Args:
        f (function): ODE function being solved
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize

    Returns:
        float: approximation for next x 
    """
    
    
    x = error_handle(f, x, t, h, deltat_max=0.5)
    
    dim = len(x)
    x_new = np.zeros(dim)
    
    t_new = t + h
    x_new = x + h*f(t_new + h/2, x + h/2*f(t, x))       
    
    return x_new, t_new


def eurler_step(f, x, t, h):
    """Single Euler step for any order ODE approximations

    Args:
        f (function): ODE function being approximated
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize

    Returns:
        float: approximation for next x
        float: next timestep 
    """


    x = error_handle(f, x, t, h, deltat_max=0.5)
    
    dim = len(x)
    x_new = np.zeros(dim)
    
    t_new = t + h
    x_new = x + h*f(t_new, x)

    return x_new, t_new


def runge_kutta(f, x, t, h):
    """Runge-Kutta 4th order method for any order ODE approximations

    Args:
        f (function): ODE function being approximated
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize

    Returns:
        float: approximation for next x
        float: next timestep 
    """
    
    
    x = error_handle(f, x, t, h, deltat_max=0.5)
    
    t_new = t + h
    
    dim = len(x)
    x_new = np.zeros(dim)

    k1 = h * f(t, x)
    k2 = h * f(t + h/2, x + k1/2)
    k3 = h * f(t + h/2, x + k2/2)
    k4 = h * f(t + h, x + k3)
    x_new = x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return x_new, t_new 
    
  
def solve_to(f, x0, t1, t2, h, method, deltat_max=0.5):
    """Numerically solves given ODE(s) from t1 to t2, in step-size h, with intitial condition(s) x0. 
    Second order and above ODE's must be converted to the equivalent system of first order ODE's.

    Args:
        f (function): ODE system to be solved. lambda t, x_1, ..., x_n : f(x_1, ..., x_n)
        x0 (float, array-like): Initial condition x_0 = a, or vector x = [a_1, ..., a_n] 
        t1 (float): Start time
        t2 (float): End time
        h (float): Step-size
        method (string): {'Euler', 'RK4', 'Midpoint'} Method of solving ODE
        deltat_max (float, optional): Maximum step-size allowed for solution. Defaults to 0.5.

    Raises:
        ValueError: h is larger than deltat_max. t2 is larger than t1
        TypeError: x0 should be given as an integer/float or array-like
        SyntaxError: method type did not match predefined methods

    Returns:
        array: approximation of ODE solution
        array: timestpes of ODE solution
        
    Example
    -------
    >>> def ode_second(t, x, y):
    ...    dxdt = x
    ...    dydt = y
    ... return dxdt, dydt
    >>> x0 = [1, 1]
    >>> t1, t2 = 0, 1.1
    >>> h = 0.5
    >>> x, t = nm.solve_to(ode_second, x0, t1, t2, h, "Euler")
    >>> print(x, t)
    [[1.    1.5   2.25  2.475]
    [1.    1.5   2.25  2.475]] 
    [0.  0.5 1.  1.1]
    """
        
        
    x0 = error_handle(f, x0, t1, h, deltat_max)
    
    if t1 > t2:
        raise ValueError("t2 must be greater than t1")
     
    no_steps = int((t2 - t1)/h) + 1        
    t = np.zeros(no_steps)
    t[0] = t1
    x = np.zeros((len(t), len(x0)))
    x[0, :] = x0
        
    ind = 0
    
    while ind < no_steps - 1:
        
        match method:
            
            case "Euler": 
                x[ind + 1, :], t[ind + 1] = eurler_step(f, x[ind, :], t[ind], h)
                
            case "RK4":
                x[ind + 1, :], t[ind + 1] = runge_kutta(f, x[ind, :], t[ind], h)
                
            case "Midpoint":
                x[ind + 1, :], t[ind + 1] = midpoint_method(f, x[ind, :], t[ind], h)
                
            case _:
                raise SyntaxError("Incorrect method type specified")
        
        ind += 1   
        
    final_h = ((t2 - t1)/h - int((t2- t1)/h))*h 
    
    if final_h:
        
        t = np.append(t, 0)
        x = np.concatenate((x, np.zeros((1, len(x0)))))
            
        match method:
            
            case "Euler": 
                x[ind + 1, :], t[ind + 1] = eurler_step(f, x[ind, :], t[ind], final_h)
                
            case "RK4":
                x[ind + 1, :], t[ind + 1] = runge_kutta(f, x[ind, :], t[ind], h)
                
            case "Midpoint":
                x[ind + 1, :], t[ind + 1] = midpoint_method(f, x[ind, :], t[ind], final_h)
                    
                    
    return x, t


def shooting(ode, x0, period, phase, method="Euler", h=0.01):
    """Numerical shooting to solve for ODE limit cycles

    Args:
        ode (function): ODE to be solved. lambda t, x_1, ..., x_n : f(x_1, ..., x_n) 
        x0 (float, array-like): Initial condition guess x_0 = a, or vector x = [a_1, ..., a_n]
        period (float): Initial guess for ODE limit cycle period time    
        phase (function): Phase condition. lambda p (= x1, ..., x_n): f(x_1, ..., x_n)
        method (string, optional): Method used to solve ODE. Defaults to "Euler"
        h (float, optional): Step-size to be used in ODE solution method. Defaults to 0.01

    Raises:
        RuntimeError: Root finder failed to converge 

    Returns:
        array: solution to ODE with found limit cycle conditions
        array: timesteps of ODE solution
        array: conditions of limit cycle (as seen from output)
    """
    
    
    x0 = error_handle(ode, x0, 0, h, deltat_max=0.5)  
        
    def root_ode(u):
        x0 = u[0:-1]
        t = u[-1]
        sols, time = solve_to(ode, x0, 0, t, h, method)
        f = np.zeros(len(x0))
        
        f = sols[0, :] - sols[-1, :]
        
        p = np.array(phase(x0))
        
        return np.append(f, p)
    
    x0 = np.append(x0, period)  
    x0, info, ier, msg = fsolve(root_ode, x0, full_output=True)
    
    if ier == 1:
        print(f"Root finder found the solution x = {x0[0:-1]}, period t = {x0[-1]}s after {info['nfev']} function calls")         
    else:
        raise RuntimeError(f"Root finder failed with error message: {msg}") 
    
    x, t = solve_to(ode, x0[0:-1], 0, x0[-1], 0.01, method)
    
    return x, t, x0