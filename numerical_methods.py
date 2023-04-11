"""
NUMERCIAL METHODS
-----------------

Created: 23/01/2023

Author: ks20447@bristol.ac.uk (Adam Morris)

numerical_methods.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
Check all documentation

Notes:
Remove requirment for users to need to specify no additional arguments 
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def error_handle(f, x0, t, h, args, deltat_max, full=True):
    """Error handling used across several functions

    Args:
        f (function): ODE being checked
        x0 (float, array_like): Initial conditions
        t (float): Time value (often initial time value)
        h (float): Step-size
        args (float, array-like): Additional ODE arguments
        deltat_max (float): Maximum step size

    Raises:
        ValueError: Step size is too large
        TypeError: Initial conditions are incorrect data type 
        ValueError: Inconsistent dimensionality of ODE and IC

    Returns:
        float, array-like: Initial conditions
        int: Dimension of ODE
    """
    
    # Checks that the provided step size is sufficiently small
    if h > deltat_max:
        raise ValueError("Given step-size exceeds maximum step-size")
    
    # Checks that provided initial conditions are the correct data type
    if isinstance(x0, (int, float)):
        dim = 1
    elif isinstance(x0, (list, tuple, np.ndarray)):
        dim = len(x0)
    else:
        raise TypeError("x is incorrect data type")
    
    if isinstance(args, (int, float, list, tuple, np.ndarray)):
        args = args
    else:
        raise TypeError("args is incorrect data type")
    
    # Checks the initial conditions watch the provided ODE
    try:
        f(t, x0, args)
    except (TypeError, ValueError):
        raise ValueError(f"Function initial condition and/or argument dimesions do not match")
        
    return x0, args, dim


def graph_format(x_label : str, y_label : str, title : str, ax=None, filename=False):
    """Matplotlib.pyplot plot formatting

    Args:
        x_label (string): x-axis label
        y_label (string): y-axis label
        title (string): title of plot
        ax (axis.Axis): axis object for use with subplots
        filename (string, optional): name given to saved .png plot
    """
    
    if not ax:
        plt.grid()
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
    else:
        ax.grid()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
    if filename:
        plt.savefig(f"results/{filename}") 
      

def midpoint_method(f, x, t, h, args=None):
    """Explicit midpoint method for any order ODE approximations

    Args:
        f (function): ODE function being solved
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize
        args (float, array-like): Additional ODE arguments

    Returns:
        float: approximation for next x 
    """
    

    x_new = np.zeros(len(x))
    
    t_new = t + h
    x_new = x + h*np.asarray(f(t_new + h/2, x + h/2*np.asarray(f(t, x, args)), args))       
    
    return x_new, t_new


def eurler_method(f, x, t, h, args=None):
    """Single Euler step for any order ODE approximations

    Args:
        f (function): ODE function being approximated
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize
        args (float, array-like): Additional ODE arguments

    Returns:
        float: approximation for next x
        float: next timestep 
    """

    t_new = t + h
    x_new = np.zeros(len(x))
    
    x_new = x + h*np.asarray(f(t_new, x, args))

    return x_new, t_new


def runge_kutta(f, x, t, h, args=None):
    """Runge-Kutta 4th order method for any order ODE approximations

    Args:
        f (function): ODE function being approximated
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize
        args (float, array-like): Additional ODE arguments

    Returns:
        float: approximation for next x
        float: next timestep 
    """
    
    
    t_new = t + h
    x_new = np.zeros(len(x))

    k1 = h * np.asarray(f(t, x, args))
    k2 = h * np.asarray(f(t + h/2, x + k1/2, args))
    k3 = h * np.asarray(f(t + h/2, x + k2/2, args))
    k4 = h * np.asarray(f(t + h, x + k3, args))
    x_new = x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return x_new, t_new 
    
  
def solve_to(f, x0, t1: float, t2: float, h: float, method, args=None, deltat_max=0.5):
    """Numerically solves given ODE from t1 to t2, in step-size h, with intitial condition(s) x0. 
    Second order and above ODE's must be converted to the equivalent system of first order ODE's.
    In the case that the time-span does not exatly divide by h, a final additional step will be calculated using the remainder.
    ODE's should specify that there are no additonal parameters in its arguments `def ode(t, x, args=None)`

    Args:
        f (function): ODE system to be solved.
        x0 (float, array-like): Initial condition(s) 
        t1 (float): Start time
        t2 (float): End time
        h (float): Step-size
        method (function): Function method for solving ODE {.euler, .midpoint, .runge_kutta}
        args (float, array-like): Additional ODE arguments
        deltat_max (float, optional): Maximum step-size allowed for solution. Defaults to 0.5.

    Raises:
        ValueError: h is larger than deltat_max. t2 is larger than t1. Function and initial condition dimesions do not macth
        TypeError: x0 should be given as an integer/float or array-like
        SyntaxError: method type did not match predefined methods

    Returns:
        array: approximation of ODE solution(s)
        array: timestpes of ODE solution
        
    Example
    -------
    >>> import numerical_methods as nm
    >>> def ode_second_order(t, u, args=None):
    ...     x, y = u
    ...     dudt = [x, y]
    ... return dudt
    >>> x, t = nm.solve_to(ode_second_order, [1, 1], 0, 1, 0.1, "Euler")
    >>> print(x[:, 0], t)
    [1.         1.1        1.21       1.331      1.4641     1.61051
    1.771561   1.9487171  2.14358881 2.35794769 2.59374246] [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
    """    
    
      
    x0, args, dim = error_handle(f, x0, t1, h, args, deltat_max)
    
    # Ensures that the initial time is before the final time
    if t1 > t2:
        raise ValueError("t2 must be greater than t1")
     
    no_steps = int((t2 - t1)/h) + 1        
    t = np.zeros(no_steps)
    t[0] = t1
    x = np.zeros((len(t), dim))
    x[0, :] = x0
        
    ind = 0
    
    while ind < no_steps - 1:
            
        x[ind + 1, :], t[ind + 1] = method(f, x[ind, :], t[ind], h, args)
        ind += 1   
    
    final_h = ((t2 - t1)/h - int((t2- t1)/h))*h 
    
    if final_h: # In the case that t2 exactly divides by h, we don't want to execute this part
        
        t = np.append(t, 0)
        x = np.concatenate((x, np.zeros((1, dim))))
        
        x[ind + 1, :], t[ind + 1] = method(f, x[ind, :], t[ind], final_h, args)
                    
                    
    return x, t


def shooting(ode, x0, period: float, phase, args=None, method=eurler_method, h=0.01, output=True):
    """Numerical shooting to solve for ODE limit cycles. Uses solve_to to produce ODE solutions 

    Args:
        ode (function): ODE to be solved  
        x0 (float, array-like): Initial condition guess 
        period (float): Initial guess for ODE limit cycle period time    
        phase (function): Phase condition
        args (float, array-like): Additional ODE arguments
        method (function, optional): Function method for solving ODE {.euler_method, .midpoint_method, .runge_kutta}. Defaults to .euler_method
        h (float, optional): Step-size to be used in ODE solution method. Defaults to 0.01

    Raises:
        RuntimeError: Root finder failed to converge 

    Returns:
        array: solution to ODE with found limit cycle conditions
        array: timesteps of ODE solution
        array: initial conditions of limit cycle (as seen from output)
        float: period of limit cycle (as seen from output)
        
    Example
    -------
    >>> import numerical_methods as nm
    >>> def hopf_normal_form(t, u, sigma, beta):
    ...     u1, u2 = u
    ...     dudt = [beta*u1 - u2 + sigma*u1*(u1**2 + u2**2), u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)]
    ...     return dudt
    >>> def hopf_phase(p, sigma, beta):
    ...     u1, u2 = p
    ...     p = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    ...     return p 
    >>> sigma, beta = -1, 1
    >>> hopf = lambda t, u: hopf_normal_form(t, u, sigma, beta)
    >>> phase = lambda p: hopf_phase(p, sigma, beta)
    >>> u, t, u0, period = nm.shooting(hopf, [1, 0], 6.3, phase)
    Root finder found the solution x = [ 1.00247384 -0.00499102], period t = 6.283080704684171s after 9 function calls
    """
    
    
    x0, args, dim = error_handle(ode, x0, 0, h, args, deltat_max=0.5)  
        
    def root_ode(u):
        x0, t = u[0:2], u[-1]
        sols = solve_to(ode, x0, 0, t, h, method, args)[0]
        f = np.zeros(dim)
        
        f = sols[0, :] - sols[-1, :]
        
        p = phase(x0, args)
        
        return np.append(f, p)
    
    u = np.append(x0, period)
    u, info, ier, msg = fsolve(root_ode, u, full_output=True)
    x0, period = u[0:2], u[-1]
    
    if ier == 1:
        if output:
            print(f"Root finder found the solution x = {x0}, period t = {period}s after {info['nfev']} function calls")         
    else:
        raise RuntimeError(f"Root finder failed with error message: {msg}") 
    
    x, t = solve_to(ode, x0, 0, period, 0.01, method, args)
    
    return x, t, x0, period


def natural_parameter(ode, x0, period: float, phase, p_range: float, p_vary: int, num_steps: int, args=None, method=eurler_method, h=0.01):
    """Natural parameter continuation investigating single parameter affect on ODE 

    Args:
        ode (function): ODE to be solved
        x0 (float, array-like): Initial state vector
        period (float): Limit cycle period guess
        phase (function): Phase condition
        p0 (float): Initial parameter value
        p1 (float): Final parameter value
        num_steps (int): Number of linearly placed steps to take between p0 and p1
        args (float, array-like): Additional ODE arguments        
        method (function, optional): Function method for solving ODE {.euler_method, .midpoint_method, .runge_kutta}. Defaults to .euler_method
        h (float, optional): Step-size to be used in ODE solution method. Defaults to 0.01

    Returns:
        array: array of parameter values
        array: array of state vector solutions
        
    Example
    -------
    import numerical_methods as nm
    >>> def hopf_normal_form(t, u, beta):
    ...     sigma = -1
    ...     u1, u2 = u
    ...     dudt = np.array([beta*u1 - u2 + sigma*u1*(u1**2 + u2**2), u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)])
    ...     return dudt
    >>> def hopf_phase(p, beta):
    ...     sigma = -1
    ...     u1, u2 = p
    ...     p = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    ...     return p 
    >>> x0 = [1.41594778, -0.0070194] 
    >>> period = 6.28308
    >>> p, x = nm.continuation(hopf_normal_form, x0, period, hopf_phase, 2, -1, 11)
    >>> print(p, x[:, 0])
    [ 2.   1.7  1.4  1.1  0.8  0.5  0.2 -0.1 -0.4 -0.7 -1. ] 
    [ 1.41594778e+00  1.30572653e+00  1.18529974e+00  1.05116564e+00    8.97197290e-01  7.10617176e-01  
    4.52758525e-01 -1.60753498e-16  3.93514178e-39  9.31616179e-61  5.74885115e-78]
    """
    
    x0, args, dim = error_handle(ode, x0, period, h, args, deltat_max=0.5)
    
    p = np.linspace(p_range[0], p_range[1], num_steps)
    x = np.zeros((len(p), dim))

    try:
        for ind, arg in enumerate(p):
            if isinstance(args, (int, float)):
                args = arg
            else:
                args = list(args)
                args[p_vary] = arg
            x_sol, t_sol, x0, period = shooting(ode, x0, period, phase, args, method, h, output=False)
            x[ind, :] = x0
    except (RuntimeError, ValueError):
        print(f"Failed to converge at parameter value {arg}", "\n Hint: vary num_steps or starting parameter value")
        return p[0:ind], x[0:ind]
        
    return p, x 


def pseudo_arclength(ode, states, periods, phase, parameters, p_vary, p_final, args=None, method=eurler_method, h=0.01):
    """Pseudo-arclength continuation investigating single parameter affect on ODE 

    Args:
        ode (function): ODE to be solved
        states (array-like): Two initial state vectors
        periods (array-like): Both limit cycle period guesses
        phase (function): Phase condition
        parameters (array-like): Two initial parameter values
        num_steps (int): Number of pseudo-arclength operations to take
        args (float, array-like): Additional ODE arguments
        method (function, optional): Function method for solving ODE {.euler_method, .midpoint_method, .runge_kutta}. Defaults to .euler_method
        h (float, optional): Step-size to be used in ODE solution method. Defaults to 0.01

    Returns:
        array: array of solutions. [x0 ... xn period parameter]
        
    Example
    -------
    >>> def hopf_normal_form(t, u, beta):
    ...     sigma = -1
    ...     u1, u2 = u
    ...     dudt = np.array([beta*u1 - u2 + sigma*u1*(u1**2 + u2**2), u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)])
    ...     return dudt
    >>> def hopf_phase(p, beta):
    ...     sigma = -1
    ...     u1, u2 = p
    ...     p = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    ...     return p 
    >>> x0 = [1.41594778, -0.0070194] 
    >>> t0 = 6.28308
    >>> p0 = 2 
    >>> x1 = [1.38018573, -0.00684505]
    >>> t1 = 6.28308
    >>> p1 = 1.9
    >>> v = nm.pseudo_arclength(hopf_normal_form, [x0, x1], [t0, t1], hopf_phase, [p0, p1], 5)    
    >>> print(v)
    [[ 1.41594778 -0.0070194   6.28308     2.        ]
    [ 1.38018573 -0.00684505  6.28308     1.9       ]
    [ 1.34347207 -0.00666584  6.2830807   1.80034032]
    [ 1.30598713 -0.00648261  6.2830807   1.70096479]
    [ 1.26748393 -0.00629417  6.2830807   1.60197335]]
    """ 
    max_iter = 100
    
    t0, t1 = periods[0], periods[1]
    p0, p1 = parameters[0], parameters[1]
    
    x0, args, dim = error_handle(ode, states[0], t0, h, args, deltat_max=0.5)
    x1, args, dim = error_handle(ode, states[1], t1, h, args, deltat_max=0.5)
    
    if isinstance(args, (int, float)):
        args0 = p0
        args1 = p1
    else:
        args = list(args)
        args[p_vary] = p0
        args0 = args
        args[p_vary] = p1
        args1 = args
    
    x0, t0 = shooting(ode, x0, t0, phase, args0, output=False)[2::]
    x1, t1 = shooting(ode, x1, t1, phase, args1, output=False)[2::]
    
    v0 = np.append(x0, [t0, p0])
    v1 = np.append(x1, [t1, p1])
    v = np.zeros((max_iter, dim + 2))
    v[0, :], v[1, :] = v0, v1
    ind = 1
    
    try:
        while  ind < max_iter:
            secant = v[ind] - v[ind - 1]
            pred =  v[ind] + secant
            
            if v[ind, -1] > v[ind - 1, -1] or pred[-1] < p_final :
                break 
            
            if isinstance(args, (int, float)):
                args = pred[-1]
            else:
                args = list(args)
                args[p_vary] = pred[-1]
            
            def ode_root(v):
                x0 = v[0:2]
                t = v[2]
                sols = solve_to(ode, x0, 0, t, h, method, args)[0]
                f = np.zeros(dim)
                
                f = sols[0, :] - sols[-1, :]
                p = phase(x0, args)
                arc = np.dot(v - pred, secant)
                
                return np.append(f, [p, arc])
            
            v[ind + 1, :], info, ier, msg = fsolve(ode_root, pred, full_output=True) 
            ind += 1          
    except (ValueError, IndexError):
        print(f"Root finding stopped after {ind + 2} steps at p = {pred[-1]}")
        return v[0:ind+1]
        
    return v[0:ind]
    