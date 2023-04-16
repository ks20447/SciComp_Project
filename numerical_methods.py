"""
NUMERCIAL METHODS
-----------------

Created: 23/01/2023

Author: ks20447@bristol.ac.uk (Adam Morris)

numerical_methods.py library to be used for Scientific Computing Coursework

\nThese include:
    - euler_method - performs one step of the euler numerical integration method
    - midpoint_method - performs one step of the midpoint numerical integration method
    - runge_kutta - performs one step of the Runge-Kutta 4th order numerical integration method
    - solve_to - evaluates a given ODE using one of the pre-defined numerical integration methods
    - shooting - finds equilibria/limit cycles of ODE's using numerical shooting
    - natural_parameter - investigates parameter effects on ODE equilibria/limit cycle solutions using natural parameter continuation
    - pseudo-arclength - investigates parameter effects on ODE equilibria/limit cycle solutions using pseudo-arclength continuation
    - graph_format - creates consistent graph formats for matplotlib.pyplot figures
    - error_handle - captures possible user generated errors across the included numerical methods

Potential Future Development:
Remove requirement for users to need to specify no additional arguments
Add ability to include other ODE solvers within methods, such as `scipy.integrate.solve_ivp`
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def error_handle(f, x0, t, h, args, deltat_max):
    """Error handling used across several functions to catch user input errors.

    Args:
        f (function): ODE being checked
        x0 (float, array_like): Initial conditions of ODE
        t (float): Any time value 
        h (float): Method step-size
        args (float, array-like): Additional ODE arguments
        deltat_max (float): Maximum step size

    Raises:
        ValueError: Step size is larger than deltat_max
        TypeError: Initial conditions are incorrect data type 
        ValueError: Inconsistent dimensionality of ODE and IC

    Returns:
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
    
    if args:
        if isinstance(args, (int, float, list, tuple, np.ndarray)):
            args = args
        else:
            raise TypeError("args is incorrect data type")
    
    # Checks the initial conditions watch the provided ODE
    try:
        f(t, x0, args)
    except (TypeError, ValueError):
        raise ValueError(f"Function initial condition and/or argument dimesions do not match")
        
    return dim


def graph_format(x_label : str, y_label : str, title : str, ax=None, filename=False):
    """Matplotlib.pyplot plot formatting. Sets title, axis labels, grid and legend.

    Args:
        x_label (string): x-axis label
        y_label (string): y-axis label
        title (string): title of plot
        ax (axis.Axis, optional): axis object for use with subplots
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
    """Single step of Midpoint numerical integration method.

    Args:
        f (function): ODE function being solved
        x (array): current x approximation
        t (float): current timestep 
        h (float): step-size
        args (float, array-like): Additional ODE arguments

    Returns:
        array: approximation for next x
        float: next time step
    """
    

    x_new = np.zeros(len(x))
    
    t_new = t + h
    x_new = x + h*np.asarray(f(t + h/2, x + h/2*np.asarray(f(t, x, args)), args))       
    
    return x_new, t_new


def eurler_method(f, x, t, h, args=None):
    """Single step of Euler numerical integration method.

    Args:
        f (function): ODE function being solved
        x (array): current x approximation
        t (float): current timestep 
        h (float): step-size
        args (float, array-like): Additional ODE arguments

    Returns:
        array: approximation for next x
        float: next time step
    """

    t_new = t + h
    x_new = np.zeros(len(x))
    
    x_new = x + h*np.asarray(f(t_new, x, args))

    return x_new, t_new


def runge_kutta(f, x, t, h, args=None):
    """Single step of Rugne-Kutta 4th order numerical integration method.

    Args:
        f (function): ODE function being solved
        x (array): current x approximation
        t (float): current timestep 
        h (float): step-size
        args (float, array-like): Additional ODE arguments

    Returns:
        array: approximation for next x
        float: next time step
    """
    
    
    t_new = t + h
    x_new = np.zeros(len(x))

    k1 = h * np.asarray(f(t, x, args))
    k2 = h * np.asarray(f(t + h/2, x + k1/2, args))
    k3 = h * np.asarray(f(t + h/2, x + k2/2, args))
    k4 = h * np.asarray(f(t + h, x + k3, args))
    x_new = x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return x_new, t_new 
    
  
def solve_to(ode, x0, t1: float, t2: float, h: float, method, args=None, deltat_max=0.5):
    """Numerically solves given `ode` from `t1` to `t2`, in step-size `h`, with initial condition(s) `x0`. 
    Second order and above ODE's must be converted to the equivalent system of first order ODE's.
    In the case that the time-span does not exactly divide by `h`, a final additional step will be calculated using the remainder.
    ODE's should specify that there are no additional parameters in its arguments `def ode(t, x, args=None)`.

    Args:
        ode (function): ODE system to be solved.
        x0 (float, array-like): Initial condition(s) 
        t1 (float): Start time
        t2 (float): End time
        h (float): Step-size
        method (function): Function method for solving ODE {.euler, .midpoint, .runge_kutta}
        args (float, array-like): Additional ODE arguments
        deltat_max (float, optional): Maximum step-size allowed for solution. Defaults to 0.5.

    Raises:
        ValueError: `t2` is larger than `t1`.

    Returns:
        array: approximation of ODE solution
        array: time steps of ODE solution
        
    Example
    -------
    >>> import numerical_methods as nm
    >>> def ode_second_order(t, u, args=None):
    ...    x, y = u
    ...    dudt = [x, y]
    ...    return dudt
    >>> x, t = nm.solve_to(ode_second_order, [1, 1], 0, 1, 0.1, nm.eurler_method)
    >>> print(x[:, 0], t)
    [1.         1.1        1.21       1.331      1.4641     1.61051
    1.771561   1.9487171  2.14358881 2.35794769 2.59374246] [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
    """    
    
      
    dim = error_handle(ode, x0, t1, h, args, deltat_max)
    
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
            
        x[ind + 1, :], t[ind + 1] = method(ode, x[ind, :], t[ind], h, args)
        ind += 1   
    
    final_h = ((t2 - t1)/h - int((t2- t1)/h))*h 
    
    if final_h: # In the case that (t2 - t1) exactly divides by h, we don't want to execute this part
        
        t = np.append(t, 0)
        x = np.concatenate((x, np.zeros((1, dim))))
        
        x[ind + 1, :], t[ind + 1] = method(ode, x[ind, :], t[ind], final_h, args)
                    
                    
    return x, t


def shooting(ode, x0, period: float, phase, args=None, method=eurler_method, h=0.01, output=True):
    """Numerical shooting to solve for ODE equilibria/limit cycles. Uses solve_to to produce ODE solutions. 

    Args:
        ode (function): ODE to be solved  
        x0 (float, array-like): Initial condition guess 
        period (float): Initial guess for ODE limit cycle period time    
        phase (function): Phase condition
        args (float, array-like): Additional ODE arguments
        method (function, optional): Function method for solving ODE {.euler_method, .midpoint_method, .runge_kutta}. Defaults to .euler_method
        h (float, optional): Step-size to be used in ODE solution method. Defaults to 0.01
        output (bool, optional): Toggle print output 

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
    >>> def predator_prey(t, u, args):
    ...     a, d, b = args
    ...     x, y = u
    ...     dxdt = x*(1 - x) - (a*x*y)/(d + x)
    ...     dydt = b*y*(1 - y/x)
    ...     return dxdt, dydt
    >>> def phase(p, args):
    ...     a, d, b = args
    ...     x, y = p
    ...     p = x*(1 - x) - (a*x*y)/(d + x)
    ...     return p
    >>> x0 = [0.39, 0.30]
    >>> t1, t2 = 0, 19
    >>> h = 0.01
    >>> args = [1, 0.1, 0.25]
    >>> period = t2 - t1
    >>> x, t, x0, period = nm.shooting(predator_prey, x0, period, phase, args)
    Root finder found the solution x = [0.39551779 0.29953169], period t = 18.38440240297539 after 13 function calls
    """
    
    
    dim = error_handle(ode, x0, 0, h, args, deltat_max=0.5)  
        
    def root_ode(u):
        x0, t = u[0:dim], u[-1]
        sols = solve_to(ode, x0, 0, t, h, method, args)[0]
        f = np.zeros(dim)
        
        f = sols[0, :] - sols[-1, :]
        
        p = phase(x0, args)
        
        return np.append(f, p)
    
    u = np.append(x0, period)
    u, info, ier, msg = fsolve(root_ode, u, full_output=True)
    x0, period = u[0:dim], u[-1]
    
    if ier == 1:
        if output:
            print(f"Root finder found the solution x = {x0}, period t = {period} after {info['nfev']} function calls")         
    else:
        raise RuntimeError(f"Root finder failed with error message: {msg}") 
    
    x, t = solve_to(ode, x0, 0, period, 0.01, method, args)
    
    return x, t, x0, period


def natural_parameter(ode, x0, period: float, phase, p_range: float, p_vary: int, num_steps: int, args=None, method=eurler_method, h=0.01):
    """Natural parameter continuation. Investigates single parameter effect on ODE by stepping through linearly spaced parameter values in `p_range`.
    Uses `shooting` to find solutions.

    Args:
        ode (function): ODE to be investigated
        x0 (float, array-like): Initial condition guess
        period (float): Limit cycle period guess
        phase (function): Phase condition
        p_range (array-like): Parameter range to be investigated
        p_vary (int): Index position of parameter to be changed in `args`
        num_steps (int): Number of linearly spaced steps to take in `p_range`
        args (float, array-like): Additional ODE arguments, including investigated parameter        
        method (function, optional): Function method for solving ODE {.euler_method, .midpoint_method, .runge_kutta}. Defaults to .euler_method
        h (float, optional): Step-size to be used in ODE solution method. Defaults to 0.01

    Returns:
        array: array of parameter values
        array: array of state vector solutions
        
    Example
    -------
    >>> import numerical_methods as nm
    >>> def hopf_normal_form(t, u, args):
    ...     b = args
    ...     u1, u2 = u
    ...     du1dt = b*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
    ...     du2dt = u1 + b*u2 + u2*(u1**2 + u2**2) - u2*(u1**2 + u2**2)**2
    ...     return [du1dt, du2dt]
    >>> def hopf_phase(p, args):
    ...     b = args
    ...     u1, u2 = p
    ...     p = b*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
    ...     return p  
    >>> x0 = [1.0, 0.0] 
    >>> period = 6.3
    >>> args = 2
    >>> p_range, p_vary = [2, -1], None
    >>> num_steps = 25
    >>> p, x = nm.natural_parameter(hopf_normal_form, x0, period, hopf_phase, p_range, p_vary, num_steps, args)
    >>> print(p, x[:, 0])
    [ 2.     1.875  1.75   1.625  1.5    1.375  1.25   1.125  1.     0.875
    0.75   0.625  0.5    0.375  0.25   0.125  0.    -0.125 -0.25  -0.375
    -0.5   -0.625 -0.75  -0.875 -1.   ] [ 1.41477051e+000  1.39977270e+000  1.38415813e+000  1.36786195e+000
    1.35080773e+000  1.33290434e+000  1.31404187e+000  1.29408574e+000
    1.27286845e+000  1.25017733e+000  1.22573583e+000  1.19917365e+000
    1.16997638e+000  1.13739444e+000  1.10026269e+000  1.05659364e+000
    1.00246149e+000  9.27639773e-001  7.55432478e-001 -9.37310408e-014
    1.77286750e-038 -4.08699198e-063  2.81256389e-092 -3.80191900e-122
    3.69048022e-152]
    """
    
    dim = error_handle(ode, x0, period, h, args, deltat_max=0.5)
    
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
    """Pseudo-arclength continuation. Investigates single parameter affect on ODE by pseudo-arclength prediction.  

    Args:
        ode (function): ODE to be investigated
        states (array-like): Two initial condition guesses
        periods (array-like): Both limit cycle period guesses
        phase (function): Phase condition
        parameters (array-like): Two initial parameter values. The difference between these will affect the parameter variation rate
        p_vary (int): Index position of parameter to be changed in `args`
        p_final (float): Final parameter value to evaluate at 
        args (float, array-like): Additional ODE arguments
        method (function, optional): Function method for solving ODE {.euler_method, .midpoint_method, .runge_kutta}. Defaults to .euler_method
        h (float, optional): Step-size to be used in ODE solution method. Defaults to 0.01

    Returns:
        array: array of solutions. [x_1 ... x_n period parameter]
        
    Example
    -------
    """ 
    max_iter = 100
    
    
    x0, x1 = states[0], states[1]
    t0, t1 = periods[0], periods[1]
    p0, p1 = parameters[0], parameters[1]
    
    dim = error_handle(ode, x0, t0, h, args, deltat_max=0.5)
    dim = error_handle(ode, x1, t1, h, args, deltat_max=0.5)
    
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
            
            if pred[-1] < p_final :
                break 
            
            if isinstance(args, (int, float)):
                args = pred[-1]
            else:
                args = list(args)
                args[p_vary] = pred[-1]
            
            def ode_root(v):
                x0 = v[0:dim]
                t = v[dim]
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
    