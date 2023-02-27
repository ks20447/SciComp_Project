"""
Created: 23/01/2023

Author: ks20447 (Adam Morris)

numerical_methods.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
4) (Optional) Add another numerical integration method

Notes:
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fsolve


def error_handle(f, x0, t, h, deltat_max):
    
    if h > deltat_max:
        raise ValueError("Given step-size exceeds maximum step-size")
    
    if isinstance(x0, (int, float)):
        x0 = [x0]
    elif isinstance(x0, (list, tuple, np.ndarray)):
        x0 = x0
    else:
        raise TypeError("x is incorrect data type")
    
    try:
        f(t, *x0)
    except TypeError:
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
    """Single Euler step for any odrer ODE approximations

    Args:
        f (function): ODE function being approximated
        x (float): current x approximation
        t (float): current timestep 
        h (float): stepsize
        
    Raises:
        TypeError: x should be given as an integer/float or array-like

    Returns:
        float: approximation for next x
        float: next timestep 
    """

    x = error_handle(f, x, t, h, deltat_max=0.5)
    
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


def step_calc(t1, t2, h):
    """Calucates number of steps needed for ODE solve. Will return final stepsize value if a remainder occurs

    Args:
        t1 (flaot): initial time
        t2 (float): final time
        h (float): stepsize

    Returns:
        int: number of steps for ODE solver
        float: final step size value
    """
    
    
    no_steps = int((t2 - t1)/h)
    final_h = ((t2 - t1)/h - no_steps)*h
    if not math.isclose(final_h, 0, rel_tol=1e-6):
        no_steps += 1
    else:
        final_h = 0
        
    return no_steps, final_h

    
  
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
        ValueError: h is larger than deltat_max. t2 is larger than t1
        TypeError: x0 should be given as an integer/float or array-like
        SyntaxError: method type did not match predefined methods

    Returns:
        array: approximation of ODE solution
        array: timestpes of ODE solution
    """
        
        
    x0 = error_handle(f, x0, t1, h, deltat_max)
    
    if t1 > t2:
        raise ValueError("t2 must be greater than t1")
    
    
    no_steps, final_h = step_calc(t1, t2, h)          
    t = np.zeros(no_steps)
    t[0] = t1
    x = np.zeros((len(x0), no_steps))
    for ind, iv in enumerate(x0):
        x[ind][0] = iv
    
    match method:
        
        case "Euler":
            t_new, ind = 0, 0
            while round(t_new + h, 3) < t2:
                x_args = []
                for j in range(len(x0)):
                    x_args.append(x[j][ind])
                    
                x_new, t_new = eurler_step(f, x_args, t[ind], h)
                
                for k in range(len(x_new)):
                    x[k][ind + 1] = x_new[k]  
                t[ind + 1] = t_new
                
                ind += 1   
            
            if final_h:
                x_new, t_new = eurler_step(f, x_args, t[-2], final_h)
                
                for k in range(len(x_new)):
                    x[k][-1] = x_new[k]
                t[-1] = t_new
                 
                          
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
        f = np.zeros(len(sols))
        
        for ind, sol in enumerate(sols):
            f[ind] = sol[0] - sol[-1]
        
        p = np.array(phase(x0))
        
        return np.append(f, p)
    
    
    x0 = x0 + [period]  
    x0, info, ier, msg = fsolve(root_ode, x0, full_output=True)
    
    if ier == 1:
        print(f"Root finder found the solution x = {x0[0:-1]}, period t = {x0[-1]}s after {info['nfev']} function calls")         
    else:
        raise RuntimeError(f"Root finder failed with error message: {msg}") 
    
    x, t = solve_to(ode, x0[0:-1], 0, x0[-1], 0.01, method)
    
    return x, t, x0