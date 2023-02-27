"""
Created: 07/02/2023

Author: ks20447 (Adam Morris)

week14.py file to complete week 14 excersises 

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
2) 3. Find step-sizes for each method that give you the same error - how long does each method take? (you can use the time command when running your Python script)
"""

import numerical_methods as nm
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots


def graph_format(x_label, y_label, title, filename):
    
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"results/{filename}") 


def euler(ode, x0, t1, t2, steps):
    
    error = np.zeros(len(steps))
    
    for idx, step in enumerate(steps):
        approx, t = nm.solve_to(ode, x0, t1, t2, step, 'Euler')
        exact = np.exp(t)
        error[idx] = abs(approx[0][-1] - exact[-1])
        
    plt.figure(0, figsize=(10, 5))    
    plt.loglog(steps, error)
        
    graph_format("Step-size", "Error", "Euler method Error against Step-size", "Euler_Error.png")
    
    return error


def runge_kutta(ode, x0, t1, t2, steps):
    
    error = np.zeros(len(steps))
    
    for idx, step in enumerate(steps):
        approx, t = nm.solve_to(ode, x0, t1, t2, step, 'RK4First')
        exact = np.exp(t)
        error[idx] = abs(approx[-1] - exact[-1])
        
    plt.figure(1, figsize=(10, 5))    
    plt.loglog(steps, error)
    
    graph_format("Step-size", "Error", "RK4 method Error against Step-Size", "RK4_Error.png")
    
    return error
            

def week14_excersises():
    
    # 1st order ODE function to be solved
    def ode(t, x):
        dxdt = x
        return dxdt
    
    # 2nd order ODE function to be solved
    def ode_second(t, x1, x2):
        dx1dt = x2
        dx2dt = -x1
        return dx1dt, dx2dt
        
    # Time start, time stop, initial condition, stepsize     
    t1, t2 = 0, 1
    x0 = 1
    h = 0.01

    # Produces loglog plot for a set of stepsizes
    steps = np.logspace(-5, -2, 50)
    error_euler = euler(ode, x0, t1, t2, steps)
    error_runge = runge_kutta(ode, x0, t1, t2, steps)
    
    plt.figure(2, figsize=(10, 5))
    plt.loglog(steps, error_euler, label="Euler Method")
    plt.loglog(steps, error_runge, label="RK4 Method")
    nm.graph_format("Step-size", "Error", "RK4 vs Euler: Error against Step-Size", "Error_Both.png")
    
    # Solves 2nd order ODE with specified method
    x1, t = nm.solve_to(ode_second, [1, 1], t1, t2, h, "Euler")
    x2, t = nm.solve_to(ode_second, [1, 1], t1, t2, h, "RK4Second")
    
    # Plots approximations against the exact ODE solution
    exact = np.cos(t) + np.sin(t)
    plt.figure(3, figsize=(10, 5))
    plt.plot(t, x1[0], label="$x(t)$: Euler Approximation")
    plt.plot(t, x2[0], label="$x(t)$: RK4 Approximation")
    plt.plot(t, exact, '--', label="$x(t) = cos(t) + sin(t)$")
    plt.plot()
    nm.graph_format("Time $(s)$", "$x(t)$", "Numerical Solutions vs Exact Solution of 2nd Order ODE", "Euler_RK4.png")
    
    # Solution to ODE over large time 
    x1, t = nm.solve_to(ode_second, [1, 1], t1, 10, h, "Euler")
    x2, t = nm.solve_to(ode_second, [1, 1], t1, 10, h, "RK4Second")
    
    plt.figure(4, figsize=(10, 5))
    plt.plot(x1[0], x1[1], label="$x(t)$: Euler Approximation")
    plt.plot(x2[0], x2[1], label="$x(t)$: RK4 Approximation")
    plt.plot()
    nm.graph_format("$x(t)$", "$\dot{x}(t)$", "$x$ vs $\dot{x}$ for large time", "Large_Time.png")
    
    
week14_excersises()