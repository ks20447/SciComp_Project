"""
Created: 07/02/2023

Author: ks20447 (Adam Morris)

testing.py file to test numerical_methods.py library and produce results

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
2) 3. Find step-sizes for each method that give you the same error - how long does each method take? (you can use the time command when running your Python script)
"""

import numerical_methods as nm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"text.usetex": True, 'font.size': 14})

def graph_format(x_label, y_label, title, filename):
    
    plt.grid()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f"results/{filename}")


def error_plot(t, exact, approx, step):
    
    error = abs(approx - exact)
    plt.loglog(t, error, label=f"Step-size {step}")
    

    return error


def euler(ode, x0, t1, t2, steps):
    
    for i in range(len(steps)):
        approx, t = nm.solve_to(ode, x0, t1, t2, steps[i], 'EulerFirst')
        exact = np.exp(t)
        plt.figure(0, figsize=(10, 5))
        error_plot(t, exact, approx, steps[i])
    
    graph_format("Time", "Error", "Euler method ODE solver with decreasing timesteps", "Euler_Error.png")


def runge_kutta(ode, x0, t1, t2, steps):
    
    for i in range(len(steps)):
        approx, t = nm.solve_to(ode, x0, t1, t2, steps[i], 'RK4First')
        exact = np.exp(t)
        plt.figure(1, figsize=(10, 5))
        error_plot(t, exact, approx, steps[i])
    
    graph_format("Time", "Error", "Runge-Kutta method ODE solver with decreasing timesteps", "RK4_Error.png")
    
    
def euler_runge(ode, x0, t1, t2, steps):
    
    for i in range(len(steps)):
        approx1, t = nm.solve_to(ode, x0, t1, t2, steps[i], 'EulerFirst')
        approx2, t = nm.solve_to(ode, x0, t1, t2, steps[i], 'RK4First')
        exact = np.exp(t)
        plt.figure(2, figsize=(10, 5))
        error_plot(t, exact, approx1, f"{steps[i]}: Euler")
        error_plot(t, exact, approx2, f"{steps[i]}: RK4")
        
    
    graph_format("Time", "Error", "Euler Method vs Runge-Kutta Error", "Error_Both.png")
            


def run():
    
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
    steps = [0.1, 0.01, 0.001, 0.0001]
    euler(ode, x0, t1, t2, steps)
    runge_kutta(ode, x0, t1, t2, steps)
    euler_runge(ode, x0, t1, t2, [0.001, 0.0001])
    
    # Solves 2nd order ODE with specified method
    x1, t = nm.solve_to(ode_second, [1, 1], t1, t2, h, "EulerSecond")
    x2, t = nm.solve_to(ode_second, [1, 1], t1, t2, h, "RK4Second")
    
    # Plots approximations against the exact ODE solution
    exact = np.cos(t) + np.sin(t)
    plt.figure(3, figsize=(10, 5))
    plt.plot(t, x1[0], label="$x(t)$: Euler Approximation")
    plt.plot(t, x2[0], label="$x(t)$: RK4 Approximation")
    plt.plot(t, exact, '--', label="$x(t) = cos(t) + sin(t)$")
    plt.plot()
    graph_format("Time $(s)$", "$x(t)$", "Numerical Solutions vs Exact Solution of 2nd Order ODE", "Euler_RK4.png")
    
    # Solution to ODE over large time 
    x1, t = nm.solve_to(ode_second, [1, 1], t1, 10, h, "EulerSecond")
    x2, t = nm.solve_to(ode_second, [1, 1], t1, 10, h, "RK4Second")
    
    plt.figure(4, figsize=(10, 5))
    plt.plot(x1[0], x1[1], label="$x(t)$: Euler Approximation")
    plt.plot(x2[0], x2[1], label="$x(t)$: RK4 Approximation")
    plt.plot()
    graph_format("$x(t)$", "$\dot{x}(t)$", "$x$ vs $\dot{x}$ for large time", "Large_Time.png")
    
    
run()

