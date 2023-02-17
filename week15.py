"""
Created: 17/02/2023

Author: ks20447 (Adam Morris)

week15.py file to complete week 15 excersises

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
4) Generalise your code so that you can use arbitrary differential equations of arbitrary dimension (assume they are always in first-order form).
    How should a user pass the differential equations to your code?
    How should a user pass the phase-condition to your code?
    What options might a user want to have access to?
"""

import numerical_methods as nm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def week15_excersises():
    
    def lokta_volterra(t, x, y, b):
        a, d = 1, 0.1
        
        dxdt = x*(1 - x) - (a*x*y)/(d + x)
        dydt = b*y*(1 - y/x)
        
        return dxdt, dydt

    x1, t1 = nm.solve_to(lambda t, x, y: lokta_volterra(t, x, y, 0.25), (0.39 , 0.30), 0, 20, 0.01, "Euler")
    x2, t2 = nm.solve_to(lambda t, x, y: lokta_volterra(t, x, y, 0.30), (0.39 , 0.30), 0, 20, 0.01, "Euler")
    
    plt.figure(1, figsize=(10, 5))
    plt.plot(t1, x1[0], label="$b = 0.25$")
    plt.plot(t2, x2[0], label="$b = 0.30$")
    nm.graph_format("Time (s)", "$x(t)$", "Lokta-Volterra Equations", "Lokta_Volterra")

    plt.figure(2, figsize=(10, 5))
    plt.plot(x1[0], x1[1], label="$b = 0.25$")
    plt.plot(x2[0], x2[1], label="$b = 0.30$")
    nm.graph_format("$x(t)$", "$\dot{x}(t)$", "Lokta-Volterra Equations", "Lokta_Volterra_Cycle")
    
    # I.C (0.39, 0.30) produces limit cycle period of ~19s (b = 0.25)
    
    def num_shooting(u):
        x, y, t = u
        a, d = 1, 0.1
        b = 0.2
        
        sol, time = nm.solve_to(lambda t, x, y: lokta_volterra(t, x, y, 0.25), (x , y), 0, t, 0.01, "Euler")
        f1 = sol[0][0] - sol[0][-1]
        f2 = sol[1][0] - sol[1][-1]
        p = x*(1 - x) - (a*x*y)/(d + x) # Phase Condition
        
        return f1, f2, p
    
    x0 = [0.39, 0.30, 19]
    x0, info, ier, msg = fsolve(num_shooting, x0, full_output=True)
    if ier == 1:
        print(f"Root finder found the solution x={x0} after {info['nfev']} function calls; the norm of the final residual is {np.linalg.norm(info['fvec'])}")         
    else:
        print(f"Root finder failed with error message: {msg}")
        return
    
    x0 = np.around(x0, decimals=3)
    x, t = nm.solve_to(lambda t, x, y: lokta_volterra(t, x, y, 0.25), (x0[0] , x0[1]), 0, x0[2], 0.01, "Euler")
    
    plt.figure(3, figsize=(10, 5))
    plt.plot(t, x[0], label="$b = 0.25$")
    nm.graph_format("Time (s)", "$x(t)$", "Lokta-Volterra Equations", "Lokta_Volterra_Exact")
    
    plt.figure(4, figsize=(10, 5))
    plt.plot(x[0], x[1], label="$b = 0.25$")
    nm.graph_format("$x(t)$", "$\dot{x}(t)$", "Lokta-Volterra Equations", "Lokta_Volterra_Cycle_Exact")



week15_excersises()