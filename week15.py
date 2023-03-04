"""
Created: 17/02/2023

Author: ks20447 (Adam Morris)

week15.py file to complete week 15 excersises

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
"""

import numerical_methods as nm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots


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
    
    x0 = [0.39, 0.30]
    
    def phase(p):
        a, d = 1, 0.1
        x, y = p
        p = x*(1 - x) - (a*x*y)/(d + x)
        
        return p
        
    x0 = [0.39, 0.30]
    x, t, x0 = nm.shooting(lambda t, x, y: lokta_volterra(t, x, y, 0.25), x0, 19, phase)
    
    plt.figure(3, figsize=(10, 5))
    plt.plot(t, x[0], label="$b = 0.25$")
    nm.graph_format("Time (s)", "$x(t)$", "Lokta-Volterra Equations", "Lokta_Volterra_Exact")
    
    plt.figure(4, figsize=(10, 5))
    plt.plot(x[0], x[1], label="$b = 0.25$")
    nm.graph_format("$x(t)$", "$\dot{x}(t)$", "Lokta-Volterra Equations", "Lokta_Volterra_Cycle_Exact")


week15_excersises()