"""
Created: 17/02/2023

Author: ks20447 (Adam Morris)

week15.py file to complete week 15 excersises

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
2) Determine an appropriate phase-condition for the limit cycle
3) Construct the shooting root-finding problem for the predator-prey example; check that it can find the periodic orbit found in 1.
4) Generalise your code so that you can use arbitrary differential equations of arbitrary dimension (assume they are always in first-order form).
    How should a user pass the differential equations to your code?
    How should a user pass the phase-condition to your code?
    What options might a user want to have access to?
"""

import numerical_methods as nm
import matplotlib.pyplot as plt


def week15_excersises():
    
    def lokta_volterra(t, x, y, b):
        a, d = 1, 0.1
        
        dxdt = x*(1 - x) - (a*x*y)/(d + x)
        dydt = b*y*(1 - y/x)
        
        return dxdt, dydt


    x1, t1 = nm.solve_to(lambda t, x, y: lokta_volterra(t, x, y, 0.25), (1 , 1), 0, 250, 0.01, "Euler")
    x2, t2 = nm.solve_to(lambda t, x, y: lokta_volterra(t, x, y, 0.27), (1 , 1), 0, 250, 0.01, "Euler")
    
    plt.figure(1, figsize=(10, 5))
    plt.plot(t1, x1[0], label="$b = 0.25$")
    plt.plot(t2, x2[0], label="$b = 0.27$")
    nm.graph_format("Time (s)", "$x(t)$", "Lokta-Volterra Equations", "Lokta_Volterra")

    plt.figure(2, figsize=(10, 5))
    plt.plot(x1[0], x1[1], label="$b = 0.25$")
    plt.plot(x2[0], x2[1], label="$b = 0.27$")
    nm.graph_format("$x(t)$", "$\dot{x}(t)$", "Lokta-Volterra Equations", "Lokta_Volterra_Cycle") 
    

week15_excersises()