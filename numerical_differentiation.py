"""
NUMERCIAL DIFFERENTATION
------------------------

Created: 10/03/2023

Author: ks20447@bristol.ac.uk (Adam Morris)

numerical_differntiation.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
Add non-linear soruce term implementation to finite difference

Notes:
Two Neumann conditions gives singular matrix, unsure how to correct this
"""


import numpy as np


class Boundary_Condition:

    def __init__(self, name : str, value) -> None:
        self.name = name
        self.value = np.array(value)
            
    def calc_values(self, dx):
        match self.name:
            
            case "Dirichlet":
                n = 0
                self.u_bound = self.value
                self.a_bound = [-2, 1]
                self.b_bound = self.value
                self.sol_bound = 1
                
            case "Neumann":
                n = 1
                self.u_bound = 0
                self.a_bound = [-2, 2]
                self.b_bound = 2*self.value*dx
                self.sol_bound = 0
                
            case "Robin":
                n = 1
                self.u_bound = 0
                self.a_bound = [-2*(1 + self.value[1]*dx), 2]
                self.b_bound = 2*self.value[0]*dx
                self.sol_bound = 0
                
            case _:
                raise SyntaxError("Incorrect boundary condition type specified")
            
        return n


def finite_difference(source, a, b, bc_left, bc_right, n):
    
    grid = np.linspace(a, b, n+1)
    dx = (b - a) / n
    
    u = np.zeros(n + 1)
    q = np.zeros(n + 1)
    q[0::] = (dx**2)*source(grid[0::])
    
    n += bc_left.calc_values(dx) + bc_right.calc_values(dx) - 1
    
    a_dd = np.zeros((n, n))
    b_dd = np.zeros(n)
    
    u[0] = bc_left.u_bound
    u[-1] = bc_right.u_bound
    
    for i in range(n):
        a_dd[i, i] = -2
    for i in range(n-1):
        a_dd[i, i+1] = 1
        a_dd[i+1, i] = 1
        
    a_dd[0, 0:2] = bc_left.a_bound
    a_dd[-1, -2::] = np.flip(bc_right.a_bound)
        
    b_dd[0] = bc_left.b_bound
    b_dd[-1] = bc_right.b_bound
    
    start = bc_left.sol_bound
    end = bc_right.sol_bound
    
    if end:
        u[start:-end] = np.linalg.solve(a_dd, -b_dd - q[start:-end])
    else:
        u[start::] = np.linalg.solve(a_dd, -b_dd - q[start::])
   

    return grid, u