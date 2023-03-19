"""
NUMERCIAL DIFFERENTATION
------------------------

Created: 10/03/2023

Author: ks20447@bristol.ac.uk (Adam Morris)

numerical_differntiation.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:

Notes:
Two Neumann conditions gives singular matrix, unsure how to correct this
"""


import numpy as np


class Boundary_Condition:
    """Boundary Condition (BC) object to be used with finite_differnce function
    """


    def __init__(self, name : str, value) -> None:
        """Initialise Boundary Condition (BC) object with given name and value

        Args:
            name (str): {"Dirichlet", "Neumann", "Robin"} Type of BC
            value (float, array-like): Value of BC. If "Robin" type BC is defined, must be array-like of two values
            
        EXAMPLE
        -------
        >>> bc_left = Boundary_Condition("Dirichlet", 0.0)  
        >>> bc_right = Boundary_Condition("Robin", [1.0, 1.0])
        """
        
        
        self.name = name
        self.value = np.array(value)
            
    def calc_values(self, dx):
        """Calculates appropriate values to be used in finite difference matrix construction depending on BC

        Args:
            dx (float): Discretization step-size

        Raises:
            SyntaxError: Incorrect BC type specified

        Returns:
            int: factor to determine size of finite difference matricies
            
        EXAMPLE
        -------
        >>> bc_left = Boundary_Condition("Dirichlet", 0.0)  
        >>> bc_right = Boundary_Condition("Robin", [1.0, 1.0])
        >>> n = 10
        >>> n += bc_left.calc_values(dx) + bc_right.calc_values(dx) - 1 # Forumla to determine size of matrices depending on BC types
        >>> print(n)
        11
        """
        
        
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


def finite_difference(source, a : float, b : float, bc_left : Boundary_Condition, bc_right : Boundary_Condition, n : int):
    """Finite difference method to solve 2nd order PDE with source term, two boundary conditions (bc) in n steps

    Args:
        source (function): source term of PDE. Function that returns a single value (can be dependant on x)
        a (float): x value that the left bc is evaluated at
        b (float): x value that the right bc is evaluated at
        bc_left (object): Initial (left) boundary conition
        bc_right (object): Final (Right) boundary conition
        n (int): Number of steps

    Returns:
        array: linearly spaced grid values from a to b
        array: solution to PDE
        
    EXAMPLE
    -------
    >>> def source(x, u):
    ...     f = 1
    ...     return f
    >>> bc_left = nd.Boundary_Condition("Dirichlet", 0.0)
    >>> bc_right = nd.Boundary_Condition("Dirichlet", 1.0)
    >>> a, b = 0, 1
    >>> x, u = nd.finite_difference(source, a, b, bc_left, bc_right, 10)
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
    [0.    0.145 0.28  0.405 0.52  0.625 0.72  0.805 0.88  0.945 1.   ]
    """
    
    
    grid = np.linspace(a, b, n+1)
    dx = (b - a) / n
    
    u = np.zeros(n + 1)
    q = np.zeros(n + 1)
    
    n += bc_left.calc_values(dx) + bc_right.calc_values(dx) - 1 # Forumla to determine size of matrices depending on BC types 
    
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
        
    u_old = np.zeros_like(u)
    
    itr = 0
        
    while np.max(np.abs(u - u_old)) > 1e-6 and itr < 100:
        u_old[:] = u[:]
        q[:] = (dx**2)*source(grid[:], u_old[:])
        
        if end:
            u[start:-end] = np.linalg.solve(a_dd, -b_dd - q[start:-end])
        else:
            u[start::] = np.linalg.solve(a_dd, -b_dd - q[start::])
            
        itr += 1
        

    return grid, u