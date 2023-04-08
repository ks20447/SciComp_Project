"""
NUMERCIAL DIFFERENTATION
------------------------

Created: 10/03/2023

Author: ks20447@bristol.ac.uk (Adam Morris)

numerical_differntiation.py library to be used for Scientific Computing Coursework

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:

Notes:
Two Neumann conditions gives singular matrix - this is a mathematical issue to do with constant of intergration
Can however verify solutions by setting dirichlet and neumann and then flipping
"""


import numpy as np
from math import ceil


class Boundary_Condition:
    """Boundary Condition (BC) object to be used with finite_differnce function
    """

    def __init__(self, name : str, value : float) -> None:
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
        self.value = value
            
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


def construct_a_b_matricies(grid, bc_left, bc_right):
    
    a = grid[0]
    b = grid[-1]
    n = len(grid) - 1
    dx = (b - a) / n
    
    n += bc_left.calc_values(dx) + bc_right.calc_values(dx) - 1 # Forumla to determine size of matrices depending on BC types 
    
    a_dd = np.zeros((n, n))
    b_dd = np.zeros(n)
    
    for i in range(n):
        a_dd[i, i] = -2
    for i in range(n-1):
        a_dd[i, i+1] = 1
        a_dd[i+1, i] = 1
        
    a_dd[0, 0:2] = bc_left.a_bound
    a_dd[-1, -2::] = np.flip(bc_right.a_bound)
        
    b_dd[0] = bc_left.b_bound
    b_dd[-1] = bc_right.b_bound
    
    return a_dd, b_dd
    

def finite_difference(source, a : float, b : float, bc_left : Boundary_Condition, bc_right : Boundary_Condition, n : int):
    """Finite difference method to solve 2nd order PDE with source term, two boundary conditions (bc) in n steps

    Args:
        source (function): source term of PDE. Function that returns a single value (can be dependant on x)
        a (float): x value that the left bc is evaluated at
        b (float): x value that the right bc is evaluated at
        bc_left (object): Initial (left) bc
        bc_right (object): Final (Right) bc
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
    
    a_dd, b_dd = construct_a_b_matricies(grid, bc_left, bc_right)
    
    u[0] = bc_left.u_bound
    u[-1] = bc_right.u_bound
    
    start = bc_left.sol_bound
    end = bc_right.sol_bound
    
    if end:
        u[start:-end] = np.linalg.solve(a_dd, -b_dd - q[start:-end])
    else:
        u[start::] = np.linalg.solve(a_dd, -b_dd - q[start::])
        
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


def explicit_methods(source, a, b, d_coef, bc_left, bc_right, ic, n, t_final, method):
    """Method of lines using Runge-Kutta 4th order to solve (linear) diffusion equation from time 0 to time t_final, in n spacial steps

    Args:
        source (func): NOT YET IMPLEMENTED. Source PDE term
        a (float): x value that the left bc is evaluated at
        b (float): x value that the right bc is evaluated at
        d_coef (float): Diffusion coefficient
        bc_left (object): Initial (left) bc
        bc_right (object): Final (Right) bc
        ic (func): Initial condition of PDE
        n (int): Number of steps
        t_final (float): Time to solve PDE until (from zero)

    Returns:
        array: spacial grid
        array: time grid
        array: solution to PDE 
    """
    
    grid = np.linspace(a, b, n+1)
    dx = (b - a) / n
    
    c = 0.49
    dt = c * dx**2 / d_coef
    
    num_time = ceil(t_final / dt)
    time = dt * np.arange(num_time)
    
    u = np.zeros((num_time, n + 1))
    u[0, :] = ic(grid, u)
    
    a_dd, b_dd = construct_a_b_matricies(grid, bc_left, bc_right)
    
    u[:, 0] = bc_left.u_bound
    u[:, -1] = bc_right.u_bound
    
    start = bc_left.sol_bound
    end = bc_right.sol_bound
    
    pde = lambda x, t, u: (d_coef/(dx**2))*(a_dd @ u + b_dd) + source(x, t, u)
    
    
    match method:
        
        case "Euler":
            if end:
                
                sol_grid = grid[start:-end]
                
                for i in range(num_time - 1):
                    sol_u = u[i, start:-end]
                    u[i + 1, start:-end] = sol_u + dt*pde(sol_grid, time[i], sol_u)
            else:
                
                sol_grid = grid[start::]
                
                for i in range(num_time - 1):
                    sol_u = u[i, start::]
                    u[i + 1, start::] = sol_u + dt*pde(sol_grid, time[i], sol_u)

                            
        case "RK4":
            if end:
                
                sol_grid = grid[start:-end]
                
                for i in range(num_time - 1):
                    sol_u = u[i, start:-end]
                    k1 = pde(sol_grid, time[i], sol_u)
                    k2 = pde(sol_grid, time[i], sol_u + dt*k1/2)
                    k3 = pde(sol_grid, time[i], sol_u + dt*k2/2)
                    k4 = pde(sol_grid, time[i], sol_u + dt*k3)
                    u[i + 1, start:-end] = sol_u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                
                sol_grid = grid[start:-end]
                
                for i in range(num_time - 1):
                    sol_u = u[i, start::]
                    k1 = pde(sol_grid, time[i], sol_u)
                    k2 = pde(sol_grid, time[i], sol_u + dt*k1/2)
                    k3 = pde(sol_grid, time[i], sol_u + dt*k2/2)
                    k4 = pde(sol_grid, time[i], sol_u + dt*k3)
                    u[i + 1, start::] = sol_u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                        
    return grid, time, u
    
    
def implicit_methods(source, a, b, d_coef, bc_left, bc_right, ic, n, dt, t_final, method):
    
    grid = np.linspace(a, b, n+1)
    dx = (b - a) / n
    
    c = (dt*d_coef)/(dx**2)
    
    num_time = ceil(t_final / dt)
    time = dt * np.arange(num_time)
    
    u = np.zeros((num_time, n + 1))
    u[0, :] = ic(grid, u)
    
    a_dd, b_dd = construct_a_b_matricies(grid, bc_left, bc_right)
    identity = np.identity(len(a_dd))
    
    u[:, 0] = bc_left.u_bound
    u[:, -1] = bc_right.u_bound
    
    start = bc_left.sol_bound
    end = bc_right.sol_bound

    
    match method:
        
        case "Euler":
            
            a = identity - c*a_dd
    
            if end:
                for i in range(num_time - 1):
                    b = u[i, start:-end] + c*b_dd + c*source(grid[start:-end], time[i])
                    u[i + 1, start:-end] = np.linalg.solve(a, b)
                    
            else:
                for i in range(num_time - 1):
                    b = u[i, start::] + c*b_dd + c*source(grid[start::], time[i])
                    u[i + 1, start::] = np.linalg.solve(a, b)

        case "Crank-Nicolson":
            
            a = identity - (c/2)*a_dd
            
            if end:
                for i in range(num_time - 1):
                    b = ((identity + (c/2)*a_dd) @ u[i, start:-end]) + c*b_dd + c*source(grid[start:-end], time[i])
                    u[i + 1, start:-end] = np.linalg.solve(a, b)
            else:
                for i in range(num_time - 1):
                    b = ((identity + (c/2)*a_dd) @ u[i, start::]) + c*b_dd + c*source(grid[start::], time[i])
                    u[i + 1, start::] = np.linalg.solve(identity - (c/2)*a_dd, 
                                            (identity + (c/2)*a_dd) @ u[i, start::] + c*b_dd)
            
            
    return grid, time, u