"""
NUMERCIAL DIFFERENTATION
------------------------

Created: 10/03/2023

Author: ks20447@bristol.ac.uk (Adam Morris)

numerical_differntiation.py library to be used for Scientific Computing Coursework

\nIncluded:
    - `Boundary_Condition` - Creates boundary value objects
    - `construct_a_b_matricies` - Constructs matrix equation equivalents for ODE/PDE's 
    - `finite_difference` - Performs finite difference method on 2nd order ODE's
    - `explicit_methods` - Perform explicit numerical differentiation methods on PDE's
    - `implicit_methods` - Perform implicit numerical differentiation methods on PDE's
    - `imex` - Perform implicit-explicit numerical differentiation methods on non-linear PDE's

Potential Future Development:
Re-use `numerical_methods.py` functions for appropriate methods
"""


import numpy as np
from math import ceil


class Boundary_Condition:
    """Boundary Condition (BC) object for use with numerical differentiation methods.
    """

    def __init__(self, name : str, value : float) -> None:
        """Initialise Boundary Condition (BC) object with given name and value.

        Args:
            name (str): {"Dirichlet", "Neumann", "Robin"} Type of BC
            value (float, array-like): Value of BC. If "Robin" type BC is defined, must be array-like of two values
            
        EXAMPLE
        -------
        >>> bc_left = Boundary_Condition("Dirichlet", 0.0) 
        >>> bc_left = Boundary_Condition("Neumann", 0.5) 
        >>> bc_right = Boundary_Condition("Robin", [1.0, 1.0])
        """
        
        
        self.name = name
        self.value = np.asarray(value)
            
    def calc_values(self, dx):
        """Calculates appropriate values to be used in matrix construction depending on BC type.

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


def construct_a_b_matricies(grid, bc_left : Boundary_Condition, bc_right : Boundary_Condition):
    """Constructs A and B matricies that represent defined ODE/PDE for use with numerical differentiation.

    Args:
        grid (array-like): Discretization grid
        bc_left (object): Initial (left) BC
        bc_right (object): Final (Right) BC

    Returns:
        array: A matrix
        array: B matrix
    """
    
    a = grid[0]
    b = grid[-1]
    n = len(grid) - 1
    dx = (b - a) / n
    
    n += bc_left.calc_values(dx) + bc_right.calc_values(dx) - 1 # Forumla to determine size of matrices depending on BC types 
    
    a_mat = np.zeros((n, n))
    b_mat = np.zeros(n)
    
    for i in range(n):
        a_mat[i, i] = -2
    for i in range(n-1):
        a_mat[i, i+1] = 1
        a_mat[i+1, i] = 1
        
    a_mat[0, 0:2] = bc_left.a_bound
    a_mat[-1, -2::] = np.flip(bc_right.a_bound)
        
    b_mat[0] = bc_left.b_bound
    b_mat[-1] = bc_right.b_bound
    
    return a_mat, b_mat
    

def finite_difference(source, a : float, b : float, bc_left : Boundary_Condition, bc_right : Boundary_Condition, n : int, args):
    """Finite difference method to solve 2nd order ODE's with source term, two boundary conditions (BC) in n steps.

    Args:
        source (function): Source term of ODE
        a (float): x value that the left BC is evaluated at
        b (float): x value that the right BC is evaluated at
        bc_left (object): Initial (left) BC
        bc_right (object): Final (Right) BC
        n (int): Number of steps
        args (float, array-like): Additional ODE parameters

    Returns:
        array: linearly spaced grid values from `a` to `b`
        array: solution to ODE
        
    EXAMPLE
    -------
    >>> def source(x, u, args):
    ...     D = args
    ...     f = 1/D
    ...     return f
    >>> bc_left = nd.Boundary_Condition("Dirichlet", 0.0)
    >>> bc_right = nd.Boundary_Condition("Dirichlet", 1.0)
    >>> a, b = 0, 1
    >>> x, u = nd.finite_difference(source, a, b, bc_left, bc_right, 10, 1)
    >>> print(x, u)
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
        q[:] = (dx**2)*source(grid[:], u_old[:], args)
        
        if end:
            u[start:-end] = np.linalg.solve(a_dd, -b_dd - q[start:-end])
        else:
            u[start::] = np.linalg.solve(a_dd, -b_dd - q[start::])
            
        itr += 1
        

    return grid, u


def explicit_methods(source, a : float, b : float, d_coef : float,
                     bc_left : Boundary_Condition, bc_right : Boundary_Condition, 
                     ic, n : float, t_final : float, method : str, args):
    """Explicit numerical differentiation used to solve 2nd Order PDE's from time 0 to time `t_final`, in `n` spacial steps, with an initial condition. 

    Args:
        source (func): PDE source term
        a (float): x value that the left BC is evaluated at
        b (float): x value that the right BC is evaluated at
        d_coef (float): Diffusion coefficient
        bc_left (object): Initial (left) BC
        bc_right (object): Final (Right) BC
        ic (func): Initial condition
        n (int): Number of steps
        t_final (float): Time to solve PDE until (from zero)
        method (str): Explicit method type. {'Euler', 'RK4'}
        args (float, array-like): Additional PDE parameters

    Returns:
        array: spacial grid
        array: time grid
        array: solution to PDE 
        
    EXAMPLE
    -------    
    >>> def pde(x, t, u, args):
    ...     return 1
    >>> a, b = 0, 1
    >>> d_coef = 0.1
    >>> bc_left = nd.Boundary_Condition("Dirichlet", 1.0)
    >>> bc_right = nd.Boundary_Condition("Neumann", 0.0)
    >>> ic = lambda x, args: 0
    >>> n = 10
    >>> t_final = 1
    >>> args = 2
    >>> grid, time, u = nd.explicit_methods(pde, a, b, d_coef, bc_left, bc_right, ic, n, t_final, "Euler", args)
    >>> print(grid, time, u[-1, :])        
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ] [0.    0.049 0.098 0.147 0.196 0.245 0.294 0.343 0.392 0.441 0.49  0.539
    0.588 0.637 0.686 0.735 0.784 0.833 0.882 0.931 0.98 ] [1.         1.13089095 1.19115512 1.19458151 1.17707801 1.13556941
    1.10204063 1.06303271 1.04156582 1.02192404 1.02022076]
    """
    
    grid = np.linspace(a, b, n+1)
    dx = (b - a) / n
    
    c = 0.49
    dt = c*dx**2 / d_coef
    
    num_time = ceil(t_final / dt)
    time = dt * np.arange(num_time)
    
    u = np.zeros((num_time, n + 1))
    u[0, :] = ic(grid, args)
    
    a_dd, b_dd = construct_a_b_matricies(grid, bc_left, bc_right)
    
    u[:, 0] = bc_left.u_bound
    u[:, -1] = bc_right.u_bound
    
    start = bc_left.sol_bound
    end = bc_right.sol_bound
    
    pde = lambda x, t, u: (d_coef/(dx**2))*(a_dd @ u + b_dd) + source(x, t, u, args)
    
    match method:
        
        case "Euler":
            if end:
                
                grid_sol = grid[start:-end]
                
                for i in range(num_time - 1):
                    u_sol = u[i, start:-end]
                    u[i + 1, start:-end] = u_sol + dt*pde(grid_sol, time[i], u_sol)
            else:
                
                grid_sol = grid[start::]
                
                for i in range(num_time - 1):
                    u_sol = u[i, start::]
                    u[i + 1, start::] = u_sol + dt*pde(grid_sol, time[i], u_sol)

                            
        case "RK4":
            if end:
                
                grid_sol = grid[start:-end]
                
                for i in range(num_time - 1):
                    u_sol = u[i, start:-end]
                    k1 = pde(grid_sol, time[i], u_sol)
                    k2 = pde(grid_sol, time[i], u_sol + dt*k1/2)
                    k3 = pde(grid_sol, time[i], u_sol + dt*k2/2)
                    k4 = pde(grid_sol, time[i], u_sol + dt*k3)
                    u[i + 1, start:-end] = u_sol + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
            else:
                
                grid_sol = grid[start::]
                
                for i in range(num_time - 1):
                    u_sol = u[i, start::]
                    k1 = pde(grid_sol, time[i], u_sol)
                    k2 = pde(grid_sol, time[i], u_sol + dt*k1/2)
                    k3 = pde(grid_sol, time[i], u_sol + dt*k2/2)
                    k4 = pde(grid_sol, time[i], u_sol + dt*k3)
                    u[i + 1, start::] = u_sol + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
                        
    return grid, time, u
    
    
def implicit_methods(source, a : float, b : float, d_coef : float,
                     bc_left : Boundary_Condition, bc_right : Boundary_Condition,
                     ic, n : int, dt : float, t_final : float, method : str, args):
    """Implicit numerical differentiation used to solve 2nd Order PDE's from time 0 to time `t_final`, in `n` spacial steps, `dt` time step-size with an initial condition.
    *Note: Should only be used for linear PDE systems* 

    Args:
        source (func): PDE source term
        a (float): x value that the left BC is evaluated at
        b (float): x value that the right BC is evaluated at
        d_coef (float): Diffusion coefficient
        bc_left (object): Initial (left) BC
        bc_right (object): Final (Right) BC
        ic (func): Initial condition
        n (int): Number of steps
        dt (float): Time step-size
        t_final (float): Time to solve PDE until (from zero)
        method (str): Explicit method type. {'Euler', 'Crank-Nicolson'}
        args (float, array-like): Additional PDE parameters

    Returns:
        array: spacial grid
        array: time grid
        array: solution to PDE 
    
    EXAMPLE
    -------
    >>> def pde(x, t, u, args):
    ...     return 1
    >>> a, b = 0, 1
    >>> d_coef = 0.1
    >>> bc_left = nd.Boundary_Condition("Dirichlet", 1.0)
    >>> bc_right = nd.Boundary_Condition("Neumann", 0.0)
    >>> ic = lambda x, args: 0
    >>> n = 10
    >>> t_final = 1
    >>> args = 2        
    >>> grid, time, u = nd.implicit_methods(pde, a, b, d_coef, bc_left, bc_right, ic, n, dt, t_final, "Crank-Nicolson", args)
    >>> print(grid, time, u[-1, :])        
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ] [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9] [1.         3.72357893 5.64365386 6.94500804 7.79100529 8.31743467
    8.63030718 8.80727205 8.90140592 8.94600416 8.95899802]    
    """
    
    
    grid = np.linspace(a, b, n+1)
    dx = (b - a) / n
    
    c = (dt*d_coef)/(dx**2)
    
    num_time = ceil(t_final / dt)
    time = dt * np.arange(num_time)
    
    u = np.zeros((num_time, n + 1))
    u[0, :] = ic(grid, args)
    
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
                grid_sol = grid[start:-end]
                for i in range(num_time - 1):
                    u_sol = u[i, start:-end]
                    b = u_sol + c*b_dd + dt*source(grid_sol, time[i], u_sol, args)
                    u[i + 1, start:-end] = np.linalg.solve(a, b)
                    
            else:
                grid_sol = grid[start::]
                for i in range(num_time - 1):
                    u_sol = u[i, start::]
                    b = u_sol + c*b_dd + source(grid_sol, time[i], u_sol, args)
                    u[i + 1, start::] = np.linalg.solve(a, b)

        case "Crank-Nicolson":
            
            a = identity - (c/2)*a_dd
            
            if end:
                grid_sol = grid[start:-end]
                for i in range(num_time - 1):
                    u_sol = u[i, start:-end]
                    b = ((identity + (c/2)*a_dd) @ u_sol) + c*b_dd + source(grid_sol, time[i], u_sol, args)
                    u[i + 1, start:-end] = np.linalg.solve(a, b)
                    
            else:
                grid_sol = grid[start::]
                for i in range(num_time - 1):
                    u_sol = u[i, start::]
                    b = ((identity + (c/2)*a_dd) @ u_sol) + c*b_dd + source(grid_sol, time[i], u_sol, args)
                    u[i + 1, start::] = np.linalg.solve(a, b) 
                                            
               
    return grid, time, u


def imex(source, a : float, b : float, d_coef : float,
         bc_left : Boundary_Condition, bc_right : Boundary_Condition,
         ic, n : int, dt : float, t_final : float, args):
    """Implicit-Explicit Euler method used to solve 2nd Order PDE's from time 0 to time `t_final`, in `n` spacial steps, `dt` time step-size with an initial condition.

    Args:
        source (func): PDE source term
        a (float): x value that the left BC is evaluated at
        b (float): x value that the right BC is evaluated at
        d_coef (float): Diffusion coefficient
        bc_left (object): Initial (left) BC
        bc_right (object): Final (Right) BC
        ic (func): Initial condition
        n (int): Number of steps
        dt (float): Time step-size
        t_final (float): Time to solve PDE until (from zero)
        args (float, array-like): Additional PDE parameters

    Returns:
        array: spacial grid
        array: time grid
        array: solution to PDE 
        
    EXAMPLE
    -------
    >>> import numerical_differentiation as nd
    >>> def pde(x, t, u, args):
    ...     return u**2
    >>> a, b = 0, 1
    >>> d_coef = 1.0
    >>> bc_left = nd.Boundary_Condition("Dirichlet", 1.0)
    >>> bc_right = nd.Boundary_Condition("Dirichlet", 1.0)
    >>> ic = lambda x, args: 1
    >>> n = 10
    >>> dt, t_final = 0.1, 1
    >>> args = None
    >>> grid, time, u = nd.imex(pde, a, b, d_coef, bc_left, bc_right, ic, n, dt, t_final, args)
    >>> print(grid, time, u[-1, :])
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ] [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9] [1.         1.060927   1.10259983 1.13343125 1.15311294 1.15992139
    1.15311294 1.13343125 1.10259983 1.060927   1.        ]
    """
    
    
    grid = np.linspace(a, b, n+1)
    dx = (b - a) / n
    
    c = (dt*d_coef)/(dx**2)
    
    num_time = ceil(t_final / dt)
    time = dt * np.arange(num_time)
    
    u = np.zeros((num_time, n + 1))
    u[0, :] = ic(grid, args)
    
    a_dd, b_dd = construct_a_b_matricies(grid, bc_left, bc_right)
    identity = np.identity(len(a_dd))
    
    u[:, 0] = bc_left.u_bound
    u[:, -1] = bc_right.u_bound
    
    start = bc_left.sol_bound
    end = bc_right.sol_bound
    
    a = identity - c*a_dd
    
    pde = lambda x, t, u: (d_coef/(dx**2))*(a_dd @ u + b_dd) + source(x, t, u, args)

    if end:
        grid_sol = grid[start:-end]
        for i in range(num_time - 1):
            u_sol = u[i, start:-end]
            b = u_sol + c*b_dd + dt*source(grid_sol, time[i], u_sol, args)
            u_inter = np.linalg.solve(a, b)
            u[i + 1, start:-end] = u_inter + dt*pde(grid_sol, time[i], u_inter)
            
            
    else:
        grid_sol = grid[start::]
        for i in range(num_time - 1):
            u_sol = u[i, start::]
            b = u_sol + c*b_dd + source(grid_sol, time[i], u_sol, args)
            u_inter = np.linalg.solve(a, b)
            u[i + 1, start::] = u_inter + dt*pde(grid_sol, time[i], u_inter)
    
    
    return grid, time, u