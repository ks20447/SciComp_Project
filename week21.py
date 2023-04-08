import numerical_differentiation as nd
import matplotlib.pyplot as plt
import numpy as np
import time
import cProfile


def week21_excercises():
    
    def pde_nonlinear(x, t, u):
        f = 0
        return f
    
    
    def pde_linear(x, t):
        f = 0
        return f
    
    bc_left = nd.Boundary_Condition("Dirichlet", 0.0)
    bc_right = nd.Boundary_Condition("Dirichlet", 0.0)
    
    ic = lambda x, u: np.sin(x*np.pi)
    a, b = 0, 1
    d_coef = 0.1
    t_final = 2
    
    # start = time.perf_counter()
    # grid_1, time_1, u_exp_euler = nd.explicit_methods(pde_nonlinear, a, b, d_coef, bc_left, bc_right, ic, 100, t_final, "Euler")
    # elapsed = time.perf_counter() - start
    # print(f"{elapsed} seconds")
    
    # start = time.perf_counter()
    # grid_2, time_2, u_runge_kuta = nd.explicit_methods(pde_nonlinear, a, b, d_coef, bc_left, bc_right, ic, 100, t_final, "RK4")
    # elapsed = time.perf_counter() - start
    # print(f"{elapsed} seconds")
        
    # start = time.perf_counter()
    # grid_3, time_3, u_imp_euler = nd.implicit_methods(pde_linear, a, b, d_coef, bc_left, bc_right, ic, 100, 0.1, t_final, "Euler")
    # elapsed = time.perf_counter() - start
    # print(f"{elapsed} seconds")   
    
    # start = time.perf_counter()
    # grid_4, time_4, u_crank = nd.implicit_methods(pde_linear, a, b, d_coef, bc_left, bc_right, ic, 100, 0.1, t_final, "Crank-Nicolson")
    # elapsed = time.perf_counter() - start
    # print(f"{elapsed} seconds")    
    
    cp = cProfile.Profile()
    cp.enable()
    
    grid_2, time_2, u_runge_kuta = nd.explicit_methods(pde_nonlinear, a, b, d_coef, bc_left, bc_right, ic, 100, t_final, "RK4")
    
    cp.disable()
    cp.print_stats(sort="cumtime")
    
    
week21_excercises()