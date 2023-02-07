import numerical_methods as nm
import matplotlib.pyplot as plt
import numpy as np


def run():
    
    # Example 2nd order ODE function to be solved
    def func_second(t, x1, x2):
        dx2dt = -x1
        return dx2dt
    
    x0_second = [0, 1]
    ode_second = func_second
    h = 0.01
    t1, t2 = 0, 3
    x_second_e, t = nm.solve_to(ode_second, x0_second, t1, t2, h, "EulerSecond")
    x_second, t = nm.solve_to(ode_second, x0_second, t1, t2, h, "RK4Second")
    plt.plot(t, x_second_e[0], label="Approx Euler")
    plt.plot(t, x_second[0], label="Approx RK4")
    plt.plot(t, np.sin(t), label='Exact')
    plt.legend()
    plt.show()


    
run()









