import numerical_differentiation as nd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numerical_methods as nm


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots


def week20_excersises():
    
    bc_left = nd.Boundary_Condition("Dirichlet", 0.0)
    bc_right = nd.Boundary_Condition("Dirichlet", 0.0)

    def source_linear(x, u):
        f = 0
        return f

    a, b = 0, 1
    d_coef = 0.5
    t_final = 1
    ic = lambda x: np.sin(np.pi*(x - a)/(b - a))

    grid, time, u = nd.method_of_lines(source_linear, a, b, d_coef, bc_left, bc_right, ic, 20, t_final)

    fig, ax = plt.subplots()
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("u")

    line, = ax.plot(grid[1:-1], u[0, :])
    line_exact = ax.plot(grid[1:-1], np.exp(-((d_coef*(np.pi**2)*0)/((b-a)**2)))*np.sin(np.pi*(grid[1:-1] - a)/(b - a)))
    line_exact_end = ax.plot(grid[1:-1], np.exp(-((d_coef*(np.pi**2)*t_final)/((b-a)**2)))*np.sin(np.pi*(grid[1:-1] - a)/(b - a)))

    def animate(i):
        line.set_data((grid[1:-1], u[i, :]))
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=len(time), blit=True)
    plt.show()
    
    
week20_excersises()