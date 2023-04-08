import numerical_differentiation as nd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numerical_methods as nm


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots


def week20_excersises():
    
    bc_left = nd.Boundary_Condition("Dirichlet", 0.0)
    bc_right = nd.Boundary_Condition("Dirichlet", 0.0)

    def bratu(x, t, u):
        f = np.exp(2*u)
        return f

    a, b = 0, 1
    d_coef = 1
    t_final = 1
    ic = lambda x, u: 0

    grid, time, u = nd.explicit_methods(bratu, a, b, d_coef, bc_left, bc_right, ic, 20, t_final, "RK4")

    fig, ax = plt.subplots()
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")

    line, = ax.plot(grid, u[0, :])

    def animate(i):
        line.set_data((grid, u[i, :]))
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=len(time), blit=True)
    plt.show()
    
    
week20_excersises()