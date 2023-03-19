import numerical_differentiation as nd
import numerical_methods as nm
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots

def week18_excersises():

    def source_linear(x, u):
        f = 1
        return f

    def source_non_linear(x, u):
        f = np.exp(u)
        return f

    bc_left = nd.Boundary_Condition("Dirichlet", 0.0)
    bc_right = nd.Boundary_Condition("Dirichlet", 1.0)

    a, b = 0, 1
    x, u_lin = nd.finite_difference(source_linear, a, b, bc_left, bc_right, 10)

    p, q = 0, 1
    D = 1
    exact = ((-1/(2*D))*(x - a)*(x - b)) + ((q - p)/(b- a))*(x - a) + p

    plt.scatter(x, u_lin, label="approx")
    plt.plot(x, exact, label="exact")
    nm.graph_format("$x$", "$u(x)$", "Poisson Equation with 2 Dirichlet BC's", "Poisson_Dirichlet")
    plt.show()

    x, u_non = nd.finite_difference(source_non_linear, a, b, bc_left, bc_right, 10)
    plt.scatter(x, u_non, label="approx")
    nm.graph_format("$x$", "$u(x)$", "Poisson Equation with non-Linear Source Term", "Poisson_Non_Linear")
    plt.show()


week18_excersises()    