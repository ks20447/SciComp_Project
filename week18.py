import numerical_differentiation as nd
import numerical_methods as nm
import matplotlib.pyplot as plt


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots

def source(x):
    f = 1
    return f

bc_left = nd.Boundary_Condition("Dirichlet", 0.0)
bc_right = nd.Boundary_Condition("Dirichlet", 1.0)

a, b = 0, 1
x, u = nd.finite_difference(source, a, b, bc_left, bc_right, 10)

p, q = 0, 1
D = 1
exact = ((-1/(2*D))*(x - a)*(x - b)) + ((q - p)/(b- a))*(x - a) + p

plt.scatter(x, u, label="approx")
plt.plot(x, exact, label="exact")
nm.graph_format("$x$", "$u(x)$", "Poisson Equation with 2 Dirichlet BC's", "Poisson_Dirichlet")


    