import numerical_methods as nm
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"text.usetex": True, 'font.size': 14}) # Use Latex fonts for all matplotlib.pyplot plots


def week17_excersises():

    # c = np.linspace(-2, 2, 11)
    # x = np.linspace(-2, 2, 51)
    # y = np.zeros((len(x), len(c)))

    # for ind, parameter in enumerate(c):
    #     equation = lambda x: x**3 - x + parameter
    #     y[:, ind] = equation(x)
        
    # for ind, y in enumerate(np.transpose(y)):
    #     plt.plot(x, y, label=f"$c = {c[ind]:.1f}$")
    # nm.graph_format("$x$", "$f(x)$", "$f(x)$ for $-2 \leq c \leq 2$", "Varying_Parameter")
    # plt.show()
    
    
    def hopf_normal_form(t, u, beta):
        sigma = -1
        u1, u2 = u
        dudt = np.array([beta*u1 - u2 + sigma*u1*(u1**2 + u2**2), u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)])
        return dudt
    def hopf_phase(p, beta):
        sigma = -1
        u1, u2 = p
        p = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
        return p 
    x0 = [1.41594778, -0.0070194] 
    period = 6.28308
    p, x = nm.continuation(hopf_normal_form, x0, period, hopf_phase, 2, -1, 11)
    print(p, x[:, 0])
    
    # plt.plot(p, x[:, 0], label="$u_{1}$")
    # plt.plot(p, x[:, 1], label="$u_{2}$")
    # nm.graph_format("$\\beta$", "$u(\\beta)$", "Natural Paramter Continuation", "Natural_Parameter")
    # plt.show()
    
    
week17_excersises()

