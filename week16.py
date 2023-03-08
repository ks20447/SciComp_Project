import numerical_methods as nm
import numpy as np
import matplotlib.pyplot as plt


def hopf_normal_form(t, u, sigma, beta):
    u1, u2 = u
    dudt = np.array([beta*u1 - u2 + sigma*u1*(u1**2 + u2**2), u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)])
    return dudt


def hopf_phase(p, sigma, beta):
    u1, u2 = p
    p = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    return p 


sigma = -1
beta = 1
u0 = np.array([1, 0])
t1, t2 = 0, 6.3
phase = lambda p: hopf_phase(p, sigma, beta)
u, t, u0 = nm.shooting(lambda t, u: hopf_normal_form(t, u, sigma, beta), u0, t2, phase)


















