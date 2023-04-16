"""
Created: 19/02/2023

Author: ks20447 (Adam Morris)

testing.py file to test numerical_methods.py functionality

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:

Notes:
"""


import numerical_methods as nm
import numpy as np
import unittest


class NumericalMethodsTesting(unittest.TestCase):
           
        
    def test_euler_success(self):
        x_new, t_new = nm.eurler_method(ode, [1, 1, 1], 0, 0.01, [1, 1, 1]) 
        np.testing.assert_array_equal(x_new, [1.0001, 1.0001, 1.0001])
        self.assertEqual(t_new, 0.01)
        
        
    def test_midpoint_success(self):
        x_new, t_new = nm.midpoint_method(ode, [1, 1, 1], 0, 0.01, [1, 1, 1])
        np.testing.assert_array_equal(x_new, [1.00005, 1.00005, 1.00005])
        self.assertEqual(t_new, 0.01)
        
    
    def test_runge_kutta_success(self):
        x_new, t_new = nm.runge_kutta(ode, [1, 1, 1], 0, 0.01, [1, 1, 1])
        np.testing.assert_array_equal(x_new, [1.0000500012500209, 1.0000500012500209, 1.0000500012500209])
        self.assertEqual(t_new, 0.01)
    
    
    def test_solve_to_success(self):
        x, t = nm.solve_to(ode, [1, 1, 1], 0, 1, 0.01, nm.eurler_method, [1, 1, 1])
        np.testing.assert_allclose(x[-1, :], np.exp(1/2), 0, 1e-2)
        self.assertAlmostEqual(t[-1], 1)
        
        x, t = nm.solve_to(ode, [1, 1, 1], 0, 1, 0.01, nm.midpoint_method, [1, 1, 1])
        np.testing.assert_allclose(x[-1, :], np.exp(1/2), 0, 1e-4)
        self.assertAlmostEqual(t[-1], 1)
        
        x, t = nm.solve_to(ode, [1, 1, 1], 0, 1, 0.01, nm.runge_kutta, [1, 1, 1])
        np.testing.assert_allclose(x[-1, :], np.exp(1/2), 0, 1e-10)
        self.assertAlmostEqual(t[-1], 1)
        
    
    def test_solve_to_exceptions(self):  
        with self.assertRaises(ValueError) as exception_context:
            nm.solve_to(ode, [1, 1, 1], 1, 0, 0.01, nm.eurler_method, [1, 1, 1])
        self.assertEqual(str(exception_context.exception),
            "t2 must be greater than t1")
        
    
    def test_shooting_success(self):
        phase = lambda u, args: u[1]
        x, t, x0, period = nm.shooting(oscillator, [1, 1], 2*np.pi, phase, 1, output=False)
        np.testing.assert_allclose(x[0, :], x[-1, :], 0, 1e-10)
        self.assertAlmostEqual(t[-1], period)
        np.testing.assert_allclose(x0, 0, 0, 1e-10)
        self.assertAlmostEqual(period, 2*np.pi, 6)
        
    
    def test_natural_parameter_success(self):
        phase = lambda u, b: b*u[0] - u[1] - u[0]*(u[0]**2 + u[1]**2)
        p, x = nm.natural_parameter(hopf, [1, 1], 6.3, phase, [2, -1], 0, 10, 2)


    def test_pseudo_arclength_success(self):
        phase = lambda u, b: b*u[0] - u[1] - u[0]*(u[0]**2 + u[1]**2)
        v = nm.pseudo_arclength(hopf, [[1, 1], [1, 1]], [6.3, 6.3], phase, [2, 1.9], 0, -1, 2)
        
        
    def test_excpetions(self):
        with self.assertRaises(ValueError) as exception_context:
            nm.error_handle(ode, [1, 1], 0, 0.01, [1, 1, 1], 0.5)
        self.assertEqual(str(exception_context.exception),
            "Function initial condition and/or argument dimesions do not match")
        
        with self.assertRaises(TypeError) as exception_context:
            nm.error_handle(ode, "test", 0, 0.01, [1, 1, 1], 0.5)
        self.assertEqual(str(exception_context.exception),
            "x is incorrect data type")

        with self.assertRaises(ValueError) as exception_context:
            nm.error_handle(ode, [1, 1, 1], 0, 1, [1, 1, 1], 0.5)
        self.assertEqual(str(exception_context.exception),
            "Given step-size exceeds maximum step-size")

        with self.assertRaises(TypeError) as exception_context:
            nm.error_handle(ode, [1, 1, 1], 0, 0.01, "test", 0.5)
        self.assertEqual(str(exception_context.exception),
            "args is incorrect data type")                


def ode(t, u, args):
    x, y, z = u
    p, q, r = args
    dxdt = p*t*x
    dydt = q*t*y
    dzdt = r*t*z
    return [dxdt, dydt, dzdt]


def oscillator(t, u, args):
    x, y = u
    w = args
    dxdt = y
    dydt = -(w**2)*x
    return [dxdt, dydt]


def hopf(t, u, args):
    b = args
    u1, u2 = u
    du1dt = b*u1 - u2 - u1*(u1**2 + u2**2)
    du2dt = u1 + b*u2 - u2*(u1**2 + u2**2)
    return [du1dt, du2dt]


unittest.main()
