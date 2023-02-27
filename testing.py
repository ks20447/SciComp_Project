"""
Created: 19/02/2023

Author: ks20447 (Adam Morris)

testing.py file to test numerical_methods.py functionality

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:

Notes:
"""


import numerical_methods as nm
import math
import unittest


class NumericalMethodsTesting(unittest.TestCase):
           
        
    def test_euler_success(self):
        first_x, first_t = nm.eurler_step(ode_first, 0, 0, 0.01)
        higer_x, higher_t = nm.eurler_step(ode_higher, [0, 0, 0], 0, 0.01)
        self.assertEqual(first_x, 0)
        self.assertEqual(first_t, 0.01)
        for x in higer_x:
            self.assertEqual(x, 0)
        self.assertEqual(higher_t, 0.01)
        
    
    def test_euler_exceptions(self):
        with self.assertRaises(ValueError) as exception_context:
            nm.eurler_step(ode_first, 0, 0, 10)
        self.assertEqual(str(exception_context.exception),
            "Given step-size exceeds maximum step-size")
        
        with self.assertRaises(TypeError) as exception_context:
            nm.eurler_step(ode_first, "test", 0, 0.01)
        self.assertEqual(str(exception_context.exception),
            "x is incorrect data type")
        
        with self.assertRaises(TypeError) as exception_context:
            nm.eurler_step(ode_second, 0, 0, 0.01)
        self.assertEqual(str(exception_context.exception),
            "Function and initial condition dimesions do not match")
    
    
    def test_midpoint_success(self):
        first_x, first_t = nm.midpoint_method(ode_first, 0, 0, 0.01)
        self.assertEqual(first_x, 0)
        self.assertEqual(first_t, 0.01)
        
    
    def test_runge_kutta_first_success(self):
        first_x, first_t = nm.runge_kutta(ode_first, 0, 0, 0.01)
        self.assertEqual(first_x, 0)
        self.assertEqual(first_t, 0.01)
        
    
    def test_runge_kutta_second_success(self):
        test_second_x, test_sceond_y, test_second_t = nm.runge_kutta_second(ode_second, 0, 0, 0, 0.01)
        self.assertEqual(test_second_x, 0)
        self.assertEqual(test_sceond_y, 0)
        self.assertEqual(test_second_t, 0.01)
        
    
    def test_solve_to_success(self):
        x1, t1 = nm.solve_to(ode_first, 1, 0, 1, 0.01, "Euler")
        self.assertEqual(x1[0][0], 1)
        self.assertEqual(t1[0], 0)
        self.assertTrue(math.isclose(x1[0][-1], math.e, rel_tol=1e-1, abs_tol=0.0))
        
        x, t = nm.solve_to(ode_higher, [1, 1, 1], 0, 1, 0.01, "Euler")
        for x in x:
            self.assertEqual(x[0], 1)
            self.assertTrue(math.isclose(x[-1], math.e, rel_tol=1e-1, abs_tol=0.0))
        self.assertEqual(t1[0], 0)
        
    
    def test_solve_to_exceptions(self):
        with self.assertRaises(ValueError) as exception_context:
            nm.solve_to(ode_higher, [1, 1, 1], 0, 1, 1, "Euler")
        self.assertEqual(str(exception_context.exception),
            "Given step-size exceeds maximum step-size")

        with self.assertRaises(TypeError) as exception_context:
            nm.solve_to(ode_higher, "test", 0, 1, 0.01, "Euler")
        self.assertEqual(str(exception_context.exception),
            "x is incorrect data type")
        
        with self.assertRaises(TypeError) as exception_context:
            nm.solve_to(ode_higher, [1, 1], 0, 1, 0.01, "Euler")
        self.assertEqual(str(exception_context.exception),
            "Function and initial condition dimesions do not match")
        
        with self.assertRaises(ValueError) as exception_context:
            nm.solve_to(ode_higher, [1, 1, 1], 1, 0, 0.01, "Euler")
        self.assertEqual(str(exception_context.exception),
            "t2 must be greater than t1")
        
        with self.assertRaises(SyntaxError) as exception_context:
            nm.solve_to(ode_higher, [1, 1, 1], 0, 1, 0.01, "test")
        self.assertEqual(str(exception_context.exception),
            "Incorrect method type specified")
      
        
    def test_shooting_success(self):
        x, t, x0 = nm.shooting(ode, [1, 0], 6.3, phase)
        self.assertTrue(math.isclose(x[0][-1], x[0][-1]))
        self.assertTrue(math.isclose(x[1][-1], x[1][-1]))
        
    
    def test_solve_to_exceptions(self):
        with self.assertRaises(ValueError) as exception_context:
            nm.shooting(ode, [1, 0], 6.3, phase, h=1)
        self.assertEqual(str(exception_context.exception),
            "Given step-size exceeds maximum step-size")

        with self.assertRaises(TypeError) as exception_context:
            nm.shooting(ode, "test", 6.3, phase)
        self.assertEqual(str(exception_context.exception),
            "x is incorrect data type")
        
        with self.assertRaises(TypeError) as exception_context:
            nm.shooting(ode, [1, 0, 0], 6.3, phase)
        self.assertEqual(str(exception_context.exception),
            "Function and initial condition dimesions do not match")
        
        with self.assertRaises(SyntaxError) as exception_context:
            nm.shooting(ode, [1, 0], 6.3, phase, method="test")
        self.assertEqual(str(exception_context.exception),
            "Incorrect method type specified")
        

def ode_first(t, x):
    dxdt = x
    return dxdt


def ode_second(t, x, y):
    dxdt = x
    dydt = y
    return dxdt, dydt


def ode_higher(t, x, y, z):
    dxdt = x
    dydt = y
    dzdt = z
    return dxdt, dydt, dzdt


def hopf_normal_form(t, u1, u2, sigma, beta):
    du1dt = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)
    return du1dt, du2dt


def hopf_phase(p, sigma, beta):
    u1, u2 = p
    p = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    return p 


ode = lambda t, x1, x2: hopf_normal_form(t, x1, x2, -1, 1)
phase = lambda p: hopf_phase(p, -1, 1)


unittest.main()
