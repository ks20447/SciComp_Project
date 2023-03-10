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
import math
import unittest


class NumericalMethodsTesting(unittest.TestCase):
           
        
    def test_euler_success(self):
        x1, t1 = nm.eurler_step(ode_first, 1, 0, 0.01)
        x, t = nm.eurler_step(ode_higher, [1, 1, 1], 0, 0.01)
        self.assertEqual(x1, 1.01)
        self.assertEqual(t1, 0.01)
        for x in x:
            self.assertEqual(x, 1.01)
        self.assertEqual(t, 0.01)
        
    
    def test_euler_exceptions(self):
        with self.assertRaises(ValueError) as exception_context:
            nm.eurler_step(ode_first, 0, 0, 10)
        self.assertEqual(str(exception_context.exception),
            "Given step-size exceeds maximum step-size")
        
        with self.assertRaises(TypeError) as exception_context:
            nm.eurler_step(ode_first, "test", 0, 0.01)
        self.assertEqual(str(exception_context.exception),
            "x is incorrect data type")
        
        with self.assertRaises(ValueError) as exception_context:
            nm.eurler_step(ode_second, 0, 0, 0.01)
        self.assertEqual(str(exception_context.exception),
            "Function and initial condition dimesions do not match")
    
    
    def test_midpoint_success(self):
        x1, t1 = nm.midpoint_method(ode_first, 1, 0, 0.01)
        x, t = nm.midpoint_method(ode_higher, [1, 1, 1], 0, 0.01)
        self.assertEqual(x1, 1.01005)
        self.assertEqual(t1, 0.01)
        for x in x:
            self.assertEqual(x, 1.01005)
        self.assertEqual(t, 0.01)
        
    
    def test_runge_kutta_success(self):
        x1, t1 = nm.runge_kutta(ode_first, 1, 0, 0.01)
        x, t = nm.runge_kutta(ode_higher, [1, 1, 1], 0, 0.01)
        self.assertEqual(x1, 1.0100501670833333)
        self.assertEqual(t1, 0.01)
        for x in x:
            self.assertEqual(x, 1.0100501670833333)
        self.assertEqual(t, 0.01)
        
    
    def test_solve_to_euler_success(self):
        x1, t1 = nm.solve_to(ode_first, 1, 0, 1, 0.01, "Euler")
        x, t = nm.solve_to(ode_higher, [1, 1, 1], 0, 1, 0.01, "Euler")
        self.assertEqual(x1[0][0], 1)
        self.assertEqual(t1[0], 0)
        self.assertAlmostEqual(x1[-1][0], math.e, 1)
        self.assertAlmostEqual(t1[-1], 1)
        for x in np.transpose(x):
            self.assertEqual(x[0], 1)
            self.assertAlmostEqual(x[-1], math.e, 1)
        self.assertEqual(t[0], 0)
        self.assertAlmostEqual(t[-1], 1)
        
    
    def test_solve_to_midpoint_success(self):
        x1, t1 = nm.solve_to(ode_first, 1, 0, 1, 0.01, "Midpoint")
        x, t = nm.solve_to(ode_higher, [1, 1, 1], 0, 1, 0.01, "Midpoint")
        self.assertEqual(x1[0][0], 1)
        self.assertEqual(t1[0], 0)
        self.assertAlmostEqual(x1[-1][0], math.e, 4)
        self.assertAlmostEqual(t1[-1], 1)
        for x in np.transpose(x):
            self.assertEqual(x[0], 1)
            self.assertAlmostEqual(x[-1], math.e, 4)
        self.assertEqual(t[0], 0)
        self.assertAlmostEqual(t[-1], 1)
     
        
    def test_solve_to_runge_kutta_success(self):
        x1, t1 = nm.solve_to(ode_first, 1, 0, 1, 0.01, "RK4")
        x, t = nm.solve_to(ode_higher, [1, 1, 1], 0, 1, 0.01, "RK4")
        self.assertEqual(x1[0][0], 1)
        self.assertEqual(t1[0], 0)
        self.assertAlmostEqual(x1[-1][0], math.e, 9)
        self.assertAlmostEqual(t1[-1], 1)
        for x in np.transpose(x):
            self.assertEqual(x[0], 1)
            self.assertAlmostEqual(x[-1], math.e, 9)
        self.assertEqual(t[0], 0)
        self.assertAlmostEqual(t[-1], 1)
        
    
    def test_solve_to_exceptions(self):
        with self.assertRaises(ValueError) as exception_context:
            nm.solve_to(ode_higher, [1, 1, 1], 0, 1, 1, "Euler")
        self.assertEqual(str(exception_context.exception),
            "Given step-size exceeds maximum step-size")

        with self.assertRaises(TypeError) as exception_context:
            nm.solve_to(ode_higher, "test", 0, 1, 0.01, "Euler")
        self.assertEqual(str(exception_context.exception),
            "x is incorrect data type")
        
        with self.assertRaises(ValueError) as exception_context:
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
        x, t, x0, period = nm.shooting(hopf, [1, 0], 6.3, phase)
        for x in np.transpose(x):
            self.assertAlmostEqual(x[0], x[-1])
        
    
    def test_shooting_exceptions(self):
        with self.assertRaises(ValueError) as exception_context:
            nm.shooting(hopf, [1, 0], 6.3, phase, h=1)
        self.assertEqual(str(exception_context.exception),
            "Given step-size exceeds maximum step-size")

        with self.assertRaises(TypeError) as exception_context:
            nm.shooting(hopf, "test", 6.3, phase)
        self.assertEqual(str(exception_context.exception),
            "x is incorrect data type")
        
        with self.assertRaises(ValueError) as exception_context:
            nm.shooting(hopf, [1, 0, 0], 6.3, phase)
        self.assertEqual(str(exception_context.exception),
            "Function and initial condition dimesions do not match")
        
        with self.assertRaises(SyntaxError) as exception_context:
            nm.shooting(hopf, [1, 0], 6.3, phase, method="test")
        self.assertEqual(str(exception_context.exception),
            "Incorrect method type specified")
        

def ode_first(t, x):
    dxdt = x
    return dxdt


def ode_second(t, u):
    x, y = u
    dudt = [x, y]
    return dudt


def ode_higher(t, u):
    x, y, z = u
    dudt = [x, y, z]
    return dudt


def hopf_normal_form(t, u, sigma, beta):
    u1, u2 = u
    dudt = [beta*u1 - u2 + sigma*u1*(u1**2 + u2**2), u1 + beta*u2 + sigma*u2*(u1**2 + u2**2)]
    return dudt


def hopf_phase(p, sigma, beta):
    u1, u2 = p
    p = beta*u1 - u2 + sigma*u1*(u1**2 + u2**2)
    return p 


hopf = lambda t, x: hopf_normal_form(t, x, -1, 1)
phase = lambda p: hopf_phase(p, -1, 1)


unittest.main()
