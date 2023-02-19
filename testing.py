"""
Created: 19/02/2023

Author: ks20447 (Adam Morris)

testing.py file to test numerical_methods.py functionality

All commits to be pushed to "working" branch before merging to "master" branch

To be completed:
solve_to exception testing
shooting success and exception testing

Notes:
"""


import numerical_methods as nm
import numpy as np
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
    

unittest.main()