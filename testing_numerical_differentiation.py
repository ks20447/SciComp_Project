"""
Created: 19/04/2023

Author: ks20447 (Adam Morris)

testing_numerical_differentiation.py file to test numerical_differentiation.py functionality

Notes:
"""


import numerical_differentiation as nd
import numpy as np
import unittest
import matplotlib.pyplot as plt


class NumericalMethodsTesting(unittest.TestCase):
           
        
    def test_Boundary_Condition_success(self):
        bc_dir = nd.Boundary_Condition("Dirichlet", 1.0)
        bc_neu = nd.Boundary_Condition("Neumann", 1.0)
        bc_rob = nd.Boundary_Condition("Robin", [1.0, 1.0])
        self.assertEqual(bc_dir.value, 1)
        self.assertEqual(bc_neu.value, 1)
        np.testing.assert_array_equal(bc_dir.value, [1.0, 1.0])

        
    def test_Boundary_Condition_calc_values_success(self):
        bc_dir = nd.Boundary_Condition("Dirichlet", 1.0)
        n = bc_dir.calc_values(0.1)
        self.assertEqual(n, 0)
        self.assertEqual(bc_dir.b_bound, bc_dir.value)
        self.assertEqual(bc_dir.u_bound, bc_dir.value)
        self.assertEqual(bc_dir.sol_bound, 1)
        np.testing.assert_array_equal(bc_dir.a_bound, [-2, 1])
        
        
    def test_Boundary_Condition_exceptions(self):
        with self.assertRaises(SyntaxError) as exception_context:
            bc = nd.Boundary_Condition("Test", 1.0)
            bc.calc_values(0.1)
        self.assertEqual(str(exception_context.exception),
            "Incorrect boundary condition type specified")    
        
        
    def test_construct_a_b_matricies_success(self):
        grid = np.linspace(0, 1, 11)
        a_mat, b_mat = nd.construct_a_b_matricies(grid, nd.Boundary_Condition("Dirichlet", 1.0),
                                                  nd.Boundary_Condition("Dirichlet", 1.0))
        self.assertEqual(a_mat.shape, (9, 9))
        self.assertEqual(b_mat.shape, (9, 1))
        
        
    def test_finite_differnces_success(self):
        grid, u = nd.finite_difference(pde_fdiff, 0, 1, nd.Boundary_Condition("Dirichlet", 0.0),
                                       nd.Boundary_Condition("Dirichlet", 1.0), 10, None)
        exact = grid
        np.testing.assert_array_almost_equal(u[:, 0], exact, 15)
        
        
    def test_explicit_methods_success(self):
        ic = lambda x, args: np.sin(np.pi*x)
        grid, time, u = nd.explicit_methods(pde, 0, 1, 1, nd.Boundary_Condition("Dirichlet", 0.0),
                                            nd.Boundary_Condition("Dirichlet", 0.0), ic, 50, 1, "Euler", None)
        exact = np.exp(-np.pi**2*time[-1])*np.sin(np.pi*grid)
        np.testing.assert_array_almost_equal(u[:, -2:-1], exact, 6)
        
        
    def test_implicit_methods_success(self):
        ic = lambda x, args: np.sin(np.pi*x)
        grid, time, u = nd.implicit_methods(pde, 0, 1, 1, nd.Boundary_Condition("Dirichlet", 0.0),
                                            nd.Boundary_Condition("Dirichlet", 0.0), ic, 50, 0.01, 1, "Crank-Nicoloson", None)
        exact = np.exp(-np.pi**2*time[-1])*np.sin(np.pi*grid)
        np.testing.assert_array_almost_equal(u[:, -2:-1], exact, 4)
        
        
    def test_imex_methods_success(self):
        ic = lambda x, args: np.sin(np.pi*x)
        grid, time, u = nd.imex(pde, 0, 1, 1, nd.Boundary_Condition("Dirichlet", 0.0),
                                            nd.Boundary_Condition("Dirichlet", 0.0), ic, 50, 0.01, 1, None)
        exact = np.exp(-np.pi**2*time[-1])*np.sin(np.pi*grid)
        np.testing.assert_array_almost_equal(u[:, -2:-1], exact, 4)
        
    
    def test_sparse_methods_success(self):
        grid, u = nd.finite_difference(pde_fdiff, 0, 1, nd.Boundary_Condition("Dirichlet", 0.0),
                                       nd.Boundary_Condition("Dirichlet", 1.0), 10, None, sparse=True)
        exact = grid
        np.testing.assert_array_almost_equal(u[:, 0], exact, 15)
        
        ic = lambda x, args: np.sin(np.pi*x)
        grid, time, u = nd.implicit_methods(pde, 0, 1, 1, nd.Boundary_Condition("Dirichlet", 0.0),
                                            nd.Boundary_Condition("Dirichlet", 0.0), ic, 50, 0.01, 1, 
                                            "Crank-Nicoloson", None, sparse=True)
        exact = np.exp(-np.pi**2*time[-1])*np.sin(np.pi*grid)
        np.testing.assert_array_almost_equal(u[:, -2:-1], exact, 4)
        
        grid, time, u = nd.imex(pde, 0, 1, 1, nd.Boundary_Condition("Dirichlet", 0.0),
                                            nd.Boundary_Condition("Dirichlet", 0.0), ic, 50, 0.01, 1, None, sparse=True)
        exact = np.exp(-np.pi**2*time[-1])*np.sin(np.pi*grid)
        np.testing.assert_array_almost_equal(u[:, -2:-1], exact, 4)
        
        
 
def pde_fdiff(x, u, args):
    return 0 

def pde(x, t, u, args)  :
    return 0           

unittest.main()


