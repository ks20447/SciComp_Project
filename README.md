# SciComp_Project

A library containing a series of numerical methods associated with the time simulating and investigating of Oridinary and Partial Differential equations (ODE/PDE's). Created as part of the Scientific Computing module coursework for Engineering Mathematics in the University of Bristol.   

## Authors
Adam Morris (ks20447@bristol.ac.uk)
## Table of Contents
- Installation
- Usage
- Example
- Contributing
- Credit

## Installation

*Note: You will need python 3.10 or later to be able to use this library*

Clone repo from git hub on your local git bash terminal: https://github.com/ks20447/SciComp_Project.git

Use the following pip commands to install the packages and libaries required to run the file: 

```bash
pip install -r requirements.txt
```
## Usage
After creating a suitable environment and cloning, the modules can be imported like any other python package as needed:

```python
import numerical_methods as nm
import numerical_differntiation as nd
```

This will give full access to the included methods.

## Example

Here is a quick example demonstrated on a simple ODE system:

```python
# ODE: y''(t) + 2y'(t) + ay(t) = 0. Convert to system of first order ODE's 
def ode(t, y, args):
    a = args
    u, v = y
    dudt = v
    dvdt = -2*v - a*u
    return [dudt, dvdt]

x, t = nm.solve_to(ode=ode, x0=[1, 1], t1=0, t2=1, h=0.01, method=nm.eurler_method, args=5)    
```

## Contributing

Any contributions before the 27th of April 2023 were produced solely by the author.

This repo will be made public after the 27th of April. If you would like to contribute, please don't hesitate to contact.

## Credit

This project was built under the teaching of David Barton (@dawbarton) and Matthew Hannessy (@hennessymatt) at the University of Bristol.