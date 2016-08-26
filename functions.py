import numpy as np


"""
Rosenbrock function https://en.wikipedia.org/wiki/Rosenbrock_function
f(x,y) = (a-x)^2 + b(y-x^2)^2
In this class
a = 0.5
b = 0.5
As result f(x,y) = 0.5(1-x)^2 + 0.5(y-x^2)^2
"""

class Rosenbrock:
    initialPoint = (-2, -2)
    camera = (41, 75)
    interval = [(-2, 2), (-2, 2)]

    @staticmethod
    def function(x):
        return 0.5*(1-x[0])**2 + 0.5*(x[1]-x[0]**2)**2

    @staticmethod
    def function_array(x):
        return np.array([1 - x[0] , x[1] - x[0] ** 2]).reshape((2,1))

    @staticmethod
    def gradient(x):
        return np.array([-(1-x[0]) - (x[1]-x[0]**2)*2*x[0], (x[1] - x[0]**2)])

    @staticmethod
    def hessian(x):
        return np.array(((1 -2*x[1] + 6*x[0]**2, -2*x[0]), (-2 * x[0], 1)))

    @staticmethod
    def jacobi(x):
        return np.array([ [-1, 0], [-2*x[0], 1]])

    @staticmethod
    def getZMeshGrid(X, Y):
        return 0.5*(1-X)**2 + 0.5*(Y - X**2)**2

