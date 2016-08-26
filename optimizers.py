# Author: Evgeny Semyonov <DragonSlights@yandex.ru>
# Repository: https://github.com/lightforever/Levenberg_Manquardt

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

from abc import ABCMeta, abstractmethod
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

class Optimizer:
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, epsilon=1e-7, function_array=None, metaclass=ABCMeta):
        self.function_array = function_array
        self.epsilon = epsilon
        self.interval = interval
        self.function = function
        self.gradient = gradient
        self.hesse = hesse
        self.jacobi = jacobi
        self.name = self.__class__.__name__.replace('Optimizer', '')
        self.x = initialPoint
        self.y = self.function(initialPoint)

    "This method will return the next point in the optimization process"
    @abstractmethod
    def next_point(self):
        pass

    """
    Moving to the next point.
    Saves in Optimizer class next coordinates
    """

    def move_next(self, nextX):
        nextY = self.function(nextX)
        self.y = nextY
        self.x = nextX
        return self.x, self.y


class SteepestDescentOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, function_array=None, learningRate=0.05):
        super().__init__(function, initialPoint, gradient, jacobi, hesse, interval, function_array=function_array)
        self.learningRate = learningRate

    def next_point(self):
        nextX = self.x - self.learningRate * self.gradient(self.x)
        return self.move_next(nextX)


class NewtonOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, function_array=None, learningRate=0.05):
        super().__init__(function, initialPoint, gradient, jacobi, hesse, interval, function_array=function_array)
        self.learningRate = learningRate

    def next_point(self):
        hesse = self.hesse(self.x)
        # if Hessian matrix if positive - Ok, otherwise we are going in wrong direction, changing to gradient descent
        if is_pos_def(hesse):
            hesseInverse = np.linalg.inv(hesse)
            nextX = self.x - self.learningRate * np.dot(hesseInverse, self.gradient(self.x))
        else:
            nextX = self.x - self.learningRate * self.gradient(self.x)

        return self.move_next(nextX)


class NewtonGaussOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, function_array=None, learningRate=1):
        super().__init__(function, initialPoint, gradient, jacobi, hesse, interval, function_array=function_array)
        self.learningRate = learningRate

    def next_point(self):
        # Solve (J_t * J)d_ng = -J*f
        jacobi = self.jacobi(self.x)
        jacobisLeft = np.dot(jacobi.T, jacobi)
        jacobiLeftInverse = np.linalg.inv(jacobisLeft)
        jjj = np.dot(jacobiLeftInverse, jacobi.T)  # (J_t * J)^-1 * J_t
        nextX = self.x - self.learningRate * np.dot(jjj, self.function_array(self.x)).reshape((-1))
        return self.move_next(nextX)


class LevenbergMarquardtOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hesse=None,
                 interval=None, function_array=None, learningRate=1):
        self.learningRate = learningRate
        functionNew = lambda x: np.array([function(x)])
        super().__init__(functionNew, initialPoint, gradient, jacobi, hesse, interval, function_array=function_array)
        self.v = 2
        self.alpha = 1e-3
        self.m = self.alpha * np.max(self.getA(jacobi(initialPoint)))

    def getA(self, jacobi):
        return np.dot(jacobi.T, jacobi)

    def getF(self, d):
        function = self.function_array(d)
        return 0.5 * np.dot(function.T, function)

    def next_point(self):
        if self.y==0: # finished. Y can't be less than zero
            return self.x, self.y

        jacobi = self.jacobi(self.x)
        A = self.getA(jacobi)
        g = np.dot(jacobi.T, self.function_array(self.x)).reshape((-1, 1))
        leftPartInverse = np.linalg.inv(A + self.m * np.eye(A.shape[0], A.shape[1]))
        d_lm = - np.dot(leftPartInverse, g) # moving direction
        x_new = self.x + self.learningRate * d_lm.reshape((-1)) # line search
        grain_numerator = (self.getF(self.x) - self.getF(x_new))
        gain_divisor = 0.5* np.dot(d_lm.T, self.m*d_lm-g) + 1e-10
        gain = grain_numerator / gain_divisor
        if gain > 0: # it's a good function approximation.
            self.move_next(x_new) # ok, step acceptable
            self.m = self.m * max(1 / 3, 1 - (2 * gain - 1) ** 3)
            self.v = 2
        else:
            self.m *= self.v
            self.v *= 2

        return self.x, self.y


def getOptimizers(function, initialPoint, gradient, jacobi, hesse, interval, function_array):
    return [optimizer(function, initialPoint, gradient=gradient, jacobi=jacobi, hesse=hesse,
                      interval=interval, function_array=function_array)
            for optimizer in [
                SteepestDescentOptimizer,
                NewtonOptimizer,
                NewtonGaussOptimizer,
                LevenbergMarquardtOptimizer
            ]]
