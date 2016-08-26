from abc import ABCMeta, abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hessian=None,
                 interval=None, epsilon=1e-7, function_array=None, metaclass=ABCMeta):
        self.function_array = function_array
        self.epsilon = epsilon
        self.interval = interval
        self.function = function
        self.gradient = gradient
        self.hessian = hessian
        self.jacobi = jacobi
        self.name = self.__class__.__name__.replace('Optimizer', '')
        self.x = initialPoint
        self.y = self.function(initialPoint)

    "Этот метод будет возвращать следующую точку в процессе оптимизации"

    @abstractmethod
    def next_point(self):
        pass

    """
    Перемещаемся к следующей точке
    """

    def move_next(self, nextX):
        nextY = self.function(nextX)
        self.y = nextY
        self.x = nextX
        return self.x, self.y


class SteepestDescentOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hessian=None,
                 interval=None, function_array=None, learningRate=0.05):
        super().__init__(function, initialPoint, gradient, jacobi, hessian, interval, function_array=function_array)
        self.learningRate = learningRate

    def next_point(self):
        nextX = self.x - self.learningRate * self.gradient(self.x)
        return self.move_next(nextX)


class NewtonOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hessian=None,
                 interval=None, function_array=None, learningRate=0.05):
        super().__init__(function, initialPoint, gradient, jacobi, hessian, interval, function_array=function_array)
        self.learningRate = learningRate

    def next_point(self):
        hessianInverse = np.linalg.inv(self.hessian(self.x))
        nextX = self.x - self.learningRate * np.dot(hessianInverse, self.gradient(self.x))
        return self.move_next(nextX)


class NewtonGaussOptimizer(Optimizer):
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hessian=None,
                 interval=None, function_array=None, learningRate=1):
        super().__init__(function, initialPoint, gradient, jacobi, hessian, interval, function_array=function_array)
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
    def __init__(self, function, initialPoint, gradient=None, jacobi=None, hessian=None,
                 interval=None, function_array=None, learningRate=1):
        self.learningRate = learningRate
        functionNew = lambda x: np.array([function(x)])
        super().__init__(functionNew, initialPoint, gradient, jacobi, hessian, interval, function_array=function_array)
        self.v = 2
        self.alpha = 1e-3
        self.m = self.alpha * np.max(self.getA(jacobi(initialPoint)))

    def getA(self, jacobi):
        return np.dot(jacobi.T, jacobi)

    def getF(self, d):
        function = self.function_array(d)
        return 0.5 * np.dot(function.T, function)

    def next_point(self):
        if self.y==0:
            return self.x, self.y

        jacobi = self.jacobi(self.x)
        A = self.getA(jacobi)
        g = np.dot(jacobi.T, self.function_array(self.x)).reshape((-1, 1))
        leftPartInverse = np.linalg.inv(A + self.m * np.eye(A.shape[0], A.shape[1]))
        d_lm = - np.dot(leftPartInverse, g)
        x_new = self.x + self.learningRate * d_lm.reshape((-1))
        grain_numerator = (self.getF(self.x) - self.getF(x_new))
        gain_divisor = 0.5* np.dot(d_lm.T, self.m*d_lm-g) + 1e-10
        gain = grain_numerator / gain_divisor
        if gain > 0:
            self.move_next(x_new)
            self.m = self.m * max(1 / 3, 1 - (2 * gain - 1) ** 3)
            self.v = 2
        else:
            self.m *= self.v
            self.v *= 2

        return self.x, self.y


def getOptimizers(function, initialPoint, gradient, jacobi, hessian, interval, function_array):
    return [optimizer(function, initialPoint, gradient=gradient, jacobi=jacobi, hessian=hessian,
                      interval=interval, function_array=function_array)
            for optimizer in [
                SteepestDescentOptimizer,
                NewtonOptimizer,
                NewtonGaussOptimizer,
                LevenbergMarquardtOptimizer
            ]]
