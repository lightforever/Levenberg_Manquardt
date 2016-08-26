# Author: Evgeny Semyonov <DragonSlights@yandex.ru>
# Repository: https://github.com/lightforever/Levenberg_Manquardt

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import webbrowser

import optimizers
import plot
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from os import path
import functions
from mpl_toolkits.mplot3d import axes3d, Axes3D

#colors for line plotting
optimizersColorLookup = {'SteepestDescent': 'green', 'Newton': 'blue', 'NewtonGauss': 'red',
                         'LevenbergMarquardt': 'yellow'}

# we can create new function(which use same interface)
functionClass = functions.Rosenbrock

"""
Plot function on the interval.
Plot legend using optimizersColorLookup for optimizers
Init camera at convenient position
All it needs is in functionClass
"""
def initAxes(optimizers):
    interval = functionClass.interval
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(interval[0][0], interval[0][1], 0.05)
    y = np.arange(interval[1][0], interval[1][1], 0.05)
    X, Y = np.meshgrid(x, y)
    Z = functionClass.getZMeshGrid(X, Y)

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=functionClass.camera[0], azim=functionClass.camera[1])
    plt.legend(
        handles=[mpatches.Patch(color=optimizersColorLookup[optimizer.name], label=optimizer.name) for optimizer in
                 optimizers])

    return fig, ax

# may be list or array
def is_sequence(arg):
    return isinstance(arg, np.ndarray) or isinstance(arg, list)


#save image after N iterations
printStep = 1
#work excactly maxIteration count. No rules to stop optimizers
maxIteration = 100
# need save images and make GIF after that. If False it' finish much faster
needSaveImage = True
currentIteration = 0
optimizers = optimizers.getOptimizers(functionClass.function, functionClass.initialPoint, functionClass.gradient,
                                      jacobi=functionClass.jacobi,
                                      hesse=functionClass.hesse,
                                      interval=functionClass.interval,
                                      function_array=functionClass.function_array
                                      )
fig, ax = initAxes(optimizers)
plotter = plot.GifPlotter(fig, ax)

while currentIteration <= maxIteration:
    for optimizer in optimizers:
        yBefore = optimizer.y
        xBefore = optimizer.x

        x, y = optimizer.next_point()
        if is_sequence(y):
            y = y[0]
        if is_sequence(yBefore):
            yBefore = yBefore[0]

        plotter.plotLine([xBefore[0], x[0]], [xBefore[1], x[1]], [yBefore, y], optimizersColorLookup[optimizer.name])
        print("Iteration = {0} Optimizer = {1} X = {2} Y = {3} Z = {4}".format(currentIteration, optimizer.name, x[0],
                                                                               x[1], y))

    if currentIteration % printStep == 0 and needSaveImage:
        plotter.fixImage(currentIteration)

    currentIteration += 1

if needSaveImage:
    plotter.savegif('movie.gif')
    webbrowser.open(path.join(path.dirname(__file__), 'movie.gif'))
