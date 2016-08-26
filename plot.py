"""
FFMPEG Required. You can get it at https://ffmpeg.zeranoe.com/builds/   (static library)
Ensure ffmpeg is available from command line. ADD IT TO YOUR $PATH if you are using Windows
"""
import subprocess
import os
from os import path

import shutil
import matplotlib.pyplot as plt


class GifPlotter:
    def __init__(self, fig, ax, movieTimeSec=4):
        self.movieTimeSec = movieTimeSec
        self.fig = fig
        self.ax = ax
        self._tempDir = path.join(path.dirname(__file__), 'temp')
        self.lastText = None

        if path.exists(self._tempDir):
            for file in os.listdir(self._tempDir):
                os.remove(path.join(self._tempDir, file))
        else:
            os.mkdir(self._tempDir)




    def plotLine(self, x, y, z, color):
        self.ax.plot(x, y, z, color=color, linewidth=3)

    def fixImage(self, iteration):
        if self.lastText is not None:
            self.lastText.remove()

        self.lastText = self.ax.text2D(0.05, 0.95, 'Iteration = {0}'.format(iteration), transform=self.ax.transAxes)
        savePath = path.join(self._tempDir, "picture_{0}.jpg".format(iteration))
        self.fig.savefig(savePath, dpi=self.fig.dpi)

    def savegif(self, fileName):
        os.chdir(self._tempDir)
        subprocess.call(['ffmpeg', '-i', 'picture_%d.jpg', fileName, '-y' ])
        shutil.copy(path.join(self._tempDir, fileName), path.join(path.dirname(__file__), fileName))
