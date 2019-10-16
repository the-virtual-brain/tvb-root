# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Script used for obtaining the benchmarks overview image.
It requires the user to first
- run tvb.interfaces.command.benchmark.py.
- manually copy from console output the results and place them in a tvb_x.y.rst file
- from tvb_x.y.rst, add times in the plot bellow.
"""
import matplotlib.pyplot as pyplot


def plot(subplot, title, ymax, data1, label1, data2, label2, data3, label3):
    pyplot.subplot(subplot)
    pyplot.plot(x, data1, 'b', label=label1, marker='o')
    pyplot.plot(x, data2, 'g', label=label2, marker='o')
    pyplot.plot(x, data3, 'r', label=label3, marker='o')
    pyplot.legend(bbox_to_anchor=(0., 1.04, 1., 1.02), loc=3, ncol=1,
                  mode="expand", borderaxespad=0., fontsize=10)
    pyplot.xticks(x, x_labels)
    pyplot.ylim(ymin=0, ymax=ymax)
    pyplot.ylabel("seconds")
    pyplot.xlabel("TVB version")
    pyplot.grid(True)
    pyplot.title(title)


def plot_neotraits():
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_labels = ["2", "4", "8", "16", "32", "64", "128", "256", "512"]
    figure = pyplot.figure(figsize=(15, 7))

    pyplot.plot(x, [20.4, 19.1, 20.3, 20.4, 19.8, 18.9, 17.2, 16.1, 14.4], 'b', label="Epileptor", linestyle="--",
                linewidth=3)
    pyplot.plot(x, [44.1, 42.7, 43.1, 42.8, 39.4, 40.4, 36.7, 33.9, 27.4], 'b', label="Epileptor neo", linewidth=3)
    pyplot.plot(x, [32.7, 32.5, 32.5, 26.3, 26.4, 27.9, 26.5, 24.7, 22.4], 'g', label="RWW", linestyle="--",
                linewidth=3)
    pyplot.plot(x, [78.9, 79.0, 78.4, 78.7, 77.6, 76.6, 68.9, 64.0, 53.9], 'g', label="RWW neo", linewidth=3)
    pyplot.plot(x, [31.4, 31.3, 30.8, 30.9, 31.3, 30.7, 29.9, 28.8, 26.2], 'r', label="G2D", linestyle="--",
                linewidth=3)
    pyplot.plot(x, [63.3, 62.4, 62.7, 61.9, 61.6, 59.2, 57.1, 52.3, 45.8], 'r', label="G2D neo", linewidth=3)
    pyplot.legend(fontsize=10)
    pyplot.xticks(x, x_labels)
    pyplot.ylim(ymin=0, ymax=80)
    pyplot.ylabel("kHz")
    pyplot.xlabel("nodes")
    pyplot.grid(True)
    pyplot.title("Simulation cycles")

    pyplot.show()
    figure.savefig("../../_static/benchmarks-neotraits.png")


x = [1, 2, 3, 4, 5, 6]
x_labels = ["1.4", "1.4.1", "1.5", "1.5.4", "1.5.9", "2.0.0"]
figure = pyplot.figure(figsize=(12, 11))

plot(131, "Generic2dOscillator 1000sec", 20,
     [14.5, 17.9, 6.6, 7.2, 7.2, 12], "nodes=192 dt=0.05",
     [9.5, 11.0, 6.2, 7.0, 7.1, 11], "nodes=96 dt=0.05",
     [5, 5.9, 3.5, 3.6, 4.2, 9], "nodes=96 dt=0.1")

plot(132, "Epileptor 1000sec", 25,
     [21.8, 25.5, 9.1, 8.9, 9.5, 14], "nodes=192 dt=0.05",
     [12.8, 16.2, 8.7, 8.7, 9.3, 20], "nodes=96 dt=0.05",
     [6.7, 7.5, 4.6, 5.0, 5.0, 14], "nodes=96 dt=0.1")

plot(133, "LarterBreakspear 10000sec", 200,
     [159.6, 145.4, 82.8, 79.2, 78.5, 137], "nodes=192 dt=0.1",
     [88.6, 89.8, 71.1, 74.6, 74.5, 120], "nodes=96 dt=0.1",
     [40.7, 39.0, 39.1, 39.4, 39.8, 94], "nodes=96 dt=0.2")

pyplot.show()
figure.savefig("../../_static/benchmarks-evolution-mac.png")
