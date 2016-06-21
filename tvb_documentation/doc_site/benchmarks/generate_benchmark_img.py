# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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


x = [1, 2, 3]
x_labels = ["1.4", "1.4.1", "1.5"]
figure = pyplot.figure()

plot(131, "Generic2dOscillator 1000sec", 20,
     [14.5, 17.9, 6.6], "nodes=192 dt=0.05",
     [9.5, 11.0, 6.2], "nodes=96 dt=0.05",
     [5, 5.9, 3.5], "nodes=96 dt=0.1")

plot(132, "Epileptor 1000sec", 30,
     [21.8, 25.5, 9.1], "nodes=192 dt=0.05",
     [12.8, 16.2, 8.7], "nodes=96 dt=0.05",
     [6.7, 7.5, 4.6], "nodes=96 dt=0.1")

plot(133, "LarterBreakspear 10000sec", 180,
     [159.6, 145.4, 82.8], "nodes=192 dt=0.1",
     [88.6, 89.8, 71.1], "nodes=96 dt=0.1",
     [40.7, 39.0, 39.1], "nodes=96 dt=0.2")

pyplot.show()
figure.savefig("../../_static/benchmarks-evolution-mac.png")
