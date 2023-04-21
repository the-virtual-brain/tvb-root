# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Template for running a demo using a 'contributed' model

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <pau.sleon@gmail.com>

"""
import numpy
from tvb.simulator.lab import *
from matplotlib.pylab import *

LOG = get_logger(__name__)

# Add the contributed models directory to the PYTHONPATH
sys.path += ["../models"]

# import the Model class
from larter_breakspear import LarterBreakspear

##----------------------------------------------------------------------------##
##-                      Perform the simulation                              -##
##----------------------------------------------------------------------------##

LOG.info("Configuring...")
# Initialise a Model, Coupling, and Connectivity.
lar = LarterBreakspear(QV_max=numpy.array([1.0]), QZ_max=numpy.array([1.0]), # t_scale=numpy.array([0.01]),
                       VT=numpy.array([0.54]), d_V=numpy.array([0.5]), C=numpy.array([0.0]))

white_matter = connectivity.Connectivity.from_file()
white_matter.speed=numpy.array([4.0])
white_matter_coupling = coupling.Linear(a=lar.C)

# Initialise an Integrator
heunint = integrators.HeunDeterministic(dt=0.2)

# Initialise some Monitors with period in physical time
mon_raw = monitors.Raw()
mon_tavg = monitors.TemporalAverage(period=1.)

# Bundle them
what_to_watch = (mon_raw, mon_tavg)

# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = simulator.Simulator(model=lar,
                          connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint,
                          monitors=what_to_watch)
sim.configure()

LOG.info("Starting simulation...")
# Perform the simulation
raw_data, raw_time = [], []
tavg_data, tavg_time = [], []

for raw, tavg in sim(simulation_length=2 ** 14):
    if not raw is None:
        raw_time.append(raw[0])
        raw_data.append(raw[1])

    if not tavg is None:
        tavg_time.append(tavg[0])
        tavg_data.append(tavg[1])

LOG.info("Finished simulation.")

##----------------------------------------------------------------------------##
##-               Plot pretty pictures of what we just did                   -##
##----------------------------------------------------------------------------##

# Make the lists numpy.arrays for easier use.
RAW = numpy.array(raw_data)
TAVG = numpy.array(tavg_data)

# Plot raw time series
figure(1)
plot(raw_time, RAW[:, 0, :, 0])
title("Raw -- State variable 0")

figure(2)
plot(raw_time, RAW[:, 1, :, 0])
title("Raw -- State variable 1")

figure(3)
plot(raw_time, RAW[:, 2, :, 0])
title("Raw -- State variable 2")

# Plot 3D trajectories

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(4)
ax = fig.gca(projection='3d')
ax.plot(RAW[:, 0, 0, 0], RAW[:, 1, 0, 0], RAW[:, 2, 0, 0])
plt.show()

# Show them
show()
###EoF###
