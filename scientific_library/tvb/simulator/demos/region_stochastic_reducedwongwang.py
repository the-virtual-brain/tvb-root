# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
#   Frontiers in Neuroinformatics (in press)
#
#

"""

References:

.. [DPA_2013]     Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca
                  Romani, Patric Hagmann and Maurizio Corbetta. *Resting-State
                  Functional Connectivity Emerges from Structurally and
                  Dynamically Shaped Slow Linear Fluctuations*. The Journal of
                  Neuroscience 32(27), 11239-11252, 2013.

Demonstrate using the simulator at the region level, stochastic
integration, ReducedWongWang model with default paramters. The large
speed conduction speed is used to neglect the time-delays as in
[DPA_2013]_

Using the Hagmann 66 ROI connectome, it's possible to obtain similar
results as those obtained with the Python code provided by the
authors.

When using the default TVB connectivity matrix, then the slope 'a' of
the Linear coupling function should be divided by a factor of at least
100 to obtain similar results (see below).


``Run time``: 

``Memory requirement``: < 1GB

.. moduleauthor::Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>

"""

from tvb.simulator.lab import *

##----------------------------------------------------------------------------##
##-                      Perform the simulation                              -##
##----------------------------------------------------------------------------##

LOG.info("Configuring...")
#Initialise a Model, Coupling, and Connectivity.
rww = models.ReducedWongWang()

#Change the state variable range to constraint the initial conditions 
rww.state_variable_range['S'] = numpy.array([0.0, 0.01])

#Intialise a Connectivity
white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([20000.0])
white_matter_coupling = coupling.Linear(a=1.05 / 100.)

#Initialise an Integrator
hiss = noise.Additive(nsig=rww.sigma_noise)
heunint = integrators.EulerStochastic(dt=0.1, noise=hiss)

#Initialise some Monitors with period in physical time
momo = monitors.Raw()
mama = monitors.TemporalAverage(period=1.)

#Bundle them
what_to_watch = (momo, mama)

#Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = simulator.Simulator(model=rww, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch)
sim.configure()

LOG.info("Starting simulation...")
#Perform the simulation
raw_data = []
raw_time = []
tavg_data = []
tavg_time = []

for raw, tavg in sim(simulation_length=6000.):
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

#Plot defaults in a few combinations

#Make the lists numpy.arrays for easier use.
RAW = numpy.array(raw_data)
TAVG = numpy.array(tavg_data)

#Plot raw time series
figure(1)
plot(raw_time, RAW[:, 0, :, 0])
title("Raw -- State variable 0")

#Plot temporally averaged time series
figure(2)
plot(tavg_time, TAVG[:, 0, :, 0])
title("Temporal average")

#Show them
show()

###EoF###
