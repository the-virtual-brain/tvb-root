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
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Generate 8.125 seconds of 2048 Hz data at the surface level, stochastic integration.

``Run time``: approximately 5 hours (workstation circa 2010, MKL.)

``Memory requirement``: ~ 7 GB
``Storage requirement``: 2.1 GB

.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""

from tvb.simulator.lab import *

##----------------------------------------------------------------------------##
##-                      Perform the simulation                              -##
##----------------------------------------------------------------------------##

#TODO: Configure this so it actually generates an interesting timeseries.
#      Start with a local_coupling that has a broader footprint than the default.

LOG.info("Configuring...")
#Initialise a Model, Coupling, and Connectivity.
oscillator = models.Generic2dOscillator()

white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([4.0])
white_matter_coupling = coupling.Linear(a=2 ** -9)

#Initialise an Integrator
hiss = noise.Additive(nsig=numpy.array([2 ** -16, ]))
heunint = integrators.HeunStochastic(dt=0.06103515625, noise=hiss)
#TODO: Make dt as big as possible to shorten runtime; restrictions of integral multiples to get desired monitor period

#Initialise a Monitor with period in physical time
what_to_watch = monitors.TemporalAverage(period=0.48828125)     # 2048Hz => period=1000.0/2048.0

#Initialise a surface
local_coupling_strength = numpy.array([0.0115])

grey_matter = surfaces.LocalConnectivity(load_default=True)
grey_matter.cutoff = 60.0
grey_matter.equation.parameters['sigma1'] = 10.0
grey_matter.equation.parameters['sigma2'] = 20.0
grey_matter.equation.parameters['amp1'] = 1.0
grey_matter.equation.parameters['amp2'] = 0.5

default_cortex = surfaces.Cortex.from_file()
default_cortex.local_connectivity = grey_matter
default_cortex.coupling_strength = local_coupling_strength

#Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = simulator.Simulator(model=oscillator, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch,
                          surface=default_cortex)
sim.configure()

#Clear initial transient
LOG.info("Initial run to clear transient...")
for _ in sim(simulation_length=125):
    pass
LOG.info("Finished initial run to clear transient.")


#Perform the simulation
tavg_data = []
tavg_time = []
LOG.info("Starting simulation...")
for tavg in sim(simulation_length=8125):
    if not tavg is None:
        tavg_time.append(tavg[0][0])
        tavg_data.append(tavg[0][1])

LOG.info("Finished simulation.")


##----------------------------------------------------------------------------##
##-                     Save the data to a file                              -##
##----------------------------------------------------------------------------##

#Make the list a numpy.array.
LOG.info("Converting result to array...")
TAVG = numpy.array(tavg_data)

#Save it
FILE_NAME = "demo_data_surface_8.125s_2048Hz.npy"
LOG.info("Saving array to %s..." % FILE_NAME)
numpy.save(FILE_NAME, TAVG)

LOG.info("Done.")

###EoF###