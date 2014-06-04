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
Demonstrate using the simulator at the region level with a stimulus.

``Run time``: approximately 7 min (workstation circa 2010).

``Memory requirement``: < 1GB

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

from tvb.simulator.lab import *

##----------------------------------------------------------------------------##
##-                      Perform the simulation                              -##
##----------------------------------------------------------------------------##

LOG.info("Configuring...")
#Initialise a Model, Coupling, and Connectivity.
oscilator = models.Generic2dOscillator()    # ReducedSetHindmarshRose() #
white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=0.0126)

#Initialise an Integrator
heunint = integrators.HeunDeterministic(dt=2 ** -4)

#Initialise some Monitors with period in physical time
momo = monitors.TemporalAverage(period=1.0)     # 1000Hz
mama = monitors.Bold(period=500)    # defaults to one data point every 2s

#Bundle them
what_to_watch = (momo, mama)

#Define the stimulus
#Specify a weighting for regions to receive stimuli... 
white_matter.configure()  # Because we want access to number_of_regions
nodes = [0, 7, 13, 33, 42]
weighting = numpy.zeros((white_matter.number_of_regions, ))     # 1
weighting[nodes] = numpy.array([2.0 ** -2, 2.0 ** -3, 2.0 ** -4, 2.0 ** -5, 2.0 ** -6])  # [:, numpy.newaxis]

eqn_t = equations.Gaussian()
eqn_t.parameters["midpoint"] = 15000.0
eqn_t.parameters["sigma"] = 4.0

stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                  connectivity=white_matter,
                                  weight=weighting)

#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and stimulus.
sim = simulator.Simulator(model=oscilator, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=what_to_watch,
                          stimulus=stimulus)    # surface=default_cortex,

sim.configure()


LOG.info("Starting simulation...")
#Perform the simulation
tavg_time = []
tavg_data = []
bold_time = []
bold_data = []
for tavg, bold in sim(simulation_length=30000):
    
    if not tavg is None:
        tavg_time.append(tavg[0])
        tavg_data.append(tavg[1])
    
    if not bold is None:
        bold_time.append(bold[0])
        bold_data.append(bold[1])

LOG.info("Finished simulation.")

###EoF###
