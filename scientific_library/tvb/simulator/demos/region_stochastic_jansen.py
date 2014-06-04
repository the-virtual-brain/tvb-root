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
Demo using the Jansen and Rit model.
White noise is added to one specific state variable to emulate the external
stochastic stimulus p(t) as described in [JanseRit_1995]

.. moduleauthor:: Paula Sanz Leon <pau.sleon@gmail.com>

"""

from tvb.simulator.lab import *

##----------------------------------------------------------------------------##
##-                      Perform the simulation                              -##
##----------------------------------------------------------------------------##

LOG.info("Configuring...")
#Initialise a Model, Coupling, and Connectivity.
jrm = models.JansenRit()
nsigma = 0.022

white_matter = connectivity.Connectivity(load_default=True)
white_matter.speed = numpy.array([4.0])

white_matter_coupling = coupling.Linear(a=0.0)

#Initialise an Integrator adding noise to only one state variable
hiss = noise.Additive(nsig=numpy.array([0., 0., 0., 0., nsigma, 0.]))
heunint = integrators.HeunStochastic(dt=2 ** -4, noise=hiss)

#Initialise some Monitors with period in physical time

momo = monitors.Raw()
mama = monitors.TemporalAverage(period=2 ** -2)

#Bundle them
what_to_watch = list((momo, mama))


#Initialise Simulator -- Model, Connectivity, Integrator, Monitors, and stimulus.
sim = simulator.Simulator(model=jrm,
                          connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint,
                          monitors=what_to_watch)
sim.configure()

LOG.info("Starting simulation...")
#Perform the simulation
raw_data = []
raw_time = []
tavg_time = []
tavg_data = []

for raw, tavg in sim(simulation_length=2 ** 10):
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

#Make the lists numpy.arrays for easier use.
RAW = numpy.array(raw_data)
TAVG = numpy.array(tavg_data)

#Plot raw time series
figure(1)
plot(raw_time, RAW[:, 0, :, 0])
title("Raw -- State variable 0")

figure(2)
plot(raw_time, RAW[:, 1, :, 0])
title("Raw -- State variable 1")

figure(3)
plot(raw_time, RAW[:, 2, :, 0])
title("Raw -- State variable 2")

figure(4)
plot(raw_time, RAW[:, 1, :, 0] - RAW[:, 2, :, 0])
title("Raw -- State variable 1 - 2")


figure(5)
plot(tavg_time, TAVG[:, 0, :, 0])
title("Temporal Average")

#Show them
show()

###EoF###
