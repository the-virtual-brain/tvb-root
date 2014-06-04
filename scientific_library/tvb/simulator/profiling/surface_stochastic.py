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
Profiling example for running a stochastic surface simulation with EEG and BOLD.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

from tvb.simulator.lab import *
from time import time

lconn = surfaces.LocalConnectivity(
    equation=equations.Gaussian(),
    cutoff=30.0,
    )

lconn.equation.parameters['sigma'] = 10.0
lconn.equation.parameters['amp'] = 0.0


sim = simulator.Simulator(
        model        = models.Generic2dOscillator(),
        connectivity = connectivity.Connectivity(speed=4.0, load_default=True),
        coupling     = coupling.Linear(a=-2 ** -9),
        integrator   = integrators.HeunStochastic(
                            dt=2 ** -4,
                            noise=noise.Additive(nsig=ones((2,)) * 0.001)
                            ),
        monitors     = (
            monitors.EEG(period=1e3/2 ** 10), # 1024 Hz
            monitors.Bold(period=500)       # 0.5  Hz
            ),
        surface      = surfaces.Cortex(
            load_default=True,
            local_connectivity = lconn,
            coupling_strength  = array([0.01])
            ),
        )

sim.configure()

# set delays to mean
print sim.connectivity.idelays
sim.connectivity.delays[:] = sim.connectivity.delays.mean()
sim.connectivity.set_idelays(sim.integrator.dt)
print sim.connectivity.idelays

ts_eeg, ys_eeg = [], []
ts_bold, ys_bold = [], []

tic = time()
for eeg, bold in sim(60e3):
    if not eeg is None:
        t, y = eeg
        ts_eeg.append(t)
        ys_eeg.append(y)
    if not bold is None:
        t, y = bold
        ts_bold.append(t)
        ys_bold.append(y)
    print t

print '1024 ms took %f s' % (time() - tic,)


save('ts_eeg.npy', squeeze(array(ts_eeg)))
save('ys_eeg.npy', squeeze(array(ys_eeg)))
save('ts_bold.npy', squeeze(array(ts_bold)))
save('ys_bold.npy', squeeze(array(ys_bold)))
