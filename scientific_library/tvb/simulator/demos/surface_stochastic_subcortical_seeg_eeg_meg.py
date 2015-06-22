# -*- coding: utf-8 -*-
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
Cortical surface with subcortical regions, sEEG, EEG & MEG, using a stochastic
integration.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

from pylab import *
from tvb.simulator.lab import *
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import ProjectionMatrix


oscillator = models.Generic2dOscillator()
white_matter = connectivity.Connectivity.from_file('connectivity_192.zip')
white_matter.speed = numpy.array([4.0])
white_matter_coupling = coupling.Difference(a=0.014)
heunint = integrators.HeunStochastic(dt=2**-4)
projmat = ProjectionMatrix()
ProjectionMatrix.load_surface_projection_matrix(projmat, 'projection_EEG_surface.npy')
monitors = (
    monitors.EEG(period=1e3/1024.0,
                 sensors=sensors.SensorsEEG.from_file('eeg-brainstorm-65.txt'),
                 projection=projmat),
)
local_coupling_strength = numpy.array([2 ** -10])
default_cortex = Cortex(region_mapping_data=RegionMapping.from_file('regionMapping_16k_192.txt'),
                        load_default=True)
default_cortex.coupling_strength = local_coupling_strength
sim = simulator.Simulator(model=oscillator, connectivity=white_matter,
                          coupling=white_matter_coupling,
                          integrator=heunint, monitors=monitors,
                          surface=default_cortex)
sim.configure()

ts, eeg = [], []
for (t, eeg_t), in sim(simulation_length=2 ** 2):
    ts.append(t)
    eeg.append(eeg_t)
ts = numpy.array(ts)
eeg = numpy.array(eeg)

figure()
plot(ts, eeg[:, 0, :, 0], 'k')
show()