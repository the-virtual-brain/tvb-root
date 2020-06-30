# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import tvb.simulator.lab as lab
from tvb.simulator.Interface_co_simulation_parallel import Interface_co_simulation
import numpy as np

# reference simulation
np.random.seed(42)
model = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
connectivity = lab.connectivity.Connectivity().from_file()
connectivity.speed = np.array([4.0])
coupling = lab.coupling.Linear(a=np.array(0.0154))
integrator = lab.integrators.HeunDeterministic(dt=0.1,bounded_state_variable_indices=np.array([0]),state_variable_boundaries=np.array([[0.0, 1.0]]))
monitors = lab.monitors.Raw(period=0.1, variables_of_interest=np.array(0,dtype=np.int))
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = lab.simulator.Simulator(model=model,
                              connectivity=connectivity,
                              coupling=coupling,
                              integrator=integrator,
                              monitors=(monitors,),
                              # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                              )
sim.configure()
result_all=sim.run(simulation_length=10.0)
result = result_all[0][1][0][0]


# New simulator without proxy
np.random.seed(42)
model_1 = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
monitors_1 = Interface_co_simulation(period=0.1,id_proxy=np.array([],dtype=np.int),time_synchronize=10.0)
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim_1 = lab.simulator.Simulator(model=model_1,
                              connectivity=connectivity,
                              coupling=coupling,
                              integrator=integrator,
                              monitors=(monitors,monitors_1,),
                              # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                              )
sim_1.configure()
result_1_all=sim_1.run(simulation_length=10.0)
result_1= result_1_all[0][1][0][0]
for i in range(100):
    diff = result_all[0][1][i][0][2:] - result_1_all[0][1][i][0][2:]
    diff_2 = result_all[0][1][i][0][:2] - result_1_all[0][1][i][0][:2]
    if np.sum(diff,where=np.logical_not(np.isnan(diff))) == 0.0 and np.sum(diff_2,where=np.logical_not(np.isnan(diff_2))) ==0.0:
        print('test succeeds')
    else:
        print('test FAIL')
