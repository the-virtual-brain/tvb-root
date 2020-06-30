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
from tvb.tests.library.simulator.test_monitor_cosimulation.co_simulation_paralelle.function_tvb import tvb_sim
import numpy as np
import numpy.random as rgn

weight = np.array([[1,1],[1,1]])
delay = np.array([[10.0,10.0],[10.0,10.0]])
resolution_simulation = 0.1
time_synchronize = 0.1*4
proxy_id_1 = [0]
not_proxy_id_1 = np.where(np.logical_not(np.isin(np.arange(0,weight.shape[0],1), proxy_id_1)))[0]
proxy_id_2 = [1]
not_proxy_id_2 = np.where(np.logical_not(np.isin(np.arange(0,weight.shape[0],1), proxy_id_2)))[0]

# full simulation
rgn.seed(42)
sim_ref = tvb_sim(weight, delay,[], resolution_simulation, time_synchronize)
time,result_ref = sim_ref(time_synchronize)

# simulation with one proxy
rgn.seed(42)
sim_1 = tvb_sim(weight, delay,proxy_id_1, resolution_simulation, time_synchronize)
time,result_1 = sim_1(time_synchronize,[time,result_ref[:,proxy_id_1][:,:,0]])

# simulation_2 with one proxy
rgn.seed(42)
sim_2 = tvb_sim(weight, delay,proxy_id_2, resolution_simulation, time_synchronize)
time,result_2 = sim_2(time_synchronize,[time,result_1[:,proxy_id_2][:,:,0]])

print("COMPARE PROXY 1 ---------------")
diff_1 = np.where(np.squeeze(result_ref,axis=2)[0] != np.squeeze(result_1,axis=2)[0])
if diff_1[0].size != 0:
    print("Fail compare")
    print(diff_1)
    print(result_ref)
    print(result_1)
else:
    print('test 0 : succeed')
print("COMPARE PROXY 2 ---------------")
diff_2 = np.where(np.squeeze(result_ref,axis=2)[0] != np.squeeze(result_2,axis=2)[0])
if diff_2[0].size != 0:
    print("Fail compare")
    print(diff_2)
    print(result_ref)
    print(result_2)
else:
    print('test 0 : succeed')

max_error_s_1 = 0.0
max_error_s_2 = 0.0
for i in range(0,100):
    time,result_ref = sim_ref(time_synchronize)
    time,result_1 = sim_1(time_synchronize,[time,result_ref[:,proxy_id_1][:,:,0]])
    time,result_2 = sim_2(time_synchronize,[time,result_1[:,proxy_id_2][:,:,0]])

    print("COMPARE PROXY 1 ---------------")
    if np.max(np.abs(result_ref- result_1)) > max_error_s_1:
        max_error_s_1 = np.max(np.abs(result_ref- result_1))
    diff_1 = np.where(np.squeeze(result_ref, axis=2)[0] != np.squeeze(result_1, axis=2)[0])
    if diff_1[0].size != 0:
        print("Fail compare")
        print(diff_1)
        print(result_ref)
        print(result_1)
    else:
        print('test' + str(i + 1) + ' : succeed')
    print("COMPARE PROXY 2 ---------------")
    if np.max(np.abs(result_ref- result_2)) > max_error_s_2:
        max_error_s_2 = np.max(np.abs(result_ref- result_2))
    diff_2 = np.where(np.squeeze(result_ref, axis=2)[0] != np.squeeze(result_2, axis=2)[0])
    if diff_2[0].size != 0:
        print("Fail compare")
        print(diff_2)
        print(result_ref)
        print(result_2)
    else:
        print('test' + str(i + 1) + ' : succeed')

print("time_synchronize : %r s"%(time_synchronize*10**-3))
print("max error S rate for 1 : %r" % max_error_s_1)
print("max error S rate for 2: %r" % max_error_s_2)