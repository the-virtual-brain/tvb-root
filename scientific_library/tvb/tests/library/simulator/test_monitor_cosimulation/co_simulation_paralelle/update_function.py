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

weight = np.array([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]])
delay = np.array([[1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5]])
resolution_simulation = 0.1
time_synchronize = 1.0
proxy_id =  [0,1]
firing_rate = np.array([[20.0,10.0]])*10**-3 # units time in tvb is ms so the rate is in KHz

# Test the the update function
sim = tvb_sim(weight, delay,proxy_id, resolution_simulation,time_synchronize)
time, result = sim(resolution_simulation,[np.array([resolution_simulation]),firing_rate])
for i in range(0,100):
    time,result = sim(time_synchronize,[np.arange(i*time_synchronize,(i+1)*time_synchronize,resolution_simulation),
                                        np.repeat(firing_rate.reshape(1,2),int(time_synchronize/resolution_simulation),axis=0)])
print('test succeeds')

# Test a fail function due to the time of simulation too long
try :
    sim(time_synchronize,[np.array([resolution_simulation]),firing_rate])
    print('test fail')
except:
    print('test succeed')
    pass