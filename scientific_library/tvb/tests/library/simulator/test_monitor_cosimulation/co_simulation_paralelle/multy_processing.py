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
from mpi4py import MPI
from tvb.tests.library.simulator.test_monitor_cosimulation.co_simulation_paralelle.function_tvb import tvb_sim
import numpy as np
import numpy.random as rgn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

weight = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
delay = np.array([[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]])*10
resolution_simulation = 0.1
time_synchronize = 0.1 * 4

# reference model
rgn.seed(42)
sim_ref = tvb_sim(weight, delay,[], resolution_simulation, time_synchronize)
time,result_ref = sim_ref(time_synchronize)

if rank == 0:
    # initialisation
    rgn.seed(42)
    proxy_id =  np.array([0])
    not_proxy_id = np.ones(len(weight),dtype=bool)
    not_proxy_id [proxy_id] = False
    sim = tvb_sim(weight, delay,proxy_id, resolution_simulation, time_synchronize)
    # simulation
    time,result = sim(time_synchronize,[time,result_ref[:,proxy_id][:,:,0]])
    for i in range(0,10):
        comm.send(result[:,not_proxy_id], dest=1, tag=11)
        rate = comm.recv(source=1, tag=11)
        if np.where(rate != result[:,not_proxy_id])[0].size!=0:
            print("process 1 : error in step %r", i)
            print("process 1 : rate %r", rate)
            print("process 1 : rate %r", result[:,proxy_id])
            exit(1)
        else:
            print("process 1 : good step %r",i)
        time, result_ref = sim_ref(time_synchronize)
        time,result = sim(time_synchronize,[time,result_ref[:,proxy_id][:,:,0]])


elif rank == 1:
    # init
    rgn.seed(42)
    proxy_id =  np.array([1,2,3])
    not_proxy_id = np.ones(len(weight),dtype=bool)
    not_proxy_id [proxy_id] = False
    sim = tvb_sim(weight, delay,proxy_id, resolution_simulation, time_synchronize)
    # simulation
    time,result = sim(time_synchronize,[time,result_ref[:,proxy_id][:,:,0]])
    for i in range(0,10):
        comm.send(result[:,not_proxy_id], dest=0, tag=11)
        rate = comm.recv(source=0, tag=11)
        if np.where(rate != result[:,proxy_id])[0].size!=0:
            print("process 2 : error in step %r", i)
            print("process 2 : rate %r", rate)
            print("process 2 : rate %r", result[:,proxy_id])
            exit(1)
            exit(1)
        else:
            print("process 2 : good step %r",i)
        time, result_ref = sim_ref(time_synchronize)
        time,result = sim(time_synchronize,[time,result_ref[:,proxy_id][:,:,0]])


