from mpi4py import MPI
from tvb.tests.library.simulator.test_monitor_cosimulation.co_simulation_paralelle.function_tvb import tvb_sim
import numpy as np
import numpy.random as rgn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

weight = np.array([[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
delay = np.array([[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1]])*10
resolution_simulation = 0.1
resolution_monitor = 0.1 * 4
time_synchronize = 0.1 * 4

# reference model
rgn.seed(42)
sim_ref = tvb_sim(weight, delay,[], resolution_simulation, resolution_monitor,time_synchronize)
time,result_ref = sim_ref(resolution_monitor)

if rank == 0:
    # initialisation
    rgn.seed(42)
    proxy_id =  np.array([0])
    not_proxy_id = np.ones(len(weight),dtype=bool)
    not_proxy_id [proxy_id] = False
    sim = tvb_sim(weight, delay,proxy_id, resolution_simulation, resolution_monitor,time_synchronize)
    # simulation
    time,result = sim(resolution_monitor,[time,result_ref[:,proxy_id][:,:,0]])
    for i in range(0,10):
        comm.send(result[:,not_proxy_id], dest=1, tag=11)
        rate = comm.recv(source=1, tag=11)
        if np.where(rate != result[:,proxy_id])[0].size!=0:
            print("process 1 : error in step %r", i)
            print("process 1 : rate %r", rate)
            print("process 1 : rate %r", result[:,proxy_id])
            exit(1)
        else:
            print("process 1 : good step %r",i)
        time, result_ref = sim_ref(resolution_monitor)
        time,result = sim(time_synchronize,[time,result_ref[:,proxy_id][:,:,0]])


elif rank == 1:
    # init
    rgn.seed(42)
    proxy_id =  np.array([1,2,3])
    not_proxy_id = np.ones(len(weight),dtype=bool)
    not_proxy_id [proxy_id] = False
    sim = tvb_sim(weight, delay,proxy_id, resolution_simulation, resolution_monitor,time_synchronize)
    # simulation
    time,result = sim(resolution_monitor,[time,result_ref[:,proxy_id][:,:,0]])
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
        time, result_ref = sim_ref(resolution_monitor)
        time,result = sim(time_synchronize,[time,result_ref[:,proxy_id][:,:,0]])


