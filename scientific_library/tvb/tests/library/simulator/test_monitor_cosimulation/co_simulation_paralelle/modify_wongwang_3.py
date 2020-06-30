#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "

import tvb.simulator.lab as lab
from nest_elephant_tvb.simulation.file_tvb.Interface_co_simulation_parallel import Interface_co_simulation
import numpy as np

# reference simulation
np.random.seed(42)
model = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
connectivity = lab.connectivity.Connectivity().from_file()
connectivity.speed = np.array([4.0])
connectivity.configure()
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

# New simulator with proxy
np.random.seed(42)
model_1 = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
monitors_1 = (Interface_co_simulation(period=0.1,id_proxy=np.array([0],dtype=np.int),time_synchronize=10.0))
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim_1 = lab.simulator.Simulator(model=model_1,
                              connectivity=connectivity,
                              coupling=coupling,
                              integrator=integrator,
                              monitors=(monitors,monitors_1,),
                              # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                              )
sim_1.configure()
result_1_all = [np.empty((0,)),np.empty((0,1,76,1))]
for j in range(5):
    result_1_all_step = sim_1.run(
        simulation_length=2.0,
        proxy_data=[(2.0*j)+np.arange(0.1,2.1,0.1),
                    np.array([ result_all[0][1][(20*j)+i][0][0] for i in range(20) ]).reshape((20,1,1,1))])
    result_1_all[0] = np.concatenate((result_1_all[0],result_1_all_step[0][0]))
    result_1_all[1] = np.concatenate((result_1_all[1], result_1_all_step[0][1]))

for i in range(100):
    diff = result_all[0][1][i][0][1:] - result_1_all[1][i,0,1:]
    diff_2 = result_all[0][1][i][0][:1] - result_1_all[1][i,0,:1]
    if np.sum(diff,where=np.logical_not(np.isnan(diff))) == 0.0  and np.sum(diff_2 ,where=np.logical_not(np.isnan(diff_2))) == 0.0:
        print('test succeeds')
    else:
        print(np.sum(diff_2))
        print('test FAIL')
