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
