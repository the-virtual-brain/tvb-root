import tvb.simulator.lab as lab
from tvb.simulator.Interface_co_simulation import ReducedWongWang_proxy,Interface_co_simulation
import numpy as np

# reference simulation
np.random.seed(42)
model = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
connectivity = lab.connectivity.Connectivity().from_file()
connectivity.speed = np.array([4.0])
connectivity.configure()
coupling = lab.coupling.Linear(a=np.array(0.0154))
integrator = lab.integrators.HeunDeterministic(dt=0.1,bounded_state_variable_indices=np.array([0]),state_variable_boundaries=np.array([[0.0, 1.0]]))
monitors = (lab.monitors.Raw(period=0.1, variables_of_interest=np.array(0,dtype=np.int)),)
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim = lab.simulator.Simulator(model=model,
                              connectivity=connectivity,
                              coupling=coupling,
                              integrator=integrator,
                              monitors=monitors,
                              # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                              )
sim.configure()
result_all=sim.run(simulation_length=10.0)
result = result_all[0][1][0][0]
# The modify model without proxy
np.random.seed(42)
model = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
# integrator = HeunDeterministic(dt=0.1)
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim_2 = lab.simulator.Simulator(model=model,
                              connectivity=connectivity,
                              coupling=coupling,
                              integrator=integrator,
                              monitors=monitors,
                              # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                              )
sim_2.configure()
model_2 = ReducedWongWang_proxy()
model_2.copy_inst(sim.model)
sim_2.model = model_2
result_2_all=sim_2.run(simulation_length=10.0)[0][1][0]
result_2= result_2_all[0]
diff = result - result_2
if np.sum(diff) == 0.0:
    print('test succeeds')
else:
    print('test FAIL')

# The modify model without proxy
np.random.seed(42)
id_proxy = range(11)
model = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim_3 = lab.simulator.Simulator(model=model,
                              connectivity=connectivity,
                              coupling=coupling,
                              integrator=integrator,
                              monitors=monitors,
                              # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                              )
sim_3.configure()
model_3 = ReducedWongWang_proxy()
model_3.copy_inst(sim.model)
model_3.set_id_proxy(id_proxy)
sim_3.model = model_3
result_3_all=sim_3.run(simulation_length=10.0)[0][1][0]
result_3= result_3_all[0]
diff = result - result_3
if np.sum(diff) == 0.0:
    print('test succeeds')
else:
    print('test FAIL')

# The modify model without proxy
np.random.seed(42)
id_proxy = range(11)
model = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim_4 = lab.simulator.Simulator(model=model,
                                connectivity=connectivity,
                                coupling=coupling,
                                integrator=integrator,
                                monitors=monitors,
                                # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                                )
sim_4.configure()
model_4 = ReducedWongWang_proxy()
model_4.copy_inst(sim.model)
model_4.set_id_proxy(np.array(id_proxy))
model_4.update_proxy(np.ones((11,1))*0.7)
sim_4.model = model_4
result_4_all = sim_4.run(simulation_length=10.0)[0][1][0]
result_4 = result_4_all[0]
diff = result - result_4
if np.sum(diff) != 0.0:
    print('test succeeds')
else:
    print('test FAIL')

# The modify model without proxy
np.random.seed(42)
id_proxy = range(11)
model = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim_5 = lab.simulator.Simulator(model=model,
                                connectivity=connectivity,
                                coupling=coupling,
                                integrator=integrator,
                                monitors=monitors,
                                # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                                )
sim_5.configure()
model_5 = ReducedWongWang_proxy()
model_5.copy_inst(sim.model)
model_5.set_id_proxy(np.array(id_proxy))
model_5.update_proxy([[0.02610815369723578  ],
 [0.007918682131383152 ],
 [0.008257260378213565 ],
 [0.023084939706151147 ],
 [0.03725706591997936  ],
 [0.017066023963743862 ],
 [0.028114124110158213 ],
 [0.010048491097557441 ],
 [0.013214675199868617 ],
 [0.0046064972150810365],
 [0.05189135144713729  ]])
sim_5.model = model_5
result_5_all = sim_5.run(simulation_length=10.0)[0][1][0]
result_5 = result_5_all[0]
diff = result - result_5
if np.sum(diff) == 0.0:
    print('test succeeds')
else:
    print('test FAIL')



# New simulator without proxy
np.random.seed(42)
model_6 = lab.models.ReducedWongWang(tau_s=np.random.rand(76))
monitors_2 = (Interface_co_simulation(period=0.1,id_proxy=np.array([0,2],dtype=np.int),time_synchronize=10.0),)
# Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
sim_6 = lab.simulator.Simulator(model=model_6,
                              connectivity=connectivity,
                              coupling=coupling,
                              integrator=integrator,
                              monitors=monitors_2,
                              # initial_conditions=np.repeat(0.0,1*1*nb_region).reshape(1,1,nb_region,1)
                              )
sim_6.configure()
result_6_all=sim_6.run(simulation_length=10.0)[0][1][0]
result_6= result_6_all[0]
diff = result - result_6
if np.sum(diff) != 0.0:
    print('test succeeds')
else:
    print('test FAIL')