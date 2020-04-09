import tvb.simulator.lab as lab
from tvb.simulator.Interface_co_simulation_parallel import Interface_co_simulation
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
result = result_all[0][1][0][0]


# New simulator without proxy
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
result_1_all=sim_1.run(simulation_length=10.0,
                       proxy_data=[
                           np.array([0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. ,  1.1,  1.2,  1.3,  1.4,
                            1.5,  1.6,  1.7,  1.8,  1.9,  2. ,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,
                            2.9,  3. ,  3.1,  3.2,  3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,
                            4.3,  4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,  5.4,  5.5,  5.6,
                            5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,  6.5,  6.6,  6.7,  6.8,  6.9,  7.,
                            7.1,  7.2,  7.3,  7.4,  7.5,  7.6,  7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,
                            8.5,  8.6,  8.7,  8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,  9.8,
                            9.9,10.0]),
                           np.array([
[0.02610815369723578],
[0.02208690565795946],
[0.01624221568694433],
[0.010065958661410385],
[0.012964439911412518],
[0.007101161304900518],
[0.007648360961273784],
[0.008577103349965191],
[0.006033154091789269],
[0.006613949465239189],
[0.004395086640358901],
[0.004270539061781448],
[0.004733953696455739],
[0.00579534875906159],
[0.003882815034260703],
[0.005026490972646495],
[0.003967695335940262],
[0.004575914423173127],
[0.0044686063697628115],
[0.0048916280189420175],
[0.005360182658502455],
[0.005687485736912752],
[0.002854816775134307],
[0.005600867371299465],
[0.004994427149810174],
[0.0031432982706472924],
[0.0040681916094510535],
[0.004883573527796498],
[0.004888651540616098],
[0.004737365227324522],
[0.005590994490112996],
[0.0026179835114235354],
[0.0037959332061937245],
[0.0043876314638697225],
[0.004772385101606284],
[0.004502705767544471],
[0.0041812104893505665],
[0.004958566316137342],
[0.003602737554072243],
[0.0035713235242163985],
[0.004036454819217265],
[0.0046974942942214735],
[0.004747146919375551],
[0.005391864083461571],
[0.005072569649262788],
[0.004699780762217837],
[0.004741833945952414],
[0.005154835212960672],
[0.004523834512180826],
[0.005296020195320874],
[0.003571234092075289],
[0.0035608551058189516],
[0.003994791126465998],
[0.003369350058198901],
[0.0029324179027583963],
[0.0033840540626525637],
[0.0036235556887715195],
[0.0036920510830784938],
[0.0036589509298598543],
[0.004554092474246084],
[0.003275960543014039],
[0.0022556403595760713],
[0.004324652989918787],
[0.0030212189407283732],
[0.002914722645734373],
[0.0033925263221681492],
[0.0038124955855517984],
[0.003945528393608132],
[0.004499536304670582],
[0.003829610588068099],
[0.002501346117324761],
[0.002994466743374786],
[0.0032028639684915316],
[0.0030982655915235695],
[0.002662392779526683],
[0.0024998255244032194],
[0.002261862849980796],
[0.0031478375590211142],
[0.0020529575668908084],
[0.0032728938859704617],
[0.0028118690785402067],
[0.0033624024185951646],
[0.0030257052437773443],
[0.0032818904443960998],
[0.0025330125257271765],
[0.002820111237270084],
[0.0037819794584374035],
[0.0023063363337395035],
[0.002151931530433212],
[0.0033707341099257675],
[0.002621056501062864],
[0.002558272887608563],
[0.002608867476845759],
[0.0030946712324493447],
[0.002627646527428595],
[0.0024575055641328633],
[0.002147820452679465],
[0.001701196664282492],
[0.0018450652752899112],
[0.002311344893216932],

                                     ]).reshape((100,1,1,1))])
result_1= result_1_all[0][1][0][0]
for i in range(100):
    diff = result_all[0][1][i][0][2:] - result_1_all[0][1][i][0][2:]
    diff_2 = result_all[0][1][i][0][:2] - result_1_all[0][1][i][0][:2]
    if np.sum(diff,where=np.logical_not(np.isnan(diff))) == 0.0  and np.sum(diff_2,where=np.logical_not(np.isnan(diff_2))) == 0.0:
        print('test succeeds')
    else:
        print(np.sum(diff_2))
        print('test FAIL')
