from tvb.tests.library.simulator.test_monitor_cosimulation.co_simulation_paralelle.function_tvb import tvb_sim
import numpy as np
import numpy.random as rgn

weight = np.array([[2,8],[3,5]])
delay = 100.0
delays = np.array([[delay,delay],[delay,delay]])
init_value = [[0.9],[0.9]]
resolution_simulation = 0.1
# resolution_monitor = 0.1*10.0
# time_synchronize = 0.1*10.0
resolution_monitor = 0.1*10.0
time_synchronize = 0.1*10.0
nb_init = (int(delay/resolution_simulation))+1
initial_condition = np.array(init_value * nb_init).reshape(nb_init,1,weight.shape[0],1)
proxy_id = [0]
not_proxy_id = np.where(np.logical_not(np.isin(np.arange(0,weight.shape[0],1), proxy_id)))[0]

# full simulation
rgn.seed(42)
sim_ref = tvb_sim(weight, delays,[], resolution_simulation, resolution_monitor,time_synchronize,initial_condition=initial_condition)
time,result_ref,s_ref = sim_ref(resolution_monitor,s=True)

# simulation with one proxy
rgn.seed(42)
sim = tvb_sim(weight, delays,proxy_id, resolution_simulation, resolution_monitor,time_synchronize,initial_condition=initial_condition)
time,result,s = sim(resolution_monitor,[time,result_ref[:,proxy_id][:,:,0]],s=True)

diff = np.where(result_ref[:,not_proxy_id] != result)
if diff[0].size != 0:
    print("Fail H compare")
    print(diff)
    print(result_ref)
    print(result)
else:
    print('test H 0 : succeed')
diff_s = np.where(s_ref != s)
if diff_s[0].size != 0:
    print("Fail S compare")
    print(diff_s)
    print(s_ref)
    print(s)
else:
    print('test S 0 : succeed')

max_error_h = 0.0
max_error_s = 0.0

for i in range(0,10000):
    time,result_ref,s_ref = sim_ref(time_synchronize,s=True)
    time, result, s = sim(time_synchronize,[time,result_ref[:,proxy_id][:,:,0]], s=True)

    diff = np.where(result_ref[:,not_proxy_id]!= result)
    if np.max(np.abs(result_ref- result)) > max_error_h:
        max_error_h = np.max(np.abs(result_ref- result))
    if diff[0].size !=0:
        print('rate compare')
        print(diff)
        print(result_ref)
        print(result)
        print(np.max(np.abs(result_ref- result)))
    else:
        print('test H '+str(i+1)+' : succeed')
    diff_s = np.where(s_ref != s)
    if np.max(np.abs(s_ref - s)) > max_error_s:
        max_error_s = np.max(np.abs(s_ref - s))
    if diff_s[0].size != 0:
        print("s_compare")
        print(diff_s)
        print(s_ref)
        print(s)
        print(np.max(np.abs(s_ref - s)))
    else:
        print('test S ' + str(i + 1) + ' : succeed')

print("time_synchronize : %r s"%(time_synchronize*10**-3))
print("time_monitor : %r s"%(resolution_monitor*10**-3))
print("max error firing rate : %r Hz" % (max_error_h*10**3))
print("max error S rate : %r" % max_error_s)