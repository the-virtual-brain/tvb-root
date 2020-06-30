#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements; and to You under the Apache License, Version 2.0. "

from nest_elephant_tvb.simulation.file_tvb.test_interface_co_parallel.function_tvb import tvb_sim
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