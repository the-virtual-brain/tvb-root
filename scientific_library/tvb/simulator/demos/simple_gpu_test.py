# -*- coding: utf-8 -*-

"""
This is a simple demo for gpu testing.

.. moduleauthor::  Timothée Proix <timpx@eml.cc>

"""


import numpy
import matplotlib.pyplot as plt
from tvb.simulator import lab
from tvb.simulator.backend import driver_conf
driver_conf.using_gpu = 1
from tvb.simulator.backend import driver
reload(driver)

sim = lab.simulator.Simulator(
    model=lab.models.Generic2dOscillator(b=-10.0, c=0., d=0.02, I=0.0),
    connectivity=lab.connectivity.Connectivity(speed=4.0),
    coupling=lab.coupling.Linear(a=1e-2),
    integrator=lab.integrators.EulerDeterministic(dt=2**-5),
    monitors=(lab.monitors.Raw(), lab.monitors.TemporalAverage(period=2**-2))
)

sim.configure()

print('configured')
dh = driver.device_handler.init_like(sim)
dh.n_thr = dh.n_rthr = 1
dh.fill_with(0, sim)

nsteps = 10000
ds = 50
ys = numpy.zeros((nsteps/ds, dh.n_node, dh.n_svar, 1))
tavg = numpy.zeros((nsteps/ds, dh.n_node, dh.n_svar, 1))
tavg2 = []
dys = numpy.zeros((nsteps/ds, dh.n_node, dh.n_svar, 1))

sim(simulation_length=1000)
for i in range(nsteps):
    dh()
    ys[i/ds, ...] = dh.x.device.get()
    tavg[i/ds, ...] = dh.tavg.device.get()
    _tavg2 = dh.tavg.value.reshape((dh.n_node, -1, dh.n_mode)).transpose((1, 0, 2))
    tavg2.append(_tavg2[0])
    dys[i/ds, ...] = dh.dx1.device.get()

ys = numpy.array(ys)
dys = numpy.array(dys)
tavg2 = numpy.array(tavg2)
plt.figure()
plt.plot(ys[:, 0 , 0, 0])
plt.plot(tavg[:, 0, 0, 0])
plt.figure()
plt.plot(tavg2[:, 0, 0])
plt.figure()
plt.plot(tavg2[7::8,0, 0,])
plt.show()
