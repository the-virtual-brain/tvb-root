# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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

"""
This demos how to go about debugging some device code. 
Rather, It is actually used for that, but could be helpful to others
eventually.

good
----

So far I've verified (on cpu) Euler & Euler stochastic w/ Gen 2D osc, 
WilsonCowan. JansenRit seems good with deterministic Euler, but diverges
numerically more quickyl with noise. Kuramoto seems fine as well.
ReducedFHN and ReducedHMR have more error, needs checking to see what the
source is.

The Heun integrators now appear to work as well as the Euler integrators.

Linear coupling is good, sigmoidal appears ok, difference & Kuramoto
are ok as well.

bad
---

The otherwise slow error creep is due to one or more of

- 64-bit vs 32-bit precision
- effects of order of operations on 
- history indexing off by one
- connectivity orientation/convention


ugly
----

HeunStoch doesn't like ReducedHMR, makes lots of NaaN.


debugging (july/2013)
---------------------


.. moduleauthor:: marmaduke woodman <mw@eml.cc>

"""

from tvb.simulator.lab import *
from tvb.simulator.backend import cee, cuda, driver

map(reload, [driver, cee, cuda])
conn = connectivity.Connectivity(load_default=True, speed=300.0)

sim = simulator.Simulator(
    model=models.Generic2dOscillator(),
    connectivity=conn,
    coupling=coupling.Linear(a=1e-2),
    integrator=integrators.HeunStochastic(
        dt=2 ** -5,
        noise=noise.Additive(nsig=numpy.ones((2, 1, 1)) * 1e-5)
    ),
    monitors=monitors.Raw()
)

sim.configure()

# then build device handler and pack it wiht simulation
dh = cuda.Handler.init_like(sim)
dh.n_thr = 64
dh.n_rthr = dh.n_thr
dh.fill_with(0, sim)
for i in range(1, dh.n_thr):
    dh.fill_with(i, sim)

print 'required mem ', dh.nbytes / 2 ** 30.

# run both with raw monitoring, compare output
simgen = sim(simulation_length=100)
cont = True

ys1, ys2, ys3 = [], [], []
et1, et2 = 0.0, 0.0
while cont:

    # simulator output
    try:
        # history & indicies
        #histidx = ((sim.current_step - 1 - sim.connectivity.idelays)%sim.horizon)[:4, :4].flat[:]*74 + r_[:4, :4, :4, :4]
        #histval = [sim.history[(sim.current_step - 1 - sim.connectivity.idelays[10,j])%sim.horizon, 0, j, 0] for j in range(dh.n_node)]
        #print 'histidx', histidx
        #print 'hist[idx]', histval

        tic = time()
        t1, y1 = next(simgen)[0]
        ys1.append(y1)
        et1 += time() - tic
    except StopIteration:
        break


    #print 'state sim', sim.integrator.X[:, -1, 0]
    #print 'state dh ', dh.x.value.transpose((1, 0, 2))[:, -1, 0]


    tic = time()
    # dh output
    cuda.gen_noise_into(dh.ns, dh.inpr.value[0])
    dh()

    #print 'I sim', sim.coupling.ret[0, 10:15, 0]

    # compare dx
    #print 'dx1 sim', sim.integrator.dX[:, -1, 0]
    #print 'dx1 dh ', dh.dx1.value.flat[:]
    
    t2 = dh.i_step * dh.inpr.value[0]
    _y2 = dh.x.value.reshape((dh.n_node, -1, dh.n_mode)).transpose((1, 0, 2))
    #ys3.append(_y2)

    # in this case where our simulations are all identical, the easiest
    # comparison, esp. to check that all threads on device behave, is to
    # randomly sample one of the threads at each step (right?)
    y2 = _y2[0]
    ys2.append(y2)
    et2 += time() - tic

    if dh.i_step % 100 == 0:
        stmt = "%4.2f\t%4.2f\t%.3f"
        print stmt % (t1, t2, ((y1 - y2) ** 2).sum()/y1.ptp())

ys1 = array(ys1)
ys2 = array(ys2)
#ys3 = array(ys3)
#print ys3.shape, ys3.nbytes/2**30.0

print et1, et2, et2 * 1. / dh.n_thr
#print ys1.flat[::450]
#print ys2.flat[::450]
savez('debug.npz', ys1=ys1, ys2=ys2)    # , ys3=ys3)

from matplotlib import pyplot as pl

pl.figure(2)
pl.clf()
pl.subplot(311), pl.imshow(ys1[:, 0, :, 0].T, aspect='auto', interpolation='nearest'), pl.colorbar()
pl.subplot(312), pl.imshow(ys2[:, :, 0].T, aspect='auto', interpolation='nearest'), pl.colorbar()
pl.subplot(313), pl.imshow(100 * ((ys1[:, 0] - ys2) / ys1.ptp())[..., 0].T, aspect='auto',
                           interpolation='nearest'), pl.colorbar()

#pl.show()
pl.savefig('debug.png')
