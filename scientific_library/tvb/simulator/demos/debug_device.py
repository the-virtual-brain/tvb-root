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

import time
import itertools

import matplotlib as mpl
mpl.use('Agg')

from numpy import *
import numpy

from tvb.simulator import lab
# can't reload individual modules, must reboot ipython
from tvb.simulator.backend import driver_conf
driver_conf.using_gpu = 1
from tvb.simulator.backend import driver, util
reload(driver)

def makesim():
    sim = lab.simulator.Simulator(
        model = lab.models.Generic2dOscillator(b=-10.0, c=0., d=0.02, I=0.0),
        connectivity = lab.connectivity.Connectivity(speed=4.0),
        coupling = lab.coupling.Linear(a=1e-2),                                         # shape must match model..
        integrator = lab.integrators.EulerDeterministic(dt=2**-5),
        #integrator = lab.integrators.HeunStochastic(dt=2**-5, noise=lab.noise.Additive(nsig=ones((2, 1, 1))*1e-2)),
        monitors = lab.monitors.Raw()
    )
    sim.configure()
    return sim

"""
Parameter sweep scenario : we sweep in two dimensions over the coupling
function's scale parameter ``a``, and the excitability parameter of the 
oscillator ``a``
"""

sims = []
for i, coupling_a in enumerate(r_[:0.1:16j]):
    for j, model_a in enumerate(r_[-2.0:2.0:16j]):
        simi = makesim()
        simi.coupling.a[:] = coupling_a
        simi.model.a[:] = model_a
        sims.append(simi)
        print 'simulation %d generated' % (i*32+j,)


# then build device handler and pack it iwht simulation
dh = driver.device_handler.init_like(sims[0])
dh.n_thr = dh.n_rthr = len(sims)
for i, simi in enumerate(sims):
    dh.fill_with(i, simi)

nsteps = 10000
ds = 50
ys1 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
ys2 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
dys1 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
dys2 = zeros((nsteps/ds, dh.n_node, dh.n_svar, len(sims)))
print ys1.nbytes/2**30.0

simgens = [s(simulation_length=1000) for s in sims]

tc, tg = util.timer(), util.timer()
for i in range(nsteps):

    # iterate each simulation
    with tc:
        for j, (sgj, smj) in enumerate(zip(simgens, sims)):
            ((t, y), ) = next(sgj)
            # y.shape==(svar, nnode, nmode)
            ys1[i/ds, ..., j] = y[..., 0].T
            dys1[i/ds, ..., j] = smj.dx[..., 0].T

    with tg:
        dh()

    ys2[i/ds, ...] = dh.x.device.get()
    dys2[i/ds, ...] = dh.dx1.device.get()

    if i/ds and not i%ds:
        err = ((ys1[:i/ds] - ys2[:i/ds])**2).sum()/ys1[:i/ds].ptp()/len(sims)
        print t, tc.elapsed, tg.elapsed, err


print tc.elapsed, tg.elapsed

print 'saving output data to ./debug.npz'
savez('debug.npz', ys1=ys1, ys2=ys2, dys1=dys1, dys2=dys2)