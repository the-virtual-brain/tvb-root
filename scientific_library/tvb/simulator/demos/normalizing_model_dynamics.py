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

from tvb.simulator.lab import models
from pylab import *
from numpy import *
reload(models)
import multiprocessing
import itertools

ms = [models.Generic2dOscillator(), 
      models.WilsonCowan(), 
      models.JansenRit(),
      models.ReducedSetFitzHughNagumo(),
      models.ReducedSetHindmarshRose(),
      models.Kuramoto(),
      models.LarterBreakspear()
      ]

[m.configure() for m in ms]

cs = r_[-10:10:40j].reshape((-1, 1, 1))

dt=2**-6
n_step=int(5000/dt)
n_skip=10
n_save=int(4000/dt)

todo = list(itertools.product(ms, cs))

def one(i):
    mi, csi = todo[i]
    if len(mi.cvar) > 1:
        csi = tile(csi, (len(mi.cvar), 1))
    ts, ys = mi.stationary_trajectory(
        coupling=csi, n_skip=n_skip, n_step=n_step, dt=dt)
    return ts[-n_save:], ys[-n_save:]

def all(n_cpus=multiprocessing.cpu_count()):
    pool = multiprocessing.Pool(n_cpus)
    out = []
    for i, (ts, ys) in enumerate(pool.map(one, range(len(todo)))):
        out.append(ys)

    pool.close()

    ret = []
    for i, m in enumerate(ms):
        ret.append(array(out[i*len(cs):(i+1)*len(cs)]))

    return ts, ret

def plot_data(ts, ys, specfun=None):
    cs_ = cs.ravel()
    figure()
    for y in ys:

        subplot(121)
        y = y.reshape((len(cs_), -1))
        y = (y - y.mean())/y.ptp()
        errorbar(cs_, y.mean(axis=1), yerr=y.std(axis=1))

        subplot(122)
        freqs = fft.fftfreq(len(ts), (ts[1] - ts[0])/1000.0)
        spec = abs(fft.fft(y.reshape((len(ts), -1)).sum(axis=1)))
        if specfun is None:
            semilogx(freqs, spec/spec.ptp())
        else:
            specfun(freqs, spec)

    subplot(121)
    #legend([m.__class__.__name__ for m in ms])
    # colors are same l-r, so use r as there's more space
    grid(True)
    xlabel('Input strength')
    ylabel('Normalized response mean and variability')
    xlim((cs_[0], cs_[-1]))

    subplot(122)
    legend([m.__class__.__name__ for m in ms])
    grid(True)
    xlabel('Frequency (Hz)')
    ylabel('Normalized response power')
    xlim([0, 100])

def main():
    plot_data(*all())


if __name__ == '__main__':
    main()