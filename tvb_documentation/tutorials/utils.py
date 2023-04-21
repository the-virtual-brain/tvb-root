# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

from pylab import *
from tvb.simulator.lab import *
import threading
import numpy
import time
import sys
from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt


class SimThread(threading.Thread):

    def set_sim(self, sim, **kwds):
        self.sim = sim
        self.kwds = kwds
        self.t = -1.0

    def run(self):
        """Convenience method to call the simulator with **kwds and collect output data."""
        tic = time.time()
        ts, xs = [], []

        for _ in self.sim.monitors:
            ts.append([])
            xs.append([])

        for data in self.sim(**self.kwds):
            for tl, xl, t_x in zip(ts, xs, data):
                if t_x is not None:
                    t, x = t_x
                    if t > self.t:
                        self.t = t
                    tl.append(t)
                    xl.append(x)

        for i in range(len(ts)):
            ts[i] = numpy.array(ts[i])
            xs[i] = numpy.array(xs[i])

        self.results = list(zip(ts, xs))
        self.wall_time = time.time() - tic


def formattime(eta):
    m = 60
    h = m * 60
    d = 24 * h
    msg = ''
    if eta > d:
        msg += '%d day(s), %d hour(s)' % (eta / d, ((eta / h) % 24))
    elif eta > h:
        msg += '%d hour(s), %d minute(s)' % (eta / h, ((eta / m) % 60))
    else:
        msg += '%d minute(s), %d seconds(s)' % (eta / m, eta % m)
    return msg


def pbpct(p=1e2, eta=None, walltime=None):
    i = int(p / 2)
    msg = '\r[%s%s] %d %%' % ('.' * i, ' ' * (50 - i), p)
    if eta:
        msg += ', ETA: ' + formattime(eta)
    elif walltime:
        msg += ' Wall Time: ' + formattime(walltime)
    sys.stdout.write(msg)
    sys.stdout.flush()


def run_sim_with_progress_bar(sim, simulation_length, polltime=1):
    tic = time.time()
    ar = SimThread()
    ar.set_sim(sim, simulation_length=simulation_length)
    ar.start()
    while True:
        prog = ar.t
        pct = prog * 1e2 / simulation_length
        toc = time.time() - tic
        if pct > 0.0:
            time_per_centile = toc / pct
            centiles_left = 100 - pct
            pbpct(pct, centiles_left * time_per_centile)
        else:
            pbpct(0.0)
        time.sleep(polltime)
        if hasattr(ar, 'wall_time'):
            pbpct(100.0, walltime=ar.wall_time)
            break
    return ar.results


def multiview(data, cortex, suptitle='', figsize=(15, 10), **kwds):

    vtx = cortex.vertices
    tri = cortex.triangles
    rm = cortex.region_mapping
    x, y, z = vtx.T
    lh_tri = tri[(rm[tri] < 38).any(axis=1)]
    lh_vtx = vtx[rm < 38]
    lh_x, lh_y, lh_z = lh_vtx.T
    lh_tx, lh_ty, lh_tz = lh_vtx[lh_tri].mean(axis=1).T
    rh_tri = tri[(rm[tri] >= 38).any(axis=1)]
    rh_vtx = vtx[rm < 38]
    rh_x, rh_y, rh_z = rh_vtx.T
    rh_tx, rh_ty, rh_tz = vtx[rh_tri].mean(axis=1).T
    tx, ty, tz = vtx[tri].mean(axis=1).T

    views = {
        'lh-lateral': Triangulation(-x, z, lh_tri[argsort(lh_ty)[::-1]]),
        'lh-medial': Triangulation(x, z, lh_tri[argsort(lh_ty)]),
        'rh-medial': Triangulation(-x, z, rh_tri[argsort(rh_ty)[::-1]]),
        'rh-lateral': Triangulation(x, z, rh_tri[argsort(rh_ty)]),
        'both-superior': Triangulation(y, x, tri[argsort(tz)]),
    }

    def plotview(i, j, k, viewkey, z=None, zlim=None, zthresh=None, suptitle='',
                 shaded=True, cmap=plt.cm.coolwarm, viewlabel=False):
        v = views[viewkey]
        ax = subplot(i, j, k)
        if z is None:
            z = rand(v.x.shape[0])
        if not viewlabel:
            axis('off')
        kwargs = {'shading': 'gouraud'} if shaded else {'edgecolors': 'k', 'linewidth': 0.1}
        if zthresh:
            z = z.copy() * (abs(z) > zthresh)
        tc = ax.tripcolor(v, z, cmap=cmap, **kwargs)
        if zlim:
            tc.set_clim(vmin=-zlim, vmax=zlim)
        ax.set_aspect('equal')
        if suptitle:
            ax.set_title(suptitle, fontsize=24)
        if viewlabel:
            xlabel(viewkey)

    figure(figsize=figsize)
    plotview(2, 3, 1, 'lh-lateral', data, **kwds)
    plotview(2, 3, 4, 'lh-medial', data, **kwds)
    plotview(2, 3, 3, 'rh-lateral', data, **kwds)
    plotview(2, 3, 6, 'rh-medial', data, **kwds)
    plotview(1, 3, 2, 'both-superior', data, suptitle=suptitle, **kwds)
    subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0, hspace=0)
