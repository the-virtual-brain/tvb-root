# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Defines a controller providing an HTTP/JSON API for the simulator
(or "burst" as is referred to in framework).

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import os
import hashlib
import json
import datetime
import functools
import threading
import multiprocessing
import h5py
import numpy
import cherrypy

# simulator imports 
from tvb.datatypes import connectivity, equations, surfaces, patterns
from tvb.simulator import noise, integrators, models, coupling, monitors, simulator
# framework
from tvb.interfaces.web.controllers.base_controller import BaseController


def threadsafe(f):
    """
    Decorate f with a re-entrant lock to ensure that only
    one CherryPy request is processed by f at once.
    """

    lock = threading.RLock()

    @functools.wraps(f)
    def synchronized(*args, **kwds):
        with lock:
            r = f(*args, **kwds)
        return r
    return synchronized


def build_sim_part(mod, opt):
    """
    Use dictionary opt to specify attributes of an instance 
    from the module mod, where an entry 'class' in opt is 
    used to identify the class required.

    If 'class' entry has multiple lines, and the first line
    is used to identify the class; see the SimulatorController.dir() 
    method below.
    """
    class_ = opt.pop('class')
    if '\n' in class_:
        class_ = class_.split('\n')[0]

    obj = getattr(mod, class_)()
    for k, v in opt.iteritems():
        if type(v) in (list,):
            v = numpy.array(v)
        setattr(obj, k, v)
    return obj


def build_and_run_(spec):
    """
    Builds a simulator from spec, run & collect output.

    Returns an HDF5 file with the results.
    """
    opt = spec['opt']
    print "pool starting ", opt

    # lenght of simulation 
    tf = float(opt.get('tf', 100))

    # model # coupling function # connectivity 
    simargs = {}
    for mod, key in [(models, 'model'), 
                     (connectivity, 'connectivity'),
                     (coupling, 'coupling')]:
        simargs[key] = build_sim_part(mod, opt[key])

    # noise # integrator 
    optint = opt['integrator']
    if 'noise' in optint:
        optint['noise'] = build_sim_part(noise, optint['noise'])
    simargs['integrator'] = build_sim_part(integrators, optint)

    # monitors 
    if not type(opt['monitors']) in (list,):
        opt['monitors'] = [opt['monitors']]
    simargs['monitors'] = []
    for mon in opt['monitors']:
        simargs['monitors'].append(build_sim_part(monitors, mon))

    # stimulus 
    # NotImplemented

    # simulator 
    sim = simulator.Simulator(**simargs)
    sim.configure()

    # TODO open HDF5 first, figure out correct sizes, etc

    # loop, writing data to h5
    ts = [[] for _ in opt['monitors']]
    ys = [[] for _ in opt['monitors']]
    for i, all_monitor_data in enumerate(sim(tf)):
        for j, mondata in enumerate(all_monitor_data):
            if not mondata is None:
                t, y = mondata
                ts[j].append(t)
                ys[j].append(y)

    # write data to hdf5 file
    path = os.path.abspath(opt.get('wd', './'))
    h5fname = os.path.join(path, "tvb_%s.h5" % (spec['md5sum'], ))
    h5 = h5py.File(h5fname, 'w')

    for i, (mon, (t, y)) in enumerate(zip(simargs['monitors'], zip(ts, ys))):
        mname = "mon_%d_%s" % (i, mon.__class__.__name__)
        g = h5.create_group(mname)
        g.create_dataset('ts', data=t)
        g.create_dataset('ys', data=y)

    h5.close()

    # return filename
    print "pool finished", opt
    return h5fname


def build_and_run(spec):
    try:
        r = build_and_run_(spec)
    except Exception as e:
        print 'launch of the simulator failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        import traceback
        traceback.print_exc()
        r = e
    return r



class SimulatorController(BaseController):

    # keep track of simulations
    nsim = 0
    sims = {}

    exposed = True


    def __init__(self, nproc=2):
        super(SimulatorController, self).__init__()
        self.reset(nproc=nproc)


    @cherrypy.expose
    def index(self):
        return 'Please see the documentation of the tvb.interfaces.web.controllers.api.simulator module'

    """
    Need to check how the burst controller queries operations for simulations

    """

    @cherrypy.expose
    def read(self, ix=None):
        """
        Get information on simulation(s)
        """

        if ix is not None:
            sims = {ix: self.sims[int(ix)]} 
        else:
            sims = self.sims

        dump = []
        for ix, sim in sims.iteritems():
            kv = {}
            for k, v in sim.iteritems():
                if k == 'async_result':
                    if v.ready():
                        try:
                            kv['result'] = v.get()
                            kv['status'] = True
                        except Exception as e:
                            kv['status'] = repr(e)
                    else:
                        kv['status'] = 'waiting'
                else:
                    kv[k] = v
            dump.append(kv)

        return json.dumps(dump)

    """
    - reformat data as received by MATLAB to what BurstController.launch_burst() expects
    - urlget on the URL directly? or inherit a burst controller...

    """

    @cherrypy.expose
    @threadsafe
    def create(self, js):
        """
        Create a new simulation and add to computational pool.
        """

        spec = json.loads(js)
        self.nsim += 1
        ix = self.nsim
        spec['ix'] = ix
        spec['datetime'] = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        spec['md5sum'] = hashlib.md5(json.dumps(spec)).hexdigest()
        spec['async_result'] = self.pool.apply_async(build_and_run, (spec.copy(), ))
        self.sims[ix] = spec
        return str(ix)

    """
    No changes needed.

    """

    @cherrypy.expose
    def dir(self):
        """
        Dump information about available models, etc, & parameters into
        a json blob and return it.

        If a particular class is requested, full doc is returned.
        """

        info = {}

        for m in [models, coupling, integrators, noise, monitors, connectivity, equations, surfaces, patterns]:
            minfo = {}
            for k in dir(m):
                v = getattr(m, k)
                if isinstance(v, type):
                    minfo[k] = k + '\n\n' + getattr(v, '__doc__', k)
            info[m.__name__.split('.')[-1]] = minfo

        return json.dumps(info)

    """
    Remove the processor pool deal, and stick to just removal one or more 
    operations already perofrmed.

    """

    @cherrypy.expose
    @threadsafe
    def reset(self, nproc=2):
        """
        Reset the simulation list & restarts the process pool with 
        certain number of processes.
        """

        nproc = int(nproc)
        if hasattr(self, 'pool'):
            # docs say GC'ing the pool will terminate() it, but let's be sure
            self.pool.terminate()
        self.pool = multiprocessing.Pool(processes=nproc)
        self.sims = {}
        self.nsim = 0
        return str(nproc)



if __name__ == '__main__':
    # not true if we're running in the TVB distribution

    class API(object):
        exposed = True

        @cherrypy.expose
        def version(self):
            return '1.1'  # TODO TVB version

    api = API()
    api.simulator = SimulatorController()

    print 'server is ready'

    cherrypy.quickstart(api, '/api', {
        'global': {
            'server.socket_host': '0.0.0.0',
            'server.socket_port': 8080,
        },
    }
    )

