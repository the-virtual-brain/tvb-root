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
Pycuda interop patterns


.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import time
import os
import string
import numpy

import driver

try:
    import pyublas
except ImportError as exc:
    global __pyublas__available__
    __pyublas__available__ = False

import pycuda.autoinit
import pycuda.driver
import pycuda.gpuarray as gary
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceData, OccupancyRecord

def orinfo(n):
    orec = OccupancyRecord(DeviceData(), n)
    return """occupancy record information
        thread blocks per multiprocessor - %d
        warps per multiprocessor - %d
        limited by - %s
        occupancy - %f
    """ % (orec.tb_per_mp, orec.warps_per_mp, orec.limited_by, orec.occupancy)

#                                       dispo=pycuda.autoinit.device.total_memory()
def estnthr(dist, vel, dt, nsv, pf=0.7, dispo=1535*2**20                           ):
    n = dist.shape[0]
    idelmax = long(dist.max()/vel/dt)
    return long( (dispo*pf - 4*n*n + 4*n*n)/(4*idelmax*n + 4*nsv*n + 4 + 4) )


class arrays_on_gpu(object):

    def __init__(self, _timed="gpu timer", _memdebug=False, **arrays):

        self.__array_names = arrays.keys()

        if _memdebug:
            memuse = sum([v.size*v.itemsize for k, v in arrays.iteritems()])/2.**20
            memavl = pycuda.autoinit.device.total_memory()/2.**20
            print 'GPU mem use %0.2f MB of %0.2f avail.' % (memuse, memavl)
            for k, v in arrays.iteritems():
                print 'gpu array %s.shape = %r' % (k, v.shape)
            assert memuse <= memavl
    
        for key, val in arrays.iteritems():
            setattr(self, key, gary.to_gpu(val))

        self._timed_msg = _timed

    def __enter__(self, *args):
        self.tic = time.time()
        return self

    def __exit__(self, *args):

        if self._timed_msg:
            print "%s %0.3f s" % (self._timed_msg, time.time() - self.tic)

        for key in self.__array_names:
            delattr(self, key)

   
class srcmod(object):
    
    def __init__(self, src, fns, debug=False):

        self.src = src

        if debug:
            print "srcmod: source is \n%s" % (self.src,)

        self._module = SourceModule(self.src)

        for f in fns:
            fn = self._module.get_function(f)
            if debug:
                def fn_(*args, **kwds):
                    try:
                        fn(*args, **kwds)
                        pycuda.driver.Context.synchronize()
                    except Exception as exc:
                        msg = 'PyCUDA launch of %r failed w/ %r'
                        msg %= (fn, exc)
                        raise Exception(msg)
            else:
                fn_ = fn
            setattr(self, f, fn_)

# FIXME was in driver.py, mix of names
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule as CUDASourceModule
from pycuda import gpuarray
import pycuda.tools

class OccupancyRecord(pycuda.tools.OccupancyRecord):
    def __repr__(self):
        ret = "Occupancy(tb_per_mp=%d, limited_by=%r, warps_per_mp=%d, occupancy=%0.3f)"
        return ret % (self.tb_per_mp, self.limited_by, self.warps_per_mp, self.occupancy)

_, total_mem = cuda.mem_get_info()
print 'GPU memory ', total_mem/2**30.
from pycuda.curandom import XORWOWRandomNumberGenerator as XWRNG
rng = XWRNG(pycuda.curandom.seed_getter_unique, 2000)

# FIXME should be on the noise objects, but has different interface
# FIXME this is additive white noise
def gen_noise_into(devary, dt):
    gary = devary.device
    rng.fill_normal(gary)
    gary.set(gary.get()*numpy.sqrt(dt))

class Code(driver.Code):
    def __init__(self, *args, **kwds):
        super(Code, self).__init__(*args, **kwds)
        self.mod = CUDASourceModule("#define TVBGPU\n" + self.source, 
                                    options=["--ptxas-options=-v"], keep=True,
                                    cache_dir=False)

class Global(driver.Global):
    """
    Encapsulates a source module CUDA global in a Python data descriptor
    for easy handling

    """

    def post_init(self):
        if self.__post_init:
            self.ptr   = self.code.mod.get_global(self.name)[0]
            self.__post_init = False

    def __get__(self, inst, ownr):
        self.post_init()
        buff = array([0]).astype(self.dtype)
        cuda.memcpy_dtoh(buff, self.ptr)
        return buff[0]

    def __set__(self, inst, val):
        self.post_init()
        cuda.memcpy_htod(self.ptr, self.dtype(val))
        buff = empty((1,)).astype(self.dtype)
        cuda.memcpy_dtoh(buff, self.ptr)

 
class Array(driver.Array):
    """
    Encapsulates an array that is on the device

    """

    @property
    def device(self):
        if not hasattr(self, '_device'):
            if self.pagelocked:
                raise NotImplementedError
            self._device = gpuarray.to_gpu(self.cpu)
        return self._device

    def set(self, ary):
        """
        In place update the device array.
        """
        _ = self.device
        self._device.set(ary)

    @property
    def value(self):
        return self.device.get()

    def __init__(self, *args, **kwds):
        super(Array, self).__init__(*args)
        self.pagelocked = kwds.get('pagelocked', False)

class Handler(driver.Handler):
    i_step_type = numpy.int32
    def __init__(self, *args, **kwds):
        kwds.update({'Code': Code, 'Global': Global, 'Array': Array})
        super(Handler, self).__init__(*args, **kwds)
        self._device_update = self.code.mod.get_function('update')
    @property
    def mem_info(self):
        return cuda.mem_get_info()
    @property
    def occupancy(self):
        try:
            return OccupancyRecord(pycuda.tools.DeviceData(), self.n_thr)
        except Exception as exc:
            return exc
    @property
    def extra_args(self):
        bs = int(self.n_thr)%1024
        gs = int(self.n_thr)/1024
        if bs == 0:
            bs = 1024
        if gs == 0:
            gs = 1
        return {'block': (bs, 1, 1),
                'grid' : (gs, 1)}

    def __call__(self, extra=None):
        args  = [self.i_step_type(self.i_step)]
        for k in self.device_state:
            args.append(getattr(self, k).device)
        kwds = extra or self.extra_args
        #try:
        import pdb; pdb.set_trace()
        self._device_update(*args, **kwds)
        """
        except cuda.LogicError as e:
            print 0, 'i_step', type(args[0])
            for i, k in enumerate(self.device_state):
                attr = getattr(self, k).device
                print i+1, k, type(attr), attr.dtype
            print kwds
            raise e
        """
        self.i_step += self.n_msik
        cuda.Context.synchronize()


