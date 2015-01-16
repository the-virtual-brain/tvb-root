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
#   Frontiers in Neuroinformatics (in press)
#
#

"""
Driver implementation for PyCUDA

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


try:
    import pyublas
except ImportError as exc:
    pyublas = None
    LOG.debug('pyublas unavailable')

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule as CUDASourceModule
from pycuda import gpuarray
import pycuda.tools

from . import base


class OccupancyRecord(pycuda.tools.OccupancyRecord):
    def __repr__(self):
        ret = ("Occupancy(tb_per_mp=%d, limited_by=%r, "
              "warps_per_mp=%d, occupancy=%0.3f)")
        return ret % (self.tb_per_mp, self.limited_by, 
                      self.warps_per_mp, self.occupancy)

_, total_mem = cuda.mem_get_info()

from pycuda.curandom import XORWOWRandomNumberGenerator as XWRNG
rng = XWRNG()

# FIXME should be on the noise objects, but has different interface
# FIXME this is additive white noise
def gen_noise_into(devary, dt):
    gary = devary.device
    rng.fill_normal(gary)
    gary.set(gary.get()*sqrt(dt))


class Code(base.Code):
    pass


class Global(base.Global):

    def post_init(self):
        if self.__post_init:
            self.ptr = self.code.mod.get_global(self.name)[0]
            super(Global, self).post_init()

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


class Array(base.Array):

    @property
    def device(self):
        if not hasattr(self, '_device'):
            if self.pagelocked:
                raise NotImplementedError
            self._device = gpuarray.to_gpu(self.cpu)
        return self._device

    def set(self, ary):
        _ = self.device
        self._device.set(ary)

    @property
    def value(self):
        return self.device.get()

    def __init__(self, name, type, dimensions, pagelocked=False):
        super(Array, self).__init__(name, type, dimensions)
        self.pagelocked = pagelocked


class RegionParallel(base.RegionParallel):
    pass

    def get_update_fun(self):
        return device_code.mod.get_function('update')

    @property
    def mem_info(self):
        return cuda.mem_get_info()

    @property
    def occupancy(self):
        return OccupancyRecord(pycuda.tools.DeviceData(), self.n_thr)


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

    def __call__(self, extra=None, step_type=int32):
        args  = [step_type(self.i_step)]
        for k in self.device_state:
            args.append(getattr(self, k).device)
        try:
            kwds = extra or self.extra_args
            self._device_update(*args, **kwds)
        except cuda.LogicError as e:
            print 0, 'i_step', type(args[0])
            for i, k in enumerate(self.device_state):
                attr = getattr(self, k).device
                print i+1, k, type(attr), attr.dtype
            print kwds
            raise e
        self.i_step += self.n_msik
