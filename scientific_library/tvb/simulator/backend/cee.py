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
This module enables execution of generated C code.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import tempfile
import subprocess
import ctypes
import psutil
import ctypes
import numpy as np

import driver

total_mem = psutil.phymem_usage().total

compilers = {
    'gcc': 'gcc -std=c99 -fPIC -shared -lm'.split(' '),
}
default_compiler = 'gcc'

def gen_noise_into(devary, dt):
    """
    Generate additive white noise into arrays.

    """
    devary.cpu[:] = np.random.normal(size=devary.shape)
    devary.cpu[:] *= sqrt(dt)

def dll(src, libname, compiler=None, debug=False):
    """
    Write src to a temporary file and compile as shared
    library that can be loaded afterwards.

    """
    args = compilers[compiler or default_compiler][:]
    if debug:
        file = open('temp.c', 'w')
    else:
        file = tempfile.NamedTemporaryFile(suffix='.c')
    
    with file as fd:
        fd.write(src)
        fd.flush()
        arg.append('-g' if debug else '-O3')
        ret = subprocess.call(args + [fd.name, '-o', libname])

    return ret

class srcmod(object):

    def __init__(self, src, fns, debug=False, printsrc=False):

        self.src = src
        if self.src.__class__.__module__ == 'cgen':
            self.src = '\n'.join(self.src.generate())

        if debug:
            print "backend.cee.srcmod:\n%s" % (self.src,)
            dll(self.src, 'temp.so', debug=debug)
            self._module = ctypes.CDLL('temp.so')
        else:
            with tempfile.NamedTemporaryFile(suffix='.so') as fd:
                dll(self.src, fd.name, debug=debug)
                self._module = ctypes.CDLL(fd.name)

        for f in fns:
            fn = getattr(self._module, f)
            if debug:
                def fn_(*args, **kwds):
                    try:
                        ret = fn(*args, **kwds)
                    except Exception as exc:
                        msg = 'ctypes call of %r failed w/ %r'
                        msg %= (f, exc)
                        raise Exception(msg)
                    return ret
            else:
                fn_ = fn
            setattr(self, f, fn_)


class Code(driver.Code):
    def __init__(self, *args, **kwds):
        super(Code, self).__init__(*args, **kwds)
        self.mod = srcmod("#include <math.h>\n" + self.source, self.fns)


class Global(driver.Global):
    """
    Encapsulates a source module C static in a Python data descriptor
    for easy handling

    """

    def post_init(self):
        if self.__post_init:
            self._cget = getattr(self.code.mod, 'get_'+self.name)
            self._cset = getattr(self.code.mod, 'set_'+self.name)
            self.__post_init = False

    def __get__(self, inst, ownr):
        self.post_init()
        return self._cget()

    def __set__(self, inst, val):
        self.post_init()
        ctype = ctypes.c_int32 if self.dtype==int32 else ctypes.c_float
        self._cset(ctype(val))

 
class Array(driver.Array):
    """
    Encapsulates an array that is on the device

    """

    @property
    def device(self):
        if not hasattr(self, '_device'):
            ctype = ctypes.c_float if self.type==float32 else ctypes.c_int32
            ptrtype = ctypes.POINTER(ctype)
            self._device = ascontiguousarray(self.cpu).ctypes.data_as(ptrtype)
        return self._device

    def set(self, ary):
        """
        In place update the device array.
        """
        _ = self.device
        delattr(self, '_device')
        self._cpu[:] = ary
        _ = self.cpu

    @property
    def value(self):
        return self.cpu.copy()


class Handler(driver.Handler):
    i_step_type = ctypes.c_int32
    def __init__(self, *args, **kwds):
        kwds.update({'Code': Code, 'Global': Global, 'Array': Array})
        super(Handler, self).__init__(*args, **kwds)
        self._device_update = self.code.mod.update
    @property
    def mem_info(self):
        if psutil:
            phymem = psutil.phymem_usage()
            return phymem.free, phymem.total
        else:
            return None, None
    @property
    def occupancy(self):
        pass
    @property
    def extra_args(self):
        return {}


