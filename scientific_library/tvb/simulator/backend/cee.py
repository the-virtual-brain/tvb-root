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
Driver implementation for C99 + OpenMP

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import tempfile
import subprocess
import ctypes
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

try:
    import psutil
    total_mem = psutil.phymem_usage().total
except Exception as exc:
    LOG.exception(exc)
    LOG.warning('psutil not available, memory limits will not be respected')
    total_mem = 42**42


from . import base


# FIXME this is additive white noise
def gen_noise_into(devary, dt):
    devary.cpu[:] = random.normal(size=devary.shape)
    devary.cpu[:] *= sqrt(dt)


def dll(src, libname,
        args=['gcc', '-std=c99', '-fPIC', '-shared', '-lm'],
        debug=False):

    if debug:
        file = open('temp.c', 'w')
    else:
        file = tempfile.NamedTemporaryFile(suffix='.c')
    LOG.debug('open C file %r', file)

    with file as fd:
        fd.write(src)
        fd.flush()
        if debug:
            args.append('-g')
        else:
            args.append('-O3')
        args += [fd.name, '-o', libname]
        LOG.debug('calling %r', args)
        proc = subprocess.Popen(args, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        proc.wait()
        LOG.debug('return code %r', proc.returncode)
        if proc.returncode > 0:
            LOG.error('failed compilation:\n%s\n%s', proc.stdout.read(), proc.stderr.read())

class srcmod(object):

    def __init__(self, src, fns, debug=False, printsrc=False):

        if src.__class__.__module__ == 'cgen':
            self.src = '\n'.join(src.generate())
        else:
            self.src = src

        if printsrc:
            print "srcmod: source is \n%s" % (self.src,)

        if debug:
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

class C99Compiler(object):
    def __call__(self, src, libname, debug=False, io=subprocess.PIPE):
        args = [self.cmd] + self.flags
        if debug:
            file = open('temp.c', 'w')
        else:
            file = tempfile.NamedTemporaryFile(suffix='.c')
        LOG.debug('open C file %r', file)
        with file as fd:
            fd.write(src)
            fd.flush()
            args.append(self.g_flag if debug else self.opt_flag)
            args += [fd.name, '-o', libname]
            LOG.debug('calling %r', args)
            proc = subprocess.Popen(args, stdout=io, stderr=io)
            proc.wait()
            LOG.debug('return code %r', proc.returncode)
            if proc.returncode > 0:
                LOG.error('failed compilation:\n%s\n%s', 
                        proc.stdout.read(), proc.stderr.read())


class GCC(C99Compiler):
    cmd = 'gcc'
    flags = ['-std=c99', '-fPIC', '-shared', '-lm']
    g_flag = '-g'
    opt_flag = '-O3'


class Code(base.Code):
    "Interface to C code"

    def __init__(self, cc=None, **kwds):
        super(Code, self).__init__(*args, **kwds)
        self.cc = cc or GCC()

    def build_module(self, fns, debug=False):

        if debug:
            self.cc(self.src, 'temp.so', debug=debug)
            self._module = ctypes.CDLL('temp.so')
        else:
            with tempfile.NamedTemporaryFile(suffix='.so') as fd:
                self.cc(self.src, fd.name, debug=debug)
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


class Global(base.Global):

    def post_init(self):
        if self.__post_init:
            self._cget = getattr(self.code.mod, 'get_'+self.name)
            self._cset = getattr(self.code.mod, 'set_'+self.name)
            super(Global, self).post_init()

    def __get__(self, inst, ownr):
        self.post_init()
        return self._cget()

    def __set__(self, inst, val):
        self.post_init()
        ctype = ctypes.c_int32 if self.dtype==int32 else ctypes.c_float
        self._cset(ctype(val))
 

class Array(base.Array):

    @property
    def device(self):
        if not hasattr(self, '_device'):
            ctype = ctypes.c_float if self.type==float32 else ctypes.c_int32
            ptrtype = ctypes.POINTER(ctype)
            self._device = ascontiguousarray(self.cpu).ctypes.data_as(ptrtype)
        return self._device

    def set(self, ary):
        _ = self.device
        delattr(self, '_device')
        self._cpu[:] = ary
        _ = self.cpu

    @property
    def value(self):
        return self.cpu.copy()


class RegionParallel(base.RegionParallel):
    pass

    def get_update_fun(self):
        return device_code.mod.update

    @property
    def mem_info(self):
        if psutil:
            phymem = psutil.phymem_usage()
            return phymem.free, phymem.total
        else:
            return None, None

    @property
    def extra_args(self):
        return {}


    def __call__(self, extra=None, step_type=ctypes.c_int32):
        args  = [step_type(self.i_step)]
        for k in self.device_state:
            args.append(getattr(self, k).device)
        kwds = extra or self.extra_args
        self._device_update(*args, **kwds)
        self.i_step += self.n_msik
