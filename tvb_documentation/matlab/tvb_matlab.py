# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

# Workarounds for specific problems in MATLAB-Python interactions

import os
import glob
import os.path
import sys
import types
import ctypes

class OutStream(object):

    def __init__(self, matlab_root=''):
        if sys.platform == 'darwin':
            libname = 'mex'
        elif sys.platform == 'win32':
            libname = 'libmex'
        elif sys.platform == 'linux2':
            libname = '%s/glnxa64/libmex.so' % (matlab_root, )
        else:
            print('unsupported platform %r' % (sys.platform,))
        self.lib = ctypes.CDLL(libname)

    def write(self, str):
        self.lib.mexPrintf(str)


class UnsupportedModule(types.ModuleType):

    def __init__(self, name):
        self.__name__ = name

    def __getattr__(self, _):
        msg = "The %s module is not currently supported."
        msg %= self.__name__
        raise ImportError(msg)


def find_potential_modules(path, modname):
    for pot_mod in glob.glob(os.path.join(path, '*')):
        ispackage = os.path.exists(os.path.sep.join([pot_mod, '__init__.py']))
        ismodule = pot_mod.endswith('.py')
        if (ispackage or ismodule) and (modname in pot_mod):
            yield pot_mod


def find_on_sys_path(modname):
    for path in sys.path:
        for pot_mod in find_potential_modules(path, modname):
            yield pot_mod


def find_sys_mod_names(root, base):
    for path in find_potential_modules(root, base):
        _, part = path.split(root)
        part = part.replace('.py', '')
        yield '%s%s' % (base, part.replace(os.path.sep, '.'))


def unsupport_module(modname):
    h5py_path, = find_on_sys_path('h5py')
    h5py_mods = find_sys_mod_names(h5py_path, 'h5py')
    sys.modules[modname] = UnsupportedModule(modname)
    for submodname in h5py_mods:
        sys.modules[submodname] = UnsupportedModule(submodname)


def setup():

    if sys.platform != 'darwin':
        unsupport_module('h5py')

    import logging
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.MATLAB_PROFILE)

    # MATLAB states the module doesn't exist if not importable and provides no traceback
    # to diagnose the import error, so we'll need to workaround this in the future. For now,
    # just try to import the simlab and report if it worked or not.
    try:
        import tvb.simulator.lab
        print('TVB modules available.')
    except Exception as exc:
        #print 'failed to import all TVB modules, not all functionality may be .'
        pass


def run_sim_with_seed(sim, length, seed):
    try:
        import numpy as np
        rstate = np.random.RandomState(int(seed)).get_state()
        return sim.run(simulation_length=length, random_state=rstate)
    except Exception as exc:
        print('unable to run: %r' % (exc,))
