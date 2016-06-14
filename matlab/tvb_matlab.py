# Workarounds for specific problems in MATLAB-Python interactions

import os
import glob
import os.path
import sys
import types


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
    unsupport_module('h5py')

    import logging
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.MATLAB_PROFILE)


