# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
A Futhark backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os
import sys
import importlib
import tempfile
import hashlib
import weakref
import subprocess

import futhark_ffi

from .templates import MakoUtilMix


class FutBackend(MakoUtilMix):

    def __init__(self, backend='c', path=None):
        self.backend = backend
        if path is None:
            self.cgdir = tempfile.TemporaryDirectory()
            self.path = self.cgdir.name
        if path not in sys.path:
            sys.path.append(self.path)
        self.modcache = {} # weakref.WeakValueDictionary()

    # def __del__(self):
    #     "Clean up sys.path when removing backend."
    #     if hasattr(self, 'cgdir'):
    #         del sys.path[self.cgdir.name]
    
    def build_module(self, template_source, content, print_source=False):
        source = self.render_template(template_source, content)
        if print_source:
            print(self.insert_line_numbers(source))
        return self.build_fut_module(source)
    
    def hash(self, str):
        h = hashlib.sha256()
        h.update(str.encode('ascii'))
        return h.hexdigest()[:10]
    
    def build_fut_module(self, source: str):
        hash = self.hash(source)
        if hash not in self.modcache:
            self.build_new_fut_module(source, hash)
        return self.modcache[hash]

    def build_new_fut_module(self, source: str, hash: str):
        fname = f'{hash}.fut'
        ffname = os.path.join(self.cgdir.name, fname)
        with open(ffname, 'w') as fd:
            fd.write(source)
        self.run(self.futhark_bin, self.backend, '--library', fname)
        self.run('build_futhark_ffi', hash)
        _mod = importlib.import_module(f'_{hash}')
        mod = futhark_ffi.Futhark(_mod)
        self.modcache[hash] = mod

    @property
    def futhark_bin(self):
        # NB use where on Windows
        return subprocess.check_output(['which', 'futhark']).decode('ascii').strip()

    def run(self, *args, **kwargs):
        env = {k: v for k, v in os.environ.items()}
        env.update(kwargs)
        cmd = list(args)
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env, cwd=self.path)
        except subprocess.CalledProcessError as exc:
            print(exc.stdout.decode('utf_8'))


def try_setup_futhark():
    download_futhark()
    pip_install_futhark_ffi()

if __name__ == '__main__':
    try_setup_futhark()