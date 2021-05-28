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
A plain NumPy backend which uses templating to generate simulation
code.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import os
import sys
import importlib
import numpy as np
import autopep8
import tempfile

from .templates import MakoUtilMix


class NpBackend(MakoUtilMix):

    def __init__(self):
        self.cgdir = tempfile.TemporaryDirectory()
        sys.path.append(self.cgdir.name)

    def build_py_func(self, template_source, content, name='kernel', print_source=False,
            modname=None, fname=None):
        "Build and retrieve one or more Python functions from template."
        source = self.render_template(template_source, content)
        source = autopep8.fix_code(source)
        if print_source:
            print(self.insert_line_numbers(source))
        if fname is not None:
            fullfname = os.path.join(self.cgdir.name, fname)
            with open(fullfname, 'w') as fd:
                fd.write(source)
        if modname is not None:
            return self.eval_module(source, name, modname)
        else:
            return self.eval_source(source, name, print_source)

    def eval_source(self, source, name, print_source):
        globals_ = {}
        try:
            exec(source, globals_)
        except Exception as exc:
            if not print_source:
                print(self._insert_line_numbers(source))
            raise exc
        fns = [globals_[n] for n in name.split(',')]
        return fns[0] if len(fns)==1 else fns

    def eval_module(self, source, name, modname):
        here = os.path.abspath(os.path.dirname(__file__))
        genp = os.path.join(here, 'templates', 'generated')
        with open(f'{genp}/{modname}.py', 'w') as fd:
            fd.write(source)
        fullmodname = f'tvb.simulator.backend.templates.generated.{modname}'
        mod = importlib.import_module(fullmodname)
        fns = [getattr(mod,n) for n in name.split(',')]
        return fns[0] if len(fns)==1 else fns


