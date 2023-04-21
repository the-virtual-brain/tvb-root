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

"""
A CUDA backend which uses templating to generate simulation
code, with PyCUDA as the driver.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""


try:
    import pycuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.driver as drv
    from pycuda.driver import Out, In, InOut
    pycuda_available = True
except Exception as exc:
	pycuda_available = False

from .templates import MakoUtilMix


class CuBackend(MakoUtilMix):

    def build_func(self, template_source, content, name='kernel', print_source=False):
        "Build and retrieve a Python function from template."
        source = self.render_template(template_source, content)
        if print_source:
            print(source)
        try:
            module = SourceModule(source)
        except pycuda.driver.CompileError as exc:
            print(self.insert_line_numbers(source))
            raise exc
        func = module.get_function(name)
        return func
