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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
from tvb.adapters.exporters.abcexporter import ABCExporter
from tvb.adapters.uploaders.obj.parser import ObjWriter
from tvb.datatypes.surfaces import Surface



class ObjSurfaceExporter(ABCExporter):
    """ 
    Exports a tvb surface geometry in the obj format.
    """


    def get_supported_types(self):
        return [Surface]


    def get_label(self):
        return "Obj Format"


    def export(self, data, export_folder, project):
        download_file_name = self.get_export_file_name(data)
        data_file = os.path.join(export_folder, download_file_name)

        with open(data_file, 'w') as f:
            w = ObjWriter(f)
            w.write(data.vertices, data.triangles, data.vertex_normals, comment="exported from %s" % str(data))

        return download_file_name, data_file, False


    def get_export_file_extension(self, data):
        return "obj"
