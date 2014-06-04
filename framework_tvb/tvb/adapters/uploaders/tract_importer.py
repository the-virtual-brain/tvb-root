# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import numpy

from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.files_helper import TvbZip
from tvb.core.entities.storage import transactional
from tvb.datatypes.tracts import Tracts


class TractsImporter(ABCUploader):
    """
    This imports geometry data stored in wavefront obj format
    """
    _ui_name = "Tracts"
    _ui_subsection = "tracts_importer"
    _ui_description = "Import tracts"


    def get_upload_input_tree(self):
        return [{'name': 'data_file', 'type': 'upload', 'required_type': '.zip',
                 'label': 'Please select file to import', 'required': True}]
        
        
    def get_output(self):
        return [Tracts]


    @transactional
    def launch(self, data_file):
        if data_file is None:
            raise LaunchException("Please select ZIP file which contains data to import")

        #todo warn: all in memory, this is memory hungry; at least twice the tractografy

        tracts = []
        max_tract_count = 0;

        with TvbZip(data_file) as zipf:
            # todo sort tract8 before tract74 parse ints out of file names
            for tractf in zipf.namelist():
                vertices_file = zipf.open(tractf)
                tract_vertices = numpy.loadtxt(vertices_file, dtype=numpy.float32)
                tracts.append(tract_vertices)
                max_tract_count = max(max_tract_count, len(tract_vertices))
                vertices_file.close()

        vertices_arr = numpy.zeros((len(tracts), max_tract_count, 3))
        counts_arr = numpy.zeros((len(tracts),))

        for i, tr in enumerate(tracts):
            vertices_arr[i, :len(tr)] = tr
            counts_arr[i] = len(tr)

        datatype = Tracts()
        datatype.storage_path = self.storage_path
        datatype.vertices = vertices_arr
        datatype.vertex_counts = counts_arr
        return datatype
