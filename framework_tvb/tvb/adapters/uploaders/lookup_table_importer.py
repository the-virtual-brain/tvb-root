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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from tvb.adapters.uploaders.abcuploader import ABCUploader
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import LaunchException
from tvb.datatypes.lookup_tables import LookUpTable, PsiTable, NerfTable


class LookupTableImporter(ABCUploader):

    _ui_name = "Lookup Table"
    _ui_subsection = "lookup_table_importer"
    _ui_description = "Import Lookup Tables datatype from *.NPZ file"
    
    PSI_TABLE = 'Psi Table'
    NERF_TABLE = 'Nerf Table'
    logger = get_logger(__name__)


    def get_upload_input_tree(self):
        """
        Define input parameters for this importer.
        """
        return [{'name': 'psi_table_file', 'type': 'upload', 'required_type': '.npz',
                 'label': 'Please upload Psi table file in NPZ format.', 'required': True,
                 'description': 'Expected a NPZ file containing Psi table data.'},

                {'name': 'table_type', 'type': 'select', 
                 'label': 'Table type: ', 'required': True,
                 'options': [{'name': self.PSI_TABLE, 'value': self.PSI_TABLE},
                             {'name': self.NERF_TABLE, 'value': self.NERF_TABLE}]
                 }]


    def get_output(self):
        return [LookUpTable]


    def launch(self, psi_table_file, table_type):
        """
        Created required sensors from the uploaded file.
        """
        if psi_table_file is None:
            raise LaunchException("Please select Psi table file which contains data to import")
        try:
            table_data = numpy.load(psi_table_file)
        except IOError, msg:
            self.logger.exception(msg)
            raise LaunchException("Input file is not a valid NPZ file.")
        
        if table_type == self.PSI_TABLE:
            table_inst = PsiTable()
        elif table_type == self.NERF_TABLE:
            table_inst = NerfTable()
        else:
            raise LaunchException("Could not determine table type from selected option %s" % table_type)

        table_inst.storage_path = self.storage_path
        table_inst.xmin = numpy.array(table_data['min_max'][0] if table_data is not None else [])
        table_inst.xmax = numpy.array(table_data['min_max'][1] if table_data is not None else [])
        table_inst.data = numpy.array(table_data['f'] if table_data is not None else [])
        table_inst.number_of_values = table_data['f'].shape[0] if table_data is not None else 0
        table_inst.df = numpy.array(table_data['df'] if table_data is not None else [])
        table_inst.dx = numpy.array(float(table_inst.xmax - table_inst.xmin) / table_inst.number_of_values)
        table_inst.invdx = numpy.array(1 / table_inst.dx)

        return [table_inst]
    
    