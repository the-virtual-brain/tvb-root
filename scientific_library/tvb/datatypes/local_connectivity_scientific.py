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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

import numpy

from tvb.datatypes.local_connectivity_data import LocalConnectivityData, LOG
from tvb.datatypes.surfaces import gdist


class LocalConnectivityScientific(LocalConnectivityData):
    """ This class exists to add scientific methods to LocalConnectivity """
    __tablename__ = None


    def _find_summary_info(self):
        """
        Gather scientifically interesting summary information from an instance
        of this datatype.
        """
        return self.get_info_about_array('matrix',
                                         [self.METADATA_ARRAY_MAX,
                                          self.METADATA_ARRAY_MIN,
                                          self.METADATA_ARRAY_MEAN,
                                          self.METADATA_ARRAY_SHAPE])


    def compute_sparse_matrix(self):
        """
        NOTE: Before calling this method, the surface field
        should already be set on the local connectivity.

        Computes the sparse matrix for this local connectivity.
        """
        if self.surface is None:
            msg = " Before calling 'compute_sparse_matrix' method, the surface field should already be set."
            LOG.error(msg)
            raise Exception(msg)

        self.matrix_gdist = gdist.local_gdist_matrix(self.surface.vertices.astype(numpy.float64),
                                                     self.surface.triangles.astype(numpy.int32),
                                                     max_distance=self.cutoff)
        self.compute()
        # Avoid having a large data-set in memory.
        self.matrix_gdist = None
