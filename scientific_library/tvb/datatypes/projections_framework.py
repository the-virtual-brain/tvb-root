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
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
"""

The Framework component of ProjectionMatrices DataTypes.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import tvb.datatypes.projections_data as data


class ProjectionMatrixFramework(data.ProjectionMatrixData):
    """ This class exists to add framework methods to ProjectionMatrixData. """
    __tablename__ = None
    

    
class ProjectionRegionEEGFramework(data.ProjectionRegionEEGData):
    """ This class exists to add framework methods to ProjectionRegionEEGData. """
    __tablename__ = None
    
    def generate_new_projection(self, connectivity_gid, storage_path):
        """
        Generate a new projection matrix with the given connectivity gid from an 
        existing Region Projection corresponding to the parent connectivity.
        """
        new_projection = self.__class__()
        new_projection.storage_path = storage_path
        new_projection._sources = connectivity_gid
        new_projection._sensors = self._sensors
        new_projection.projection_data = self.projection_data
        return new_projection



class ProjectionSurfaceEEGFramework(data.ProjectionSurfaceEEGData):
    """ This class exists to add framework methods to ProjectionSurfaceEEGData. """
    __tablename__ = None