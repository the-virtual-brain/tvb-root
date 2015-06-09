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
The ProjectionMatrices DataTypes. This brings together the scientific and framework 
methods that are associated with the surfaces data.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import tvb.datatypes.projections_scientific as scientific
import tvb.datatypes.projections_framework as framework
from tvb.basic.readers import try_get_absolute_path
import numpy
import scipy.io



class ProjectionMatrix(framework.ProjectionMatrixFramework, scientific.ProjectionMatrixScientific):
    """
    This class brings together the scientific and framework methods that are
    associated with the ProjectionMatrix DataType.
    
    ::
        
                        ProjectionMatrixData
                                 |
                                / \\
        ProjectionMatrixFramework   ProjectionMatrixScientific
                                \ /
                                 |
                            ProjectionMatrix
        
    
    """
    @property
    def shape(self):
        return self.projection_data.shape

    @staticmethod
    def load_region_projection_matrix(result, source_file):
        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        if source_file.endswith(".mat"):
            # consider we have a brainstorm format
            raise Exception(
                    'Please import your Brainstorm Projection matrix by '
                    'instantiating the surface projection Class')
        elif source_file.endswith(".npy"):
            # numpy array with the projectino matrix arleady computed
            result.projection_data = numpy.load(source_full_path)
        else:
            raise Exception(
                    'The projection matrix must be either a numpy array'
                    ' or a brainstorm .mat file')
        return result

    @staticmethod
    def load_surface_projection_matrix(result, source_file):
        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        if source_file.endswith(".mat"):
            # consider we have a brainstorm format
            mat = scipy.io.loadmat(source_full_path)
            gain, loc, ori = (mat[field] for field in 'Gain GridLoc GridOrient'.split())
            result.projection_data = (gain.reshape((gain.shape[0], -1, 3)) * ori).sum(axis=-1)
        elif source_file.endswith(".npy"):
            # numpy array with the projectino matrix arleady computed
            result.projection_data = numpy.load(source_full_path)
        else:
            raise Exception(
                    'The projection matrix must be either a numpy array'
                    ' or a brainstorm mat file')
        return result

    


class ProjectionSurfaceEEG(framework.ProjectionSurfaceEEGFramework, 
                           scientific.ProjectionSurfaceEEGScientific, ProjectionMatrix):
    """
    This class brings together the scientific and framework methods that are
    associated with the ProjectionMatrix DataType.
    
    ::
        
                          ProjectionSurfaceEEGData
                                      |
                                     / \\
        ProjectionSurfaceEEGFramework   ProjectionSurfaceEEGScientific
                                     \ /
                                      |
                          ProjectionSurfaceEEG
        
    
    """

    @staticmethod
    def from_file(source_file='projection_EEG_surface.npy', instance=None):
        if instance is None:
            result = ProjectionSurfaceEEG()
        else:
            result = instance
        result = ProjectionMatrix.load_surface_projection_matrix(result, source_file=source_file)
        return result 


class ProjectionSurfaceMEG(framework.ProjectionSurfaceMEGFramework,
                           scientific.ProjectionSurfaceMEGScientific, ProjectionMatrix):
    """
    This class brings together the scientific and framework methods that are
    associated with the ProjectionMatrix DataType.

    ::

                          ProjectionSurfaceMEGData
                                      |
                                     / \\
        ProjectionSurfaceMEGFramework   ProjectionSurfaceMEGScientific
                                     \ /
                                      |
                          ProjectionSurfaceMEG


    """

    @staticmethod
    def from_file(source_file='projection_MEG_surface.npy', instance=None):
        if instance is None:
            result = ProjectionSurfaceMEG()
        else:
            result = instance
        result = ProjectionMatrix.load_surface_projection_matrix(result, source_file=source_file)
        return result 


class ProjectionRegionSEEG(framework.ProjectionRegionMEGFramework,
                          scientific.ProjectionRegionMEGScientific, ProjectionMatrix):
    """
    This class brings together the scientific and framework methods that are
    associated with the ProjectionMatrix DataType.

    ::

                          ProjectionRegionMEGData
                                     |
                                    / \\
        ProjectionRegionMEGFramework   ProjectionRegionMEGScientific
                                    \ /
                                     |
                           ProjectionRegionMEG


    """

    @staticmethod
    def from_file(source_file='projection_SEEG_region.npy', instance=None):
        if instance is None:
            result = ProjectionRegionSEEG()
        else:
            result = instance
        result = ProjectionMatrix.load_region_projection_matrix(result, source_file=source_file)
        return result 


class ProjectionSurfaceSEEG(framework.ProjectionSurfaceMEGFramework,
                           scientific.ProjectionSurfaceMEGScientific, ProjectionMatrix):
    """
    This class brings together the scientific and framework methods that are
    associated with the ProjectionMatrix DataType.

    ::

                          ProjectionSurfaceMEGData
                                      |
                                     / \\
        ProjectionSurfaceMEGFramework   ProjectionSurfaceMEGScientific
                                     \ /
                                      |
                          ProjectionSurfaceMEG


    """

    @staticmethod
    def from_file(source_file='projection_SEEG_surface.npy', instance=None):
        if instance is None:
            result = ProjectionSurfaceSEEG()
        else:
            result = instance
        result = ProjectionMatrix.load_surface_projection_matrix(result, source_file=source_file)
        return result 
