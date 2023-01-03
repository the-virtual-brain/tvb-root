# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""
from enum import Enum

from tvb.contrib.scripts.datatypes.base import BaseModel
from tvb.datatypes.projections import ProjectionMatrix as TVBProjectionMatrix
from tvb.datatypes.projections import ProjectionSurfaceEEG as TVBProjectionSurfaceEEG
from tvb.datatypes.projections import ProjectionSurfaceSEEG as TVBProjectionSurfaceSEEG
from tvb.datatypes.projections import ProjectionSurfaceMEG as TVBProjectionSurfaceMEG
from tvb.datatypes.sensors import SensorTypes


class TvbProjectionType(Enum):
    eeg = TVBProjectionSurfaceEEG
    internal = TVBProjectionSurfaceSEEG
    meg = TVBProjectionSurfaceMEG


def get_TVB_proj_type(s_type):
    try:
        return TvbProjectionType[s_type.value.lower()].value
    except (KeyError, AttributeError):
        return TVBProjectionMatrix


class ProjectionMatrix(TVBProjectionMatrix, BaseModel):

    def to_tvb_instance(self, datatype=TVBProjectionMatrix, **kwargs):
        return super(ProjectionMatrix, self).to_tvb_instance(datatype, **kwargs)

    @classmethod
    def _from_tvb_projection_file(cls, s_type, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        if source_file.endswith("mat") and matlab_data_name is None:
            if len(s_type) > 0:
                matlab_data_name = kwargs.pop("%s_matlab_data_name" % s_type.lower(),
                                              kwargs.get("matlab_data_name", "ProjectionMatrix"))
            else:
                matlab_data_name = kwargs.get("matlab_data_name", "ProjectionMatrix")
        tvb_instance = get_TVB_proj_type(s_type).from_file(source_file, matlab_data_name, is_brainstorm)
        result = cls.from_tvb_instance(tvb_instance, **kwargs)
        if return_tvb_instance:
            return result, tvb_instance
        else:
            return result

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file("", source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)


class ProjectionSurfaceEEG(ProjectionMatrix, TVBProjectionSurfaceEEG):

    def to_tvb_instance(self, **kwargs):
        return super(ProjectionSurfaceEEG, self).to_tvb_instance(TVBProjectionSurfaceEEG, **kwargs)

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file(SensorTypes.TYPE_EEG, source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)


class ProjectionSurfaceSEEG(ProjectionMatrix, TVBProjectionSurfaceSEEG):

    def to_tvb_instance(self, **kwargs):
        return super(ProjectionSurfaceSEEG, self).to_tvb_instance(TVBProjectionSurfaceSEEG, **kwargs)

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file(SensorTypes.TYPE_INTERNAL, source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)


class ProjectionSurfaceMEG(ProjectionMatrix, TVBProjectionSurfaceMEG):

    def to_tvb_instance(self, **kwargs):
        return super(ProjectionSurfaceMEG, self).to_tvb_instance(TVBProjectionSurfaceMEG, **kwargs)

    @classmethod
    def from_tvb_file(cls, source_file, matlab_data_name=None, is_brainstorm=False,
                      return_tvb_instance=False, **kwargs):
        return cls._from_tvb_projection_file(SensorTypes.TYPE_MEG, source_file, matlab_data_name, is_brainstorm,
                                             return_tvb_instance, **kwargs)
