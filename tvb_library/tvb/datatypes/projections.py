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
The ProjectionMatrices DataTypes.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.basic.readers import try_get_absolute_path, FileReader
from tvb.datatypes import surfaces, sensors
from tvb.basic.neotraits.api import HasTraits, TVBEnum, Attr, NArray, Final


class ProjectionsTypeEnum(TVBEnum):
    EEG = "projEEG"
    MEG = "projMEG"
    SEEG = "projSEEG"


class ProjectionMatrix(HasTraits):
    """
    Base DataType for representing a ProjectionMatrix.
    The projection is between a source of type CorticalSurface and a set of Sensors.
    """

    projection_type = Final(field_type=str)

    brain_skull = Attr(
        field_type=surfaces.BrainSkull,
        label="Brain Skull", default=None, required=False,
        doc="""Boundary between skull and cortex domains.""")

    skull_skin = Attr(
        field_type=surfaces.SkullSkin,
        label="Skull Skin", default=None, required=False,
        doc="""Boundary between skull and skin domains.""")

    skin_air = Attr(
        field_type=surfaces.SkinAir,
        label="Skin Air", default=None, required=False,
        doc="""Boundary between skin and air domains.""")

    conductances = Attr(
        field_type=dict, label="Domain conductances", required=False,
        default={'air': 0.0, 'skin': 1.0, 'skull': 0.01, 'brain': 1.0},
        doc=""" A dictionary representing the conductances of ... """)

    sources = Attr(
        field_type=surfaces.CorticalSurface,
        label="surface or region", default=None)

    sensors = Attr(
        field_type=sensors.Sensors,
        label="Sensors", default=None, required=False,
        doc=""" A set of sensors to compute projection matrix for them. """)

    projection_data = NArray(label="Projection Matrix Data", default=None, required=True)

    @property
    def shape(self):
        return self.projection_data.shape

    @classmethod
    def from_file(cls, source_file, matlab_data_name=None, is_brainstorm=False):

        proj = cls()

        source_full_path = try_get_absolute_path("tvb_data.projectionMatrix", source_file)
        reader = FileReader(source_full_path)
        if is_brainstorm:
            proj.projection_data = reader.read_gain_from_brainstorm()
        else:
            proj.projection_data = reader.read_array(matlab_data_name=matlab_data_name)
        return proj


class ProjectionSurfaceEEG(ProjectionMatrix):
    """
    Specific projection, from a CorticalSurface to EEG sensors.
    """

    projection_type = Final(field_type=str, default=ProjectionsTypeEnum.EEG.value)

    sensors = Attr(field_type=sensors.SensorsEEG)

    @classmethod
    def from_file(cls, source_file='projection_eeg_65_surface_16k.npy', matlab_data_name="ProjectionMatrix",
                  is_brainstorm=False):
        return ProjectionMatrix.from_file.__func__(cls, source_file, matlab_data_name, is_brainstorm)


class ProjectionSurfaceMEG(ProjectionMatrix):
    """
    Specific projection, from a CorticalSurface to MEG sensors.
    """

    projection_type = Final(field_type=str, default=ProjectionsTypeEnum.MEG.value)

    sensors = Attr(field_type=sensors.SensorsMEG)

    @classmethod
    def from_file(cls, source_file='projection_meg_276_surface_16k.npy', matlab_data_name=None, is_brainstorm=False):
        return ProjectionMatrix.from_file.__func__(cls, source_file, matlab_data_name, is_brainstorm)


class ProjectionSurfaceSEEG(ProjectionMatrix):
    """
    Specific projection, from a CorticalSurface to SEEG sensors.
    """

    projection_type = Final(field_type=str, default=ProjectionsTypeEnum.SEEG.value)

    sensors = Attr(field_type=sensors.SensorsInternal)

    @classmethod
    def from_file(cls, source_file='projection_seeg_588_surface_16k.npy', matlab_data_name=None, is_brainstorm=False):
        return ProjectionMatrix.from_file.__func__(cls, source_file, matlab_data_name, is_brainstorm)


def make_proj_matrix(proj_type):
    """
    Build a ProjectionMatrix instance, based on an input type
    :param proj_type: one of the supported subtypes
    :return: Instance of the corresponding projectiion matrix class, or None
    """
    if proj_type == ProjectionsTypeEnum.EEG.value:
        return ProjectionSurfaceEEG()
    elif proj_type == ProjectionsTypeEnum.MEG.value:
        return ProjectionSurfaceMEG()
    elif proj_type == ProjectionsTypeEnum.SEEG.value:
        return ProjectionSurfaceSEEG()
    return None
