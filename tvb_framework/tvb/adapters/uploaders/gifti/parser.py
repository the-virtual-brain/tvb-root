# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Calin Pavel <calin.pavel@codemart.ro>
"""

import os
import nibabel
import numpy
from nibabel.nifti1 import intent_codes, data_type_codes
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.exceptions import ParseException
from tvb.datatypes.surfaces import CorticalSurface, center_vertices, make_surface
from tvb.datatypes.time_series import TimeSeriesSurface


OPTION_READ_METADATA = "ReadFromMetaData"


class GIFTIParser(object):
    """
    This class reads content of a GIFTI file and builds / returns a Surface instance
    filled with details.
    """

    UNIQUE_ID_ATTR = "UniqueID"
    SUBJECT_ATTR = "SubjectID"
    ASP_ATTR = "AnatomicalStructurePrimary"
    DATE_ATTR = "Date"
    DESCRIPTION_ATTR = "Description"
    NAME_ATTR = "Name"
    TIME_STEP_ATTR = "TimeStep"

    def __init__(self, operation_id):
        self.logger = get_logger(__name__)
        self.operation_id = operation_id

    @staticmethod
    def _get_meta_dict(data_array):
        data_array_meta = data_array.meta
        if data_array_meta is None:
            return {}
        return data_array_meta

    @staticmethod
    def _is_surface_gifti(data_arrays):
        return (len(data_arrays) == 2
                and intent_codes.code["NIFTI_INTENT_POINTSET"] == data_arrays[0].intent
                and data_type_codes.code["NIFTI_TYPE_FLOAT32"] == data_arrays[0].datatype
                and intent_codes.code["NIFTI_INTENT_TRIANGLE"] == data_arrays[1].intent
                and data_type_codes.code["NIFTI_TYPE_INT32"] == data_arrays[1].datatype)

    @staticmethod
    def _is_timeseries_gifti(data_arrays):
        return (len(data_arrays) > 1
                and intent_codes.code["NIFTI_INTENT_TIME_SERIES"] == data_arrays[0].intent
                and data_type_codes.code["NIFTI_TYPE_FLOAT32"] == data_arrays[0].datatype)

    def _parse_surface(self, data_arrays, data_arrays_part2, surface_type, should_center):
        meta_dict = self._get_meta_dict(data_arrays[0])
        anatomical_structure_primary = meta_dict.get(self.ASP_ATTR)
        subject = meta_dict.get(self.SUBJECT_ATTR)
        title = meta_dict.get(self.NAME_ATTR)

        # Now try to determine what type of surface we have
        # If a surface type is not explicitly given we use the type specified in the metadata
        if surface_type == OPTION_READ_METADATA:
            surface_type = anatomical_structure_primary
        if surface_type is None:
            raise ParseException("Please specify the type of the surface")

        surface = make_surface(surface_type)
        if surface is None:
            raise ParseException("Could not determine surface type! %s" % surface_type)

        # Now fill TVB data type with metadata
        if subject is not None:
            surface.subject = subject
        if title is not None:
            surface.title = title

        surface.zero_based_triangles = True

        # Now fill TVB data type with geometry data
        vertices = data_arrays[0].data
        triangles = data_arrays[1].data
        vertices_in_lh = len(vertices)
        # If a second file is present append that data
        if data_arrays_part2 is not None:
            # offset the indices
            offset = len(vertices)
            vertices = numpy.vstack([vertices, data_arrays_part2[0].data])
            triangles = numpy.vstack([triangles, offset + data_arrays_part2[1].data])

        if should_center:
            vertices = center_vertices(vertices)

        # set hemisphere mask if cortex
        if isinstance(surface, CorticalSurface):
            # if there was a 2nd file then len(vertices) != vertices_in_lh
            surface.hemisphere_mask = numpy.zeros(len(vertices), dtype=numpy.bool_)
            surface.hemisphere_mask[vertices_in_lh:] = 1

        surface.vertices = vertices
        surface.number_of_vertices = surface.vertices.shape[0]
        surface.triangles = triangles
        surface.number_of_triangles = surface.triangles.shape[0]
        return surface

    def _parse_timeseries(self, data_arrays):
        # Create TVB time series to be filled
        time_series = TimeSeriesSurface()
        time_series.start_time = 0.0
        time_series.sample_period = 1.0

        # First process first data_array and extract important data from it's metadata
        meta_dict = self._get_meta_dict(data_arrays[0])
        sample_period = meta_dict.get(self.TIME_STEP_ATTR)
        time_series.subject = meta_dict.get(self.SUBJECT_ATTR)
        time_series.title = meta_dict.get(self.NAME_ATTR)

        if sample_period:
            time_series.sample_period = float(sample_period)
            time_series.sample_rate = 1 / time_series.sample_period

        return time_series, data_arrays

    # TODO: data_file_part2 should be optional and if it's not given, then it should be None, but it is actually taken as '' from h5 file, so it  acts as if it is required
    def parse(self, data_file, data_file_part2=None, surface_type=OPTION_READ_METADATA, should_center=False):
        """
        Parse NIFTI file(s) and returns A Surface or a TimeSeries for it.
        :param surface_type: one of "Cortex" "Head" "ReadFromMetaData"
        :param data_file_part2: a file containing the second part of the surface
        """
        self.logger.debug("Start to parse GIFTI file: %s" % data_file)
        if data_file is None:
            raise ParseException("Please select GIFTI file which contains data to import")
        if not os.path.exists(data_file):
            raise ParseException("Provided file %s does not exists" % data_file)
        if data_file_part2 is not None and not os.path.exists(data_file_part2):
            raise ParseException("Provided file part %s does not exists" % data_file_part2)

        try:
            gifti_image = nibabel.load(data_file)
            data_arrays = gifti_image.darrays

            self.logger.debug("File parsed successfully")
            if data_file_part2 is not None:
                data_arrays_part2 = nibabel.load(data_file_part2).darrays
            else:
                data_arrays_part2 = None
        except Exception as excep:
            self.logger.exception(excep)
            msg = "File: %s does not have a valid GIFTI format." % data_file
            raise ParseException(msg)

        self.logger.debug("Determine data type stored in GIFTI file")
        
        # First check if it's a surface
        if self._is_surface_gifti(data_arrays):
            # If a second part exists is must be of the same type
            if data_arrays_part2 is not None and not self._is_surface_gifti(data_arrays_part2):
                raise ParseException("Second file must be a surface too")
            return self._parse_surface(data_arrays, data_arrays_part2, surface_type, should_center)
        elif self._is_timeseries_gifti(data_arrays):
            return self._parse_timeseries(data_arrays)
        else:
            raise ParseException("Could not map data from GIFTI file to a TVB data type")