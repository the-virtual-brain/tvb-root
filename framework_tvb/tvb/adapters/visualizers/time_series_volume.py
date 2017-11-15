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
Backend-side for TS Visualizer of TS Volume DataTypes.

.. moduleauthor:: Robert Parcus <betoparcus@gmail.com>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ciprian Tomoiaga <ciprian.tomoiaga@codemart.ro>

"""

import json
from tvb.adapters.visualizers.region_volume_mapping import _MappedArrayVolumeBase
from tvb.basic.filters.chain import FilterChain
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.entities.storage import dao
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.time_series import TimeSeries, TimeSeriesVolume



class TimeSeriesVolumeVisualiser(ABCDisplayer):

    _ui_name = "Time Series Volume Visualizer"
    _ui_subsection = "volume"


    def get_input_tree(self):
        return [{'name': 'time_series', 'label': 'Time Series', 'type': TimeSeries, 'required': True,
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._has_volume_mapping'],
                                           operations=["=="], values=[True])},
                _MappedArrayVolumeBase.get_background_input_tree()]


    def get_required_memory_size(self, **kwargs):
        """Return required memory."""
        return -1


    def launch(self, time_series, background=None):

        min_value, max_value = time_series.get_min_max_values()
        url_volume_data = self.paths2url(time_series, "get_volume_view", parameter="")
        url_timeseries_data = self.paths2url(time_series, "get_voxel_time_series", parameter="")

        if isinstance(time_series, TimeSeriesVolume):
            volume = time_series.volume
            volume_shape = time_series.read_data_shape()
        else:
            volume = time_series.region_mapping_volume.volume
            volume_shape = [time_series.read_data_shape()[0]]
            volume_shape.extend(time_series.region_mapping_volume.shape)

        params = dict(title="Volumetric Time Series",
                      ts_title=time_series.title,
                      labelsStateVar=time_series.labels_dimensions.get(time_series.labels_ordering[1], []),
                      labelsModes=range(time_series.read_data_shape()[3]),
                      minValue=min_value, maxValue=max_value,
                      urlVolumeData=url_volume_data,
                      urlTimeSeriesData=url_timeseries_data,
                      samplePeriod=time_series.sample_period,
                      samplePeriodUnit=time_series.sample_period_unit,
                      volumeShape=json.dumps(volume_shape),
                      volumeOrigin=json.dumps(volume.origin.tolist()),
                      voxelUnit=volume.voxel_unit,
                      voxelSize=json.dumps(volume.voxel_size.tolist()))

        if background is None:
            background = dao.try_load_last_entity_of_type(self.current_project_id, StructuralMRI)

        params.update(_MappedArrayVolumeBase._compute_background(background))

        return self.build_display_result("time_series_volume/view", params,
                                         pages=dict(controlPage="time_series_volume/controls"))

