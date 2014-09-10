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
Backend-side for TS Visualizer of TS Volume DataTypes.

.. moduleauthor:: Robert Parcus <betoparcus@gmail.com>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Ciprian Tomoiaga <ciprian.tomoiaga@codemart.ro>

"""

import json
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.time_series import TimeSeriesVolume



class TimeSeriesVolumeVisualiser(ABCDisplayer):

    _ui_name = "Time Series Volume Visualizer"
    _ui_subsection = "ts_volume"


    def get_input_tree(self):
        return [{'name': 'time_series_volume',
                 'label': 'Time Series Volume',
                 'type': TimeSeriesVolume,
                 'required': True}]


    def get_required_memory_size(self, **kwargs):
        """Return required memory."""
        return -1


    def launch(self, time_series_volume):

        volume = time_series_volume.volume
        minValue, maxValue = time_series_volume.get_min_max_values()
        dataUrls = [self.paths2url(time_series_volume, "get_rotated_volume_slice", parameter=""),
                    self.paths2url(time_series_volume, "get_volume_view", parameter=""),
                    self.paths2url(time_series_volume, "get_voxel_time_series", parameter="")
                    ]

        params = dict(title="Volumetric Time Series",
                      minValue=minValue, maxValue=maxValue,
                      dataUrls=json.dumps(dataUrls),
                      samplePeriod=time_series_volume.sample_period,
                      samplePeriodUnit=time_series_volume.sample_period_unit,
                      volumeShape=json.dumps(time_series_volume.read_data_shape()),
                      volumeOrigin=json.dumps(volume.origin.tolist()),
                      voxelUnit=volume.voxel_unit,
                      voxelSize=json.dumps(volume.voxel_size.tolist()))

        return self.build_display_result("time_series_volume/view", params,
                                         pages=dict(controlPage="time_series_volume/controls"))

