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
Backend-side for Visualizers that display measures on regions in the brain volume.

.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""

import json
from tvb.basic.filters.chain import FilterChain
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.datatypes.region_mapping import RegionVolumeMapping



class RegionVolumeMappingVisualiser(ABCDisplayer):

    _ui_name = "Region Volume Mapping Visualizer"
    _ui_subsection = "ts_volume"


    def get_input_tree(self):
        return [{'name': 'region_mapping_volume', 'label': 'Region mapping', 'type': RegionVolumeMapping, 'required': True,},
                {'name': 'connectivity_measure', 'label': 'Connectivity measure',
                 'type': ConnectivityMeasure, 'required': False,
                 'description': 'A connectivity measure',
                 'conditions': FilterChain(fields=[FilterChain.datatype + '._nr_dimensions'],
                                           operations=["=="], values=[1])},]


    def get_required_memory_size(self, **kwargs):
        return -1


    def launch(self, region_mapping_volume, connectivity_measure=None):
        if connectivity_measure is None:
            min_value, max_value = [0, region_mapping_volume.connectivity.number_of_regions]
            url_volume_data = self.paths2url(region_mapping_volume, "get_volume_view", parameter="")
        else:
            min_value, max_value =[connectivity_measure.array_data.min(), connectivity_measure.array_data.max()]
            datatype_kwargs = json.dumps({'mapped_array': connectivity_measure.gid})
            url_volume_data = ABCDisplayer.paths2url(region_mapping_volume, "get_mapped_array_volume_view") + '/' + datatype_kwargs + '?'

        volume = region_mapping_volume.volume
        volume_shape = region_mapping_volume.read_data_shape()
        volume_shape = (1, ) + volume_shape

        params = dict(title="Volumetric Region Volume Mapping Visualizer",
                      minValue=min_value, maxValue=max_value,
                      urlVolumeData=url_volume_data,
                      volumeShape=json.dumps(volume_shape),
                      volumeOrigin=json.dumps(volume.origin.tolist()),
                      voxelUnit=volume.voxel_unit,
                      voxelSize=json.dumps(volume.voxel_size.tolist()))

        return self.build_display_result("time_series_volume/staticView", params,
                                         pages=dict(controlPage="time_series_volume/controls"))
