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

import numpy
from tvb.datatypes.fcd import Fcd
from tvb.datatypes.time_series import TimeSeries
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.fcd import FcdIndex
from tvb.adapters.datatypes.db.region_mapping import RegionMappingIndex
from tvb.adapters.datatypes.db.sensors import SensorsIndex
from tvb.adapters.datatypes.db.surface import SurfaceIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesIndex


def test_store_load_region_mapping(session, connectivity_factory, surface_factory, region_mapping_factory, sensors_factory):
    connectivity = connectivity_factory(2)
    conn_idx = ConnectivityIndex()
    conn_idx.fill_from_has_traits(connectivity)
    session.add(conn_idx)

    surface = surface_factory(5)
    surf_idx = SurfaceIndex()
    surf_idx.fill_from_has_traits(surface)
    session.add(surf_idx)

    region_mapping = region_mapping_factory(surface, connectivity)
    rm_idx = RegionMappingIndex()
    rm_idx.fill_from_has_traits(region_mapping)
    rm_idx.connectivity = conn_idx
    rm_idx.surface = surf_idx
    session.add(rm_idx)

    sensors = sensors_factory("SEEG", 3)
    sensors_seeg_idx = SensorsIndex()
    sensors_seeg_idx.fill_from_has_traits(sensors)
    session.add(sensors_seeg_idx)

    sensors_eeg = sensors_factory("EEG", 3)
    sensors_eeg_idx = SensorsIndex()
    sensors_eeg_idx.fill_from_has_traits(sensors_eeg)
    session.add(sensors_eeg_idx)

    time_series = TimeSeries(data=numpy.arange(5))

    fcd = Fcd(
        array_data=numpy.arange(5),
        source=time_series,
    )

    ts_index = TimeSeriesIndex()
    ts_index.fill_from_has_traits(time_series)
    session.add(ts_index)

    fcd_index = FcdIndex()
    fcd_index.fill_from_has_traits(fcd)
    fcd_index.source = ts_index
    session.add(fcd_index)

    session.commit()

    res = session.query(ConnectivityIndex)
    assert res.count() == 1
    assert res[0].number_of_regions == 2
    assert res[0].number_of_connections == 4
    assert res[0].undirected is True
    assert res[0].weights_min == 0

    res = session.query(SurfaceIndex)
    assert res.count() == 1

    res = session.query(RegionMappingIndex)
    assert res.count() == 1
