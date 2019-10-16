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

import numpy
from tvb.datatypes.fcd import Fcd
from tvb.datatypes.time_series import TimeSeries
from tvb.core.entities.model.datatypes.connectivity import ConnectivityIndex
from tvb.core.entities.model.datatypes.fcd import FcdIndex
from tvb.core.entities.model.datatypes.region_mapping import RegionMappingIndex
from tvb.core.entities.model.datatypes.sensors import SensorsIndex
from tvb.core.entities.model.datatypes.surface import SurfaceIndex
from tvb.core.entities.model.datatypes.time_series import TimeSeriesIndex


def test_store_load_region_mapping(session, connectivityFactory, surfaceFactory, regionMappingFactory, sensorsFactory):
    connectivity = connectivityFactory(2)
    conn_idx = ConnectivityIndex()
    conn_idx.fill_from_has_traits(connectivity)
    session.add(conn_idx)

    surface = surfaceFactory(5)
    surf_idx = SurfaceIndex()
    surf_idx.fill_from_has_traits(surface)
    session.add(surf_idx)

    region_mapping = regionMappingFactory(surface, connectivity)
    rm_idx = RegionMappingIndex()
    rm_idx.fill_from_has_traits(region_mapping)
    rm_idx.connectivity = conn_idx
    rm_idx.surface = surf_idx
    session.add(rm_idx)

    sensors = sensorsFactory("SEEG", 3)
    sensors_seeg_idx = SensorsIndex()
    sensors_seeg_idx.fill_from_has_traits(sensors)
    session.add(sensors_seeg_idx)

    sensors_eeg = sensorsFactory("EEG", 3)
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
