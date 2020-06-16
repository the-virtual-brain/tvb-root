# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import pytest
import scipy
import tvb
from tvb.basic.neotraits.ex import TraitAttributeError
from tvb.adapters.datatypes.h5.local_connectivity_h5 import LocalConnectivityH5
from tvb.adapters.datatypes.h5.projections_h5 import ProjectionMatrixH5
from tvb.adapters.datatypes.h5.structural_h5 import StructuralMRIH5
from tvb.adapters.datatypes.h5.volumes_h5 import VolumeH5
from tvb.adapters.datatypes.h5.connectivity_h5 import ConnectivityH5
from tvb.adapters.datatypes.h5.region_mapping_h5 import RegionMappingH5
from tvb.adapters.datatypes.h5.sensors_h5 import SensorsH5
from tvb.adapters.datatypes.h5.surface_h5 import SurfaceH5
from tvb.core.entities.file.simulator.simulation_history_h5 import SimulationHistoryH5, SimulationHistory
from tvb.datatypes.local_connectivity import LocalConnectivity
from tvb.datatypes.projections import ProjectionMatrix
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import Sensors
from tvb.datatypes.surfaces import Surface
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.volumes import Volume


def test_store_load_region_mapping(tmph5factory, region_mapping_factory):
    region_mapping = region_mapping_factory()
    rm_h5 = RegionMappingH5(tmph5factory())
    rm_h5.store(region_mapping)
    rm_h5.close()

    rm_stored = RegionMapping()
    with pytest.raises(TraitAttributeError):
        rm_stored.array_data
    rm_h5.load_into(rm_stored)  # loads connectivity/surface as None inside rm_stored
    assert rm_stored.array_data.shape == (5,)


def test_store_load_complete_region_mapping(tmph5factory, connectivity_factory, surface_factory, region_mapping_factory):
    connectivity = connectivity_factory(2)
    surface = surface_factory(5)
    region_mapping = region_mapping_factory(surface, connectivity)

    with ConnectivityH5(tmph5factory('Connectivity_{}.h5'.format(connectivity.gid))) as conn_h5:
        conn_h5.store(connectivity)
        conn_stored = Connectivity()
        conn_h5.load_into(conn_stored)

    with SurfaceH5(tmph5factory('Surface_{}.h5'.format(surface.gid))) as surf_h5:
        surf_h5.store(surface)
        surf_stored = Surface()
        surf_h5.load_into(surf_stored)

    with RegionMappingH5(tmph5factory('RegionMapping_{}.h5'.format(region_mapping.gid))) as rm_h5:
        rm_h5.store(region_mapping)
        rm_stored = RegionMapping()
        rm_h5.load_into(rm_stored)

    # load_into will not load dependent datatypes. connectivity and surface are undefined
    with pytest.raises(TraitAttributeError):
        rm_stored.connectivity
    with pytest.raises(TraitAttributeError):
        rm_stored.surface

    rm_stored.connectivity = conn_stored
    rm_stored.surface = surf_stored
    assert rm_stored.connectivity is not None
    assert rm_stored.surface is not None


def test_store_load_sensors(tmph5factory, sensors_factory):
    sensors = sensors_factory("SEEG", 3)
    tmp_file = tmph5factory("Sensors_{}.h5".format(sensors.gid))
    with SensorsH5(tmp_file) as f:
        f.store(sensors)

    sensors_stored = Sensors()
    with pytest.raises(TraitAttributeError):
        sensors_stored.labels

    with SensorsH5(tmp_file) as f:
        f.load_into(sensors_stored)
        assert sensors_stored.labels is not None


def test_store_load_partial_sensors(tmph5factory):
    sensors = Sensors(
        sensors_type="SEEG",
        labels=numpy.array(["s1", "s2", "s3"]),
        locations=numpy.zeros((3, 3)),
        number_of_sensors=3
    )

    tmp_file = tmph5factory("Sensors_{}.h5".format(sensors.gid))
    with SensorsH5(tmp_file) as f:
        f.store(sensors)

    sensors_stored = Sensors()
    with pytest.raises(TraitAttributeError):
        sensors_stored.labels
    with SensorsH5(tmp_file) as f:
        f.load_into(sensors_stored)
    assert sensors_stored.labels is not None


def test_store_load_volume(tmph5factory):
    volume = Volume(
        origin=numpy.zeros((3, 3)),
        voxel_size=numpy.zeros((3, 3))
    )

    tmp_file = tmph5factory("Volume_{}.h5".format(volume.gid))

    with VolumeH5(tmp_file) as f:
        f.store(volume)

    volume_stored = Volume()
    with pytest.raises(TraitAttributeError):
        volume_stored.origin

    with VolumeH5(tmp_file) as f:
        f.load_into(volume_stored)
    assert volume_stored.origin is not None


def test_store_load_structuralMRI(tmph5factory):
    volume = Volume(
        origin=numpy.zeros((3, 3)),
        voxel_size=numpy.zeros((3, 3))
    )

    structural_mri = StructuralMRI(
        array_data=numpy.zeros((3, 3)),
        weighting="T1",
        volume=volume
    )

    tmp_file = tmph5factory("StructuralMRI_{}.h5".format(volume.gid))

    with StructuralMRIH5(tmp_file) as f:
        f.store(structural_mri)

    structural_mri_stored = StructuralMRI()
    with pytest.raises(TraitAttributeError):
        structural_mri_stored.array_data
    with pytest.raises(TraitAttributeError):
        structural_mri_stored.volume

    with StructuralMRIH5(tmp_file) as f:
        f.load_into(structural_mri_stored)
    assert structural_mri_stored.array_data.shape == (3, 3)
    # referenced datatype is not loaded
    with pytest.raises(TraitAttributeError):
        structural_mri_stored.volume


def test_store_load_simulation_state(tmph5factory):
    history = SimulationHistory(
        history=numpy.arange(4),
        current_state=numpy.arange(4),
        current_step=42)

    tmp_file = tmph5factory("SimulationHistory_{}.h5".format(history.gid))

    with SimulationHistoryH5(tmp_file) as f:
        f.store(history)

    history_retrieved = SimulationHistory()
    assert history_retrieved.history is None
    with SimulationHistoryH5(tmp_file) as f:
        f.load_into(history_retrieved)
    assert history_retrieved.history is not None
    assert history_retrieved.history.shape == (4,)
    assert history_retrieved.current_step == 42


def test_store_load_projection_matrix(tmph5factory, sensors_factory, surface_factory):
    sensors = sensors_factory("SEEG", 3)
    cortical_surface = surface_factory(5, cortical=True)

    projection_matrix = ProjectionMatrix(
        projection_type="projSEEG",
        sources=cortical_surface,
        sensors=sensors,
        projection_data=numpy.zeros((5, 3))
    )

    tmp_file = tmph5factory("ProjectionMatrix_{}.h5".format(projection_matrix.gid))

    with ProjectionMatrixH5(tmp_file) as f:
        f.store(projection_matrix)


def test_store_load_local_connectivity(tmph5factory, surface_factory):
    tmp_file = tmph5factory()
    cortical_surface = surface_factory(5, cortical=True)

    local_connectivity = LocalConnectivity(
        surface=cortical_surface,
        matrix=scipy.sparse.csc_matrix(numpy.eye(8) + numpy.eye(8)[:, ::-1]),
        cutoff=12,
    )

    with LocalConnectivityH5(tmp_file) as f:
        f.store(local_connectivity)
        lc = LocalConnectivity()
        f.load_into(lc)
        assert type(lc.equation) == tvb.datatypes.equations.Gaussian
