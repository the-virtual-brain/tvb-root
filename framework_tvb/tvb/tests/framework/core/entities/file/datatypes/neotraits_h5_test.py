import numpy
import tvb
from tvb.core.entities.file.datatypes.local_connectivity_h5 import LocalConnectivityH5
from tvb.datatypes.simulation_state import SimulationState
from tvb.datatypes.structural import StructuralMRI
from tvb.datatypes.volumes import Volume
from tvb.core.entities.file.datatypes.projections_h5 import ProjectionMatrixH5
from tvb.core.entities.file.datatypes.simulation_state_h5 import SimulationStateH5
from tvb.core.entities.file.datatypes.structural_h5 import StructuralMRIH5
from tvb.core.entities.file.datatypes.volumes_h5 import VolumeH5
from .import testdatatypes
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import Sensors
from tvb.core.entities.file.datatypes.connectivity_h5 import ConnectivityH5
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.surfaces import Surface
from tvb.core.entities.file.datatypes.region_mapping_h5 import RegionMappingH5
from tvb.core.entities.file.datatypes.sensors_h5 import SensorsH5
from tvb.core.entities.file.datatypes.surface_h5 import SurfaceH5
from tvb.datatypes.local_connectivity import LocalConnectivity



def test_store_load_region_mapping(tmph5factory):
    rm_h5 = RegionMappingH5(tmph5factory())
    rm_h5.store(testdatatypes.region_mapping)
    rm_h5.close()

    rm_stored = RegionMapping()
    assert rm_stored.array_data is None
    rm_h5.load_into(rm_stored)  # loads connectivity/surface as None inside rm_stored
    assert rm_stored.array_data.shape == (5,)


def test_store_load_complete_region_mapping(tmph5factory):
    conn_h5 = ConnectivityH5(tmph5factory('Connectivity_{}.h5'.format(testdatatypes.connectivity.gid)))
    surf_h5 = SurfaceH5(tmph5factory('Surface_{}.h5'.format(testdatatypes.surface.gid)))
    rm_h5 = RegionMappingH5(tmph5factory('RegionMapping_{}.h5'.format(testdatatypes.region_mapping.gid)))

    conn_h5.store(testdatatypes.connectivity)
    conn_h5.close()  # use with

    surf_h5.store(testdatatypes.surface)
    surf_h5.close()

    rm_h5.store(testdatatypes.region_mapping)
    rm_h5.close()

    conn_stored = Connectivity()
    surf_stored = Surface()
    rm_stored = RegionMapping()

    conn_h5.load_into(conn_stored)
    surf_h5.load_into(surf_stored)
    rm_h5.load_into(rm_stored)
    assert rm_stored.connectivity is None
    assert rm_stored.surface is None
    rm_stored.connectivity = conn_stored
    rm_stored.surface = surf_stored
    assert rm_stored.connectivity is not None
    assert rm_stored.surface is not None


def test_store_load_sensors(tmph5factory):
    tmp_file = tmph5factory("Sensors_{}.h5".format(testdatatypes.sensors.gid))
    with SensorsH5(tmp_file) as f:
        f.store(testdatatypes.sensors)

    sensors_stored = Sensors()
    assert sensors_stored.labels is None
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
    assert sensors_stored.labels is None
    with SensorsH5(tmp_file) as f:
        f.load_into(sensors_stored)
    assert sensors_stored.labels is not None


def test_store_load_volume(tmph5factory):
    tmp_file = tmph5factory("Volume_{}.h5".format(testdatatypes.volume.gid))

    with VolumeH5(tmp_file) as f:
        f.store(testdatatypes.volume)

    volume_stored = Volume()
    assert volume_stored.origin is None
    with VolumeH5(tmp_file) as f:
        f.load_into(volume_stored)
    assert volume_stored.origin is not None


def test_store_load_structuralMRI(tmph5factory):
    structural_mri = StructuralMRI(
        array_data=numpy.zeros((3, 3)),
        weighting="T1",
        volume=testdatatypes.volume
    )

    tmp_file = tmph5factory("StructuralMRI_{}.h5".format(testdatatypes.volume.gid))

    with StructuralMRIH5(tmp_file) as f:
        f.store(structural_mri)

    structural_mri_stored = StructuralMRI()
    assert structural_mri_stored.array_data is None
    assert structural_mri_stored.volume is None
    with StructuralMRIH5(tmp_file) as f:
        f.load_into(structural_mri_stored)
    assert structural_mri_stored.array_data.shape == (3, 3)
    assert structural_mri_stored.volume is None


def test_store_load_simulation_state(tmph5factory):
    simulation_state = SimulationState(
        history=numpy.arange(4),
        current_state=numpy.arange(4),
        current_step=1
    )

    tmp_file = tmph5factory("SimulationState_{}.h5".format(simulation_state.gid))

    with SimulationStateH5(tmp_file) as f:
        f.store(simulation_state)

    simulation_state_stored = SimulationState()
    assert simulation_state_stored.history is None
    with SimulationStateH5(tmp_file) as f:
        f.load_into(simulation_state_stored)
    assert simulation_state_stored.history is not None


def test_store_load_projection_matrix(tmph5factory):
    tmp_file = tmph5factory("ProjectionMatrix_{}.h5".format(testdatatypes.projection_matrix.gid))

    with ProjectionMatrixH5(tmp_file) as f:
        f.store(testdatatypes.projection_matrix)


def test_store_load_local_connectivity(tmph5factory):
    tmp_file = tmph5factory()

    with LocalConnectivityH5(tmp_file) as f:
        f.store(testdatatypes.local_connectivity)
        lc = LocalConnectivity()
        f.load_into(lc)
        assert type(lc.equation) == tvb.datatypes.equations.Gaussian


