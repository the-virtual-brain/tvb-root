import numpy
from types import SimpleNamespace

from tvb.adapters.simulator.hybrid_simulator_adapter import HybridSimulatorAdapter
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.file.simulator.hybrid_view_model import HybridProjectionViewModel, \
    HybridSimulatorAdapterModel, HybridSubnetworkViewModel
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.services.project_service import initialize_storage
from tvb.simulator.hybrid import InterProjection, IntraProjection
from tvb.simulator.models import Generic2dOscillator, Kuramoto
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.tests.framework.core.factory import TestFactory


def test_builds_intra_and_inter_projections_in_connectivity_orientation():
    first = HybridSubnetworkViewModel(name="First", node_indices=numpy.array([0, 2]),
                                      model=Generic2dOscillator(), observable="V")
    second = HybridSubnetworkViewModel(name="Second", node_indices=numpy.array([1]),
                                       model=Kuramoto(), observable="theta")
    weights = numpy.arange(9, dtype=float).reshape((3, 3))
    lengths = weights + 1
    projections = []
    for source in (first, second):
        for target in (first, second):
            rows = target.node_indices
            columns = source.node_indices
            projections.append(HybridProjectionViewModel(
                kind="intra" if source is target else "inter",
                source_id=source.stable_id,
                target_id=target.stable_id,
                weights=weights[numpy.ix_(rows, columns)],
                tract_lengths=lengths[numpy.ix_(rows, columns)],
            ))
    configuration = HybridSimulatorAdapterModel(subnetworks=[first, second], projections=projections,
                                                 simulation_length=0.2)
    adapter = HybridSimulatorAdapter()

    simulator = adapter.build_simulator(configuration)

    assert len(simulator.nets.subnets[0].projections) == 1
    assert isinstance(simulator.nets.subnets[0].projections[0], IntraProjection)
    assert len(simulator.nets.projections) == 2
    assert all(isinstance(projection, InterProjection) for projection in simulator.nets.projections)
    assert simulator.nets.projections[0].weights.shape == (1, 2)
    assert simulator.nets.subnets[0].node_indices.tolist() == [0, 2]


def test_python_hybrid_simulator_returns_original_node_order():
    first = HybridSubnetworkViewModel(name="First", node_indices=numpy.array([0]), observable="V")
    second = HybridSubnetworkViewModel(name="Second", node_indices=numpy.array([1]), observable="V")
    projections = []
    for source in (first, second):
        for target in (first, second):
            projections.append(HybridProjectionViewModel(
                kind="intra" if source is target else "inter",
                source_id=source.stable_id,
                target_id=target.stable_id,
                weights=numpy.zeros((1, 1)),
                tract_lengths=numpy.zeros((1, 1)),
            ))
    configuration = HybridSimulatorAdapterModel(subnetworks=[first, second], projections=projections,
                                                 simulation_length=1.0)

    simulator = HybridSimulatorAdapter().build_simulator(configuration)
    simulator.configure()
    times, data = simulator.run(random_state=1)[0]

    assert len(times) == 1
    assert data.shape == (1, 1, 2, 1)


class TestHybridSimulatorAdapterLaunch(TransactionalTestCase):
    def transactional_setup_method(self):
        initialize_storage()
        algorithm = dao.get_algorithm_by_module(HybridSimulatorAdapter.__module__, HybridSimulatorAdapter.__name__)
        self.adapter = ABCAdapter.build_adapter(algorithm)
        self.user = TestFactory.create_user("Hybrid_Adapter_User")
        self.project = TestFactory.create_project(self.user, "Hybrid_Adapter_Project")

    def test_synchronous_launch_stores_region_time_series(self, connectivity_index_factory, operation_factory,
                                                          monkeypatch):
        monkeypatch.setattr("psutil.swap_memory", lambda: SimpleNamespace(free=0, total=0))
        connectivity_index = connectivity_index_factory(4)
        connectivity = h5.load_from_index(connectivity_index)
        first = HybridSubnetworkViewModel(name="First", node_indices=numpy.array([0, 2]))
        second = HybridSubnetworkViewModel(name="Second", node_indices=numpy.array([1, 3]))
        configuration = HybridSimulatorAdapterModel(connectivity=connectivity_index.gid,
                                                     subnetworks=[first, second], simulation_length=1.0)
        projections = []
        for source in (first, second):
            for target in (first, second):
                projections.append(HybridProjectionViewModel(
                    kind="intra" if source is target else "inter",
                    source_id=source.stable_id,
                    target_id=target.stable_id,
                    weights=connectivity.weights[numpy.ix_(target.node_indices, source.node_indices)],
                    tract_lengths=connectivity.tract_lengths[numpy.ix_(target.node_indices, source.node_indices)],
                ))
        configuration.projections = projections

        TestFactory.launch_synchronously(self.user.id, self.project, self.adapter, configuration)

        result = dao.get_generic_entity(TimeSeriesRegionIndex, "TimeSeriesRegion", "time_series_type")[0]
        assert (result.data_length_1d, result.data_length_2d, result.data_length_3d,
                result.data_length_4d) == (1, 1, 4, 1)
        assert result.fk_connectivity_gid == connectivity_index.gid
