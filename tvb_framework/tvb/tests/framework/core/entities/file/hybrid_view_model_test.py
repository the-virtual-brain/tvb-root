import numpy

from tvb.core.entities.file.simulator.hybrid_view_model import HybridProjectionViewModel, \
    HybridSimulatorAdapterModel, HybridSubnetworkViewModel
from tvb.core.neocom import h5


def _configuration():
    first = HybridSubnetworkViewModel(name="Cortex", node_indices=numpy.array([0, 2]))
    second = HybridSubnetworkViewModel(name="Thalamus", node_indices=numpy.array([1, 3]))
    projections = []
    for source in (first, second):
        for target in (first, second):
            projections.append(HybridProjectionViewModel(
                kind="intra" if source is target else "inter",
                source_id=source.stable_id,
                target_id=target.stable_id,
                weights=numpy.ones((2, 2)),
                tract_lengths=numpy.ones((2, 2)),
            ))
    return HybridSimulatorAdapterModel(subnetworks=[first, second], projections=projections)


def test_hybrid_view_model_round_trip(tmp_path):
    configuration = _configuration()
    h5.store_view_model(configuration, str(tmp_path))
    loaded = h5.load_view_model(configuration.gid, str(tmp_path))

    assert [subnet.name for subnet in loaded.subnetworks] == ["Cortex", "Thalamus"]
    assert loaded.subnetworks[0].node_indices.tolist() == [0, 2]
    assert len(loaded.projections) == 4
    assert loaded.projections[2].weights.shape == (2, 2)


def test_hybrid_validation_reports_node_and_projection_errors():
    configuration = _configuration()
    configuration.subnetworks[1].node_indices = numpy.array([0, 3])
    configuration.projections[0].weights = numpy.ones((1, 2))

    errors = configuration.validate(node_count=4)

    assert any("exactly one" in error for error in errors)
    assert any("weights shape" in error for error in errors)
