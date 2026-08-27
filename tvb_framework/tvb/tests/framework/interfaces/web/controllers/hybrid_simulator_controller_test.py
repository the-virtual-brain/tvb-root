from pathlib import Path

import numpy
import pytest

from tvb.core.entities.file.simulator.hybrid_view_model import HybridSimulatorAdapterModel, HybridSubnetworkViewModel
from tvb.datatypes.connectivity import Connectivity
from tvb.interfaces.web.controllers.simulator.hybrid_simulator_controller import HybridSimulatorController


HYBRID_TEMPLATES = Path(__file__).parents[5] / "interfaces/web/templates/jinja2/burst/hybrid"
HYBRID_SCRIPT = Path(__file__).parents[5] / "interfaces/web/static/js/hybrid_simulator.js"


def test_default_projections_cover_all_directed_subnetwork_pairs():
    connectivity = Connectivity(
        weights=numpy.arange(9, dtype=float).reshape((3, 3)),
        tract_lengths=numpy.arange(9, dtype=float).reshape((3, 3)) + 10,
        centres=numpy.zeros((3, 3)),
        region_labels=numpy.array(["A", "B", "C"]),
    )
    first = HybridSubnetworkViewModel(name="First", node_indices=numpy.array([0, 2]))
    second = HybridSubnetworkViewModel(name="Second", node_indices=numpy.array([1]))
    configuration = HybridSimulatorAdapterModel(subnetworks=[first, second])

    projections = HybridSimulatorController._default_projections(configuration, connectivity)

    assert len(projections) == 4
    inter = next(projection for projection in projections
                 if projection.source_id == first.stable_id and projection.target_id == second.stable_id)
    assert inter.kind == "inter"
    assert inter.weights.tolist() == [[3.0, 5.0]]
    assert inter.tract_lengths.tolist() == [[13.0, 15.0]]


@pytest.mark.parametrize("template", ["connectivity", "projections", "dynamics", "run_settings", "review"])
def test_hybrid_pages_use_the_scrollable_main_container(template):
    content = (HYBRID_TEMPLATES / f"{template}.html").read_text()

    assert '<div id="main">' in content


def test_projection_page_posts_to_the_selected_projection_and_initializes_after_page_load():
    template = (HYBRID_TEMPLATES / "projections.html").read_text()
    script = HYBRID_SCRIPT.read_text()

    assert "projections?projection_index={{ projection_index }}" in template
    assert 'name="projection_index"' not in template
    assert "document.readyState === 'loading'" in script


def test_subnetwork_select_all_and_launch_use_the_interactive_page_contracts():
    review = (HYBRID_TEMPLATES / "review.html").read_text()
    script = HYBRID_SCRIPT.read_text()

    assert "hybrid-select-all" in script
    assert 'data-launch-url="{{ deploy_context | safe }}/burst/hybrid/launch"' in review
    assert 'data-burst-url="{{ deploy_context | safe }}/burst/"' in review
    assert "fetch(button.dataset.launchUrl" in script
