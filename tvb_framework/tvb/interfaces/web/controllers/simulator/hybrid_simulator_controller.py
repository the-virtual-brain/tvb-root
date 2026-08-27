import json
import threading
from datetime import datetime

import cherrypy
import numpy

from tvb.adapters.forms.hybrid_simulator_forms import HybridConnectivityForm
from tvb.adapters.forms.integrator_forms import get_form_for_integrator
from tvb.adapters.forms.model_forms import get_form_for_model, get_model_to_form_dict
from tvb.adapters.forms.simulator_fragments import SimulatorFinalFragment
from tvb.adapters.simulator.hybrid_simulator_adapter import HybridSimulatorAdapter
from tvb.core.entities import load
from tvb.core.entities.file.simulator.hybrid_view_model import HybridProjectionViewModel, HybridSubnetworkViewModel
from tvb.core.entities.file.simulator.view_model import EulerDeterministicViewModel, HeunDeterministicViewModel, \
    RawViewModel, TemporalAverageViewModel
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.neocom import h5
from tvb.core.services.burst_service import BurstService
from tvb.core.services.simulator_service import SimulatorService
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import context_selected, expose_json, expose_page, handle_error
from tvb.interfaces.web.controllers.simulator.hybrid_simulator_wizard_urls import HybridSimulatorWizardURLs
from tvb.interfaces.web.entities.context_hybrid_simulator import HybridSimulatorContext


@traced
class HybridSimulatorController(BurstBaseController):
    def __init__(self):
        super().__init__()
        self.context = HybridSimulatorContext()
        self.simulator_service = SimulatorService()
        self.burst_service = BurstService()

    def _page(self, template, step, **kwargs):
        params = {"title": "Hybrid Simulation", "mainContent": f"burst/hybrid/{template}",
                  "hybrid_step": step, "section": "burst", "subsection": "hybridsimulation"}
        params.update(kwargs)
        return self.fill_default_attributes(params)

    @staticmethod
    def _model_classes():
        return {model_class.__name__: model_class for model_class in get_model_to_form_dict()}

    @staticmethod
    def _connectivity(configuration):
        if configuration.connectivity is None:
            return None
        return h5.load_from_index(load.load_entity_by_gid(configuration.connectivity))

    @expose_page
    @context_selected
    def index(self):
        self.context.initialize()
        self.redirect(HybridSimulatorWizardURLs.CONNECTIVITY)

    @expose_page
    @context_selected
    @handle_error(redirect=True)
    def connectivity(self, **data):
        configuration = self.context.initialize()
        form = self.algorithm_service.prepare_adapter_form(
            form_instance=HybridConnectivityForm(), project_id=common.get_current_project().id)
        if cherrypy.request.method == "POST":
            form.fill_from_post(data)
            selected_gid = form.connectivity.value
            connectivity = h5.load_from_index(load.load_entity_by_gid(selected_gid))
            if configuration.connectivity != selected_gid:
                configuration.connectivity = selected_gid
                configuration.subnetworks = []
                configuration.projections = []
                self.context.store(configuration, 1)
                self.redirect(HybridSimulatorWizardURLs.CONNECTIVITY)
            if data.get("action") == "load":
                self.redirect(HybridSimulatorWizardURLs.CONNECTIVITY)
            grouping = json.loads(data.get("subnetworks", "[]"))
            configuration.connectivity = selected_gid
            configuration.subnetworks = [HybridSubnetworkViewModel(
                name=item["name"].strip(), node_indices=numpy.asarray(item["nodes"], dtype=numpy.int_))
                for item in grouping]
            configuration.projections = self._default_projections(configuration, connectivity)
            errors = configuration.validate(connectivity.number_of_regions)
            if errors:
                raise ValueError(" ".join(errors))
            self.context.store(configuration, 2)
            self.redirect(HybridSimulatorWizardURLs.PROJECTIONS)

        form.fill_from_trait(configuration)
        connectivity = self._connectivity(configuration)
        labels = connectivity.region_labels.tolist() if connectivity is not None else []
        initial = [{"name": subnet.name, "nodes": subnet.node_indices.tolist()}
                   for subnet in configuration.subnetworks]
        return self._page("connectivity", 1, form=form, labels_json=json.dumps(labels),
                          subnetworks_json=json.dumps(initial))

    @staticmethod
    def _default_projections(configuration, connectivity):
        projections = []
        for source in configuration.subnetworks:
            for target in configuration.subnetworks:
                rows = numpy.asarray(target.node_indices, dtype=int)
                columns = numpy.asarray(source.node_indices, dtype=int)
                projections.append(HybridProjectionViewModel(
                    kind="intra" if source.stable_id == target.stable_id else "inter",
                    source_id=source.stable_id, target_id=target.stable_id,
                    weights=connectivity.weights[numpy.ix_(rows, columns)].copy(),
                    tract_lengths=connectivity.tract_lengths[numpy.ix_(rows, columns)].copy()))
        return projections

    @expose_page
    @context_selected
    @handle_error(redirect=True)
    def projections(self, projection_index=0, **data):
        configuration = self.context.initialize()
        if not configuration.projections:
            self.redirect(HybridSimulatorWizardURLs.CONNECTIVITY)
        projection_index = int(projection_index)
        projection = configuration.projections[projection_index]
        if cherrypy.request.method == "POST":
            projection.weights = numpy.asarray(json.loads(data["weights"]), dtype=float)
            projection.tract_lengths = numpy.asarray(json.loads(data["tract_lengths"]), dtype=float)
            errors = configuration.validate(self._connectivity(configuration).number_of_regions)
            projection_errors = [error for error in errors if error.startswith("Projection")]
            if projection_errors:
                raise ValueError(" ".join(projection_errors))
            self.context.store(configuration, 2)
            if data.get("next"):
                self.redirect(HybridSimulatorWizardURLs.DYNAMICS)
            self.redirect(f"{HybridSimulatorWizardURLs.PROJECTIONS}?projection_index={projection_index}")
        names = {subnet.stable_id: subnet.name for subnet in configuration.subnetworks}
        subnet_by_id = {subnet.stable_id: subnet for subnet in configuration.subnetworks}
        labels = self._connectivity(configuration).region_labels
        source_labels = [str(labels[index]) for index in subnet_by_id[projection.source_id].node_indices]
        target_labels = [str(labels[index]) for index in subnet_by_id[projection.target_id].node_indices]
        choices = [{"index": index,
                    "label": f"{names[item.source_id]} -> {names[item.target_id]} ({item.kind})"}
                   for index, item in enumerate(configuration.projections)]
        return self._page("projections", 2, projection=projection, choices=choices,
                          projection_index=projection_index, weights=json.dumps(projection.weights.tolist()),
                          tract_lengths=json.dumps(projection.tract_lengths.tolist()),
                          source_labels=json.dumps(source_labels), target_labels=json.dumps(target_labels))

    @expose_page
    @context_selected
    @handle_error(redirect=True)
    def dynamics(self, subnetwork_id=None, **data):
        configuration = self.context.initialize()
        if not configuration.subnetworks:
            self.redirect(HybridSimulatorWizardURLs.CONNECTIVITY)
        subnet = next((item for item in configuration.subnetworks if item.stable_id == subnetwork_id),
                      configuration.subnetworks[0])
        model_classes = self._model_classes()
        if cherrypy.request.method == "POST":
            model_class = model_classes[data["model_class"]]
            model_changed = type(subnet.model) is not model_class
            if model_changed:
                subnet.model = model_class()
                subnet.observable = subnet.model.variables_of_interest[0]
            if model_changed or data.get("action") == "change_model":
                self.context.store(configuration, 3)
                self.redirect(f"{HybridSimulatorWizardURLs.DYNAMICS}?subnetwork_id={subnet.stable_id}")
            data["variables_of_interest"] = [data["observable"]]
            model_form = get_form_for_model(type(subnet.model))()
            model_form.fill_from_post(data)
            model_form.fill_trait(subnet.model)
            integrator_class = HeunDeterministicViewModel if data["integrator"] == \
                "HeunDeterministicViewModel" else EulerDeterministicViewModel
            if type(subnet.integrator) is not integrator_class:
                subnet.integrator = integrator_class()
            integrator_form = get_form_for_integrator(type(subnet.integrator))()
            integrator_form.fill_from_post(data)
            integrator_form.fill_trait(subnet.integrator)
            subnet.observable = data["observable"]
            self.context.store(configuration, 3)
            if data.get("next"):
                errors = configuration.validate(self._connectivity(configuration).number_of_regions)
                dynamics_errors = [error for error in errors if any(
                    token in error.lower() for token in ("integrat", "observable", "coupling"))]
                if dynamics_errors:
                    raise ValueError(" ".join(dynamics_errors))
                self.redirect(HybridSimulatorWizardURLs.RUN)
            self.redirect(f"{HybridSimulatorWizardURLs.DYNAMICS}?subnetwork_id={subnet.stable_id}")

        model_form = get_form_for_model(type(subnet.model))()
        model_form.fill_from_trait(subnet.model)
        integrator_form = get_form_for_integrator(type(subnet.integrator))()
        integrator_form.fill_from_trait(subnet.integrator)
        return self._page("dynamics", 3, configuration=configuration, subnet=subnet,
                          model_form=model_form, integrator_form=integrator_form,
                          model_classes=sorted(model_classes),
                          observables=list(type(subnet.model).variables_of_interest.element_choices))

    @expose_page
    @context_selected
    @handle_error(redirect=True)
    def run_settings(self, **data):
        configuration = self.context.initialize()
        if cherrypy.request.method == "POST":
            name_validation = SimulatorFinalFragment.is_burst_name_ok(data["simulation_name"])
            if name_validation is not True:
                raise ValueError(name_validation)
            configuration.simulation_length = float(data["simulation_length"])
            monitors = []
            if data.get("raw"):
                monitors.append(RawViewModel(period=float(data.get("raw_period", 0.1))))
            if data.get("temporal_average"):
                monitors.append(TemporalAverageViewModel(period=float(data["temporal_average_period"])))
            configuration.monitors = monitors
            errors = configuration.validate(self._connectivity(configuration).number_of_regions)
            monitor_errors = [error for error in errors if "monitor" in error.lower() or
                              "simulation length" in error.lower()]
            if monitor_errors:
                raise ValueError(" ".join(monitor_errors))
            self.context.store(configuration, 4)
            common.add2session("hybrid_simulation_name", data["simulation_name"])
            self.redirect(HybridSimulatorWizardURLs.REVIEW)
        selected = {type(monitor).__name__: monitor for monitor in configuration.monitors}
        return self._page("run_settings", 4, configuration=configuration, selected=selected,
                          simulation_name=common.get_from_session("hybrid_simulation_name") or "hybrid_simulation")

    @expose_page
    @context_selected
    def review(self):
        configuration = self.context.initialize()
        errors = configuration.validate(self._connectivity(configuration).number_of_regions)
        return self._page("review", 5, configuration=configuration, validation_errors=errors,
                          simulation_name=common.get_from_session("hybrid_simulation_name") or "hybrid_simulation")

    @expose_json
    @context_selected
    def launch(self):
        configuration = self.context.initialize()
        errors = configuration.validate(self._connectivity(configuration).number_of_regions)
        if errors:
            return {"error": " ".join(errors)}
        algorithm = self.algorithm_service.get_algorithm_by_module_and_class(
            HybridSimulatorAdapter.__module__, HybridSimulatorAdapter.__name__)
        if algorithm is None:
            return {"error": "Hybrid simulator adapter is not registered. Reinitialize TVB first."}
        burst = BurstConfiguration(common.get_current_project().id,
                                   name=common.get_from_session("hybrid_simulation_name") or "hybrid_simulation")
        burst.start_time = datetime.now()
        burst = self.burst_service.store_burst(burst)
        thread = threading.Thread(target=self.simulator_service.async_launch_and_prepare_simulation,
                                  kwargs={"burst_config": burst, "user": common.get_logged_user(),
                                          "project": common.get_current_project(), "simulator_algo": algorithm,
                                          "simulator": configuration})
        thread.start()
        self.context.reset()
        common.remove_from_session("hybrid_simulation_name")
        return {"id": burst.id}

    @expose_page
    @context_selected
    def reset(self):
        self.context.reset()
        self.redirect(HybridSimulatorWizardURLs.CONNECTIVITY)
