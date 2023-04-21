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

"""
Steps which a user needs to execute for achieving a 
given action are described here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import json
import sys
import cherrypy
import formencode
import numpy
import six
from tvb.adapters.forms.equation_forms import get_form_for_equation

from tvb.basic.neotraits.api import TVBEnum, SubformEnum
from tvb.basic.neotraits.ex import TraitValueError
from tvb.core.adapters import constants
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.neocom import h5
from tvb.core.neocom.h5 import REGISTRY
from tvb.core.neotraits.forms import TraitDataTypeSelectField
from tvb.core.neotraits.view_model import DataTypeGidAttr
from tvb.core.services.exceptions import OperationException
from tvb.core.services.operation_service import OperationService, RANGE_PARAMETER_1, RANGE_PARAMETER_2
from tvb.core.services.project_service import ProjectService
from tvb.core.utils import url2path
from tvb.storage.storage_interface import StorageInterface
from tvb.storage.h5.utils import string2bool
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.common import InvalidFormValues
from tvb.interfaces.web.controllers.decorators import expose_fragment, handle_error, check_user, expose_json, \
    using_template
from tvb.interfaces.web.controllers.decorators import expose_page, settings, context_selected, expose_numpy_array
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorController
from tvb.interfaces.web.entities.context_selected_adapter import SelectedAdapterContext
from tvb.adapters.creators.local_connectivity_creator import LocalConnectivityCreatorModel, KEY_LCONN

KEY_CONTENT = ABCDisplayer.KEY_CONTENT
FILTER_FIELDS = "fields"
FILTER_TYPE = "type"
FILTER_VALUES = "values"
FILTER_OPERATIONS = "operations"
KEY_CONTROLLS = "controlPage"

MAXIMUM_DATA_TYPES_DISPLAYED = 50
KEY_WARNING = "warning"
WARNING_OVERFLOW = "Too many entities in storage; some of them were not returned, to avoid overcrowding. " \
                   "Use filters, to make the list small enough to fit in here!"


@traced
class FlowController(BaseController):
    """
    This class takes care of executing steps in projects.
    """

    def __init__(self):
        BaseController.__init__(self)
        self.context = SelectedAdapterContext()
        self.operation_services = OperationService()
        self.simulator_controller = SimulatorController()
        self.enum_members = SubformEnum.get_enum_members()

        analyze_category, groups = self.algorithm_service.get_analyze_groups()
        adapters_list = []
        for adapter_group in groups:

            if len(adapter_group.children) > 1:
                ids = [str(child.id) for child in adapter_group.children]
                ids = ','.join(ids)
                adapter_link = self.build_path('/flow/show_group_of_algorithms/' + str(analyze_category.id) + "/" + ids)
            else:
                adapter_link = self.get_url_adapter(analyze_category.id, adapter_group.children[0].id)

            adapters_list.append({common.KEY_TITLE: adapter_group.name,
                                  'link': adapter_link,
                                  'description': adapter_group.description,
                                  'subsection': adapter_group.children[0].subsection_name})
        self.analyze_adapters = adapters_list

    @expose_page
    @settings
    @context_selected
    def step_analyzers(self):
        """
        Choose exact action/adapter for current step.
        """
        try:
            analyze_category, groups = self.algorithm_service.get_analyze_groups()
            step_name = analyze_category.displayname.lower()
            template_specification = dict(mainContent="header_menu", section_name=step_name, controlPage=None,
                                          title="Select an analyzer", displayControl=False)
            template_specification[common.KEY_SUBMENU_LIST] = self.analyze_adapters
            return self.fill_default_attributes(template_specification)

        except ValueError:
            message = 'Could not load analyzers!'
            common.set_warning_message(message)
            self.logger.warning(message)
            self.redirect('/tvb')

    @expose_page
    @settings
    @context_selected
    def step_connectivity(self):
        """
        Display menu for Connectivity Footer tab.
        """
        template_specification = dict(mainContent="header_menu", section_name='connectivity', controlPage=None,
                                      title="Select an algorithm", displayControl=False, subsection_name='step',
                                      submenu_list=self.connectivity_submenu)
        common.add2session(KEY_LCONN, LocalConnectivityCreatorModel)
        return self.fill_default_attributes(template_specification)

    @staticmethod
    def _compute_back_link(back_indicator, project):
        """
        Based on a simple indicator, compute URL for anchor BACK.
        """
        if back_indicator is None:
            # This applies to Connectivity and other visualizers when RELAUNCH button is used from Operation page.
            back_page_link = None
        elif back_indicator == 'burst':
            back_page_link = "/burst"
        elif back_indicator == 'operations':
            back_page_link = '/project/viewoperations/' + str(project.id)
        else:
            back_page_link = '/project/editstructure/' + str(project.id)
        return BaseController.build_path(back_page_link)

    @expose_page
    @settings
    @context_selected
    def show_group_of_algorithms(self, step_key, algorithm_ids):

        project = common.get_current_project()
        category = self.algorithm_service.get_category_by_id(step_key)
        algorithms = []
        for i in algorithm_ids.split(','):
            algorithm_id = int(i)
            algorithm = self.algorithm_service.get_algorithm_by_identifier(algorithm_id)
            algorithm.link = self.get_url_adapter(step_key, algorithm_id)
            adapter_instance = self.algorithm_service.prepare_adapter(algorithm)
            adapter_form = self.algorithm_service.prepare_adapter_form(adapter_instance=adapter_instance,
                                                                       project_id=project.id)
            algorithm.form = self.render_adapter_form(adapter_form)
            algorithms.append(algorithm)

        template_specification = dict(mainContent="flow/algorithms_list", algorithms=algorithms,
                                      title="Select an algorithm", section_name=category.displayname.lower())
        self._populate_section(algorithms[0], template_specification)
        self.fill_default_attributes(template_specification, algorithms[0].group_name)
        return template_specification

    @expose_page
    @settings
    @context_selected
    def prepare_group_launch(self, group_gid, step_key, algorithm_id, **data):
        """
        Receives as input a group gid and an algorithm given by category and id, along
        with data that gives the name of the required input parameter for the algorithm.
        Having these generate a range of GID's for all the DataTypes in the group and
        launch a new operation group.
        """
        dt_group = self.project_service.get_datatypegroup_by_gid(group_gid)
        datatypes = self.project_service.get_datatypes_from_datatype_group(dt_group.id)
        range_param_name = data.pop('range_param_name')
        data[RANGE_PARAMETER_1] = range_param_name
        data[range_param_name] = ','.join(dt.gid for dt in datatypes)
        self.operation_services.group_operation_launch(common.get_logged_user().id, common.get_current_project(),
                                                       int(algorithm_id), int(step_key), **data)
        redirect_url = self._compute_back_link('operations', common.get_current_project())
        raise cherrypy.HTTPRedirect(redirect_url)

    @expose_page
    @settings
    @context_selected
    def default(self, step_key, adapter_key, cancel=False, back_page=None, **data):
        """
        Render a specific adapter.
        'data' are arguments for POST
        """
        project = common.get_current_project()
        algorithm = self.algorithm_service.get_algorithm_by_identifier(adapter_key)
        back_page_link = self._compute_back_link(back_page, project)

        if algorithm is None:
            self.redirect("/tvb?error=True")

        if cherrypy.request.method == 'POST' and cancel:
            raise cherrypy.HTTPRedirect(back_page_link)

        submit_link = self.get_url_adapter(step_key, adapter_key, back_page)
        is_burst = back_page not in ['operations', 'data']
        if cherrypy.request.method == 'POST':
            data[common.KEY_ADAPTER] = adapter_key
            template_specification = self.execute_post(project.id, submit_link, step_key, algorithm, **data)
            self._populate_section(algorithm, template_specification, is_burst)
        else:
            template_specification = self.get_template_for_adapter(project.id, step_key, algorithm,
                                                                   submit_link, is_burst=is_burst)
        if template_specification is None:
            self.redirect('/tvb')

        if KEY_CONTROLLS not in template_specification:
            template_specification[KEY_CONTROLLS] = None
        if common.KEY_SUBMIT_LINK not in template_specification:
            template_specification[common.KEY_SUBMIT_LINK] = submit_link
        if KEY_CONTENT not in template_specification:
            template_specification[KEY_CONTENT] = "flow/full_adapter_interface"
            template_specification[common.KEY_DISPLAY_MENU] = False
        else:
            template_specification[common.KEY_DISPLAY_MENU] = True
            template_specification[common.KEY_BACK_PAGE] = back_page_link

        template_specification[common.KEY_ADAPTER] = adapter_key
        template_specification[ABCDisplayer.KEY_IS_ADAPTER] = True
        self.fill_default_attributes(template_specification, algorithm.displayname)
        return template_specification

    @expose_fragment('form_fields/options_field')
    @settings
    @context_selected
    def get_filtered_datatypes(self, dt_module, dt_class, filters, has_all_option, has_none_option):
        """
        Given the name from the input tree, the dataType required and a number of
        filters, return the available dataType that satisfy the conditions imposed.
        """
        index_class = getattr(sys.modules[dt_module], dt_class)()
        filters_dict = json.loads(filters)

        for idx in range(len(filters_dict['fields'])):
            if filters_dict['values'][idx] in ['True', 'False']:
                filters_dict['values'][idx] = string2bool(filters_dict['values'][idx])

        filter = FilterChain(fields=filters_dict['fields'], operations=filters_dict['operations'],
                             values=filters_dict['values'])
        project = common.get_current_project()

        data_type_gid_attr = DataTypeGidAttr(linked_datatype=REGISTRY.get_datatype_for_index(index_class))
        data_type_gid_attr.required = not string2bool(has_none_option)

        select_field = TraitDataTypeSelectField(data_type_gid_attr, conditions=filter,
                                                has_all_option=string2bool(has_all_option))
        self.algorithm_service.fill_selectfield_with_datatypes(select_field, project.id)

        return {'options': select_field.options()}

    def execute_post(self, project_id, submit_url, step_key, algorithm, **data):
        """ Execute HTTP POST on a generic step."""
        errors = None
        adapter_instance = ABCAdapter.build_adapter(algorithm)
        user = common.get_logged_user()
        try:
            form = self.algorithm_service.fill_adapter_form(adapter_instance, data, project_id, user)
            if form.validate():
                try:
                    view_model = form.get_view_model()()
                    form.fill_trait(view_model)
                    self.context.add_view_model_to_session(view_model)
                except NotImplementedError:
                    self.logger.exception("Form and/or ViewModel not fully implemented for " + str(form))
                    raise InvalidFormValues("Invalid form inputs! Could not find a model for this form!",
                                            error_dict=form.get_errors_dict())
            else:
                raise InvalidFormValues("Invalid form inputs! Could not fill algorithm from the given inputs!",
                                        error_dict=form.get_errors_dict())

            adapter_instance.submit_form(form)

            if not self.operation_services.fits_max_operation_size(adapter_instance, view_model, project_id):
                common.set_error_message(self.MAX_SIZE_ERROR_MSG)
                return {}

            if issubclass(type(adapter_instance), ABCDisplayer):
                adapter_instance.current_project_id = project_id
                adapter_instance.user_id = user.id
                result = adapter_instance.launch(view_model)
                if isinstance(result, dict):
                    return result
                else:
                    common.set_error_message("Invalid result returned from Displayer! Dictionary is expected!")
                return {}

            self.operation_services.fire_operation(adapter_instance, user, project_id, view_model=view_model)
            common.set_important_message("Launched an operation.")

        except formencode.Invalid as excep:
            errors = excep.unpack_errors()
            common.set_error_message("Invalid form inputs")
            self.logger.warning("Invalid form inputs %s" % errors)
        except (OperationException, LaunchException, TraitValueError) as excep1:
            self.logger.exception("Error while executing a Launch procedure:" + excep1.message)
            common.set_error_message(excep1.message)
        except InvalidFormValues as excep2:
            message, errors = excep2.display_full_errors()
            common.set_error_message(message)
            self.logger.warning("%s \n %s" % (message, errors))

        template_specification = self.get_template_for_adapter(project_id, step_key, algorithm, submit_url)
        if (errors is not None) and (template_specification is not None):
            template_specification[common.KEY_ERRORS] = errors
        return template_specification

    def get_template_for_adapter(self, project_id, step_key, stored_adapter, submit_url, is_burst=True,
                                 is_callout=False):
        """ Get Input HTML Interface template or a given adapter """
        try:
            group = None
            category = self.algorithm_service.get_category_by_id(step_key)
            title = "Fill parameters for step " + category.displayname.lower()
            if group:
                title = title + " - " + group.displayname

            adapter_instance = self.algorithm_service.prepare_adapter(stored_adapter)
            user = common.get_logged_user()
            adapter_form = self.algorithm_service.prepare_adapter_form(adapter_instance=adapter_instance,
                                                                       project_id=project_id, user=user)
            vm = self.context.get_view_model_from_session()
            if vm and type(vm) == adapter_form.get_view_model():
                adapter_form.fill_from_trait(vm)
            else:
                self.context.clean_from_session()
            template_specification = dict(submitLink=submit_url,
                                          adapter_form=self.render_adapter_form(adapter_form, is_callout=is_callout),
                                          title=title)

            self._populate_section(stored_adapter, template_specification, is_burst)
            return template_specification
        except OperationException as oexc:
            self.logger.error("Inconsistent Adapter")
            self.logger.exception(oexc)
            common.set_warning_message('Inconsistent Adapter!  Please review the link (development problem)!')
        return None

    @cherrypy.expose
    @handle_error(redirect=False)
    @check_user
    def readserverstaticfile(self, coded_path):
        """
        Retrieve file from Local storage, having a File System Path.
        """
        try:
            with open(url2path(coded_path), "rb") as f:
                return f.read()
        except Exception as excep:
            self.logger.error("Could not retrieve file from path:" + str(coded_path))
            self.logger.exception(excep)

    def _read_datatype_attribute(self, entity_gid, dataset_name, datatype_kwargs='null', **kwargs):

        self.logger.debug("Starting to read HDF5: " + entity_gid + "/" + dataset_name + "/" + str(kwargs))
        entity_dt = h5.load_from_gid(entity_gid)

        datatype_kwargs = json.loads(datatype_kwargs)
        if datatype_kwargs:
            for key, value in six.iteritems(datatype_kwargs):
                kwargs[key] = load_entity_by_gid(value)

        result = getattr(entity_dt, dataset_name)
        if callable(result):
            if kwargs:
                result = result(**kwargs)
            else:
                result = result()
        return result

    @expose_json
    def invoke_adapter(self, algo_id, method_name, entity_gid, **kwargs):
        algorithm = self.algorithm_service.get_algorithm_by_identifier(algo_id)
        adapter_instance = ABCAdapter.build_adapter(algorithm)
        entity = load_entity_by_gid(entity_gid)
        storage_path = StorageInterface().get_project_folder(entity.parent_operation.project.name,
                                                             str(entity.fk_from_operation))
        adapter_instance.storage_path = storage_path
        method = getattr(adapter_instance, method_name)
        if kwargs:
            return method(entity_gid, **kwargs)
        return method(entity_gid)

    def _read_from_h5(self, entity_gid, method_name, datatype_kwargs='null', **kwargs):
        self.logger.debug("Starting to read HDF5: " + entity_gid + "/" + method_name + "/" + str(kwargs))

        datatype_kwargs = json.loads(datatype_kwargs)
        if datatype_kwargs:
            for key, value in six.iteritems(datatype_kwargs):
                kwargs[key] = load_entity_by_gid(value)

        with h5.h5_file_for_gid(entity_gid) as entity_h5:
            result = getattr(entity_h5, method_name)
            if kwargs:
                result = result(**kwargs)
            else:
                result = result()

        return result

    @expose_json
    def read_from_h5_file(self, entity_gid, method_name, flatten=False, datatype_kwargs='null', **kwargs):
        result = self._read_from_h5(entity_gid, method_name, datatype_kwargs, **kwargs)
        return self._prepare_result(result, flatten)

    @expose_json
    def read_datatype_attribute(self, entity_gid, dataset_name, flatten=False, datatype_kwargs='null', **kwargs):
        """
        Retrieve from a given DataType a property or a method result.

        :returns: JSON representation of the attribute.
        :param entity_gid: GID for DataType entity
        :param dataset_name: name of the dataType property /method
        :param flatten: result should be flatten before return (use with WebGL data mainly e.g vertices/triangles)
            Ignored if the attribute is not an ndarray
        :param datatype_kwargs: if passed, will contain a dictionary of type {'name' : 'gid'}, and for each such
            pair, a load_entity will be performed and kwargs will be updated to contain the result
        :param kwargs: extra parameters to be passed when dataset_name is method.

        """
        result = self._read_datatype_attribute(entity_gid, dataset_name, datatype_kwargs, **kwargs)
        return self._prepare_result(result, flatten)

    def _prepare_result(self, result, flatten):
        if isinstance(result, numpy.ndarray):
            # for ndarrays honor the flatten kwarg and convert to lists as ndarrs are not json-able
            if flatten is True or flatten == "True":
                result = result.flatten()
            return result.tolist()
        else:
            return result

    @expose_numpy_array
    def read_binary_datatype_attribute(self, entity_gid, method_name, datatype_kwargs='null', **kwargs):
        return self._read_from_h5(entity_gid, method_name, datatype_kwargs, **kwargs)

    @expose_fragment("flow/genericAdapterFormFields")
    def get_simple_adapter_interface(self, algorithm_id, parent_div='', is_uploader=False):
        """
        AJAX exposed method. Will return only the interface for a adapter, to
        be used when tabs are needed.
        """
        curent_project = common.get_current_project()
        is_uploader = string2bool(is_uploader)
        template_specification = self.get_adapter_template(curent_project.id, algorithm_id, is_uploader)
        template_specification[common.KEY_PARENT_DIV] = parent_div
        return self.fill_default_attributes(template_specification)

    @expose_fragment("flow/full_adapter_interface")
    def getadapterinterface(self, project_id, algorithm_id, back_page=None):
        """
        AJAX exposed method. Will return only a piece of a page,
        to be integrated as part in another page.
        """
        template_specification = self.get_adapter_template(project_id, algorithm_id, False, back_page, is_callout=True)
        template_specification["isCallout"] = True
        return self.fill_default_attributes(template_specification)

    def get_adapter_template(self, project_id, algorithm_id, is_upload=False, back_page=None, is_callout=False):
        """
        Get the template for an adapter based on the algo group id.
        """
        if not (project_id and int(project_id) and (algorithm_id is not None) and int(algorithm_id)):
            return ""

        algorithm = self.algorithm_service.get_algorithm_by_identifier(algorithm_id)
        if is_upload:
            submit_link = BaseController.build_path("/project/launchloader/" + str(project_id) + "/" + str(algorithm_id))
        else:
            submit_link = self.get_url_adapter(algorithm.fk_category, algorithm.id, back_page)

        template_specification = self.get_template_for_adapter(project_id, algorithm.fk_category, algorithm,
                                                               submit_link, is_callout=is_callout)
        if template_specification is None:
            return ""
        template_specification[common.KEY_DISPLAY_MENU] = not is_upload
        return template_specification

    @cherrypy.expose
    @handle_error(redirect=True)
    @context_selected
    def reloadoperation(self, operation_id, **_):
        """Redirect to Operation Input selection page, with input data already selected."""
        operation = OperationService.load_operation(operation_id)
        # Reload previous parameters in session
        adapter_instance = ABCAdapter.build_adapter(operation.algorithm)
        view_model = adapter_instance.load_view_model(operation)
        self.context.add_view_model_to_session(view_model)
        # Display the inputs tree for the current op
        category_id = operation.algorithm.fk_category
        algo_id = operation.fk_from_algo
        self.redirect("/flow/" + str(category_id) + "/" + str(algo_id))

    @cherrypy.expose
    @handle_error(redirect=True)
    @context_selected
    def reload_burst_operation(self, operation_id, is_group, **_):
        """
        Find out from which burst was this operation launched. Set that burst as the selected one and
        redirect to the burst page.
        """
        is_group = int(is_group)
        if not is_group:
            operation = OperationService.load_operation(int(operation_id))
        else:
            op_group = ProjectService.get_operation_group_by_id(operation_id)
            first_op = ProjectService.get_operations_in_group(op_group)[0]
            operation = OperationService.load_operation(int(first_op.id))
        self.simulator_controller.copy_simulator_configuration(operation.burst.id)

        self.redirect("/burst/")

    @expose_json
    def cancel_or_remove_operation(self, operation_id, is_group, remove_after_stop=False):
        """
        Stop the operation given by operation_id. If is_group is true stop all the
        operations from that group.
        """
        operation_id = int(operation_id)
        is_group = int(is_group) != 0
        if isinstance(remove_after_stop, str):
            remove_after_stop = bool(remove_after_stop)
        return self.simulator_controller.cancel_or_remove_operation(operation_id, is_group, remove_after_stop)

    def fill_default_attributes(self, template_dictionary, title='-'):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        if common.KEY_TITLE not in template_dictionary:
            template_dictionary[common.KEY_TITLE] = title

        if common.KEY_PARENT_DIV not in template_dictionary:
            template_dictionary[common.KEY_PARENT_DIV] = ''
        if common.KEY_PARAMETERS_CONFIG not in template_dictionary:
            template_dictionary[common.KEY_PARAMETERS_CONFIG] = False

        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'flow/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary

    NEW_SELECTION_NAME = 'New selection'

    def _get_available_selections(self, datatype_gid):
        """
        selection retrieval common to selection component and connectivity selection
        """
        curent_project = common.get_current_project()
        selections = self.algorithm_service.get_selections_for_project(curent_project.id, datatype_gid)
        names, sel_values = [], []

        for selection in selections:
            names.append(selection.ui_name)
            sel_values.append(selection.selected_nodes)

        return names, sel_values

    @expose_fragment('visualizers/commons/channel_selector_opts')
    def get_available_selections(self, **data):
        sel_names, sel_values = self._get_available_selections(data['datatype_gid'])
        return dict(namedSelections=list(zip(sel_names, sel_values)))

    @expose_json
    def store_measure_points_selection(self, ui_name, **data):
        """
        Save a MeasurePoints selection (new or update existing entity).
        """
        if ui_name and ui_name != self.NEW_SELECTION_NAME:
            sel_project_id = common.get_current_project().id
            # client sends integers as strings:
            selection = json.dumps([int(s) for s in json.loads(data['selection'])])
            datatype_gid = data['datatype_gid']
            self.algorithm_service.save_measure_points_selection(ui_name, selection, datatype_gid, sel_project_id)
            return [True, 'Selection saved successfully.']

        else:
            error_msg = self.NEW_SELECTION_NAME + " or empty name are not  valid as selection names."
            return [False, error_msg]

    @expose_fragment("visualizers/pse_discrete/inserting_new_threshold_spec_bar")
    def create_row_of_specs(self, count):
        return dict(id_increment_count=count)

    @expose_json
    def store_pse_filter(self, config_name, **data):
        # this will need to be updated in such a way that the expose_json actually gets used
        ## also this is going to be changed to be storing through the flow service and dao. Stay updated
        try:
            ##this is to check whether there is already an entry in the
            for i, (name, Val) in enumerate(self.PSE_names_list):
                if name == config_name:
                    self.PSE_names_list[i] = (config_name, (
                            data['threshold_value'] + "," + data['threshold_type'] + "," + data[
                        'not_presence']))  # replace the previous occurence of the config name, and carry on.
                    self.get_pse_filters()
                    return [True, 'Selected Text stored, and selection updated']
            self.PSE_names_list.append(
                (config_name, (data['threshold_value'] + "," + data['threshold_type'] + "," + data['not_presence'])))
        except AttributeError:
            self.PSE_names_list = [
                (config_name, (data['threshold_value'] + "," + data['threshold_type'] + "," + data['not_presence']))]

        self.get_pse_filters()
        return [True, 'Selected Text stored, and selection updated']

    @expose_fragment("visualizers/commons/channel_selector_opts")
    def get_pse_filters(self):
        try:
            return dict(namedSelections=self.PSE_names_list)
        except AttributeError:
            return dict(namedSelections=[])  # this will give us back atleast the New Selection option in the select
        except:
            raise

    @expose_json
    def store_exploration_section(self, val_range, step, dt_group_guid):
        """
        Launching method for further simulations.
        """
        range_list = [float(num) for num in val_range.split(",")]
        step_list = [float(num) for num in step.split(",")]

        datatype_group_ob = self.project_service.get_datatypegroup_by_gid(dt_group_guid)
        operation_grp = datatype_group_ob.parent_operation_group
        operation_obj = OperationService.load_operation(datatype_group_ob.fk_from_operation)
        parameters = {}

        range1name, range1_dict = json.loads(operation_grp.range1)
        range2name, range2_dict = json.loads(operation_grp.range2)
        parameters[RANGE_PARAMETER_1] = range1name
        parameters[RANGE_PARAMETER_2] = range2name

        # change the existing simulator parameters to be min max step types
        range1_dict = {constants.ATT_MINVALUE: range_list[0],
                       constants.ATT_MAXVALUE: range_list[1],
                       constants.ATT_STEP: step_list[0]}
        range2_dict = {constants.ATT_MINVALUE: range_list[2],
                       constants.ATT_MAXVALUE: range_list[3],
                       constants.ATT_STEP: step_list[1]}
        parameters[range1name] = json.dumps(range1_dict)  # this is for the x axis parameter
        parameters[range2name] = json.dumps(range2_dict)  # this is for the y axis parameter

        OperationService().group_operation_launch(common.get_logged_user().id, common.get_current_project(),
                                                  operation_obj.algorithm.id, operation_obj.algorithm.fk_category,
                                                  **parameters)

        return [True, 'Stored the exploration material successfully']

    @cherrypy.expose
    @using_template('form_fields/form_field')
    @handle_error(redirect=False)
    @check_user
    def refresh_subform(self, data_name, subform_label, spatial_model_key):
        data_class = TVBEnum.string_to_enum(self.enum_members, data_name).value

        spatial_model = common.get_from_session(spatial_model_key)
        equation_info = spatial_model.get_equation_information()
        setattr(spatial_model, equation_info[subform_label], data_class())

        adapter_form = get_form_for_equation(data_class)()

        return {'adapter_form': adapter_form}
