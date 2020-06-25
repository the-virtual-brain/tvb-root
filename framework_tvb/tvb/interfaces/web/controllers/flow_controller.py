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

"""
Steps which a user needs to execute for achieving a 
given action are described here.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import copy
import json
import os
from http import HTTPStatus
import cherrypy
import formencode
import numpy
import six
from cherrypy.lib.static import serve_file
from tvb.core.adapters import constants
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcdisplayer import ABCDisplayer
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.adapters import constants
from tvb.core.entities.model.model_burst import BurstConfiguration
from tvb.core.entities.model.model_operation import OperationPossibleStatus
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.operation_hpc_launcher import UPDATE_STATUS_KEY
from tvb.core.services.burst_service import BurstService
from tvb.core.services.encryption_handler import EncryptionHandler
from tvb.core.services.exceptions import OperationException
from tvb.core.services.operation_service import OperationService, RANGE_PARAMETER_1, RANGE_PARAMETER_2
from tvb.core.services.project_service import ProjectService
from tvb.core.utils import url2path, parse_json_parameters, string2date, string2bool
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.common import InvalidFormValues
from tvb.interfaces.web.controllers.decorators import expose_page, settings, context_selected, expose_numpy_array
from tvb.interfaces.web.controllers.decorators import expose_fragment, handle_error, check_user, expose_json, \
    expose_endpoint
from tvb.interfaces.web.entities.context_selected_adapter import SelectedAdapterContext

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
        self.files_helper = FilesHelper()
        self.operation_services = OperationService()

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
            adapters_list = []
            for adapter_group in groups:

                if len(adapter_group.children) > 1:
                    ids = [str(child.id) for child in adapter_group.children]
                    ids = ','.join(ids)
                    adapter_link = '/flow/show_group_of_algorithms/' + str(analyze_category.id) + "/" + ids
                else:
                    adapter_link = self.get_url_adapter(analyze_category.id, adapter_group.children[0].id)

                adapters_list.append({common.KEY_TITLE: adapter_group.name,
                                      'link': adapter_link,
                                      'description': adapter_group.description,
                                      'subsection': adapter_group.children[0].subsection_name})
            self.analyze_adapters = adapters_list
            template_specification[common.KEY_SUBMENU_LIST] = adapters_list
            return self.fill_default_attributes(template_specification)

        except ValueError:
            message = 'Could not load analyzers!'
            common.set_warning_message(message)
            self.logger.warning(message)
            raise cherrypy.HTTPRedirect('/tvb')

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
        return back_page_link

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
            adapter_form = self.algorithm_service.prepare_adapter_form(adapter_instance, project.id)
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
        prj_service = ProjectService()
        dt_group = prj_service.get_datatypegroup_by_gid(group_gid)
        datatypes = prj_service.get_datatypes_from_datatype_group(dt_group.id)
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
    def default(self, step_key, adapter_key, cancel=False, back_page=None, not_reset=False, **data):
        """
        Render a specific adapter.
        'data' are arguments for POST
        """
        project = common.get_current_project()
        algorithm = self.algorithm_service.get_algorithm_by_identifier(adapter_key)
        back_page_link = self._compute_back_link(back_page, project)

        if algorithm is None:
            raise cherrypy.HTTPRedirect("/tvb?error=True")

        if cherrypy.request.method == 'POST' and cancel:
            raise cherrypy.HTTPRedirect(back_page_link)

        submit_link = self.get_url_adapter(step_key, adapter_key, back_page)
        is_burst = back_page not in ['operations', 'data']
        if cherrypy.request.method == 'POST':
            data[common.KEY_ADAPTER] = adapter_key
            template_specification = self.execute_post(project.id, submit_link, step_key, algorithm, **data)
            self._populate_section(algorithm, template_specification, is_burst)
        else:
            if (('Referer' not in cherrypy.request.headers or
                ('Referer' in cherrypy.request.headers and 'step' not in cherrypy.request.headers['Referer']))
                    and 'View' in algorithm.algorithm_category.displayname):
                # Avoid reset in case of Visualizers, as a supplementary GET
                not_reset = True
            template_specification = self.get_template_for_adapter(project.id, step_key, algorithm,
                                                                   submit_link, not not_reset, is_burst=is_burst)
        if template_specification is None:
            raise cherrypy.HTTPRedirect('/tvb')

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

    @expose_fragment("flow/reduce_dimension_select")
    def gettemplatefordimensionselect(self, entity_gid=None, select_name="", reset_session='False',
                                      parameters_prefix="dimensions", required_dimension=1,
                                      expected_shape="", operations=""):
        """
        Returns the HTML which contains the selects components which allows the user
        to reduce the dimension of a multi-dimensional array.

        We try to obtain the aggregation_functions from the entity, which is a list of lists.
        For each dimension should be a list with the supported aggregation functions. We
        create a DICT for each of those lists. The key will be the name of the function and
        the value will be its label.

        entity_gid 
            the GID of the entity for which is displayed the component
        
        select_name
            the name of the parent select. The select in which
            is displayed the entity with the given GID
  
        parameters_prefix 
            a string which will be used for computing the names of the component

        required_dimension
            the expected dimension for the resulted array

        expected_shape and operations
            used for applying conditions on the resulted array
            e.g.: If the resulted array is a 3D array and we want that the length of the second
            dimension to be smaller then 512 then the expected_shape and operations should be:
            ``expected_shape=x,512,x`` and ``operations='x,&lt;,x``
        """
        template_params = {"select_name": "",
                           "data": [],
                           "parameters_prefix": parameters_prefix,
                           "array_shape": "",
                           "required_dimension": required_dimension,
                           "currentDim": "",
                           "required_dim_msg": "",
                           "expected_shape": expected_shape,
                           "operations": operations}

        # if reload => populate the selected values
        session_dict = self.context.get_current_default()
        dimensions = {1: [0], 3: [0]}
        selected_agg_functions = {}
        if not string2bool(str(reset_session)) and session_dict is not None:
            starts_with_str = select_name + "_" + parameters_prefix + "_"
            ui_sel_items = dict((k, v) for k, v in session_dict.items() if k.startswith(starts_with_str))
            from tvb.datatypes.arrays import MappedArray

            dimensions, selected_agg_functions, required_dimension, _ = MappedArray().parse_selected_items(ui_sel_items)
        template_params["selected_items"] = dimensions
        template_params["selected_functions"] = selected_agg_functions

        aggregation_functions = []
        default_agg_functions = self.accepted__aggregation_functions()
        labels_set = ["Time", "Channel", "Line"]
        if entity_gid is not None:
            actual_entity = ABCAdapter.load_entity_by_gid(entity_gid)
            if hasattr(actual_entity, 'shape'):
                array_shape = actual_entity.shape
                new_shape, current_dim = self._compute_current_dimension(list(array_shape), dimensions,
                                                                         selected_agg_functions)
                if required_dimension is not None and current_dim != int(required_dimension):
                    template_params["required_dim_msg"] = "Please select a " + str(required_dimension) + "D array"
                if not current_dim:
                    template_params["currentDim"] = "1 element"
                else:
                    template_params["currentDim"] = str(current_dim) + "D array"
                template_params["array_shape"] = json.dumps(new_shape)
                if hasattr(actual_entity, 'dimensions_labels') and actual_entity.dimensions_labels is not None:
                    labels_set = actual_entity.dimensions_labels
                    # make sure there exists labels for each dimension
                    while len(labels_set) < len(array_shape):
                        labels_set.append("Undefined")
                if (hasattr(actual_entity, 'aggregation_functions') and actual_entity.aggregation_functions is not None
                        and len(actual_entity.aggregation_functions) == len(array_shape)):
                    # will be a list of lists of aggregation functions
                    defined_functions = actual_entity.aggregation_functions
                    for function in defined_functions:
                        if not len(function):
                            aggregation_functions.append({})
                        else:
                            func_dict = {}
                            for function_key in function:
                                func_dict[function_key] = default_agg_functions[function_key]
                            aggregation_functions.append(func_dict)
                else:
                    for _ in array_shape:
                        aggregation_functions.append(default_agg_functions)
                result = []
                for i, shape in enumerate(array_shape):
                    labels = []
                    values = []
                    for j in range(shape):
                        labels.append(labels_set[i] + " " + str(j))
                        values.append(entity_gid + "_" + str(i) + "_" + str(j))
                    result.append([labels, values, aggregation_functions[i]])
                template_params["select_name"] = select_name
                template_params["data"] = result
                return template_params

        return template_params

    @staticmethod
    def _compute_current_dimension(array_shape, selected_items, selected_functions):
        """
        If the user reloads an operation we have to compute the current dimension of the array
        and also the shape of the array based on his selections
        """
        current_dim = len(array_shape)
        for i in range(len(array_shape)):
            if i in selected_items and len(selected_items[i]) > 0:
                array_shape[i] = len(selected_items[i])
                if len(selected_items[i]) == 1:
                    current_dim -= 1
            if i in selected_functions and selected_functions[i] != 'none':
                array_shape[i] = 1
                if i not in selected_items or len(selected_items[i]) > 1:
                    current_dim -= 1
        return array_shape, current_dim

    @staticmethod
    def accepted__aggregation_functions():
        """
        Returns the list of aggregation functions that may be
        applied on arrays.
        """
        return {"sum": "Sum", "average": "Average"}


    # @expose_fragment("flow/type2component/datatype2select_simple")
    def getfiltereddatatypes(self, name, parent_div, tree_session_key, filters):
        # TODO: fix this use-case
        """
        Given the name from the input tree, the dataType required and a number of
        filters, return the available dataType that satisfy the conditions imposed.
        """
        previous_tree = self.context.get_session_tree_for_key(tree_session_key)
        if previous_tree is None:
            common.set_error_message("Adapter Interface not in session for filtering!")
            raise cherrypy.HTTPRedirect("/tvb?error=True")
        current_node = self._get_node(previous_tree, name)
        if current_node is None:
            raise Exception("Could not find node :" + name)
        datatype = current_node[ABCAdapter.KEY_DATATYPE]

        filters = json.loads(filters)
        availablefilter = json.loads(FilterChain.get_filters_for_type(datatype))
        for i, filter_ in enumerate(filters[FILTER_FIELDS]):
            # Check for filter input of type 'date' as these need to be converted
            if filter_ in availablefilter and availablefilter[filter_][FILTER_TYPE] == 'date':
                try:
                    temp_date = string2date(filters[FILTER_VALUES][i], False)
                    filters[FILTER_VALUES][i] = temp_date
                except ValueError:
                    raise
        # In order for the filter object not to "stack up" on multiple calls to
        # this method, create a deepCopy to work with
        if constants.ELEM_CONDITIONS in current_node:
            new_filter = copy.deepcopy(current_node[constants.ELEM_CONDITIONS])
        else:
            new_filter = FilterChain()
        new_filter.fields.extend(filters[FILTER_FIELDS])
        new_filter.operations.extend(filters[FILTER_OPERATIONS])
        new_filter.values.extend(filters[FILTER_VALUES])
        # Get dataTypes that match the filters from DB then populate with values
        values, total_count = [], 0
        # Create a dictionary that matches what the template expects
        parameters = {ABCAdapter.KEY_NAME: name,
                      ABCAdapter.KEY_FILTERABLE: availablefilter,
                      ABCAdapter.KEY_TYPE: constants.TYPE_SELECT,
                      ABCAdapter.KEY_OPTIONS: values,
                      ABCAdapter.KEY_DATATYPE: datatype}

        if total_count > MAXIMUM_DATA_TYPES_DISPLAYED:
            parameters[KEY_WARNING] = WARNING_OVERFLOW

        if constants.ATT_REQUIRED in current_node:
            parameters[constants.ATT_REQUIRED] = current_node[constants.ATT_REQUIRED]
            if len(values) > 0 and string2bool(str(parameters[constants.ATT_REQUIRED])):
                parameters[ABCAdapter.KEY_DEFAULT] = str(values[-1][ABCAdapter.KEY_VALUE])
        previous_selected = self.context.get_current_default(name)
        if previous_selected in [str(vv['value']) for vv in values]:
            parameters[ABCAdapter.KEY_DEFAULT] = previous_selected

        template_specification = {"inputRow": parameters, "disabled": False,
                                  "parentDivId": parent_div, common.KEY_SESSION_TREE: tree_session_key}
        return self.fill_default_attributes(template_specification)

    def _get_node(self, input_tree, name):
        """
        Given a input tree and a variable name, check to see if any default filters exist.
        """
        for entry in input_tree:
            if (ABCAdapter.KEY_DATATYPE in entry and ABCAdapter.KEY_NAME in entry
                    and str(entry[ABCAdapter.KEY_NAME]) == str(name)):
                return entry
            if entry.get(ABCAdapter.KEY_ATTRIBUTES) is not None:
                in_attr = self._get_node(entry[ABCAdapter.KEY_ATTRIBUTES], name)
                if in_attr is not None:
                    return in_attr
            if entry.get(ABCAdapter.KEY_OPTIONS) is not None:
                in_options = self._get_node(entry[ABCAdapter.KEY_OPTIONS], name)
                if in_options is not None:
                    return in_options
        return None

    def execute_post(self, project_id, submit_url, step_key, algorithm, **data):
        """ Execute HTTP POST on a generic step."""
        errors = None
        adapter_instance = ABCAdapter.build_adapter(algorithm)

        try:
            form = adapter_instance.get_form()(project_id=project_id)
            if 'fill_defaults' in data:
                form.fill_from_post_plus_defaults(data)
            else:
                form.fill_from_post(data)
            view_model = None
            if form.validate():
                try:
                    view_model = form.get_view_model()()
                    form.fill_trait(view_model)
                except NotImplementedError:
                    self.logger.exception("Form and/or ViewModel not fully implemented for " + str(form))
                    raise InvalidFormValues("Invalid form inputs! Could not find a model for this form!",
                                            error_dict=form.get_errors_dict())
            else:
                raise InvalidFormValues("Invalid form inputs! Could not fill algorithm from the given inputs!",
                                        error_dict=form.get_errors_dict())

            adapter_instance.submit_form(form)

            if issubclass(type(adapter_instance), ABCDisplayer):
                adapter_instance.current_project_id = project_id
                adapter_instance.user_id = common.get_logged_user().id
                result = adapter_instance.launch(view_model)
                if isinstance(result, dict):
                    return result
                else:
                    common.set_error_message("Invalid result returned from Displayer! Dictionary is expected!")
                return {}

            result = self.operation_services.fire_operation(adapter_instance, common.get_logged_user(),
                                                            project_id, view_model=view_model)
            # Store input data in session, for informing user of it.
            step = self.algorithm_service.get_category_by_id(step_key)
            if not step.rawinput:
                self.context.add_adapter_to_session(None, None, copy.deepcopy(data))
            if isinstance(result, list):
                result = "Launched %s operations." % len(result)
            common.set_important_message(str(result))

        except formencode.Invalid as excep:
            errors = excep.unpack_errors()
            common.set_error_message("Invalid form inputs")
            self.logger.warning("Invalid form inputs %s" % errors)
        except OperationException as excep1:
            self.logger.exception("Error while executing a Launch procedure:" + excep1.message)
            common.set_error_message(excep1.message)
        except LaunchException as excep3:
            self.logger.exception("Error while executing a Launch procedure:" + excep3.message)
            common.set_error_message(excep3.message)
        except InvalidFormValues as excep2:
            message, errors = excep2.display_full_errors()
            common.set_error_message(message)
            self.logger.warning("%s \n %s" % (message, errors))

        previous_step = self.context.get_current_substep()
        should_reset = previous_step is None or data.get(common.KEY_ADAPTER) != previous_step
        template_specification = self.get_template_for_adapter(project_id, step_key, algorithm,
                                                               submit_url, should_reset)
        if (errors is not None) and (template_specification is not None):
            template_specification[common.KEY_ERRORS] = errors
        template_specification[common.KEY_OPERATION_ID] = adapter_instance.operation_id
        return template_specification

    def get_template_for_adapter(self, project_id, step_key, stored_adapter, submit_url, session_reset=True,
                                 is_burst=True):
        """ Get Input HTML Interface template or a given adapter """
        try:
            if session_reset:
                self.context.clean_from_session()

            group = None
            category = self.algorithm_service.get_category_by_id(step_key)
            title = "Fill parameters for step " + category.displayname.lower()
            if group:
                title = title + " - " + group.displayname

            adapter_instance = self.algorithm_service.prepare_adapter(stored_adapter)

            adapter_form = self.algorithm_service.prepare_adapter_form(adapter_instance, project_id)
            template_specification = dict(submitLink=submit_url, adapter_form=self.render_adapter_form(adapter_form), title=title)

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
        entity = ABCAdapter.load_entity_by_gid(entity_gid)
        entity_dt = h5.load_from_index(entity)

        datatype_kwargs = json.loads(datatype_kwargs)
        if datatype_kwargs:
            for key, value in six.iteritems(datatype_kwargs):
                kwargs[key] = ABCAdapter.load_entity_by_gid(value)

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
        entity = ABCAdapter.load_entity_by_gid(entity_gid)
        storage_path = self.files_helper.get_project_folder(entity.parent_operation.project,
                                                            str(entity.fk_from_operation))
        adapter_instance.storage_path = storage_path
        method = getattr(adapter_instance, method_name)
        if kwargs:
            return method(entity_gid, **kwargs)
        return method(entity_gid)

    @expose_json
    def read_from_h5_file(self, entity_gid, method_name, flatten=False, datatype_kwargs='null', **kwargs):
        self.logger.debug("Starting to read HDF5: " + entity_gid + "/" + method_name + "/" + str(kwargs))
        entity = ABCAdapter.load_entity_by_gid(entity_gid)
        entity_h5 = h5.h5_file_for_index(entity)

        datatype_kwargs = json.loads(datatype_kwargs)
        if datatype_kwargs:
            for key, value in six.iteritems(datatype_kwargs):
                kwargs[key] = ABCAdapter.load_entity_by_gid(value)

        result = getattr(entity_h5, method_name)
        if kwargs:
            result = result(**kwargs)
        else:
            result = result()

        entity_h5.close()
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
    def read_binary_datatype_attribute(self, entity_gid, dataset_name, datatype_kwargs='null', **kwargs):
        return self._read_datatype_attribute(entity_gid, dataset_name, datatype_kwargs, **kwargs)

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
        template_specification = self.get_adapter_template(project_id, algorithm_id, False, back_page)
        template_specification["isCallout"] = True
        return self.fill_default_attributes(template_specification)

    def get_adapter_template(self, project_id, algorithm_id, is_upload=False, back_page=None):
        """
        Get the template for an adapter based on the algo group id.
        """
        if not (project_id and int(project_id) and (algorithm_id is not None) and int(algorithm_id)):
            return ""

        algorithm = self.algorithm_service.get_algorithm_by_identifier(algorithm_id)
        if is_upload:
            submit_link = "/project/launchloader/" + str(project_id) + "/" + str(algorithm_id)
        else:
            submit_link = self.get_url_adapter(algorithm.fk_category, algorithm.id, back_page)

        current_step = self.context.get_current_substep()
        if current_step is None or str(current_step) != str(algorithm_id):
            self.context.clean_from_session()
        template_specification = self.get_template_for_adapter(project_id, algorithm.fk_category, algorithm,
                                                               submit_link, is_upload)
        if template_specification is None:
            return ""
        template_specification[common.KEY_DISPLAY_MENU] = not is_upload
        return template_specification

    @cherrypy.expose
    @handle_error(redirect=True)
    @context_selected
    def reloadoperation(self, operation_id, **_):
        """Redirect to Operation Input selection page, 
        with input data already selected."""
        operation = OperationService.load_operation(operation_id)
        data = parse_json_parameters(operation.parameters)
        self.context.add_adapter_to_session(operation.algorithm, None, data)
        category_id = operation.algorithm.fk_category
        algo_id = operation.fk_from_algo
        raise cherrypy.HTTPRedirect("/flow/" + str(category_id) + "/" + str(algo_id) + "?not_reset=True")

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
        operation.burst.prepare_after_load()
        common.add2session(common.KEY_BURST_CONFIG, operation.burst)
        raise cherrypy.HTTPRedirect("/burst/")

    @expose_json
    def cancel_or_remove_operation(self, operation_id, is_group, remove_after_stop=False):
        """
        Stop the operation given by operation_id. If is_group is true stop all the
        operations from that group.
        """
        operation_id = int(operation_id)
        is_group = int(is_group) != 0
        # Load before we remove, to have its data in memory here
        burst_config = BurstService.get_burst_for_operation_id(operation_id)

        result = OperationService.stop_operation(operation_id, is_group, remove_after_stop)
        if remove_after_stop:
            current_burst = common.get_from_session(common.KEY_BURST_CONFIG)
            if current_burst is not None and burst_config is not None and current_burst.id == burst_config.id:
                common.remove_from_session(common.KEY_BURST_CONFIG)
                common.add2session(common.KEY_BURST_CONFIG, BurstConfiguration(burst_config.project.id))
        return result

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

    ##### Below this point are operations that might be moved to different #####
    ##### controller                                                       #####

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

        datatype_group_ob = ProjectService().get_datatypegroup_by_gid(dt_group_guid)
        operation_grp = datatype_group_ob.parent_operation_group
        operation_obj = OperationService.load_operation(datatype_group_ob.fk_from_operation)
        parameters = json.loads(operation_obj.parameters)

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
                                                  datatype_group_ob, **parameters)

        return [True, 'Stored the exploration material successfully']

    @expose_endpoint
    def update_status(self, simulator_gid, **data):
        if cherrypy.request.method != 'PUT':
            raise cherrypy.HTTPError(HTTPStatus.METHOD_NOT_ALLOWED)

        if UPDATE_STATUS_KEY not in data:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST,
                                     "Invalid request. {} body param is missing.".format(UPDATE_STATUS_KEY))

        new_status = data[UPDATE_STATUS_KEY]
        if new_status not in OperationPossibleStatus:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST,
                                     "Invalid status.")

        burst_config = dao.get_generic_entity(BurstConfiguration, simulator_gid, "simulator_gid")
        if len(burst_config) == 0:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST,
                                     "Invalid simulator gid.")

        OperationService.handle_hpc_status_changed(burst_config[0], new_status)

    @expose_endpoint
    def encryption_config(self, simulator_gid):
        if cherrypy.request.method != 'GET':
            raise cherrypy.HTTPError(HTTPStatus.METHOD_NOT_ALLOWED)

        burst_config = dao.get_generic_entity(BurstConfiguration, simulator_gid, "simulator_gid")
        if len(burst_config) == 0:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST,
                                     "Invalid simulator gid.")

        encryption_handler = EncryptionHandler(simulator_gid)
        file_path = encryption_handler.get_password_file()

        return serve_file(file_path, "application/x-download", "attachment", os.path.basename(file_path))
