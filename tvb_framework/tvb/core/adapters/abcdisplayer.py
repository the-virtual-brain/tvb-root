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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""
import importlib
import json
import os
import numpy
from abc import ABCMeta
from threading import Lock
from uuid import UUID
from six import add_metaclass
from tvb.core.adapters.abcadapter import AdapterLaunchModeEnum, ABCAdapter
from tvb.core.adapters.exceptions import LaunchException
from tvb.core.neocom import h5

LOCK_CREATE_FIGURE = Lock()


class URLGenerator(object):
    FLOW = 'flow'
    INVOKE_ADAPTER = 'invoke_adapter'
    H5_FILE = 'read_from_h5_file'
    DATATYPE_ATTRIBUTE = 'read_datatype_attribute'
    BINARY_DATATYPE_ATTRIBUTE = 'read_binary_datatype_attribute'

    @staticmethod
    def build_base_h5_url(entity_gid):
        url_regex = '/{}/{}/{}'
        return url_regex.format(URLGenerator.FLOW, URLGenerator.H5_FILE, entity_gid)

    @staticmethod
    def build_url(adapter_id, method_name, entity_gid, parameter=None):
        if isinstance(entity_gid, UUID):
            entity_gid = entity_gid.hex
        url_regex = '/{}/{}/{}/{}/{}'
        url = url_regex.format(URLGenerator.FLOW, URLGenerator.INVOKE_ADAPTER, adapter_id, method_name, entity_gid)

        if parameter is not None:
            url += "?" + str(parameter)

        return url

    @staticmethod
    def build_h5_url(entity_gid, method_name, flatten=False, datatype_kwargs=None, parameter=None):
        json_kwargs = json.dumps(datatype_kwargs)
        if isinstance(entity_gid, UUID):
            entity_gid = entity_gid.hex

        url_regex = '/{}/{}/{}/{}/{}/{}'
        url = url_regex.format(URLGenerator.FLOW, URLGenerator.H5_FILE, entity_gid, method_name, flatten, json_kwargs)

        if parameter is not None:
            url += "?" + str(parameter)

        return url

    @staticmethod
    def paths2url(datatype_gid, attribute_name, flatten=False, parameter=None):
        """
        Prepare a File System Path for passing into an URL.
        """
        if isinstance(datatype_gid, UUID):
            datatype_gid = datatype_gid.hex
        url_regex = '/{}/{}/{}/{}/{}'
        url = url_regex.format(URLGenerator.FLOW, URLGenerator.DATATYPE_ATTRIBUTE,
                               datatype_gid, attribute_name, flatten)
        if parameter is not None:
            url += "?" + str(parameter)
        return url

    @staticmethod
    def build_binary_datatype_attribute_url(datatype_gid, attribute_name, parameter=None):
        if isinstance(datatype_gid, UUID):
            datatype_gid = datatype_gid.hex
        url_regex = '/{}/{}/{}/{}'
        url = url_regex.format(URLGenerator.FLOW, URLGenerator.BINARY_DATATYPE_ATTRIBUTE,
                               datatype_gid, attribute_name)
        if parameter is not None:
            url += "?" + str(parameter)
        return url


@add_metaclass(ABCMeta)
class ABCDisplayer(ABCAdapter, metaclass=ABCMeta):
    """
    Abstract class, for marking Adapters used for UI display only.
    """
    KEY_CONTENT = "mainContent"
    KEY_IS_ADAPTER = "isAdapter"
    VISUALIZERS_ROOT = ''
    launch_mode = AdapterLaunchModeEnum.SYNC_SAME_MEM

    def get_output(self):
        return []

    def _prelaunch(self, operation, view_model, available_disk_space=0):
        """
        Shortcut in case of visualization calls.
        """
        self.current_project_id = operation.project.id
        self.user_id = operation.fk_launched_by
        return self.launch(view_model=view_model), 0

    def get_required_disk_size(self, view_model):
        """
        Visualizers should no occupy any additional disk space.
        """
        return 0

    def build_display_result(self, template, parameters, pages=None):
        """
        Helper method for building the result of the ABCDisplayer.
        :param template : relative path towards the HTML template to display
        :param parameters : dictionary with parameters for "template"
        :param pages : dictionary of pages to be used with <xi:include>
        """
        module_ref = importlib.import_module(self.VISUALIZERS_ROOT)
        relative_path = os.path.basename(os.path.dirname(module_ref.__file__))
        jinja_separator = '/'

        template = relative_path + jinja_separator + template
        if pages:
            for key, value in pages.items():
                if value is not None:
                    value = relative_path + jinja_separator + value
                parameters[key] = value
        parameters[self.KEY_CONTENT] = template
        parameters[self.KEY_IS_ADAPTER] = True

        return parameters

    @staticmethod
    def get_one_dimensional_list(list_of_elements, expected_size, error_msg):
        """
        Used for obtaining a list of 'expected_size' number of elements from the
        list 'list_of_elements'. If the list 'list_of_elements' doesn't have 
        sufficient elements then an exception will be thrown.

        list_of_elements - a list of one or two dimensions
        expected_size - the number of elements that should have the returned list
        error_msg - the message that will be used for the thrown exception.
        """
        if len(list_of_elements) > 0 and isinstance(list_of_elements[0], list):
            if len(list_of_elements[0]) < expected_size:
                raise LaunchException(error_msg)
            return list_of_elements[0][:expected_size]
        else:
            if len(list_of_elements) < expected_size:
                raise LaunchException(error_msg)
            return list_of_elements[:expected_size]

    @staticmethod
    def dump_with_precision(xs, precision=3):
        """
        Dump a list of numbers into a string, each at the specified precision.
        """
        format_str = "%0." + str(precision) + "g"
        return "[" + ",".join(format_str % s for s in xs) + "]"

    @staticmethod
    def handle_infinite_values(data):
        """
        Replace positive infinite values with the biggest float number available in numpy, and negative infinite
        values with the smallest float number available in numpy.
        """
        float_max = numpy.finfo(numpy.float64).max
        data[data == numpy.PINF] = float_max

        float_min = numpy.finfo(numpy.float64).min
        data[data == numpy.NINF] = float_min

        return data

    @staticmethod
    def prepare_shell_surface_params(shell_surface, surface_url_generator):
        """
        Prepares urls that are necessary for a shell surface.
        """
        if shell_surface:
            shell_h5 = h5.h5_file_for_index(shell_surface)
            vertices, normals, lines, triangles, _ = surface_url_generator.get_urls_for_rendering(shell_h5)
            shell_object = json.dumps([vertices, normals, triangles, lines])
            shell_h5.close()
            return shell_object
        return None
