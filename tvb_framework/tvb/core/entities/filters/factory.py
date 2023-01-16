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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>    
"""

import json
import six
import tvb.core.entities.model.model_operation as model
from tvb.core.entities.filters.chain import FilterChain


class StaticFiltersFactory(object):
    """
    Factory class to build lists with static used filters through the application.
    """
    RELEVANT_VIEW = "Relevant view"
    FULL_VIEW = "Full view"

    @staticmethod
    def build_datatype_filters(selected=RELEVANT_VIEW, single_filter=None):
        """
        Return all visibility filters for data structure page, or only one filter.
        """
        filters = {StaticFiltersFactory.FULL_VIEW: FilterChain(StaticFiltersFactory.FULL_VIEW),
                   StaticFiltersFactory.RELEVANT_VIEW: FilterChain(StaticFiltersFactory.RELEVANT_VIEW,
                                                                   [FilterChain.datatype + '.visible'],
                                                                   [True], operations=["=="])}
        if selected is None or len(selected) == 0:
            selected = StaticFiltersFactory.RELEVANT_VIEW
        if selected in filters:
            filters[selected].selected = True
        if single_filter is not None:
            if single_filter in filters:
                return filters[single_filter]
            else:
                # We have some custom filter to build
                return StaticFiltersFactory._build_custom_filter(single_filter)
        return list(filters.values())

    @staticmethod
    def _build_custom_filter(filter_data):
        """
        Param filter_data should be at this point a dictionary of the form:
        {'type' : 'filter_type', 'value' : 'filter_value'}
        If 'filter_type' is not handled just return None.
        """
        filter_data = json.loads(filter_data)
        if filter_data['type'] == 'from_burst':
            return FilterChain('Burst', ['BurstConfiguration.id'],
                               [filter_data['value']], operations=["=="])
        return None

    @staticmethod
    def build_operations_filters(simulation_algorithm, logged_user_id):
        """
        :returns: list of filters that can be applied on Project View Operations page.
        """
        new_filters = []

        # Filter by algorithm / categories
        new_filter = FilterChain("Omit Views", [FilterChain.algorithm_category + '.display'],
                                 [False], operations=["=="])
        new_filters.append(new_filter)

        new_filter = FilterChain("Only Upload", [FilterChain.algorithm_category + '.rawinput'],
                                 [True], operations=["=="])
        new_filters.append(new_filter)
        if simulation_algorithm is not None:
            new_filter = FilterChain("Only Simulations", [FilterChain.algorithm + '.id'],
                                     [simulation_algorithm.id], operations=["=="])
            new_filters.append(new_filter)

        # Filter by operation status
        filtered_statuses = {model.STATUS_STARTED: "Only Running",
                             model.STATUS_ERROR: "Only with Errors",
                             model.STATUS_CANCELED: "Only Canceled",
                             model.STATUS_FINISHED: "Only Finished",
                             model.STATUS_PENDING: "Only Pending"}
        for status, title in six.iteritems(filtered_statuses):
            new_filter = FilterChain(title, [FilterChain.operation + '.status'], [status], operations=["=="])
            new_filters.append(new_filter)

        # Filter by author
        new_filter = FilterChain("Only mine", [FilterChain.operation + '.fk_launched_by'],
                                 [logged_user_id], operations=["=="])
        new_filters.append(new_filter)

        # Filter by other flags
        new_filter = FilterChain("Only relevant", [FilterChain.operation + '.visible'], [True], operations=["=="])
        new_filter.selected = True
        new_filters.append(new_filter)

        return new_filters
