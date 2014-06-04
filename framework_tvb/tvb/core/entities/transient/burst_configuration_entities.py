# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
Entities transient and related to a Burst Configuration.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""



class PortletConfiguration():
    """
    Helper entity that hold the configuration of a portlet.
    Keeps track of the portlet id and the parameters for a given portlet.
    """


    def __init__(self, portlet_id):
        self.portlet_id = portlet_id
        self.visualizer = None
        self.analyzers = []


    @property
    def name(self):
        """
        The name of the portlet configuration is taken from the name of the
        workflow steps that were stored for it.
        """
        if self.visualizer is not None:
            return self.visualizer.ui_name
        if len(self.analyzers) > 0:
            return self.analyzers[0].ui_name
        return None


    @property
    def index_in_tab(self):
        """
        If it was a burst-launched portlet, it should have a visualizer and then
        the index in tab from that entity is considered the portlet configuration index in tab.
        """
        if self.visualizer is not None:
            return self.visualizer.index_in_tab
        return None


    def set_visualizer(self, visualizer):
        """
        Called in case we are just viewing an old burst launch, where a the 
        results of the analyzer part of a portlet are available, and a visualization
        could be launched.
        """
        self.visualizer = visualizer


    def set_analyzers(self, analyzers):
        """ Selected analyzers"""
        self.analyzers = analyzers


    def clone(self):
        """
        Return an exact copy of the entity with the exception than none of it's
        sub-entities (workflow steps) are persisted in db.
        """
        new_config = PortletConfiguration(self.portlet_id)
        if self.visualizer is not None:
            new_config.visualizer = self.visualizer.clone()
        for analyzer in self.analyzers:
            new_config.analyzers.append(analyzer.clone())
        return new_config


    def __repr__(self):
        return "Portlet(id=%s, analyzers=%s, visualizer=%s)" % (str(self.portlet_id),
                                                                str(self.analyzers),
                                                                str(self.visualizer))



class AdapterConfiguration():
    """
    Helper entity that holds the configuration for an adapter from a Portlet-Chain.
    Keeps track of the adapter interface, the group and the UI name. In case of 
    sub-algorithm also keep the prefix, and the pair {sub-algorithm : name}
    """


    def __init__(self, interface, group, ui_name=None, prefix='', subalgorithm_field=None, subalgorithm_value=None):
        self.interface = interface
        self.group = group
        self._ui_name = ui_name
        self.prefix = prefix
        self.subalgorithm_field = subalgorithm_field
        self.subalgorithm_value = subalgorithm_value


    @property
    def ui_name(self):
        """ UI title """
        if self._ui_name is not None:
            return self._ui_name
        return self.group.displayname



class WorkflowStepConfiguration():
    """
    Helper entity that holds the configuration needed to build a WorkflowStep.
    Holds the algorithm id for the corresponding adapter, plus the dynamic and 
    static parameters that are required to launch this step.
    """

    STEP_INDEX_KEY = "step_index"
    DATATYPE_INDEX_KEY = "datatype_idx"


    def __init__(self, algorithm_id, static_params=None, dynamic_params=None):
        self.algorithm_id = algorithm_id
        self.static_params = static_params
        self.dynamic_params = dynamic_params


    def __repr__(self):
        return str("WorkflowStepConfiguration(algorithm_id:%s, static_params:%s, dynamic_" +
                   "params:%s") % (str(self.algorithm_id), str(self.static_params), str(self.dynamic_params))
        

