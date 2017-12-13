# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import copy
import json
import numpy
import pytest
import cherrypy
from time import sleep
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseControllersTest
from tvb.config import SIMULATOR_MODULE, SIMULATOR_CLASS
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.burst.burst_controller import BurstController
from tvb.datatypes.connectivity import Connectivity
from tvb.core.entities import model
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.storage import dao
from tvb.core.entities.model.model_burst import BurstConfiguration, NUMBER_OF_PORTLETS_PER_TAB
from tvb.core.entities.transient.burst_configuration_entities import AdapterConfiguration
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.services.burst_service import BurstService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.flow_service import FlowService
from tvb.tests.framework.adapters.storeadapter import StoreAdapter
from tvb.tests.framework.adapters.simulator.simulator_adapter_test import SIMULATOR_PARAMETERS



class TestBurstContoller(BaseControllersTest):
    """ Unit tests for burst_controller """


    def setup_method(self):
        """
        Sets up the environment for testing;
        creates a `BurstController`
        """
        BaseControllersTest.init(self)
        self.burst_c = BurstController()


    def teardown_method(self):
        """
        Cleans up the environment after testing is done
        """
        self.cleanup()
        self.clean_database()


    def test_index(self):
        """
        Test that index returns a dict with all required keys. Also check
        that the default portlets are populated, with only the first being
        the TimeSeries portlet and the rest are empty.
        """
        result_dict = self.burst_c.index()
        assert 'burst_list' in result_dict and result_dict['burst_list'] == []
        assert 'available_metrics' in result_dict and isinstance(result_dict['available_metrics'], list)
        assert 'portletList' in result_dict and isinstance(result_dict['portletList'], list)
        assert result_dict[common.KEY_SECTION] == "burst"
        assert 'burstConfig' in result_dict and isinstance(result_dict['burstConfig'], BurstConfiguration)
        portlets = json.loads(result_dict['selectedPortlets'])
        portlet_id = dao.get_portlet_by_identifier("TimeSeries").id
        for tab_idx, tab in enumerate(portlets):
            for index_in_tab, value in enumerate(tab):
                if tab_idx == 0 and index_in_tab == 0:
                    assert value == [portlet_id, "TimeSeries"]
                else:
                    assert value == [-1, "None"]
        assert result_dict['draw_hidden_ranges']


    def test_load_burst_history(self):
        """
        Create two burst, load the burst and check that we get back
        the same stored bursts.
        """
        self._store_burst(self.test_project.id, 'started', {'test': 'test'}, 'burst1')
        burst = self._store_burst(self.test_project.id, 'started', {'test': 'test'}, 'burst2')
        cherrypy.session[common.KEY_BURST_CONFIG] = burst
        result_dict = self.burst_c.load_burst_history()
        burst_history = result_dict['burst_list']
        assert len(burst_history) == 2
        for burst in burst_history:
            assert burst.name in ('burst1', 'burst2')


    def test_get_selected_burst(self):
        """
        Create burst, add it to session, then check that get_selected_burst
        return the same burst. Also check that for an unstored entity we get
        back 'None'
        """
        burst_entity = BurstConfiguration(self.test_project.id, 'started', {}, 'burst1')
        cherrypy.session[common.KEY_BURST_CONFIG] = burst_entity
        stored_id = self.burst_c.get_selected_burst()
        assert stored_id == 'None'
        burst_entity = dao.store_entity(burst_entity)
        cherrypy.session[common.KEY_BURST_CONFIG] = burst_entity
        stored_id = self.burst_c.get_selected_burst()
        assert str(stored_id) == str(burst_entity.id)


    def test_get_portlet_configurable_interface(self):
        """
        Look up that an AdapterConfiguration is returned for the default
        portlet configuration, if we look at index (0, 0) where TimeSeries portlet
        should be default.
        """
        self.burst_c.index()
        result = self.burst_c.get_portlet_configurable_interface(0)
        assert common.KEY_PARAMETERS_CONFIG in result
        assert not result[common.KEY_PARAMETERS_CONFIG]
        adapter_config = result['adapters_list']
        # Default TimeSeries portlet should be available, so we expect
        # adapter_config to be a list of AdapterConfiguration with one element
        assert len(adapter_config) == 1
        assert isinstance(adapter_config[0], AdapterConfiguration)


    def test_portlet_tab_display(self):
        """
        Update the default portlet configuration, by storing a TimeSeries
        portlet for all postions. Then check that we get the same configuration.
        """
        self.burst_c.index()
        portlet_id = dao.get_portlet_by_identifier("TimeSeries").id
        one_tab = [[portlet_id, "TimeSeries"] for _ in range(NUMBER_OF_PORTLETS_PER_TAB)]
        full_tabs = [one_tab for _ in range(BurstConfiguration.nr_of_tabs)]
        data = {'tab_portlets_list': json.dumps(full_tabs)}
        result = self.burst_c.portlet_tab_display(**data)
        selected_portlets = result['portlet_tab_list']
        for entry in selected_portlets:
            assert entry.id == portlet_id


    def test_get_configured_portlets_no_session(self):
        """
        Test that if we have no burst stored in session, an empty
        portlet list is reduced.
        """
        result = self.burst_c.get_configured_portlets()
        assert 'portlet_tab_list' in result
        assert result['portlet_tab_list'] == []


    def test_get_configured_portlets_default(self):
        """
        Check that the default configuration holds one portlet
        and it's identifier is 'TimeSeries'.
        """
        self.burst_c.index()
        result = self.burst_c.get_configured_portlets()
        assert 'portlet_tab_list' in result
        portlets_list = result['portlet_tab_list']
        assert len(portlets_list) == 1
        assert portlets_list[0].algorithm_identifier == 'TimeSeries'


    def test_get_portlet_session_configuration(self):
        """
        Test that the default portlet session sonciguration is generated
        as expected, with a default TimeSeries portlet and rest empty.
        """
        self.burst_c.index()
        result = json.loads(self.burst_c.get_portlet_session_configuration())
        portlet_id = dao.get_portlet_by_identifier("TimeSeries").id
        for tab_idx, tab in enumerate(result):
            for index_in_tab, value in enumerate(tab):
                if tab_idx == 0 and index_in_tab == 0:
                    assert value == [portlet_id, "TimeSeries"]
                else:
                    assert value == [-1, "None"]


    def test_save_parameters_no_relaunch(self):
        """
        Test the save parameters for the default TimeSeries portlet and
        pass an empty dictionary as the 'new' data. In this case a relaunch
        should not be required.
        """
        self.burst_c.index()
        assert 'noRelaunch' == self.burst_c.save_parameters(0, portlet_parameters="{}")


    def test_rename_burst(self):
        """
        Create and store a burst, then rename it and check that it
        works as expected.
        """
        burst = self._store_burst(self.test_project.id, 'started', {'test': 'test'}, 'burst1')
        self.burst_c.rename_burst(burst.id, "test_new_burst_name")
        renamed_burst = dao.get_burst_by_id(burst.id)
        assert renamed_burst.name == "test_new_burst_name"


    def test_launch_burst(self):
        """
        Launch a burst and check that it finishes correctly and before timeout (100)
        """
        self.burst_c.index()
        connectivity = self._burst_create_connectivity()
        launch_params = copy.deepcopy(SIMULATOR_PARAMETERS)
        launch_params['connectivity'] = connectivity.gid
        launch_params['simulation_length'] = '10'
        launch_params = {"simulator_parameters": json.dumps(launch_params)}
        burst_id = json.loads(self.burst_c.launch_burst("new", "test_burst", **launch_params))['id']
        waited = 1
        timeout = 100
        burst_config = dao.get_burst_by_id(burst_id)
        while burst_config.status == BurstConfiguration.BURST_RUNNING and waited <= timeout:
            sleep(0.5)
            waited += 0.5
            burst_config = dao.get_burst_by_id(burst_config.id)
        if waited > timeout:
            raise AssertionError("Timed out waiting for simulations to finish.")
        if burst_config.status != BurstConfiguration.BURST_FINISHED:
            BurstService().stop_burst(burst_config)
            raise AssertionError("Burst should have finished successfully.")


    def test_load_burst(self):
        """
        Test loading and burst and checking you get expected dictionary.
        """
        self.burst_c.index()
        burst = self._store_burst(self.test_project.id, 'started', {'test': 'test'}, 'burst1')
        result = json.loads(self.burst_c.load_burst(burst.id))
        assert result["status"] == "started"
        assert result['group_gid'] == None
        assert result['selected_tab'] == 0


    def test_load_burst_removed(self):
        """
        Add burst to session, then remove burst from database. Try to load
        burst and check that it will raise exception and remove it from session.
        """
        burst = self._store_burst(self.test_project.id, 'started', {'test': 'test'}, 'burst1')
        cherrypy.session[common.KEY_BURST_CONFIG] = burst
        burst_id = burst.id
        BurstService().cancel_or_remove_burst(burst_id)
        with pytest.raises(Exception):
            self.burst_c.load_burst(burst_id)
        assert common.KEY_BURST_CONFIG not in cherrypy.session


    def test_remove_burst_not_session(self):
        """
        Test removing a burst that is not the one currently stored in 
        session. SHould just remove and return a 'done' string.
        """
        burst = self._store_burst(self.test_project.id, 'finished', {'test': 'test'}, 'burst1')
        cherrypy.session[common.KEY_BURST_CONFIG] = burst
        another_burst = self._store_burst(self.test_project.id, 'finished', {'test': 'test'}, 'burst1')
        result = self.burst_c.cancel_or_remove_burst(another_burst.id)
        assert result == 'done'


    def test_remove_burst_in_session(self):
        """
        Test that if we remove the burst that is the current one from the
        session, we get a 'reset-new' string as result.
        """
        burst = self._store_burst(self.test_project.id, 'finished', {'test': 'test'}, 'burst1')
        cherrypy.session[common.KEY_BURST_CONFIG] = burst
        result = self.burst_c.cancel_or_remove_burst(burst.id)
        assert result == 'reset-new'


    def _store_burst(self, proj_id, status, sim_config, name):
        """
        Create and store a burst entity, for the project given project_id, having the
        given status and simulator parames config, under the given name.
        """
        burst = BurstConfiguration(proj_id, status, sim_config, name)
        burst.prepare_before_save()
        return dao.store_entity(burst)


    def _burst_create_connectivity(self):
        """
        Create a connectivity that will be used in "non-dummy" burst launches (with the actual simulator).
        TODO: This is duplicate code from burstservice_test. Should go into the 'generic' DataType factory
        once that is done.
        """
        meta = {DataTypeMetaData.KEY_SUBJECT: "John Doe", DataTypeMetaData.KEY_STATE: "RAW_DATA"}
        algorithm = FlowService().get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS)
        self.operation = model.Operation(self.test_user.id, self.test_project.id, algorithm.id,
                                         json.dumps(''), meta=json.dumps(meta), status=model.STATUS_STARTED)
        self.operation = dao.store_entity(self.operation)
        storage_path = FilesHelper().get_project_folder(self.test_project, str(self.operation.id))
        connectivity = Connectivity(storage_path=storage_path)
        connectivity.weights = numpy.ones((74, 74))
        connectivity.centres = numpy.ones((74, 3))
        adapter_instance = StoreAdapter([connectivity])
        OperationService().initiate_prelaunch(self.operation, adapter_instance, {})
        return connectivity
