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
.. moduleauthor:: bogdan.neacsa <bogdan.neacsa@codemart.ro>
"""

from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.entities.transient.burst_configuration_entities import WorkflowStepConfiguration as wf_cfg
from tvb.core.services.flow_service import FlowService
from tvb.core.services.workflow_service import WorkflowService
from tvb.core.services.burst_service import BurstService
from tvb.core.services.operation_service import OperationService
from tvb.tests.framework.datatypes.datatype1 import Datatype1
from tvb.tests.framework.datatypes import datatypes_factory
from tvb.tests.framework.core.factory import TestFactory



class TestWorkflow(TransactionalTestCase):
    """
    Test that workflow conversion methods are valid.
    """


    def transactional_setup_method(self):
        """
        Sets up the testing environment;
        saves config file;
        creates a test user, a test project;
        creates burst, operation, flow and workflow services
        """
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user)
        self.workflow_service = WorkflowService()
        self.burst_service = BurstService()
        self.operation_service = OperationService()
        self.flow_service = FlowService()


    def transactional_teardown_method(self):
        """
        Remove project folders and clean up database.
        """
        FilesHelper().remove_project_structure(self.test_project.name)
        self.delete_project_folders()


    def __create_complex_workflow(self, workflow_step_list):
        """
        Creates a burst with a complex workflow with a given list of workflow steps.
        :param workflow_step_list: a list of workflow steps that will be used in the
            creation of a new workflow for a new burst
        """
        burst_config = TestFactory.store_burst(self.test_project.id)

        stored_dt = datatypes_factory.DatatypesFactory()._store_datatype(Datatype1())

        first_step_algorithm = self.flow_service.get_algorithm_by_module_and_class("tvb.tests.framework.adapters.testadapter1",
                                                                                   "TestAdapterDatatypeInput")
        metadata = {DataTypeMetaData.KEY_BURST: burst_config.id}
        kwargs = {"test_dt_input": stored_dt.gid, 'test_non_dt_input': '0'}
        operations, group = self.operation_service.prepare_operations(self.test_user.id, self.test_project.id,
                                                                      first_step_algorithm,
                                                                      first_step_algorithm.algorithm_category,
                                                                      metadata, **kwargs)

        workflows = self.workflow_service.create_and_store_workflow(project_id=self.test_project.id,
                                                                    burst_id=burst_config.id,
                                                                    simulator_index=0,
                                                                    simulator_id=first_step_algorithm.id,
                                                                    operations=operations)
        self.operation_service.prepare_operations_for_workflowsteps(workflow_step_list, workflows, self.test_user.id,
                                                                    burst_config.id, self.test_project.id, group,
                                                                    operations)
        #fire the first op
        if len(operations) > 0:
            self.operation_service.launch_operation(operations[0].id, False)
        return burst_config.id


    def test_workflow_generation(self):
        """
        A simple test just for the fact that a workflow is created an ran, 
        no dynamic parameters are passed. In this case we create a two steps
        workflow: step1 - tvb.tests.framework.adapters.testadapter2.TestAdapter2
                  step2 - tvb.tests.framework.adapters.testadapter1.TestAdapter1
        The first adapter doesn't return anything and the second returns one
        tvb.datatypes.datatype1.Datatype1 instance. We check that the steps
        are actually ran by checking that two operations are created and that
        one dataType is stored.
        """
        workflow_step_list = [TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter2",
                                                               "TestAdapter2", step_index=1,
                                                               static_kwargs={"test2": 2}),
                              TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter1",
                                                               "TestAdapter1", step_index=2,
                                                               static_kwargs={"test1_val1": 1, "test1_val2": 1})]
        self.__create_complex_workflow(workflow_step_list)
        stored_datatypes = dao.get_datatypes_in_project(self.test_project.id)
        assert  len(stored_datatypes) == 2, "DataType from second step was not stored."
        assert  stored_datatypes[0].type == 'Datatype1', "Wrong type was stored."
        assert  stored_datatypes[1].type == 'Datatype1', "Wrong type was stored."

        finished, started, error, _, _ = dao.get_operation_numbers(self.test_project.id)
        assert  finished == 3, "Didnt start operations for both adapters in workflow."
        assert  started == 0, "Some operations from workflow didnt finish."
        assert  error == 0, "Some operations finished with error status."


    def test_workflow_dynamic_params(self):
        """
        A simple test just for the fact that dynamic parameters are passed properly
        between two workflow steps: 
                  step1 - tvb.tests.framework.adapters.testadapter1.TestAdapter1
                  step2 - tvb.tests.framework.adapters.testadapter3.TestAdapter3
        The first adapter returns a tvb.datatypes.datatype1.Datatype1 instance. 
        The second adapter has this passed as a dynamic workflow parameter.
        We check that the steps are actually ran by checking that two operations 
        are created and that two dataTypes are stored.
        """
        workflow_step_list = [TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter1",
                                                               "TestAdapter1", step_index=1,
                                                               static_kwargs={"test1_val1": 1, "test1_val2": 1}),
                              TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter3",
                                                               "TestAdapter3", step_index=2,
                                                               dynamic_kwargs={
                                                                   "test": {wf_cfg.DATATYPE_INDEX_KEY: 0,
                                                                            wf_cfg.STEP_INDEX_KEY: 1}})]

        self.__create_complex_workflow(workflow_step_list)
        stored_datatypes = dao.get_datatypes_in_project(self.test_project.id)
        assert  len(stored_datatypes) == 3, "DataType from all step were not stored."
        for result_row in stored_datatypes:
            assert  result_row.type in ['Datatype1', 'Datatype2'], "Wrong type was stored."

        finished, started, error, _, _ = dao.get_operation_numbers(self.test_project.id)
        assert  finished == 3, "Didn't start operations for both adapters in workflow."
        assert  started == 0, "Some operations from workflow didn't finish."
        assert  error == 0, "Some operations finished with error status."


    def test_configuration2workflow(self):
        """
        Test that building a WorkflowStep from a WorkflowStepConfiguration. Make sure all the data is
        correctly passed. Also check that any base_wf_step is incremented to dynamic parameters step index.
        """
        workflow_step = TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter1", "TestAdapter1",
                                                         static_kwargs={"static_param": "test"},
                                                         dynamic_kwargs={"dynamic_param": {wf_cfg.STEP_INDEX_KEY: 0,
                                                                                           wf_cfg.DATATYPE_INDEX_KEY: 0}},
                                                         step_index=1, base_step=5)
        assert  workflow_step.step_index == 1, "Wrong step index in created workflow step."
        assert  workflow_step.static_param == {'static_param': 'test'}, 'Different static parameters on step.'
        assert  workflow_step.dynamic_param == {'dynamic_param': {wf_cfg.STEP_INDEX_KEY: 5,
                                                                         wf_cfg.DATATYPE_INDEX_KEY: 0}},\
                         "Dynamic parameters not saved properly, or base workflow index not added to step index."


    def test_create_workflow(self):
        """
        Test that a workflow with all the associated workflow steps is actually created.
        """
        workflow_step_list = [TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter2",
                                                               "TestAdapter2", step_index=1,
                                                               static_kwargs={"test2": 2}),
                              TestFactory.create_workflow_step("tvb.tests.framework.adapters.testadapter1",
                                                               "TestAdapter1", step_index=2,
                                                               static_kwargs={"test1_val1": 1, "test1_val2": 1})]
        burst_id = self.__create_complex_workflow(workflow_step_list)
        workflow_entities = dao.get_workflows_for_burst(burst_id)
        assert  len(workflow_entities) == 1, "For some reason workflow was not stored in database."
        workflow_steps = dao.get_workflow_steps(workflow_entities[0].id)
        assert  len(workflow_steps) == len(workflow_step_list) + 1, "Wrong number of workflow steps created."
