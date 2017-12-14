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
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.portlets.portlet_configurer import PortletConfigurer


class TestPythonPortlets(TransactionalTestCase):
    
    
    def transactional_setup_method(self):
        """
        Sets up the environment for testing;
        creates a test user, a test project and saves config file
        """
#        self.clean_database()
        user = model.User("test_user", "test_pass", "test_mail@tvb.org", True, "user")
        self.test_user = dao.store_entity(user) 
        project = model.Project("test_proj", self.test_user.id, "description")
        self.test_project = dao.store_entity(project)
        
        
    def transactional_teardown_method(self):
        """
        Remove project folders and restore config file
        """
        FilesHelper().remove_project_structure(self.test_project.name)
        
        
    def test_portlet_configurable_interface(self):
        """
        A simple test for the get configurable interface method.
        """        
        test_portlet = dao.get_portlet_by_identifier("TA1TA2")
        
        result = PortletConfigurer(test_portlet).get_configurable_interface()
        assert len(result) == 2, "Length of the resulting interface not as expected"
        for one_entry in result:
            for entry in one_entry.interface:
                if entry['name'] == 'test1':
                    assert entry['default'] == 'step_0[0]', "Overwritten default not in effect."
                if entry['name'] == 'test2':
                    assert entry['default'] == '0', "Value that was not overwritten changed."

