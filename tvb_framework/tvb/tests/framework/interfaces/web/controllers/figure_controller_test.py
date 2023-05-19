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

import cherrypy
from tvb.core.entities.storage import dao
from tvb.interfaces.web.controllers.project.figure_controller import FigureController
from tvb.tests.framework.core.factory import TestFactory
from tvb.tests.framework.interfaces.web.controllers.base_controller_test import BaseTransactionalControllerTest


class TestFigureController(BaseTransactionalControllerTest):
    """ Unit tests for FigureController """

    def transactional_setup_method(self):
        self.init()
        self.figure_c = FigureController()

    def transactional_teardown_method(self):
        """ Cleans the testing environment """
        self.cleanup()

    def test_displayresultfigures(self):
        """
        Tests result dictionary for the expected key/value
        """
        figure1 = TestFactory.create_figure(self.test_user.id, self.test_project.id, name="figure1",
                                            path="path-to-figure1", session_name="test")
        figure2 = TestFactory.create_figure(self.test_user.id, self.test_project.id, name="figure2",
                                            path="path-to-figure2", session_name="test")

        result_dict = self.figure_c.displayresultfigures()
        figures = result_dict['selected_sessions_data']['test']
        assert set([fig.id for fig in figures]) == {figure1.id, figure2.id}

    def test_editresultfigures_remove_fig(self):
        """
        Tests call to `editresultfigures` correctly redirects to '/project/figure/displayresultfigures'
        on figure removal
        """
        cherrypy.request.method = 'POST'
        figure1 = TestFactory.create_figure(self.test_user.id, self.test_project.id, name="figure1",
                                            path="path-to-figure1", session_name="test42")
        figs, _ = dao.get_previews(self.test_project.id, self.test_user.id, "test42")
        assert len(figs['test42']) == 1
        data = {'figure_id': figure1.id}

        self._expect_redirect('/project/figure/displayresultfigures', self.figure_c.editresultfigures,
                              remove_figure=True, **data)

        figs, _ = dao.get_previews(self.test_project.id, self.test_user.id, "test42")
        assert len(figs['test42']) == 0

    def test_editresultfigures_rename_session(self):
        """
        Tests result dictionary has the expected keys / values and call to `editresultfigures`
        correctly redirects to '/project/figure/displayresultfigures' on session renaming
        """
        cherrypy.request.method = 'POST'
        TestFactory.create_figure(self.test_user.id, self.test_project.id, name="figure1",
                                  path="path-to-figure1", session_name="test")
        TestFactory.create_figure(self.test_user.id, self.test_project.id, name="figure2",
                                  path="path-to-figure2", session_name="test")
        figs, _ = dao.get_previews(self.test_project.id, self.test_user.id, "test")
        assert len(figs['test']) == 2
        data = {'old_session_name': 'test', 'new_session_name': 'test_renamed'}
        self._expect_redirect('/project/figure/displayresultfigures', self.figure_c.editresultfigures,
                              rename_session=True, **data)
        figs, previews = dao.get_previews(self.test_project.id, self.test_user.id, "test")
        assert len(figs['test']) == 0
        assert previews['test_renamed'] == 2

    def test_editresultfigures_remove_session(self):
        """
        Tests result dictionary has the expected keys / values and call to `editresultfigures`
        correctly redirects to '/project/figure/displayresultfigures' on session removal
        """
        cherrypy.request.method = 'POST'
        TestFactory.create_figure(self.test_user.id, self.test_project.id, name="figure1",
                                  path="path-to-figure1", session_name="test")
        TestFactory.create_figure(self.test_user.id, self.test_project.id, name="figure2",
                                  path="path-to-figure2", session_name="test")
        figs, _ = dao.get_previews(self.test_project.id, self.test_user.id, "test")
        assert len(figs['test']) == 2
        data = {'old_session_name': 'test', 'new_session_name': 'test_renamed'}
        self._expect_redirect('/project/figure/displayresultfigures', self.figure_c.editresultfigures,
                              remove_session=True, **data)
        figs, previews = dao.get_previews(self.test_project.id, self.test_user.id, "test")
        assert len(figs['test']) == 0
        assert previews == {}
