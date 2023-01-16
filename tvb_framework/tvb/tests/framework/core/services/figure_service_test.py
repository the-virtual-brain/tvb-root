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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

from PIL import Image
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core import utils
from tvb.core.services.figure_service import FigureService
from tvb.tests.framework.core.factory import TestFactory

IMG_DATA = ("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAJElEQVQYV2Pcv3"
            "//fwYk4OjoyIjMZ6SDAmT7QGx0K1EcRBsFAFAcHPlrTpAmAAAAAElFTkSuQmCC")


class TestFigureService(TransactionalTestCase):
    """
    Tests for the figure service
    """

    def transactional_setup_method(self):
        self.figure_service = FigureService()
        self.user = TestFactory.create_user()
        self.project = TestFactory.create_project(admin=self.user)

    def transactional_teardown_method(self):
        self.delete_project_folders()

    def assertCanReadImage(self, image_path):
        try:
            Image.open(image_path).load()
        except (IOError, ValueError):
            raise AssertionError("Could not open %s as a image" % image_path)

    def store_test_png(self):
        self.figure_service.store_result_figure(self.project, self.user, "png", IMG_DATA, "test-figure")

    def retrieve_images(self):
        figures_by_session, _ = self.figure_service.retrieve_result_figures(self.project, self.user)
        # flatten image session grouping
        figures = []
        for fg in figures_by_session.values():
            figures.extend(fg)
        return figures

    def test_store_image(self):
        self.store_test_png()

    def test_store_and_retrieve_image(self):
        self.store_test_png()
        figures = self.retrieve_images()
        assert 1 == len(figures)
        image_path = utils.url2path(figures[0].file_path)
        self.assertCanReadImage(image_path)

    def test_load_figure(self):
        self.store_test_png()
        figures = self.retrieve_images()
        self.figure_service.load_figure(figures[0].id)

    def test_edit_figure(self):
        session_name = 'the altered ones'
        name = 'altered'
        self.store_test_png()
        figures = self.retrieve_images()
        self.figure_service.edit_result_figure(figures[0].id, session_name=session_name, name=name)
        figures_by_session, _ = self.figure_service.retrieve_result_figures(self.project, self.user)
        assert [session_name] == list(figures_by_session)
        assert name == list(figures_by_session.values())[0][0].name

    def test_remove_figure(self):
        self.store_test_png()
        figures = self.retrieve_images()
        assert 1 == len(figures)
        self.figure_service.remove_result_figure(figures[0].id)
        figures = self.retrieve_images()
        assert 0 == len(figures)
