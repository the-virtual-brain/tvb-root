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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import os
import unittest
from PIL import Image
from tvb.tests.framework.core.base_testcase import TransactionalTestCase
from tvb.core import utils
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.storage import dao
from tvb.core.services.figure_service import FigureService
from tvb.tests.framework.core.test_factory import TestFactory


IMG_DATA = ("iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAJElEQVQYV2Pcv3"
            "//fwYk4OjoyIjMZ6SDAmT7QGx0K1EcRBsFAFAcHPlrTpAmAAAAAElFTkSuQmCC")


class FigureServiceTest(TransactionalTestCase):
    """
    Tests for the figure service
    """
    def setUp(self):
        self.figure_service = FigureService()
        self.user = TestFactory.create_user()
        self.project = TestFactory.create_project(admin=self.user)
        self.files_helper = FilesHelper()


    def tearDown(self):
        self.delete_project_folders()


    def assertCanReadImage(self, image_path):
        try:
            Image.open(image_path).load()
        except (IOError, ValueError):
            self.fail("Could not open %s as a image" % image_path)


    def store_test_png(self):
        self.figure_service.store_result_figure(self.project, self.user, "png", IMG_DATA, image_name="test-figure")


    def retrieve_images(self):
        figures_by_session, _ = self.figure_service.retrieve_result_figures(self.project, self.user)
        # flatten image session grouping
        figures = []
        for fg in figures_by_session.itervalues():
            figures.extend(fg)
        return figures


    def test_store_image(self):
        self.store_test_png()


    def test_store_image_from_operation(self):
        # test that image can be retrieved from operation
        test_operation = TestFactory.create_operation(test_user=self.user, test_project=self.project)

        self.figure_service.store_result_figure(self.project, self.user, "png",
                                                IMG_DATA, operation_id=test_operation.id)
        figures = dao.get_figures_for_operation(test_operation.id)
        self.assertEqual(1, len(figures))
        image_path = self.files_helper.get_images_folder(self.project.name)
        image_path = os.path.join(image_path, figures[0].file_path)
        self.assertCanReadImage(image_path)


    def test_store_and_retrieve_image(self):
        self.store_test_png()
        figures = self.retrieve_images()
        self.assertEqual(1, len(figures))
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
        self.assertEqual([session_name], figures_by_session.keys())
        self.assertEqual(name, figures_by_session.values()[0][0].name)


    def test_remove_figure(self):
        self.store_test_png()
        figures = self.retrieve_images()
        self.assertEqual(1, len(figures))
        self.figure_service.remove_result_figure(figures[0].id)
        figures = self.retrieve_images()
        self.assertEqual(0, len(figures))



def suite():
    """
    Gather all the tests in a test suite.
    """
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(FigureServiceTest))
    return test_suite


if __name__ == "__main__":
    #So you can run tests from this package individually.
    TEST_RUNNER = unittest.TextTestRunner()
    TEST_SUITE = suite()
    TEST_RUNNER.run(TEST_SUITE)
