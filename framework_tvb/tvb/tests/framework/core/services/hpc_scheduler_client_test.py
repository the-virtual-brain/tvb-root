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
.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
"""

import os
import shutil

import numpy
from tvb.adapters.simulator.hpc_simulator_adapter import HPCSimulatorAdapter
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.view_model import EEGViewModel
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.operation_hpc_launcher import do_operation_launch
from tvb.core.services.backend_clients.hpc_scheduler_client import HPCSchedulerClient
from tvb.core.services.encryption_handler import EncryptionHandler
from tvb.datatypes.projections import ProjectionSurfaceEEG
from tvb.tests.framework.core.base_testcase import BaseTestCase
from tvb.tests.framework.core.factory import TestFactory


def _update_operation_status(status, simulator_gid, op_id, base_url):
    pass


def _request_passfile_dummy(simulator_gid, op_id, base_url, passfile_folder):
    pass


class TestHPCSchedulerClient(BaseTestCase):

    def setup_method(self):
        self.files_helper = FilesHelper()
        self.encryption_handler = EncryptionHandler('123')
        self.clean_database()
        self.test_user = TestFactory.create_user()
        self.test_project = TestFactory.create_project(self.test_user)

    def _prepare_dummy_files(self, tmpdir):
        dummy_file1 = os.path.join(str(tmpdir), 'dummy1.txt')
        open(dummy_file1, 'a').close()
        dummy_file2 = os.path.join(str(tmpdir), 'dummy2.txt')
        open(dummy_file2, 'a').close()
        job_inputs = [dummy_file1, dummy_file2]
        return job_inputs

    def test_encrypt_inputs(self, tmpdir):
        job_inputs = self._prepare_dummy_files(tmpdir)
        job_encrypted_inputs = self.encryption_handler.encrypt_inputs(job_inputs)
        # Encrypted folder has 2 more files are more then plain folder
        assert len(job_encrypted_inputs) == len(job_inputs)

    def test_decrypt_results(self, tmpdir):
        # Prepare encrypted dir
        job_inputs = self._prepare_dummy_files(tmpdir)
        self.encryption_handler.encrypt_inputs(job_inputs)
        encrypted_dir = self.encryption_handler.get_encrypted_dir()

        # Unencrypt data
        out_dir = os.path.join(str(tmpdir), 'output')
        os.mkdir(out_dir)
        self.encryption_handler.decrypt_results_to_dir(out_dir)
        list_plain_dir = os.listdir(out_dir)
        assert len(list_plain_dir) == len(os.listdir(encrypted_dir))
        assert 'dummy1.txt' in list_plain_dir
        assert 'dummy2.txt' in list_plain_dir

    def test_do_operation_launch(self, simulator_factory, operation_factory, mocker, tmpdir):
        # Prepare encrypted dir
        op = operation_factory(test_user=self.test_user, test_project=self.test_project)
        sim_folder, sim_gid = simulator_factory(op=op)
        self.encryption_handler = EncryptionHandler(sim_gid)
        job_encrypted_inputs = HPCSchedulerClient()._prepare_input(op, sim_gid)
        self.encryption_handler.encrypt_inputs(job_encrypted_inputs)
        encrypted_dir = self.encryption_handler.get_encrypted_dir()

        mocker.patch('tvb.core.operation_hpc_launcher._request_passfile', _request_passfile_dummy)
        mocker.patch('tvb.core.operation_hpc_launcher._update_operation_status', _update_operation_status)

        # Call do_operation_launch similarly to CSCS env
        plain_dir = self.files_helper.get_project_folder(self.test_project, 'plain')
        do_operation_launch(sim_gid.hex, 1000, False, '', op.id, plain_dir)
        assert len(os.listdir(encrypted_dir)) == 7
        output_path = os.path.join(encrypted_dir, HPCSimulatorAdapter.OUTPUT_FOLDER)
        assert os.path.exists(output_path)
        assert len(os.listdir(output_path)) == 2

    def test_do_operation_launch_pse(self, simulator_factory, operation_factory, mocker, tmpdir):
        # Prepare encrypted dir
        op = operation_factory(test_user=self.test_user, test_project=self.test_project)
        sim_folder, sim_gid = simulator_factory(op=op)
        self.encryption_handler = EncryptionHandler(sim_gid)
        job_encrypted_inputs = HPCSchedulerClient()._prepare_input(op, sim_gid)
        self.encryption_handler.encrypt_inputs(job_encrypted_inputs)
        encrypted_dir = self.encryption_handler.get_encrypted_dir()

        mocker.patch('tvb.core.operation_hpc_launcher._request_passfile', _request_passfile_dummy)
        mocker.patch('tvb.core.operation_hpc_launcher._update_operation_status', _update_operation_status)

        # Call do_operation_launch similarly to CSCS env
        plain_dir = self.files_helper.get_project_folder(self.test_project, 'plain')
        do_operation_launch(sim_gid.hex, 1000, True, '', op.id, plain_dir)
        assert len(os.listdir(encrypted_dir)) == 7
        output_path = os.path.join(encrypted_dir, HPCSimulatorAdapter.OUTPUT_FOLDER)
        assert os.path.exists(output_path)
        assert len(os.listdir(output_path)) == 2

    def test_prepare_inputs(self, operation_factory, simulator_factory):
        op = operation_factory(test_user=self.test_user, test_project=self.test_project)
        sim_folder, sim_gid = simulator_factory(op=op)
        hpc_client = HPCSchedulerClient()
        input_files = hpc_client._prepare_input(op, sim_gid)
        assert len(input_files) == 6

    def test_prepare_inputs_with_surface(self, operation_factory, simulator_factory):
        op = operation_factory(test_user=self.test_user, test_project=self.test_project)
        sim_folder, sim_gid = simulator_factory(op=op, with_surface=True)
        hpc_client = HPCSchedulerClient()
        input_files = hpc_client._prepare_input(op, sim_gid)
        assert len(input_files) == 9

    def test_prepare_inputs_with_eeg_monitor(self, operation_factory, simulator_factory, surface_index_factory,
                                             sensors_index_factory):
        surface_idx, surface = surface_index_factory(cortical=True)
        sensors_idx, sensors = sensors_index_factory()
        proj = ProjectionSurfaceEEG(sensors=sensors, sources=surface, projection_data=numpy.ones(3))

        op = operation_factory()
        storage_path = FilesHelper().get_project_folder(op.project, str(op.id))
        prj_db_db = h5.store_complete(proj, storage_path)
        prj_db_db.fk_from_operation = op.id
        dao.store_entity(prj_db_db)

        eeg_monitor = EEGViewModel(projection=proj.gid, sensors=sensors.gid)

        sim_folder, sim_gid = simulator_factory(op=op, monitor=eeg_monitor)
        hpc_client = HPCSchedulerClient()
        input_files = hpc_client._prepare_input(op, sim_gid)
        assert len(input_files) == 10

    def teardown_method(self):
        encrypted_dir = self.encryption_handler.get_encrypted_dir()
        if os.path.isdir(encrypted_dir):
            shutil.rmtree(encrypted_dir)
        passfile = self.encryption_handler.get_password_file()
        if os.path.exists(passfile):
            os.remove(passfile)
        self.files_helper.remove_project_structure(self.test_project.name)
        self.clean_database()
