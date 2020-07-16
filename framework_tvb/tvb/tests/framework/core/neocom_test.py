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
import os

import numpy
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.file.simulator.view_model import SimulatorAdapterModel, EEGViewModel, HeunStochasticViewModel
from tvb.core.entities.storage import dao
from tvb.core.neocom import h5
from tvb.core.neocom.h5 import load, store, load_from_dir, store_to_dir
from tvb.datatypes.projections import ProjectionSurfaceEEG


def test_store_load(tmpdir, connectivity_factory):
    path = os.path.join(str(tmpdir), 'interface.conn.h5')
    connectivity = connectivity_factory(2)
    store(connectivity, path)
    con2 = load(path)
    numpy.testing.assert_equal(connectivity.weights, con2.weights)


def test_store_load_rec(tmpdir, connectivity_factory, region_mapping_factory):
    connectivity = connectivity_factory(2)
    region_mapping = region_mapping_factory(connectivity=connectivity)
    store_to_dir(region_mapping, str(tmpdir), recursive=True)

    rmap = load_from_dir(str(tmpdir), region_mapping.gid, recursive=True)
    numpy.testing.assert_equal(connectivity.weights, rmap.connectivity.weights)


def test_store_simulator_view_model(connectivity_index_factory, operation_factory):
    conn = connectivity_index_factory()
    sim_view_model = SimulatorAdapterModel()
    sim_view_model.connectivity = conn.gid

    op = operation_factory()
    storage_path = FilesHelper().get_project_folder(op.project, str(op.id))

    h5.store_view_model(sim_view_model, storage_path)

    loaded_sim_view_model = h5.load_view_model(sim_view_model.gid, storage_path)

    assert isinstance(sim_view_model, SimulatorAdapterModel)
    assert isinstance(loaded_sim_view_model, SimulatorAdapterModel)


def test_store_simulator_view_model_noise(connectivity_index_factory, operation_factory):
    conn = connectivity_index_factory()
    sim_view_model = SimulatorAdapterModel()
    sim_view_model.connectivity = conn.gid
    sim_view_model.integrator = HeunStochasticViewModel()
    sim_view_model.integrator.noise.noise_seed = 45

    op = operation_factory()
    storage_path = FilesHelper().get_project_folder(op.project, str(op.id))

    h5.store_view_model(sim_view_model, storage_path)

    loaded_sim_view_model = h5.load_view_model(sim_view_model.gid, storage_path)

    assert isinstance(sim_view_model, SimulatorAdapterModel)
    assert isinstance(loaded_sim_view_model, SimulatorAdapterModel)
    assert sim_view_model.integrator.noise.noise_seed == loaded_sim_view_model.integrator.noise.noise_seed == 45


def test_store_simulator_view_model_eeg(connectivity_index_factory, surface_index_factory, region_mapping_factory,
                                        sensors_index_factory, operation_factory):
    conn = connectivity_index_factory()
    surface_idx, surface = surface_index_factory(cortical=True)
    region_mapping = region_mapping_factory()
    sensors_idx, sensors = sensors_index_factory()
    proj = ProjectionSurfaceEEG(sensors=sensors, sources=surface, projection_data=numpy.ones(3))

    op = operation_factory()
    storage_path = FilesHelper().get_project_folder(op.project, str(op.id))
    prj_db_db = h5.store_complete(proj, storage_path)
    prj_db_db.fk_from_operation = op.id
    dao.store_entity(prj_db_db)

    seeg_monitor = EEGViewModel(projection=proj.gid, sensors=sensors.gid)
    seeg_monitor.region_mapping = region_mapping.gid.hex
    sim_view_model = SimulatorAdapterModel()
    sim_view_model.connectivity = conn.gid
    sim_view_model.monitors = [seeg_monitor]

    op = operation_factory()
    storage_path = FilesHelper().get_project_folder(op.project, str(op.id))

    h5.store_view_model(sim_view_model, storage_path)

    loaded_sim_view_model = h5.load_view_model(sim_view_model.gid, storage_path)

    assert isinstance(sim_view_model, SimulatorAdapterModel)
    assert isinstance(loaded_sim_view_model, SimulatorAdapterModel)
    assert sim_view_model.monitors[0].projection == loaded_sim_view_model.monitors[0].projection
