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
import pytest
from tvb.config.logger.elasticsearch_handler import _retrieve_user_gid


@pytest.mark.benchmark
def test_retrieve_user_gid(benchmark):
    text = "2023-02-17 23:34:46,563 - TRACE - tvb_user_actions - USER: 7dd030bb-6f70-40ca-9539-396943278560 | METHOD: <bound method ProjectController.fill_default_attributes of <tvb.interfaces.web.controllers.project.project_controller.ProjectController object at 0x7fb0619ef220>> | PARAMS: ({'selectedProject': <Project('Default_Project', '2')>},) *{"
    result = benchmark(_retrieve_user_gid, text)

    assert result == "7dd030bb-6f70-40ca-9539-396943278560"

@pytest.mark.benchmark
def test_retrieve_no_user_gid(benchmark):
    text = "2023-02-17 23:34:46,563 - TRACE - tvb_user_actions - | METHOD: <bound method ProjectController.fill_default_attributes of <tvb.interfaces.web.controllers.project.project_controller.ProjectController object at 0x7fb0619ef220>> | PARAMS: ({'selectedProject': <Project('Default_Project', '2')>},) *{"
    result = benchmark(_retrieve_user_gid, text)

    assert result == ""



