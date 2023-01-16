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
This is just a shortcut, in order to make all DAO functions 
accessible from a  single point, in an uniform manner, 
without supplementary name space.
We want DAO functions to be separated in multiple files, 
because they are too many to easily follow.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""
from tvb.core.entities.storage.burst_dao import BurstDAO
from tvb.core.entities.storage.session_maker import transactional, SA_SESSIONMAKER
from tvb.core.entities.storage.project_dao import CaseDAO
from tvb.core.entities.storage.datatype_dao import DatatypeDAO
from tvb.core.entities.storage.operation_dao import OperationDAO
from tvb.core.entities.storage.workflow_dao import WorkflowDAO


class DAO(DatatypeDAO, OperationDAO, CaseDAO, BurstDAO, WorkflowDAO):
    """
    Empty class, build only for inheriting from all DAO classes.
    """
    pass


dao = DAO()
