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
Service layer for Caching mechanism.

.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""
from functools import lru_cache
from sqlalchemy.exc import SQLAlchemyError
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_operation import OperationGroup
from tvb.core.entities.storage import dao
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.kube_service import KubeService

LOGGER = get_logger(__name__)


class CacheService:
    """
    Service layer for caching mechanism.
    """
    user_cache = {}

    def __init__(self):
        self.alg_service = AlgorithmService()

    @lru_cache()
    def cached_operation_group(self, op_group_id):
        try:
            return dao.get_generic_entity(OperationGroup, op_group_id)[0]
        except SQLAlchemyError:
            return None

    @lru_cache()
    def cached_dt_group(self, op_group_id):
        return dao.get_datatypegroup_by_op_group_id(op_group_id)

    @lru_cache()
    def cached_visualizers_for_group(self, datatype_group_gid):
        return self.alg_service.get_visualizers_for_group(datatype_group_gid)

    @lru_cache()
    def cached_algorithm(self, alg_id):
        return dao.get_algorithm_by_id(alg_id)

    @lru_cache()
    def cached_user(self, user_id):
        return dao.get_user_by_id(user_id)

    @lru_cache()
    def cached_operation_results(self, op_id):
        return dao.get_results_for_operation(op_id)

    def clear_cache(self, notify_others=True):
        if notify_others:
            KubeService.notify_pods("/kube/clear_cache")
        else:
            self.cached_operation_group.cache_clear()
            self.cached_dt_group.cache_clear()
            self.cached_visualizers_for_group.cache_clear()
            self.cached_algorithm.cache_clear()
            self.cached_user.cache_clear()
            self.cached_operation_results.cache_clear()


cache = CacheService()
