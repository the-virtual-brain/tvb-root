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
Demo script on how to load a TVB DataType by Id and modify metadata

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

if __name__ == "__main__":
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

import datetime
from tvb.core.entities.storage import dao
from tvb.core.traits.types_mapped import SparseMatrix
from tvb.datatypes.local_connectivity import LocalConnectivity


def update_localconnectivity_metadata(dt_id):
    dt = dao.get_generic_entity(LocalConnectivity, dt_id)[0]

    mtx = dt.matrix
    info_dict = SparseMatrix.extract_sparse_matrix_metadata(mtx)

    data_group_path = SparseMatrix.ROOT_PATH + 'matrix'
    dt.set_metadata(info_dict, '', True, data_group_path)



def update_dt(dt_id, new_create_date):
    dt = dao.get_datatype_by_id(dt_id)
    dt.create_date = new_create_date
    dao.store_entity(dt)
    # Update MetaData in H5 as well.
    dt = dao.get_datatype_by_gid(dt.gid)
    dt.persist_full_metadata()



if __name__ == "__main__":
    update_localconnectivity_metadata(7)
    update_dt(35, datetime.datetime(2014, 9, 29, 14, 17, 50))
    update_dt(36, datetime.datetime(2014, 9, 29, 14, 17, 52))