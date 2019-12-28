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
Change of DB structure to TVB version 1.1.1

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy import Column, String, Boolean
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.sql import text
from migrate.changeset.schema import create_column, drop_column
from tvb.basic.logger.builder import get_logger
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER

meta = model.Base.metadata

DEFAULT = '{"0": {"hemisphere": "NONE", "vertices": {"end_idx": 45000, "start_idx": 0}, ' \
          '"triangles": {"end_idx": 80000, "start_idx": 0}}}'

COL_1 = Column('_hemisphere_mask', String, default='hemisphere_mask')
COL_2 = Column('_split_slices', String, default=DEFAULT)
COL_3 = Column('_bi_hemispheric', Boolean, default=False)


def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    try:
        meta.bind = migrate_engine
        table1 = meta.tables['MAPPED_SURFACE_DATA']

        create_column(COL_1, table1)
        create_column(COL_2, table1)
        create_column(COL_3, table1)

        try:
            session = SA_SESSIONMAKER()
            session.execute(text("UPDATE \"DATA_TYPES\" SET invalid=1 WHERE exists "
                                 "(SELECT * FROM \"MAPPED_SURFACE_DATA\" WHERE  _number_of_split_slices > 1 "
                                 "and \"DATA_TYPES\".id = \"MAPPED_SURFACE_DATA\".id)"))
            session.commit()
            session.close()
        except ProgrammingError:
            # PostgreSQL
            session = SA_SESSIONMAKER()
            session.execute(text("UPDATE \"DATA_TYPES\" SET invalid=TRUE WHERE exists "
                                 "(SELECT * FROM \"MAPPED_SURFACE_DATA\" WHERE  _number_of_split_slices > 1 "
                                 "and \"DATA_TYPES\".id = \"MAPPED_SURFACE_DATA\".id)"))
            session.commit()
            session.close()

    except Exception:
        logger = get_logger(__name__)
        logger.exception("Cold not create new column required by the update")
        raise


def downgrade(migrate_engine):
    """
    Operations to reverse the above upgrade go here.
    """
    try:
        meta.bind = migrate_engine
        table1 = meta.tables['MAPPED_SURFACE_DATA']

        drop_column(COL_1, table1)
        drop_column(COL_2, table1)
        drop_column(COL_3, table1)

    except Exception:
        logger = get_logger(__name__)
        logger.warning("Cold not remove column as required by the downgrade")
        raise


