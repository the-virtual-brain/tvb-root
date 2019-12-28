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
Change of DB structure from TVB version 1.2.1 to 1.2.2

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>

"""

from sqlalchemy import Column, String, Boolean
from migrate.changeset.schema import create_column, drop_column
from tvb.core.entities import model


meta = model.Base.metadata

COLUMN_BURST = Column('_dynamic_ids', String, default="[]")
COLUMN_GROUP = Column("removed", Boolean, default=False)
COLUMN_CATEGORY = Column("removed", Boolean, default=False)
COLUMN_VALID_SURFACE = Column("_valid_for_simulations", Boolean, default=True)


def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    meta.bind = migrate_engine

    table1 = meta.tables['BURST_CONFIGURATIONS']
    create_column(COLUMN_BURST, table1)

    table2 = meta.tables['ALGORITHM_CATEGORIES']
    create_column(COLUMN_CATEGORY, table2)

    table3 = meta.tables['ALGORITHM_GROUPS']
    create_column(COLUMN_GROUP, table3)

    table4 = meta.tables['MAPPED_SURFACE_DATA']
    create_column(COLUMN_VALID_SURFACE, table4)


def downgrade(migrate_engine):
    """Operations to reverse the above upgrade go here."""
    meta.bind = migrate_engine

    table1 = meta.tables['BURST_CONFIGURATIONS']
    drop_column(COLUMN_BURST, table1)

    table2 = meta.tables['ALGORITHM_CATEGORIES']
    drop_column(COLUMN_CATEGORY, table2)

    table3 = meta.tables['ALGORITHM_GROUPS']
    drop_column(COLUMN_GROUP, table3)

    table4 = meta.tables['MAPPED_SURFACE_DATA']
    drop_column(COLUMN_VALID_SURFACE, table4)
