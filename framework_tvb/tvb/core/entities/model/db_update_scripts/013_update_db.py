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
Change of DB structure from TVB version 1.2.2 to 1.2.3

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>

"""

from sqlalchemy import Column, Float
from sqlalchemy.sql import text
from migrate.changeset.schema import create_column, drop_column
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER

meta = model.Base.metadata

COLUMN_N1 = Column('_edge_mean_length', Float)
COLUMN_N2 = Column('_edge_min_length', Float)
COLUMN_N3 = Column('_edge_max_length', Float)


def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    meta.bind = migrate_engine

    table = meta.tables['MAPPED_SURFACE_DATA']
    create_column(COLUMN_N1, table)
    create_column(COLUMN_N2, table)
    create_column(COLUMN_N3, table)

    session = SA_SESSIONMAKER()
    session.execute(text("""UPDATE "OPERATIONS" SET status='5-FINISHED' WHERE status = '4-FINISHED' """))
    session.commit()
    session.close()


def downgrade(migrate_engine):
    """Operations to reverse the above upgrade go here."""
    meta.bind = migrate_engine

    table = meta.tables['MAPPED_SURFACE_DATA']
    drop_column(COLUMN_N1, table)
    drop_column(COLUMN_N2, table)
    drop_column(COLUMN_N3, table)

    session = SA_SESSIONMAKER()
    session.execute(text("""UPDATE "OPERATIONS" SET status='4-FINISHED' WHERE status = '5-FINISHED' """))
    session.execute(text("""UPDATE "OPERATIONS" SET status='3-STARTED' WHERE status = '4-PENDING' """))
    session.commit()
    session.close()

