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
Change of DB structure from TVB version 1.2 to 1.2.1

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.sql import text
from migrate.changeset.schema import create_column, drop_column
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER


meta = model.Base.metadata

COL_OLD = Column('_unidirectional', Integer)
COL_NEW = Column('_undirected', Integer)
COL_NOSE_CORRECTION = Column('_nose_correction', String)


def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    meta.bind = migrate_engine
    table1 = meta.tables['MAPPED_CONNECTIVITY_DATA']

    create_column(COL_NEW, table1)

    session = SA_SESSIONMAKER()
    session.execute(text("UPDATE \"MAPPED_CONNECTIVITY_DATA\" set _undirected=_unidirectional"))
    session.commit()
    session.close()

    drop_column(COL_OLD, table1)
    drop_column(COL_NOSE_CORRECTION, table1)



def downgrade(migrate_engine):
    """Operations to reverse the above upgrade go here."""
    meta.bind = migrate_engine
    table1 = meta.tables['MAPPED_CONNECTIVITY_DATA']

    create_column(COL_OLD, table1)

    session = SA_SESSIONMAKER()
    session.execute(text("UPDATE \"MAPPED_CONNECTIVITY_DATA\" set _unidirectional=_undirected"))
    session.commit()
    session.close()

    drop_column(COL_NEW, table1)
    create_column(COL_NOSE_CORRECTION, table1)