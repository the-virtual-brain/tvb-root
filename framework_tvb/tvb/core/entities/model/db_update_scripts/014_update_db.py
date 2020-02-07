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
Change of DB structure from TVB version 1.3 to 1.3.1

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>

"""

from sqlalchemy import Column, Float, Integer
from sqlalchemy.sql import text
from migrate.changeset.schema import create_column, drop_column, alter_column
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER
from tvb.basic.logger.builder import get_logger

meta = model.Base.metadata
LOGGER = get_logger(__name__)

COLUMN_N1 = Column('used_disk_space', Float)
COLUMN_N2 = Column('disk_size', Integer)

COLUMN_N3_OLD = Column('result_disk_size', Integer)
COLUMN_N3_NEW = Column('estimated_disk_size', Integer)



def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    meta.bind = migrate_engine

    table = meta.tables['USERS']
    drop_column(COLUMN_N1, table)
    table = meta.tables['BURST_CONFIGURATIONS']
    drop_column(COLUMN_N2, table)
    table = meta.tables['OPERATIONS']
    alter_column(COLUMN_N3_OLD, table=table, name=COLUMN_N3_NEW.name)

    try:
        meta.bind = migrate_engine
        session = SA_SESSIONMAKER()
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.region_mapping' WHERE "type" = 'RegionMapping' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.local_connectivity' WHERE "type" = 'LocalConnectivity' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.cortex' WHERE "type" = 'Cortex' """))
        session.commit()
        session.close()

    except Exception:
        LOGGER.exception("Cold not update datatypes")
        raise



def downgrade(migrate_engine):
    """Operations to reverse the above upgrade go here."""
    meta.bind = migrate_engine

    table = meta.tables['USERS']
    create_column(COLUMN_N1, table)
    table = meta.tables['BURST_CONFIGURATIONS']
    create_column(COLUMN_N2, table)
    table = meta.tables['OPERATIONS']
    alter_column(COLUMN_N3_NEW, table=table, name=COLUMN_N3_OLD.name)

    try:
        meta.bind = migrate_engine
        session = SA_SESSIONMAKER()
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.surfaces' WHERE "type" = 'RegionMapping' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.surfaces' WHERE "type" = 'LocalConnectivity' """))
        session.execute(text("""UPDATE "DATA_TYPES" SET module='tvb.datatypes.surfaces' WHERE "type" = 'Cortex' """))
        session.commit()
        session.close()
    except Exception:
        LOGGER.exception("Cold not update datatypes")
        raise

