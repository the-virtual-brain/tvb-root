# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
Change of DB structure from TVB version 1.0.1 to TVB 1.0.2.

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import sqlalchemy
from sqlalchemy import Column, Integer, String
from sqlalchemy.sql import text
from migrate.changeset.schema import create_column, drop_column
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER

meta = model.Base.metadata
COL_1 = Column('_length_1d', Integer)
COL_2 = Column('_length_2d', Integer)
COL_3 = Column('_length_3d', Integer)
COL_4 = Column('_length_4d', Integer)
COL_5 = Column('_labels_dimensions', String, default="{}")
COL_6 = Column('_labels_ordering', String, default="[]")
COL_7 = Column('_unidirectional', Integer)

TABLE_RENAMES = [('BURST_CONFIGURATION', 'BURST_CONFIGURATIONS'),
                 ('CONNECTIVITY_SELECTION', 'CONNECTIVITY_SELECTIONS'),
                 ('USER_TO_PROJECT', 'USERS_TO_PROJECTS'),
                 ('OPERATION_PROCESS_IDENTIFIER', 'OPERATION_PROCESS_IDENTIFIERS')]



def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    meta.bind = migrate_engine
    table1 = meta.tables['MAPPED_TIME_SERIES_DATA']

    create_column(COL_1, table1)
    create_column(COL_2, table1)
    create_column(COL_3, table1)
    create_column(COL_4, table1)
    create_column(COL_5, table1)

    session = SA_SESSIONMAKER()
    try:
        # We have a database that supports renaming columns. This way we save data from old timeseries.
        session.execute(text("ALTER TABLE \"MAPPED_TIME_SERIES_DATA\" "
                             "RENAME COLUMN _dim_labels to _labels_ordering"))
        session.execute(text("ALTER TABLE \"MAPPED_CROSS_CORRELATION_DATA\" "
                             "RENAME COLUMN _dim_labels to _labels_ordering"))
    except sqlalchemy.exc.OperationalError:
        # We have a database like sqlite. Just create a new column, we're gonna miss old data in this case.
        session.execute(text("ALTER TABLE \"MAPPED_TIME_SERIES_DATA\" "
                             "ADD COLUMN _labels_ordering VARYING CHARACTER(255)"))
        session.execute(text("ALTER TABLE \"MAPPED_CROSS_CORRELATION_DATA\" "
                             "ADD COLUMN _labels_ordering VARYING CHARACTER(255)"))

    session.execute(text("DROP TABLE \"MAPPED_PSI_TABLE_DATA\""))
    session.execute(text("DROP TABLE \"MAPPED_NERF_TABLE_DATA\""))
    session.execute(text("DROP TABLE \"MAPPED_LOOK_UP_TABLES_DATA\""))
    session.commit()
    session.close()

    table2 = meta.tables['MAPPED_CONNECTIVITY_DATA']
    create_column(COL_7, table2)

    for mapping in TABLE_RENAMES:
        session = SA_SESSIONMAKER()
        session.execute(text("ALTER TABLE \"%s\" RENAME TO \"%s\"" % (mapping[0], mapping[1])))
        session.commit()
        session.close()



def downgrade(migrate_engine):
    """Operations to reverse the above upgrade go here."""
    meta.bind = migrate_engine

    table1 = meta.tables['MAPPED_TIME_SERIES_DATA']
    drop_column(COL_1, table1)
    drop_column(COL_2, table1)
    drop_column(COL_3, table1)
    drop_column(COL_4, table1)

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("ALTER TABLE \"MAPPED_TIME_SERIES_DATA\" "
                             "RENAME COLUMN _labels_ordering to _dim_labels"))
        session.execute(text("ALTER TABLE \"MAPPED_CROSS_CORRELATION_DATA\" "
                             "RENAME COLUMN _labels_ordering to _dim_labels"))

    except sqlalchemy.exc.OperationalError:
        session.execute(text("ALTER TABLE \"MAPPED_TIME_SERIES_DATA\" "
                             "ADD COLUMN _dim_labels VARYING CHARACTER(255)"))
        session.execute(text("ALTER TABLE \"MAPPED_CROSS_CORRELATION_DATA\" "
                             "ADD COLUMN _dim_labels VARYING CHARACTER(255)"))
    session.commit()
    session.close()

    table2 = meta.tables['MAPPED_CONNECTIVITY_DATA']
    drop_column(COL_7, table2)

    for mapping in TABLE_RENAMES:
        session = SA_SESSIONMAKER()
        session.execute(text("ALTER TABLE \"%s\" RENAME TO \"%s\"" % (mapping[1], mapping[0])))
        session.commit()
        session.close()
        
        
        