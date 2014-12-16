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
Change of DB structure from TVB version 1.2.3 to 1.2.4

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>

"""

from sqlalchemy import Column, Float, Integer
from migrate.changeset.schema import create_column, drop_column, alter_column
from tvb.core.entities import model

meta = model.Base.metadata

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


def downgrade(migrate_engine):
    """Operations to reverse the above upgrade go here."""
    meta.bind = migrate_engine

    table = meta.tables['USERS']
    create_column(COLUMN_N1, table)
    table = meta.tables['BURST_CONFIGURATIONS']
    create_column(COLUMN_N2, table)
    table = meta.tables['OPERATIONS']
    alter_column(COLUMN_N3_NEW, table=table, name=COLUMN_N3_OLD.name)

