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
Revision ID: 32d4bf9f8def
Create Date: 2022-11-04

"""
from alembic import op
from sqlalchemy import Column, Boolean, Integer, String
from tvb.core.neotraits.db import Base

# revision identifiers, used by Alembic.
revision = '32d4bf9f8def'
down_revision = '32d4bf9f8cab'

conn = op.get_bind()


def upgrade():
    # Get tables
    tables = Base.metadata.tables

    new_column_1 = Column('disable_imports', Boolean, default=False)
    op.add_column('PROJECTS', new_column_1)
    new_column_2 = Column('max_operation_size', Integer, default=None)
    op.add_column('PROJECTS', new_column_2)

    existent_column = Column('name', String)

    projects_table = tables['PROJECTS']
    conn.execute(projects_table.update().values({"disable_imports": False}))
    conn.execute(projects_table.update().values({"disable_imports": True, "max_operation_size": 1500}).where(
        existent_column == 'Default_Project'))

    conn.execute('COMMIT')


def downgrade():
    """
    Downgrade currently not supported
    """
    pass
