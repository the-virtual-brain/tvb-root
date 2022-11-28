# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
