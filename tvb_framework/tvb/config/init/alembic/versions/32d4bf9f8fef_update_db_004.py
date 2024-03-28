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
Revision ID: 32d4bf9f8fef
Create Date: 2024-03-28

"""
from alembic import op
from sqlalchemy import Column, String

# revision identifiers, used by Alembic.
revision = '32d4bf9f8gaf'
down_revision = '32d4bf9f8def'

conn = op.get_bind()


def upgrade():
    new_column_1 = Column('host_ip', String, default="")
    op.add_column('OPERATION_PROCESS_IDENTIFIERS', new_column_1)


def downgrade():
    """
    Downgrade currently not supported
    """
    pass
