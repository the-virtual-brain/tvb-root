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
Revision ID: 32d4bf9f8cab
Create Date: 2021-06-03

"""

from alembic import op
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import Column, Boolean
from tvb.core.neotraits.db import Base
from tvb.basic.logger.builder import get_logger

# revision identifiers, used by Alembic.
revision = '32d4bf9f8cab'
down_revision = 'ec2859bb9114'
conn = op.get_bind()
LOGGER = get_logger(__name__)


def upgrade():
    # Get tables
    inspector = Inspector.from_engine(conn)
    table_names = inspector.get_table_names()
    tables = Base.metadata.tables

    if 'DataTypeMatrix' in table_names:
        new_column_1 = Column('has_valid_time_series', Boolean, default=True)
        op.add_column('DataTypeMatrix', new_column_1)
        datatype_matrix_table = tables['DataTypeMatrix']
        conn.execute(datatype_matrix_table.update().values({"has_valid_time_series": True}))

    new_column_2 = Column('queue_full', Boolean, default=False)
    op.add_column('OPERATIONS', new_column_2)

    operations_table = tables['OPERATIONS']
    conn.execute(operations_table.update().values({"queue_full": False}))
    conn.execute('COMMIT')


def downgrade():
    """
    Downgrade currently not supported
    """
    pass
