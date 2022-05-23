"""
Revision ID: 32d4bf9f8cab
Create Date: 2021-06-03

"""
from alembic import op
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import Column, Boolean
import uuid
import json

from sqlalchemy.sql.schema import UniqueConstraint
from tvb.basic.profile import TvbProfile
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
