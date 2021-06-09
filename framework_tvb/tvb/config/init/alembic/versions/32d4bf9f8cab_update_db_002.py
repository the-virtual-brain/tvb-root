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
    new_column_1 = Column('has_valid_time_series', Boolean, default=True)
    op.add_column('DataTypeMatrix', new_column_1)

    new_column_2 = Column('queue_full', Boolean, default=False)
    op.add_column('OPERATIONS', new_column_2)




def downgrade():
    """
    Downgrade currently not supported
    """
    pass
