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
