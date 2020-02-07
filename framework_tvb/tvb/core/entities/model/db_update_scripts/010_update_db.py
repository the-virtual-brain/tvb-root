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
Change of DB structure from TVB version 1.1.2 to 1.1.3

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Andrei Mihai <mihai.andrei@codemart.ro>
"""

from sqlalchemy import Column, Integer, text
from migrate.changeset.schema import create_column, drop_column
from tvb.basic.logger.builder import get_logger
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER


meta = model.Base.metadata

COL_NR_OF_CONNECTIONS = Column('_number_of_connections', Integer)

LOGGER = get_logger(__name__)


def upgrade(migrate_engine):
    """
    Upgrade operations go here.
    Don't create your own engine; bind migrate_engine to your metadata.
    """
    try:
        meta.bind = migrate_engine
        table = meta.tables['MAPPED_CONNECTIVITY_DATA']
        create_column(COL_NR_OF_CONNECTIONS, table)

        remove_visualizer_references()

    except Exception:
        LOGGER.exception("Cold not create new column required by the update")
        raise


def downgrade(migrate_engine):
    """
    Operations to reverse the above upgrade go here.
    """
    try:
        meta.bind = migrate_engine
        table = meta.tables['MAPPED_CONNECTIVITY_DATA']
        drop_column(COL_NR_OF_CONNECTIONS, table)
    except Exception:
        LOGGER.warning("Cold not remove column as required by the downgrade")
        raise



def remove_visualizer_references():
    """
    As we removed an algorithm, remove left-overs.
    """

    LOGGER.info("Starting to remove references towards old viewer ....")

    session = SA_SESSIONMAKER()
    try:
        session.execute(text(
            """DELETE FROM "OPERATIONS" WHERE fk_from_algo IN
               (SELECT A.id FROM "ALGORITHMS" A, "ALGORITHM_GROUPS" AG
               WHERE  A.fk_algo_group = AG.id AND module = 'tvb.adapters.visualizers.cross_correlation'
                      AND classname = 'PearsonCorrelationCoefficientVisualizer');"""))

        session.execute(text(
            """DELETE FROM "WORKFLOW_VIEW_STEPS" WHERE fk_algorithm IN
               (SELECT A.id FROM "ALGORITHMS" A, "ALGORITHM_GROUPS" AG
               WHERE  A.fk_algo_group = AG.id AND module = 'tvb.adapters.visualizers.cross_correlation'
                      AND classname = 'PearsonCorrelationCoefficientVisualizer');"""))
        session.commit()
    except Exception as excep:
        LOGGER.exception(excep)
    finally:
        session.close()

    LOGGER.info("References removed.")