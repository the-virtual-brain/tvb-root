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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
# Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
# Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Change of DB structure from TVB version 1.4.1 to TVB 1.4.2

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from sqlalchemy.sql import text
from tvb.basic.logger.builder import get_logger
from tvb.core.entities import model
from tvb.core.entities.storage import SA_SESSIONMAKER


meta = model.Base.metadata

LOGGER = get_logger(__name__)

SQL_CREATE_TABLE = """
CREATE TABLE "ALGORITHMS2"
(
  id INTEGER NOT NULL,
  module VARCHAR,
  classname VARCHAR,
  fk_category INTEGER,
  group_name VARCHAR,
  group_description VARCHAR,
  displayname VARCHAR,
  description VARCHAR,
  subsection_name VARCHAR,
  required_datatype VARCHAR,
  datatype_filter VARCHAR,
  parameter_name VARCHAR,
  outputlist VARCHAR,
  last_introspection_check timestamp without time zone,
  removed BOOLEAN,
  PRIMARY KEY(id),
  FOREIGN KEY (fk_category) REFERENCES "ALGORITHM_CATEGORIES" (id) ON DELETE CASCADE
);
         """

SQL_INSERT = """
INSERT INTO "ALGORITHMS2"
SELECT A.id, G.module, G.classname, G.fk_category, null, null ,A.name, A.description,
        G.subsection_name, A.required_datatype, A.datatype_filter, A.parameter_name, A.outputlist, null, 0
FROM "ALGORITHMS" A, "ALGORITHM_GROUPS" G
WHERE A.fk_algo_group = G.id;
               """


def upgrade(migrate_engine):
    """
    Migrate can not handle these (adding new columns with FK, etc.) so we need to find a more solid solution
    """
    meta.bind = migrate_engine

    session = SA_SESSIONMAKER()
    try:
        session.execute(text("PRAGMA foreign_keys=OFF"))
        session.execute(text(SQL_CREATE_TABLE))
        session.execute(text(SQL_INSERT))
        #session.execute(text("""DROP TABLE "ALGORITHM_GROUPS";"""))
        #session.execute(text("""DROP TABLE "ALGORITHMS";"""))
        session.execute(text("""ALTER TABLE "ALGORITHMS" RENAME TO "ALGORITHMS_old";"""))
        session.execute(text("""ALTER TABLE "ALGORITHMS2" RENAME TO "ALGORITHMS";"""))
        session.execute(text("PRAGMA foreign_keys=OFF"))
        session.commit()

    except Exception, excep:
        LOGGER.exception(excep)
    finally:
        session.close()



def downgrade(_):
    """
    Downgrade currently not supported
    """
    pass
