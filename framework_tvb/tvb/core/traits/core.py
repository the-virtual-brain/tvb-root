# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: marmaduke <mw@eml.cc>
"""

import re
import warnings
from tvb.basic.logger.builder import get_logger


LOG = get_logger(__name__)

def compute_table_name(class_name):
    """
    Given a class name compute the name of the corresponding SQL table.
    """
    tablename = 'MAPPED' + re.sub('((?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z]))', '_', class_name).upper()
    if tablename.count('MAPPED_') > 1:
        tablename = tablename.replace('MAPPED_', '', 1)

    # Preserve table names from when diamond datatypes, to avoid DB update scripts
    if not tablename.endswith("_DATA"):
        tablename = tablename + '_DATA'
    return tablename


def log_warnings(message, category, filename, lineno, file=None, line=None):
    LOG.warning("%s -- %s " % (category, message))
    LOG.debug("%s : %d " % (filename, lineno))
    if file is not None or line is not None:
        LOG.debug("%s : %s " % (file, line))


## Disable SqlAlchemy recent warnings from appearing in the console, and make them respect logging strategy in TVB
## We have them generated from DeclarativeMetaType.__new__
warnings.showwarning = log_warnings
