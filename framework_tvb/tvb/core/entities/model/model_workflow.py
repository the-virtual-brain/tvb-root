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
Here we define entities related to workflows and portlets.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import datetime
from sqlalchemy import Integer, String, Column, DateTime
from tvb.core.neotraits.db import Base


class Portlet(Base):
    """
    Store the Portlet entities. 
    One entity will hold:
    - portlet id;
    - path to the XML file that declares it;
    - a name given in the XML
    - an unique identifier (also from XML)
    - last date of introspection.
    """
    __tablename__ = 'PORTLETS'

    id = Column(Integer, primary_key=True)
    algorithm_identifier = Column(String)
    xml_path = Column(String)
    name = Column(String)
    last_introspection_check = Column(DateTime)


    def __init__(self, algorithm_identifier, xml_path, name="TVB Portlet"):
        self.algorithm_identifier = algorithm_identifier
        self.xml_path = xml_path
        self.name = name
        self.last_introspection_check = datetime.datetime.now()
