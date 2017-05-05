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
Created on Nov 1, 2011

.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.core.entities.storage import dao
from tvb.core.adapters.abcremover import ABCRemover
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.time_series import TimeSeriesRegion
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes.graph import ConnectivityMeasure
from tvb.core.services.exceptions import RemoveDataTypeException


class ConnectivityRemover(ABCRemover):
    """
    Connectivity specific validations at remove time.
    """
    
    def remove_datatype(self, skip_validation=False):
        """
        Called when a Connectivity is to be removed.
        """
        if not skip_validation:
            associated_ts = dao.get_generic_entity(TimeSeriesRegion, self.handled_datatype.gid, '_connectivity')
            associated_rm = dao.get_generic_entity(RegionMapping, self.handled_datatype.gid, '_connectivity')
            associated_stim = dao.get_generic_entity(StimuliRegion, self.handled_datatype.gid, '_connectivity')
            associated_mes = dao.get_generic_entity(ConnectivityMeasure, self.handled_datatype.gid, '_connectivity')
            msg = "Connectivity cannot be removed as it is used by at least one "

            if len(associated_ts) > 0:
                raise RemoveDataTypeException(msg + " TimeSeriesRegion.")
            if len(associated_rm) > 0:
                raise RemoveDataTypeException(msg + " RegionMapping.")
            if len(associated_stim) > 0:
                raise RemoveDataTypeException(msg + " StimuliRegion.")
            if len(associated_mes) > 0:
                raise RemoveDataTypeException(msg + " ConnectivityMeasure.")

        #### Update child Connectivities, if any.
        child_conns = dao.get_generic_entity(Connectivity, self.handled_datatype.gid, '_parent_connectivity')
        
        if len(child_conns) > 0:
            for one_conn in child_conns[1:]:
                one_conn.parent_connectivity = child_conns[0].gid
            if child_conns and child_conns[0]:
                child_conns[0].parent_connectivity = self.handled_datatype.parent_connectivity
            for one_child in child_conns:
                dao.store_entity(one_child)
        ABCRemover.remove_datatype(self, skip_validation)
        
        