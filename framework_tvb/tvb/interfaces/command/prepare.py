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
.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

Shortcut for activating the console profile on the console and conveniently
importing interacting with the full framework.

Provides convenient access to framework from the console.

"""

## Select the profile with storage enabled, but without web interface:

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE, try_reload=False)

from tvb.basic.logger.builder import get_logger
from tvb.core.traits import db_events
from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.core.services.flow_service import FlowService
from tvb.core.services.operation_service import OperationService

# Hook DB events (like prepare json attributes on traited DataTypes):
db_events.attach_db_events()

LOG = get_logger(__name__)
