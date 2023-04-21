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

import sys
from tvb.basic.logger.builder import get_logger
from tvb.interfaces.rest.bids_monitor.bids_dir_monitor import BIDSDirWatcher

logger = get_logger(__name__)


def get_bids_dir():
    bids_dir = None
    if len(sys.argv) > 0:
        for arg in sys.argv:
            if arg.startswith('--bids-dir'):
                bids_dir = arg.split('=')[1]
    return bids_dir


def monitor_dir(bids_dir):
    # A sample code to how to monitor a directory using BIDSDirWatcher
    # and build BIDS dataset whenever new files are added
    logger.info('Starting bids monitor')
    # Set IMPORT_DATA_IN_TVB to True to enable importing dataset into TVB
    bids_dir_watcher = BIDSDirWatcher(
        DIRECTORY_TO_WATCH=bids_dir,
        UPLOAD_TRIGGER_INTERVAL=20,
        IMPORT_DATA_IN_TVB=True
    )
    bids_dir_watcher.init_watcher()


if __name__ == '__main__':
    """
    Receives as arguments the BIDS directory to monitor and the TVB REST server URL
    e.g. python launch_bids_monitor.py --bids-dir=user/doc/BIDS_SAMPLE --rest-url=http://localhost:9090

    """
    monitor_dir(get_bids_dir())
