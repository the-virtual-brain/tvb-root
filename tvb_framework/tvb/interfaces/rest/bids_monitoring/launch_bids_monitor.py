# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import sys
from tvb.interfaces.rest.bids_monitoring.bids_data_builder import BIDSDataBuilder
from tvb.interfaces.rest.bids_monitoring.bids_dir_monitor import BIDSDirWatcher
from tvb.adapters.uploaders.bids_importer import BIDSImporter


BIDS_UPLOAD_CONTENT = BIDSImporter.NET_TOKEN
BIDS_DIR = ""


def get_bids_dir():
    if len(sys.argv) > 0:
        for arg in sys.argv:
            if arg.startswith('--bids-dir'):
                BIDS_DIR = arg.split('=')[1]
    return BIDS_DIR


def build_bids_dataset():
    # A sample code to how to build BIDS dataset for each datatype using BIDSDataBuilder

    bids_data_builder = BIDSDataBuilder(BIDS_UPLOAD_CONTENT, BIDS_DIR)
    zip_file_location = bids_data_builder.create_dataset_subjects()
    print(zip_file_location)


def monitor_dir():
    # A sample code to how to monitor a directory using BIDSDirWatcher
    # and build BIDS dataset whenever new files are added

    # Set IMPORT_DATA_IN_TVB to True to enable importing dataset into TVB
    bids_dir_watcher = BIDSDirWatcher(
        DIRECTORY_TO_WATCH=BIDS_DIR,
        UPLOAD_TRIGGER_INTERVAL=20,
        IMPORT_DATA_IN_TVB=True
    )
    bids_dir_watcher.init_watcher()


if __name__ == '__main__':

    # If bids dir is provided as command line args
    BIDS_DIR = get_bids_dir()

    monitor_dir()

    # build_bids_dataset()
