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

"""
Demo script on how to filter datatypes and later export them.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

import os
import shutil
from sys import argv
from datetime import datetime
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.core.entities.filters.chain import FilterChain
from tvb.core.entities.storage import dao
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.neocom import h5

def _retrieve_entities_by_filters(kind, project_id, filters):
    named_tuple_array, counter = dao.get_values_of_datatype(project_id, kind, filters)
    print("Found " + str(counter) + " entities of type " + str(kind))

    result = []
    for named_tuple in named_tuple_array:
        dt_id = named_tuple[0]
        result.append(dao.get_generic_entity(kind, dt_id)[0])

    return result


def search_and_export_ts(project_id, export_folder=os.path.join("~", "TVB")):
    # This is the simplest filter you could write: filter and entity by Subject
    filter_connectivity = FilterChain(fields=[FilterChain.datatype + '.subject'],
                                      operations=["=="],
                                      values=[DataTypeMetaData.DEFAULT_SUBJECT])

    connectivities = _retrieve_entities_by_filters(ConnectivityIndex, project_id, filter_connectivity)

    # A more complex filter: by linked entity (connectivity), saompling, operation date:
    filter_timeseries = FilterChain(fields=[FilterChain.datatype + '.fk_connectivity_gid',
                                            FilterChain.datatype + '.sample_period',
                                            FilterChain.operation + '.create_date'
                                            ],
                                    operations=["==", ">=", "<="],
                                    values=[connectivities[0].gid,
                                            0,
                                            datetime.now()
                                            ]
                                    )

    # If you want to filter another type of TS, change the kind class bellow,
    # instead of TimeSeriesRegion use TimeSeriesEEG, or TimeSeriesSurface, etc.
    timeseries = _retrieve_entities_by_filters(TimeSeriesRegionIndex, project_id, filter_timeseries)

    for ts in timeseries:
        print("=============================")
        print(ts.summary_info)
        storage_h5 = h5.path_for_stored_index(ts)
        print(" Original file: " + str(storage_h5))
        destination_folder = os.path.expanduser(export_folder)
        shutil.copy2(storage_h5, destination_folder)
        print("File {0} exported in {1}".format(storage_h5, destination_folder))


if __name__ == '__main__':
    from tvb.interfaces.command.lab import *

    if len(argv) < 2:
        PROJECT_ID = 1
    else:
        PROJECT_ID = int(argv[1])

    print("We will try to search datatypes in project with ID:" + str(PROJECT_ID))

    search_and_export_ts(PROJECT_ID)
