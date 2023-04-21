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
Find a TS in current project (by Subject) and later run an analyzer on it.

__main__ will contain the code.

.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
"""

from tvb.adapters.analyzers.fourier_adapter import FourierAdapter, FFTAdapterModel
from tvb.adapters.datatypes.db.spectral import FourierSpectrumIndex
from tvb.adapters.datatypes.db.time_series import TimeSeriesRegionIndex
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.transient.structure_entities import DataTypeMetaData
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model.model_operation import STATUS_FINISHED
from tvb.core.entities.storage import dao
from time import sleep
from tvb.interfaces.command.lab import *


def run_analyzer():
    log = get_logger(__name__)

    # This ID of a project needs to exists in DB, and it can be taken from the WebInterface:
    project = dao.get_project_by_id(1)
    # Prepare the Adapter
    adapter_instance = ABCAdapter.build_adapter_from_class(FourierAdapter)

    # Prepare the input algorithms as if they were coming from web UI submit:
    time_series = dao.get_generic_entity(TimeSeriesRegionIndex, DataTypeMetaData.DEFAULT_SUBJECT, "subject")
    if len(time_series) < 1:
        log.error("We could not find a compatible TimeSeries Datatype!")

    fourier_model = FFTAdapterModel()
    fourier_model.time_series = time_series[0].gid
    fourier_model.window_function = 'hamming'
    fourier_model.segment_length = 100

    # launch an operation and have the results stored both in DB and on disk
    launched_operation = OperationService().fire_operation(adapter_instance, project.administrator,
                                                           project.id, view_model=fourier_model)

    # wait for the operation to finish
    while not launched_operation.has_finished:
        sleep(5)
        launched_operation = dao.get_operation_by_id(launched_operation.id)

    if launched_operation.status == STATUS_FINISHED:
        fourier_spectrum = dao.get_generic_entity(FourierSpectrumIndex, launched_operation.id, "fk_from_operation")[0]
        log.info("Fourier Spectrum result is: %s " % fourier_spectrum)
    else:
        log.warning("Operation ended with problems [%s]: [%s]" % (launched_operation.status,
                                                                  launched_operation.additional_info))


if __name__ == "__main__":
    run_analyzer()
