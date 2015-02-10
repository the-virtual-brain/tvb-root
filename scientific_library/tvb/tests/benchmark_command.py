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
This module is used to measure simulation performance. Some standardized simulations are run and a report is generated.
"""

from time import sleep
from datetime import datetime
from os import path
import tvb_data

if __name__ == "__main__":
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.COMMAND_PROFILE)

from tvb.core.entities import model
from tvb.core.entities.storage import dao
from tvb.datatypes.connectivity import Connectivity
from tvb.interfaces.command import lab


def fire_simulation(project_id, **kwargs):
    launched_operation = lab.fire_simulation(project_id, **kwargs)

    # wait for the operation to finish
    while not launched_operation.has_finished:
        sleep(5)
        launched_operation = dao.get_operation_by_id(launched_operation.id)

    if launched_operation.status != model.STATUS_FINISHED:
        raise Exception('simulation failed: ' + launched_operation.additional_info)

    return launched_operation


def run_benchmark(project_id, model_kws, connectivities, conductions, int_dts, sim_lengths):
    running_times = []
    for model_kw in model_kws:
        for conn in connectivities:
            for conduction in conductions:
                for dt in int_dts:
                    for length in sim_lengths:
                        launch_args = model_kw.copy()
                        launch_args.update({
                            "connectivity": conn.gid,
                            "simulation_length": str(length),
                            "integrator": "HeunDeterministic",
                            "integrator_parameters_option_HeunDeterministic_dt": str(dt),
                            "conduction_speed": str(conduction),
                        })
                        operation = fire_simulation(project_id, **launch_args)
                        running_times.append( operation.completion_date - operation.start_date )

    return running_times


def report(running_times, model_kws, connectivities, conductions, int_dts, sim_lengths):
    FS = "| %24s | %8s | %8s | %8s | %8s | %12s |"
    i = 0
    print FS % ("model", "regions", "speed", "dt", "length",  "time" )
    for model_kw in model_kws:
        for conn in connectivities:
            for conduction in conductions:
                for dt in int_dts:
                    for length in sim_lengths:
                        print FS % (model_kw['model'], conn.number_of_regions, conduction, dt, length, running_times[i])
                        i += 1


def main():
    """
    Launches a set of standardized simulations and prints a report of their running time.
    Creates a new project for these simulations.
    """
    prj = lab.new_project("benchmark_project_ %s" % datetime.now())
    data_dir = path.abspath(path.dirname(tvb_data.__file__))
    zip_path = path.join(data_dir, 'connectivity', 'connectivity_68.zip')
    lab.import_conn_zip(prj.id, zip_path)
    zip_path = path.join(data_dir, 'connectivity', 'connectivity_96.zip')
    lab.import_conn_zip(prj.id, zip_path)
    zip_path = path.join(data_dir, 'connectivity', 'connectivity_190.zip')
    lab.import_conn_zip(prj.id, zip_path)

    conn68 = dao.get_generic_entity(Connectivity, 68, "_number_of_regions")[0]
    conn96 = dao.get_generic_entity(Connectivity, 96, "_number_of_regions")[0]
    conn190 = dao.get_generic_entity(Connectivity, 190, "_number_of_regions")[0]

    print 'Benchmark 1'
    model_kws = [
        {"model": "Generic2dOscillator", },
        {"model": "Epileptor", },
    ]
    connectivities = [conn68, conn96, conn190]
    conductions = [30.0, 3.0]
    int_dts = [0.2, 0.1]
    sim_lengths = [1000]
    times = run_benchmark(prj.id, model_kws, connectivities, conductions, int_dts, sim_lengths)
    report(times, model_kws, connectivities, conductions, int_dts, sim_lengths)

    print 'Paula pdf'
    model_kws = [
        {"model": "LarterBreakspear", },
    ]
    connectivities = [conn68, conn96, conn190]
    conductions = [10.0]
    int_dts = [0.2, 0.1]
    sim_lengths = [10000]

    times = run_benchmark(prj.id, model_kws, connectivities, conductions, int_dts, sim_lengths)
    report(times, model_kws, connectivities, conductions, int_dts, sim_lengths)


if __name__ == "__main__":
    main()
