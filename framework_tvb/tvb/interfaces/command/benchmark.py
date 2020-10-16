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
This module is used to measure simulation performance with the Command Profile.
Some standardized simulations are run and a report is generated in the console output.
"""

import tvb_data
from os import path
from tvb.simulator.coupling import HyperbolicTangent
from tvb.simulator.integrators import HeunDeterministic
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.interfaces.command.lab import *
from tvb.simulator.models import ModelsEnum


def _fire_simulation(project_id, **kwargs):
    # Prepare a Simulator instance with defaults and configure it to use the previously loaded Connectivity
    simulator = Simulator(**kwargs)
    launched_operation = fire_simulation(project_id, simulator)

    # wait for the operation to finish
    while not launched_operation.has_finished:
        sleep(5)
        launched_operation = dao.get_operation_by_id(launched_operation.id)

    if launched_operation.status != STATUS_FINISHED:
        raise Exception('simulation failed: ' + launched_operation.additional_info)

    return launched_operation


def _create_bench_project():
    prj = new_project("benchmark_project_ %s" % datetime.now())
    data_dir = path.abspath(path.dirname(tvb_data.__file__))
    zip_path = path.join(data_dir, 'connectivity', 'connectivity_68.zip')
    import_conn_zip(prj.id, zip_path)
    zip_path = path.join(data_dir, 'connectivity', 'connectivity_96.zip')
    import_conn_zip(prj.id, zip_path)
    zip_path = path.join(data_dir, 'connectivity', 'connectivity_192.zip')
    import_conn_zip(prj.id, zip_path)

    conn68 = dao.get_generic_entity(ConnectivityIndex, 68, "number_of_regions")[0]
    conn68 = h5.load_from_index(conn68)
    conn96 = dao.get_generic_entity(ConnectivityIndex, 96, "number_of_regions")[0]
    conn96 = h5.load_from_index(conn96)
    conn190 = dao.get_generic_entity(ConnectivityIndex, 192, "number_of_regions")[0]
    conn190 = h5.load_from_index(conn190)
    return prj, [conn68, conn96, conn190]


HEADER = """
+------------------------+--------+-------+-----------+---------+-----------+
|      Results                                                              |
+------------------------+--------+-------+-----------+---------+-----------+
|        Model           | Sim.   | Nodes |Conduction | time    | Execution |
|                        | Length |       |speed      | step    | time      |
+------------------------+--------+-------+-----------+---------+-----------+
|                        |    (ms)|       |    (mm/ms)|     (ms)| min:sec   |
+========================+========+=======+===========+=========+===========+"""


class Bench(object):
    LINE = HEADER.splitlines()[1]
    COLW = [len(col) - 2 for col in LINE.split('+')[1:-1]]  # the widths of columns based on the first header line
    FS = ' | '.join('%' + str(cw) + 's' for cw in COLW)  # builds a format string like  "| %6s | %6s | %6s ... "
    FS = '| ' + FS + ' |'

    def __init__(self, model_kws, connectivities, conductions, int_dts, sim_lengths):
        self.model_kws = model_kws
        self.connectivities = connectivities
        self.conductions = conductions
        self.int_dts = int_dts
        self.sim_lengths = sim_lengths
        self.running_times = []

    def run(self, project_id):
        for model_kw in self.model_kws:
            for conn in self.connectivities:
                for length in self.sim_lengths:
                    for dt in self.int_dts:
                        for conduction in self.conductions:
                            launch_args = model_kw.copy()
                            launch_args.update({
                                "connectivity": conn,
                                "simulation_length": length,
                                "integrator": HeunDeterministic(dt=dt),
                                "conduction_speed": conduction,
                            })
                            operation = _fire_simulation(project_id, **launch_args)
                            self.running_times.append(operation.completion_date - operation.start_date)

    def report(self):
        i = 0
        print(HEADER)
        # use the same iteration order as run_benchmark to interpret running_times
        for model_kw in self.model_kws:
            for conn in self.connectivities:
                for length in self.sim_lengths:
                    for dt in self.int_dts:
                        for conduction in self.conductions:
                            timestr = str(self.running_times[i])[2:-5]
                            print(self.FS % (model_kw['model'].__class__.__name__, length,
                                             conn.number_of_regions, conduction, dt, timestr))
                            print(self.LINE)
                            i += 1


def main():
    """
    Launches a set of standardized simulations and prints a report of their running time.
    Creates a new project for these simulations.
    """
    prj, connectivities = _create_bench_project()

    g2d_epi = Bench(
        model_kws=[
            {"model": ModelsEnum.GENERIC_2D_OSCILLATOR.get_class()()},
            {"model": ModelsEnum.EPILEPTOR.get_class()()},
        ],
        connectivities=connectivities,
        conductions=[30.0, 3.0],
        int_dts=[0.1, 0.05],
        sim_lengths=[1000],
    )

    larter = Bench(
        model_kws=[
            {"model": ModelsEnum.LARTER_BREAKSPEAR.get_class()(), "coupling": HyperbolicTangent()},
        ],
        connectivities=connectivities,
        conductions=[10.0],
        int_dts=[0.2, 0.1],
        sim_lengths=[10000]
    )

    print('Generic2dOscillator and Epileptor')
    g2d_epi.run(prj.id)
    g2d_epi.report()

    print('LarterBreakspear')
    larter.run(prj.id)
    larter.report()


if __name__ == "__main__":
    main()
