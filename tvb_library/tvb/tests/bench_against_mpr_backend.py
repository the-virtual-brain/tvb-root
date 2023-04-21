# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
This file benchmarks the default vs mpr backend for simulation in the scientific library.

"""

import numpy
import time
from tvb.simulator.lab import *
from tvb.simulator.backend import ReferenceBackend
from tvb.simulator.backend.nb_mpr import NbMPRBackend

HEADER = """
+------------------+------------------------+--------+-------+-----------+---------+-----------+
|      Results                                                                                 |
+------------------+------------------------+--------+-------+-----------+---------+-----------+
|     Backend      |        Model           | Sim.   | Nodes |Conduction | time    | Execution |
|                  |                        | Length |       |speed      | step    | time      |
+------------------+------------------------+--------+-------+-----------+---------+-----------+
|                  |                        |    (ms)|       |    (mm/ms)|     (ms)| min:sec   |
+==================+========================+========+=======+===========+=========+===========+"""


class Bench(object):
    LINE = HEADER.splitlines()[1]
    COLW = [len(col) - 2 for col in LINE.split('+')[1:-1]]  # the widths of columns based on the first header line
    FS = ' | '.join('%' + str(cw) + 's' for cw in COLW)  # builds a format string like  "| %6s | %6s | %6s ... "
    FS = '| ' + FS + ' |'

    def __init__(self, backends, models, connectivities, conductions, int_dts, sim_lengths, coupling=None):
        self.backends = backends
        self.models = models
        self.connectivities = connectivities
        self.conductions = conductions
        self.int_dts = int_dts
        self.sim_lengths = sim_lengths
        self.running_times = []
        self.coupling = coupling

    def run(self):
        for backend in self.backends:
            for model in self.models:
                for conn in self.connectivities:
                    for length in self.sim_lengths:
                        for dt in self.int_dts:
                            for conduction in self.conductions:
                                sim = simulator.Simulator()
                                sim.connectivity = conn
                                sim.model = model
                                sim.integrator = integrators.HeunStochastic()
                                sim.integrator.dt = dt
                                sim.conduction_speed = conduction
                                sim.simulation_length = length

                                if self.coupling:
                                    sim.coupling = self.coupling

                                sim.monitors[0].period = 0.1

                                sim.configure()

                                start_time = time.time()
                                if backend is NbMPRBackend:
                                    backend().run_sim(sim, simulation_length=sim.simulation_length)
                                else:
                                    sim.run()
                                end_time = time.time()

                                self.running_times.append(end_time - start_time)

    def report(self):
        i = 0
        print(HEADER)
        # use the same iteration order as run_benchmark to interpret running_times
        for backend in self.backends:
            for model in self.models:
                for conn in self.connectivities:
                    for length in self.sim_lengths:
                        for dt in self.int_dts:
                            for conduction in self.conductions:
                                timestr = time.strftime('%H:%M:%S', time.gmtime(self.running_times[i]))
                                print(self.FS % (backend.__name__, type(model).__name__, length,
                                                 conn.number_of_regions, conduction, dt, timestr))
                                print(self.LINE)
                                i += 1


def read_connectivity(file_path):
    weights = numpy.loadtxt(file_path)

    magic_number = 124538.470647693
    weights_orig = weights / magic_number
    conn = connectivity.Connectivity(
        weights=weights_orig,
        region_labels=numpy.array(numpy.zeros(numpy.shape(weights_orig)[0]), dtype='<U128'),
        tract_lengths=numpy.zeros(numpy.shape(weights_orig)),
        areas=numpy.zeros(numpy.shape(weights_orig)[0]),
        speed=numpy.array(numpy.Inf, dtype=float),
        centres=numpy.zeros(numpy.shape(weights_orig)[0]))  # default 76 regions
    conn.configure()

    return conn


def main():
    """
    Launches a simulation and prints a report of their running time.
    """

    conn = connectivity.Connectivity.from_file()  # read_connectivity("SC_Schaefer7NW100p_nolog10.txt")
    connectivities = [conn]

    montbrio = Bench(
        backends=[ReferenceBackend, NbMPRBackend],
        models=[models.ModelsEnum.MONTBRIO_PAZO_ROXIN.get_class()()],
        connectivities=connectivities,
        conductions=[numpy.Inf],
        int_dts=[0.005],
        sim_lengths=[1000],
    )

    print('MONTBRIO_PAZO_ROXIN')
    montbrio.run()
    montbrio.report()


if __name__ == "__main__":
    main()
