# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
What:
     Reproduces Figure 17, second column, of Sanz-Leon P., Knock, S. A., Spiegler, A. and Jirsa V.
     Mathematical framework for large-scale brain network modelling in The Virtual Brain.
     Neuroimage, 2014, (in review)

Needs:
     A working installation of tvb

Run:
     python region_deterministic_bnm_wc_b.py -s True -f True

     #Subsequent calls can be made with: 
     python region_deterministic_bnm_wc_b.py -f True

.. author:: Paula Sanz-Leon
"""

import numpy
import argparse
from tvb.simulator.lab import *
import matplotlib.pylab as pylab
from matplotlib.pylab import *

LOG = get_logger(__name__)

pylab.rcParams['figure.figsize'] = 10, 7  # that's default image size for this interactive session
pylab.rcParams.update({'font.size': 22})
pylab.rcParams.update({'axes.linewidth': 3})

parser = argparse.ArgumentParser(description='Reproduce results of Figure XX presented in Sanz-Leon et al 2014')
parser.add_argument('-s', '--sim', help='Run the simulations', default=False)
parser.add_argument('-f', '--fig', help='Plot the figures', default=False)
args = vars(parser.parse_args())

speed = 4.0
simulation_length = 512

idx = ['b0', 'b1', 'b2']
gcs = [0.0, 0.0042, 0.042]

if args['sim']:
    for i in range(3):

        oscilator = models.WilsonCowan(c_ee=numpy.array([13.]), c_ei=numpy.array([4.]),
                                       c_ie=numpy.array([22.]), c_ii=numpy.array([2.]),
                                       tau_e=numpy.array([10.]), tau_i=numpy.array([10.]),
                                       a_e=numpy.array([1.5]), a_i=numpy.array([6.]),
                                       b_e=numpy.array([2.6]), b_i=numpy.array([4.3]),
                                       variables_of_interest=["E", "I"])
        oscilator.state_variable_range["E"] = numpy.array([0.2, 0.3])
        oscilator.state_variable_range["I"] = numpy.array([0.2, 0.3])
        white_matter = connectivity.Connectivity.from_file()
        white_matter.speed = numpy.array([speed])
        # 0, 0.0042, 0.042
        white_matter_coupling = coupling.Linear(a=numpy.array([gcs[i]]))

        # Initialise an Integrator
        heunint = integrators.HeunDeterministic(dt=2 ** -4)

        # Initialise some Monitors with period in physical time
        momo = monitors.Raw()
        mama = monitors.TemporalAverage(period=2 ** -2)

        # Bundle them
        what_to_watch = (momo, mama)

        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim = simulator.Simulator(model=oscilator, connectivity=white_matter,
                                  coupling=white_matter_coupling,
                                  integrator=heunint, monitors=what_to_watch)
        sim.configure()

        LOG.info("Starting simulation...")
        # Perform the simulation
        raw_data = []
        raw_time = []
        tavg_data = []
        tavg_time = []

        for raw, tavg in sim(simulation_length=simulation_length):
            if not raw is None:
                raw_time.append(raw[0])
                raw_data.append(raw[1])
            if not tavg is None:
                tavg_time.append(tavg[0])
                tavg_data.append(tavg[1])

        LOG.info("Finished simulation.")

        # Make the lists numpy.arrays for easier use.
        RAW = numpy.asarray(raw_data)
        TAVG = numpy.asarray(tavg_data)

        numpy.save('region_deterministic_bnm_wc_raw_' + idx[i] + '.npy', RAW)
        numpy.save('region_deterministic_bnm_wc_tavg_' + idx[i] + '.npy', TAVG)
        numpy.save('region_deterministic_bnm_wc_rawtime_' + idx[i] + '.npy', raw_time)
        numpy.save('region_deterministic_bnm_wc_tavgtime_' + idx[i] + '.npy', tavg_time)

if args['fig']:
    for i in range(3):
        RAW = numpy.load('region_deterministic_bnm_wc_raw_' + idx[i] + '.npy')
        raw_time = numpy.load('region_deterministic_bnm_wc_rawtime_' + idx[i] + '.npy')

        fig = figure(1)
        clf()
        ax1 = subplot(1, 2, 1)
        plot(raw_time, RAW[:, 0, :, 0], 'k', alpha=0.042, linewidth=3)
        plot(raw_time, RAW[:, 1, :, 0], 'r', alpha=0.042, linewidth=3)
        plot(raw_time, RAW[:, 0, :, 0].mean(axis=1), 'k', linewidth=3)
        plot(raw_time, RAW[:, 1, :, 0].mean(axis=1), 'r', linewidth=3)
        title('TS')
        xlabel('time[ms]')
        ylabel('[au]')
        ylim([0, 1.])
        xlim([0, simulation_length])
        xticks((0, simulation_length / 2., simulation_length),
               ('0', str(int(simulation_length // 2)), str(simulation_length)))
        yticks((0, 0.5, 1), ('0', '0.5', '1'))
        for label in ax1.get_yticklabels():
            label.set_fontsize(24)
        for label in ax1.get_xticklabels():
            label.set_fontsize(24)

        ax = subplot(1, 2, 2)
        plot(RAW[:, 0, :, 0], RAW[:, 1, :, 0], 'b', alpha=0.042, linewidth=3)
        plot(RAW[:, 0, :, 0].mean(axis=1), RAW[:, 1, :, 0].mean(axis=1), 'b', alpha=1., linewidth=3)
        plot(RAW[0, 0, :, 0], RAW[0, 1, :, 0], 'bo', alpha=0.15)
        title('PP')
        ylim([0, 1])
        xlim([0, 1])
        xticks((0, 0.5, 1), ('0', '0.5', '1'))
        yticks((0, 0.5, 1), ('0', '0.5', '1'))
        for label in ax.get_yticklabels():
            label.set_fontsize(24)
        for label in ax.get_xticklabels():
            label.set_fontsize(24)
        ax.yaxis.set_label_position("right")
        xlabel(r'$E$')
        ylabel(r'$I$')

        fig_name = 'WC_default_speed_' + str(int(speed)) + '-config_gcs-' + idx[i] + '.pdf'
        savefig(fig_name)

##EoF