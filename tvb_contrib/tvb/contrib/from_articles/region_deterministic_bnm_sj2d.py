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
     Reproduces Figures XX Sanz-Leon P., PhD Thesis

Needs:
     A working installation of tvb

Run:
     python region_deterministic_bnm_sjd2d.py -s True -f True

     #Subsequent calls can be made with: 
     python region_deterministic_bnm_sj2d.py -f True

.. author:: Paula Sanz-Leon
"""

import numpy
import argparse
from tvb.simulator.lab import *
import matplotlib.pylab as pylab
from matplotlib.pylab import *

LOG = get_logger(__name__)

pylab.rcParams['figure.figsize'] = 20, 15  # that's default image size for this interactive session
pylab.rcParams.update({'font.size': 22})
pylab.rcParams.update({'lines.linewidth': 3})
pylab.rcParams.update({'axes.linewidth': 3})

parser = argparse.ArgumentParser(description='Reproduce results of Figure XX presented in Sanz-Leon 2014 PhD Thesis')
parser.add_argument('-s', '--sim', help='Run the simulations', default=False)
parser.add_argument('-f', '--fig', help='Plot the figures', default=False)
args = vars(parser.parse_args())

idx = ['a0', 'a1', 'a2']
gcs = [0.0, 0.5, 1.0]

simulation_length = 2e3
speed = 10.

if args['sim']:
    for i in range(3):
        oscilator = models.ReducedSetFitzHughNagumo()
        oscilator.variables_of_interest = ["xi", "eta", "alpha", "beta"]
        white_matter = connectivity.Connectivity.from_file("connectivity_66.zip")
        white_matter.speed = numpy.array([speed])
        white_matter_coupling = coupling.Linear(a=numpy.array([gcs[i]]))

        # Initialise an Integrator
        heunint = integrators.HeunDeterministic(dt=0.1)

        # Initialise some Monitors with period in physical time
        momo = monitors.Raw()
        mama = monitors.TemporalAverage(period=1.)

        # Bundle them
        what_to_watch = (momo, mama)

        # Initialise a Simulator -- Model, Connectivity, Integrator, and Monitors.
        sim = simulator.Simulator(model=oscilator, connectivity=white_matter,
                                  coupling=white_matter_coupling,
                                  integrator=heunint, monitors=what_to_watch)

        sim.configure()
        # LOG.info("Starting simulation...")
        # #Perform the simulation
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
        TAVG = numpy.asarray(tavg_data)
        RAW = numpy.asarray(raw_data)

        LOG.info("Saving simulated data ...")
        numpy.save('region_deterministic_bnm_sj2d_raw_' + idx[i] + '.npy', RAW)
        numpy.save('region_deterministic_bnm_sj2d_tavg_' + idx[i] + '.npy', TAVG)
        numpy.save('region_deterministic_bnm_sj2d_rawtime_' + idx[i] + '.npy', raw_time)
        numpy.save('region_deterministic_bnm_sj2d_tavgtime_' + idx[i] + '.npy', tavg_time)

if args['fig']:
    for i in range(3):

        start_point = int(simulation_length // 4)
        end_point = int(simulation_length // 4 + start_point // 2)

        LOG.info("Generating pretty pictures ...")
        TAVG = numpy.load('region_deterministic_bnm_sj2d_tavg_' + idx[i] + '.npy')
        tavg_time = numpy.load('region_deterministic_bnm_sj2d_tavgtime_' + idx[i] + '.npy')[start_point:end_point]

        fig = figure(1)
        clf()

        for k in range(3):
            # load data
            # compute time and use sim_length

            ax = subplot(3, 3, 4 + k)
            plot(tavg_time, TAVG[start_point:end_point, 0, :, k], 'k', alpha=0.042, linewidth=3)
            plot(tavg_time, TAVG[start_point:end_point, 1, :, k], 'r', alpha=0.042, linewidth=3)
            plot(tavg_time, TAVG[start_point:end_point, 0, :, k].mean(axis=1), 'k')
            plot(tavg_time, TAVG[start_point:end_point, 1, :, k].mean(axis=1), 'r')
            ylim([-5, 2])
            xlim([start_point, int(end_point)])
            for label in ax.get_yticklabels():
                label.set_fontsize(20)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if k == 0:
                ylabel('[au]')
                yticks((-4, 0, 1), ('-4', '0', '1'))
                title(r'TS ($m=1$)')

            ax = subplot(3, 3, 7 + k)
            plot(tavg_time, TAVG[start_point:end_point, 2, :, k], 'k', alpha=0.042, linewidth=3)
            plot(tavg_time, TAVG[start_point:end_point, 3, :, k], 'r', alpha=0.042, linewidth=3)
            plot(tavg_time, TAVG[start_point:end_point, 2, :, k].mean(axis=1), 'k')
            plot(tavg_time, TAVG[start_point:end_point, 3, :, k].mean(axis=1), 'r')
            ylim([-5, 2])
            xlim([start_point, int(end_point)])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            xticks((start_point, end_point), (str(int(start_point)), str(int(end_point))))
            xlabel('time[ms]')
            if k == 0:
                ylabel('[au]')
                yticks((-4, 0, 1), ('-4', '0', '1'))
                title(r'TS ($m=2$)')

            ax = subplot(3, 3, 1 + k)
            plot(TAVG[start_point:end_point, 0, :, k], TAVG[start_point:end_point, 1, :, k], 'b', alpha=0.042)
            plot(TAVG[start_point:end_point, 0, :, k].mean(axis=1), TAVG[start_point:end_point, 1, :, k].mean(axis=1),
                 'b')
            title(r'PP ($o=%s$)' % str(k))
            ax.yaxis.set_label_position("right")
            ylim([-5, 2])
            xlim([-5, 2])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            if k == 1:
                xticks((-4, 0, 1), ('-4', '0', '1'))
                ax.xaxis.labelpad = -10
                xlabel(r'$\xi$')
                yticks((-4, 0, 1), ('-4', '0', '1'))
                ylabel(r'$\eta$')

        fig_name = 'SJ2D_default_speed_' + str(int(speed)) + '-config_gcs-' + idx[i] + '.png'
        savefig(fig_name)

###EoF###
