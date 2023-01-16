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
     Reproduces Figures 23 and 24 of Sanz-Leon P., Knock, S. A., Spiegler, A. and Jirsa V.
     Mathematical framework for large-scale brain network modelling in The Virtual Brain.
     Neuroimage, 2014, (in review)

Needs:
     A working installation of tvb

Run:
     python region_deterministic_bnm_wc.py -s True -f True

     #Subsequent calls can be made with: 
     python region_deterministic_bnm_wc.py -f True

.. author:: Paula Sanz-Leon
"""

import numpy
import argparse
from tvb.simulator.lab import *
import matplotlib.pylab as pylab
from matplotlib.pylab import *

LOG = get_logger(__name__)

pylab.rcParams['figure.figsize'] = 19.42, 12  # that's default image size for this interactive session
pylab.rcParams.update({'font.size': 22})

parser = argparse.ArgumentParser(description='Reproduce results of Figure XX presented in Sanz-Leon et al 2014')
parser.add_argument('-s', '--sim', help='Run the simulations', default=False)
parser.add_argument('-f', '--fig', help='Plot the figures', default=False)
args = vars(parser.parse_args())

speed = 4.0
simulation_length = 512

# TODO change the input params, as c_1 ... c_4 are no longer compatible with the model class
oscilator = models.WilsonCowan(c_1=numpy.array([16.]), c_2=numpy.array([12.]), c_3=numpy.array([15.]),
                               c_4=numpy.array([3]), tau_e=numpy.array([8.]), tau_i=numpy.array([8.]),
                               a_e=numpy.array([1.3]), a_i=numpy.array([2.]), theta_e=numpy.array([4.]),
                               theta_i=numpy.array([3.7]))
white_matter = connectivity.Connectivity.from_file()
white_matter.speed = numpy.array([speed])
gcs = 8
white_matter_coupling = coupling.Linear(a=numpy.array([2 ** -gcs]))

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
RAW = numpy.array(raw_data)
TAVG = numpy.array(tavg_data)

# <codecell>

numpy.save('region_deterministic_bnm_article_wc_raw.npy', RAW)
numpy.save('region_deterministic_bnm_article_wc_rawtime.npy', raw_time)
numpy.save('region_deterministic_bnm_article_wc_tavg.npy', TAVG)
numpy.save('region_deterministic_bnm_article_wc_tavgtime.npy', tavg_time)

if args['fig']:
    RAW = numpy.load('region_deterministic_bnm_article_wc_raw.npy')
    raw_time = numpy.load('region_deterministic_bnm_article_wc_rawtime.npy')
    # Plot temporally averaged time series
    figure(1)
    subplot(1, 2, 1)
    plot(raw_time, RAW[:, 0, :, 0], 'k', alpha=0.042, linewidth=3)
    plot(raw_time, RAW[:, 1, :, 0], 'r', alpha=0.042, linewidth=3)
    plot(raw_time, RAW[:, 0, :, 0].mean(axis=1), 'k', linewidth=3)
    plot(raw_time, RAW[:, 1, :, 0].mean(axis=1), 'r', linewidth=3)

    xlabel('time[ms]')
    # ylim([-25, 5])
    xlim([0, sim.simulation_length])
    subplot(1, 2, 2)
    plot(RAW[:, 0, :, 0], RAW[:, 1, :, 0], alpha=0.042)
    plot(RAW[:, 0, :, 0].mean(axis=1), RAW[:, 1, :, 0].mean(axis=1), alpha=1.)
    plot(RAW[0, 0, :, 0], RAW[0, 1, :, 0], 'bo', alpha=0.15)

    xlabel(r'$E$')
    ylabel(r'$I$')

    show()

    fig_name = 'wc_default_speed_' + str(int(white_matter.speed)) + '_gcs_2**-' + str(gcs) + '.pdf'
    savefig(fig_name)

###EoF###
