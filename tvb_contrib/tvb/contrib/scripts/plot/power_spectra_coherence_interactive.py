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
An interactive power spectra plot generated from a TVB TimeSeries datatype.

Usage
::

    #Load the demo data
    import numpy
    data = numpy.load("demos/demo_data_region_16s_2048Hz.npy")
    period = 0.00048828125 #NOTE: Providing period in seconds

    #Create a tvb TimeSeries object
    import tvb.datatypes.time_series
    tsr = tvb.datatypes.time_series.TimeSeriesRegion()
    tsr.data = data
    tsr.sample_period = period
    tsr.sample_period_unit = 's'

    #Create and launch the interactive visualiser
    import tvb.simulator.power_spectra_interactive as ps_int
    psi = ps_int.PowerSpectraCoherenceInteractive()
    psi.time_series = tsr
    psi.show()

.. moduleauthor:: Lionel Kusch <lkusch@thevirtualbrain.org>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import os
import numpy
import matplotlib.pyplot as plt
import ipywidgets as widgets
import tvb.datatypes.time_series as time_series_datatypes
from tvb.simulator.common import get_logger
from tvb.simulator.plot.utils import generate_region_demo_data
from tvb.basic.neotraits.api import HasTraits, Int, Attr

LOG = get_logger(__name__)

BACKGROUNDCOLOUR = "slategrey"
EDGECOLOUR = "darkslateblue"
AXCOLOUR = "steelblue"
BUTTONCOLOUR = "steelblue"
HOVERCOLOUR = "blue"


class PowerSpectraCoherenceInteractive(HasTraits):
    """
    The graphical interface for visualising the power-spectra (FFT) of a
    timeseries provide controls for setting:

        - which state-variable and mode to display [sets]
        - log or linear scaling for the power or frequency axis [binary]
        - segmentation length [set]
        - windowing function [set]
        - power normalisation [binary] (emphasise relative frequency contribution)
        - show std or sem [binary]

    """

    time_series = Attr(
        field_type=time_series_datatypes.TimeSeries,
        label="Timeseries",
        default=None,
        doc=""" The timeseries to which the FFT is to be applied.""")

    first_n = Int(
        label="Display the first 'n'",
        default=0,
        doc="""Primarily intended for displaying the first N components of a 
            surface PCA timeseries. Defaults to -1, meaning it'll display all
            of 'space' (ie, regions or vertices or channels). In other words,
            for Region or M/EEG timeseries you can ignore this, but, for a 
            surface timeseries it really must be set.""")

    def __init__(self, **kwargs):
        """
        Initialise based on provided keywords or their traited defaults. Also,
        initialise the place-holder attributes that aren't filled until the
        show() method is called.

        """
        LOG.debug(str(kwargs))

        # figure
        self.fig = None

        # time-series
        self.fft_ax = None
        self.coh_ax = None

        # Current state
        self.units = "Hz"
        self.period = 1.0  # s
        self.max_freq = 0.5 / self.period

        # Selectors
        self.freq_range_selector = None
        self.reg_selector = None

    def _assert_window_length(self):
        self.window_length = numpy.minimum(self.window_length, self.window_length_max)
        print("window_length = %g sec" % self.window_length)
        self.freq_step = 1.0 / self.window_length
        print("freq_step = %g %s" % (self.freq_step, self.units))

    def _configure_time_from_time_series(self):
        if self.time_series is not None:
            if isinstance(self.time_series.time, numpy.ndarray) and \
                    len(self.time_series.time) == self.tpts:
                self.time = self.time_series.time.copy()
        self.window_length_max = self.time_series_length
        self._assert_window_length()

    def configure(self):
        """ Separate configure cause traits be busted... """
        LOG.debug("time_series shape: %s" % str(self.time_series.data.shape))

        self.data = self.time_series.data[:, :, :self.first_n, :]
        self.period = 1 / self.time_series.sample_rate
        self.period_unit = "s"
        self.max_freq = 0.5 / self.period
        self.units = "Hz"
        self.tpts = self.data.shape[0]
        self.nsrs = self.data.shape[2]
        self.time_series_length = self.tpts * self.period
        self.time = numpy.arange(self.tpts) * self.period

        print('data.shape = %s' % str(self.data.shape))
        print("period = %g ms" % self.period)
        print("max_freq = %g %s" % (self.max_freq, self.units))
        print("time_series_length = %g ms" % self.time_series_length)

        self.regions = numpy.arange(self.data.shape[2]).astype('i')
        self.reg_xy_inds = {}
        self.data_xy = []
        ii_xy = 0
        for reg1 in self.regions:
            for reg2 in range(reg1 + 1, self.regions[-1] + 1):
                self.reg_xy_inds[(reg1, reg2)] = int(ii_xy)
                ii_xy += 1
                self.data_xy.append(self.data[:, :, reg1, :] * self.data[:, :, reg2, :])
        self.data_xy = numpy.array(self.data_xy).transpose(1, 2, 0, 3)
        self.reg_inds_sel = slice(0, self.regions.shape[0])
        self.reg_xy_inds_sel = numpy.arange(self.data_xy.shape[2]).astype('i')
        self.nsrs_xy = self.nsrs * (self.nsrs - 1) / 2
        self.labels = ["channel_%0.3d" % k for k in range(self.nsrs)]
        self._configure_time_from_time_series()

    def add_window_length_selector(self):
        """
        Generate radio selector buttons to set the window length is seconds.
        """
        self._assert_window_length()
        self.possible_window_lengths = numpy.unique(numpy.concatenate((self.possible_window_lengths,
                                                                       numpy.array([self.window_length]))))
        self.possible_window_lengths = \
            self.possible_window_lengths[self.possible_window_lengths < self.window_length_max]
        active = numpy.argmin(numpy.abs(self.possible_window_lengths - self.window_length))
        wl_tup = tuple(self.possible_window_lengths)
        self.window_length_selector = widgets.RadioButtons(options=wl_tup, value=wl_tup[active],
                                                           layout=self.other_layout)
        self.window_length_selector.observe(self.update_window_length, 'value')

        self.wls_box = widgets.VBox([widgets.Label('Segment Length'), self.window_length_selector],
                                    layout=self.box_layout)

    def add_freq_range_selector(self):
        self.freq_range_selector = widgets.Text(value="%g, " % 0.0)
        self.freq_range_selector.observe(self.update_freq_range, 'value')

        self.frs_box = widgets.VBox([widgets.Label('Frequency range'), self.freq_range_selector],
                                    layout=self.box_layout)

    def add_regions_selector(self):
        self.reg_selector = widgets.Text(value="%d:%d:%d" % (0, self.data.shape[2] - 1, 1))
        self.reg_selector.observe(self.update_regions, 'value')

        self.rgs_box = widgets.VBox([widgets.Label('Regions'), self.reg_selector], layout=self.box_layout)

    def add_selectors_widgets_to_lc_items(self):

        self.add_mode_selector()
        self.add_normalise_power_selector()
        self.add_xscale_selector()
        self.add_yscale_selector()
        self.add_variable_selector()
        self.add_window_function_selector()
        self.add_window_length_selector()

        self.mode_sv_box = widgets.VBox([self.ms_box, self.vs_box], layout=self.outer_box_layout)
        self.xs_ys_box = widgets.VBox([self.xss_box, self.yss_box], layout=self.outer_box_layout)
        self.np_box = widgets.VBox([self.nps_box], layout=self.outer_box_layout)
        self.wf_box = widgets.VBox([self.wfs_box], layout=self.outer_box_layout)
        self.wl_box = widgets.VBox([self.wls_box], layout=self.outer_box_layout)

        self.lc_items.extend([self.mode_sv_box, self.xs_ys_box, self.np_box, self.wf_box, self.wl_box])

        self.add_freq_range_selector()
        self.add_regions_selector()
        self.rg_box = widgets.VBox([self.rgs_box], layout=self.outer_box_layout)
        self.fr_box = widgets.VBox([self.frs_box], layout=self.outer_box_layout)
        self.lc_items.extend([self.rg_box, self.fr_box])

    ##------------------------------------------------------------------------##
    ##------------------ Functions for updating the state --------------------##
    ##------------------------------------------------------------------------##
    def calc_fft(self):
        """
        Calculate FFT using current state of the window_length, window_function,
        """
        # Segment time-series, overlapping if necessary
        nseg = int(numpy.ceil(self.time_series_length / self.window_length))
        print("nseg = \n%s" % str(nseg))
        if nseg != 1:
            seg_tpts = int(numpy.round(self.window_length / self.period))
            print("seg_tpts = \n%s" % str(seg_tpts))
            overlap = int(numpy.round(((seg_tpts * nseg) - self.tpts) / (nseg - 1)))
            print("overlap = \n%s" % str(overlap))
            starts = [int(numpy.round(max(seg * (seg_tpts - overlap), 0)))
                      for seg in range(nseg)]
            segments = [self.data[start:start + seg_tpts] for start in starts]
            segments = [segment[:, :, :, numpy.newaxis] for segment in segments]
            time_series = numpy.concatenate(segments, axis=4)
            segments_xy = [self.data_xy[start:start + seg_tpts] for start in starts]
            segments_xy = [segment_xy[:, :, :, numpy.newaxis] for segment_xy in segments_xy]
            time_series_xy = numpy.concatenate(segments_xy, axis=4)
        else:
            time_series = self.data[:, :, :, :, numpy.newaxis]
            time_series_xy = self.data_xy[:, :, :, :, numpy.newaxis]
            seg_tpts = time_series.shape[0]

        # Base-line correct segmented time-series
        time_series = time_series - time_series.mean(axis=0)[numpy.newaxis, :]
        time_series_xy = time_series_xy - time_series_xy.mean(axis=0)[numpy.newaxis, :]

        # Apply windowing function
        if self.window_function != "None":
            window_function = eval("".join(("numpy.", self.window_function)))
            window_mask = numpy.reshape(window_function(seg_tpts),
                                        (seg_tpts, 1, 1, 1, 1))
            time_series = time_series * window_mask
            time_series_xy = time_series_xy * window_mask

        # Calculate the FFT
        result = numpy.fft.fft(time_series, axis=0)
        result_xy = numpy.fft.fft(time_series_xy, axis=0)
        nfreq = int(numpy.round(len(result) / 2))

        self.frequency = numpy.arange(0, self.max_freq, self.freq_step)
        self.plot_freqs_slice = slice(0, self.frequency.size)
        LOG.debug("frequency shape: %s" % str(self.frequency.shape))
        print("frequency shape: %s" % str(self.frequency.shape))

        self.spectra = numpy.mean(numpy.abs(result[1:nfreq + 1]) ** 2, axis=-1)
        LOG.debug("spectra shape: %s" % str(self.spectra.shape))
        print("spectra shape: %s" % str(self.spectra.shape))

        self.spectra_norm = (self.spectra / numpy.sum(self.spectra, axis=0))
        LOG.debug("spectra_norm shape: %s" % str(self.spectra_norm.shape))
        print("spectra_norm shape: %s" % str(self.spectra_norm.shape))

        spectra_norm_xy = numpy.mean(numpy.abs(result_xy[1:nfreq + 1]) ** 2, axis=-1)
        spectra_norm_xy = (spectra_norm_xy / numpy.sum(spectra_norm_xy, axis=0))
        self.coherence = []
        for xy_inds, ii in self.reg_xy_inds.items():
            self.coherence.append(
                spectra_norm_xy[:, :, ii, :] ** 2 /
                (self.spectra_norm[:, :, xy_inds[0], :] *
                 self.spectra_norm[:, :, xy_inds[1], :]),
            )
        self.coherence = numpy.array(self.coherence).transpose(1, 2, 0, 3)
        LOG.debug("coherence shape: %s" % str(self.spectra.shape))

    ##------------------------------------------------------------------------##
    ##------------------ Functions for updating the figure -------------------##
    ##------------------------------------------------------------------------##
    def update_xscale(self, value):
        """
        Update the FFT axes' xscale to either log or linear based on radio
        button selection.
        """
        self.xscale = value['new']
        self.fft_ax.set_xscale(self.xscale)
        self.coh_ax.set_xscale(self.xscale)
        plt.draw()

    def update_yscale(self, value):
        """
        Update the FFT axes' yscale to either log or linear based on radio
        button selection.
        """
        self.yscale = value['new']
        self.fft_ax.set_yscale(self.yscale)
        self.coh_ax.set_yscale(self.yscale)
        plt.draw()

    def update_window_length(self, value):
        """
        Update timeseries window length based on the selected value.
        """
        # TODO: need this casting but not sure why, don't need int() with mode...
        self.window_length = numpy.float64(value['new'])
        self._assert_window_length()
        # import pdb; pdb.set_trace()
        self.update_spectra()

    def _find_nearest(self, value):
        return (numpy.abs(self.frequency - value)).argmin()

    def update_freq_range(self, value):
        freq_range_text = value['new']
        if "," in freq_range_text:
            print(freq_range_text)
            freqs = freq_range_text.split(",")
            print(freqs)
            for iF, (new_freq, current_freq) in \
                    enumerate(zip(freqs, [self.plot_freqs_slice.start, self.plot_freqs_slice.stop])):
                print(new_freq)
                if len(new_freq):
                    freqs[iF] = self._find_nearest(float(new_freq))
                else:
                    freqs[iF] = current_freq
                self.plot_freqs_slice = slice(freqs[0], freqs[1])
        else:
            if len(freq_range_text):
                freqs = float(freq_range_text)
                if freqs > 0.0:
                    self.plot_freqs_slice = slice(0, self._find_nearest(freqs))
                else:
                    self.plot_freqs_slice = slice(0, self.frequency.size)
        self.plot_spectra()

    def _update_xy_regions(self):
        self.reg_xy_inds_sel = []
        for ir1, reg1 in enumerate(self.reg_inds_sel[:-1]):
            for reg2 in self.reg_inds_sel[ir1 + 1:]:
                self.reg_xy_inds_sel.append(self.reg_xy_inds[(reg1, reg2)])
        self.reg_xy_inds_sel = numpy.array(self.reg_xy_inds_sel).astype('i')

    def update_regions(self, value):
        regions_slice_text = value['new']
        """ Update the visualised regions based on radio button selection. """
        if ":" in regions_slice_text:
            regions_slice = regions_slice_text.split(":")
            start = None
            stop = None
            step = None
            if len(regions_slice) > 0:
                try:
                    start = int(regions_slice[0])
                except:
                    pass
                if len(regions_slice) > 1:
                    try:
                        stop = int(regions_slice[1])
                    except:
                        pass
                    if len(regions_slice) > 2:
                        try:
                            step = int(regions_slice[2])
                        except:
                            pass
            self.reg_inds_sel = self.regions[slice(start, stop, step)]
        else:
            if len(regions_slice_text):
                self.reg_inds_sel = numpy.array(regions_slice_text.split(",")).astype('i')
        print("Indices of regions selected for plotting:\n%s" % str(self.reg_inds_sel))
        self._update_xy_regions()
        self.plot_spectra()

    def add_axes(self):
        self.fft_ax = self.fig.add_axes([0.15, 0.15, 0.75, 0.75])
        self.coh_ax = self.fig.add_axes([0.15, 0.2, 0.7, 0.3])

    def fft_ax_plot(self):
        # Plot the power spectra
        if self.normalise_power == "yes":
            self.fft_ax.clear()
            self.fft_ax.set(ylabel="PSD")
            self.fft_ax.plot(self._freqs, self.spectra_norm[self.plot_freqs_slice, self.variable,
                                                            self.reg_inds_sel, self.mode])
        else:
            self.fft_ax.clear()
            self.fft_ax.set(ylabel="Power")
            self.fft_ax.plot(self._freqs, self.spectra[self.plot_freqs_slice, self.variable,
                                                       self.reg_inds_sel, self.mode])

    def plot_coh(self):
        self.coh_ax.clear()
        self.coh_ax.set(ylabel="Coherence")
        self.coh_ax.set(xlabel="Frequency (%s)" % self.units)
        if self.reg_xy_inds_sel.size:
            self.coh_ax.plot(self._freqs,
                             self.coherence[self.plot_freqs_slice, self.variable,
                                            self.reg_xy_inds_sel, self.mode])

        # Set x and y scale based on curent radio button selection.
        self.coh_ax.set_xscale(self.xscale)
        self.coh_ax.set_yscale(self.yscale)
        if hasattr(self.coh_ax, 'autoscale'):
            self.coh_ax.autoscale(enable=True, axis='both', tight=True)

    def plot_figures(self):
        self_freqs = self.frequency[self.plot_freqs_slice]
        print("Frequency range for plotting: [%g, %g]" % (self_freqs[0], self_freqs[-1]))
        self.plot_fft()
        self.plot_coh()


def main_function():
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    file_path = os.path.join(os.getcwd(), "demo_data_region_16s_2048Hz.npy")
    try:
        data = numpy.load(file_path)
    except IOError:
        LOG.error("Can't load demo data. It will be created now running the generate_region_demo_data() function.")
        generate_region_demo_data(file_path=file_path)
        data = numpy.load(file_path)

    period = 0.00048828125  # NOTE: Providing period in seconds
    tsr = time_series_datatypes.TimeSeriesRegion()
    tsr.data = data
    tsr.sample_period = period
    tsr.sample_period_unit = 's'

    psi = PowerSpectraCoherenceInteractive()
    psi.time_series = tsr
    psi.show()


if __name__ == "__main__":
    main_function()
