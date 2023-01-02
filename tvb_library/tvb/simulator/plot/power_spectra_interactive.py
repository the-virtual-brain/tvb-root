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
    psi = ps_int.PowerSpectraInteractive()
    psi.time_series = tsr
    psi.show()


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import matplotlib.pyplot as plt
import ipywidgets as widgets
from deprecated import deprecated
from IPython.core.display import display
from tvb.simulator.common import get_logger
import tvb.datatypes.time_series as time_series_datatypes
from tvb.basic.neotraits.api import HasTraits, Attr, Int
from tvb.simulator.plot.utils import generate_region_demo_data

LOG = get_logger(__name__)

# Define a colour theme... see: matplotlib.colors.cnames.keys()
BACKGROUNDCOLOUR = "white"
EDGECOLOUR = "darkslateblue"
AXCOLOUR = "steelblue"
BUTTONCOLOUR = "steelblue"
HOVERCOLOUR = "blue"


@deprecated(reason="Use from tvb.contrib.scripts.plot PowerSpectraCoherenceInteractive")
class PowerSpectraInteractive(HasTraits):
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
        default=-1,
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
        super(PowerSpectraInteractive, self).__init__(**kwargs)
        LOG.debug(str(kwargs))
        # figure
        self.fig = None

        # time-series
        self.fft_ax = None

        # Current state
        self.xscale = "linear"
        self.yscale = "log"
        self.mode = 0
        self.variable = 0
        self.show_sem = False
        self.show_std = False
        self.normalise_power = "no"
        self.window_length = 0.25
        self.window_function = "None"

        # Selectors
        self.xscale_selector = None
        self.yscale_selector = None
        self.mode_selector = None
        self.variable_selector = None
        self.show_sem_selector = None
        self.show_std_selector = None
        self.normalise_power_selector = None
        self.window_length_selector = None
        self.window_function_selector = None

        #
        possible_freq_steps = [2 ** x for x in range(-2, 7)]  # Hz
        self.possible_window_lengths = 1.0 / numpy.array(possible_freq_steps)  # s
        self.freq_step = 1.0 / self.window_length
        self.frequency = None
        self.spectra = None
        self.spectra_norm = None

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
        self.labels = ["channel_%0.3d" % k for k in range(self.nsrs)]

    def show(self):
        """ Generate the interactive power-spectra figure. """
        # Make sure everything is configured
        self.configure()

        # Make the figure:
        fig = self.create_figure()

        self.calc_fft()
        self.plot_spectra()

        display(fig)

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

    ##------------------------------------------------------------------------##
    ##------------------ Functions for building the figure -------------------##
    ##------------------------------------------------------------------------##
    def create_figure(self):
        """ Create the figure and time-series axes. """

        self.outer_box_layout = widgets.Layout(border='solid 1px white',
                                               margin='3px 3px 3px 3px',
                                               padding='2px 2px 2px 2px')

        self.box_layout = widgets.Layout(border='solid 1px black',
                                         margin='3px 3px 3px 3px',
                                         padding='2px 2px 2px 2px')

        self.other_layout = widgets.Layout(width='90%')

        self.lc_items = []

        self.add_selectors_widgets_to_lc_items()

        self.lc_box = widgets.HBox(self.lc_items)
        self.lc_box.layout = self.box_layout

        # item 2
        self.fig = None
        self.op = widgets.Output()
        self.op.layout = self.box_layout

        items = [self.op, self.lc_box]
        grid = widgets.GridBox(items, layout=widgets.Layout(grid_template_rows="450px 250px"))
        return grid

    def add_xscale_selector(self):
        """
        Add a radio button to the figure for selecting which scaling the x-axes
        should use.
        """
        xscale_tuple = ("log", "linear")
        self.xscale_selector = widgets.RadioButtons(options=xscale_tuple, value=xscale_tuple[1],
                                                    layout=self.other_layout)
        self.xscale_selector.observe(self.update_xscale, 'value')

        self.xss_box = widgets.VBox([widgets.Label('X Axis Scale'), self.xscale_selector], layout=self.box_layout)

    def add_yscale_selector(self):
        """
        Add a radio button to the figure for selecting which scaling the y-axes
        should use.
        """
        yscale_tuple = ("log", "linear")
        self.yscale_selector = widgets.RadioButtons(options=yscale_tuple, value=yscale_tuple[0],
                                                    layout=self.other_layout)
        self.yscale_selector.observe(self.update_yscale, 'value')

        self.yss_box = widgets.VBox([widgets.Label('Y Axis Scale'), self.yscale_selector], layout=self.box_layout)

    def add_mode_selector(self):
        """
        Add a radio button to the figure for selecting which mode of the model
        should be displayed.
        """
        mode_tuple = tuple(range(self.data.shape[3]))
        self.mode_selector = widgets.RadioButtons(options=mode_tuple, value=mode_tuple[0], layout=self.other_layout)
        self.mode_selector.observe(self.update_mode, 'value')

        self.ms_box = widgets.VBox([widgets.Label('Mode'), self.mode_selector], layout=self.box_layout)

    def add_variable_selector(self):
        """
        Generate radio selector buttons to set which state variable is 
        displayed.
        """
        noc = self.data.shape[1]
        self.variable_selector = widgets.RadioButtons(options=tuple(range(noc)), value=0, layout=self.other_layout)
        self.variable_selector.observe(self.update_variable, 'value')

        self.vs_box = widgets.VBox([widgets.Label('State Variable'), self.variable_selector], layout=self.box_layout)

    def add_window_length_selector(self):
        """
        Generate radio selector buttons to set the window length is seconds.
        """
        wl_tup = tuple(self.possible_window_lengths)
        self.window_length_selector = widgets.RadioButtons(options=wl_tup, value=wl_tup[4], layout=self.other_layout)
        self.window_length_selector.observe(self.update_window_length, 'value')

        self.wls_box = widgets.VBox([widgets.Label('Segment Length'), self.window_length_selector],
                                    layout=self.box_layout)

    def add_window_function_selector(self):
        """
        Generate radio selector buttons to set the windowing function.
        """
        wf_tup = ("None", "hamming", "bartlett", "blackman", "hanning")
        self.window_function_selector = widgets.RadioButtons(options=wf_tup, value=wf_tup[0], layout=self.other_layout)
        self.window_function_selector.observe(self.update_window_function, 'value')

        self.wfs_box = widgets.VBox([widgets.Label('Windowing Function'), self.window_function_selector],
                                    layout=self.box_layout)

    def add_normalise_power_selector(self):
        """
        Add a radio button to chose whether or not the power of all spectra 
        should be normalised to 1.
        """
        np_tuple = ("yes", "no")
        self.normalise_power_selector = widgets.RadioButtons(options=np_tuple, value=np_tuple[1],
                                                             layout=self.other_layout)
        self.normalise_power_selector.observe(self.update_normalise_power, 'value')

        self.nps_box = widgets.VBox([widgets.Label('Normalise'), self.normalise_power_selector], layout=self.box_layout)

    ##------------------------------------------------------------------------##
    ##------------------ Functions for updating the state --------------------##
    ##------------------------------------------------------------------------##
    def calc_fft(self):
        """
        Calculate FFT using current state of the window_length, window_function,
        """
        # Segment time-series, overlapping if necessary
        nseg = int(numpy.ceil(self.time_series_length / self.window_length))
        if nseg != 1:
            seg_tpts = numpy.ceil(self.window_length / self.period)  # use ceil to avoid dimensions mismatch
            overlap = ((seg_tpts * nseg) - self.tpts) / (nseg - 1)
            starts = [max(seg * (seg_tpts - overlap), 0) for seg in range(nseg)]
            segments = [self.data[int(start):int(start + seg_tpts)] for start in starts]
            segments = [segment[:, :, :, numpy.newaxis] for segment in segments]
            time_series = numpy.concatenate(segments, axis=4)
        else:
            time_series = self.data[:, :, :, :, numpy.newaxis]
            seg_tpts = time_series.shape[0]

        # Base-line correct segmented time-series
        time_series = time_series - time_series.mean(axis=0)[numpy.newaxis, :]

        # Apply windowing function
        if self.window_function != "None":
            window_function = eval("".join(("numpy.", self.window_function)))
            window_mask = numpy.reshape(window_function(seg_tpts),
                                        (int(seg_tpts), 1, 1, 1, 1))
            time_series = time_series * window_mask

        # Calculate the FFT
        result = numpy.fft.fft(time_series, axis=0)
        nfreq = numpy.ceil(len(result) / 2)  # use ceil to avoid dimensions mismatch

        self.frequency = numpy.arange(0, self.max_freq, self.freq_step)
        LOG.debug("frequency shape: %s" % str(self.frequency.shape))

        self.spectra = numpy.mean(numpy.abs(result[1:int(nfreq) + 1]) ** 2, axis=-1)
        LOG.debug("spectra shape: %s" % str(self.spectra.shape))

        self.spectra_norm = (self.spectra / numpy.sum(self.spectra, axis=0))
        LOG.debug("spectra_norm shape: %s" % str(self.spectra_norm.shape))

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
        plt.draw()

    def update_yscale(self, value):
        """ 
        Update the FFT axes' yscale to either log or linear based on radio
        button selection.
        """
        self.yscale = value['new']
        self.fft_ax.set_yscale(self.yscale)
        plt.draw()

    def update_mode(self, value):
        """ Update the visualised mode based on radio button selection. """
        self.mode = value['new']
        self.plot_spectra()

    def update_variable(self, value):
        """ 
        Update state variable being plotted based on radio button selection.
        """
        self.variable = value['new']
        self.plot_spectra()

    def update_normalise_power(self, value):
        """ Update whether to normalise based on radio button selection. """
        self.normalise_power = value['new']
        self.plot_spectra()

    def update_window_length(self, value):
        """
        Update timeseries window length based on the selected value.
        """
        self.window_length = numpy.float64(value['new'])
        self.freq_step = 1.0 / self.window_length
        self.update_spectra()

    def update_window_function(self, value):
        """
        Update windowing function based on the radio button selection.
        """
        self.window_function = value['new']
        self.update_spectra()

    def update_spectra(self):
        """ Clear the axes and redraw the power-spectra. """
        self.calc_fft()
        self.plot_spectra()

    def add_axes(self):
        self.fft_ax = self.fig.add_axes([0.15, 0.15, 0.75, 0.75])

    def plot_figure(self):
        if not self.fig:
            time_series_type = self.time_series.__class__.__name__
            try:
                figure_window_title = "Interactive power spectra: " + time_series_type
                plt.close(figure_window_title)
                self.fig = plt.figure(num=figure_window_title,
                                      figsize=(9, 4),
                                      facecolor=BACKGROUNDCOLOUR,
                                      edgecolor=EDGECOLOUR)
            except ValueError:
                LOG.info("My life would be easier if you'd update your Pyplot...")
                figure_number = 42
                plt.close(figure_number)
                self.fig = plt.figure(num=figure_number,
                                      figsize=(9, 4),
                                      facecolor=BACKGROUNDCOLOUR,
                                      edgecolor=EDGECOLOUR)
            self.add_axes()

    def fft_ax_plot(self):
        # import pdb; pdb.set_trace()
        # Plot the power spectra
        if self.normalise_power == "yes":
            self.fft_ax.plot(self.frequency,
                             self.spectra_norm[:, self.variable, :, self.mode])
        else:
            self.fft_ax.plot(self.frequency,
                             self.spectra[:, self.variable, :, self.mode])

    def plot_fft(self):
        self.fft_ax.clear()
        # Set title and axis labels
        time_series_type = self.time_series.__class__.__name__
        self.fft_ax.set(title=time_series_type)
        self.fft_ax.set(xlabel="Frequency (%s)" % self.units)
        self.fft_ax.set(ylabel="Power")

        # Set x and y scale based on curent radio button selection.
        self.fft_ax.set_xscale(self.xscale)
        self.fft_ax.set_yscale(self.yscale)

        if hasattr(self.fft_ax, 'autoscale'):
            self.fft_ax.autoscale(enable=True, axis='both', tight=True)

        self.fft_ax_plot()

    def plot_figures(self):
        self.plot_fft()

    def plot_spectra(self):
        """ Plot the power spectra. """
        self.op.clear_output(wait=True)
        with plt.ioff():
            if not self.fig:
                time_series_type = self.time_series.__class__.__name__
                try:
                    figure_window_title = "Interactive power spectra: " + time_series_type
                    plt.close(figure_window_title)
                    self.fig = plt.figure(num=figure_window_title,
                                          figsize=(9, 4),
                                          facecolor=BACKGROUNDCOLOUR,
                                          edgecolor=EDGECOLOUR)
                except ValueError:
                    LOG.info("My life would be easier if you'd update your Pyplot...")
                    figure_number = 42
                    plt.close(figure_number)
                    self.fig = plt.figure(num=figure_number,
                                          figsize=(9, 4),
                                          facecolor=BACKGROUNDCOLOUR,
                                          edgecolor=EDGECOLOUR)
            self.add_axes()
            self.plot_figures()

        with self.op:
            display(self.fig.canvas)


def main_function(class_type=PowerSpectraInteractive):
    import os
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

    psi = PowerSpectraInteractive()
    psi.time_series = tsr
    psi.show()


if __name__ == "__main__":
    main_function()
