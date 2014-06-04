# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

    #Create and launch the interactive visualiser
    import tvb.simulator.power_spectra_interactive as ps_int
    psi = ps_int.PowerSpectraInteractive(time_series=tsr)
    psi.show()


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
#TODO: There are fence-posts...
#TODO: add a save button, for current powerspectra view (data more than fig)
#TODO: channel/region selection, with surface time-series vertex slection 
#      grouped by region 

import numpy
import pylab
import matplotlib.widgets as widgets

#The Virtual Brain
from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

import tvb.datatypes.time_series as time_series_datatypes

import tvb.basic.traits.core as core
import tvb.basic.traits.types_basic as basic

# Define a colour theme... see: matplotlib.colors.cnames.keys()
BACKGROUNDCOLOUR = "slategrey"
EDGECOLOUR = "darkslateblue"
AXCOLOUR = "steelblue"
BUTTONCOLOUR = "steelblue"
HOVERCOLOUR = "blue"


class PowerSpectraInteractive(core.Type):
    """
    The graphical interface for visualising the power-spectra (FFT) of a
    timeseries provide controls for setting:

        - which state-variable and mode to display [sets]
        - log or linear scaling for the power or frequency axis [binary]
        - sementation lenth [set]
        - windowing function [set]
        - power normalisation [binary] (emphasise relative frequency contribution)
        - show std or sem [binary]


    """

    time_series = time_series_datatypes.TimeSeries(
        label = "Timeseries",
        default = None,
        required = True,
        doc = """ The timeseries to which the FFT is to be applied.""")

    first_n = basic.Integer(
        label = "Display the first 'n'",
        default = -1,
        required = True,
        doc = """Primarily intended for displaying the first N components of a 
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
        #figure
        self.ifft_fig = None

        #time-series
        self.fft_ax = None

        #Current state
        self.xscale = "linear"
        self.yscale = "log"
        self.mode = 0
        self.variable = 0
        self.show_sem = False
        self.show_std = False
        self.normalise_power = "no"
        self.window_length = 0.25
        self.window_function = "None"

        #Selectors
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
        possible_freq_steps = [2**x for x in range(-2, 7)] #Hz
        #possible_freq_steps.append(1.0 / self.time_series_length) #Hz
        self.possible_window_lengths = 1.0 / numpy.array(possible_freq_steps) #s
        self.freq_step = 1.0 / self.window_length
        self.frequency = None
        self.spectra = None
        self.spectra_norm = None

        #Sliders
        #self.window_length_slider = None


    def configure(self):
        """ Seperate configure cause ttraits be busted... """
        LOG.debug("time_series shape: %s" % str(self.time_series.data.shape))
        #TODO: if isinstance(self.time_series, TimeSeriesSurface) and self.first_n == -1: #LOG.error, return.
        self.data = self.time_series.data[:, :, :self.first_n, :]
        self.period = self.time_series.sample_period
        self.max_freq = 0.5 / self.period
        self.units = "Hz"
        self.tpts = self.data.shape[0]
        self.nsrs = self.data.shape[2]
        self.time_series_length = self.tpts * self.period
        self.time = numpy.arange(self.tpts) * self.period
        self.labels = ["channel_%0.3d" % k for k in range(self.nsrs)]


    def show(self):
        """ Generate the interactive power-spectra figure. """
        #Make sure everything is configured
        self.configure()

        #Make the figure:
        self.create_figure()

        #Selectors
        self.add_xscale_selector()
        self.add_yscale_selector()
        self.add_mode_selector()
        self.add_variable_selector()
        self.add_normalise_power_selector()
        self.add_window_length_selector()
        self.add_window_function_selector()

        #Sliders
        #self.add_window_length_slider() #Want discrete values
        #self.add_scaling_slider()

        #...
        self.calc_fft()

        #Plot timeseries
        self.plot_spectra()

        pylab.show()

    ##------------------------------------------------------------------------##
    ##------------------ Functions for building the figure -------------------##
    ##------------------------------------------------------------------------##
    def create_figure(self):
        """ Create the figure and time-series axes. """
        time_series_type = self.time_series.__class__.__name__
        try:
            figure_window_title = "Interactive power spectra: " + time_series_type
            pylab.close(figure_window_title)
            self.ifft_fig = pylab.figure(num = figure_window_title,
                                        figsize = (16, 8),
                                        facecolor = BACKGROUNDCOLOUR, 
                                        edgecolor = EDGECOLOUR)
        except ValueError:
            LOG.info("My life would be easier if you'd update your PyLab...")
            figure_number = 42
            pylab.close(figure_number)
            self.ifft_fig = pylab.figure(num = figure_number, 
                                         figsize = (16, 8), 
                                         facecolor = BACKGROUNDCOLOUR, 
                                         edgecolor = EDGECOLOUR)

        self.fft_ax = self.ifft_fig.add_axes([0.15, 0.2, 0.7, 0.75])


    def add_xscale_selector(self):
        """
        Add a radio button to the figure for selecting which scaling the x-axes
        should use.
        """
        pos_shp = [0.45, 0.02, 0.05, 0.104]
        rax = self.ifft_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="xscale")
        xscale_tuple = ("log", "linear")
        self.xscale_selector = widgets.RadioButtons(rax, xscale_tuple, active=1)
        self.xscale_selector.on_clicked(self.update_xscale)


    def add_yscale_selector(self):
        """
        Add a radio button to the figure for selecting which scaling the y-axes
        should use.
        """
        pos_shp = [0.02, 0.5, 0.05, 0.104]
        rax = self.ifft_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="yscale")
        yscale_tuple = ("log", "linear")
        self.yscale_selector = widgets.RadioButtons(rax, yscale_tuple, active=0)
        self.yscale_selector.on_clicked(self.update_yscale)


    def add_mode_selector(self):
        """
        Add a radio button to the figure for selecting which mode of the model
        should be displayed.
        """
        pos_shp = [0.02, 0.07, 0.05, 0.1+0.002*self.data.shape[3]]
        rax = self.ifft_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="Mode")
        mode_tuple = tuple(range(self.data.shape[3]))
        self.mode_selector = widgets.RadioButtons(rax, mode_tuple, active=0)
        self.mode_selector.on_clicked(self.update_mode)


    def add_variable_selector(self):
        """
        Generate radio selector buttons to set which state variable is 
        displayed.
        """
        noc = self.data.shape[1] # number of choices
        #State variable for the x axis
        pos_shp = [0.02, 0.22, 0.05, 0.12+0.008*noc]
        rax = self.ifft_fig.add_axes(pos_shp, axisbg=AXCOLOUR, 
                                     title="state variable")
        self.variable_selector = widgets.RadioButtons(rax, tuple(range(noc)), 
                                                    active=0)
        self.variable_selector.on_clicked(self.update_variable)


    def add_window_length_selector(self):
        """
        Generate radio selector buttons to set the window length is seconds.
        """
        noc = self.possible_window_lengths.shape[0] # number of choices
        #State variable for the x axis
        pos_shp = [0.88, 0.07, 0.1, 0.12+0.02*noc]
        rax = self.ifft_fig.add_axes(pos_shp, axisbg=AXCOLOUR, 
                                     title="Segment length")
        wl_tup = tuple(self.possible_window_lengths)
        self.window_length_selector = widgets.RadioButtons(rax, wl_tup, active=4)
        self.window_length_selector.on_clicked(self.update_window_length)


    def add_window_function_selector(self):
        """
        Generate radio selector buttons to set the windowing function.
        """
        #TODO: add support for kaiser, requiers specification of beta.
        wf_tup = ("None", "hamming", "bartlett", "blackman", "hanning") 
        noc = len(wf_tup) # number of choices
        #State variable for the x axis
        pos_shp = [0.88, 0.77, 0.085, 0.12+0.01*noc]
        rax = self.ifft_fig.add_axes(pos_shp, axisbg=AXCOLOUR, 
                                    title="Windowing function")
        self.window_function_selector = widgets.RadioButtons(rax, wf_tup, active=0)
        self.window_function_selector.on_clicked(self.update_window_function)


    def add_normalise_power_selector(self):
        """
        Add a radio button to chose whether or not the power of all spectra 
        shouold be normalised to 1.
        """
        pos_shp = [0.02, 0.8, 0.05, 0.104]
        rax = self.ifft_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="normalise")
        np_tuple = ("yes", "no")
        self.normalise_power_selector = widgets.RadioButtons(rax, np_tuple, active=1)
        self.normalise_power_selector.on_clicked(self.update_normalise_power)


    ##------------------------------------------------------------------------##
    ##------------------ Functions for updating the state --------------------##
    ##------------------------------------------------------------------------##
    def calc_fft(self):
        """
        Calculate FFT using current state of the window_length, window_function,
        """
        #Segment time-series, overlapping if necessary
        nseg = int(numpy.ceil(self.time_series_length / self.window_length))
        if nseg != 1:
            seg_tpts = self.window_length / self.period
            overlap = ((seg_tpts * nseg) - self.tpts) / (nseg-1)
            starts = [max(seg*(seg_tpts - overlap), 0) for seg in range(nseg)]
            segments = [self.data[start:start+seg_tpts] for start in starts] 
            segments = [segment[:, :, :, numpy.newaxis] for segment in segments]
            time_series = numpy.concatenate(segments, axis=4)
        else:
            time_series = self.data[:, :, :, :, numpy.newaxis]
            seg_tpts = time_series.shape[0]

        #Base-line correct segmented time-series
        time_series = time_series - time_series.mean(axis=0)[numpy.newaxis, :]

        #Apply windowing function
        if self.window_function != "None":
            window_function = eval("".join(("numpy.", self.window_function)))
            window_mask = numpy.reshape(window_function(seg_tpts), 
                                        (seg_tpts, 1, 1, 1, 1))
            time_series = time_series * window_mask

        #Calculate the FFT
        result =  numpy.fft.fft(time_series, axis=0)
        nfreq = len(result)/2

        self.frequency = numpy.arange(0, self.max_freq, self.freq_step)
        LOG.debug("frequency shape: %s" % str(self.frequency.shape))

        self.spectra = numpy.mean(numpy.abs(result[1:nfreq+1])**2, axis=-1)
        LOG.debug("spectra shape: %s" % str(self.spectra.shape))

        self.spectra_norm = (self.spectra / numpy.sum(self.spectra, axis=0))
        LOG.debug("spectra_norm shape: %s" % str(self.spectra_norm.shape))

        #import pdb; pdb.set_trace()
#        self.spectra_std = numpy.std(numpy.abs(result[:nfreq]), axis=4)
#        self.spectra_sem = self.spectra_std / time_series.shape[4]

    ##------------------------------------------------------------------------##
    ##------------------ Functions for updating the figure -------------------##
    ##------------------------------------------------------------------------##
    def update_xscale(self, xscale):
        """ 
        Update the FFT axes' xscale to either log or linear based on radio
        button selection.
        """
        self.xscale = xscale
        self.fft_ax.set_xscale(self.xscale)
        pylab.draw()


    def update_yscale(self, yscale):
        """ 
        Update the FFT axes' yscale to either log or linear based on radio
        button selection.
        """
        self.yscale = yscale
        self.fft_ax.set_yscale(self.yscale)
        pylab.draw()


    def update_mode(self, mode):
        """ Update the visualised mode based on radio button selection. """
        self.mode = mode
        self.plot_spectra()


    def update_variable(self, variable):
        """ 
        Update state variable being plotted based on radio buttton selection.
        """
        self.variable = variable
        self.plot_spectra()


    def update_normalise_power(self, normalise_power):
        """ Update whether to normalise based on radio button selection. """
        self.normalise_power = normalise_power
        self.plot_spectra()


    def update_window_length(self, length):
        """
        Update timeseries window length based on the selected value.
        """
        #TODO: need this casting but not sure why, don't need int() with mode...
        self.window_length = numpy.float64(length) 
        #import pdb; pdb.set_trace()
        self.freq_step = 1.0 / self.window_length
        self.update_spectra()


    def update_window_function(self, window_function):
        """
        Update windowing function based on the radio button selection.
        """
        self.window_function = window_function
        self.update_spectra()


    def update_spectra(self):
        """ Clear the axes and redraw the power-spectra. """
        self.calc_fft()
        self.plot_spectra()


#    def plot_std(self):
#        """ Plot """
#        std = (self.spectra[:, self.variable, :, self.mode] +
#               self.spectra_std[:, self.variable, :, self.mode])
#        self.fft_ax.plot(self.frequency, std, "--")
#    
#    
#    def plot_sem(self):
#        """  """
#        sem = (self.spectra[:, self.variable, :, self.mode] +
#               self.spectra_sem[:, self.variable, :, self.mode])
#        self.fft_ax.plot(self.frequency, sem, ":")


    def plot_spectra(self):
        """ Plot the power spectra. """
        self.fft_ax.clear()
        # Set title and axis labels
        time_series_type = self.time_series.__class__.__name__
        self.fft_ax.set(title = time_series_type)
        self.fft_ax.set(xlabel = "Frequency (%s)" % self.units)
        self.fft_ax.set(ylabel = "Power")

        # Set x and y scale based on curent radio button selection.
        self.fft_ax.set_xscale(self.xscale)
        self.fft_ax.set_yscale(self.yscale)

        if hasattr(self.fft_ax, 'autoscale'):
            self.fft_ax.autoscale(enable=True, axis='both', tight=True)

        #import pdb; pdb.set_trace()
        #Plot the power spectra
        if self.normalise_power == "yes":
            self.fft_ax.plot(self.frequency, 
                             self.spectra_norm[:, self.variable, :, self.mode])
        else:
            self.fft_ax.plot(self.frequency, 
                             self.spectra[:, self.variable, :, self.mode])

#        #TODO: Need to ensure colour matching... and allow region selection.
#        #If requested, add standard deviation
#        if self.show_std:
#            self.plot_std(self)
#            
#        #If requested, add standard error in mean
#        if self.show_sem:
#            self.plot_sem(self)

        pylab.draw()


if __name__ == "__main__":
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    try:
        data = numpy.load("../demos/demo_data_region_16s_2048Hz.npy")
    except IOError:
        LOG.error("Can't load demo data. Run demos/generate_region_demo_data.py")
        raise

    period = 0.00048828125 #NOTE: Providing period in seconds
    tsr = time_series_datatypes.TimeSeriesRegion()
    tsr.data = data
    tsr.sample_period = period

    psi = PowerSpectraInteractive(time_series=tsr)
    psi.show()

