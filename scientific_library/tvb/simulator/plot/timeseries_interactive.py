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
An interactive time-series plot generated from a TVB TimeSeries datatype.

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
    import tvb.simulator.timeseries_interactive as ts_int
    tsi = ts_int.TimeSeriesInteractive(time_series=tsr)
    tsi.configure()
    tsi.show()


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

#TODO: Add state-variable and mode selection RadioButtons

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

TIME_RESOLUTION = 1024 #512 is too coarse; 2048 is a bit slow... ?Make it a traited attribute??

#TODO: check for fence-posts, I did this tired...

class TimeSeriesInteractive(core.Type):
    """
    For generating an interactive time-series figure, given one of TVB's 
    TimeSeries datatypes to initialise it. The graphical interface for 
    visualising a timeseries provides controls for setting:

        - Window length
        - Amplitude scaling
        - Stepping forward/backward through time.


    """

    time_series = time_series_datatypes.TimeSeries(
        label = "Timeseries",
        default = None,
        required = True,
        doc = """The TVB TimeSeries datatype to be displayed.""")

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
        Doc me...

        """
        super(TimeSeriesInteractive, self).__init__(**kwargs) 
        LOG.debug(str(kwargs))

        #figure
        self.its_fig = None

        #time-series
        self.ts_ax = None
        self.ts_view = None
        self.whereami_ax = None
        self.hereiam = None

        #Current state
        self.window_length = None
        self.scaling = 0.42
        self.offset = None
        self.view_step = None
        self.time_view = None
        self.channel_view = None
        #self.mode = 0

        #Selectors
        #self.mode_selector = None

        #Sliders
        self.window_length_slider = None
        self.scaling_slider = None
        self.time_slider = None

        #time-view buttons
        self.step_back_button = None
        self.step_forward_button = None
        self.big_step_back_button = None
        self.big_step_forward_button = None
        self.start_button = None
        self.end_button = None


    def configure(self):
        """ Seperate configure cause ttraits be busted... """
        #TODO: if isinstance(self.time_series, TimeSeriesSurface) and self.first_n == -1: #LOG.error, return.
        self.data = (self.time_series.data[:, :, :self.first_n, :]  -
                     self.time_series.data[:, :, :self.first_n, :].mean(axis=0)[numpy.newaxis, :])
        self.period = self.time_series.sample_period
        self.tpts = self.data.shape[0]
        self.nsrs = self.data.shape[2]
        self.time = numpy.arange(self.tpts) * self.period
        self.start_time = self.time[0]
        self.end_time = self.time[-1]
        self.time_series_length = self.end_time - self.start_time
        self.peak_to_peak = (numpy.max(self.data) - numpy.min(self.data))
        
        #Use actual labels if they exist.
        if (isinstance(self.time_series, time_series_datatypes.TimeSeriesRegion) and 
            (not self.time_series.connectivity is None)):
            self.labels = self.time_series.connectivity.region_labels
        elif (isinstance(self.time_series, (time_series_datatypes.TimeSeriesEEG,
            time_series_datatypes.TimeSeriesMEG) and (not self.time_series.sensors is None))):
            self.labels = self.time_series.sensors.labels
        else:
            self.labels = ["channel_%0.2d"%k for k in range(self.nsrs)] 

        #Current state
        self.window_length = self.tpts * self.period
        self.view_step = max(int(self.tpts / TIME_RESOLUTION), 1)
        self.time_view = range(0, self.tpts, self.view_step)


    def show(self):
        """ Generate the interactive time-series figure. """
        time_series_type = self.time_series.__class__.__name__
        msg = "Generating an interactive time-series plot for %s"
        if isinstance(self.time_series, time_series_datatypes.TimeSeriesSurface):
            LOG.warning("Intended for region and sensors, not surfaces.")
        LOG.info(msg % time_series_type)

        #Make the figure:
        self.create_figure()

        #Selectors
        #self.add_mode_selector()

        #Sliders
        self.add_window_length_slider()
        self.add_scaling_slider()
        #self.add_time_slider()

        #time-view buttons
        self.add_step_back_button()
        self.add_step_forward_button()
        self.add_big_step_back_button()
        self.add_big_step_forward_button()
        self.add_start_button()
        self.add_end_button()

        #Plot timeseries
        self.plot_time_series()

        pylab.show()


    ##------------------------------------------------------------------------##
    ##------------------ Functions for building the figure -------------------##
    ##------------------------------------------------------------------------##
    def create_figure(self):
        """ Create the figure and time-series axes. """
        #time_series_type = self.time_series.__class__.__name__
        try:
            figure_window_title = "Interactive time series: " #+ time_series_type
#            pylab.close(figure_window_title)
            self.its_fig = pylab.figure(num = figure_window_title,
                                        figsize = (14, 8),
                                        facecolor = BACKGROUNDCOLOUR, 
                                        edgecolor = EDGECOLOUR)
        except ValueError:
            LOG.info("My life would be easier if you'd update your PyLab...")
            figure_number = 42
            pylab.close(figure_number)
            self.its_fig = pylab.figure(num = figure_number, 
                                        figsize = (14, 8), 
                                        facecolor = BACKGROUNDCOLOUR, 
                                        edgecolor = EDGECOLOUR)

        self.ts_ax = self.its_fig.add_axes([0.1, 0.1, 0.85, 0.85])

        self.whereami_ax = self.its_fig.add_axes([0.1, 0.95, 0.85, 0.025],
                                                 axisbg=BACKGROUNDCOLOUR)
        self.whereami_ax.set_axis_off()
        if hasattr(self.whereami_ax, 'autoscale'):
            self.whereami_ax.autoscale(enable=True, axis='both', tight=True)
        self.whereami_ax.plot(self.time_view, 
                              numpy.zeros((len(self.time_view),)), 
                              color="0.3", linestyle="--")
        self.hereiam = self.whereami_ax.plot(self.time_view, 
                                             numpy.zeros((len(self.time_view),)), 
                                             'b-', linewidth=4)

#    def add_mode_selector(self):
#        """
#        Add a radio button to the figure for selecting which mode of the model
#        should be displayed.
#        """
#        pos_shp = [0.02, 0.07, 0.04, 0.1+0.002*self.data.shape[3]]]
#        mode_ax = self.its_fig.add_axes(pos_shp, axisbg=AXCOLOUR, title="Mode")
#        mode_tuple = tuple(range(self.model.number_of_modes))
#        self.mode_selector = widgets.RadioButtons(mode_ax, mode_tuple, active=0)
#        self.mode_selector.on_clicked(self.update_mode)


#    def add_time_sliders(self):
#        """
#        Add a slider to allow the time-series window length to be adjusted.
#        """
#        pos_shp = [0.2, 0.02, 0.7, 0.025]
#        slax = self.its_fig.add_axes(pos_shp, axisbg=AXCOLOUR)
#        
#        self.current_time_slider = widgets.Slider(slax, "Time", self.start_time,
#                                          self.end_time, 
#                                          valinit = self.current_time)
#        self.current_time.on_changed(self.update_time)


    def add_window_length_slider(self):
        """
        Add a slider to allow the time-series window length to be adjusted.
        """
        pos_shp = [0.15, 0.02, 0.175, 0.035]
        slax = self.its_fig.add_axes(pos_shp, axisbg=AXCOLOUR)

        self.window_length_slider = widgets.Slider(slax, "Window length", 
                                                   TIME_RESOLUTION*self.period, 
                                                   self.time_series_length,
                                                   valinit = self.window_length,
                                                   valfmt = "%d")
        self.window_length_slider.on_changed(self.update_window_length)

    #TODO: Add a conversion so this is an amplitude scaling, say 1.0-20.0
    def add_scaling_slider(self):
        """ Add a slider to allow scaling of the offset of time-series. """
        pos_shp = [0.75, 0.02, 0.175, 0.035]
        sax = self.its_fig.add_axes(pos_shp, axisbg=AXCOLOUR)

        self.scaling_slider = widgets.Slider(sax, "Spacing", 0.0, 1.25, 
                                             valinit = self.scaling,
                                             valfmt = "%4.2f")
        self.scaling_slider.on_changed(self.update_scaling)


    def add_step_back_button(self):
        """ Add a button to step back by 4 view_steps. """
        bax = self.its_fig.add_axes([0.5, 0.015, 0.04, 0.045])
        self.step_back_button = widgets.Button(bax, '<', color=BUTTONCOLOUR, 
                                               hovercolor=HOVERCOLOUR)

        self.step_back_button.on_clicked(self.step_back)


    def add_step_forward_button(self):
        """ Add a button to step forward by 4 view_steps. """
        bax = self.its_fig.add_axes([0.54, 0.015, 0.04, 0.045])
        self.step_forward_button = widgets.Button(bax, '>', color=BUTTONCOLOUR, 
                                                  hovercolor=HOVERCOLOUR)

        self.step_forward_button.on_clicked(self.step_forward)


    def add_big_step_back_button(self):
        """ Add a button to step back by 1/4 window_length. """
        bax = self.its_fig.add_axes([0.46, 0.015, 0.04, 0.045])
        self.big_step_back_button = widgets.Button(bax, '<<',
                                                   color=BUTTONCOLOUR, 
                                                   hovercolor=HOVERCOLOUR)

        self.big_step_back_button.on_clicked(self.bigstep_back)


    def add_big_step_forward_button(self):
        """ Add a button to step forward by 1/4 window_length. """
        bax = self.its_fig.add_axes([0.58, 0.015, 0.04, 0.045])
        self.big_step_forward_button = widgets.Button(bax, '>>',
                                                      color=BUTTONCOLOUR, 
                                                      hovercolor=HOVERCOLOUR)

        self.big_step_forward_button.on_clicked(self.bigstep_forward)


    def add_start_button(self):
        """ Add a button to jump back to the start of the timeseries. """
        bax = self.its_fig.add_axes([0.42, 0.015, 0.04, 0.045])
        self.start_button = widgets.Button(bax, '|<<<', color=BUTTONCOLOUR, 
                                           hovercolor=HOVERCOLOUR)

        self.start_button.on_clicked(self.jump_to_start)


    def add_end_button(self):
        """ Add a button to jump forward to the end of the timeseries. """
        bax = self.its_fig.add_axes([0.62, 0.015, 0.04, 0.045])
        self.end_button = widgets.Button(bax, '>>>|', color=BUTTONCOLOUR, 
                                         hovercolor=HOVERCOLOUR)

        self.end_button.on_clicked(self.jump_to_end)


    ##------------------------------------------------------------------------##
    ##------------------ Functions for updating the state --------------------##
    ##------------------------------------------------------------------------##

    def step_back(self, event=None):
        """ Step the timeview back by a single view step. """
        LOG.debug("step_back accessed with event: %s" % str(event))
        step = 4*self.view_step
        if self.time_view[0]-step >= 0:
            self.time_view = [k-step for k in self.time_view]
            self.update_time_series()


    def step_forward(self, event=None):
        """ Step the timeview forward by a single view step. """
        LOG.debug("step_forward accessed with event: %s" % str(event))
        step = 4*self.view_step
        if self.time_view[-1]+step < self.tpts:
            self.time_view = [k+step for k in self.time_view]
            self.update_time_series()


    def bigstep_back(self, event=None):
        """ Step the timeview back by 1/4 window length. """
        LOG.debug("bigstep_back accessed with event: %s" % str(event))
        step = self.view_step * TIME_RESOLUTION / 4
        if self.time_view[0]-step >= 0:
            self.time_view = [k-step for k in self.time_view]
            self.update_time_series()
        else:
            self.jump_to_start()


    def bigstep_forward(self, event=None):
        """ Step the timeview forward by 1/4 window length. """
        LOG.debug("bigstep_forward accessed with event: %s" % str(event))
        step = self.view_step * TIME_RESOLUTION / 4
        if self.time_view[-1]+step < self.tpts:
            self.time_view = [k+step for k in self.time_view]
            self.update_time_series()
        else:
            self.jump_to_end()


    def jump_to_start(self, event=None):
        """ Jump to the start of the timeseries. """
        LOG.debug("jump_to_start accessed with event: %s" % str(event))
        step = self.time_view[0]
        self.time_view = [k-step for k in self.time_view]
        self.update_time_series()


    def jump_to_end(self, event=None):
        """ Jump to the end of the timeseries."""
        LOG.debug("jump_to_end accessed with event: %s" % str(event))
        step = self.tpts-1 - self.time_view[-1]
        self.time_view = [k+step for k in self.time_view]
        self.update_time_series()


    def update_time_view(self):
        """ Update the time_view when window length is changed. """
        tpts = self.window_length / self.period
        self.view_step = max(int(tpts / TIME_RESOLUTION), 1)
        window_start = self.time_view[0]
        window_end = min(window_start + self.view_step * (TIME_RESOLUTION-1),
                         self.tpts)
        self.time_view = range(window_start, window_end, self.view_step)


    ##------------------------------------------------------------------------##
    ##------------------ Functions for updating the figure -------------------##
    ##------------------------------------------------------------------------##
#    def update_mode(self, label):
#        """ Update the visualised mode based on radio button selection. """
#        self.mode = label
#        self.update_time_series()


    def update_window_length(self, length):
        """
        Update timeseries window length based on the time window slider value.
        """
        self.window_length = length
        self.update_time_view()
        self.update_time_series()


    def update_scaling(self, scaling):
        """
        Update timeseries scaling based on the scaling slider value.
        """
        self.scaling = scaling
        self.update_time_series()


    def update_time_series(self):
        """ Clear the axes and redraw the time-series. """
        self.ts_ax.clear()
        self.plot_time_series()


    def plot_time_series(self):
        """ Plot a view on the timeseries. """
        # Set title and axis labels
        #time_series_type = self.time_series.__class__.__name__
        #self.ts_ax.set(title = time_series_type)
        #self.ts_ax.set(xlabel = "Time (%s)" % self.units)

        # This assumes shape => (time, space)
        step = self.scaling * self.peak_to_peak
        if step == 0:
            offset = 0.0
        else: #NOTE: specifying step in arange is faster, but it fence-posts.
            offset = numpy.arange(0, self.nsrs) * step
        if hasattr(self.ts_ax, 'autoscale'):
            self.ts_ax.autoscale(enable=True, axis='both', tight=True)

        self.ts_ax.set_yticks(offset)
        self.ts_ax.set_yticklabels(self.labels, fontsize=10)
        #import pdb; pdb.set_trace()

        #Light gray guidelines
        self.ts_ax.plot([self.nsrs*[self.time[self.time_view[0]]], 
                         self.nsrs*[self.time[self.time_view[-1]]]], 
                        numpy.vstack(2*(offset,)), "0.85")

        #Plot the timeseries
        self.ts_view = self.ts_ax.plot(self.time[self.time_view], 
                                       offset + self.data[self.time_view, 0, :, 0])

        self.hereiam[0].remove()
        self.hereiam = self.whereami_ax.plot(self.time_view, 
                                             numpy.zeros((len(self.time_view),)), 
                                             'b-', linewidth=4)

        pylab.draw()


if __name__ == "__main__":
    # Do some stuff that tests or makes use of this module...
    LOG.info("Testing %s module..." % __file__)
    try:
        data = numpy.load("../demos/demo_data_region_16s_2048Hz.npy") #
    except IOError:
        LOG.error("Can't load demo data. Run demos/generate_region_demo_data.py")
        raise

    period = 0.00048828125 #NOTE: Providing period in s
    tsr = time_series_datatypes.TimeSeriesRegion()
    tsr.data = data
    tsr.sample_period = period

    tsi = TimeSeriesInteractive(time_series=tsr)
    tsi.configure()
    tsi.show()


