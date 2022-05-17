# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
    from tvb.datatypes.connectivity import Connectivity
    tsr.connectivity = Connectivity.from_file()

    #Create and launch the interactive visualiser
    import tvb.simulator.plot.timeseries_interactive as ts_int
    tsi = ts_int.TimeSeriesInteractive(time_series=tsr)
    tsi.configure()
    tsi.show()


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

import numpy
import ipywidgets as widgets
import matplotlib.pyplot as plt
import tvb.datatypes.time_series as time_series_datatypes
from IPython.display import display
from matplotlib import rcParams
from tvb.basic.neotraits.api import HasTraits, Attr, Int
from tvb.simulator.common import get_logger
from tvb.simulator.plot.utils import ensure_list, rotate_n_list_elements

LOG = get_logger(__name__)
# Define a colour theme... see: matplotlib.colors.cnames.keys()
BACKGROUNDCOLOUR = "white"
EDGECOLOUR = "darkslateblue"
AXCOLOUR = "steelblue"
BUTTONCOLOUR = "steelblue"
HOVERCOLOUR = "blue"
MARGIN_2PX = '2px 2px 2px 2px'
MARGIN_3PX = '3px 3px 3px 3px'
BORDER_BLACK = 'solid 1px black'
TIME_RESOLUTION = 1024  # 512 is too coarse; 2048 is a bit slow... ?Make it a traited attribute??


class TimeSeriesInteractive(HasTraits):
    """
    For generating an interactive time-series figure, given one of TVB's 
    TimeSeries datatypes to initialise it. The graphical interface for 
    visualising a timeseries provides controls for setting:

        - Window length
        - Amplitude scaling
        - Stepping forward/backward through time.
    """

    time_series = Attr(
        field_type=time_series_datatypes.TimeSeries,
        label="Timeseries",
        default=None,
        doc="""The TVB TimeSeries datatype to be displayed.""")

    first_n = Int(
        label="Display the first 'n'",
        default=-1,
        doc="""Primarily intended for displaying the first N components of a 
            surface PCA timeseries. Defaults to -1, meaning it'll display all
            of 'space' (ie, regions or vertices or channels). In other words,
            for Region or M/EEG timeseries you can ignore this, but, for a 
            surface timeseries it really must be set.""")

    def __init__(self, **kwargs):
        super(TimeSeriesInteractive, self).__init__(**kwargs)
        LOG.debug(str(kwargs))

        # figure
        self.its_fig = None

        # t ime-series
        self.ts_ax = None
        self.ts_view = None
        self.whereami_ax = None
        self.hereiam = None

        # Current state
        self.window_length = None
        self.scaling = 0.42
        self.view_step = None
        self.time_view = None
        self.channel_view = None
        # self.mode = 0

        # Selectors
        # self.mode_selector = None

        # Sliders
        self.window_length_slider = None
        self.scaling_slider = None
        self.time_slider = None

        # time-view buttons
        self.step_back_button = None
        self.step_forward_button = None
        self.big_step_back_button = None
        self.big_step_forward_button = None
        self.start_button = None
        self.end_button = None

        self.region_cb_stack = list()

    def configure(self):
        """ Prepare for usage """
        self.data = (self.time_series.data[:, :, :, :] -
                     self.time_series.data[:, :, :, :].mean(axis=0)[numpy.newaxis, :])
        self.period = self.time_series.sample_period
        self.tpts = self.data.shape[0]
        self.nsrs = self.data.shape[2]
        self.time = numpy.arange(self.tpts) * self.period
        self.start_time = self.time[0]
        self.end_time = self.time[-1]
        self.time_series_length = self.end_time - self.start_time
        self.peak_to_peak = (numpy.max(self.data) - numpy.min(self.data))

        # Use actual labels if they exist.
        if (isinstance(self.time_series, time_series_datatypes.TimeSeriesRegion) and
                (not self.time_series.connectivity is None)):
            self.labels = self.time_series.connectivity.region_labels
        elif (isinstance(self.time_series, (time_series_datatypes.TimeSeriesEEG,
                                            time_series_datatypes.TimeSeriesMEG)) and (
                      not self.time_series.sensors is None)):
            self.labels = self.time_series.sensors.labels
        else:
            self.labels = ["channel_%0.2d" % k for k in range(self.nsrs)]

            # Current state
        self.window_length = self.tpts * self.period
        self.view_step = max(int(self.tpts / TIME_RESOLUTION), 1)
        self.time_view = list(range(0, self.tpts, self.view_step))

        self.plot_labels = self.labels

    def show(self, block=True, **kwargs):
        """ Generate the interactive time-series figure. """
        time_series_type = self.time_series.__class__.__name__
        msg = "Generating an interactive time-series plot for %s"
        if isinstance(self.time_series, time_series_datatypes.TimeSeriesSurface):
            LOG.warning("Intended for region and sensors, not surfaces.")
        LOG.info(msg % time_series_type)

        # Make the figure:
        fig = self.create_figure()

        # Plot timeseries
        self.plot_time_series()

        display(fig)

    def ensure_list(self, arg):
        if not (isinstance(arg, list)):
            if hasattr(arg, '__iter__'):
                arg = list(arg)
            else:
                arg = [arg]
        return arg

    def rotate_n_list_elements(self, lst, n):
        lst = self.ensure_list(lst)
        n_lst = len(lst)
        if n_lst != n and n_lst != 0:
            if n_lst == 1:
                lst *= n
            elif n_lst > n:
                lst = lst[:n]
            else:
                old_lst = list(lst)
                while n_lst < n:
                    lst += old_lst[0]
                    old_lst = old_lst[1:] + old_lst[:1]
        return lst

    # ------------------------------------------------------------------------##
    # ------------------ Functions for building the figure -------------------##
    # ------------------------------------------------------------------------##
    def _create_inner_figure(self):
        """ Create the figure - common part with subclass"""
        self.outer_box_layout = widgets.Layout(border='solid 1px white',
                                               margin=MARGIN_3PX,
                                               padding=MARGIN_2PX)
        self.box_layout = widgets.Layout(border=BORDER_BLACK,
                                         margin=MARGIN_3PX,
                                         padding=MARGIN_2PX)
        self.button_layout = widgets.Layout(width='auto')

        self.its_fig = None
        self.out = widgets.Output()
        self.out.layout = self.box_layout

        # WIDGETS
        self.control_widgets = []
        self.time_control_buttons = []

        # Sliders
        self.add_window_length_slider()

        # time-view buttons
        self.add_start_button()
        self.add_big_step_back_button()
        self.add_step_back_button()
        self.add_step_forward_button()
        self.add_big_step_forward_button()
        self.add_end_button()

        self.tc_box = widgets.HBox(self.time_control_buttons, layout=self.outer_box_layout)
        self.control_widgets.append(self.tc_box)

        self.add_scaling_slider()

    def create_figure(self):
        """ Create the figure and time-series axes. """
        self._create_inner_figure()

        self.control_box = widgets.GridBox(self.control_widgets,
                                           layout=widgets.Layout(grid_template_columns="290px 350px 290px",
                                                                 border=BORDER_BLACK))
        self.region_select_buttons = []
        self.add_region_selectall_button()
        self.add_region_unselectall_button()
        self.add_region_checkboxes()

        self.space_label = widgets.Label('Space labels selection')
        self.label_buttons_box = widgets.HBox([self.space_label] + self.region_select_buttons)
        self.region_box = widgets.HBox(self.region_cb_box_list)
        self.space_labels_box = widgets.VBox([self.label_buttons_box, self.region_box], layout=self.box_layout)

        items = [self.out, self.control_box, self.space_labels_box]
        grid = widgets.GridBox(items, layout=widgets.Layout(grid_template_rows="570px 75px 365px"))

        return grid

    #    def add_mode_selector(self):
    #        """
    #        Add a radio button to the figure for selecting which mode of the model
    #        should be displayed.
    #        """
    #        pos_shp = [0.02, 0.07, 0.04, 0.1+0.002*self.data.shape[3]]]
    #        mode_ax = self.its_fig.add_axes(pos_shp, facecolor=AXCOLOUR, title="Mode")
    #        mode_tuple = tuple(range(self.model.number_of_modes))
    #        self.mode_selector = widgets.RadioButtons(mode_ax, mode_tuple, active=0)
    #        self.mode_selector.on_clicked(self.update_mode)

    #    def add_time_sliders(self):
    #        """
    #        Add a slider to allow the time-series window length to be adjusted.
    #        """
    #        pos_shp = [0.2, 0.02, 0.7, 0.025]
    #        slax = self.its_fig.add_axes(pos_shp, facecolor=AXCOLOUR)
    #
    #        self.current_time_slider = widgets.Slider(slax, "Time", self.start_time,
    #                                          self.end_time,
    #                                          valinit = self.current_time)
    #        self.current_time.on_changed(self.update_time)

    def add_region_checkboxes(self):

        self.region_checkboxes = dict()
        self.region_cb_box_list = []
        cb_mw = 950 // (len(self.labels) // 10 + 1)
        for i, label in enumerate(self.labels):
            self.region_checkboxes[label] = widgets.Checkbox(value=True,
                                                             description=label,
                                                             disabled=False,
                                                             indent=False,
                                                             layout={'max_width': f'{cb_mw}px'})
            self.region_checkboxes[label].observe(self.update_region_plots, names=['value'])

            if i and i % 10 == 0:
                self.region_cb_box_list.append(widgets.VBox(self.region_cb_stack))
                self.region_cb_stack = []
            self.region_cb_stack.append(self.region_checkboxes[label])
        self.region_cb_box_list.append(widgets.VBox(self.region_cb_stack))

    def add_region_selectall_button(self):

        def update_select_region_selectors(_):
            for checkbox in self.region_checkboxes:
                self.region_checkboxes[checkbox].value = True

        self.region_checkbox_select_button = widgets.Button(description='Select All', layout={'description': 'initial'})
        self.region_checkbox_select_button.on_click(update_select_region_selectors)
        self.region_select_buttons.append(self.region_checkbox_select_button)

    def add_region_unselectall_button(self):
        def update_unselect_region_selectors(_):
            for checkbox in self.region_checkboxes:
                self.region_checkboxes[checkbox].value = False

        self.region_checkbox_unselect_button = widgets.Button(description='Unselect All',
                                                              layout={'description': 'initial'})
        self.region_checkbox_unselect_button.on_click(update_unselect_region_selectors)
        self.region_select_buttons.append(self.region_checkbox_unselect_button)

    def add_window_length_slider(self):
        """
        Add a slider to allow the time-series window length to be adjusted.
        """
        self.window_length_slider = widgets.IntSlider(min=TIME_RESOLUTION * self.period,
                                                      max=self.time_series_length,
                                                      value=self.window_length,
                                                      layout=widgets.Layout(width="90%"),
                                                      continuous_update=False)
        self.window_length_slider.observe(self.update_window_length, names=['value'])
        self.wls_label = widgets.Label("Window Length")
        self.wls_box = widgets.VBox([self.wls_label, self.window_length_slider], layout=self.outer_box_layout)

        self.control_widgets.append(self.wls_box)

    # TODO: Add a conversion so this is an amplitude scaling, say 1.0-20.0
    def add_scaling_slider(self):
        """ Add a slider to allow scaling of the offset of time-series. """
        self.scaling_slider = widgets.FloatSlider(min=0.0,
                                                  max=1.25,
                                                  value=self.scaling,
                                                  layout=widgets.Layout(width="90%"),
                                                  continuous_update=False)
        self.scaling_slider.observe(self.update_scaling, names=['value'])
        self.ss_label = widgets.Label("Spacing")
        self.ss_box = widgets.VBox([self.ss_label, self.scaling_slider], layout=self.outer_box_layout)

        self.control_widgets.append(self.ss_box)

    def add_step_back_button(self):
        """ Add a button to step back by 4 view_steps. """
        self.step_back_button = widgets.Button(description='<', layout=self.button_layout)

        self.step_back_button.on_click(self.step_back)
        self.time_control_buttons.append(self.step_back_button)

    def add_step_forward_button(self):
        """ Add a button to step forward by 4 view_steps. """
        self.step_forward_button = widgets.Button(description='>', layout=self.button_layout)

        self.step_forward_button.on_click(self.step_forward)
        self.time_control_buttons.append(self.step_forward_button)

    def add_big_step_back_button(self):
        """ Add a button to step back by 1/4 window_length. """
        self.big_step_back_button = widgets.Button(description='<<', layout=self.button_layout)

        self.big_step_back_button.on_click(self.bigstep_back)
        self.time_control_buttons.append(self.big_step_back_button)

    def add_big_step_forward_button(self):
        """ Add a button to step forward by 1/4 window_length. """
        self.big_step_forward_button = widgets.Button(description='>>', layout=self.button_layout)

        self.big_step_forward_button.on_click(self.bigstep_forward)
        self.time_control_buttons.append(self.big_step_forward_button)

    def add_start_button(self):
        """ Add a button to jump back to the start of the timeseries. """
        self.start_button = widgets.Button(description='|<<<', layout=self.button_layout)

        self.start_button.on_click(self.jump_to_start)
        self.time_control_buttons.append(self.start_button)

    def add_end_button(self):
        """ Add a button to jump forward to the end of the timeseries. """
        self.end_button = widgets.Button(description='>>>|', layout=self.button_layout)

        self.end_button.on_click(self.jump_to_end)
        self.time_control_buttons.append(self.end_button)

    # ------------------------------------------------------------------------##
    # ------------------ Functions for updating the state --------------------##
    # ------------------------------------------------------------------------##

    def step_back(self, event=None):
        """ Step the timeview back by a single view step. """
        LOG.debug("step_back accessed with event: %s" % str(event))
        step = 4 * self.view_step
        if self.time_view[0] - step >= 0:
            self.time_view = [k - step for k in self.time_view]
            self.update_time_series()

    def step_forward(self, event=None):
        """ Step the timeview forward by a single view step. """
        LOG.debug("step_forward accessed with event: %s" % str(event))
        step = 4 * self.view_step
        if self.time_view[-1] + step < self.tpts:
            self.time_view = [k + step for k in self.time_view]
            self.update_time_series()

    def bigstep_back(self, event=None):
        """ Step the timeview back by 1/4 window length. """
        LOG.debug("bigstep_back accessed with event: %s" % str(event))
        step = int(self.view_step * TIME_RESOLUTION / 4)
        if self.time_view[0] - step >= 0:
            self.time_view = [k - step for k in self.time_view]
            self.update_time_series()
        else:
            self.jump_to_start()

    def bigstep_forward(self, event=None):
        """ Step the timeview forward by 1/4 window length. """
        LOG.debug("bigstep_forward accessed with event: %s" % str(event))
        step = int(self.view_step * TIME_RESOLUTION / 4)
        if self.time_view[-1] + step < self.tpts:
            self.time_view = [k + step for k in self.time_view]
            self.update_time_series()
        else:
            self.jump_to_end()

    def jump_to_start(self, event=None):
        """ Jump to the start of the timeseries. """
        LOG.debug("jump_to_start accessed with event: %s" % str(event))
        step = self.time_view[0]
        self.time_view = [k - step for k in self.time_view]
        self.update_time_series()

    def jump_to_end(self, event=None):
        """ Jump to the end of the timeseries."""
        LOG.debug("jump_to_end accessed with event: %s" % str(event))
        step = self.tpts - 1 - self.time_view[-1]
        self.time_view = [k + step for k in self.time_view]
        self.update_time_series()

    def update_time_view(self):
        """ Update the time_view when window length is changed. """
        tpts = self.window_length / self.period
        self.view_step = max(int(tpts / TIME_RESOLUTION), 1)
        window_start = self.time_view[0]
        window_end = min(window_start + self.view_step * (TIME_RESOLUTION - 1),
                         self.tpts)
        self.time_view = list(range(window_start, window_end, self.view_step))

    def update_region_plots(self, _):
        self.plot_labels = []
        for cb in list(self.region_checkboxes.values()):
            if cb.value:
                self.plot_labels.append(cb.description)
        self.update_time_series()

    # ------------------------------------------------------------------------##
    # ------------------ Functions for updating the figure -------------------##
    # ------------------------------------------------------------------------##

    def update_window_length(self, length):
        """
        Update timeseries window length based on the time window slider value.
        """
        self.window_length = length['new']
        self.update_time_view()
        self.update_time_series()

    def update_scaling(self, scaling):
        """
        Update timeseries scaling based on the scaling slider value.
        """
        self.scaling = scaling['new']
        self.update_time_series()

    def update_time_series(self):
        """ Clear the axes and redraw the time-series. """
        self.ts_ax.clear()
        self.plot_time_series()

    def _plot_inner_ts(self, fig_size=(9, 5), ts_axes=[0.1, 0.05, 0.85, 0.90]):
        """ Plot a view on the timeseries - common part with subclass. """
        self.out.clear_output(wait=True)
        with plt.ioff():
            if not self.its_fig:
                try:
                    figure_window_title = "Interactive time series"
                    self.its_fig = plt.figure(num=figure_window_title,
                                              figsize=fig_size,
                                              facecolor=BACKGROUNDCOLOUR,
                                              edgecolor=EDGECOLOUR)
                except ValueError:
                    LOG.info("My life would be easier if you'd update your Pyplot...")
                    figure_number = 42
                    plt.close(figure_number)
                    self.its_fig = plt.figure(num=figure_number,
                                              figsize=fig_size,
                                              facecolor=BACKGROUNDCOLOUR,
                                              edgecolor=EDGECOLOUR)

                self.ts_ax = self.its_fig.add_axes(ts_axes)

                self.whereami_ax = self.its_fig.add_axes([0.1, 0.95, 0.85, 0.025],
                                                         facecolor=BACKGROUNDCOLOUR)
                self.whereami_ax.set_axis_off()
                if hasattr(self.whereami_ax, 'autoscale'):
                    self.whereami_ax.autoscale(enable=True, axis='both', tight=True)
                self.whereami_ax.plot(self.time_view,
                                      numpy.zeros((len(self.time_view),)),
                                      color="0.3", linestyle="--")
                self.hereiam = self.whereami_ax.plot(self.time_view,
                                                     numpy.zeros((len(self.time_view),)),
                                                     'b-', linewidth=4)

    def plot_time_series(self, **kwargs):
        """ Plot a view on the timeseries. """
        self._plot_inner_ts()

        # This assumes shape => (time, space)
        step = self.scaling * self.peak_to_peak
        if hasattr(self.ts_ax, 'autoscale'):
            self.ts_ax.autoscale(enable=True, axis='both', tight=True)

        num_labels = len(self.plot_labels)
        if num_labels:
            offset = numpy.arange(0, num_labels) * step
            self.ts_ax.set_yticks(offset)
            self.ts_ax.set_yticklabels(self.labels[:num_labels], fontsize=10)

            # Light gray guidelines
            self.ts_ax.plot([num_labels * [self.time[self.time_view[0]]],
                             num_labels * [self.time[self.time_view[-1]]]],
                            numpy.vstack(2 * (offset,)), "0.85")

            # Determine colors and linestyles for each variable of the Timeseries
            linestyle = self.ensure_list(kwargs.pop("linestyle", "-"))
            colors = kwargs.pop("linestyle", None)
            if colors is not None:
                colors = self.ensure_list(colors)
            if self.data.shape[1] > 1:
                linestyle = self.rotate_n_list_elements(linestyle, self.data.shape[1])
                if not isinstance(colors, list):
                    colors = (rcParams['axes.prop_cycle']).by_key()['color']
                colors = self.rotate_n_list_elements(colors, self.data.shape[1])
            else:
                # If no color or a color sequence is given in the input, but there is only one variable to plot,
                # choose the black color
                if colors is None or len(colors) > 1:
                    colors = ["k"]
                linestyle = linestyle[:1]

            # Determine the alpha value depending on the number of modes/samples of the Timeseries
            alpha = 1.0
            if len(self.data.shape) > 3 and self.data.shape[3] > 1:
                alpha /= self.data.shape[3]

            # Plot the timeseries (per variable and sample)
            if kwargs:
                self.ts_view = []
                for variable_value in range(self.data.shape[1]):
                    for sample_value in range(self.data.shape[3]):
                        self.ts_view.append(self.ts_ax.plot(self.time[self.time_view],
                                                            offset + self.data[self.time_view, variable_value, :,
                                                                     sample_value],
                                                            alpha=alpha, color=colors[variable_value],
                                                            linestyle=linestyle[variable_value],
                                                            **kwargs))
            else:
                self.ts_view = self.ts_ax.plot(self.time[self.time_view],
                                               offset + self.data[self.time_view, 0, :num_labels, 0])

            self.hereiam[0].remove()
            self.hereiam = self.whereami_ax.plot(self.time_view,
                                                 numpy.zeros((len(self.time_view),)),
                                                 'b-', linewidth=4)

            with self.out:
                display(self.its_fig.canvas)


class TimeSeriesInteractivePlotter(TimeSeriesInteractive):

    def create_figure(self, **kwargs):
        """ Create the figure and time-series axes. """

        self._create_inner_figure()

        self.control_box = widgets.GridBox(self.control_widgets,
                                           layout=widgets.Layout(grid_template_columns="300px 350px 300px",
                                                                 border=BORDER_BLACK))

        items = [self.out, self.control_box]
        grid = widgets.GridBox(items, layout=widgets.Layout(grid_template_rows="800px 100px"))

        return grid

    def plot_time_series(self, **kwargs):
        """ Plot a view on the timeseries. """
        self._plot_inner_ts(fig_size=(9, 7), ts_axes=[0.1, 0.1, 0.85, 0.85])

        # This assumes shape => (time, space)
        step = self.scaling * self.peak_to_peak
        if step == 0:
            offset = 0.0
        else:  # NOTE: specifying step in arange is faster, but it fence-posts.
            offset = numpy.arange(0, self.nsrs) * step
        if hasattr(self.ts_ax, 'autoscale'):
            self.ts_ax.autoscale(enable=True, axis='both', tight=True)

        self.ts_ax.set_yticks(offset)
        self.ts_ax.set_yticklabels(self.labels, fontsize=10)

        # Light gray guidelines
        self.ts_ax.plot([self.nsrs * [self.time[self.time_view[0]]],
                         self.nsrs * [self.time[self.time_view[-1]]]],
                        numpy.vstack(2 * (offset,)), "0.85")

        # Determine colors and linestyles for each variable of the Timeseries
        linestyle = ensure_list(kwargs.pop("linestyle", "-"))
        colors = kwargs.pop("linestyle", None)
        if colors is not None:
            colors = ensure_list(colors)
        if self.data.shape[1] > 1:
            linestyle = rotate_n_list_elements(linestyle, self.data.shape[1])
            if not isinstance(colors, list):
                colors = (rcParams['axes.prop_cycle']).by_key()['color']
            colors = rotate_n_list_elements(colors, self.data.shape[1])
        else:
            # If no color,
            # or a color sequence is given in the input
            # but there is only one variable to plot,
            # choose the black color
            if colors is None or len(colors) > 1:
                colors = ["k"]
            linestyle = linestyle[:1]

        # Determine the alpha value depending on the number of modes/samples of the Timeseries
        alpha = 1.0
        if len(self.data.shape) > 3 and self.data.shape[3] > 1:
            alpha /= self.data.shape[3]

        # Plot the timeseries per variable and sample
        self.ts_view = []
        for i_var in range(self.data.shape[1]):
            for ii in range(self.data.shape[3]):
                # Plot the timeseries
                self.ts_view.append(self.ts_ax.plot(self.time[self.time_view],
                                                    offset + self.data[self.time_view, i_var, :, ii],
                                                    alpha=alpha, color=colors[i_var], linestyle=linestyle[i_var],
                                                    **kwargs))

        self.hereiam[0].remove()
        self.hereiam = self.whereami_ax.plot(self.time_view,
                                             numpy.zeros((len(self.time_view),)),
                                             'b-', linewidth=4)

        with self.out:
            display(self.its_fig.canvas)
