# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Lia Domide <lia.domide@codemart.ro>
.. moduleauthor:: Stuart A. Knock <stuart.knock@gmail.com>

"""
import numpy
from matplotlib.widgets import RadioButtons
from tvb.core.adapters.abcdisplayer import ABCMPLH5Displayer
from tvb.datatypes.spectral import FourierSpectrum


BACKGROUNDCOLOUR = "slategrey"
EDGECOLOUR = "darkslateblue"
AXCOLOUR = "steelblue"
BUTTONCOLOUR = "steelblue"
HOVERCOLOUR = "blue"

CONTOLS_START_X = 0.02
CONTROLS_WIDTH = 0.06
CONTROLS_HEIGHT = 0.104

NR_OF_PREVIEW_CHANS = 5



class FourierSpectrumDisplay(ABCMPLH5Displayer):
    """
    This viewer takes as inputs a result form FFT analysis, and returns
    required parameters for a MatplotLib representation.
    """

    _ui_name = "Fourier Visualizer"
    _ui_subsection = "fourier"


    def get_input_tree(self):
        """ 
        Accept as input result from FFT Analysis.
        """
        return [{'name': 'input_data', 'label': 'Fourier Result',
                 'type': FourierSpectrum, 'required': True,
                 'description': 'Fourier Analysis to display'}]


    def get_required_memory_size(self, **kwargs):
        """
        Return the required memory to run this algorithm.
        """
        return numpy.prod(kwargs['input_data'].read_data_shape()) * 8


    def plot(self, figure, **kwargs):
        """
        Draw interactive display.
        """
        self.log.debug("Plot started...")
        self.input_data = kwargs['input_data']
        self.figure = figure
        figure.facecolor = BACKGROUNDCOLOUR
        figure.edgecolor = EDGECOLOUR
        self.axes = figure.add_axes([CONTOLS_START_X + CONTROLS_WIDTH + 0.065, 0.07, 0.85, 0.85])

        self.xscale = "log"
        self.yscale = "log"
        self.mode = 0
        self.variable = 0
        self.normalise_power = "no"

        #Selectors
        if not self.is_preview:
            self._add_xscale_selector()
            self._add_yscale_selector()
            self._add_mode_selector()
            self._add_variable_selector()
            self._add_normalise_power_selector()
        #Plot timeSeries
        self.plot_spectra()


    #------------------------------- X_SCALE ---------------------------------#
    def _add_xscale_selector(self):
        """
        Add a radio button to the figure for selecting which scaling the x-axes
        should use.
        """
        pos_shp = [CONTOLS_START_X, 0.8, CONTROLS_WIDTH, CONTROLS_HEIGHT]
        rax = self.figure.add_axes(pos_shp, axisbg=AXCOLOUR, title="X Scale")
        xscale_tuple = ("log", "linear")
        self.xscale_selector = RadioButtons(rax, xscale_tuple, active=xscale_tuple.index(self.xscale))
        self.xscale_selector.on_clicked(self._update_xscale)


    def _update_xscale(self, xscale):
        """ 
        Update the FFT axes' xscale to either log or linear based on radio
        button selection.
        """
        self.xscale = xscale
        self.axes.set_xscale(self.xscale)
        self.figure.canvas.draw()


    #------------------------------- Y_SCALE ---------------------------------#
    def _add_yscale_selector(self):
        """
        Add a radio button to the figure for selecting which scaling the y-axes
        should use.
        """
        pos_shp = [CONTOLS_START_X, 0.65, CONTROLS_WIDTH, CONTROLS_HEIGHT]
        rax = self.figure.add_axes(pos_shp, axisbg=AXCOLOUR, title="Y Scale")
        yscale_tuple = ("log", "linear")
        self.yscale_selector = RadioButtons(rax, yscale_tuple, active=yscale_tuple.index(self.yscale))
        self.yscale_selector.on_clicked(self._update_yscale)


    def _update_yscale(self, yscale):
        """ 
        Update the FFT axes' yscale to either log or linear based on radio
        button selection.
        """
        self.yscale = yscale
        self.axes.set_yscale(self.yscale)
        self.figure.canvas.draw()


    #---------------------------- MODE SELECTOR ------------------------------#
    def _add_mode_selector(self):
        """
        Add a radio button to the figure for selecting which mode of the model
        should be displayed.
        """
        pos_shp = [CONTOLS_START_X, 0.07, CONTROLS_WIDTH, 0.1 + 0.002 * self.input_data.read_data_shape()[3]]
        rax = self.figure.add_axes(pos_shp, axisbg=AXCOLOUR, title="Mode")
        mode_tuple = tuple(range(self.input_data.read_data_shape()[3]))
        self.mode_selector = RadioButtons(rax, mode_tuple, active=mode_tuple.index(self.mode))
        self.mode_selector.on_clicked(self._update_mode)


    def _update_mode(self, mode):
        """ Update the visualized mode based on radio button selection. """
        self.mode = int(mode)
        self.plot_spectra()


    #-------------------------- VARIABLE SELECTOR ----------------------------#
    def _add_variable_selector(self):
        """
        Generate radio selector buttons to set which state variable is 
        displayed.
        """
        noc = self.input_data.read_data_shape()[1]  # number of choices
        #State variable for the x axis
        pos_shp = [CONTOLS_START_X, 0.22, CONTROLS_WIDTH, 0.12 + 0.008 * noc]
        rax = self.figure.add_axes(pos_shp, axisbg=AXCOLOUR, title="State Variable")
        self.variable_selector = RadioButtons(rax, tuple(range(noc)), active=0)
        self.variable_selector.on_clicked(self._update_variable)


    def _update_variable(self, variable):
        """ 
        Update state variable being plotted based on radio button selection.
        """
        self.variable = int(variable)
        self.plot_spectra()


    #-------------------------- NORMALIZE SELECTOR ---------------------------#
    def _add_normalise_power_selector(self):
        """
        Add a radio button to chose whether or not the power of all spectra 
        should be normalized to 1.
        """
        pos_shp = [CONTOLS_START_X, 0.5, CONTROLS_WIDTH, CONTROLS_HEIGHT]
        rax = self.figure.add_axes(pos_shp, axisbg=AXCOLOUR, title="Normalise")
        np_tuple = ("yes", "no")
        self.normalise_power_selector = RadioButtons(rax, np_tuple, active=np_tuple.index(self.normalise_power))
        self.normalise_power_selector.on_clicked(self._update_normalise_power)


    def _update_normalise_power(self, normalise_power):
        """ Update whether to normalize based on radio button selection. """
        self.normalise_power = str(normalise_power)
        self.plot_spectra()


    #------------------------------ MAIN PLOT --------------------------------#
    def plot_spectra(self):
        """ 
        Main plot.
        """
        self.axes.clear()
        # Set title and axis labels
        title = " ".join(("source:", self.input_data.source.type.upper()))
        if self.input_data.windowing_function is not None and self.input_data.windowing_function.lower() != "none":
            title = " ".join((title, " window function:", self.input_data.windowing_function.upper()))

        self.axes.set(title=title)
        self.axes.set(xlabel="Frequency (kHz)")  # TODO: Somewhere we should convert requencies back to Hz...
        self.axes.set(ylabel="Power")

        # Set x and y scale based on current radio button selection.
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)

        if hasattr(self.axes, 'autoscale'):
            self.axes.autoscale(enable=True, axis='both', tight=True)

        shape = list(self.input_data.read_data_shape())
        if self.is_preview:
            shape[2] = NR_OF_PREVIEW_CHANS

        slices = (slice(shape[0]),
                  slice(self.variable, min(self.variable + 1, shape[1]), None),
                  slice(shape[2]),
                  slice(self.mode, min(self.mode + 1, shape[3]), None))

        #Plot the power spectra
        if self.normalise_power == "yes":
            data_matrix = self.input_data.get_data('normalised_average_power', slices)
        else:
            data_matrix = self.input_data.get_data('average_power', slices)

        data_matrix = data_matrix.reshape((shape[0], shape[2]))
        self.axes.plot(self.input_data.frequency[slice(shape[0])], data_matrix)
        self.figure.canvas.draw()



