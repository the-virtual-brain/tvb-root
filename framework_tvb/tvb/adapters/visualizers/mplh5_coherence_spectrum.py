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
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""
import numpy
from matplotlib.widgets import RadioButtons
from tvb.core.adapters.abcdisplayer import ABCMPLH5Displayer
from tvb.datatypes.spectral import ComplexCoherenceSpectrum


BACKGROUNDCOLOUR = "slategrey"
EDGECOLOUR = "darkslateblue"
AXCOLOUR = "steelblue"
BUTTONCOLOUR = "steelblue"
HOVERCOLOUR = "blue"

CONTOLS_START_X = 0.02
CONTROLS_WIDTH = 0.06
CONTROLS_HEIGHT = 0.104

NR_OF_PREVIEW_CHANS = 5



class ImaginaryCoherenceDisplay(ABCMPLH5Displayer):
    """
    This viewer takes as inputs a result from complex coherence analysis, 
    and returns required parameters for a MatplotLib representation.
    """

    _ui_name = "Complex (Imaginary) Coherence Visualizer"
    _ui_subsection = "complex_coherence"


    def get_input_tree(self):
        """ 
        Accept as input result from ComplexCoherence Analysis.
        """
        return [{'name': 'input_data',
                 'label': 'Complex Coherence Result',
                 'type': ComplexCoherenceSpectrum,
                 'required': True,
                 'description': 'Imaginary Coherence Analysis to display'}]


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

        self.xscale = "linear"
        self.yscale = "linear"
        self.spectrum = "Imag"

        #Selectors
        if not self.is_preview:
            self._add_xscale_selector()
            #self._add_yscale_selector()
            self._add_spectrum_mode_selector()
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

    ##------------------------ SPECTRUM MODE SELECTOR --------------------------#
    def _add_spectrum_mode_selector(self):
        """
        Add a radio button to the figure for selecting which part of the spectrum
        should be displayed: 0:real, 1:imag, 2:abs
        """
        pos_shp = [CONTOLS_START_X, 0.07, CONTROLS_WIDTH, 0.1 + 0.002 * 3]
        rax = self.figure.add_axes(pos_shp, axisbg=AXCOLOUR, title="Spectrum Mode")
        spectrum_mode_tuple = ("Re", "Imag", "Abs")
        self.spectrum_mode_selector = RadioButtons(rax, spectrum_mode_tuple,
                                                   active=spectrum_mode_tuple.index(self.spectrum))
        self.spectrum_mode_selector.on_clicked(self._update_spectrum_mode)


    def _update_spectrum_mode(self, spectrum_mode):
        """ Update the visualized spectrum mode based on radio button selection. """
        self.spectrum = str(spectrum_mode)
        self.plot_spectra()


    #------------------------------ MAIN PLOT --------------------------------#
    def plot_spectra(self):
        """ 
        Main plot.
        """
        self.axes.clear()
        # Set title and axis labels
        self.axes.set(title=self.input_data.source.type)
        self.axes.set(xlabel="Frequency [kHz]")
        #TODO: Somewhere we should convert requencies back to Hz...
        self.axes.set(ylabel="CohSpec")

        # Set x and y scale based on current radio button selection.
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)

        if hasattr(self.axes, 'autoscale'):
            self.axes.autoscale(enable=True, axis='x', tight=True)

        shape = list(self.input_data.read_data_shape())

        slices = (slice(shape[0]), slice(shape[1]), slice(shape[2]), )

        #Plot the power spectra
        if self.spectrum == "Imag":
            data_matrix = self.input_data.get_data('array_data', slices).imag
            indices = numpy.triu_indices(shape[0], 1)
            data_matrix = data_matrix[indices]
            HEX_COLOR = '#0F94DB'
            HEX_FACE_COLOR = '#469EEB'

        elif self.spectrum == "Re":
            data_matrix = self.input_data.get_data('array_data', slices).real
            data_matrix = data_matrix.reshape(shape[0] * shape[0], shape[2])
            HEX_COLOR = '#16C4B9'
            HEX_FACE_COLOR = '#0CF0E1'

        else:
            data_matrix = self.input_data.get_data('array_data', slices)
            data_matrix = numpy.absolute(data_matrix)
            data_matrix = data_matrix.reshape(shape[0] * shape[0], shape[2])
            HEX_COLOR = '#CC4F1B'
            HEX_FACE_COLOR = '#FF9848'

        #Get the upper off-diagonal indices

        coh_spec_sd = numpy.std(data_matrix, axis=0)
        coh_spec_av = numpy.mean(data_matrix, axis=0)

        self.axes.plot(self.input_data.frequency[slice(shape[2])],
                       coh_spec_av,
                       color=HEX_COLOR,
                       lw=2,
                       label='Mean +/- SD of CohSpec')
        self.axes.fill_between(self.input_data.frequency[slice(shape[2])],
                               coh_spec_av - coh_spec_sd,
                               coh_spec_av + coh_spec_sd,
                               alpha=0.5,
                               edgecolor=HEX_COLOR,
                               facecolor=HEX_FACE_COLOR)

        self.axes.legend()
        self.figure.canvas.draw()