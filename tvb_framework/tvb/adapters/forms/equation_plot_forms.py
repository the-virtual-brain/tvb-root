# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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

from tvb.core.neotraits.forms import Form, FloatField
from tvb.basic.neotraits.api import Float


class EquationPlotForm(Form):
    def __init__(self):
        super(EquationPlotForm, self).__init__()
        self.min_x = FloatField(Float(label='Min 0x value', default=0,
                                      doc="The minimum value of the x-axis for the equation plot."),
                                name='min_x')
        self.max_x = FloatField(Float(label='Max 0x value', default=100,
                                      doc="The maximum value of the x-axis for the equation plot."),
                                name='max_x')

    def fill_from_post(self, form_data):
        if self.min_x.name in form_data:
            self.min_x.fill_from_post(form_data)
        if self.max_x.name in form_data:
            self.max_x.fill_from_post(form_data)


class EquationSpatialPlotForm(Form):
    def __init__(self):
        super(EquationSpatialPlotForm, self).__init__()
        self.min_space_x = FloatField(Float(label='Spatial Start Distance(mm)', default=0,
                                            doc="The minimum value of the x-axis for spatial equation plot."),
                                      name='min_space_x')
        self.max_space_x = FloatField(Float(label='Spatial End Distance(mm)', default=100,
                                            doc="The maximum value of the x-axis for spatial equation plot."),
                                      name='max_space_x')


class EquationTemporalPlotForm(Form):
    def __init__(self):
        super(EquationTemporalPlotForm, self).__init__()
        self.min_tmp_x = FloatField(Float(label='Temporal Start Time(ms)', default=0,
                                          doc="The minimum value of the x-axis for temporal equation plot. " \
                                              "Not persisted, used only for visualization."),
                                    name='min_tmp_x')
        self.max_tmp_x = FloatField(Float(label='Temporal End Time(ms)', default=100,
                                          doc="The maximum value of the x-axis for temporal equation plot. " \
                                              "Not persisted, used only for visualization."),
                                    name='max_tmp_x')
