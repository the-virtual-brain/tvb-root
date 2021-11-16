# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
Dict to be used to rerender subforms when a parent form is refreshed.

.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from tvb.adapters.forms.equation_forms import get_form_for_equation, SpatialEquationsEnum, TemporalEquationsEnum, \
    SurfaceModelEquationsEnum
from tvb.adapters.forms.pipeline_forms import CommonPipelineForm, ParticipantPipelineForm, GroupPipelineForm, \
    IPPipelineAnalysisLevelsEnum

SPATIAL_EQ_KEY = "SPATIAL_EQ"
TEMPORAL_EQ_KEY = "TEMPORAL_EQ"
SURFACE_EQ_KEY = "SURFACE_EQ"
PIPELINE_KEY = "PIPELINE"


def get_form_method_by_name(form_name):
    form_name_to_form_methods = {
        SPATIAL_EQ_KEY: (get_form_for_equation, SpatialEquationsEnum),
        TEMPORAL_EQ_KEY: (get_form_for_equation, TemporalEquationsEnum),
        SURFACE_EQ_KEY: (get_form_for_equation, SurfaceModelEquationsEnum),
        # We have to do this trick because here we have the form directly in the Enum
        PIPELINE_KEY: ((lambda x: {CommonPipelineForm: CommonPipelineForm,
                                   ParticipantPipelineForm: ParticipantPipelineForm,
                                   GroupPipelineForm: GroupPipelineForm}[x]), IPPipelineAnalysisLevelsEnum)
    }

    return form_name_to_form_methods.get(form_name)
