# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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


class SimulatorWizzardURLs(object):
    SET_CONNECTIVITY_URL = '/burst/set_connectivity'
    SET_COUPLING_PARAMS_URL = '/burst/set_coupling_params'
    SET_SURFACE_URL = '/burst/set_surface'
    SET_STIMULUS_URL = '/burst/set_stimulus'
    SET_CORTEX_URL = '/burst/set_cortex'
    SET_MODEL_URL = '/burst/set_model'
    SET_MODEL_PARAMS_URL = '/burst/set_model_params'
    SET_INTEGRATOR_URL = '/burst/set_integrator'
    SET_INTEGRATOR_PARAMS_URL = '/burst/set_integrator_params'
    SET_NOISE_PARAMS_URL = '/burst/set_noise_params'
    SET_NOISE_EQUATION_PARAMS_URL = '/burst/set_noise_equation_params'
    SET_MONITORS_URL = '/burst/set_monitors'
    SET_MONITOR_PARAMS_URL = '/burst/set_monitor_params'
    SET_MONITOR_EQUATION_URL = '/burst/set_monitor_equation'
    SETUP_PSE_URL = '/burst/setup_pse'
    SET_PSE_PARAMS_URL = '/burst/set_pse_params'
    LAUNCH_PSE_URL = '/burst/launch_pse'
