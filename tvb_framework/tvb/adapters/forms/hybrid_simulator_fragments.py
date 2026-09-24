# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.forms.monitor_forms import get_monitor_to_ui_name_dict, get_ui_name_to_monitor_dict
from tvb.adapters.forms.simulator_fragments import SimulatorFinalFragment
from tvb.basic.neotraits.api import Attr, List
from tvb.core.adapters.abcadapter import ABCAdapterForm
from tvb.core.entities.file.simulator.view_model import HybridSimulatorAdapterModel
from tvb.core.neotraits.forms import FloatField, MultiSelectField, StrField, TraitDataTypeSelectField


class HybridConnectivityFragment(ABCAdapterForm):

    def __init__(self):
        super(HybridConnectivityFragment, self).__init__()
        self.connectivity = TraitDataTypeSelectField(HybridSimulatorAdapterModel.connectivity,
                                                     name=self.get_input_name(),
                                                     conditions=self.get_filters())
        self.ordered_fields = (self.connectivity,)

    @staticmethod
    def get_view_model():
        return HybridSimulatorAdapterModel

    @staticmethod
    def get_input_name():
        return 'connectivity'

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_required_datatype():
        return ConnectivityIndex


class HybridSubnetworksFragment(ABCAdapterForm):
    """
    The Subnetwork grouping step has no traited fields, the Connectivity regions are assigned to Subnetworks
    through a dedicated interactive component. This form only keeps the step inside the Hybrid Simulator wizard.
    """

    @staticmethod
    def get_view_model():
        return HybridSimulatorAdapterModel


class HybridSubnetworkDynamicsFragment(ABCAdapterForm):
    """
    The wizard step under which each Subnetwork's Model and Integrator are configured.

    Its only field is the integration step size, which is shared by every Subnetwork:
    tvb.simulator.hybrid.Simulator refuses a NetworkSet whose Subnetworks disagree on dt, so it is
    configured once here instead of being editable on each Subnetwork's Integrator form. The
    per-Subnetwork configuration itself happens in the contextual column.
    """

    def __init__(self):
        super(HybridSubnetworkDynamicsFragment, self).__init__()
        self.dt = FloatField(HybridSimulatorAdapterModel.dt)
        self.ordered_fields = (self.dt,)

    @staticmethod
    def get_view_model():
        return HybridSimulatorAdapterModel


class HybridMonitorsFragment(ABCAdapterForm):
    """
    The wizard step saying what the simulation records, and for how long.

    The Monitors are global: tvb.simulator.hybrid.Simulator observes every Subnetwork at once and hands
    one array to each Monitor, so there is one list for the whole configuration rather than one per
    Subnetwork.

    The simulation length sits here rather than on a step of its own. The classic Cockpit keeps it for
    last because it shares that step with the Launch button; there is nothing to launch yet, and a
    Monitor's sampling period only means something next to the length it is sampling.
    """

    # The Hybrid Simulator has no surface, so the Monitors are the ones the classic Cockpit offers for a
    # region simulation. That leaves out BOLD Region ROI, which is a surface only Monitor.
    IS_SURFACE_SIMULATION = False

    def __init__(self):
        super(HybridMonitorsFragment, self).__init__()
        self.simulation_length = FloatField(HybridSimulatorAdapterModel.simulation_length)
        self.monitor_choices = get_ui_name_to_monitor_dict(self.IS_SURFACE_SIMULATION)
        self.monitors = MultiSelectField(List(of=str, label='Monitors', choices=tuple(self.monitor_choices.keys())),
                                         name='monitors')
        self.ordered_fields = (self.simulation_length, self.monitors)

    def fill_from_trait(self, trait):
        # type: (HybridSimulatorAdapterModel) -> None
        super(HybridMonitorsFragment, self).fill_from_trait(trait)
        names = get_monitor_to_ui_name_dict(self.IS_SURFACE_SIMULATION)
        self.monitors.data = [names[type(monitor)] for monitor in trait.monitors or []
                              if type(monitor) in names]

    def monitors_from_post(self):
        """
        :return: one Monitor view model per chosen name, in the order they were chosen. The ad hoc List
                 the selector is built on carries no field_name, so fill_trait skips it and the Monitors
                 are built here instead - the same split the classic Cockpit makes.
        """
        return [self.monitor_choices[name]() for name in self.monitors.value or []
                if name in self.monitor_choices]

    @staticmethod
    def get_view_model():
        return HybridSimulatorAdapterModel


class HybridLaunchFragment(ABCAdapterForm):
    """
    The closing step of the Hybrid Simulator: what this simulation is called, next to the Launch button.

    The name is validated by the classic Cockpit's own rule, so that a Hybrid simulation cannot be named
    something the burst history could not show.
    """

    def __init__(self, default_simulation_name="simulation_1"):
        super(HybridLaunchFragment, self).__init__()
        self.simulation_name = StrField(
            Attr(str, doc='Name for the current Hybrid simulation', default=default_simulation_name,
                 label='Simulation name'),
            name='input_simulation_name_id')
        self.ordered_fields = (self.simulation_name,)

    def fill_from_post(self, form_data):
        super(HybridLaunchFragment, self).fill_from_post(form_data)
        validation_result = SimulatorFinalFragment.is_burst_name_ok(self.simulation_name.value)
        if validation_result is not True:
            raise ValueError(validation_result)

    @staticmethod
    def get_view_model():
        return HybridSimulatorAdapterModel
