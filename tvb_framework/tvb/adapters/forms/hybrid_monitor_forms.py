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


"""
The Monitor forms of the Hybrid Simulator.

They are the classic Simulator Cockpit Monitor forms with their ``variables_of_interest`` field taken
away. ``tvb.simulator.hybrid.Simulator`` assigns ``monitor.voi = slice(None)`` to every Monitor it is
given, so a selection made on that field would be discarded before the first step is integrated, and
the field would only describe something the simulation does not do.

Everything else each Monitor offers is kept, and kept by inheriting the classic form rather than by
restating its fields.

.. moduleauthor:: TVB Team
"""

from tvb.adapters.forms.monitor_forms import BoldMonitorForm, EEGMonitorForm, MEGMonitorForm, MonitorForm, \
    SpatialAverageMonitorForm, iEEGMonitorForm
from tvb.core.entities.file.simulator.view_model import BoldViewModel, EEGViewModel, GlobalAverageViewModel, \
    MEGViewModel, RawViewModel, SpatialAverageViewModel, SubSampleViewModel, TemporalAverageViewModel, \
    iEEGViewModel
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.neotraits.forms import Form
from tvb.simulator.monitors import DefaultMasks


class HybridMonitorForm(MonitorForm):
    """
    A classic Monitor form without its ``variables_of_interest``.

    The three methods ``MonitorForm`` adds on top of ``Form`` all exist to carry that one field, and all
    three assume a classic ``SimulatorAdapterModel`` with a single Model - which the Hybrid Simulator,
    holding one Model per Subnetwork, does not have. Each is therefore taken back down to ``Form``:

    * ``fill_from_post`` resolves the posted variable names against ``simulator.model``;
    * ``fill_trait`` writes the resolved indices onto the Monitor. With nothing resolved it writes an
      empty float array, which the ``int`` typed ``Monitor.variables_of_interest`` refuses outright;
    * ``fill_from_trait`` reads them back into the field.

    The Monitor's own ``variables_of_interest`` is therefore left at ``None``, which is what
    ``Monitor._config_vois`` reads as 'all of them' - and what the Hybrid Simulator would impose anyway.

    Subclasses of the classic forms (BOLD, EEG, MEG, iEEG, Spatial average) list this class **after**
    their classic sibling, so that their own ``fill_trait`` and ``fill_from_trait`` still run and reach
    this one through ``super()`` instead of reaching ``MonitorForm``.
    """

    def __init__(self, *args, **kwargs):
        super(HybridMonitorForm, self).__init__(*args, **kwargs)
        # Form.fields yields what is on the instance, so dropping the attribute is what keeps the field
        # off the page. Nothing reaches it any more: all three methods that used it are overridden here.
        del self.variables_of_interest

    def fill_from_post(self, form_data):
        Form.fill_from_post(self, form_data)

    def fill_from_trait(self, trait):
        Form.fill_from_trait(self, trait)

    def fill_trait(self, datatype):
        Form.fill_trait(self, datatype)


class HybridSpatialAverageMonitorForm(SpatialAverageMonitorForm, HybridMonitorForm):
    """
    Spatial average, whose default mask choices depend on what the Connectivity carries.

    The classic form prunes them in ``fill_from_trait`` out of the session stored classic Simulator, so
    that pruning is done here instead, against the Connectivity the Hybrid Simulator was given.
    """

    def __init__(self, connectivity_gid=None):
        super(HybridSpatialAverageMonitorForm, self).__init__()
        self.connectivity_gid = connectivity_gid

    def fill_from_trait(self, trait):
        # deliberately not SpatialAverageMonitorForm.fill_from_trait: that one prunes out of
        # self.session_stored_simulator, which a Hybrid Simulator configuration never sets
        HybridMonitorForm.fill_from_trait(self, trait)

        # the Hybrid Simulator has no surface, so a mask over one is never on offer
        self._discard_choice(DefaultMasks.REGION_MAPPING)

        if self.connectivity_gid is None:
            return
        connectivity_index = load_entity_by_gid(self.connectivity_gid)
        if connectivity_index is None:
            return
        if connectivity_index.has_cortical_mask is False:
            self._discard_choice(DefaultMasks.CORTICAL)
        if connectivity_index.has_hemispheres_mask is False:
            self._discard_choice(DefaultMasks.HEMISPHERES)

    def _discard_choice(self, choice):
        if choice in self.default_mask.choices:
            self.default_mask.choices.remove(choice)


class HybridEEGMonitorForm(EEGMonitorForm, HybridMonitorForm):
    pass


class HybridMEGMonitorForm(MEGMonitorForm, HybridMonitorForm):
    pass


class HybridiEEGMonitorForm(iEEGMonitorForm, HybridMonitorForm):
    pass


class HybridBoldMonitorForm(BoldMonitorForm, HybridMonitorForm):
    pass


def get_hybrid_monitor_to_form_dict():
    return {
        RawViewModel: HybridMonitorForm,
        SubSampleViewModel: HybridMonitorForm,
        SpatialAverageViewModel: HybridSpatialAverageMonitorForm,
        GlobalAverageViewModel: HybridMonitorForm,
        TemporalAverageViewModel: HybridMonitorForm,
        EEGViewModel: HybridEEGMonitorForm,
        MEGViewModel: HybridMEGMonitorForm,
        iEEGViewModel: HybridiEEGMonitorForm,
        BoldViewModel: HybridBoldMonitorForm
    }


def get_form_for_hybrid_monitor(monitor_class):
    return get_hybrid_monitor_to_form_dict().get(monitor_class)
