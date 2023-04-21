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

"""
The adapter in this module creates new Structural and Functional Connectivities by extracting data from
the EBRAINS Knowledge Graph using siibra

.. moduleauthor:: Romina Baila <romina.baila@codemart.ro>
"""

import os
from siibra.retrieval.requests import SiibraHttpRequestError
from tvb.adapters.creators import siibra_base
from tvb.adapters.datatypes.db.connectivity import ConnectivityIndex
from tvb.adapters.datatypes.db.graph import ConnectivityMeasureIndex
from tvb.basic.neotraits._attr import Attr, EnumAttr
from tvb.basic.neotraits._core import TVBEnum
from tvb.core.adapters.abcadapter import ABCAdapterForm, ABCAdapter
from tvb.core.neotraits.forms import StrField, SelectField, BoolField, UserSessionStrField
from tvb.core.neotraits.view_model import ViewModel, Str
from tvb.core.services.user_service import KEY_AUTH_TOKEN

CLB_AUTH_TOKEN_KEY = 'HBP_AUTH_TOKEN'


# Following code is executed only once, when the application starts running
def init_siibra_options():
    """"
    Initialize siibra options for atlas and parcellations
    """
    # should use `atlases = [a.name for a in list(siibra.atlases)]`, but only the default one has data
    atlases = [siibra_base.DEFAULT_ATLAS]  # list with atlases names
    # should get only valid parcellations for default atlas, but only newest version of Julich parcellation
    # has data and corresponds with the current API of siibra
    parcellations = [siibra_base.DEFAULT_PARCELLATION]

    atlas_dict = {}
    parcellation_dict = {}

    for a_name in atlases:
        atlas_dict[a_name] = a_name

    for p_name in parcellations:
        parcellation_dict[p_name] = p_name

    atlas_options = TVBEnum('AtlasOptions', atlas_dict)
    parcellation_options = TVBEnum('ParcellationOptions', parcellation_dict)

    return atlas_options, parcellation_options


if 'SIIBRA_INIT_DONE' not in globals():
    ATLAS_OPTS, PARCELLATION_OPTS = init_siibra_options()
    SIIBRA_INIT_DONE = True


class SiibraModel(ViewModel):
    ebrains_token = Str(
        label='EBRAINS token',
        required=True,
        doc='Auth Token provided by EBRAINS lab `clb_oauth.get_token()` for accessing the Knowledge Graph'
    )

    atlas = EnumAttr(
        field_type=ATLAS_OPTS,
        default=ATLAS_OPTS[siibra_base.DEFAULT_ATLAS],
        label='Atlas',
        required=True,
        doc='Atlas to be used (only the compatible ones listed)'
    )

    parcellation = EnumAttr(
        field_type=PARCELLATION_OPTS,
        default=PARCELLATION_OPTS[siibra_base.DEFAULT_PARCELLATION],
        label='Parcellation',
        required=True,
        doc='Parcellation to be used (only TVB compatible ones listed here)'
    )

    subject_ids = Str(
        label='Subjects',
        required=True,
        default='000',
        doc="""The list of all subject IDs for which the structural and optionally functional connectivities
        The IDs can be specified in 3 ways: <br/>
        1. individually, delimited by a semicolon symbol: 000;001;002<br/>
        2. As a range, specifying the first and last IDs: 000-050 will retrieve all the subjects starting with 
        subject 000 until subject 050 (51 subjects)<br/>
        A combination of the 2 methods is also supported: 000-005;010 will retrieve all the subjects starting with 
        subject 000 until subject 005 (6 subjects) AND subject 010 (so 7 subjects in total)<br/>
        3. Using the keyword 'all' (without apostrophes) to get the connectivities for all the available subjects """)

    fc = Attr(
        field_type=bool,
        label="Compute Functional Connectivities",
        default=True,
        required=True,
        doc="Set if the functional connectivities for the specified subjects should also be computed"
    )


class SiibraCreatorForm(ABCAdapterForm):
    def __init__(self):
        super(SiibraCreatorForm, self).__init__()
        self.ebrains_token = UserSessionStrField(SiibraModel.ebrains_token, name="ebrains_token", key=KEY_AUTH_TOKEN)
        self.atlas = SelectField(SiibraModel.atlas, name='atlas')
        self.parcellation = SelectField(SiibraModel.parcellation, name='parcellation')
        self.subject_ids = StrField(SiibraModel.subject_ids, name='subject_ids')
        self.fc = BoolField(SiibraModel.fc, name='fc')

    @staticmethod
    def get_view_model():
        return SiibraModel

    @staticmethod
    def get_required_datatype():
        return None

    @staticmethod
    def get_filters():
        return None

    @staticmethod
    def get_input_name():
        return None


class SiibraCreator(ABCAdapter):
    """ The purpose of this creator is to use siibra in order to create Structural and Functional Connectivities """

    _ui_name = "Siibra Connectivity Creator"
    _ui_description = "Create Structural and Functional Connectivities from the EBRAINS KG using siibra"

    def get_form_class(self):
        return SiibraCreatorForm

    def get_output(self):
        return [ConnectivityIndex, ConnectivityMeasureIndex]

    def launch(self, view_model):
        ebrains_token = view_model.ebrains_token
        atlas = view_model.atlas.value
        parcellation = view_model.parcellation.value
        subject_ids = view_model.subject_ids
        compute_fc = view_model.fc

        os.environ[CLB_AUTH_TOKEN_KEY] = ebrains_token

        # list of all resulting indices for connectivities and possibly connectivity measures
        results = []

        try:
            conn_dict, conn_measures_dict = siibra_base.get_connectivities_from_kg(atlas, parcellation, subject_ids,
                                                                                    compute_fc)
        except SiibraHttpRequestError as e:
            if e.response.status_code in [401, 403]:
                raise ConnectionError('Invalid EBRAINS authentication token. Please provide a new one.')
            else:
                raise ConnectionError('We could not complete the operation. '
                                      'Please check the logs and contact the development team from TVB, siibra or EBRAINS KG.')


        # list of indexes after store_complete() is called on each struct. conn. and conn. measures
        conn_indices = []
        conn_measures_indices = []

        for subject_id, conn in conn_dict.items():
            generic_attrs = view_model.generic_attributes
            generic_attrs.subject = subject_id

            conn_index = self.store_complete(conn, generic_attrs)
            conn_index.fixed_generic_attributes = True
            conn_indices.append(conn_index)
            if compute_fc:
                conn_measures = conn_measures_dict[subject_id]
                for conn_measure in conn_measures:
                    conn_measure_index = self.store_complete(conn_measure, generic_attrs)
                    conn_measure_index.fixed_generic_attributes = True
                    conn_measures_indices.append(conn_measure_index)
        results.extend(conn_indices)
        if conn_measures_indices:
            results.extend(conn_measures_indices)
        return results

    def get_required_memory_size(self, view_model):
        return -1

    def get_required_disk_size(self, view_model):
        return -1
