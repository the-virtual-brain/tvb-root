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

import os
import numpy as np
import pandas as pd
import pandas.core.frame
import pytest
import siibra
from tvb.adapters.creators import siibra_base as sb
from tvb.datatypes import connectivity, graph
from tvb.tests.framework.core.base_testcase import BaseTestCase

HUMAN_ATLAS = 'Multilevel Human Atlas'
MONKEY_ATLAS = 'Monkey Atlas'
JULICH_PARCELLATION_3_0 = 'Julich 3.0'
MONKEY_PARCELLATION = 'MEBRAINS population-based monkey parcellation'
DEFAULT_HCP_SUBJECT = ['101309']
DEFAULT_1000BRAINS_SUBJECT = ['0001_1']


class TestSiibraBase(BaseTestCase):

    @pytest.fixture()
    def create_test_atlases_and_parcellations(self):
        self.human_atlas = siibra.atlases[HUMAN_ATLAS]
        self.monkey_atlas = siibra.atlases[MONKEY_ATLAS]
        self.julich_parcellation_3_0 = siibra.parcellations[JULICH_PARCELLATION_3_0]
        self.julich_parcellation_3_1 = siibra.parcellations[sb.JULICH_3]
        self.julich_parcellation_2_9 = siibra.parcellations[sb.JULICH_2_9]
        self.monkey_parcellation = siibra.parcellations[MONKEY_PARCELLATION]

    @pytest.fixture()
    def create_weights_and_tracts(self, create_test_atlases_and_parcellations):
        """
        Return all the weights and tracts available in siibra for default atlas and parcellation
        """
        weights = siibra.get_features(self.julich_parcellation_3_0, siibra.modalities.StreamlineCounts)
        tracts = siibra.get_features(self.julich_parcellation_3_0, siibra.modalities.StreamlineLengths)
        self.weights = weights
        self.tracts = tracts

    @pytest.fixture()
    def create_siibra_functional_connectivities(self, create_test_atlases_and_parcellations):
        """
        Return all the functional connectivities available in siibra for default atlas and parcellation
        """
        features = siibra.features.get(self.julich_parcellation_3_0,
                                       siibra.features.connectivity.FunctionalConnectivity)
        features_for_cohort = features[0]
        f = features_for_cohort[0]
        fc = f.data
        self.fc = fc

    def test_check_atlas_parcellation_compatible(self, create_test_atlases_and_parcellations):
        assert sb.check_atlas_parcellation_compatible(self.human_atlas, self.julich_parcellation_3_0)
        assert sb.check_atlas_parcellation_compatible(self.monkey_atlas, self.monkey_parcellation)
        assert not sb.check_atlas_parcellation_compatible(self.monkey_atlas, self.julich_parcellation_3_0)

    def test_get_atlases_for_parcellation(self, create_test_atlases_and_parcellations):
        atlas_list = sb.get_atlases_for_parcellation(self.julich_parcellation_3_0)
        assert atlas_list
        assert self.human_atlas in atlas_list
        assert self.monkey_atlas not in atlas_list

    def test_get_parcellations_for_atlas(self, create_test_atlases_and_parcellations):
        parcellation_list = sb.get_parcellations_for_atlas(self.human_atlas)
        assert parcellation_list
        assert self.julich_parcellation_3_1 in parcellation_list
        assert self.monkey_parcellation not in parcellation_list

    def test_get_cohorts_for_sc(self, create_test_atlases_and_parcellations):
        cohorts_julich_3 = sb.get_cohorts_for_sc(self.julich_parcellation_3_0)
        assert sb.HCP_COHORT in cohorts_julich_3
        assert sb.THOUSAND_BRAINS_COHORT not in cohorts_julich_3

        cohorts_julich_2_9 = sb.get_cohorts_for_sc(self.julich_parcellation_2_9)
        assert sb.HCP_COHORT in cohorts_julich_2_9
        assert sb.THOUSAND_BRAINS_COHORT in cohorts_julich_2_9

    def test_parse_subject_ids(self):
        # for HCP cohort
        single_id = '000'
        assert sb.parse_subject_ids(single_id, sb.HCP_COHORT) == ['000']

        multiple_ids = '000;010'
        assert sb.parse_subject_ids(multiple_ids, sb.HCP_COHORT) == ['000', '010']

        range_ids = '000-002'
        assert sb.parse_subject_ids(range_ids, sb.HCP_COHORT) == ['000', '001', '002']

        range_and_multiple_ids = '000-002;010'
        assert sb.parse_subject_ids(range_and_multiple_ids, sb.HCP_COHORT) == ['000', '001', '002', '010']

        range_and_multiple_ids2 = '100;000-002;010'
        assert sb.parse_subject_ids(range_and_multiple_ids2, sb.HCP_COHORT) == ['000', '001', '002', '010', '100']

        # for 1000BRAINS cohort
        single_full_id = '0001_1'
        assert sb.parse_subject_ids(single_full_id, sb.THOUSAND_BRAINS_COHORT) == ['0001_1']

        single_partial_id = '0017'
        assert sb.parse_subject_ids(single_partial_id, sb.THOUSAND_BRAINS_COHORT) == ['0017']

        multiple_full_ids = '0010_1;0017_2'
        assert sb.parse_subject_ids(multiple_full_ids, sb.THOUSAND_BRAINS_COHORT) == ['0010_1', '0017_2']

        multiple_partial_ids = '0010;0017'
        assert sb.parse_subject_ids(multiple_partial_ids, sb.THOUSAND_BRAINS_COHORT) == ['0010', '0017']

        range_ids = '0001-0003'
        assert sb.parse_subject_ids(range_ids, sb.THOUSAND_BRAINS_COHORT) == ['0001', '0002', '0003']

        range_and_ids = '0010;0001-0003;0017'
        assert sb.parse_subject_ids(range_and_ids, sb.THOUSAND_BRAINS_COHORT) == ['0001', '0002', '0003', '0010',
                                                                                  '0017']

    def test_init_siibra_params_no_selections(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when no sellection was made for atlas, parcellation or subject ids
        """
        with pytest.raises(ValueError) as error_info:
            sb.init_siibra_params(None, None, None, None)
        assert str(error_info.value) == 'Please provide at least one subject ID!'

    def test_init_siibra_params_for_defaults(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when no sellection was made for atlas, parcellation and cohort, but
        a subject ID was mentioned
        """
        # subject id is mandatory, otherwise execption
        empty_params_config = sb.init_siibra_params(None, None, None, '000')
        atlas, parcellation, cohort, subject_ids = empty_params_config
        assert atlas == self.human_atlas
        assert parcellation == self.julich_parcellation_3_1
        assert cohort == sb.HCP_COHORT
        assert subject_ids == ['000']

    def test_init_siibra_params_atlas_selected(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when only the atlas was selected
        """
        _, parcellation, cohort, subject_ids = sb.init_siibra_params(self.human_atlas, None, None, '000')
        assert parcellation is not None
        assert parcellation is self.julich_parcellation_3_1
        assert parcellation in list(self.human_atlas.parcellations)
        assert cohort == sb.HCP_COHORT
        assert subject_ids == ['000']

    def test_init_siibra_params_parcellation_selected(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when only the parcellation was selected
        """
        atlas, _, cohort, subject_ids = sb.init_siibra_params(None, self.julich_parcellation_2_9, None, '000')
        assert atlas == self.human_atlas
        assert cohort == sb.HCP_COHORT
        assert subject_ids == ['000']

    def test_init_siibra_params_subjects_selected(self, create_test_atlases_and_parcellations):
        """"
        Test initialization of siibra paramas when only the subjects were selected
        """
        atlas, parcellation, cohort, subject_ids = sb.init_siibra_params(None, None, None, '000;001')
        assert atlas == self.human_atlas
        assert parcellation in list(atlas.parcellations)
        assert cohort == sb.HCP_COHORT
        assert subject_ids == ['000', '001']

    def test_get_connectivity_matrix(self, create_test_atlases_and_parcellations):
        """
        Test the retrieval of structural connectivities (weights and tracts) and functional connectivities
        """
        weights = sb.get_connectivity_matrix(self.julich_parcellation_3_1, sb.HCP_COHORT, DEFAULT_HCP_SUBJECT,
                                             sb.Component2Modality.WEIGHTS)
        assert len(weights) > 0
        assert DEFAULT_HCP_SUBJECT[0] in weights
        assert isinstance(weights[DEFAULT_HCP_SUBJECT[0]], pandas.core.frame.DataFrame)

        tracts = sb.get_connectivity_matrix(self.julich_parcellation_3_1, sb.HCP_COHORT, DEFAULT_HCP_SUBJECT,
                                            sb.Component2Modality.TRACTS)
        assert len(tracts) > 0
        assert DEFAULT_HCP_SUBJECT[0] in tracts
        assert isinstance(tracts[DEFAULT_HCP_SUBJECT[0]], pandas.core.frame.DataFrame)

    def test_get_functional_connectivity_matrix(self, create_test_atlases_and_parcellations):
        fcs, fcs_names = sb.get_functional_connectivity_matrix(self.julich_parcellation_3_1, sb.HCP_COHORT,
                                                               DEFAULT_HCP_SUBJECT[0])
        assert len(fcs) > 0
        assert len(fcs_names) > 0
        assert len(fcs) == len(fcs_names)
        assert isinstance(fcs[0], pandas.core.frame.DataFrame)

    def test_get_hemispheres_for_regions(self):
        reg_names = ['reg1_right', 'reg1_left', 'reg_2']
        hemi = sb.get_hemispheres_for_regions(reg_names)
        assert hemi == [1, 0, 0]

    def test_get_regions_positions(self, create_test_atlases_and_parcellations):
        region = self.julich_parcellation_3_1.get_region('v1')
        assert region.name == 'Area hOc1 (V1, 17, CalcS)'
        reg_coord = sb.get_regions_positions([region])[0]
        assert len(reg_coord) == 3

    def test_create_tvb_structural_connectivity(self):
        """
        Test the creation of TVB Structural Connectivity using dummy data
        """
        weights_data = np.random.randint(0, 5, size=(2, 2))
        tracts_data = np.random.randint(0, 5, size=(2, 2))
        regions = ['reg1', 'reg2']
        hemi = [1, 0]
        positions = [(2.8424532907291535, -82.22873119424844, 2.1326498912705745),
                     (4.8424532907291535, -52.22873119424844, 4.1326498912705745)]

        weights = pd.DataFrame(data=weights_data, index=regions, columns=regions)
        tracts = pd.DataFrame(data=tracts_data, index=regions, columns=regions)

        tvb_conn = sb.create_tvb_structural_connectivity(weights, tracts, regions, hemi, positions)

        assert (tvb_conn.region_labels == regions).all()
        assert tvb_conn.number_of_regions == 2
        assert (tvb_conn.centres == positions).all()
        assert (tvb_conn.hemispheres == hemi).all()

    def test_get_structural_connectivities_from_kg(self, create_test_atlases_and_parcellations):
        tvb_conns = sb.get_structural_connectivities_from_kg(self.human_atlas, self.julich_parcellation_2_9,
                                                             sb.THOUSAND_BRAINS_COHORT, '0001_1')

        assert len(tvb_conns) == 1
        assert list(tvb_conns.keys()) == ['0001_1']
        assert isinstance(tvb_conns['0001_1'], connectivity.Connectivity)

    def test_create_tvb_connectivity_measure(self, create_siibra_functional_connectivities):
        conn = connectivity.Connectivity.from_file("connectivity_192.zip")
        fc = self.fc

        # the FC and SC are not compatible, but are used together only for testing purposes
        tvb_conn_measure = sb.create_tvb_connectivity_measure(fc, conn, 'StreamlineCounts (StreamlineCounts) anchored '
                                                                        'at Julich-Brain Cytoarchitectonic Atlas '
                                                                        '(v2.9) with cohort HCP - 000')
        assert (tvb_conn_measure.array_data == fc.to_numpy()).all()
        assert tvb_conn_measure.connectivity is conn
        assert tvb_conn_measure.title == ('StreamlineCounts (StreamlineCounts) anchored at Julich-Brain '
                                          'Cytoarchitectonic Atlas (v2.9) with cohort HCP - 000')

    def test_get_connectivity_measures_from_kg(self, create_test_atlases_and_parcellations):
        sc1 = connectivity.Connectivity.from_file("connectivity_76.zip")
        scs = {'001': sc1}

        tvb_conn_measures = sb.get_connectivity_measures_from_kg(self.human_atlas, self.julich_parcellation_3_0,
                                                                 sb.HCP_COHORT, '001', scs)

        assert len(tvb_conn_measures) == 1
        assert len(tvb_conn_measures['001']) == 5
        assert (list(tvb_conn_measures.keys()) == ['001'])
        assert isinstance(tvb_conn_measures['001'][0], graph.ConnectivityMeasure)
        assert tvb_conn_measures['001'][0].connectivity is sc1

        sc2 = connectivity.Connectivity.from_file("connectivity_66.zip")
        sc3 = connectivity.Connectivity.from_file("connectivity_68.zip")
        scs2 = {'0017_1': sc2, '0017_2': sc3}
        tvb_conn_measures2 = sb.get_connectivity_measures_from_kg(self.human_atlas, self.julich_parcellation_2_9,
                                                                  sb.THOUSAND_BRAINS_COHORT, '0017', scs2)

        assert len(tvb_conn_measures2) == 2
        assert (list(tvb_conn_measures2.keys()) == ['0017_1', '0017_2'])
        assert len(tvb_conn_measures2['0017_1']) == 1
        assert isinstance(tvb_conn_measures2['0017_1'][0], graph.ConnectivityMeasure)
        assert tvb_conn_measures2['0017_1'][0].connectivity is sc2
        assert tvb_conn_measures2['0017_2'][0].connectivity is sc3

    def test_get_connectivities_from_kg_no_fc(self, create_test_atlases_and_parcellations):
        """
        Test retrieval of just structural connectivities
        """
        scs, fcs = sb.get_connectivities_from_kg(self.human_atlas, self.julich_parcellation_3_1, sb.HCP_COHORT, '102311')

        assert len(scs) == 1
        assert not fcs

        assert (list(scs.keys()) == ['102311'])
        assert isinstance(scs['102311'], connectivity.Connectivity)

    def test_get_connectivities_from_kg_with_fc(self, create_test_atlases_and_parcellations):
        """
        Test retrieval of both structural and functional connectivities
        """
        scs, fcs = sb.get_connectivities_from_kg(self.human_atlas, self.julich_parcellation_3_1, sb.HCP_COHORT, '102311',
                                                 True)

        assert len(scs) == 1
        assert len(fcs) == 1
        assert len(fcs['102311']) == 5

        assert (list(scs.keys()) == ['102311'])
        assert isinstance(scs['102311'], connectivity.Connectivity)

        assert (list(fcs.keys()) == ['102311'])
        assert isinstance(fcs['102311'][4], graph.ConnectivityMeasure)
