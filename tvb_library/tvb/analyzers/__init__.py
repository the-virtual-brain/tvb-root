# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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

from tvb.analyzers.metric_variance_global import compute_variance_global_metric
from tvb.analyzers.metric_kuramoto_index import compute_kuramoto_index_metric
from tvb.analyzers.metric_proxy_metastability import compute_proxy_metastability_metric
from tvb.analyzers.metric_variance_of_node_variance import compute_variance_of_node_variance_metric

METRICS = {'GlobalVariance': compute_variance_global_metric,
           'KuramotoIndex': compute_kuramoto_index_metric,
           'ProxyMetastabilitySynchrony': compute_proxy_metastability_metric,
           'VarianceNodeVariance': compute_variance_of_node_variance_metric}
