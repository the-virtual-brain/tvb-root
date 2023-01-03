# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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
Some math tools

.. moduleauthor:: Dionysios Perdikis <Denis@tvb.invalid>
"""

from itertools import product
import numpy as np
from tvb.contrib.scripts.utils.data_structures_utils import is_integer
from sklearn.cluster import AgglomerativeClustering
from tvb.basic.logger.builder import get_logger
from tvb.simulator.plot.config import FiguresConfig

logger = get_logger(__name__)


def weighted_vector_sum(weights, vectors, normalize=True):
    if isinstance(vectors, np.ndarray):
        vectors = list(vectors.T)
    if normalize:
        weights /= np.sum(weights)
    vector_sum = weights[0] * vectors[0]
    for iv in range(1, len(weights)):
        vector_sum += weights[iv] * vectors[iv]
    return np.array(vector_sum)


def normalize_weights(weights, percentile, remove_diagonal=True, ceil=1.0):
    # Create the normalized connectivity weights:
    if len(weights) > 0:
        normalized_w = np.array(weights)
        if remove_diagonal:
            # Remove diagonal elements
            n_regions = normalized_w.shape[0]
            normalized_w *= (1.0 - np.eye(n_regions))
        # Normalize with the 95th percentile
        normalized_w = np.array(normalized_w / np.percentile(normalized_w, percentile))
        if ceil:
            if ceil is True:
                ceil = 1.0
            normalized_w[normalized_w > ceil] = ceil
        return normalized_w
    else:
        return np.array([])


def compute_in_degree(weights):
    return np.expand_dims(np.sum(weights, axis=1), 1).T


def compute_gain_matrix(locations1, locations2, normalize=100.0, ceil=False):
    n1 = locations1.shape[0]
    n2 = locations2.shape[0]
    projection = np.zeros((n1, n2))
    dist = np.zeros((n1, n2))
    for i1, i2 in product(range(n1), range(n2)):
        dist[i1, i2] = np.abs(np.sum((locations1[i1, :] - locations2[i2, :]) ** 2))
        projection[i1, i2] = 1 / dist[i1, i2]
    if normalize:
        projection /= np.percentile(projection, normalize)
    if ceil:
        if ceil is True:
            ceil = 1.0
        projection[projection > ceil] = ceil
    return projection


def get_greater_values_array_inds(values, n_vals=1):
    return np.argsort(values)[::-1][:n_vals]


def select_greater_values_array_inds(values, threshold=None, percentile=None, nvals=None, verbose=False):
    if threshold is None and percentile is not None:
        threshold = np.percentile(values, percentile)
    if threshold is not None:
        return np.where(values > threshold)[0]
    else:
        if is_integer(nvals):
            return get_greater_values_array_inds(values, nvals)
        if verbose:
            logger.warning("Switching to curve elbow point method since threshold=" + str(threshold))
        elbow_point = curve_elbow_point(values)
        return get_greater_values_array_inds(values, elbow_point)


def select_greater_values_2Darray_inds(values, threshold=None, percentile=None, nvals=None, verbose=False):
    return np.unravel_index(
        select_greater_values_array_inds(values.flatten(), threshold, percentile, nvals, verbose), values.shape)


def select_by_hierarchical_group_metric_clustering(distance, disconnectivity=np.array([]), metric=None,
                                                   n_groups=10, members_per_group=1):
    if disconnectivity.shape == distance.shape:
        distance += disconnectivity * distance.max()

    n_groups = np.minimum(np.maximum(n_groups, 3), n_groups // members_per_group)
    clustering = AgglomerativeClustering(n_groups, affinity="precomputed", linkage="average")
    clusters_labels = clustering.fit_predict(distance)
    selection = []
    for cluster_id in range(len(np.unique(clusters_labels))):
        # For each cluster, select the first...
        cluster_inds = np.where(clusters_labels == cluster_id)[0]
        # ... at least members_per_group elements...
        n_select = np.minimum(members_per_group, len(cluster_inds))
        if metric is not None and len(metric) == distance.shape[0]:
            # ...optionally according to some metric
            inds_select = np.argsort(metric[cluster_inds])[-n_select:]
        else:
            # ...otherwise, randomly
            inds_select = range(n_select)
        selection.append(cluster_inds[inds_select])
    return np.unique(np.hstack(selection)).tolist()


def curve_elbow_point(vals, interactive):
    # Solution found in
    # https://www.analyticbridge.datasciencecentral.com/profiles/blogs/identifying-the-number-of-clusters-finally-a-solution
    vals = np.array(vals).flatten()
    if np.any(vals[0:-1] - vals[1:] < 0):
        vals = np.sort(vals)
        vals = vals[::-1]
    cumsum_vals = np.cumsum(vals)
    grad = np.gradient(np.gradient(np.gradient(cumsum_vals)))
    elbow = np.argmax(grad)
    # alternatively:
    # dif = np.diff(np.diff(np.diff(cumsum_vals)))
    # elbow = np.argmax(dif) + 2
    if interactive:
        from matplotlib import pyplot
        pyplot.ion()
        fig, ax = pyplot.subplots()
        xdata = range(len(vals))
        lines = []
        lines.append(ax.plot(xdata, cumsum_vals, 'g*', picker=None, label="values' cumulative sum")[0])
        lines.append(ax.plot(xdata, vals, 'bo', picker=None, label="values in descending order")[0])
        lines.append(ax.plot(elbow, vals[elbow], "rd",
                             label="suggested elbow point (maximum of third central difference)")[0])
        lines.append(ax.plot(elbow, cumsum_vals[elbow], "rd")[0])
        pyplot.legend(handles=lines[:2])

        class MyClickableLines(object):

            def __init__(self, figure_no, axe, lines_list):
                self.x = None
                # self.y = None
                self.ax = axe
                title = "Mouse lef-click please to select the elbow point..." + \
                        "\n...or click ENTER to continue accepting our automatic choice in red..."
                self.ax.set_title(title)
                self.lines = lines_list
                self.fig = figure_no

            def event_loop(self):
                self.fig.canvas.mpl_connect('button_press_event', self.onclick)
                self.fig.canvas.mpl_connect('key_press_event', self.onkey)
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(timeout=-1)
                return

            def onkey(self, event):
                if event.key == "enter":
                    self.fig.canvas.stop_event_loop()
                return

            def onclick(self, event):
                if event.inaxes != self.lines[0].axes:
                    return
                dist = np.sqrt((self.lines[0].get_xdata() - event.xdata) ** 2.0)
                # + (self.lines[0].get_ydata() - event.ydata) ** 2.)
                self.x = np.argmin(dist)
                self.fig.canvas.stop_event_loop()
                return

        click_point = MyClickableLines(fig, ax, lines)
        click_point.event_loop()
        if click_point.x is not None:
            elbow = click_point.x
            logger.info("\nmanual selection: " + str(elbow))
        else:
            logger.info("\nautomatic selection: " + str(elbow))
        return elbow
    else:
        return elbow


def spikes_events_to_time_index(spike_time, time):
    if spike_time < time[0] or spike_time > time[-1]:
        return None
    return np.argmin(np.abs(time - spike_time))


def compute_spikes_counts(spikes_times, time):
    spikes_counts = np.zeros(time.shape)
    for spike_time in spikes_times:
        ind = spikes_events_to_time_index(spike_time, time)
        if ind is not None:
            spikes_counts[ind] += 1
    return spikes_counts


def spikes_rate_convolution(spike, spikes_kernel):
    if (spike != 0).any():
        if len(spikes_kernel) > 1:
            return np.convolve(spike, spikes_kernel, mode="same")
        else:
            return spike * spikes_kernel
    else:
        return np.zeros(spike.shape)
