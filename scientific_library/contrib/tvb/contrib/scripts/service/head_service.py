# -*- coding: utf-8 -*-

from copy import deepcopy
from six import string_types
import numpy as np
from tvb.basic.logger.builder import get_logger


class HeadService(object):
    logger = get_logger(__name__)

    def _assert_indices_from_labels(self, labels_or_indices, labels):
        indices = []
        labels_list = list(labels)
        for lbl_or_ind in labels_or_indices:
            if isinstance(lbl_or_ind, string_types):
                indices.append(labels_list.index(lbl_or_ind))
            else:
                indices.append(lbl_or_ind)
        return indices

    def slice_connectivity(self, connectivity, labels_or_indices):
        labels_or_indices = np.array(self._assert_indices_from_labels(labels_or_indices, connectivity.region_labels))
        out_conn = deepcopy(connectivity)
        out_conn.weights = connectivity.weights[labels_or_indices][:, labels_or_indices]
        out_conn.tract_lengths = connectivity.tract_lengths[labels_or_indices][:, labels_or_indices]
        out_conn.centres = connectivity.centres[labels_or_indices]
        out_conn.areas = connectivity.areas[labels_or_indices]
        out_conn.orientations = connectivity.orientations[labels_or_indices]
        out_conn.cortical = connectivity.cortical[labels_or_indices]
        out_conn.hemispheres = connectivity.hemispheres[labels_or_indices]
        out_conn.region_labels = connectivity.region_labels[labels_or_indices]
        out_conn.configure()
        return out_conn

    def slice_sensors(self, sensors, labels_or_indices):
        labels_or_indices = np.array(self._assert_indices_from_labels(labels_or_indices, sensors.labels))
        out_sensors = deepcopy(sensors)
        out_sensors.labels = sensors.labels[labels_or_indices]
        out_sensors.locations = sensors.locations[labels_or_indices]
        if out_sensors.has_orientations:
            out_sensors.orientations = sensors.orientations[labels_or_indices]
        out_sensors.configure()
        return out_sensors

    def sensors_in_electrodes_disconnectivity(self, sensors, sensors_labels=[]):
        if len(sensors_labels) < 2:
            sensors_labels = sensors.labels
        n_sensors = len(sensors_labels)
        elec_labels, elec_inds = sensors.group_sensors_to_electrodes(sensors_labels)
        if len(elec_labels) >= 2:
            disconnectivity = np.ones((n_sensors, n_sensors))
            for ch in elec_inds:
                disconnectivity[np.meshgrid(ch, ch)] = 0.0
        return disconnectivity
