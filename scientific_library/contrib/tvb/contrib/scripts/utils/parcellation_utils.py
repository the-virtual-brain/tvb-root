# coding=utf-8
import numpy as np


def parcellation_correspondence(inds_from, labels_from, labels_to):
    inds_to = []
    for ind in inds_from:
        lbl = labels_from[ind]
        inds_to.append(np.where(labels_to == lbl)[0][0])
    if len(inds_to) != 1:
        return inds_to
    else:
        return inds_to[0]
