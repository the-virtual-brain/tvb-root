# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
import collections
import numpy
from scipy import sparse
from tvb.basic.traits import util
from tvb.datatypes.cortex_data import CortexData
from tvb.datatypes.local_connectivity_scientific import LocalConnectivityScientific
from tvb.datatypes.surfaces import LOG


class CortexScientific(CortexData):
    """ This class exists to add scientific methods to Cortex """
    __tablename__ = None

    #TODO: Prob. should implement these in the @property way...
    region_areas = None
    region_orientation = None


    def configure(self):
        """
        Invoke the compute methods for computable attributes that haven't been
        set during initialisation.
        """
        super(CortexScientific, self).configure()

        if self.region_orientation is None:
            self.compute_region_orientation()

        if self.region_areas is None:
            self.compute_region_areas()

        if self.local_connectivity is None:
            self.local_connectivity = LocalConnectivityScientific(cutoff=40.0, use_storage=False, surface=self)

        if self.local_connectivity.matrix.size == 0:
            self.compute_local_connectivity()


        # Pad the local connectivity matrix with zeros when non-cortical regions
        # are included in the long range connectivity...
        if self.local_connectivity.matrix.shape[0] < self.region_mapping.shape[0]:
            LOG.info("There are non-cortical regions, will pad local connectivity")
            padding = sparse.csc_matrix((self.local_connectivity.matrix.shape[0],
                                         self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0]))
            self.local_connectivity.matrix = sparse.hstack([self.local_connectivity.matrix, padding])

            padding = sparse.csc_matrix((self.region_mapping.shape[0] - self.local_connectivity.matrix.shape[0],
                                         self.local_connectivity.matrix.shape[1]))
            self.local_connectivity.matrix = sparse.vstack([self.local_connectivity.matrix, padding])


    def _find_summary_info(self):
        """
        Extend the base class's scientific summary information dictionary.
        """
        summary = super(CortexScientific, self)._find_summary_info()
        summary["Number of regions"] = numpy.sum(self.region_areas > 0.0)
        summary["Region area, mean (mm:math:`^2`)"] = self.region_areas.mean()
        summary["Region area, minimum (mm:math:`^2`)"] = self.region_areas.min()
        summary["Region area, maximum (mm:math:`^2`)"] = self.region_areas.max()
        return summary


    def compute_local_connectivity(self):
        """
        """
        LOG.info("Computing local connectivity matrix")
        loc_con_cutoff = self.local_connectivity.cutoff
        self.compute_geodesic_distance_matrix(max_dist=loc_con_cutoff)

        self.local_connectivity.matrix_gdist = self.geodesic_distance_matrix.copy()
        self.local_connectivity.compute()  # Evaluate equation based distance
        self.local_connectivity.trait["matrix"].log_debug(owner=self.__class__.__name__ + ".local_connectivity")

        #HACK FOR DEBUGGING CAUSE TRAITS REPORTS self.local_connectivity.trait["matrix"] AS BEING EMPTY...
        lcmat = self.local_connectivity.matrix
        sts = str(lcmat.__class__)
        name = ".".join((self.__class__.__name__ + ".local_connectivity", self.local_connectivity.trait.name))
        shape = str(lcmat.shape)
        sparse_format = str(lcmat.format)
        nnz = str(lcmat.nnz)
        dtype = str(lcmat.dtype)
        if lcmat.data.any() and lcmat.data.size > 0:
            array_max = lcmat.data.max()
            array_min = lcmat.data.min()
        else:
            array_max = array_min = 0.0
        LOG.debug("%s: %s shape: %s" % (sts, name, shape))
        LOG.debug("%s: %s format: %s" % (sts, name, sparse_format))
        LOG.debug("%s: %s number of non-zeros: %s" % (sts, name, nnz))
        LOG.debug("%s: %s dtype: %s" % (sts, name, dtype))
        LOG.debug("%s: %s maximum: %s" % (sts, name, array_max))
        LOG.debug("%s: %s minimum: %s" % (sts, name, array_min))



    #--------------------------------------------------------------------------#
    #TODO: May be better to have these return values for assignment to the associated Connectivity...
    #TODO: These will need to do something sensible with non-cortical regions.
    def compute_region_areas(self):
        """
        """
        regions = numpy.unique(self.region_mapping)
        number_of_regions = len(regions)
        region_surface_area = numpy.zeros((number_of_regions, 1))
        avt = numpy.array(self.vertex_triangles)
        #NOTE: Slightly overestimates as it counts overlapping border triangles,
        #      but, not really a problem provided triangle-size << region-size.

        #NOTE: Check if there are non-cortical regions.

        if len(self.region_mapping) > len(self.vertex_normals):
            vertices_per_region = numpy.bincount(self.region_mapping)
            # Assume non-cortical regions will have len 1.
            non_cortical_regions, = numpy.where(vertices_per_region == 1)
            cortical_regions, = numpy.where(vertices_per_region > 1)
            #Average orientation of the region
            cortical_region_mapping = [x for x in self.region_mapping if x in cortical_regions]

            for nk in non_cortical_regions:
                region_surface_area[nk, :] = 0.0
            for k in cortical_regions:
                regs = map(set, avt[cortical_region_mapping == k])
                region_triangles = set.union(*regs)
                region_surface_area[k] = self.triangle_areas[list(region_triangles)].sum()
        else:
            for k in regions:
                regs = map(set, avt[self.region_mapping == k])
                region_triangles = set.union(*regs)
                region_surface_area[k] = self.triangle_areas[list(region_triangles)].sum()

        util.log_debug_array(LOG, region_surface_area, "region_areas", owner=self.__class__.__name__)
        self.region_areas = region_surface_area


    def compute_region_orientation(self):
        """
        """
        regions = numpy.unique(self.region_mapping)
        #import pdb;pdb.set_trace()
        average_orientation = numpy.zeros((len(regions), 3))

        if len(self.region_mapping) > len(self.vertex_normals):
            # Count how many vertices each region has.
            counter = collections.Counter(self.region_mapping)
            # Presumably non-cortical regions will have len 1 vertex assigned.
            vertices_per_region = numpy.asarray(counter.values())
            non_cortical_regions = numpy.where(vertices_per_region == 1)
            cortical_regions = numpy.where(vertices_per_region > 1)
            cortical_region_mapping = [x for x in self.region_mapping if x in cortical_regions[0]]
            #Average orientation of the region
            for k in cortical_regions[0]:
                orient = self.vertex_normals[cortical_region_mapping == k, :]
                avg_orient = numpy.mean(orient, axis=0)
                average_orientation[k, :] = avg_orient / numpy.sqrt(numpy.sum(avg_orient ** 2))
            for nk in non_cortical_regions[0]:
                average_orientation[nk, :] = numpy.zeros((1, 3))
        else:
            #Average orientation of the region
            for k in regions:
                orient = self.vertex_normals[self.region_mapping == k, :]
                avg_orient = numpy.mean(orient, axis=0)
                average_orientation[k, :] = avg_orient / numpy.sqrt(numpy.sum(avg_orient ** 2))

        util.log_debug_array(LOG, average_orientation, "region_orientation", owner=self.__class__.__name__)
        self.region_orientation = average_orientation
