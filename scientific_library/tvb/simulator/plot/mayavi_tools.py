# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
A collection of plotting functions used by tvb demos to plot 3D
Mayavi library is necessary in order for this module to work.

.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <paula.sanz-leon@univ-amu.fr>
"""

import numpy
import networkx as nx
from tvb.basic.logger.builder import get_logger

LOG = get_logger(__name__)

try:
    from mayavi import mlab


    @mlab.animate(delay=41, ui=True)
    def surface_timeseries(surface, data, step=1):
        """
        
        """
        fig = mlab.figure(figure="surface_timeseries", fgcolor=(0.5, 0.5, 0.5))
        # Plot an initial surface and colourbar
        surf_mesh = mlab.triangular_mesh(surface.vertices[:, 0],
                                         surface.vertices[:, 1],
                                         surface.vertices[:, 2],
                                         surface.triangles,
                                         scalars=data[0, :],
                                         vmin=data.min(), vmax=data.max(),
                                         figure=fig)
        mlab.colorbar(object=surf_mesh, orientation="vertical")

        # Handle for the surface object and figure
        surf = surf_mesh.mlab_source

        # Time
        tpts = data.shape[0]
        time_step = mlab.text(0.85, 0.125, ("0 of %s" % str(tpts)),
                              width=0.0625, color=(1, 1, 1), figure=fig,
                              name="counter")

        # Movie
        k = 0
        while 1:
            if abs(k) >= tpts:
                k = 0
            surf.set(scalars=data[k, :])
            time_step.set(text=("%s of %s" % (str(k), str(tpts))))
            k += step
            yield
        # mlab.show()


    def plot_surface(surface, fig=None, name=None, op=1.0, rep='surface'):
        """
        """
        if fig is None:
            fig = mlab.figure(figure=name, fgcolor=(0.5, 0.5, 0.5))

        surf_mesh = mlab.triangular_mesh(surface.vertices[:, 0],
                                         surface.vertices[:, 1],
                                         surface.vertices[:, 2],
                                         surface.triangles,
                                         color=(0.7, 0.67, 0.67),
                                         opacity=op,
                                         representation=rep,
                                         figure=fig)

        return surf_mesh


    def surface_orientation(surface, normals="triangles", name=None):
        """
        """
        fig = mlab.figure(figure=name, fgcolor=(0.5, 0.5, 0.5))
        surf_mesh = mlab.triangular_mesh(surface.vertices[:, 0],
                                         surface.vertices[:, 1],
                                         surface.vertices[:, 2],
                                         surface.triangles,
                                         color=(0.7, 0.67, 0.67),
                                         figure=fig)
        surf_orient = None
        if normals == "triangles":
            surf_orient = mlab.quiver3d(surface.triangle_centres[:, 0],
                                        surface.triangle_centres[:, 1],
                                        surface.triangle_centres[:, 2],
                                        surface.triangle_normals[:, 0],
                                        surface.triangle_normals[:, 1],
                                        surface.triangle_normals[:, 2])
        elif normals == "vertices":
            surf_orient = mlab.quiver3d(surface.vertices[:, 0],
                                        surface.vertices[:, 1],
                                        surface.vertices[:, 2],
                                        surface.vertex_normals[:, 0],
                                        surface.vertex_normals[:, 1],
                                        surface.vertex_normals[:, 2])
        else:
            LOG.error("normals must be either 'triangles' or 'vertices'")

        return surf_mesh, surf_orient


    def surface_parcellation(cortex_boundaries, colouring, mapping_colours, colour_rgb, interaction=False):
        """
        """
        number_of_vertices = cortex_boundaries.cortex.vertices.shape[0]
        number_of_triangles = cortex_boundaries.cortex.triangles.shape[0]

        number_of_regions = len(cortex_boundaries.region_neighbours)
        alpha = 255
        lut = numpy.zeros((number_of_regions, 4), dtype=numpy.uint8)
        for k in range(number_of_regions):
            lut[k] = numpy.hstack((colour_rgb[mapping_colours[colouring[k]]], alpha))

        fig = mlab.figure(figure="surface parcellation", bgcolor=(0.0, 0.0, 0.0), fgcolor=(0.5, 0.5, 0.5))
        surf_mesh = mlab.triangular_mesh(cortex_boundaries.cortex.vertices[:number_of_vertices // 2, 0],
                                         cortex_boundaries.cortex.vertices[:number_of_vertices // 2, 1],
                                         cortex_boundaries.cortex.vertices[:number_of_vertices // 2, 2],
                                         cortex_boundaries.cortex.triangles[:number_of_triangles // 2, :],
                                         scalars=cortex_boundaries.cortex.region_mapping[:number_of_vertices // 2],
                                         figure=fig)
        surf_mesh.module_manager.scalar_lut_manager.lut.number_of_colors = number_of_regions

        x = cortex_boundaries.boundary[:, 0]
        y = cortex_boundaries.boundary[:, 1]
        z = cortex_boundaries.boundary[:, 2]
        bpts = mlab.points3d(x, y, z, color=(0.25, 0.25, 0.25), scale_factor=1)
        mlab.show(stop=interaction)
        return surf_mesh, bpts


    def surface_pattern(surface, vertex_colours, custom_lut=None, foci=None):
        """
        Plot a surface and colour it based on a vector of length number of 
        vertices (vertex_colours).

        * How to obtain a pretty picture (from Mayavi's gui): 
        
          - set surf_mesh color to rgb(237, 217, 221)
          - add a surface module derived from surf_mesh; set 'Actor' 
            representation to wireframe; colour 'gray'.
          - enable contours of scalar_surf  
        """
        fig = mlab.figure(figure="surface pattern", fgcolor=(0.5, 0.5, 0.5))
        surf_mesh = mlab.triangular_mesh(surface.vertices[:, 0],
                                         surface.vertices[:, 1],
                                         surface.vertices[:, 2],
                                         surface.triangles,
                                         figure=fig)
        sm_obj = surf_mesh.mlab_source
        scalar_data = surf_mesh.mlab_source.dataset.point_data
        scalar_data.scalars = vertex_colours
        scalar_data.scalars.name = 'Scalar data'
        scalar_data.update()
        scalar_mesh = mlab.pipeline.set_active_attribute(surf_mesh, point_scalars='Scalar data')
        scalar_surf = mlab.pipeline.surface(scalar_mesh)

        if custom_lut is not None:
            # and finally we put this LUT back in the surface object. We could have
            # added any 255*4 array rather than modifying an existing LUT.
            scalar_surf.module_manager.scalar_lut_manager.lut.table = custom_lut

        if foci is not None:
            mlab.points3d(foci[:, 0],
                          foci[:, 1],
                          foci[:, 2],
                          scale_factor=2.,
                          scale_mode='none',
                          resolution=5,
                          opacity=0.01)

        mlab.show(stop=True)
        return sm_obj


    def xmas_balls(connectivity,
                   labels=True, labels_indices=None,
                   balls_colormap='Blues',
                   bgcolor=(0.5, 0.5, 0.5),
                   node_data=None, node_size=4.,
                   edge_data=True, edge_color=(0.8, 0.8, 0.8), edge_size=0.2,
                   text_size=0.042, text_color=(0, 0, 0),
                   remove_nodes=False, nbunch=[],
                   remove_edges=False, ebunch=[]):
        """
        Plots coloured balls at the region centres of connectivity, colour and
        size is determined by a vector of length number of regions (node_data).
        
        Optional: adds the connections between pair of nodes.
        
        """

        mlab.figure(1, bgcolor=bgcolor)

        # Get graph
        grf = nx.from_numpy_matrix(numpy.matrix(connectivity.weights))

        # Get the subgraph of nodes in nbunch
        if remove_nodes:
            grf.remove_nodes_from([n for n in grf if n not in set(nbunch)])
            # G.remove_nodes_from([node for node,degree in G.degree().items() if degree < 2])

        if remove_edges:
            grf.remove_edges_from([e for e in grf.edges() if e not in ebunch])

        # scalar colors
        if node_data is not None:
            scalars = node_data
            # mlab.colorbar(orientation="vertical")
        else:
            scalars = numpy.array(grf.nodes()) * 20

        pts = mlab.points3d(connectivity.centres[:, 0],
                            connectivity.centres[:, 1],
                            connectivity.centres[:, 2],
                            scalars,
                            # mask_points=1,
                            scale_factor=node_size,
                            scale_mode='none',
                            colormap=balls_colormap,
                            resolution=5,
                            opacity=0.01)

        if labels:
            if labels_indices is not None:
                for i, (idx) in enumerate(labels_indices):
                    x = connectivity.centres[idx, 0]
                    y = connectivity.centres[idx, 1]
                    z = connectivity.centres[idx, 2]
                    label = mlab.text(x, y, connectivity.region_labels[idx],
                                      z=z,
                                      width=text_size,
                                      name=str(connectivity.region_labels[idx]),
                                      color=text_color)
                    label.property.shadow = False
            else:
                for i, (x, y, z) in enumerate(connectivity.centres):
                    label = mlab.text(x, y, connectivity.region_labels[i],
                                      z=z,
                                      width=text_size,
                                      name=str(connectivity.region_labels[i]),
                                      color=text_color)
                    label.property.shadow = False

        if edge_data:
            pts.mlab_source.dataset.lines = numpy.array(grf.edges())
            tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
            mlab.pipeline.surface(tube, color=edge_color, representation='wireframe', opacity=0.3)

        # mlab.show()

        # stop the scene
        # mlab.show(stop=True)


    def connectivity_3d(connectivity, edge_cutoff=None):
        """
        Plots a 3D representation of the delayed-connectivity structure.
        See Fig. 3 in (Knock et al 2009)

        [Nodes x Nodes x Delays]
        
        Original script can be found at: 
        BrainNetworkModels_3.1/PlottingTools/PlotConnectivity3D.m
        
        """

        mlab.figure(figure="Connectivity 3D", bgcolor=(0.0, 0.0, 0.0))

        if connectivity.delays is None:
            connectivity.configure()

        n_reg_half = connectivity.number_of_regions // 2

        min_d = connectivity.delays.min()
        max_d = connectivity.delays.max()
        step_d = (max_d - min_d) / 10.

        if edge_cutoff is None:
            edge_cutoff = connectivity.weights.min()

        # Loop over connectivity matrix, colouring and one cube per matrix element
        k = []
        d = []
        m = []
        s = []
        for k_idx in range(n_reg_half):
            for m_idx in range(n_reg_half):
                if connectivity.weights[k_idx, m_idx] != 0:
                    if k_idx != m_idx:

                        if connectivity.weights[k_idx, m_idx] > edge_cutoff:
                            k.append(k_idx + 2.)
                            d.append(connectivity.delays[k_idx, m_idx] + step_d)
                            m.append(m_idx + 2.0)
                            s.append(connectivity.weights[k_idx, m_idx])
        mlab.points3d(k, d, m, s, mode='cube')
        mlab.show(stop=True)


except ImportError:
    LOG.warning(
        "Mayavi is needed for this demo but due to sizing and packaging constraints we are not distributing it. "
        "If you want to see the actual plot you should use the github version and install all the required "
        "dependencies as described here: (advanced users only)"
        "http://docs.thevirtualbrain.com/manuals/ContributorsManual/ContributorsManual.html#the-unaided-setup")
