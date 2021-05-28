# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Useful graph analyses.

.. moduleauthor:: Paula Sanz Leon <pau.sleon@gmail.com>

"""

import numpy
import networkx


def betweenness_bin(A):
    """

    Node betweenness centrality is the fraction of all shortest paths in 
    the network that contain a given node. Nodes with high values of 
    betweenness centrality participate in a large number of shortest paths.
    
    :param A: binary (directed/undirected) connection matrix (array)
            
    :returns: BC: a vector representing node between centrality vector.


    **Notes:**

    Betweenness centrality may be normalised to the range [0,1] as
    BC/[(N-1)(N-2)], where N is the number of nodes in the network.
    
    Original Mika Rubinov, UNSW/U Cambridge, 2007-2012 - From BCT 2012-12-04


    **Reference:**    [1] Kintali (2008) arXiv:0809.1906v2 [cs.DS] (generalization to directed and disconnected graphs)
    
    **Author:**        Paula Sanz Leon
    
    """
    
    n   = len(A)
    I   = numpy.eye(n)        # logical ID matrix                                        
    d   = 1                   # path length
    number_of_paths_d         = A.copy()           # number of paths of length |d|
    number_of_shortest_paths_d= A.copy()           # number of shortest paths of length |d|
    number_of_shortest_paths  = A.copy()           # number of shortest paths of any length
    length_of_shortest_paths  = A.copy()           # length of shortest paths
        
    number_of_shortest_paths[numpy.where(I)] = 1
    length_of_shortest_paths[numpy.where(I)] = 1

    #compute the number of shortest paths and their lengths
    while numpy.any(number_of_shortest_paths_d):
        d += 1
        number_of_paths_d          = numpy.dot(number_of_paths_d, A)
        number_of_shortest_paths_d = number_of_paths_d * (length_of_shortest_paths == 0)
        number_of_shortest_paths  += number_of_shortest_paths_d
        length_of_shortest_paths   = length_of_shortest_paths + d * (number_of_shortest_paths_d != 0)

    length_of_shortest_paths[length_of_shortest_paths == 0] = numpy.inf       # length for disconnected vertices is inf
    length_of_shortest_paths[numpy.where(I)] = 0.0
    number_of_shortest_paths[number_of_shortest_paths == 0] = 1.0  # number of shortest paths for disconn vertices is 1

    dependency = numpy.zeros((n, n))  # node on node dependency
    diam = d - 1                      # graph diameter
    
    #calculate dependency
    for d in range(diam, 1, -1):
        temp = (numpy.dot(((length_of_shortest_paths == d) * (1 + dependency) / number_of_shortest_paths), A.T)
                * ((length_of_shortest_paths == (d - 1)) * number_of_shortest_paths))
                #temp: dependencies on vertices |d-1| from source
        dependency += temp
                
    return dependency.sum(axis=0)



def efficiency_bin(A, compute_local_efficiency=False):
    """
    
    Computes global efficiency or local efficiency of a connectivity matrix.
    The global efficiency is the average of inverse shortest path length, 
    and is inversely related to the characteristic path length.
    
    The local efficiency is the global efficiency computed on the
    neighborhood of the node, and is related to the clustering coefficient.
    
    :param A: array; binary undirected connectivity matrix.
    
    :param compute_local_efficiency: bool, optional
        flag to compute either local or global efficiency of the network.
         
    
    :returns:
        -  global efficiency (float)
        - local efficiency (array)
             

    **References:** [1] Latora and Marchiori (2001) Phys Rev Lett 87:198701.
    
    
    .. note:: Algorithm: algebraic path count
    .. note:: Original: Mika Rubinov, UNSW, 2008-2010 - From BCT 2012-12-04
    .. note:: Tested with  Numpy 1.7
    
    .. warning:: tested against Matlab version... needs indexing improvement
    
    
    **Example:**
         
    >>> import numpy.random
    >>> A = np.random.rand(5, 5)
    >>> E = efficiency_bin(A)
    >>> E.shape == (1, ) 
    >>> True
    
    If you want to compute the local efficiency for every node in the network:
    
    >>> E = efficiency_bin(A, compute_local_efficiency=True)
    >>> E.shape == (5, 1)
    >>> True
    
    
    **Author:**        Paula Sanz Leon

    """
    
    # Binarize without modifying the original matrix (in case A is weighted)
    G = A.copy()
    G[G > 0] = 1.0 

    number_of_nodes = G.shape[0]     
    if compute_local_efficiency:
        E = numpy.zeros((number_of_nodes,1))  
        k = G.sum(axis=1)   # degree
        for u in range(number_of_nodes):
            indices = (G[u, :] > 0)
            if k[u] >= 2:   # degree must be at least two
                e = distance_inv(G[numpy.ix_(indices,indices)])
                E[u,:] = e.sum() / (k[u] ** 2 - k[u])     # local efficiency
        return E
    else:
        e = distance_inv(G)
        E = e.sum() / (number_of_nodes ** 2 - number_of_nodes)
        
        return E



def distance_inv(G):
    """
    Compute the inverse shortest path lengths of G.

    :param G: binary undirected connection matrix
    :returns: D: matrix of inverse distances
    """
    L = numpy.zeros(G.shape)
    D = numpy.eye(G.shape[0])
    n = 1
    nPATH = G.copy()                     # n-path matrix
    L[nPATH != 0] = 1.                   # shortest n-path matrix

    while L.sum() != 0:
        D += n * L
        n += 1
        nPATH = numpy.dot(nPATH, G)
        L = (nPATH > 0) * (D == 0)

    D[D == 0] = numpy.inf
    D = 1 / D                                # invert distance
    D = D - numpy.eye(G.shape[0])
    return D



def get_components_sizes(A):
    """
    Get connected components sizes.
    Returns the size of the largest component of an undirected graph specified by the *binary* and 
    *undirected* connection matrix A.
    
    
    :param A: array
        - binary undirected (BU) connectivity matrix.
    
    :returns:
        - largest component (float)
        - size  of the largest component
    
    :raises: Value Error - If A is not square.
              
    .. warning::       Requires NetworkX

    **Author:**        Paula Sanz Leon
    
    """


    # Just to preserve the original code. Check if the input matrix is square.
    # Further checks should include: check if it's binary for the functions that require so
    # and check that is undirected. 

    # Check if it is square
    if A.shape[0] != A.shape[1]:
        raise ValueError('The input matrix is not square')
    else:
        pass

    # Binarize without modifying the original matrix (in case A is weighted)
    temp_A = A.copy()
    temp_A[temp_A > 0] = 1.0    

    # Set diagonal elements to one
    if numpy.diag(A).sum() != A.shape[0]:
       numpy.fill_diagonal(temp_A, 1.0)
    
    # build a networkX graph to get largest connected component.
    components = networkx.connected_components(networkx.from_numpy_matrix(numpy.matrix(temp_A)))
    # For the time being returns the size of the largest component
    component_sizes = [len(x) for x in components][0]
    return component_sizes



def sequential_random_deletion(white_matter, random_sequence, nor):
    """
    
    A strategy to lesion a connectivity matrix.
    
    A single node is removed at each step until the network is reduced to only 2
    nodes. This method represents a structural failure analysis and it should
    be run several times with different random sequences.
    
    :param white_matter: tvb Connectivity DataType (yes, it's an example for TVB!)
                  a connectivity DataType that has a 'weights' attribute.
                  
    :param random_sequence: int array; a sequence of random integer numbers indicating which
                     the nodes will be deleted at each step.
    :param nor:      number of nodes of the original connectivity matrix.
                     
    :returns:
        - Node strength     (number_of_nodes, number_of_nodes -2)
        - Node degree       (number_of_nodes, number_of_nodes -2)
        - Global efficiency (number_of_nodes, )
        - Size of the largest component (number_of_nodes, )

    **References:**    Alstott et al. (2009).

    **Author:**        Paula Sanz Leon
    
    """

    node_strength = numpy.zeros((nor, nor - 2))
    node_degree   = numpy.zeros((nor, nor - 2))
    global_efficieny = numpy.zeros(nor - 2)
    largest_component = numpy.zeros(nor - 2)
    temp_strength = white_matter.weights.copy()
    temp_degree   = white_matter.weights.copy()
    temp_degree[temp_degree > 0.0] = 1.0

    for i, idx in enumerate(random_sequence):
            # delete rows
            temp_strength[idx, :] = 0.0
            temp_degree[idx, :]   = 0.0
        
            # delete columns
            temp_strength[:, idx] = 0.0
            temp_degree[:, idx]   = 0.0
        
            # strength
            in_strength     =  temp_strength.sum(axis=1)  
            out_strength    =  temp_strength.sum(axis=0)
            node_strength[:, i] = in_strength + out_strength
        
            # degree
            in_degree       = temp_degree.sum(axis=1)
            out_degree      = temp_degree.sum(axis=0)
            node_degree[:, i] = in_degree + out_degree
        
            # efficiency
            global_efficieny[i] = efficiency_bin(temp_degree)
        
            # largest connected component
            largest_component[i] = get_components_sizes(temp_degree)
            
    return node_strength, node_degree, global_efficieny, largest_component



def sequential_targeted_deletion(white_matter, nor):
    """
    
    A strategy to lesion a connectivity matrix.
    
    A single node is removed at each step until the network is reduced to only 2
    nodes. At each step different graph metrics are computed (degree, strength and
    betweenness centrality). The single node with the highest degree, strength or 
    centrality is removed.

    
    :param white_matter: tvb Connectivity datatype (yes, it's an example for TVB!)
                  a connectivity datatype that has a 'weights' attribute.
                  
    :param nor: number of nodes of the original connectivity matrix.
                     
    :returns:
        - Node strength          (number_of_nodes, number_of_nodes -2) array
        - Node degree            (number_of_nodes, number_of_nodes -2) array
        - Betweenness centrality (number_of_nodes, number_of_nodes -2) array
        - Global efficiency      (number_of_nodes, 3) array
        - Size of the largest component (number_of_nodes, 3) array
    
    
    **See also:**  sequential_random_deletion, localized_area_deletion

    **References:**    Alstott et al. (2009).

    **Author:**        Paula Sanz Leon
    
    """

    node_strength = numpy.zeros((nor, nor - 2))
    node_degree   = numpy.zeros((nor, nor - 2))
    node_betweenness_centrality = numpy.zeros((nor, nor - 2))
    global_efficiency = numpy.zeros((nor - 2, 3))
    largest_component = numpy.zeros((nor - 2, 3))
    temp_strength = white_matter.weights.copy()
    temp_degree   = white_matter.weights.copy()
    temp_bc       = white_matter.weights.copy()
    temp_degree[temp_degree > 0.0] = 1.0

    for idx in range(nor - 2):

            # strength
            in_strength     = temp_strength.sum(axis=1)
            out_strength    = temp_strength.sum(axis=0)
            node_strength[:, idx] = in_strength + out_strength
        
            # degree
            in_degree       = temp_degree.sum(axis=1)
            out_degree      = temp_degree.sum(axis=0)
            node_degree[:, idx] = in_degree + out_degree
            
            # betweeness centrality
            
            node_betweenness_centrality[:, idx] = betweenness_bin(temp_bc)
            
            # define target index
            sorted_strength_indices    = numpy.argsort(node_strength[:, idx])
            sorted_degree_indices      = numpy.argsort(node_degree[:, idx])
            sorted_bc_indices          = numpy.argsort(node_betweenness_centrality[:, idx])
            
            # lesion
            temp_strength[sorted_strength_indices[-1], :] = 0.0
            temp_strength[:, sorted_strength_indices[-1]] = 0.0

            temp_degree[sorted_degree_indices[-1], :] = 0.0
            temp_degree[:, sorted_degree_indices[-1]] = 0.0
            
            temp_bc[sorted_bc_indices[-1], :] = 0.0
            temp_bc[:, sorted_bc_indices[-1]] = 0.0
            
            # global efficiency (BU)
            global_efficiency[idx, 0] = efficiency_bin(temp_strength)  # compute the global eff of the binary matrix ver
            global_efficiency[idx, 1] = efficiency_bin(temp_degree)
            global_efficiency[idx, 2] = efficiency_bin(temp_bc)
        
            # largest connected component (BU)
            largest_component[idx, 0] = get_components_sizes(temp_strength)
            largest_component[idx, 1] = get_components_sizes(temp_degree)
            largest_component[idx, 2] = get_components_sizes(temp_bc)
            
    return node_strength, node_degree, node_betweenness_centrality, global_efficiency, largest_component

