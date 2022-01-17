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
A Numba backend which handles inner and outer batching via
unrolling and numba.prange respectively.  This backend is
ideal for CPU parameter sweeps: when benchmarked across a
range of machines, this approach achieves a significant
portion or exceeds the CPU's state memory bandwidth; on an
i7-8665U, it is nearly 50x faster throughput than single
threaded Numba backend.

... moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""


from .nb import NbBackend
import numba, mako.template, time, tqdm, numpy as np

class NbbuBackend(NbBackend):

    def __init__(self, blocks=4, lanes=4, chunklen=1000, progress=False):
        self.blocks = blocks
        self.lanes = lanes
        self.chunklen = chunklen
        self.progress = progress

    def run_sim(self, sim, **kwargs):
        "Given a template simulator, run batches from parameters **kwargs."
        nchunks = int(np.ceil(sim.simulation_length / sim.integrator.dt / self.chunklen))
        chiter = (tqdm.trange if self.progress else range)(nchunks)
        kernel, states, parameters, nh = self.prep_kernel(sim, nchunks)
        outputs = []
        for i in chiter:
            kernel(states, parameters)
            outputs.append(states[nh:])
            self.shuffle_chunk(sim, states, nh)
        return outputs

    def prep_kernel(self, sim, nchunks):
        "Prepare a kernel and associated arrays for use."
        # TODO
        kernel = lambda x, p: 0
        states = np.array([]) # (k, ..., nl)
        parameters = np.array([]) # (k, ..., nl)
        nh = sim.connectivity.idelays.max() + 1
        if isinstance(sim.integrator, IntegratorStochastic):
            self.noise_chunk(sim, states[nh:])
        return kernel, states, parameters, nh

    def shuffle_chunk(self, sim, states, nh):
        "Prepare the next chunk from the previous one."
        chunk[:nh] = chunk[-nh:]  # maybe could use it's own kernel
        if isinstance(sim.integrator, IntegratorStochastic):
            self.noise_chunk(sim, chunk[nh:])

    def noise_chunk(self, sim, states):
        "Generate noise in chunk for integration."
        # replace by bitgenerator
        z = np.random.randn(*states.shape)
        z = z * sim.integrator.noise.nsig
        states[:] = z

    def prep_poc_bench(self, conn, dt=0.01,
            nt=1000, cv=10.0, nl=1, k=1, pars=(1.0,1.0,-5.0,15.0,0.0)):
        # prep connectome
        weights = conn.weights.astype('f')
        idelays = (conn.tract_lengths / cv / dt).astype('i')
        nh = idelays.max() + 1
        nn = weights.shape[0]
        # prep template
        template = '<%include file="nbbu-poc.py.mako"/>'
        params = {'nl': nl, 'nt': nt, 'nh': nh}
        delays = self.build_py_func(template, params, name='delays')
        # prep arrays for one chunk of sim
        g = np.linspace(0,1,k*nl).reshape((k,nl))
        r, V = np.random.randn(2,k,nn,nh+nt, nl).astype('f')
        return g, r, V, lambda r, V, g: delays(dt, r, V, weights, idelays, g, *pars)

# TODO migrate test code into backend
# TODO chunking for any sim length
# TODO autotune k & nl to choose best Miter/s
# TODO expand all paramers to (k,nl) matrices, check perf
# TODO generalize dfun, gfun & cfun
# TODO generalize integrator
# TODO add bold & tavg
# TODO add stim
# TODO add driver code to handle arbitrary parameter spaces / sequences
# TODO memmap sharded output files
# TODO add cli

# parameters may be constant, varying per thread, varying per node
