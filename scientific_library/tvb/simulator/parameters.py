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
#   The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
A collection of parameter related classes and functions.

"""

from copy import deepcopy
from typing import List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from tvb.datatypes.connectivity import Connectivity
from .simulator import Simulator

class ParamGetter:
    pass

@dataclass
class SimSeq:
    "A sequence of simulator configurations."
    template: Simulator
    params: List[str]
    values: List[List[Any]]
    getters: Optional[List[Optional[ParamGetter]]] = None # is the first Optional needed?
    # TODO consider transpose, so a param can have a remote data source
    # to load when constructing the sequence

    def __iter__(self):
        self.pos = 0
        return self

    def __post_init__(self):
        self.template.configure() # deepcopy doesn't work on un-configured simulator o_O
        if self.getters is None:
            self.getters = [None]*len(self.params)
        else:
            assert len(self.getters) == len(self.params)

    def __next__(self):
        if self.pos >= len(self.values):
            raise StopIteration
        obj = deepcopy(self.template) 
        updates = zip(self.params, self.getters, self.values[self.pos])
        for key, getter, val in updates:
            if getter is not None:
                val = getter(val)
            exec(f'obj.{key} = val',
                 {'obj': obj, 'val': val})
        self.pos += 1
        return obj

class Metric:
    "A summary statistic for a simulation."
    def __call__(self, t, y) -> np.ndarray: # what about multi metric returning dict of statistics? Also, chaining?
        pass

class NodeVariability(Metric):
    "A simplistic simulation statistic."
    def __call__(self, t, y):
        return np.std(y[t > (t[-1] / 2), 0, :, 0], axis=0)

class Reduction:
    pass

@dataclass
class SaveMetricsToDisk(Reduction):
    filename: str

    def __call__(self, metrics_mat: np.ndarray) -> None:
        np.save(self.filename, metrics_mat)

# or save to a bucket or do SNPE then to a bucket, etc.

@dataclass
class PostProcess:
    metrics: List[Metric]
    reduction: Reduction

class Exec:
    pass

@dataclass
class JobLibExec:
    seq: SimSeq
    post: PostProcess

    def __call__(self, n_jobs=-1):
        from joblib import Parallel, delayed
        pool = Parallel(n_jobs)
        @delayed
        def job(sim):
            (t, y), = sim.configure().run()
            return np.hstack([m(t, y) for m in self.post.metrics])
        metrics = pool(job(_) for _ in self.seq)
        self.post.reduction(metrics)

if __name__ == '__main__':

    sim = Simulator(connectivity=Connectivity.from_file()).configure() # deepcopy doesn't work on un-configured simulator o_O
    seq = SimSeq(
        template=sim,
        params=['coupling.a'],
        values=[[np.r_[_]] for _ in 10**np.r_[-4:-1:10j]]
    )
    pp = PostProcess(
        metrics=[NodeVariability()],
        reduction=SaveMetricsToDisk('foo.npz'),
    )  
    exe = JobLibExec(seq=seq, post=pp)
    exe(n_jobs=4)
