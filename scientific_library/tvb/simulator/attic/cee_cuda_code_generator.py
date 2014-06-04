# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
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
The genc module implements code generation for integrating a
system's equations in different execution contexts and strategies.

    - C
    - CUDA

Below, functions exist emitting various code:

    - wrap - implements Python's % op in C
    - step - advance integration one step
    - model - differential equations for a particular node model

Going forward, 

    - Refactor into better pieces
    - Generalize coupling as f(x_post(t), x_pre(t-tau)), insert into
        step's inner loop over delayed step
    - Generalize integration scheme (use better schemes)
    - Add noise
    - Add flexible monitor components


.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

from cgen import *
from cgen.cuda import *


class RPointer(Pointer):
    c99 = True
    gpu = False
    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        restrict = "" if self.gpu else (" restrict " if self.c99 else " __restricted__ ")
        return sub_tp, "*%s%s" % (restrict, sub_decl)

def wrap(horizon, gpu=False):
    """
    Emits a function that behaves like Python's modulo operator.
    """

    h = horizon

    fndecl = FunctionDeclaration(Value('int', 'wrap'), [Value('int', 'i')])
    if gpu:
        fndecl = CudaDevice(fndecl)

    body = [If('i>=0', Block([Statement('return i %% %d' % h)]),
            Block([If('(-i) %% %d == 0 /* 1 McSlow w/ correct fries plz */' % h, 
                Block([Statement('return 0')]),
                Block([Statement('return %d + (i %% %d)' % (h, h))]))]))]

    return FunctionBody(fndecl, Block(body))

# should be able to inline this if we wish.. 
def model(eqns, pars, name='model', dt=0.1, gpu=False, dtype=None, noise=False):

    dtype = dtype or ('float' if gpu else 'double')

    args = [RPointer(Value(dtype, 'X')), RPointer(Value('void', 'pars')), Value(dtype, 'input')]\
         + ([Value('int', 'nthr'),Value('int', 'parij')] if gpu else [])\
         + ([RPointer(Value(dtype, 'noise'))] if noise else  [])

    fndecl = FunctionDeclaration(Value('void', name), args)
    if gpu:
        fndecl = CudaDevice(fndecl)

    Xrefs = [("X[nthr*%d + parij]" if gpu else "X[%d]") % i for i in range(len(eqns))]

    body = [Initializer(Value(dtype, p), "((%s*) pars)[%d]" % (dtype, i)) for i, p in enumerate(pars)]\
         + [Initializer(Value(dtype, eqn[0]), Xref) for Xref, eqn in zip(Xrefs, eqns)]\
         + [Initializer(Value(dtype, 'd'+var), deriv) for var, deriv in eqns]\
         +([Statement("d%s += noise[%s]" % (eqn[0], ("nthr*%d + parij"%i if gpu else "%d"%i))) for i, eqn in enumerate(eqns)] if noise else [])\
         + [Assign(Xref, '%s + %f*d%s' % (eqn[0], dt, eqn[0])) for Xref, eqn in zip(Xrefs, eqns)]

    return FunctionBody(fndecl, Block(body))


def step(n, nsv, cvar=0, gpu=False, nunroll=1, model='model', dtype=None, noise=False):
    """
    Emits a function that advances the system one step.
    """

    dtype = dtype or ('float' if gpu else 'double')

    varargs = ['hist', 'conn', 'X', 'gsc', 'exc'] + (['noise'] if noise else [])
    fndecl = FunctionDeclaration(Value('void', 'step'),
               [Value('int', 'step'), RPointer(Value('int', 'idel'))]\
             + [RPointer(Value(dtype, arg)) for arg in varargs])

    if gpu:
        fndecl = CudaGlobal(fndecl)

    body = []

    if gpu:
        body += [Initializer(Value('int', 'parij'), "blockDim.x*blockIdx.x + threadIdx.x"),
                 Initializer(Value('int', 'nthr'),  "blockDim.x*gridDim.x")]

    body += [Value('int', 'hist_idx'), Value(dtype, 'input'),
             Value('int', 'i'), Value('int', 'j')]

    hist_idx = '%d*nthr*%s + nthr*j + parij' if gpu else '%d*%s + j'

    inner_loop_body = [Assign('hist_idx', hist_idx % (n, 'wrap(step - 1 - *idel)')),
                       Statement('input += (*conn)*hist[hist_idx]')]\
                    + [Statement(v+'++') for v in ['j', 'idel', 'conn']]

    update_loop = [
        Assign('hist[' + ('nthr*%d*wrap(step) + nthr*i + parij'%n if gpu else '%d*wrap(step) + i'%n) + ']', 
               'X[' + ('%d*nthr*i + nthr*%d + parij' if gpu else '%d*i + %d')%(nsv, cvar) + ']'),
        Statement('i++')
    ]

    model_args = ['X + %s*i' % (('nthr*%d' if gpu else '%d')%nsv, ),
                  'exc' + (' + parij' if gpu else ''),
                  '%s*input' % ('gsc[parij]' if gpu else '(*gsc)',)]\
               + (['nthr', 'parij'] if gpu else [])\
               + (['noise+%s' % ('i*%d*nthr + parij'%(n*nsv,) if gpu else 'i*%d'%(nsv,))] if noise else [])
    # TODO: make noise passing customizable

    body += [For('i=0', 'i<%d' % n, 'i++', Block([ 
        Assign('input', '0.0'),
        For('j=0', 'j<%d' % (n - n%nunroll,), '', Block(inner_loop_body*nunroll))
    ] + inner_loop_body*(n%nunroll) + [Statement('%s(%s)' % (model, ', '.join(model_args), ))]
    )),

    # I thought this needed to be in a separate CUDA kernel to synch correctly, but I'm an idiot
    For('i=0', 'i<%d' % (n - n%nunroll), '', Block(update_loop*nunroll))
    ] + update_loop * (n%nunroll)

    return FunctionBody(fndecl, Block(body))


def module(model, step, wrap, gpu=False):
    return Module([wrap, model, step, Line('\n')])

# put this into descr.py or somewhere else
fhn = dict(
    name='fhn',
    eqns=[('x', '(x - x*x*x/3.0 + y)*3.0/5.0'),
          ('y', '(a - x)/3.0/5.0 + input')],
    pars = ['a']
)

pitch = dict(
    name='pitch',
    eqns=[('x', '(x - x*x*x/3.0)/5.0 + lambda + input')],
    pars=['lambda']
)

    

