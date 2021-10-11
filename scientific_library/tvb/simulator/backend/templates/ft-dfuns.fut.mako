-- TheVirtualBrain-Scientific Package. This package holds all simulators, and
-- analysers necessary to run brain-simulations. You can use it stand alone or
-- in conjunction with TheVirtualBrain-Framework Package. See content of the
-- documentation-folder for more details. See also http://www.thevirtualbrain.org
--
-- (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
--
-- This program is free software: you can redistribute it and/or modify it under the
-- terms of the GNU General Public License as published by the Free Software Foundation,
-- either version 3 of the License, or (at your option) any later version.
-- This program is distributed in the hope that it will be useful, but WITHOUT ANY
-- WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
-- PARTICULAR PURPOSE.  See the GNU General Public License for more details.
-- You should have received a copy of the GNU General Public License along with this
-- program.  If not, see <http://www.gnu.org/licenses/>.
--
--
--   CITATION:
-- When using The Virtual Brain for scientific publications, please cite it as follows:
--
--   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
--   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
--       The Virtual Brain: a simulator of primate brain network dynamics.
--   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

<%
    node_t = ', '.join(['f32' for _ in sim.model.state_variables])
    cvar_t = ', '.join(['f32' for _ in sim.model.coupling_terms])
    pars_t = ', '.join(['f32' for _ in sim.model.spatial_parameter_names])
%>

type node = (${node_t})
type cvar = (${cvar_t})
type pars = (${pars_t})

let dfun (y:node) (c:cvar) (p:pars): node =
    let (${','.join(sim.model.coupling_terms)}) = c
% if sim.model.spatial_parameter_names:
    let (${','.join(sim.model.spatial_parameter_names)}) = p
% endif
% for par in sim.model.global_parameter_names:
    let ${par} = ${getattr(sim.model, par)[0]}f32
% endfor
    let pi = f32.pi
    let (${','.join(sim.model.state_variables)}) = y
    in (
% for i, svar in enumerate(sim.model.state_variables):
        ${sim.model.state_variable_dfuns[svar]}${',' if i < (len(sim.model.state_variables)-1) else ''}
% endfor
    )