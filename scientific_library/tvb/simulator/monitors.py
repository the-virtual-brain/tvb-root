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
#   The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#
"""
Monitors record significant values from the simulation. In their simplest form
they return all the simulated data, Raw(), directly subsampled data, SubSample()
spatially averaged temporally subsampled, GlobalAverage(), or temporally
averaged subsamples, TemporalAverage(). The more elaborate monitors instantiate
a physically realistic measurement process on the simulation, such as EEG, MEG,
and fMRI (BOLD).

Conversion of power of 2 sample-rates(Hz) to Monitor periods(ms)
::

    4096 Hz => 0.244140625 ms
    2048 Hz => 0.48828125 ms
    1024 Hz => 0.9765625 ms
     512 Hz => 1.953125 ms
     256 Hz => 3.90625 ms
     128 Hz => 7.8125 ms


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>
.. moduleauthor:: Paula Sanz Leon <Paula@tvb.invalid>
.. moduleauthor:: Noelia Montejo <Noelia@tvb.invalid>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
.. moduleauthor:: Jan Fousek <izaak@mail.muni.cz>

"""

# Standard python libraries

# Third party python libraries
import numpy

#The Virtual Brain
from tvb.datatypes.time_series import (TimeSeries, TimeSeriesRegion,
    TimeSeriesEEG, TimeSeriesMEG, TimeSeriesSEEG, TimeSeriesSurface)
from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

import tvb.datatypes.sensors as sensors_module
from tvb.datatypes.sensors import SensorsMEG, SensorsInternal, SensorsEEG, Sensors
import tvb.datatypes.arrays as arrays
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.connectivity import Connectivity
from tvb.datatypes.projections import (
    ProjectionMatrix, ProjectionSurfaceEEG, ProjectionSurfaceMEG)
import tvb.datatypes.equations as equations

import tvb.basic.traits.util as util
import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.core as core

from tvb.simulator.common import iround


class MonitorTransforms(object):
    """
    Monitor pre & post expr transform rules

    - pre/post are string fields
    - both contain (possibly comma separated) expressions in numexpr syntax
    - no reduction operations (sum/prod) because these result in shape changes
    - if comma separated, assert len(pre.split(','))==len(post.split(',')), pre-post expression are paired
    - output 4D ts has shape (time, len(pre.split(',')), node, mode)
    - by default pre & post is identity
    - n pre & n post can be different, but only if one of them is 1 or 0
      in which case it is applied to each of the others (or identity if none given)

    additional rules on special variable names in the expressions

    - pre can't contain empty
    - post can, but is identity
    - post should ref 'mon' as corresponding pre expr

    """

    def __init__(self, pre_exprs, post_exprs, model_svars=(), delim=';'):
        self.pre = self._prep_exprs(pre_exprs, delim)
        self.post = self._prep_exprs(post_exprs, delim)
        self.svars = model_svars
        self._check_num_exprs()
        self._check_syntax(self.pre, allow_empty=False)
        self._check_syntax(self.post, allow_empty=True)
        self._np_ns = {}
        exec "from numpy import *" in self._np_ns

    def _prep_exprs(self, exprs, delim):
        return [e.strip() for e in exprs.split(delim)]

    def _check_num_exprs(self):
        n_pre, n_post = len(self.pre), len(self.post)
        if n_pre != n_post:
            if n_pre == 1:
                self.pre *= n_post
            elif n_post == 1:
                self.post *= n_pre
            else:
                msg = ("%d pre and %d post monitor expressions provided. Only an equal number or 1 for either "
                       "should be provided.")
                msg %= n_pre, n_post
                raise Exception(msg)
        else:
            pass

    def _check_syntax(self, exprs, allow_empty):
        for expr in exprs:
            if allow_empty and len(expr) == 0:
                continue
                # this expr will be no-op / identity
            try:
                compile(expr, '<expr>', 'eval')
            except SyntaxError:
                raise SyntaxError('%r is not a valid expression' % (expr, ))

    def _exec_in(self, code, ns):
        ns = ns.copy()
        ns.update(self._np_ns)
        try:
            exec code in ns
        except Exception as exc:
            msg = '%s: %r' % (exc.message, code)
            raise Exception(msg)

    def apply_pre(self, state):
        """
        Apply pre-monitor transformations to state.

        """

        ns = {}
        svars = self.svars or map('x{0}'.format, range(len(state)))
        ok = '__out__'
        assert ok not in svars
        ns[ok] = numpy.empty((len(self.pre), ) + state.shape[1:])
        for key, val in zip(svars, state):
            ns[key] = val
        for i, expr in enumerate(self.pre):
            self._exec_in("%s[%d] = %s" % (ok, i, expr), ns)
        return ns[ok]

    def apply_post(self, monitor_output, inkey='mon'):
        """
        Apply post-monitor transformations to monitor output.

        """

        ok = '__out__'
        assert inkey != ok
        time, sample = monitor_output
        ns = {inkey: sample}
        ns[ok] = numpy.empty((len(self.post), ) + sample.shape[1:])
        for i, expr in enumerate(self.post):
            ns[inkey] = sample[i]
            if len(expr) == 0:
                expr = inkey
            self._exec_in('%s[%d] = %s' % (ok, i, expr), ns)
        return time, ns[ok]


class Monitor(core.Type):
    """
    Base Monitor class, all Monitors inherit from this class.

    **private attribute:**
        ``_stock``: 
            A kind of internal memory of the monitor, used for when monitors
            are non-instantaneous, such as TemporalAverage or fMRI.

    .. #Currently there seems to be a clash betwen traits and autodoc, autodoc
    .. #can't find the methods of the class, the class specific names below get
    .. #us around this...
    .. automethod:: Monitor.__init__
    .. automethod:: Monitor.config_for_sim
    .. automethod:: Monitor.record

    """

    # list of class names not shown in UI
    _base_classes = ['Monitor', 'BoldMultithreaded', 'BalloonWindkesselAccordingToKJFristonEtAl2003NeuroImage',
                     'Projection']

    period = basic.Float(
        label = "Sampling period (ms)",
        default = 0.9765625, #ms. 0.9765625 => 1024Hz #ms, 0.5 => 2000Hz
        doc = """Sampling period in milliseconds, must be an integral multiple
        of integration-step size. As a guide: 2048 Hz => 0.48828125 ms ;  
        1024 Hz => 0.9765625 ms ; 512 Hz => 1.953125 ms.""")

    variables_of_interest = arrays.IntegerArray(
        label = "Model variables to watch",
        doc = """This can be used to override, for each Monitor, the default 
        monitored state-variables. NOTE: Any specified indices must be consistent 
        with the Model being monitored. By default, if left unspecified, this will
        be set based on the variables_of_interest attribute on the Model.""",
        order = -1)

    pre_expr = basic.String(
        label="Pre-monitor expression(s)",
        required=False,
        doc="Expression to evaluate on state variables prior to applying a "
            "monitor's observation model. Several expressions may be provided "
            "when separated by semicolons (';'), each paired with a corresponding post_expr "
            "if provided. Expressions must be standard NumPy syntax or using functions from numpy, and "
            "reductions (sum, prod) are not available. If no set of expressions are "
            "given, then the model's variables of interest are used. For example, the dipole moment prior "
            "to applying an EEG gain matrix for the generic 2D oscillator might use the expression "
            "`1e-6 * (V + W / 2)`."
    )

    post_expr = basic.String(
        label="Post-monitor expression(s)",
        required=False,
        doc="Expression to evaluate on monitor values (named `mon`) after applying a "
            "monitor's observation model. Several expressions of 'mon' may be provided "
            "when separated by semicolons (';'), each paired with a corresponding pre_expr "
            "if provided. Expressions must be standard NumPy syntax or using functions from numpy, and "
            "reductions (sum, prod) are not available. If no expressions are given "
            "then the result of the monitor's observation model results. For example, the sine of squaring the"
            " result of a monitor's observation model is achieved which the expression `sin(mon**2)`"
    )

    def __init__(self,  **kwargs):
        """
        Initialise the place holder attributes that will be filled when the
        Monitor is configured for a given Simulation -- ie, for a specific
        Model, Connectivity, Integrator, etc.

        """
        super(Monitor, self).__init__(**kwargs) 
        LOG.debug(str(kwargs))

        self.istep = None #Monitor period in integration time-steps (integer).
        self.dt = None  #Integration time-step in "physical" units.
        self.voi = None
        self._stock = numpy.array([], dtype=numpy.float64)

    def __repr__(self):
        """A formal, executable, representation of a Monitor object."""
        class_name = self.__class__.__name__
        traited_kwargs = self.trait.keys()
        formal = class_name + "(" + "=%s, ".join(traited_kwargs) + "=%s)"
        return formal % eval("(self." + ", self.".join(traited_kwargs) + ")")

    def __str__(self):
        """An informal, 'human readable', representation of a Monitor object."""
        class_name = self.__class__.__name__ 
        traited_kwargs = self.trait.keys()
        informal = class_name + "(" + ", ".join(traited_kwargs) + ")"
        return informal

    def config_for_sim(self, simulator):
        """
        Grab the Simulator's integration step size. Set the monitor's variables
        of interest based on the Monitor's 'variables_of_interest' attribute, if
        it was specified, otherwise use the 'variables_of_interest' specified 
        for the Model. Calculate the number of integration steps (isteps)
        between returns by the record method. This method is called from within
        the the Simulator's configure() method.

        Args:
            ``simulator`` (Simulator): a Simulator object.

        """
        self.dt = simulator.integrator.dt
        self.istep = iround(self.period / self.dt)
        LOG.info("%s: istep of monitor is %d" % (str(self), self.istep))

        svars = simulator.model.state_variable_range.keys()

        # handle transforms, "expressions of interest"
        if self.pre_expr or self.post_expr:
            pre, post = self.pre_expr, self.post_expr
        else:
            # TODO convert voi to pre expr
            if self.variables_of_interest.size == 0:
                self.voi = numpy.array([simulator.model.state_variables.index(var)
                                        for var in simulator.model.variables_of_interest])
            else:
                self.voi = self.variables_of_interest
            pre = ';'.join([svar for i, svar in enumerate(svars) if i in self.voi])
            post = ''

        self._transforms = mt = MonitorTransforms(pre, post, svars)
        self.voi = numpy.r_[:len(mt.pre)]
        LOG.info('%r - pre %r, post %r, svars %r', self, mt.pre, mt.post, svars)

        dbg_voi = numpy.array([0, 3])
        if type(self) in (TemporalAverage, ) and self.voi.shape == dbg_voi.shape and numpy.allclose(self.voi, dbg_voi):
            pass

    def record(self, step, state):
        """
        This is a final method called by the simulator to obtain samples from a
        monitor instance. Monitor subclasses should not override this method, but
        rather implement the `sample` method.

        """

        mt = self._transforms
        sample = self.sample(step, mt.apply_pre(state))
        if sample is not None:
            return mt.apply_post(sample)

    def sample(self, step, state):
        """
        This method provides monitor output, and should be overridden by subclasses.

        """

        raise NotImplementedError(
            "The Monitor base class does not provide any observation model and "
            "should be subclasses with an implementation of the `sample` method.")

    def create_time_series(self, storage_path, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        """
        Create a time series instance that will be populated by this monitor
        :param surface: if present a TimeSeriesSurface is returned
        :param connectivity: if present a TimeSeriesRegion is returned
        Otherwise a plain TimeSeries will be returned
        """
        if surface is not None:
            return TimeSeriesSurface(storage_path=storage_path,
                                     surface=surface,
                                     sample_period=self.period,
                                     title='Surface ' + self.__class__.__name__,
                                     **self._transform_user_tags())
        if connectivity is not None:
            return TimeSeriesRegion(storage_path=storage_path,
                                    connectivity=connectivity,
                                    region_mapping=region_map,
                                    region_mapping_volume=region_volume_map,
                                    sample_period=self.period,
                                    title='Regions ' + self.__class__.__name__,
                                    **self._transform_user_tags())

        return TimeSeries(storage_path=storage_path,
                          sample_period=self.period,
                          title=' ' + self.__class__.__name__,
                          **self._transform_user_tags())

    def _transform_user_tags(self):
        "Return transforms as user tags."
        values = []
        if self.pre_expr:
            values.append(self.pre_expr)
        if self.post_expr:
            values.append(self.post_expr)
        return {'user_tag_%d' % (i + 1, ): value for i, value in enumerate(values)}


class Raw(Monitor):
    """
    A monitor that records the output raw data from a tvb simulation:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based 
        - all the integration time steps

    """
    _ui_name = "Raw recording"

    period = basic.Float(
        label = "Sampling period is ignored for Raw Monitor",
        order = -1)

    variables_of_interest = arrays.IntegerArray(
        label = "Raw Monitor sees all!!! Resistance is futile...",
        order = -1)
    
    def __init__(self, **kwargs):
        """Initialise the Raw monitor from the base Monitor class."""
        LOG.info("%s: initing..." % str(self))
        super(Raw, self).__init__(**kwargs)
        LOG.debug("%s: inited." % repr(self))

    def config_for_sim(self, simulator):
        """
        Initialise the istep to 1, regardless of any ``period`` provided. Also,
        sets the monitor's variables of interest to be all state variables, 
        regardless of any ``variables_of_interest`` provided.

        """

        super(Raw, self).config_for_sim(simulator)

        # Simulation time step
        self.dt = simulator.integrator.dt
        self.period = simulator.integrator.dt #Needed for Monitor consistency

        # state-variables to monitor
        self.voi = numpy.arange(simulator.model.nvar)
        LOG.info("%s: variable of interest is %s"%(str(self), str(self.voi)))

        # monitor period in integration steps
        self.istep = 1
        LOG.info("%s: istep of monitor is %d"%(str(self), self.istep))

    def sample(self, step, state):
        """
        Records all state-variables, nodes and modes for every step of the
        integration.

        """
        time = step * self.dt
        return [time, state]


class SubSample(Monitor):
    """
    Discretely sub-sample the simulation every `istep` integration steps. Time 
    steps that are not modulo `istep` are completely ignored.

    """
    _ui_name = "Temporally sub-sample"

    def __init__(self, **kwargs):
        """Initialise a SubSample monitor from the base Monitor class."""
        LOG.info("%s: initing..." % str(self))
        super(SubSample, self).__init__(**kwargs)
        LOG.debug("%s: inited." % repr(self))


    def sample(self, step, state):
        """Records if integration step corresponds to sampling period."""
        if step % self.istep == 0:
            time = step * self.dt
            return [time, state[self.voi, :]]


class SpatialAverage(Monitor):
    """
    Monitors the averaged value for the models variable of interest over sets of
    nodes -- defined by spatial_mask. This is primarily intended for use with
    surface simulations, with a default behaviour, when no spatial_mask is
    specified, of using surface.region_mapping in order to reduce a surface
    simulation back to a single average timeseries for each region in the
    associated Connectivity. However, any vector of length nodes containing
    integers, from a set contiguous from zero, specifying the new grouping to
    which each node belongs should work.

    Additionally, this monitor temporally sub-samples the simulation every `istep` 
    integration steps.

    """
    _ui_name = "Spatial average with temporal sub-sample"

    spatial_mask = arrays.IntegerArray( #TODO: Check it's a vector of length Nodes (like region mapping for surface)
        label = "An index mask of nodes into areas",
        doc = """A vector of length==nodes that assigns an index to each node
            specifying the "region" to which it belongs. The default usage is
            for mapping a surface based simulation back to the regions used in 
            its `Long-range Connectivity.`""")
    
    default_mask = basic.Enumerate(
                              label = "Default Mask",
                              options = ["cortical", "hemispheres"],
                              default = ["hemispheres"],
                              doc = r"""Fallback in case spatial mask is none and no surface provided 
                              to use either connectivity hemispheres or cortical attributes.""",
                              order = -1)



    def __init__(self, **kwargs):
        """
        Initialise a SpatialAverage monitor from the base Monitor class. Add an
        additional place holder attribute, specific to this Monitor, for the
        array which averages nodes of the simulation to the new set of
        time-series specified by the spatial_mask.

        """
        LOG.info("%s: initing..." % str(self))
        super(SpatialAverage, self).__init__(**kwargs)

        self.spatial_mean = None
        LOG.debug("%s: inited." % repr(self))


    def config_for_sim(self, simulator):
        """
        Sets sampling period, variable/s of interest, and the array used to
        group nodes, by default the surface's region_mapping is used.

        """
        super(SpatialAverage, self).config_for_sim(simulator)
        self.is_default_special_mask = False

        if self.spatial_mask.size == 0:
            self.is_default_special_mask = True
            if not (simulator.surface is None):
                self.spatial_mask = simulator.surface.region_mapping
            else:
                conn = simulator.connectivity
                if self.default_mask[0] == 'cortical':
                    if conn is not None and conn.cortical is not None and conn.cortical.size > 0:
                        ## Use as spatial-mask cortical/non cortical areas
                        self.spatial_mask = [int(c) for c in conn.cortical]
                    else:
                        msg = "Must fill Spatial Mask parameter for non-surface simulations when using SpatioTemporal monitor!"
                        LOG.error(msg)
                        raise Exception(msg)
                if self.default_mask[0] == 'hemispheres':
                    if conn is not None and conn.hemispheres is not None and conn.hemispheres.size > 0:
                        ## Use as spatial-mask left/right hemisphere
                        self.spatial_mask = [int(h) for h in conn.hemispheres]
                    else:
                        msg = "Must fill Spatial Mask parameter for non-surface simulations when using SpatioTemporal monitor!"
                        LOG.error(msg)
                        raise Exception(msg)

        number_of_nodes = simulator.number_of_nodes
        LOG.debug("%s: number_of_nodes = %s" % (str(self), number_of_nodes))
        if self.spatial_mask.size != number_of_nodes:
            msg = "spatial_mask must be a vector of length number_of_nodes."
            LOG.error(msg)
            raise Exception(msg)

        areas = numpy.unique(self.spatial_mask)
        number_of_areas = len(areas)
        LOG.debug("%s: number_of_areas = %s" % (str(self), number_of_areas))
        if not numpy.all(areas == numpy.arange(number_of_areas)):
            msg = ("Areas in the spatial_mask must be specified as a "
                    "contiguous set of indices starting from zero.")
            LOG.error(msg)
            raise Exception(msg)

        util.log_debug_array(LOG, self.spatial_mask, "spatial_mask", owner=self.__class__.__name__)

        spatial_sum = numpy.zeros((number_of_nodes, number_of_areas))
        spatial_sum[numpy.arange(number_of_nodes), self.spatial_mask] = 1

        spatial_sum = spatial_sum.T

        util.log_debug_array(LOG, spatial_sum, "spatial_sum")

        nodes_per_area = numpy.sum(spatial_sum, axis=1)[:, numpy.newaxis]
        self.spatial_mean = spatial_sum / nodes_per_area

        util.log_debug_array(LOG, self.spatial_mean, "spatial_mean", owner=self.__class__.__name__)


    def sample(self, step, state):
        """
        This method records the state of the simulation if the integration step
        corresponds to the sampling period of this Monitor. The nodes of the
        simulation are effectively combined into groups, and then time-series
        representing the average activity within these groups is returned.

        """
        if step % self.istep == 0:
            time = step * self.dt
            monitored_state = numpy.dot(self.spatial_mean, state[self.voi, :])
            return [time, monitored_state.transpose((1, 0, 2))]

    def create_time_series(self, storage_path, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        if self.is_default_special_mask:
            return TimeSeriesRegion(storage_path=storage_path,
                                    sample_period=self.period,
                                    region_mapping=region_map,
                                    region_mapping_volume=region_volume_map,
                                    title='Regions ' + self.__class__.__name__,
                                    connectivity=connectivity,
                                    **self._transform_user_tags())
        else:
            # mask does not correspond to the number of regions
            # let the parent create a plain TimeSeries
            return super(SpatialAverage, self).create_time_series(storage_path)


class GlobalAverage(Monitor):
    """
    Monitors the averaged value for the model's variables of interest over all
    the nodes at each sampling period. This mainly exists as a "convenience"
    monitor for quickly checking the "global" state of a simulation.

    """
    _ui_name = "Global average"


    def __init__(self, **kwargs):
        """Initialise a GlobalAverage monitor from the base Monitor class."""
        LOG.info("%s: initing..." % str(self))
        super(GlobalAverage, self).__init__(**kwargs)
        LOG.debug("%s: inited." % repr(self))


    def sample(self, step, state):
        """Records if integration step corresponds to sampling period."""
        if step % self.istep == 0:
            time = step * self.dt
            data = numpy.mean(state[self.voi, :], axis=1)[:, numpy.newaxis, :]
            return [time, data]

    def create_time_series(self, storage_path, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        # ignore connectivity and surface and let parent create a TimeSeries
        return super(GlobalAverage, self).create_time_series(storage_path)


class TemporalAverage(Monitor):
    """
    Monitors the averaged value for the model's variable/s of interest over all
    the nodes at each sampling period. Time steps that are not modulo ``istep``
    are stored temporarily in the ``_stock`` attribute and then that temporary
    store is averaged and returned when time step is modulo ``istep``.

    """
    _ui_name = "Temporal average"


    def __init__(self, **kwargs):
        """Initialise a TemporalAverage monitor from the base Monitor class."""
        LOG.info("%s: initing..." % str(self))
        super(TemporalAverage, self).__init__(**kwargs)
        LOG.debug("%s: inited." % repr(self))


    def config_for_sim(self, simulator):
        """
        Set the monitor's variables of interest based on the model
        specification. Calculates the number of integration steps (isteps)
        between returns by the record method. And initialises the stock array
        over which the temporal averaging will be performed.

        """
        super(TemporalAverage, self).config_for_sim(simulator)

        stock_size = (self.istep, self.voi.shape[0],
                      simulator.number_of_nodes,
                      simulator.model.number_of_modes)
        LOG.debug("%s: stock_size is %s" % (str(self), str(stock_size)))

        self._stock = numpy.zeros(stock_size)


    def sample(self, step, state):
        """
        Records if integration step corresponds to sampling period, Otherwise
        just update the monitor's stock. When the step corresponds to the sample
        period, the ``_stock`` is averaged over time for return. 

        """
        try:
            self._stock[((step % self.istep) - 1), :] = state[self.voi]
        except IndexError:
            self.voi
        if step % self.istep == 0:
            avg_stock = numpy.mean(self._stock, axis=0)
            time = (step - self.istep / 2.0) * self.dt
            return [time, avg_stock]


class Projection(Monitor):
    "Base class monitor providing lead field suppport."
    _ui_name = "Projection matrix"

    projection = ProjectionMatrix(
        default=None, label='Projection matrix',
        doc='Projection matrix to apply to sources.')

    region_mapping = RegionMapping(
        required=False,
        label="region mapping",
        doc="""An index vector of length equal to the number_of_vertices + the
            number of non-cortical regions, with values that index into an
            associated connectivity matrix.""")


    @staticmethod
    def oriented_gain(gain, orient):
        "Apply orientations to gain matrix."
        return (gain.reshape((gain.shape[0], -1, 3)) * orient).sum(axis=-1)

    @classmethod
    def from_files(cls, sensors_fname, projection_fname, **kwds):
        """
        Build Projection-based monitor from sensors and projection files, and
        any extra keyword arguments are passed to the monitor class constructor.

        """

        return cls(
            sensors=type(cls.sensors).from_file(sensors_fname),
            projection=ProjectionMatrix.from_file(projection_fname),
            **kwds
        )

    def analytic(self, loc, ori):
        "Construct analytic or default set of spatial filters for simulation."
        # this will not be implemented but kept for API uniformity
        raise NotImplementedError(
            "No general purpose analytic formula available for spatial "
            "projection matrices. Please select an appropriate projection "
            "matrix."
        )

    def config_for_sim(self, simulator):
        "Configure projection matrix monitor for given simulation."

        super(Projection, self).config_for_sim(simulator)
        self._sim = simulator

        if hasattr(self, 'sensors'):
            self.sensors.configure()

        # setup convenient locals
        surf = simulator.surface
        conn = simulator.connectivity
        using_cortical_surface = surf is not None
        if using_cortical_surface:
            non_cortical_indices, = numpy.where(numpy.bincount(surf.region_mapping) == 1)
            self.region_mapping = surf.region_mapping
        else:
            non_cortical_indices, = numpy.where(~conn.cortical)
            if self.region_mapping is None:
                raise Exception("Please specify a region mapping on the EEG/MEG/iEEG monitor when "
                                "performing a region simulation.")

        have_subcortical = len(non_cortical_indices) > 0

        # determine source space
        if using_cortical_surface:
            sources = {'loc': surf.vertices, 'ori': surf.vertex_normals}
        else:
            sources = {'loc': conn.centres, 'ori': conn.orientations}

        # compute analytic if not provided
        if self.projection is None:
            self.projection = ProjectionMatrix(projection_data=self.analytic(**sources))

        # reduce to region lead field if region sim
        if not using_cortical_surface:
            proj = numpy.zeros((self.gain.shape[0], conn.number_of_regions))
            numpy.add.at(proj.T, self.region_mapping, self.projection.T)
            self.projection = proj

        # append analytic sub-cortical to lead field
        if have_subcortical:
            # need matrix of shape (proj.shape[0], len(sc_ind))
            src = conn.centres[non_cortical_indices], conn.orientations[non_cortical_indices]
            self.gain = numpy.hstack((self.gain, self.analytic(*src)))

        # zero out unusable sensors
        if self.sensors.usable is not None and not self.sensors.usable.all():
            self.gain[~self.sensors.usable] = 0.0

        # attrs used for recording
        self._state = numpy.zeros((self.gain.shape[0], len(self._transforms.pre)))
        self._period_in_steps = int(self.period / self.dt)

    def sample(self, step, state):
        "Record state, returning sample at sampling frequency / period."
        self._state += self.gain.dot(state[self.voi].sum(axis=-1).T)
        if step % self._period_in_steps == 0:
            time = (step - self._period_in_steps / 2.0) * self.dt
            sample = self._state.copy() / self._period_in_steps
            self._state[:] = 0.0
            return time, sample.reshape((1, -1, 1)) # for compatibility

    def _get_gain(self):
        return self.projection.projection_data

    def _set_gain(self, new_gain):
        self.projection.projection_data = new_gain

    gain = property(_get_gain, _set_gain)


class EEG(Projection):
    """Forward solution monitor for electroencephalogy (EEG). If a
    precomputed lead field is not available, a single sphere analytic
    formula due to Sarvas 1987 is used.

    .. [Sarvas_1987] Sarvas, J., *Basic mathematical and electromagnetic
    concepts of the biomagnetic inverse problem*, Physics in Medicine and
    Biology, 1987.

    """
    _ui_name = "EEG"

    reference = basic.String(required=False, label="EEG Reference",
                             doc='EEG Electrode to be used as reference, or "average" to '
                                 'apply an average reference. If none is provided, the '
                                 'produced time-series are the idealized or reference-free.')

    sensors = SensorsEEG(required=True, label="EEG Sensors",
                         doc='Sensors to use for this EEG monitor')

    sigma = basic.Float(label="Conductivity (w/o projection)", default=1.0,
                        doc='When a projection matrix is not used, this provides '
                            'the value of conductivity in the formula for the single '
                            'sphere approximation of the head (Sarvas 1987).')

    def config_for_sim(self, simulator):
        super(EEG, self).config_for_sim(simulator)
        self._ref_vec = numpy.zeros((self.sensors.number_of_sensors, ))
        if self.reference:
            if self.reference.lower() != 'average':
                sensor_names = self.sensors.labels.tolist()
                self._ref_vec[sensor_names.indexof(self.reference)] = 1.0
            else:
                self._ref_vec[:] = 1.0 / self.sensors.number_of_sensors
        self._ref_vec_mask = numpy.isfinite(self.gain).all(axis=1)
        self._ref_vec = self._ref_vec[self._ref_vec_mask]

    def analytic(self, loc, ori):
        "Equation 12 of [Sarvas_1987]_"
        # r => sensor positions
        # r_0 => source positions
        # a => vector from sources_to_sensor
        # Q => source unit vectors
        r_0, Q = loc, ori
        center = numpy.mean(r_0, axis=0)[numpy.newaxis, ]
        radius = 1.05125 * max(numpy.sqrt(numpy.sum((r_0 - center)**2, axis=1)))
        loc = self.sensors.locations.copy()
        sen_dis = numpy.sqrt(numpy.sum((loc)**2, axis=1))
        loc = loc / sen_dis[:, numpy.newaxis] * radius + center
        V_r = numpy.zeros((loc.shape[0], r_0.shape[0]))
        for sensor_k in numpy.arange(loc.shape[0]):
            a = loc[sensor_k, :] - r_0
            na = numpy.sqrt(numpy.sum(a**2, axis=1))[:, numpy.newaxis]
            V_r[sensor_k, :] = numpy.sum(Q * (a / na**3), axis=1 ) / (4.0 * numpy.pi * self.sigma)
        return V_r

    def sample(self, step, state):
        maybe_sample = super(EEG, self).sample(step, state)
        if maybe_sample is not None:
            time, sample = maybe_sample
            sample -= self._ref_vec.dot(sample[:, self._ref_vec_mask])[:, numpy.newaxis]
            return time, sample.reshape((1, -1, 1))

    def create_time_series(self, storage_path, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        return TimeSeriesEEG(storage_path=storage_path,
                             sensors=self.sensors,
                             sample_period=self.period,
                             title=' ' + self.__class__.__name__,
                             **self._transform_user_tags())


class MEG(Projection):
    "Forward solution monitor for magnetoencephalography (MEG)."
    _ui_name = "MEG"

    sensors = sensors_module.SensorsMEG(
        label = "MEG Sensors",
        default = None,
        required = True,
        doc = """The set of MEG sensors for which the forward solution will be
        calculated.""")

    def analytic(self, loc, ori):
        """Compute single sphere analytic form of MEG lead field.
        Equation 25 of [Sarvas_1987]_."""
        # the magnetic constant = 1.25663706 × 10-6 m kg s-2 A-2  (H/m)
        mu_0 = 1.25663706e-6 #mH/mm
        # r => sensor positions
        # r_0 => source positions
        # a => vector from sources_to_sensor
        # Q => source unit vectors
        r_0, Q = loc, ori
        centre = numpy.mean(r_0, axis=0)[numpy.newaxis, :]
        radius = 1.01 * max(numpy.sqrt(numpy.sum((r_0 - centre)**2, axis=1)))
        sensor_locations = self.sensors.locations.copy()
        sen_dis = numpy.sqrt(numpy.sum((sensor_locations)**2, axis=1))
        sensor_locations = sensor_locations / sen_dis[:, numpy.newaxis]
        sensor_locations = sensor_locations * radius
        sensor_locations = sensor_locations + centre
        B_r = numpy.zeros((sensor_locations.shape[0], r_0.shape[0], 3))
        for sensor_k in numpy.arange(sensor_locations.shape[0]):
            a = sensor_locations[sensor_k,:] - r_0
            na = numpy.sqrt(numpy.sum(a**2, axis=1))[:, numpy.newaxis]
            rsk = sensor_locations[sensor_k,:][numpy.newaxis, :]
            nr = numpy.sqrt(numpy.sum(rsk**2, axis=1))[:, numpy.newaxis]

            F = a * (nr * a + nr**2 - numpy.sum(r_0 * rsk, axis=1)[:, numpy.newaxis])
            adotr = numpy.sum((a / na) * rsk, axis=1)[:, numpy.newaxis]
            delF = ((na**2 / nr + adotr + 2.0 * na + 2.0 * nr) * rsk -
                    (a + 2.0 * nr + adotr * r_0))

            B_r[sensor_k, :] = ((mu_0 / (4.0 * numpy.pi * F**2)) *
                                (numpy.cross(F * Q, r_0) - numpy.sum(numpy.cross(Q, r_0) *
                                                                     (rsk * delF), axis=1)[:, numpy.newaxis]))
        return numpy.sqrt(numpy.sum(B_r**2, axis=2))

    def create_time_series(self, storage_path, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        return TimeSeriesMEG(storage_path=storage_path,
                             sensors=self.sensors,
                             sample_period=self.period,
                             title=' ' + self.__class__.__name__,
                             **self._transform_user_tags())


class iEEG(Projection):
    "Forward solution for intracranial EEG (not ECoG!)."

    _ui_name = "Intracerebral / Stereo EEG"

    sigma = basic.Float(label="conductivity", default=1.0)

    sensors = sensors_module.SensorsInternal(
        label="Internal brain sensors", default=None, required=True,
        doc="The set of SEEG sensors for which the forward solution will be calculated.")

    def analytic(self, loc, ori):
        """Compute the projection matrix -- simple distance weight for now.
        Equation 12 from sarvas1987basic (point dipole in homogeneous space):
          V(r) = 1/(4*pi*\sigma)*Q*(r-r_0)/|r-r_0|^3
        """
        r_0, Q = loc, ori
        V_r = numpy.zeros((self.sensors.locations.shape[0], r_0.shape[0]))
        for sensor_k in numpy.arange(self.sensors.locations.shape[0]):
            a = self.sensors.locations[sensor_k, :] - r_0
            na = numpy.sqrt(numpy.sum(a ** 2, axis=1))[:, numpy.newaxis]
            V_r[sensor_k, :] = numpy.sum(Q * (a / na ** 3), axis=1) / (4.0 * numpy.pi * self.sigma)
        return V_r

    def create_time_series(self, storage_path, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        return TimeSeriesSEEG(storage_path=storage_path,
                              sensors=self.sensors,
                              sample_period=self.period,
                              title=' ' + self.__class__.__name__,
                              **self._transform_user_tags())


"""
#NOTE: It's probably best to do voxelization as an offline "analysis" style
#      process, returning region or surface timeseries for BOLD based on the
#      simulation. The voxelization only really makes sense for surface anyway,
#      and it should be interesting/benificial to be able to compare the effect
#      of different voxelizations on the same underlying surface.

#SK:   For a number of reasons, it's probably best to avoid returning TimeSeriesVolume ,
#      from a simulation directly, instead just stick with the source, i.e. Region and Surface,
#      then later we can add a voxelization "analyser" to produce TimeSeriesVolume on which Volume
#      based analysers and visualisers (which don't exist yet) can operate.
"""


class Bold(Monitor):
    """

    Base class for the Bold monitor.

    **Attributes**
    
        hrf_kernel: the haemodynamic response function (HRF) used to compute 
                    the BOLD (Blood Oxygenation Level Dependent) signal.

        length    : duration of the hrf in seconds.

        period    : the monitor's period

    **References**:
      
    .. [B_1997] Buxton, R. and Frank, L., *A Model for the Coupling between 
        Cerebral Blood Flow and Oxygen Metabolism During Neural Stimulation*,
        17:64-72, 1997.

    .. [Fr_2000] Friston, K., Mechelli, A., Turner, R., and Price, C., *Nonlinear
        Responses in fMRI: The Balloon Model, Volterra Kernels, and Other 
        Hemodynamics*, NeuroImage, 12, 466 - 477, 2000.

    .. [Bo_1996] Geoffrey M. Boynton, Stephen A. Engel, Gary H. Glover and David
        J. Heeger (1996). Linear Systems Analysis of Functional Magnetic Resonance 
        Imaging in Human V1. J Neurosci 16: 4207-4221

    .. [Po_2000] Alex Polonsky, Randolph Blake, Jochen Braun and David J. Heeger
        (2000). Neuronal activity in human primary visual cortex correlates with
        perception during binocular rivalry. Nature Neuroscience 3: 1153-1159

    .. [Gl_1999] Glover, G. *Deconvolution of Impulse Response in Event-Related BOLD fMRI*.
        NeuroImage 9, 416-429, 1999.

  
    .. VJ derivation in the review paper.

    .. note:: LIMITATIONS: sampling period must be integer multiple of 500ms

    .. note:: CONSIDERATIONS: It is  sensible to use this monitor if your 
              simulation length is > 30s (30000ms)

    .. note:: gamma and polonsky are based on the nitime implementation
              http://nipy.org/nitime/api/generated/nitime.fmri.hrf.html

    .. note:: see Tutorial_Exploring_The_Bold_Monitor

    .. warning:: Not yet tested, debugged, generalised etc...
    .. wisdom and plagiarism

    """
    _ui_name = "BOLD"

    #Over-ride the Monitor baseclass period...
    period = basic.Float(
        label = "Sampling period (ms)",
        default = 2000.0,
        doc = """For the BOLD monitor, sampling period in milliseconds must be
        an integral multiple of 500. Typical measurment interval (repetition 
        time TR) is between 1-3 s. If TR is 2s, then Bold period is 2000ms.""")

    hrf_kernel = equations.HRFKernelEquation(
        label = "Haemodynamic Response Function",
        default = equations.FirstOrderVolterra,
        required = True,
        doc = """A tvb.datatypes.equation object which describe the haemodynamic
        response function used to compute the BOLD signal.""") 

    hrf_length = basic.Float(
        label = "Duration (ms)",
        default = 20000.,
        doc= """Duration of the hrf kernel""",
        order=-1)


    def __init__(self, **kwargs):
        """
        Initialise from the base Monitor class.

        """
        LOG.info("%s: initing..." % str(self))
        super(Bold, self).__init__(**kwargs)
        LOG.warning("%s: Needs testing, debugging, etc..." % repr(self))

        #Bold measurement period is much larger than our simulation's dt, so we
        #need an interim downsampling to support it. 
        #TODO: adapt monitors to support chaining, so we can just stick a 
        #      temporal average followed by the bold response convolution and
        #      then a spatial average (node to voxel, only really relevant for 
        #      surface simulations).
        self._interim_period = None
        self._interim_istep = None
        self._interim_stock = None #I hate bold.
        self._stock_steps = None
        self._stock_time = None    # Me too
        self._stock_sample_rate = 2**-2

        self.hemodynamic_response_function = None

        LOG.debug("%s: inited." % repr(self))


    def compute_hrf(self):
        """
        Compute the heamodynamic response function.
        """

        #TODO: Current traits limitations require this moved to config_for_sim()
        if numpy.mod(self.period, 500.0): #TODO: This is a temporary limit, need to fix configure...
            msg = "%s: BOLD.period must be a multiple of 500.0, period = %s"
            LOG.error(msg % (str(self), str(self.period)))

        #downsample average over simulator steps to give a fixed sampling rate (256Hz)
        #then (18.75 * tau_s) * 256  ==> 3840 required length of _stock

        # simulation in ms therefore 1000.0/256.0 ==> 3.90625 _interim_period in ms
        LOG.warning("%s: Needs testing, debugging, etc..." % repr(self))

        self._stock_sample_rate = 2.0**-2 #/ms    # NOTE: An integral multiple of dt
        magic_number = self.hrf_length #* 0.8      # truncates G, volterra kernel, once ~zero 

        #Length of history needed for convolution in steps @ _stock_sample_rate
        required_history_length = self._stock_sample_rate * magic_number # 3840 for tau_s=0.8
        self._stock_steps = numpy.ceil(required_history_length).astype(int)
        stock_time_max    = magic_number/1000.0                                # [s]
        stock_time_step   = stock_time_max / self._stock_steps                 # [s]
        self._stock_time  = numpy.arange(0.0, stock_time_max, stock_time_step) # [s]

        LOG.debug("%s: Required history length for performing convolution = %s" % 
                (str(self), self._stock_steps))

        
        #Compute the HRF kernel
        self.hrf_kernel.pattern = self._stock_time
        G = self.hrf_kernel.pattern

        #Reverse it, need it into the past for matrix-multiply of stock
        G = G[::-1]
        self.hemodynamic_response_function = G[numpy.newaxis, :]


        util.log_debug_array(LOG, self.hemodynamic_response_function,
                             "hemodynamic_response_function",
                             owner=self.__class__.__name__)

        #Interim stock configuration
        self._interim_period = 1.0 / self._stock_sample_rate #period in ms
        self._interim_istep = int(round(self._interim_period / self.dt)) # interim period in integration time steps


    def config_for_sim(self, simulator):
        """
        Set up stock arrays
       
        """
        super(Bold, self).config_for_sim(simulator)

        self.compute_hrf() # now we have self.hemodynamic_response_function

        interim_stock_size = (self._interim_istep, self.voi.shape[0],
                              simulator.number_of_nodes,
                              simulator.model.number_of_modes) 
        LOG.debug("%s: interim_stock_size is %s" % (str(self), str(interim_stock_size)))

        self._interim_stock = numpy.zeros(interim_stock_size)

        #Stock configuration
        stock_size = (self._stock_steps, self.voi.shape[0],
                      simulator.number_of_nodes,
                      simulator.model.number_of_modes)
        LOG.debug("%s: stock_size is %s" % (str(self), str(stock_size)))

        #Set the initingal _stock based on simulator.history
        mean_history = numpy.mean(simulator.history[:, self.voi, :, :], axis=0)
        self._stock = mean_history[numpy.newaxis,:] * numpy.ones(stock_size)
        #NOTE: BOLD can have a long (~15s) transient that is mainly due to the
        #      initial dynamic transient from simulations that are started with 
        #      imperfect initial conditions.
        #import pdb; pdb.set_trace()


    def sample(self, step, state):
        """
        Returns a result if integration step corresponds to sampling period.
        Updates the interim-stock on every step and updates the stock if the 
        step corresponds to the interim period.

        """
        #Update the interim-stock at every step
        self._interim_stock[((step % self._interim_istep) - 1), :] = state[self.voi, :]

        #At stock's period update it with the temporal average of interim-stock
        if step % self._interim_istep == 0:
            avg_interim_stock = numpy.mean(self._interim_stock, axis=0)
            self._stock[((step/self._interim_istep % self._stock_steps) - 1), :] = avg_interim_stock

        #At the monitor's period, apply the heamodynamic response function to
        #the stock and return the resulting BOLD signal.
        if step % self.istep == 0:
            time = step * self.dt
            hrf = numpy.roll(self.hemodynamic_response_function,
                             ((step/self._interim_istep % self._stock_steps) - 1),
                             axis=1)
            if isinstance(self.hrf_kernel, equations.FirstOrderVolterra):
                k1_V0 = self.hrf_kernel.parameters["k_1"] * self.hrf_kernel.parameters["V_0"]
                bold = (numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3))) - 1.0) * k1_V0
            else:
                bold = numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3)))
            bold = bold.reshape(self._stock.shape[1:])
            bold = bold.sum(axis=0)[numpy.newaxis,:,:] #state-variables
            bold = bold.sum(axis=2)[:,:,numpy.newaxis] #modes
            return [time, bold]


class BoldRegionROI(Bold):
    """
    The BoldRegionROI monitor assumes that it is being used on a surface and 
    uses the region mapping of the surface to generate regional signals which
    are the spatial average of all vertices in the region. 

    This was originated to compare the results of a Bold monitor with a 
    region level simulation with that of an otherwise identical surface
    simulation.

    """
    _ui_name = "BOLD Region ROI (only with surface)"

    def config_for_sim(self, simulator):
        super(BoldRegionROI, self).config_for_sim(simulator)
        self.region_mapping = simulator.surface.region_mapping

    def sample(self, step, state, array=numpy.array):
        result = super(BoldRegionROI, self).sample(step, state)
        if result:
            t, data = result
            return [t, array([data.flat[self.region_mapping==i].mean()
                              for i in xrange(self.region_mapping.max())])]
        else:
            return None


class BalloonWindkesselAccordingToKJFristonEtAl2003NeuroImage(Monitor):
    """

    The full Balloon-Windkessel model, following Friston et al's paper
    detailing the equations describing hemodynamic state. 

    Currently, this monitor models the BOLD activity of the nodes of the
    simulation, not voxels. 

    .. [F_2003] Friston, K., Harrison, L., and Penny, W., *Dynamic causal
        modeling*, NeuroImage, 19, 1273 - 1302, 2003.

    """
    _ui_name = "Balloon-Windkessel"

    class config:

        period = basic.Float(
            label = "Sampling period (ms)",
            default = 1000.0,
            doc = """Sampling period of the Balloon""")

        nvcvar = basic.Integer(
            label = "Neurovascular coupling variable index",
            default = 0,
            doc = ""
            )

        #                                   prior mean          prior variance

        kappa = basic.Float(
            label="",
            default=                        0.65, # per/s       0.015
            doc="Rate of signal decay"
            )

        gamma = basic.Float(
            label="",
            default=                        0.41, # per/s       0.002
            doc="Rate of flow-dependent elimination"
            )

        tau = basic.Float(
            label="",
            default=                        0.98, # s           0.0568
            doc="Haemodynamic transit time"
            )

        alpha = basic.Float(
            label="",
            default=                        0.32, #             0.0015
            doc="Grubb's exponent"
            )

        rho = basic.Float(
            label="",
            default=                        0.34, #             0.0024
            doc="Resting oxygen extraction fraction"
            )

    """
    class state:

        z_i = FloatArray()
        s_i = FloatArray()
        f_i = FloatArray()
        v_i = FloatArray()
        q_i = FloatArray()
        y_i = FloatArray()

    class ddt:
        pass

    """
        
    def __init__(self, *args, **kwds):
        raise NotImplementedError
    
    def sample(self, step, state):
        """
        Returns a result if integration step corresponds to sampling period.
        Updates the interim-stock on every step and updates the stock if the 
        step corresponds to the interim period.

        """
        #Update the interim-stock at every step
        self._interim_stock[((step % self._interim_istep) - 1), :] = state[self.voi, :]

        #At stock's period update it with the temporal average of interim-stock
        if step % self._interim_istep == 0:
            #import pdb; pdb.set_trace()
            avg_interim_stock = numpy.mean(self._interim_stock, axis=0)
            self._stock[((step/self._interim_istep % self._stock_steps) - 1), :] = avg_interim_stock

        #At the monitor's period, apply the heamodynamic response function to
        #the stock and return the resulting BOLD signal.
        if step % self.istep == 0:
            time = step * self.dt
            hrf = numpy.roll(self.hemodynamic_response_function,
                             ((step/self._interim_istep % self._stock_steps) - 1),
                             axis=1)
            bold = numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3)))
            bold = bold.reshape(self._stock.shape[1:])
            bold = bold.sum(axis=0)[numpy.newaxis, :, :]    #state-variables
            bold = bold.sum(axis=2)[:, :, numpy.newaxis]    #modes
            return [time, bold]
