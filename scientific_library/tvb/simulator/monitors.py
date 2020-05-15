# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and 
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

import abc
import numpy
from tvb.datatypes.time_series import (TimeSeries, TimeSeriesRegion, TimeSeriesEEG, TimeSeriesMEG, TimeSeriesSEEG,
                                       TimeSeriesSurface)
from tvb.simulator import noise
import tvb.datatypes.sensors as sensors_module
import tvb.datatypes.projections as projections_module
from tvb.datatypes.region_mapping import RegionMapping
import tvb.datatypes.equations as equations
from tvb.simulator.common import iround, numpy_add_at
from tvb.basic.neotraits.api import HasTraits, Attr, NArray, Float, narray_describe



class Monitor(HasTraits):
    """
    Abstract base class for monitor implementations.
    """

    period = Float(
        label="Sampling period (ms)",  # order = 10
        default=0.9765625,  # ms. 0.9765625 => 1024Hz #ms, 0.5 => 2000Hz
        doc="""Sampling period in milliseconds, must be an integral multiple
        of integration-step size. As a guide: 2048 Hz => 0.48828125 ms ;  
        1024 Hz => 0.9765625 ms ; 512 Hz => 1.953125 ms.""")

    variables_of_interest = NArray(
        dtype=int,
        label="Model variables to watch",  # order=11,
        doc=("Indices of model's variables of interest (VOI) that this monitor should record. "
             "Note that the indices should start at zero, so that if a model offers VOIs V, W and "
             "V+W, and W is selected, and this monitor should record W, then the correct index is 0."),
        required=False)

    istep = None
    dt = None
    voi = None
    _stock = numpy.empty([])

    def __str__(self):
        clsname = self.__class__.__name__
        return '%s(period=%f, voi=%s)' % (clsname, self.period, self.variables_of_interest.tolist())

    def config_for_sim(self, simulator):
        """Configure monitor for given simulator.

        Grab the Simulator's integration step size. Set the monitor's variables
        of interest based on the Monitor's 'variables_of_interest' attribute, if
        it was specified, otherwise use the 'variables_of_interest' specified 
        for the Model. Calculate the number of integration steps (isteps)
        between returns by the record method. This method is called from within
        the the Simulator's configure() method.

        """
        self.dt = simulator.integrator.dt
        self.istep = iround(self.period / self.dt)
        self.voi = self.variables_of_interest
        if self.voi is None or self.voi.size == 0:
            self.voi = numpy.r_[:len(simulator.model.variables_of_interest)]

    def record(self, step, observed):
        """Record a sample of the observed state at given step.

        This is a final method called by the simulator to obtain samples from a
        monitor instance. Monitor subclasses should not override this method, but
        rather implement the `sample` method.

        """
        return self.sample(step, observed)

    @abc.abstractmethod
    def sample(self, step, state):
        """
        This method provides monitor output, and should be overridden by subclasses.

        """

    def create_time_series(self, connectivity=None, surface=None, region_map=None, region_volume_map=None):
        """
        Create a time series instance that will be populated by this monitor
        :param surface: if present a TimeSeriesSurface is returned
        :param connectivity: if present a TimeSeriesRegion is returned
        Otherwise a plain TimeSeries will be returned
        """
        if surface is not None:
            return TimeSeriesSurface(surface=surface.region_mapping_data.surface,
                                     sample_period=self.period,
                                     title='Surface ' + self.__class__.__name__)
        if connectivity is not None:
            return TimeSeriesRegion(connectivity=connectivity,
                                    region_mapping=region_map,
                                    region_mapping_volume=region_volume_map,
                                    sample_period=self.period,
                                    title='Regions ' + self.__class__.__name__)

        return TimeSeries(sample_period=self.period,
                          title=' ' + self.__class__.__name__)


class Raw(Monitor):
    """
    A monitor that records the output raw data from a tvb simulation:
    It collects:

        - all state variables and modes from class :Model:
        - all nodes of a region or surface based 
        - all the integration time steps

    """
    _ui_name = "Raw recording"

    period = Float(default=0.0, label="Sampling period is ignored for Raw Monitor")
    # order = -1

    variables_of_interest = NArray(
        dtype=int,
        label="Raw Monitor sees all!!! Resistance is futile...",
        required=False)
    # order = -1

    def config_for_sim(self, simulator):
        if self.period != simulator.integrator.dt:
            self.log.debug('Raw period not equal to integration time step, overriding')
        self.period = simulator.integrator.dt
        super(Raw, self).config_for_sim(simulator)
        self.istep = 1
        self.voi = numpy.arange(len(simulator.model.variables_of_interest))

    def sample(self, step, state):
        time = step * self.dt
        return [time, state]


class SubSample(Monitor):
    """
    Sub-samples or decimates the solution in time.

    """
    _ui_name = "Temporally sub-sample"

    def sample(self, step, state):
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

    spatial_mask = NArray(  #TODO: Check it's a vector of length Nodes (like region mapping for surface)
        dtype=int,
        label="An index mask of nodes into areas",
        required=False,
        doc="""A vector of length==nodes that assigns an index to each node
            specifying the "region" to which it belongs. The default usage is
            for mapping a surface based simulation back to the regions used in 
            its `Long-range Connectivity.`""")

    default_mask = Attr(
        str,
        choices=("cortical", "hemispheres"),
        default="hemispheres",
        label="Default Mask",
        doc=("Fallback in case spatial mask is none and no surface provided" 
             "to use either connectivity hemispheres or cortical attributes."))
        # order = -1)

    def config_for_sim(self, simulator):

        # initialize base attributes
        super(SpatialAverage, self).config_for_sim(simulator)
        self.is_default_special_mask = False

        # setup given spatial mask or default to region mapping
        if self.spatial_mask is None:
            self.is_default_special_mask = True
            if not (simulator.surface is None):
                self.spatial_mask = simulator.surface.region_mapping
            else:
                conn = simulator.connectivity
                if self.default_mask == 'cortical':
                    if conn is not None and conn.cortical is not None and conn.cortical.size > 0:
                        ## Use as spatial-mask cortical/non cortical areas
                        self.spatial_mask = numpy.array([int(c) for c in conn.cortical])
                    else:
                        msg = "Must fill Spatial Mask parameter for non-surface simulations when using SpatioTemporal monitor!"
                        raise Exception(msg)
                if self.default_mask == 'hemispheres':
                    if conn is not None and conn.hemispheres is not None and conn.hemispheres.size > 0:
                        ## Use as spatial-mask left/right hemisphere
                        self.spatial_mask = numpy.array([int(h) for h in conn.hemispheres])
                    else:
                        msg = "Must fill Spatial Mask parameter for non-surface simulations when using SpatioTemporal monitor!"
                        raise Exception(msg)

        number_of_nodes = simulator.number_of_nodes
        if self.spatial_mask.size != number_of_nodes:
            msg = "spatial_mask must be a vector of length number_of_nodes."
            raise Exception(msg)

        areas = numpy.unique(self.spatial_mask)
        number_of_areas = len(areas)
        if not numpy.all(areas == numpy.arange(number_of_areas)):
            msg = ("Areas in the spatial_mask must be specified as a "
                    "contiguous set of indices starting from zero.")
            raise Exception(msg)

        self.log.debug("spatial_mask")
        self.log.debug(narray_describe(self.spatial_mask))
        spatial_sum = numpy.zeros((number_of_nodes, number_of_areas))
        spatial_sum[numpy.arange(number_of_nodes), self.spatial_mask] = 1
        spatial_sum = spatial_sum.T
        self.log.debug("spatial_sum")
        self.log.debug(narray_describe(spatial_sum))
        nodes_per_area = numpy.sum(spatial_sum, axis=1)[:, numpy.newaxis]
        self.spatial_mean = spatial_sum / nodes_per_area
        self.log.debug("spatial_mean")
        self.log.debug(narray_describe(self.spatial_mean))


    def sample(self, step, state):
        if step % self.istep == 0:
            time = step * self.dt
            monitored_state = numpy.dot(self.spatial_mean, state[self.voi, :])
            return [time, monitored_state.transpose((1, 0, 2))]

    def create_time_series(self, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        if self.is_default_special_mask:
            return TimeSeriesRegion(sample_period=self.period,
                                    region_mapping=region_map,
                                    region_mapping_volume=region_volume_map,
                                    title='Regions ' + self.__class__.__name__,
                                    connectivity=connectivity)
        else:
            # mask does not correspond to the number of regions
            # let the parent create a plain TimeSeries
            return super(SpatialAverage, self).create_time_series()


class GlobalAverage(Monitor):
    """
    Monitors the averaged value for the model's variables of interest over all
    the nodes at each sampling period. This mainly exists as a "convenience"
    monitor for quickly checking the "global" state of a simulation.

    """
    _ui_name = "Global average"

    def sample(self, step, state):
        """Records if integration step corresponds to sampling period."""
        if step % self.istep == 0:
            time = step * self.dt
            data = numpy.mean(state[self.voi, :], axis=1)[:, numpy.newaxis, :]
            return [time, data]

    def create_time_series(self, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        # ignore connectivity and surface and let parent create a TimeSeries
        return super(GlobalAverage, self).create_time_series()


class TemporalAverage(Monitor):
    """
    Monitors the averaged value for the model's variable/s of interest over all
    the nodes at each sampling period. Time steps that are not modulo ``istep``
    are stored temporarily in the ``_stock`` attribute and then that temporary
    store is averaged and returned when time step is modulo ``istep``.

    """
    _ui_name = "Temporal average"

    def config_for_sim(self, simulator):
        super(TemporalAverage, self).config_for_sim(simulator)
        stock_size = (self.istep, self.voi.shape[0],
                      simulator.number_of_nodes,
                      simulator.model.number_of_modes)
        self.log.debug("Temporal average stock_size is %s" % (str(stock_size), ))
        self._stock = numpy.zeros(stock_size)


    def sample(self, step, state):
        """
        Records if integration step corresponds to sampling period, Otherwise
        just update the monitor's stock. When the step corresponds to the sample
        period, the ``_stock`` is averaged over time for return. 

        """
        self._stock[((step % self.istep) - 1), :] = state[self.voi]
        if step % self.istep == 0:
            avg_stock = numpy.mean(self._stock, axis=0)
            time = (step - self.istep / 2.0) * self.dt
            return [time, avg_stock]


# mhtodo: this is not a proper superclass but a mixin, it refers to fields that don't exist

class Projection(Monitor):
    "Base class monitor providing lead field suppport."
    _ui_name = "Projection matrix"

    region_mapping = Attr(
        RegionMapping,
        required=False,
        label="region mapping",  #order=3,
        doc="A region mapping specifies how vertices of a surface correspond to given regions in the"
            " connectivity. For iEEG/EEG/MEG monitors, this must be specified when performing a region"
            " simulation but is optional for a surface simulation.")

    obsnoise = Attr(
        noise.Noise,
        label="Observation Noise",
        default=noise.Additive(),
        required=False,
        doc="""The monitor's noise source. It incorporates its
        own instance of Numpy's RandomState.""")

    @staticmethod
    def oriented_gain(gain, orient):
        "Apply orientations to gain matrix."
        return (gain.reshape((gain.shape[0], -1, 3)) * orient).sum(axis=-1)

    @classmethod
    def projection_class(cls):
        if hasattr(cls, 'projection'):
            return cls.projection.field_type
        else:
            return projections_module.ProjectionMatrix

    @classmethod
    def from_file(cls, sensors_fname, projection_fname, rm_f_name="regionMapping_16k_76.txt",
                  period=1e3/1024.0, **kwds):
        """
        Build Projection-based monitor from sensors and projection files, and
        any extra keyword arguments are passed to the monitor class constructor.

        """
        result = cls(period=period, **kwds)
        result.sensors = cls.sensors.field_type.from_file(sensors_fname)
        result.projection = cls.projection_class().from_file(projection_fname)
        result.region_mapping = RegionMapping.from_file(rm_f_name)
        return result

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

        # handle observation noise and configure white/coloured noise
        # pass in access to the: i) dt and ii) sample shape
        if self.obsnoise is not None:
            # configure the noise level
            if self.obsnoise.ntau > 0.0:
                noiseshape = self.sensors.labels[:,numpy.newaxis].shape
                self.obsnoise.configure_coloured(dt=self.dt, shape=noiseshape)
            else:
                self.obsnoise.configure_white(dt=self.dt)

        # handle region vs simulation, analytic vs numerical proj, cortical vs subcortical.
        # setup convenient locals
        surf = simulator.surface
        conn = simulator.connectivity
        using_cortical_surface = surf is not None
        if using_cortical_surface:
            non_cortical_indices, = numpy.where(numpy.bincount(surf.region_mapping) == 1)
            self.rmap = surf.region_mapping
        else:
            # assume all cortical if no info
            if conn.cortical.size == 0:
                conn.cortical = numpy.array([True] * conn.weights.shape[0])
            non_cortical_indices, = numpy.where(~conn.cortical)
            if self.region_mapping is None:
                raise Exception("Please specify a region mapping on the EEG/MEG/iEEG monitor when "
                                "performing a region simulation.")
            else:
                self.rmap = self.region_mapping

            self.log.debug('Projection used in region sim has %d non-cortical regions', non_cortical_indices.size)

        have_subcortical = len(non_cortical_indices) > 0

        # determine source space
        if using_cortical_surface:
            sources = {'loc': surf.vertices, 'ori': surf.vertex_normals}
        else:
            sources = {'loc': conn.centres[conn.cortical], 'ori': conn.orientations[conn.cortical]}

        # compute analytic if not provided
        if not hasattr(self, 'projection'):
            self.log.debug('Precomputed projection not unavailable using analytic approximation.')
            self.gain = self.analytic(**sources)

        # reduce to region lead field if region sim
        if not using_cortical_surface and self.gain.shape[1] == self.rmap.size:
            gain = numpy.zeros((self.gain.shape[0], conn.number_of_regions))
            numpy_add_at(gain.T, self.rmap, self.gain.T)
            self.log.debug('Region mapping gain shape %s to %s', self.gain.shape, gain.shape)
            self.gain = gain

        # append analytic sub-cortical to lead field
        if have_subcortical:
            # need matrix of shape (proj.shape[0], len(sc_ind))
            src = conn.centres[non_cortical_indices], conn.orientations[non_cortical_indices]
            self.gain = numpy.hstack((self.gain, self.analytic(*src)))
            self.log.debug('Added subcortical analytic gain, for final shape %s', self.gain.shape)

        if self.sensors.usable is not None and not self.sensors.usable.all():
            mask_unusable = ~self.sensors.usable
            self.gain[mask_unusable] = 0.0
            self.log.debug('Zeroed gain coefficients for %d unusable sensors', mask_unusable.sum())

        # unconditionally zero NaN elements; framework not prepared for NaNs.
        nan_mask = numpy.isfinite(self.gain).all(axis=1)
        self.gain[~nan_mask] = 0.0
        self.log.debug('Zeroed %d NaN gain coefficients', nan_mask.sum())

        # attrs used for recording
        self._state = numpy.zeros((self.gain.shape[0], len(self.voi)))
        self._period_in_steps = int(self.period / self.dt)
        self.log.debug('State shape %s, period in steps %s', self._state.shape, self._period_in_steps)

        self.log.info('Projection configured gain shape %s', self.gain.shape)


    def configure(self, *args, **kwargs):
        self.sensors.configure()


    def sample(self, step, state):
        "Record state, returning sample at sampling frequency / period."
        self._state += self.gain.dot(state[self.voi].sum(axis=-1).T)
        if step % self._period_in_steps == 0:
            time = (step - self._period_in_steps / 2.0) * self.dt
            sample = self._state.copy() / self._period_in_steps

            # add observation noise if available
            if self.obsnoise is not None:
                sample += self.obsnoise.generate(shape=sample.shape)

            self._state[:] = 0.0
            return time, sample.T[..., numpy.newaxis] # for compatibility

    _gain = None

    @property
    def gain(self):
        if self._gain is None:
            self._gain = self.projection.projection_data
        return self._gain

    @gain.setter
    def gain(self, new_gain):
        self._gain = new_gain

    _rmap = None

    def _reg_map_data(self, reg_map):
        return getattr(reg_map, 'array_data', reg_map)

    @property
    def rmap(self):
        "Normalize obtaining reg map vector over various possibilities."
        if self._rmap is None:
            self._rmap = self._reg_map_data(self.region_mapping)
        return self._rmap

    @rmap.setter
    def rmap(self, reg_map):
        if self._rmap is not None:
            self._rmap = self._reg_map_data(self.region_mapping)



class EEG(Projection):
    """
    Forward solution monitor for electroencephalogy (EEG). If a
    precomputed lead field is not available, a single sphere analytic
    formula due to Sarvas 1987 is used.

    **References**:

    .. [Sarvas_1987] Sarvas, J., *Basic mathematical and electromagnetic
        concepts of the biomagnetic inverse problem*, Physics in Medicine and
        Biology, 1987.

    """
    _ui_name = "EEG"

    projection = Attr(
        projections_module.ProjectionSurfaceEEG,
        default=None, label='Projection matrix',  #order=2,
        doc='Projection matrix to apply to sources.')

    reference = Attr(
        str, required=False,
        label="EEG Reference",  #order=5,
        doc='EEG Electrode to be used as reference, or "average" to '
            'apply an average reference. If none is provided, the '
            'produced time-series are the idealized or reference-free.')

    sensors = Attr(sensors_module.SensorsEEG, required=True, label="EEG Sensors",  #order=1,
                   doc='Sensors to use for this EEG monitor')

    sigma = Float(
        default=1.0,  #order=4,
        label="Conductivity (w/o projection)",
        doc='When a projection matrix is not used, this provides '
            'the value of conductivity in the formula for the single '
            'sphere approximation of the head (Sarvas 1987).')


    @classmethod
    def from_file(cls, sensors_fname='eeg_brainstorm_65.txt', projection_fname='projection_eeg_65_surface_16k.npy', **kwargs):
        return Projection.from_file.__func__(cls, sensors_fname, projection_fname, **kwargs)

    def config_for_sim(self, simulator):
        super(EEG, self).config_for_sim(simulator)
        self._ref_vec = numpy.zeros((self.sensors.number_of_sensors, ))
        if self.reference:
            if self.reference.lower() != 'average':
                sensor_names = self.sensors.labels.tolist()
                self._ref_vec[sensor_names.index(self.reference)] = 1.0
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
            return time, sample.reshape((state.shape[0], -1, 1))

    def create_time_series(self, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        return TimeSeriesEEG(sensors=self.sensors,
                             sample_period=self.period,
                             title=' ' + self.__class__.__name__)


class MEG(Projection):
    "Forward solution monitor for magnetoencephalography (MEG)."
    _ui_name = "MEG"

    projection = Attr(
        projections_module.ProjectionSurfaceMEG,
        default=None, label='Projection matrix', # order=2,
        doc='Projection matrix to apply to sources.')

    sensors = Attr(
        sensors_module.SensorsMEG,
        label = "MEG Sensors",
        default = None,
        required = True,  #order=1,
        doc="The set of MEG sensors for which the forward solution will be calculated.")


    @classmethod
    def from_file(cls, sensors_fname='meg_brainstorm_276.txt',
                   projection_fname='projection_meg_276_surface_16k.npy', **kwargs):
        return Projection.from_file.__func__(cls, sensors_fname, projection_fname, **kwargs)

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

    def create_time_series(self, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        return TimeSeriesMEG(sensors=self.sensors,
                             sample_period=self.period,
                             title=' ' + self.__class__.__name__)


class iEEG(Projection):
    "Forward solution for intracranial EEG (not ECoG!)."

    _ui_name = "Intracerebral / Stereo EEG"

    projection = Attr(
        projections_module.ProjectionSurfaceSEEG,
        default=None, label='Projection matrix',  #order=2,
        doc='Projection matrix to apply to sources.')

    sigma = Float(label="conductivity", default=1.0)  #, order=4)

    sensors = Attr(
        sensors_module.SensorsInternal,
        label="Internal brain sensors", default=None, required=True,  #order=1,
        doc="The set of SEEG sensors for which the forward solution will be calculated.")


    @classmethod
    def from_file(cls, sensors_fname='seeg_588.txt',
                   projection_fname='projection_seeg_588_surface_16k.npy', **kwargs):
        return Projection.from_file.__func__(cls, sensors_fname, projection_fname, **kwargs)

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

    def create_time_series(self, connectivity=None, surface=None,
                           region_map=None, region_volume_map=None):
        return TimeSeriesSEEG(sensors=self.sensors,
                              sample_period=self.period,
                              title=' ' + self.__class__.__name__)


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

    .. note:: gamma and polonsky are based on the nitime implementation
              http://nipy.org/nitime/api/generated/nitime.fmri.hrf.html

    .. note:: see Tutorial_Exploring_The_Bold_Monitor

    """
    _ui_name = "BOLD"

    period = Float(
        label="Sampling period (ms)",
        default=2000.0,
        doc="""For the BOLD monitor, sampling period in milliseconds must be
        an integral multiple of 500. Typical measurment interval (repetition
        time TR) is between 1-3 s. If TR is 2s, then Bold period is 2000ms.""")

    hrf_kernel = Attr(
        equations.HRFKernelEquation,
        label="Haemodynamic Response Function",
        default=equations.FirstOrderVolterra(),
        required=True,
        doc="""A tvb.datatypes.equation object which describe the haemodynamic
        response function used to compute the BOLD signal.""")

    hrf_length = Float(
        label="Duration (ms)",
        default=20000.,
        doc= """Duration of the hrf kernel""",)
        #order=-1)

    _interim_period = None
    _interim_istep = None
    _interim_stock = None
    _stock_steps = None
    _stock_time = None
    _stock_sample_rate = 2 ** -2
    hemodynamic_response_function = None

    def compute_hrf(self):
        """
        Compute the hemodynamic response function.

        """
        self._stock_sample_rate = 2.0**-2 #/ms    # NOTE: An integral multiple of dt
        magic_number = self.hrf_length #* 0.8      # truncates G, volterra kernel, once ~zero
        #Length of history needed for convolution in steps @ _stock_sample_rate
        required_history_length = self._stock_sample_rate * magic_number # 3840 for tau_s=0.8
        self._stock_steps = numpy.ceil(required_history_length).astype(int)
        stock_time_max    = magic_number/1000.0                                # [s]
        stock_time_step   = stock_time_max / self._stock_steps                 # [s]
        self._stock_time  = numpy.arange(0.0, stock_time_max, stock_time_step) # [s]
        self.log.debug("Bold requires %d steps for HRF kernel convolution", self._stock_steps)
        #Compute the HRF kernel
        G = self.hrf_kernel.evaluate(self._stock_time)
        #Reverse it, need it into the past for matrix-multiply of stock
        G = G[::-1]
        self.hemodynamic_response_function = G[numpy.newaxis, :]
        #Interim stock configuration
        self._interim_period = 1.0 / self._stock_sample_rate #period in ms
        self._interim_istep = int(round(self._interim_period / self.dt)) # interim period in integration time steps
        self.log.debug('Bold HRF shape %s, interim period & istep %d & %d',
                  self.hemodynamic_response_function.shape, self._interim_period, self._interim_istep)

    def config_for_sim(self, simulator):
        super(Bold, self).config_for_sim(simulator)
        self.compute_hrf()
        sample_shape = self.voi.shape[0], simulator.number_of_nodes, simulator.model.number_of_modes
        self._interim_stock = numpy.zeros((self._interim_istep,) + sample_shape)
        self.log.debug("BOLD inner buffer %s %.2f MB" % (
            self._interim_stock.shape, self._interim_stock.nbytes/2**20))
        self._stock = numpy.zeros((self._stock_steps,) + sample_shape)
        self.log.debug("BOLD outer buffer %s %.2f MB" % (
            self._stock.shape, self._stock.nbytes/2**20))


    def sample(self, step, state):
        # Update the interim-stock at every step
        self._interim_stock[((step % self._interim_istep) - 1), :] = state[self.voi, :]
        # At stock's period update it with the temporal average of interim-stock
        if step % self._interim_istep == 0:
            avg_interim_stock = numpy.mean(self._interim_stock, axis=0)
            self._stock[((step//self._interim_istep % self._stock_steps) - 1), :] = avg_interim_stock
        # At the monitor's period, apply the heamodynamic response function to
        # the stock and return the resulting BOLD signal.
        if step % self.istep == 0:
            time = step * self.dt
            hrf = numpy.roll(self.hemodynamic_response_function,
                             ((step//self._interim_istep % self._stock_steps) - 1),
                             axis=1)
            if isinstance(self.hrf_kernel, equations.FirstOrderVolterra):
                k1_V0 = self.hrf_kernel.parameters["k_1"] * self.hrf_kernel.parameters["V_0"]
                bold = (numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3))) - 1.0) * k1_V0
            else:
                bold = numpy.dot(hrf, self._stock.transpose((1, 2, 0, 3)))
            bold = bold.reshape(self._stock.shape[1:])
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
            # TODO use reduceat
            return [t, array([data.flat[self.region_mapping==i].mean()
                              for i in range(self.region_mapping.max())])]
        else:
            return None


class ProgressLogger(Monitor):
    "Logs progress of simulation; only for use in console scripts."

    def __init__(self, **kwargs):
        super(ProgressLogger, self).__init__(**kwargs)

    def config_for_sim(self, simulator):
        self._dt = simulator.integrator.dt
        self._istep = int(self.period / self._dt)

    def record(self, step, state):
        try:
            self._last_step
        except:
            self._last_step = step
        if (step - self._last_step) % self._istep == 0:
            self.log.info('step %d time %.4f s', step, step * self._dt / 1e3)

    def sample(self, step, state):
        raise NotImplementedError