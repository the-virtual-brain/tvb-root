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
Provides an interface of sorts between TVB and OpenMEEG.

NOTE: Under Linux, using a custom ATLAS for your OpenMEEG install, as opposed
to a generic distribution package, can easily result in runtimes decreasing by
a factor of 3 to 5. The approximate runtimes mentioned below are for TVB's
default dataset running on a highend workstation, circa 2010 (custom ATLAS).

An example for EEG
::

    import tvb.datatypes.surfaces as surfaces_module
    import tvb.datatypes.sensors as sensors_module
    import tvb.datatypes.projections as projections_module

    brain_skull = surfaces_module.BrainSkull()
    skull_skin = surfaces_module.SkullSkin()
    skin_air = surfaces_module.SkinAir()
    sources = surfaces_module.Cortex()
    sensors = sensors_module.SensorsEEG()
    conductances = {'air': 0.0, 'skin': 1.0, 'skull': 0.01, 'brain': 1.0}

    proj_mat = ProjectionMatrix(brain_skull = brain_skull,
                                skull_skin = skull_skin,
                                skin_air = skin_air,
                                conductances = conductances,
                                sources = sources,
                                sensors = sensors)

    # NOTE: ~3 hours run time (45min head +  1h15 invert + 1h source matrices)
    proj_mat.configure()

    eeg_projection_matrix = proj_mat()


.. moduleauthor:: Stuart A. Knock <Stuart@tvb.invalid>

"""

#TODO: Check all the methods actually work...
#TODO: Fix repeated filename append issue... Resolve naming and temp location
#TODO: Docstrings and comments, currently more for me than general consumption...
#TODO: Add checks, try-excepts, error-messages, etc...
#TODO: Consider extending to include internal potential sensors
#TODO: Benchmark thread count efficienccy for the OpenMP parts, such as 
#      calculation of source and head matrices. NOTE: inversion of the head 
#      matrix uses ATLAS under Linux (and MKL under MS), so considerations of 
#      performance for this step are left to a properly optimised ATLAS install.


#NOTE: All the compute time is in the source+head geometry, including 
#      conductances, so these should only be expected to be set once  at 
#      initialisation...However, support for changing sensors on a given 
#      ProjectionMatrix object probably makes sense

import os

#NOTE: Using built in saves, a single run with saved data will require ~10 GB of
#      free space,.. 
OM_SAVE_SUFFIX = ".bin"  #NOTE: ".mat" segfaults, ".txt" truncates precision. 
OM_STORAGE_DIR = os.path.expanduser(os.path.join("~", "TVB", "openmeeg"))
if not os.path.isdir(OM_STORAGE_DIR):
    os.mkdir(OM_STORAGE_DIR)

import numpy

from tvb.simulator.common import get_logger
LOG = get_logger(__name__)

#From externals.openmeeg.build.src.Wrapping
import openmeeg as om # allez oohh-eemmeuh, allez oohh-eemmeuh...


#The Virtual Brain

import tvb.datatypes.surfaces as surfaces_module
import tvb.datatypes.sensors as sensors_module
import tvb.datatypes.connectivity as connectivity_module

import tvb.basic.traits.types_basic as basic
import tvb.basic.traits.core as core

#Unique compound name/id where id would be  either TVB GID or filename, val is 
#the numerical vaue of those parameters
#bs_id_ss_id_sa_id_src_id_ca_val_csn_val_csl_val_cb_val_sen_id

#THe long  calculations depend on the geometry of the Head and sources but are 
# independent of the sensors...???
class ProjectionMatrix(core.Type):
    """
    Provides the mechanisms necessary to access OpenMEEG for the calculation of
    EEG and MEG projection matrices, ie matrices that map source activity to 
    sensor activity. It is initialised with datatypes of TVB and ultimately 
    returns the projection matrix as a Numpy ndarray. 
    """

    brain_skull = surfaces_module.BrainSkull(
        label = "Boundary between skull and skin domains",
        default = None,
        required = True,
        doc = """A ... surface on which ... including ...""")

    skull_skin = surfaces_module.SkullSkin(
        label = "surface and auxillary for surface sim",
        default = None,
        required = True,
        doc = """A ... surface on which ... including ...""")

    skin_air = surfaces_module.SkinAir(
        label = "surface and auxillary for surface sim",
        default = None,
        required = True,
        doc = """A ... surface on which ... including ...""")

    conductances = basic.Dict(
        label = "Domain conductances",
        default = {'air': 0.0, 'skin': 1.0, 'skull': 0.01, 'brain': 1.0},
        required = True,
        doc = """A dictionary representing the conductances of ...""")

    sources = surfaces_module.Cortex(
        label = "surface and auxillary for surface sim",
        default = None,
        required = True,
        doc = """A cortical surface on which ... including ...""")

    sensors = sensors_module.Sensors(
        label = "surface and auxillary for surface sim",
        default = None,
        required = False,
        doc = """A cortical surface on which ... including ...  
            If left as None then EEG is assumed and skin_air is expected to 
            already has sensors associated""")


    def __init__(self, **kwargs):
        """
        Initialse traited attributes and attributes that will hold OpenMEEG 
        objects.
        """
        super(ProjectionMatrix, self).__init__(**kwargs) 
        LOG.debug(str(kwargs))

        #OpenMEEG attributes
        self.om_head = None
        self.om_sources = None
        self.om_sensors = None
        self.om_head2sensor = None

        self.om_inverse_head = None
        self.om_source_matrix = None
        self.om_source2sensor = None #For MEG, not used for EEG


    def configure(self):
        """
        Converts TVB objects into a for accessible to OpenMEEG, then uses the 
        OpenMEEG library to calculate the intermediate matrices needed in 
        obtaining the final projection matrix.
        """
        super(ProjectionMatrix, self).configure()
        if self.sensors is None:
            self.sensors = self.skin_air.sensors

        if isinstance(self.sensors, sensors_module.SensorsEEG):
            self.skin_air.sensors = self.sensors
            self.skin_air.sensor_locations = self.sensors.sensors_to_surface(self.skin_air)

        # Create OpenMEEG objects from TVB objects.
        self.om_head = self.create_om_head()
        self.om_sources = self.create_om_sources()
        self.om_sensors = self.create_om_sensors()

        # Calculate based on type of sources
        if isinstance(self.sources, surfaces_module.Cortex):
            self.om_source_matrix = self.surface_source() #NOTE: ~1 hr
        elif isinstance(self.sources, connectivity_module.Connectivity):
            self.om_source_matrix = self.dipole_source()

        # Calculate based on type of sensors
        if isinstance(self.sensors, sensors_module.SensorsEEG):
            self.om_head2sensor = self.head2eeg()
        elif isinstance(self.sensors, sensors_module.SensorsMEG):
            self.om_head2sensor = self.head2meg()
            if isinstance(self.sources, surfaces_module.Cortex):
                self.om_source2sensor = self.surf2meg()
            elif isinstance(self.sources, connectivity_module.Connectivity):
                self.om_source2sensor = self.dip2meg()

        #NOTE: ~1 hr
        self.om_inverse_head = self.inverse_head(inv_head_mat_file = "hminv_uid")


    def __call__(self):
        """
        Having configured the ProjectionMatrix instance, that is having run the 
        configure() method or otherwise provided the intermedite OpenMEEG (om_*)
        attributes, the oblect can be called as a function -- returning a 
        projection matrix as a Numpy array.
        """
        #Check source type and sensor type, then call appripriate methods to 
        #generate intermediate data, cascading all the way back to geometry 
        #calculation if it wasn't already done.
        #Then return a projection matrix...

        # NOTE: returned projection_matrix is a numpy.ndarray
        if isinstance(self.sensors, sensors_module.SensorsEEG):
            projection_matrix = self.eeg_gain()
        elif isinstance(self.sensors, sensors_module.SensorsMEG):
            projection_matrix = self.meg_gain()

        return projection_matrix


    ##------------------------------------------------------------------------##
    ##--------------- Methods for creating openmeeg objects ------------------##
    ##------------------------------------------------------------------------##

    def create_om_head(self): #TODO: Prob. need to make file names specifiable
        """
        Generates 5 files::

            skull_skin.tri
            skin_air.tri
            brain_skull.tri
            head_model.geom
            head_model.cond

        Containing the specification of a head in a form that can be read by 
        OpenMEEG, then creates and returns an OpenMEEG Geometry object containing 
        this information.
        """
        surface_files = []
        surface_files.append(self._tvb_surface_to_tri("skull_skin.tri"))
        surface_files.append(self._tvb_surface_to_tri("brain_skull.tri"))
        surface_files.append(self._tvb_surface_to_tri("skin_air.tri"))

        geometry_file = self._write_head_geometry(surface_files,
                                                  "head_model.geom")
        conductances_file = self._write_conductances("head_model.cond")

        LOG.info("Creating OpenMEEG Geometry object for the head...")
        om_head = om.Geometry()
        om_head.read(geometry_file, conductances_file)
        #om_head.selfCheck() #Didn't catch bad order...
        LOG.info("OpenMEEG Geometry object for the head successfully created.")
        return om_head


    def create_om_sources(self): #TODO: Prob. should make file names specifiable
        """
        Take a TVB Connectivity or Cortex object and return an OpenMEEG object
        that specifies sources, a Matrix object for region level sources or a
        Mesh object for a cortical surface source.
        """
        if isinstance(self.sources, connectivity_module.Connectivity):
            sources_file = self._tvb_connectivity_to_txt("sources.txt")
            om_sources = om.Matrix()
        elif isinstance(self.sources, surfaces_module.Cortex):
            sources_file = self._tvb_surface_to_tri("sources.tri")
            om_sources = om.Mesh()
        else:
            LOG.error("sources must be either a Connectivity or Cortex.")

        om_sources.load(sources_file)
        return om_sources


    def create_om_sensors(self, file_name=None):
        """
        Take a TVB Sensors object and return an OpenMEEG Sensors object.
        """
        if isinstance(self.sensors, sensors_module.SensorsEEG):
            file_name = file_name or "eeg_sensors.txt"
            sensors_file = self._tvb_eeg_sensors_to_txt(file_name)
        elif isinstance(self.sensors, sensors_module.SensorsMEG):
            file_name = file_name or "meg_sensors.squid"
            sensors_file = self._tvb_meg_sensors_to_squid(file_name)
        else:
            LOG.error("sensors should be either SensorsEEG or SensorsMEG")

        LOG.info("Wrote sensors to temporary file: %s" % str(file_name))

        om_sensors = om.Sensors()
        om_sensors.load(sensors_file)
        return om_sensors



    ##------------------------------------------------------------------------##
    ##--------- Methods for calling openmeeg methods, with logging. ----------##
    ##------------------------------------------------------------------------##

    def surf2meg(self):
        """
        Create a matrix that can be used to map an OpenMEEG surface source to an 
        OpenMEEG MEG Sensors object.

        NOTE: This source to sensor mapping is not required for EEG.

        """
        LOG.info("Computing DipSource2MEGMat...")
        surf2meg_mat = om.SurfSource2MEGMat(self.om_sources, self.om_sensors)
        LOG.info("surf2meg: %d x %d" % (surf2meg_mat.nlin(),
                                        surf2meg_mat.ncol()))
        return surf2meg_mat


    def dip2meg(self):
        """
        Create an OpenMEEG Matrix that can be used to map OpenMEEG dipole sources 
        to an OpenMEEG MEG Sensors object.

        NOTE: This source to sensor mapping is not required for EEG.

        """
        LOG.info("Computing DipSource2MEGMat...")
        dip2meg_mat = om.DipSource2MEGMat(self.om_sources, self.om_sensors)
        LOG.info("dip2meg: %d x %d" % (dip2meg_mat.nlin(), dip2meg_mat.ncol()))
        return dip2meg_mat


    def head2eeg(self):
        """
        Call OpenMEEG's Head2EEGMat method to calculate the head to EEG sensor
        matrix.
        """     
        LOG.info("Computing Head2EEGMat...")
        h2s_mat = om.Head2EEGMat(self.om_head, self.om_sensors)
        LOG.info("head2eeg: %d x %d" % (h2s_mat.nlin(), h2s_mat.ncol()))
        return h2s_mat


    def head2meg(self):
        """
        Call OpenMEEG's Head2MEGMat method to calculate the head to MEG sensor
        matrix.
        """     
        LOG.info("Computing Head2MEGMat...")
        h2s_mat = om.Head2MEGMat(self.om_head, self.om_sensors)
        LOG.info("head2meg: %d x %d" % (h2s_mat.nlin(), h2s_mat.ncol()))
        return h2s_mat


    def surface_source(self, gauss_order = 3, surf_source_file=None):
        """ 
        Call OpenMEEG's SurfSourceMat method to calculate a surface source 
        matrix. Optionaly saving the matrix for later use.
        """
        LOG.info("Computing SurfSourceMat...")
        ssm = om.SurfSourceMat(self.om_head, self.om_sources, gauss_order)
        LOG.info("surface_source_mat: %d x %d" % (ssm.nlin(), ssm.ncol()))
        if surf_source_file is not None:
            LOG.info("Saving surface_source matrix as %s..." % surf_source_file)
            ssm.save(os.path.join(OM_STORAGE_DIR,
                                  surf_source_file + OM_SAVE_SUFFIX)) #~3GB
        return ssm


    def dipole_source(self, gauss_order = 3, use_adaptive_integration = True,
                      dip_source_file=None):
        """ 
        Call OpenMEEG's DipSourceMat method to calculate a dipole source matrix.
        Optionaly saving the matrix for later use.
        """
        LOG.info("Computing DipSourceMat...")
        dsm   = om.DipSourceMat(self.om_head, self.om_sources, gauss_order, 
                                use_adaptive_integration)
        LOG.info("dipole_source_mat: %d x %d" % (dsm.nlin(), dsm.ncol()))
        if dip_source_file is not None:
            LOG.info("Saving dipole_source matrix as %s..." % dip_source_file)
            dsm.save(os.path.join(OM_STORAGE_DIR,
                                  dip_source_file + OM_SAVE_SUFFIX))
        return dsm


    def inverse_head(self, gauss_order = 3, inv_head_mat_file = None):
        """
        Call OpenMEEG's HeadMat method to calculate a head matrix. The inverse 
        method of the head matrix is subsequently called to invert the matrix.
        Optionaly saving the inverted matrix for later use.

        Runtime ~8 hours, mostly in martix inverse as I just use a stock ATLAS 
        install which doesn't appear to be multithreaded (custom building ATLAS
        should sort this)... Under Windows it should use MKL, not sure for Mac

        For reg13+potato surfaces, saved file size: hminv ~ 5GB, ssm ~ 3GB.
        """

        LOG.info("Computing HeadMat...")
        head_matrix = om.HeadMat(self.om_head, gauss_order)
        LOG.info("head_matrix: %d x %d" % (head_matrix.nlin(), 
                                           head_matrix.ncol()))

        LOG.info("Inverting HeadMat...")
        hminv = head_matrix.inverse()
        LOG.info("inverse head_matrix: %d x %d" % (hminv.nlin(), hminv.ncol()))

        if inv_head_mat_file is not None:
            LOG.info("Saving inverse_head matrix as %s..." % inv_head_mat_file)
            hminv.save(os.path.join(OM_STORAGE_DIR,
                                    inv_head_mat_file + OM_SAVE_SUFFIX)) #~5GB
        return hminv


    def eeg_gain(self, eeg_file=None):
        """
        Call OpenMEEG's GainEEG method to calculate the final projection matrix.
        Optionaly saving the matrix for later use. The OpenMEEG matrix is 
        converted to a Numpy array before return. 
        """
        LOG.info("Computing GainEEG...")
        eeg_gain = om.GainEEG(self.om_inverse_head, self.om_source_matrix,
                              self.om_head2sensor)
        LOG.info("eeg_gain: %d x %d" % (eeg_gain.nlin(), eeg_gain.ncol()))
        if eeg_file is not None:
            LOG.info("Saving eeg_gain as %s..." % eeg_file)
            eeg_gain.save(os.path.join(OM_STORAGE_DIR,
                                       eeg_file + OM_SAVE_SUFFIX))
        return om.asarray(eeg_gain)


    def meg_gain(self, meg_file=None):
        """
        Call OpenMEEG's GainMEG method to calculate the final projection matrix.
        Optionaly saving the matrix for later use. The OpenMEEG matrix is 
        converted to a Numpy array before return. 
        """
        LOG.info("Computing GainMEG...")
        meg_gain = om.GainMEG(self.om_inverse_head, self.om_source_matrix,
                              self.om_head2sensor, self.om_source2sensor)
        LOG.info("meg_gain: %d x %d" % (meg_gain.nlin(), meg_gain.ncol()))
        if meg_file is not None:
            LOG.info("Saving meg_gain as %s..." % meg_file)
            meg_gain.save(os.path.join(OM_STORAGE_DIR,
                                       meg_file + OM_SAVE_SUFFIX))
        return om.asarray(meg_gain)



    ##------------------------------------------------------------------------##
    ##------- Methods for writting temporary files loaded by openmeeg --------##
    ##------------------------------------------------------------------------##

    def _tvb_meg_sensors_to_squid(self, sensors_file_name):
        """
        Write a tvb meg_sensor datatype to a .squid file, so that OpenMEEG can
        read it and compute the projection matrix for MEG...
        """
        sensors_file_path = os.path.join(OM_STORAGE_DIR, sensors_file_name)
        meg_sensors = numpy.hstack((self.sensors.locations, 
                                    self.sensors.orientations))
        numpy.savetxt(sensors_file_path, meg_sensors)
        return sensors_file_path


    def _tvb_connectivity_to_txt(self, dipoles_file_name):
        """
        Write position and orientation information from a TVB connectivity object 
        to a text file that can be read as source dipoles by OpenMEEG. 

        NOTE: Region level simulations lack sufficient detail of source orientation, 
            etc, to provide anything but superficial relevance. It's probably better
            to do a mapping of region level simulations to a surface and then
            perform the EEG projection from the mapped data...

        """
        NotImplementedError


    def _tvb_surface_to_tri(self, surface_file_name):
        """
        Write a tvb surface datatype to .tri format, so that OpenMEEG can read
        it and compute projection matrices for EEG/MEG/...
        """
        surface_file_path = os.path.join(OM_STORAGE_DIR, surface_file_name)

        #TODO: check file doesn't already exist
        LOG.info("Writing TVB surface to .tri file: %s" % surface_file_path)
        file_handle = file(surface_file_path, "a")

        file_handle.write("- %d \n" % self.sources.number_of_vertices)
        verts_norms = numpy.hstack((self.sources.vertices, 
                                    self.sources.vertex_normals))
        numpy.savetxt(file_handle, verts_norms)

        tri_str = "- " + (3 * (str(self.sources.number_of_triangles) + " ")) + "\n"
        file_handle.write(tri_str)
        numpy.savetxt(file_handle, self.sources.triangles, fmt="%d")

        file_handle.close()
        LOG.info("%s written successfully." % surface_file_name)

        return surface_file_path


    def _tvb_eeg_sensors_to_txt(self, sensors_file_name):
        """
        Write a tvb eeg_sensor datatype (after mapping to the head surface to be 
        used) to a .txt file, so that OpenMEEG can read it and compute 
        leadfield/projection/forward_solution matrices for EEG...
        """
        sensors_file_path = os.path.join(OM_STORAGE_DIR, sensors_file_name)
        LOG.info("Writing TVB sensors to .txt file: %s" % sensors_file_path)
        numpy.savetxt(sensors_file_path, self.skin_air.sensor_locations)
        LOG.info("%s written successfully." % sensors_file_name)
        return sensors_file_path


    #TODO: enable specifying ?or determining? domain surface relationships... 
    def _write_head_geometry(self, boundary_file_names, geom_file_name):
        """
        Write a geometry file that is read in by OpenMEEG, this file specifies
        the files containng the boundary surfaces and there relationship to the
        domains that comprise the head.

        NOTE: Currently the list of files is expected to be in a specific order, 
        namely::

            skull_skin
            brain_skull
            skin_air

        which is reflected in the static setting of domains. Should be generalised.
        """

        geom_file_path = os.path.join(OM_STORAGE_DIR, geom_file_name)

        #TODO: Check that the file doesn't already exist.
        LOG.info("Writing head geometry file: %s" % geom_file_path)
        file_handle = file(geom_file_path, "a")

        file_handle.write("# Domain Description 1.0\n\n")
        file_handle.write("Interfaces %d Mesh\n\n" % len(boundary_file_names))

        for file_name in boundary_file_names:
            file_handle.write("%s\n" % file_name)

        file_handle.write("\nDomains %d\n\n" % (len(boundary_file_names) + 1))
        file_handle.write("Domain Scalp %s %s\n" % (1, -3))
        file_handle.write("Domain Brain %s %s\n" % ("-2", "shared"))
        file_handle.write("Domain Air %s\n" % 3)
        file_handle.write("Domain Skull %s %s\n" % (2, -1))

        file_handle.close()
        LOG.info("%s written successfully." % geom_file_path)

        return geom_file_path


    def _write_conductances(self, cond_file_name):
        """
        Write a conductance file that is read in by OpenMEEG, this file 
        specifies the conductance of each of the domains making up the head.

        NOTE: Vaules are restricted to have 2 decimal places, ie #.##, setting 
            values of the form 0.00# will result in 0.01 or 0.00, for numbers 
            greater or less than ~0.00499999999999999967, respecitvely...

        """
        cond_file_path = os.path.join(OM_STORAGE_DIR, cond_file_name)

        #TODO: Check that the file doesn't already exist.
        LOG.info("Writing head conductance file: %s" % cond_file_path)
        file_handle = file(cond_file_path, "a")

        file_handle.write("# Properties Description 1.0 (Conductivities)\n\n")
        file_handle.write("Air         %4.2f\n" % self.conductances["air"])
        file_handle.write("Scalp       %4.2f\n" % self.conductances["skin"])
        file_handle.write("Brain       %4.2f\n" % self.conductances["brain"])
        file_handle.write("Skull       %4.2f\n" % self.conductances["skull"])

        file_handle.close()
        LOG.info("%s written successfully." % cond_file_path)

        return cond_file_path


    #TODO: Either make these utility functions or have them load directly into
    #      the appropriate attribute...
    ##------------------------------------------------------------------------##
    ##---- Methods for loading precomputed matrices into openmeeg objects ----##
    ##------------------------------------------------------------------------##

    def _load_om_inverse_head_mat(self, file_name):
        """
        Load a previously stored inverse head matrix into an OpenMEEG SymMatrix
        object.
        """
        inverse_head_martix = om.SymMatrix()
        inverse_head_martix.load(file_name)
        return inverse_head_martix


    def _load_om_source_mat(self, file_name):
        """
        Load a previously stored source matrix into an OpenMEEG Matrix object.
        """
        source_matrix = om.Matrix()
        source_matrix.load(file_name)
        return source_matrix


