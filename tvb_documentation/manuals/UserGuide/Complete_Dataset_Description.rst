A Description of a Complete Dataset
===================================

The primary purpose of The Virtual Brain platform is to simulate whole brain
dynamics. A simulation pipeline has different stages. The most fundamental stage
is building a realization of a computational model which we call a large-scale
brain network model. This model is constituted by a set of structural and
functional components linked together, completely creating a particular model of
the brain.

The following document is a generic description of what we often call a "minimal
structural dataset for TVB" and "a complete dataset". The aim of this document
is to specify the different pieces of data required to derive the structural
skeleton/substrate of a BNM. Hopefully, experts in the field of data acquisition
will help us completing and improving our description.

A complete dataset should:

+ provide the history of the acquisition/processing protocols (traceability);
+ be in a standard format to perform analysis with different toolboxes;
+ have the *Minimal Structural Dataset* to derive a self-consistent
  set of files used in TheVirtualBrain. This minla dataset will permit
  users to build large-scale brain network models and save their
  simulated data under different output modalities (eg, EEG, MEG, fMRI).
+ increase reproducibility of the results.


General Considerations
----------------------

A dataset should be made available on one single place (eg, through XNAT) to
ensure traceability and if required for clinical/privacy reasons, restricted
access.

.. Note::
    the following definitions are not definitive, if you come across with a better
    categorization, please let us know.

**Raw Structural Dataset**

  a collection of files describing a subject's head (eg, structural MRI AND DTI Data)

**Structural Dataset**

  (a collection of T1/T2 weighted MRIscans + DTI data + parcellation )

**Minimal (Preprocessed) Structural Dataset**

  (surface mesh, parcellation, region centres, region mapping, connectivity matrix)

**Complete Structural Dataset**

  in addition to the structures mentioned above, head model surfaces and
  information about the units (areas, lengths, connectivity strengths). Info
  about the EEG/MEG/iEEG sensors if users wish to compare simulated data to
  empirical data. *Complete Dataset*   all of the above + functional data (eeg,
  rsfMRI, meg)


In general, all the steps in the processing pipeline should be documented, so
it's possible to apply the same treatment to subsequent datasets.  Sending
pieces of informations in different files without descriptions from where they
came and how they were processed is a bad practice and only detrimental for your
own research project (takes a lot of time and it's not reproducible). We can't
provide any meaningful help to integrate/check or validate incomplete datasets.


Ideally, any volumetric data (eg, in NIFTI format), surface data (eg, GIFTI
format) or combination thereof (eg, CIFTI format) should be provided in their
RAW format, and if any pre-processing was performed on the raw data,  associated
data such as the region centres and parcellation mask should be provided in the
same coordinate system as the cortical mesh (ie, self-consistent dataset). The
meaning of the (x,y,z) coordinates depends entirely on how the volumetric file
was generated. It is possible to set any coordinate system you want ("native",
"mni", "talaraich") depending on the processing you apply to your data.  A
region centre, for example, would be a single spatial location in 3D. This
location is specified by three numbers (x,y,z), these numbers should ideally
represent mm and must be relative to an origin (x=0, y=0, z=0). The "same
coordinate system" means that the origin is in the same location relative to the
head, and that the axis(x,y,z) point in the same direction with the same
orientation.


Parcellation Mask
-----------------

**What is the purpose of the mask?**

  A brain mask basically covers the standard brain. The mask needs to
  partition/parcellate the volume of space containing the head/brain into regions.
  It can be used to partition the cortical mesh into regions, consistently with
  the derived parcellated connectivity matrices. So, for each row in a
  connectivity matrix the mask needs to specify a region of space (region of
  interest). The  mask can then be used to map the thousands of vertices which
  make up the  surface mesh to the hundred or so regions in the connectivity
  matrix,  that is, each region/row- in-a-connectivity-matrix is associated with
  hundreds of vertices in the cortical mesh. So the mask provides a way for us to
  generate a one to many mapping (region mapping) -- from a row in the
  connectivity matrix to the many vertices of the surface mesh which lie  in that
  region. This anatomical parcellation is also used to obtain finer parcellations
  and further divide the cortical surface into small regions of interest (REF to
  Zalesky) and distinguish subcortical structures.


**What should be the format?**

  NIFTI is a standard format for volumetric time-series, and it is widely used
  in the neuroimaging commmunity. Originally, NIFTI-1 file format was based on
  16-bit signed integers and was limited to 32,767 in each dimension. The
  NIFTI-2 format is based on 64-bit integers and can handle very large volumes
  and matrices. The more recent CIFTI format is compatible with the NIFTI-2
  format and it also has the extension .nii. We encourage to use the NIFTI-1
  format (.nii and .nii.gz).I

  .. Note::
    add references to the libraries and softwares that are available for
    NIFTI-1 TVB also has a reader. Not the same case for NIFTI-2 and CIFTI.
    FieldTrip is the only one providing CIFTI i/o functionalities.

In the case of a parcellation mask, each voxel contains  an integer
corresponding to a specific region (numeric labeling). This means, assuming
voxels of 2x2x2 mm, the mask would consist of roughly 100x100x100 (ie, 1
million) voxels. The range of the integer number in each of these voxels should
correspond to the number of regions  (or rows ) in the parcellated connectivity
matrix.  Some voxels may contain no number or say -1, to specify that that piece
of space doesn't belong to a region, for example if it lies outside of  the
head. These type of conventions should be specified and documented.
A list with the region names/labels and corresponding integer index should be provided.

**How should region labels and names look like provided with the mask?**

  A region label is a short unique identifier, a region or area name usually
  refers to a human readable description. Examples for one region/name would
  be something like 'label: RM-TCpol_R' / 'name: right temporal polar cortex'.
  Ideally, a reference to the original atlas/template should be provided as
  well. Notice that the correspondance between integrers values in the
  parcellation mask and anatomical/human readable labels should be provided if
  they are not specified in the volume file.

**Are region labels essential?**

  From the point of view of the  implementation of The Virtual Brain the labels
  are essential?

**Are region names essential?**

  The region names on the other hand are  primarily a matter of usability,
  though a very valuable one, when you want to identify an area that you wish to
  modify in a  simulation (eg, modeling lesions). Unless a user is an anatomist
  and acquainted with the labels, then the names are much clearer.


**Why is information on cortical vs. subcortical regions needed?**

  We need a means of distinguishing cortical from subcortical regions within
  the mask, so that when we apply the mask to a cortical mesh we don't
  inadvertently associate parts of the cortex with subcortical regions in  the
  connectivity matrix. Ultimately a vector of the length of the number  of
  regions is needed, specifying whether each region is part of the  cortex or
  not. If the labels or names clearly include this information,  that is they
  clearly state whether they are cortical regions or not,  then the vector could
  be generated on this basis.

**Is the parcellation mask unique?**

  No. Currently, there are several parcellation being used in the community.
  NOTE: REF parcellation papers. One of the main problems is that parcellations
  are often custom made and subsequently modified, so it becomes very difficult
  to track the origins.  To begin with, we suggest to use parcellation mask
  provided by neuroimaging software tools like FSL AAL 90. If you want to use a
  custom made parcellation, then it should have the characteristics mentioned
  above. Also, having the structural raw data it is possible to derive
  connectivity matrices from the same dataset, but at different resolutions.
  NOTE: (reference to Hagmann and Zalesky).

**What is the coordinate system of the parcellation mask?**

  It depends on how the parcellation mask was obtained. In principle, it should
  be registered to a standard space such as MNI. This coordinate systems should
  be consistent with the surface's coordinate systems.


Connectivity and path length data
---------------------------------

**What is it required to build a connectivity matrix (parcellated connectome)?**

  Diffusion data, a parcellation mask and probably the white matter surface (in
  the same space, aligned). In TVB, we are not providing the tractography tools to
  create structural connectivity matrices.

**Are the tract lengths essential for using TVB?**

  Yes. The simulations in TVB take into account time delays, and their magnitude is given
  by the distance between pairs of regions scaled by the conduction speed.

**Are the region centres important?**

  Yes! If for a reason unbeknown to you, you happen to not have the white matter
  fibre lengths, then TVB uses the region centres to compute a tract lengths
  matrix based on the Euclidean distance between region pairs. The region centres
  are merely a list of Cartesian triplets (x,y,z) that specify the spatial
  location relative to the consistent coordinate system mentioned above. Each
  region centre is just a single point in space, corresponding to the centre of
  the region. The region itself might be spatially extended (if we have the
  cortical surface), and thus not a single point.

**What is the parcellated connectome?**

  This term was introduced by the HCP, and it refers to the connectivity matrix.
  For TVB a Connectivity refers to a set of two matrices (of size "anatomical
  regions x anatomical regions "), one with weights giving the strength of     the
  connections between anatomical regions and a second matrix with the     white
  matter fibre lengths between regions;


Cortical Mesh
-------------

We encourage to use the MNI brain template (eg, MNI152) to register your
subjects data and extract the corresponding cortical surface.

**Is the cortical surface essential?**

  Yes! Strictly speaking, TVB can perform simulations using only a parcellated
  connectome as spatial support. From a scientific point of view MODELING THE
  ELECTRICAL ACTIVITY ON THE FOLDED CORTICAL SURFACE is one of the most
  interesting capabilities to exploit in TVB.  Modeling work where different
  output modalities (like EEG and BOLD) are compared need a certain level of
  geometrical detail that is not provided by a coarse-grained connectome. While
  in the field of macroconnectomics, the parcellated connectome is sufficient
  (debatable subject, see the paper by Zalesky), the cortical surface is
  necessary to work with neural field modeling and to account for spatial
  inhomogeneities.

  The cortical surface, represents the outer surface of the gray matter. It's
  often called 'pial surface'.

**How is a surface represented?**

  A way of representing 2D meshes embedded in 3D space is by storing two arrays,
  one for vertices, and one for triangles. Tha latter is an array with triplets
  of indices into the first array of vertices. So, basically a surface mesh is
  given by a set of vertices (triplets (x,y,z) defining the location of those
  vertices). And alternatively, the mesh can be represented by triagle arrays
  which are indices into the vertex arrays; three indices for each triangle.

  Then there are other 'attributes' that can be derived from these two main
  arrays, for instance 'normals'. A normal determine's the orientation of a vertex.

  All vertex-related/derived information is calculated and stored in separate
  arrays, although bound to the surface instance they were derived from. Read
  more about normals here: http://user.xmission.com/~nate/smooth.html

  .. Note::
    and the upcoming publication where surface regularization is explained for the case of the pial surface.


Region Mapping
--------------

**What is the Region Mapping?**

  The region mapping is just a relationship between the two pieces of data,
  mapping regions of a connectivity onto the nodes of a surface simulation, one
  to  many for the vertices of the cortical surface and one to one for the
  remaining  noncortical regions.  NOTE: A region mapping could be between two
  connectomes of different resolution (eg, the connectomes presented in Hagmann
  998 to 66 regions).

**How is the Region Mapping obtained?**

  Good question!

  TODO: Add links to relevant documentation.


Head model
----------

**What is the purpose of the head model**

  **Head**: the bucket that contains the brain. The head is often represented as
  a set of concentric spheres, in order to compute the electric field or
  potential on the skin surface (eg, as recorded with EEG electrodes). The
  concentric spheres (surfaces) represent the boundaries between the brain and
  the skull; the skull and the skin; and, the skin and the air mesh.

**What should be the format?**

  A surface format like GIFTI, or in the same format used for the cortical mesh.

**Is the head model essential?**

  From a scientific point of view, it is essential to compute the lead-field
  matrices which will  project the neural activity time-series into sensor space
  (eg EEG).  The boundary surfaces are then required to assist Open MEEG (or
  any other similar tool like FieldTrip) to generate good forward models for EEG/MEG)

  The surfaces describing a subject's head: skin, skull, cortical surface. See
  the description below.

**A Minimal Structural Dataset For TVB:**

  All 3D coordinates should be consistent, ie., vertices, parcellation mask, and
  region centres should be in the same units, axis orientations, alignment, etc.

**A minimally-complete connectivity data set for TVB**

  should include the following:

* Mesh surface for the cortex (regularised, continuous and complete per
  hemisphere, that is, there should be no holes in the surface and it should be
  possible to unambiguously define an inside and an outside, in other words,
  each hemisphere should be topologically spherical):

        + vertices (Cartesian (x,y,z))
        + triangles (triplets of indices into the vertices array, TRIANGLES, but not
                generalised polygons)

* Parcellation:
   + Spatial mask, 3D, PROPERLY ALIGNED WITH THE SURFACE, ie coordinates,
     orientation should be IN THE SAME SPACE.
   + Labels for all regions composing the parcellation/connectivity data.
   + A clear delineation, if not explicit in the labels, between cortical
     regions and subcortical structures.

* Region centres (Cartesian (x,y,z), consistent with surface, mask, etc), for
  all regions composing the parcellation/connectivity data.

* Connectivity (DSI):
   + Connection strength/s between regions.
   + Tract length between regions.


**Ideally**

  For a complete structural dataset, we should also have:

  * Connectivity: mainly Connection strength between regions.
        - This should include information specifying the directionality. That
          is, if the data is provided as a matrix rather than a file format
          including meta-data such as graphml, directionality should be clearly
          and unambiguously specified.

  * Mesh surfaces for:
        - inner-skull: boundary between the brain and the skull,
        - outer-skull: the boundary of between the skull and the skin
        - outer-skin:  boundary surface between the skin and the air
            (for EEG/MEG monitors)

  * Basic additional information:
        - Units: tract lengths, coordinates etc (mm).
        - Units: strength/weights units, (au) if none.
        - additional relevant information...


**Guidelines to import the data into TVB**

  Currently we have some guidelines describing what data fields and in which
  format users can import different components of a compelte dataset
  (connectome, surface, sensors, gain matrix for eeg, etc...).

  .. Note::
    Check the DataExchange chapter of the User Guide manual.



The TVB demonstration dataset
-----------------------------

**DISCLAIMER:** This dataset was custom made and built to serve the purpose of
numerically testing the simulator, as well as for theoretical exploration. It
does have, however, certain issues with regard to biophysical realism and so
shouldn't be used/relied-upon for that purpose. References, where appropiate,
are given. Also, this is an open source project and contributions are greatly
appreciated. If you see an error, please leave a comment or make corresponding
modifications [please give proper references and argument your corrections].

+ The parcellation was chosen to be as homologous as possible between Macaque
  and Human. (See the [scalable brain atlas interactive tool]
  (http://scalablebrainatlas.incf.org/main/coronal3d.php?template=PHT00&plugin=CoCoMac))

+ Weights are primarily CoCoMac -- exceptions are colossal connections. These
  are DSI fibre bundle widths scaled to fill the 0-3 of CoCoMac.

+ Most colossal connection are missing. Tract-lengths are actual DSI tracts
  where possible and Euclidean distance used where explicit DSI/DTI tract-
  lengths weren't available.

+ Region centres were generated to be consistent with the demo cortical
  surface.

+ In the current parcellated connectome all the non-cortical regions were
  stripped.

+ The CoCoMac connectivity belongs to a single hemisphere, so the
  weights matrix is symmetric (weighted undirected graph), but the DSI was
  "whole" brain and so there is probably hemispheric asymmetry in tract lengths
  and the cortical surface is hemispherically asymmetric so region centres aren't
  the same for both hemispheres. (this item is maybe deprecated...)


The default TVB connectivity is a bi-hemispheric hybrid CoCoMac/DSI matrix.
Subcortical regions (e.g. thalamus and other subcortical nuclei) are not
included in this matrix.

Anatomical labels and names:
    * A1: Primary auditory cortex
    * A2: Secondary auditory cortex
    * Amyg: Amygdala
    * CCa: Gyrus cinguli anterior
    * CCp: Gyrus cinguli posterior
    * CCr: Gyrus cinguli retrosplenialis
    * CCs: Gyrus cinguli subgenualis
    * FEF: Frontal eye field
    * G: Gustatory cortex
    * HC: Hippocampal cortex
    * IA: Anterior insula
    * IP: Posterior insula
    * M1: Primary motor area
    * PCi: Inferior parietal cortex
    * PCip: Cortex of the intraparietal sulcus
    * PCm: Medial parietal cortex (Precuneus)
    * PCs: Superior parietal cortex
    * PFCcl: Centrolateral prefrontal cortex
    * PFCdl: Dorsolateral prefrontal cortex
    * PFCdm: Dorsomedial prefrontal cortex
    * PFCm: Medial prefrontal cortex
    * PFCorb: Orbital prefrontal cortex
    * PFCpol: Pole of prefrontal cortex

And more:
    * PFCvl: Ventrolateral prefrontal cortex
    * PHC: Parahippocampal cortex
    * PMCdl: Dorsolateral premotor cortex
    * PMCm: Medial premotor cortex (supplementary motor cortex)
    * PMCvl: Ventrolateral premotor cortex
    * S1: Primary somatosensory cortex
    * S2: Secondary somatosensory cortex
    * TCc: Central temporal cortex
    * TCi: Inferior temporal cortex
    * TCpol: Pole of temporal cortex
    * TCs: superior temporal cortex
    * TCv: ventral temporal cortex
    * V1: Primary visual cortex
    * V2: Secondary visual cortex

We have:
 - An importer for RegionMapping (externally computed);

We need:
 - At least one, preferably multiple, complete datasets to serve as a default
   dataset available to users who can't or aren't interested in providing their
   own. Of specific importance here is the Connectivity Parcellation Mask, as
   well as a specification of hemisphere and cortical vs non-cortical regions.
   If you are intetrested in contributing a dataset, please contact paupau.

 - Algorithm for calculating the region mapping, given a coregistered Cortex
   and ParcellationMask, including an "island" removal/correction mechanism to
   deal with the imperfect alignment that will exist, even with coregistered
   data, between an individual's cortical surface and the "generic"
   parcellation mask.



Other datasets
--------------

Hagmann
.......

What has been provided/shown :

* A 998 ROIs connectome (weights + resampled distances)
* A mapping to the parcellated connectome of 66 regions.
* Label and anatomical names.
* Info about the coordinate system: Talaraich.

What's missing:

* The parcellation mask file.
* The cortical surface.
* The head model.

Permissions:

* On request to the authors.


The Human Connectome Project
............................


So far, the most complete datasets available.  We aim to integrate some of the
datasets provided by the HCP. Structural connectivity is the fundamental
substrate for building large-scale brain network models, and being able to use
these high quality, standardized and equally pre-processed data would be
ideal.

However, "advanced" HCP datasets will be hopefully released next year.  The HCP
data release does not include extensively processed connectivity data for
individual subjects, but mainly "an average dataset". In the current release, Q3,
there are dense ("grayordinate-to-grayordinate") functional connectivity
datasets based on resting state fMRI from individual subjects. However, HCP
people are still working on improving many of the steps for generating
structural connectivity datasets, based on diffusion imaging and probabilistic
tractography. In the future, they will release probabilistic tractography and
"dense structural connectome" datasets ( perhaps with the Q4 release, Q3 release
was made available on September 20th, 2013)

There are ongoing efforts both within and outside the HCP consortium to
generate improved methods of brain parcellation, especially cerebral cortex.
"HCP- sanctioned" parcellated connectome datasets (based on improved cortical
parcellations) will be made publicly available in the future (no target date
announced yet). Once these (plus the dense connectome datasets) are released,
users will be able to generate parcellated connectomes based on their own
preferred parcellation scheme.

They do plan to make a (FieldTrip-compatible) head model available for each
subject scanned using MEG.

What they have:

* Almost everything: raw, minimally processed and processed data.

What's missing:

* Preprocessed diffusion data (eg, fiber orientation, fiber tracts) and derived
  structural connectomes and individual based parcellations.

Permissions:

* available after agreeing with the privacy and sharing conditions. In principle,
  datasets can be distributed as long as we make users sign the terms required by
  the HCP. I would suggest, once the dense and some parcellated connectomes are
  available, to buy the connectome in a box and have a copy in a centralized
  storage server so TVB can read these data in.


Brain-mapping softwares:
    * FreeSurfer: http://surfer.nmr.mgh.harvard.edu/
    * FSL: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
    * CIVET: http://www.bic.mni.mcgill.ca/ServicesSoftware/CIVET
    * CARET: http://brainvis.wustl.edu/wiki/index.php/Caret:About
    * The Human Connectome Toolkit (CMK): http://cmtk.org/
    * NiPy: http://nipy.sourceforge.net/
    * MRtrix: http://www.brain.org.au/software/mrtrix/
    * CAmino: http://cmic.cs.ucl.ac.uk/camino/
    * BrainVisa: http://brainvisa.info/

MRI Processing/Analysis/Modeling platforms:
    * SPM: http://www.fil.ion.ucl.ac.uk/spm/
    * Fieldtrip: http://fieldtrip.fcdonders.nl/
    * Brainstorm: http://neuroimage.usc.edu/brainstorm/

Data exchange/db platforms:
    * The Human Connectome Project: http://www.humanconnectome.org/data/
    * XNAT: http://xnat.org/


**Glossary**

Space Coordinate systems:
    * MNI (we encourage to use this one)
    * Talaraich
    * ref: http://fieldtrip.fcdonders.nl/faq/how_are_the_different_head_and_mri_coordinate_systems_defined

Atlases:
    * In order to compare different brains, it is necessary to register them to a common space by using a template.
    * See http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases


Structural Anatomical Parcellations:
  * Kotter (macaque)
  * Broadmann
  * FSL AAL 90
  * Hagmann (based on Desikan)

