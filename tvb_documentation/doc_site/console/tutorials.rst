.. _console_tutorials:



******************************************
Tutorials
******************************************


Below are a set of links, with brief descriptions, to static html versions of
tutorials for the scripting interface to TVB. The first
set of "basic" tutorials are listed roughly in the order they should be read,
and cover the basic functionality of TVB's simulator package using very simple
examples. The second set target specific, more realistic, simulations which
either reproduce published modelling work or attempt to create a simulated
equivalent of a published experiment, modelling the experimental paradigm
and performing similar analysis on the simulated EEG/MEG/fMRI to that performed
on the experimental data.

Tutorials are written as IPython Notebooks (.ipynb), the links below simply
pass the github version of these .ipynb files to the
[nbviewer](http://nbviewer.ipython.org) site.

If you for or clone TVB's scientific library, these tutorials can be run
interactively. Assuming you have a version of IPython that supports notebooks
installed, the IPython notebook interface can be launched by running the
following command:

    ipython notebook --pylab

which will result in separate figure windows unless the notebook includes
specific inlining commands, or with:

    ipython notebook --pylab inline

which will automatically inline figures.

NOTE:
    When inlined the figures are static images in the notebook, so interactive
    or movie type plots aren't fully functional in this mode. However, using
    the inline form allows for static .html pages to be "printed" that include
    the figures.


Basic
=============================

* `All Tutorials <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/>`_

* `Anatomy Of A Region Simulation <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Anatomy_Of_A_Region_Simulation/Tutorial_Anatomy_Of_A_Region_Simulation.ipynb>`_
    This tutorial covers the basic overview of components required for setting
    up and running a region level simulation.

* `Anatomy Of A Surface Simulation <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Anatomy_Of_A_Surface_Simulation/Tutorial_Anatomy_Of_A_Surface_Simulation.ipynb>`_
    This tutorial covers the basic overview of components required for setting
    up and running a surface level simulation.

* `Exploring A Model <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Exploring_A_Model/Tutorial_Exploring_A_Model.ipynb>`_
    This tutorial covers using the "phase_plane_interactive" plotting tool to
    investigate the dynamic properties of a Model and at the same time set the
    parameters of a specific instance for use in a simulation.

* `Exploring The Bold Monitor <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Exploring_The_Bold_Monitor/Tutorial_Exploring_The_Bold_Monitor.ipynb>`_
    This tutotrial covers the basic overview of the Bold monitor to investigate
    its parameters and the influence on the HRF (Haemodynamic Response Function)

* `Looking At Longer TimeSeries <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Looking_At_Longer_TimeSeries/Tutorial_Looking_At_Longer_TimeSeries.ipynb>`_
    This tutorial covers the use of two interactive visualisers that become
    useful when looking at time-series produced by longer simulation runs, ie.,
    when a simple static plot of the time-series doesn't quite cut it.

* `Region Stimuli <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Region_Stimuli/Tutorial_Region_Stimuli.ipynb>`_
    This tutorial covers the basics of defining a Stimulus at the region level
    and applying it to a simulation.

* `Surface Stimuli <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Surface_Stimuli/Tutorial_Surface_Stimuli.ipynb>`_
    This tutorial covers the basics of defining a Stimulus at the surface level
    and applying it to a surface simulation.

* `Smooth Parameter Variations <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Smooth_Parameter_Variation/Tutorial_Smooth_Parameter_Variation.ipynb>`_
    This tutorial covers he application of predefined (ie, not state dependent)
    variations of simulation parameters as a function of time.

* `Stochastic Simulations <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Stochastic_Simulation/Tutorial_Stochastic_Simulation.ipynb>`_
    This tutorial covers the basics of running stochastic or Noise driven
    simulations, this mostly revolves around defining a noise process and
    selecting a noise function for a stochastic integration method.

* `Getting To Know Your Mesh Surface <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Getting_To_Know_Your_Mesh_Surface/Tutorial_Getting_To_Know_Your_Surface_Mesh.ipynb>`_
    This tutorial covers some of the methods available for discovering the
    properties of a mesh surface, which are used in TVB to represent the folded
    cortical surface as well as the boundaries between skin, skull, and brain
    used in the computation of forward solutions to EEG and MEG.

* `Using Your Own Connectivity <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/tree/trunk/tvb/simulator/doc/tutorials/Tutorial_Using_Your_Own_Connectivity/Tutorial_Using_Your_Own_Connectivity.ipynb>`_
    This tutorial covers the basics of importing non-default connectivity data
    and ways of manipulating and inspecting it.

-------------------------------------------------------------------------------


Reproducibility
=============================

* `Reproducing Sanz Leon et al. 2013 -- Evoked Responses <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Evoked_Responses_In_The_Visual_Cortex/Tutorial_Evoked_Responses_In_The_Visual_Cortex.ipynb>`_
    This tutorial reproduces the Evoked Responses in the Visual Cortex example
    presented in [Sanz Leon *etal* 2013][sl2013a]:

        Sanz Leon, P.; Woodman, M; Knock, S.; Domide, L.; Mersmann, J.; McIntosh, A. and Jirsa, V. (2013).
        The Virtual Brain: a simulator of primate brain dynamics. Frontiers in Neuroinformatics.

[sl2013a]: http://www.frontiersin.org/neuroinformatics/10.3389/fninf.2013.00010/full


-------------------------------------------------------------------------------

Use Cases
=============================

* `Evoked Responses <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Evoked_Responses_In_The_Visual_Cortex/Tutorial_Evoked_Responses_In_The_Visual_Cortex.ipynb>`_
    This tutorial explains how to model evoked reponses when stimulating the primary visual cortex (V1)
    with a rectangular pulse train. Two simulations are launched to produce resting state and
    stimulation-driven activity.

* `Modeling The Impact Of Structural Lesions`:

    * `Part I: Modeling Lesions <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Modeling_The_Impact_Of_Structural_Lesions/Tutorial_Modeling_The_Impact_Of_Structural_Lesions_Part_I.ipynb>`_

    * `Part II: The Brain Network Model <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Modeling_The_Impact_Of_Structural_Lesions/Tutorial_Modeling_The_Impact_Of_Structural_Lesions_Part_II.ipynb>`_

    * `Part III: Offline Analysis <http://nbviewer.ipython.org/github/the-virtual-brain/tvb-library/blob/trunk/tvb/simulator/doc/tutorials/Tutorial_Modeling_The_Impact_Of_Structural_Lesions/Tutorial_Modeling_The_Impact_Of_Structural_Lesions_Part_III.ipynb>`_

    In this tutorial we aimed to:
        + reproduce the results presented in [(Alstott et al. 2009)][alstott2009] using TVB aiming to
          define a set of standardized lesion strategies for simulation studies of
          Brain Network Models (BNMs);

        + extend the original work by studying the role of conduction speed (i.e.,
          time delays) on the healthy and lesioned networks;

        + systematically explore the impact of structural changes in the dynamics.
          In TVB architecture, structural connectivity is one parameter of a BNM; and

        + to create a reproducible project, make it publicly available to improve
          the experience of the reviewers and readers. A folder with the `data <https://www.dropbox.com/sh/44e8k1t8hpb1r9z/KO5YRW7_Pg>`_ is included .

[alstott2009]: http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1000408




