.. include:: /manuals/templates/pdf_constants.rst

GUI Demos
=========

The links in this section showcase TVB's web interface.

Simulation
----------

.. Sphinx limitations
      A :target: option on figure allows for links. But not for sphinx refs.
      So clicking the images does not follow the link.
      The images cannot be arranged in a flowing grid.
      To work around the grid issue we have added a css class and a rule for it in default.css
      To work around the target issue we have added direct html links. These are not robust and
      will be weird if this document will be rendered to pdf.


.. figure:: /manuals/UserGuide/screenshots/simulator_phase_plane_interactive.jpg
      :height: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Simulator.html#phase-plane

      :ref:`Get familiar with the behaviour of a model by exploring it's phase space. <phase_plane>`


.. figure:: /manuals/UserGuide/screenshots/simulator.jpg
      :height: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Simulator.html#simulator-ui

      :ref:`Launch a simulation. <simulator_ui>`

Data Management
---------------

.. figure:: /manuals/UserGuide/screenshots/data.jpg
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Project.html#tree-view-ui

      :ref:`View the data types in a project <tree_view_ui>`

.. figure:: /manuals/UserGuide/screenshots/default_operations.jpg
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Project.html#operations-ui

      :ref:`The operations that were executed in the project <operations_ui>`

Visualizers
-----------

.. figure:: /manuals/UserGuide/screenshots/visualizer_timeseries_svgd3.jpg
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Simulator-Visualizers.html#ts-svg-ui

      :ref:`Time series view <ts_svg_ui>`



.. figure:: /manuals/UserGuide/screenshots/visualizer_brain.jpg
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Simulator-Visualizers.html#brain-activity-view

      :ref:`3D brain activity view <brain_activity_view>`



.. figure:: /manuals/UserGuide/screenshots/visualizer_dual_head_eeg.jpg
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Simulator-Visualizers.html#brain-activity-view

      :ref:`brain_dual_view`



.. figure:: /manuals/UserGuide/screenshots/visualizer_tsv.jpg
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Simulator-Visualizers.html#brain-volumetric

      :ref:`brain_volumetric`



.. figure:: /manuals/UserGuide/screenshots/connectivity_editor.jpg
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Connectivity.html#connectivity-matrix-ui

      :ref:`connectivity_matrix_ui`

Scripting Tutorials
===================

These tutorials are written as IPython Notebooks and they use the scripting interface of TVB.
They can be run interactively if you have TVB’s scientific library and ipython installed.

The first set of "basic" tutorials are listed roughly in the order they should be read,
and cover the basic functionality of TVB's simulator package using very simple
examples.

.. _Anatomy Of A Region Simulation: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Anatomy_Of_A_Region_Simulation.ipynb
.. _Anatomy Of A Surface Simulation: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Anatomy_Of_A_Surface_Simulation.ipynb
.. _Exploring A Model: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Exploring_A_Model.ipynb
.. _Exploring A Model Reduced Wong Wang: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Exploring_A_Model_ReducedWongWang.ipynb
.. _Exploring The Bold Monitor: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Exploring_The_Bold_Monitor.ipynb
.. _Looking At Longer TimeSeries: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Looking_At_Longer_TimeSeries.ipynb
.. _Region Stimuli: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Region_Stimuli.ipynb
.. _Surface Stimuli: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Surface_Stimuli.ipynb
.. _Smooth Parameter Variations: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Smooth_Parameter_Variation.ipynb
.. _Stochastic Simulations: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Stochastic_Simulation.ipynb
.. _Getting To Know Your Mesh Surface: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Getting_To_Know_Your_Surface_Mesh.ipynb
.. _Using Your Own Connectivity: http://nbviewer.ipython.org/url/docs.thevirtualbrain.org/demos/Tutorial_Using_Your_Own_Connectivity.ipynb


.. figure:: /manuals/styles/TVB_logo.svg
      :width: 200px
      :figclass: demo-figure
      :target: `Anatomy Of A Region Simulation`_

      `Anatomy Of A Region Simulation`_


.. figure:: /manuals/styles/TVB_logo.svg
      :width: 200px
      :figclass: demo-figure
      :target: `Anatomy Of A Surface Simulation`_

      `Anatomy Of A Surface Simulation`_



.. figure:: /manuals/styles/TVB_logo.svg
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring A Model`_

      `Exploring A Model`_


.. figure:: /manuals/UserGuide/screenshots/demo_wong_wang.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring A Model Reduced Wong Wang`_

      `Exploring A Model Reduced Wong Wang`_


.. figure:: /manuals/UserGuide/screenshots/demo_bold.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring The Bold Monitor`_

      `Exploring The Bold Monitor`_


.. figure:: /manuals/styles/TVB_logo.svg
      :width: 200px
      :figclass: demo-figure
      :target: `Looking At Longer TimeSeries`_

      `Looking At Longer TimeSeries`_


.. figure:: /manuals/UserGuide/screenshots/demo_stimuli.png
      :width: 200px
      :figclass: demo-figure
      :target: `Region Stimuli`_

      `Region Stimuli`_


.. figure:: /manuals/UserGuide/screenshots/demo_surf_stimuli.png
      :width: 200px
      :figclass: demo-figure
      :target: `Surface Stimuli`_

      `Surface Stimuli`_


.. figure:: /manuals/UserGuide/screenshots/demo_smooth_param.png
      :width: 200px
      :figclass: demo-figure
      :target: `Smooth Parameter Variations`_

      `Smooth Parameter Variations`_


.. figure:: /manuals/UserGuide/screenshots/demo_stoch.png
      :width: 200px
      :figclass: demo-figure
      :target: `Stochastic Simulations`_

      `Stochastic Simulations`_


.. figure:: /manuals/UserGuide/screenshots/demo_mesh_stats.png
      :width: 200px
      :figclass: demo-figure
      :target: `Getting To Know Your Mesh Surface`_

      `Getting To Know Your Mesh Surface`_


.. figure:: /manuals/UserGuide/screenshots/demo_conn.png
      :width: 200px
      :figclass: demo-figure
      :target: `Using Your Own Connectivity`_

      `Using Your Own Connectivity`_
