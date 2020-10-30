
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
      :width: 200px
      :figclass: demo-figure
      :target: ../manuals/UserGuide/UserGuide-UI_Simulator.html#phase-plane

      :ref:`Exploring a Model phase space. <phase_plane>`


.. figure:: /manuals/UserGuide/screenshots/simulator.jpg
      :width: 200px
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

      :ref:`Operations executed in the project <operations_ui>`

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



.. _scripting_demos:


Python Demos
============

These tutorials are written as IPython Notebooks and they use the scripting interface of TVB.
They can be run interactively if you have TVB’s scientific library and ipython installed.

The first set of "basic" tutorials are listed roughly in the order they should be read,
and cover the basic functionality of TVB's simulator package using very simple
examples.


.. _Analyze Region Corrcoef: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/analyze_region_corrcoef.ipynb
.. _Compare Connectivity Normalization: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/compare_connectivity_normalization.ipynb
.. _Compare Integrators: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/compare_integrators.ipynb
.. _Display Source Sensor Geometry: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/display_source_sensor_geometry.ipynb
.. _Display Surface Local Connectivity: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/display_surface_local_connectivity.ipynb
.. _Encrypt Files before upload in TVB Web GUI: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/encrypt_data.ipynb
.. _Exploring Epileptor 2D: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/exploring_epileptor2D.ipynb
.. _Exploring Longer Time Series: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/exploring_longer_time_series.ipynb
.. _Exploring Resting State in Epilepsy: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/exploring_resting_state_in_epilepsy.ipynb
.. _Exploring The Bold Monitor: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/exploring_the_bold_monitor.ipynb
.. _Exploring the Epileptor 3D: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/exploring_the_epileptor_codim_3_model.ipynb
.. _Exporing A Surface Mesh: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/exporing_a_surface_mesh.ipynb
.. _Generate a new model using DSL: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/model_generation_using_dsl.ipynb
.. _Generate TimeSeries for import in Web GUI: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/generate_ts_h5_from_library.ipynb
.. _Generate Surrogate Connectivity: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/generate_surrogate_connectivity.ipynb
.. _Interacting With The Framework: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/interacting_with_the_framework.ipynb
.. _Interacting With The Framework and Allen: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/interacting_with_Allen.ipynb
.. _Simulate for Mouse: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_for_mouse.ipynb
.. _Applying multiple Stimuli: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/multiple_stimuli.ipynb
.. _Simulate Bold Continuation: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_bold_continuation.ipynb
.. _Simulate Reduced Wong Wang: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_reduced_wong_wang.ipynb
.. _Simulate Region Bold Stimulus: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_region_bold_stimulus.ipynb
.. _Simulate Region Jansen Rit: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_region_jansen_rit.ipynb
.. _Simulate Region Stimulus: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_region_stimulus.ipynb
.. _Simulate SEEG for Epileptor with Observation Noise: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_seeg_from_epileptors_with_observation_noise.ipynb
.. _Simulate Surface Seeg Eeg Meg: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/simulate_surface_seeg_eeg_meg.ipynb
.. _Using Your Own Connectivity: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/using_your_own_connectivity.ipynb
.. _Interacting with TVBClient API by launching simulation and analyzer methods: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/interacting_with_rest_api_fire_simulation.ipynb
.. _Interacting with TVBClient API importers: https://nbviewer.codemart.ro/url/docs.thevirtualbrain.org/demos/interacting_with_rest_api_launch_operations.ipynb


.. figure:: figures/analyze_region_corrcoef.png
      :width: 200px
      :figclass: demo-figure
      :target: `Analyze Region Corrcoef`_

      `Analyze Region Corrcoef`_


.. figure:: figures/compare_connectivity_normalization.png
      :width: 200px
      :figclass: demo-figure
      :target: `Compare Connectivity Normalization`_

      `Compare Connectivity Normalization`_


.. figure:: figures/compare_integrators.png
      :width: 200px
      :figclass: demo-figure
      :target: `Compare Integrators`_

      `Compare Integrators`_


.. figure:: figures/display_source_sensor_geometry.png
      :width: 200px
      :figclass: demo-figure
      :target: `Display Source Sensor Geometry`_

      `Display Source Sensor Geometry`_


.. figure:: figures/display_surface_local_connectivity.png
      :width: 200px
      :figclass: demo-figure
      :target: `Display Surface Local Connectivity`_

      `Display Surface Local Connectivity`_


.. figure:: figures/exploring_longer_time_series.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring Longer Time Series`_

      `Exploring Longer Time Series`_


.. figure:: figures/exploring_the_bold_monitor.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring The Bold Monitor`_

      `Exploring The Bold Monitor`_


.. figure:: figures/exporing_a_surface_mesh.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exporing A Surface Mesh`_

      `Exporing A Surface Mesh`_


.. figure:: figures/generate_surrogate_connectivity.png
      :width: 200px
      :figclass: demo-figure
      :target: `Generate Surrogate Connectivity`_

      `Generate Surrogate Connectivity`_


.. figure:: figures/interact_client_api_simulation.png
      :width: 200px
      :figclass: demo-figure
      :target: `Generate a new model using DSL`_

      `Generate a new model using DSL`_


.. figure:: figures/simulate_bold_continuation.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate Bold Continuation`_

      `Simulate Bold Continuation`_


.. figure:: figures/simulate_reduced_wong_wang.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate Reduced Wong Wang`_

      `Simulate Reduced Wong Wang`_


.. figure:: figures/simulate_region_bold_stimulus.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate Region Bold Stimulus`_

      `Simulate Region Bold Stimulus`_


.. figure:: figures/simulate_region_jansen_rit.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate Region Jansen Rit`_

      `Simulate Region Jansen Rit`_


.. figure:: figures/simulate_region_stimulus.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate Region Stimulus`_

      `Simulate Region Stimulus`_


.. figure:: figures/simulate_surface_seeg_eeg_meg.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate Surface Seeg Eeg Meg`_

      `Simulate Surface Seeg Eeg Meg`_


.. figure:: figures/using_your_own_connectivity.png
      :width: 200px
      :figclass: demo-figure
      :target: `Using Your Own Connectivity`_

      `Using Your Own Connectivity`_


.. figure:: figures/stimuli.png
      :width: 200px
      :figclass: demo-figure
      :target: `Applying multiple Stimuli`_

      `Applying multiple Stimuli`_


.. figure:: figures/epileptic_signal.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring Epileptor 2D`_

      `Exploring Epileptor 2D`_


.. figure:: figures/epileptic_signal.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring Resting State in Epilepsy`_

      `Exploring Resting State in Epilepsy`_


.. figure:: figures/epileptic_signal.png
      :width: 200px
      :figclass: demo-figure
      :target: `Exploring the Epileptor 3D`_

      `Exploring the Epileptor 3D`_


.. figure:: figures/epileptic_signal.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate SEEG for Epileptor with Observation Noise`_

      `Simulate SEEG for Epileptor with Observation Noise`_



Interact with TVB Framework
---------------------------

.. figure:: figures/interacting_with_the_framework.png
      :width: 200px
      :figclass: demo-figure
      :target: `Interacting With The Framework`_

      `Interacting With The Framework`_


.. figure:: figures/interacting_with_the_framework.png
      :width: 200px
      :figclass: demo-figure
      :target: `Generate TimeSeries for import in Web GUI`_

      `Generate TimeSeries for import in Web GUI`_

.. figure:: figures/interacting_with_the_framework.png
      :width: 200px
      :figclass: demo-figure
      :target: `Encrypt Files before upload in TVB Web GUI`_

      `Encrypt Files before upload in TVB Web GUI`_

Interact with TVB REST Client API
---------------------------------

.. figure:: figures/interact_client_api_simulation.png
      :width: 200px
      :figclass: demo-figure
      :target: `Interacting with TVBClient API by launching simulation and analyzer methods`_

      `Interacting with TVBClient API by launching simulation and analyzer methods`_

.. figure:: figures/interact_client_api_operation.png
      :width: 200px
      :figclass: demo-figure
      :target: `Interacting with TVBClient API importers`_

      `Interacting with TVBClient API importers`_

Mouse
-----

.. figure:: figures/interacting_with_the_framework.png
      :width: 200px
      :figclass: demo-figure
      :target: `Interacting With The Framework and Allen`_

      `Interacting With The Framework and Allen`_

.. figure:: figures/simulate_for_mouse.png
      :width: 200px
      :figclass: demo-figure
      :target: `Simulate for Mouse`_

      `Simulate for Mouse`_



.. toctree::
      :hidden:

      Demos_Matlab



.. _matlab_demos:

MATLAB Demos
============

These are the first demos of the experimental use of TVB from the MATLAB environment, and they will
be expanded in the future.

.. figure:: ../matlab/html/tvb_demo_region_rww_01.png
      :width: 200px
      :figclass: demo-figure
      :target: Demos_Matlab.html#tvb-demo-region-rww

      :ref:`tvb_demo_region_rww`


.. figure:: ../matlab/html/tvb_demo_two_epi_01.png
      :width: 200px
      :figclass: demo-figure
      :target: Demos_Matlab.html#tvb-demo-two-epi

      :ref:`tvb_demo_two_epi`