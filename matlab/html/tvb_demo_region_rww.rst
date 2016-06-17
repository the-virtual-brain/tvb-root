.. _tvb_demo_region_rww:

.. raw:: html

   <div class="content">

.. rubric:: Region-level simulation with reduced Wong-Wang.
   :name: region-level-simulation-with-reduced-wong-wang.

In this demo, we show how to perform a simulation with the reduced
Wong-Wang model, using the default connectivity.

.. rubric:: Contents
   :name: contents

.. raw:: html

   <div>

-  `Ensure TVB is set up <#1>`__
-  `Build simulator <#2>`__
-  `Plot connectivity weights and tract lengths <#3>`__
-  `Run simulation <#4>`__
-  `Convert data to MATLAB format <#5>`__
-  `Plot results <#6>`__

.. raw:: html

   </div>

.. rubric:: Ensure TVB is set up\ ` <>`__
   :name: ensure-tvb-is-set-up

.. code:: codeinput

    tvb_setup

.. code:: codeoutput

    [tvb_setup] using Python 2.7 C:\Users\mw\Downloads\TVB_Distribution\tvb_data\python.exe
    TVB modules available.

.. rubric:: Build simulator\ ` <>`__
   :name: build-simulator

.. code:: codeinput

    model = py.tvb.simulator.models.ReducedWongWang();
    coupling = py.tvb.simulator.coupling.Linear;
    conn = py.tvb.datatypes.connectivity.Connectivity(...
        pyargs('load_default', py.True));
    noise = py.tvb.simulator.noise.Additive(pyargs('nsig', 1e-4));

    sim = py.tvb.simulator.simulator.Simulator(pyargs(...
        'integrator', py.tvb.simulator.integrators.HeunStochastic(...
            pyargs('dt', 0.1, 'noise', noise)),...
        'model', model, ...
        'coupling', coupling, ...
        'connectivity', conn, ...
        'simulation_length', 1000));

    configure(sim);

.. rubric:: Plot connectivity weights and tract lengths\ ` <>`__
   :name: plot-connectivity-weights-and-tract-lengths

.. code:: codeinput

    figure('Position', [500 500 1000 400])
    subplot 121, imagesc(np2m(conn.weights)), colorbar, title('Weights')
    subplot 122, imagesc(np2m(conn.tract_lengths)), colorbar
    title('Tract Lengths (mm)')

|image0|
.. rubric:: Run simulation\ ` <>`__
   :name: run-simulation

.. code:: codeinput

    data = run(sim);

.. rubric:: Convert data to MATLAB format\ ` <>`__
   :name: convert-data-to-matlab-format

.. code:: codeinput

    t = np2m(data{1}{1});
    y = np2m(data{1}{2});

.. rubric:: Plot results\ ` <>`__
   :name: plot-results

NB Dimensions will be [mode, node, state var, time]

.. code:: codeinput

    figure()
    plot(t, squeeze(y(1, :, 1, :)), 'k')
    ylabel('S(t)')
    xlabel('Time (ms)')
    title(sprintf('Reduced Wong-Wang, %d Regions', conn.weights.shape{1}*1))

|image1|
| 
| `Published with MATLAB®
  R2016a <http://www.mathworks.com/products/matlab/>`__

.. raw:: html

   </div>

.. |image0| image:: tvb_demo_region_rww_01.png
.. |image1| image:: tvb_demo_region_rww_02.png

