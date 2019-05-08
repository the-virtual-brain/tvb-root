.. _tvb_demo_region_rww:

=======================
Reduced Wong-Wang model
=======================


In this demo, we show how to perform a region level simulation with the reduced
Wong-Wang model, using the default connectivity.

--------------------
Ensure TVB is set up
--------------------
::

    tvb_setup


|  [tvb_setup] using Python 2.7 C:\Users\mw\Downloads\TVB_Distribution\tvb_data\python.exe
|  TVB modules available.

---------------
Build simulator
---------------
::

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

-------------------------------------------
Plot connectivity weights and tract lengths
-------------------------------------------
::

    figure('Position', [500 500 1000 400])
    subplot 121, imagesc(np2m(conn.weights)), colorbar, title('Weights')
    subplot 122, imagesc(np2m(conn.tract_lengths)), colorbar
    title('Tract Lengths (mm)')

.. figure:: ../matlab/html/tvb_demo_region_rww_01.png
      :width: 1000px
      :figclass: demo-figure

--------------
Run simulation
--------------
::

    data = run(sim);

-----------------------------
Convert data to MATLAB format
-----------------------------
::

    t = np2m(data{1}{1});
    y = np2m(data{1}{2});

------------
Plot results
------------

NB Dimensions will be [mode, node, state var, time]::

    figure()
    plot(t, squeeze(y(1, :, 1, :)), 'k')
    ylabel('S(t)')
    xlabel('Time (ms)')
    title(sprintf('Reduced Wong-Wang, %d Regions', conn.weights.shape{1}*1))


.. figure:: ../matlab/html/tvb_demo_region_rww_02.png
      :width: 560px
      :figclass: demo-figure


