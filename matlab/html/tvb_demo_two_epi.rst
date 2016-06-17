.. _tvb_demo_two_epi:

.. raw:: html

   <div class="content">

.. rubric:: Two Epileptor simulation.
   :name: two-epileptor-simulation.

In this demo, we show how to perform a simulation with two Epileptors.

.. rubric:: Contents
   :name: contents

.. raw:: html

   <div>

-  `Ensure TVB is set up <#1>`__
-  `Build simulator <#2>`__
-  `Run simulation <#3>`__
-  `Convert data to MATLAB format <#4>`__
-  `Plot 2 kHz LFP & metabolic variables <#5>`__

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

    % Create epileptor model.
    model = py.tvb.simulator.models.Epileptor();

    % Difference coupling between nodes' coupling variables
    coupling = py.tvb.simulator.coupling.Difference(pyargs('a', 1e-3));

    % 2 nodes, random connection weights, zero tract lengths
    conn = py.tvb.datatypes.connectivity.Connectivity();
    conn.weights = py.numpy.random.rand(2, 2);
    conn.tract_lengths = py.numpy.zeros([2 2]);

    % Noise per state variable
    noise = py.tvb.simulator.noise.Additive(...
        pyargs('nsig', py.numpy.array([0.003 0.003 0 0.003 0.003 0])));

    % Monitor neural time series at 2 kHz
    monitor = py.tvb.simulator.monitors.TemporalAverage(...
        pyargs('period', 0.5));

    % Create simulator
    sim = py.tvb.simulator.simulator.Simulator(pyargs(...
        'integrator', py.tvb.simulator.integrators.HeunStochastic(...
            pyargs('dt', 0.1, 'noise', noise)),...
        'model', model, ...
        'coupling', coupling, ...
        'connectivity', conn, ...
        'monitors', monitor, ...
        'simulation_length', 5000));

    % Perform internal configuration
    configure(sim);

    % Spatialize epileptor excitability
    model.x0 = [-2.0, -1.6];

.. rubric:: Run simulation\ ` <>`__
   :name: run-simulation

.. code:: codeinput

    monitor_output = run(sim);

.. rubric:: Convert data to MATLAB format\ ` <>`__
   :name: convert-data-to-matlab-format

.. code:: codeinput

    time = np2m(monitor_output{1}{1});
    signal = np2m(monitor_output{1}{2});

.. rubric:: Plot 2 kHz LFP & metabolic variables\ ` <>`__
   :name: plot-2-khz-lfp-metabolic-variables

NB dimensions will be [mode, node, state var, time]

.. code:: codeinput

    figure()

    subplot 311
    plot(time, squeeze(signal(1, :, 1, :)), 'k')
    ylabel('x2(t) - x1(t)')
    set(gca, 'XTickLabel', {})

    title('Two Epileptors')

    % plot high-pass filtered LFP
    subplot 312
    [b, a] = butter(3, 2/2000*5.0, 'high');
    hpf = filter(b, a, squeeze(signal(1, :, 1, :))');
    plot(time, hpf(:, 1), 'k')
    hold on
    plot(time, hpf(:, 2), 'k')
    hold off
    set(gca, 'XTickLabel', {})
    ylabel('HPF LFP')

    subplot 313
    plot(time, squeeze(signal(1, :, 2, :)), 'k')
    ylabel('Z(t)')
    xlabel('Time (ms)')

|image0|
| 
| `Published with MATLAB®
  R2016a <http://www.mathworks.com/products/matlab/>`__

.. raw:: html

   </div>

.. |image0| image:: tvb_demo_two_epi_01.png

