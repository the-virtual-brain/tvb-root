Examples
--------

We present here some basic simulation scenarios that the user should be able to
reproduce through the |TVB| interface.



.. admonition:: |example| Example 1

    **Generating resting activity data using the default simulator configuration**

    **Objective**: generate 16 seconds of resting state activity data, sampled at 
    2048Hz, by launching a region-based simulation using a stochastic 
    integration method.


  #. Use the default long-range connectivity data.
  #. Apply a linear coupling function (parameters: :math:`a`- rescale connection 
     strength=0.016, :math:`b`- shift=0).
  #. Leave the *Cortical Surface* and the *Stimulus* set to **None**.
  #. Use the default population model and its parameters.
  #. Select the Heun Stochastic integrator. Set the integration step to 
     :math:`dt=0.06103515625` and additive noise (with null correlation time 
     and standard deviation :math:`D=2^{-10}`, use the default 
     random number generator parameters).
  #. Use the default model initial conditions (basically random initial 
     conditions).
  #. Select a Temporal Average Monitor (sampling period = 0.48828125 ms)
  #. Configure the `View` tabs: select a Time Series Visualizer.
  #. Set the simulation length to 16000 ms
  #. Launch the simulation
  #. Observe the results: the default model state variable.

  |

    Run time is approximately 4 minutes using one computing core of an Intel 
    Xeon Westemere 3.2 GHz, memory requirement < 1GB, data storage requirement
    ~ 19MB.