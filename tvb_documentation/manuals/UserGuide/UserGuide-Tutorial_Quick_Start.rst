|example| Tutorial
------------------

**Sanity check**

**Objective**: 
getting started with |TVB|. Quickly evaluate if the produced 
data is rational.

**Summary**: 
Configuring parameters for a simulation will mainly depend on the nature of 
your scientific question. Assume that you want to simulate data to
compare to some real MEG data. 


1. Choose a neural population model that shows oscillatory dynamics. 
 
2. Invest some time analysing its phase plane to set the right set of parameters.

  .. figure:: screenshots/tutorial_sanity_01.jpg
     :width: 80%
     :align: center

     The Generic 2D Oscillator model in an oscillatory regime (limit cycle).
     Dynamics are similar to the Fitzhugh-Nagumo model.


3. Test the network model for a coupling strength=0 and make sure that it does 
   what it should do, i.e, that it reflects the node local dynamics, since the 
   long-range coupling component will be absent. 

4. You might want to start by using a deterministic integration scheme such as the
   Heun method.
   
5. If you are interested in seeing the effect of time delays, test the network 
   model for a time delay = 0. Set the `Conduction Speed` to a very large value
   even if it is unrealistic. You can also launch parallel simultions using a
   range of condution speeds -- see the `Parameter Space Exploration` tutorial. 

|

.. figure:: screenshots/tutorial_sanity_02_cs_1000.jpg
   :width: 70%
   :align: center

.. figure:: screenshots/tutorial_sanity_02_cs_42.jpg
   :width: 70%
   :align: center

.. figure:: screenshots/tutorial_sanity_02_cs_3.jpg
   :width: 70%
   :align: center

   Conduction speed is 1000, 42 and 3 [mm/ms] for the upper, middle and bottom
   panel respectively.


6. For MEG signals, a good sanity test is the application of mode decompositions
   algorithms (PCA or ICA) to the simulated time series. 
 
7. Check the power spectra as well to distinguish frequency components.  
