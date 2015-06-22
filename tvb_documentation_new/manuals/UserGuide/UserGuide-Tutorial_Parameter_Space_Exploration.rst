|example| Tutorial
------------------

**Performing parameter space exploration**

**Objective**: 
learn how to sweep and search across different simulation 
settings that run on a distributed infrastructure. 

**Summary**: 
In |TVB| platform it is possible to launch parallel simulations to explore the 
changes in dynamics of the brain as a function of the local dynamics parameters.
We illustrate this process with a case study in larg-scale brain simulations 
within |TVB|.


  - Assuming that you have already created your project:
  - Enter the `Simulator` area.
  - Select the default connectivity matrix. 
  - Choose the local dynamic model `2D Generic Oscillator`
  - Click on the 'V' button next to the model parameters to unfold the available 
    parameter range and step size. At present, a maximum of 2
    parameters can be explored at the same time.
  - For :math:`I_{ext}` set the step size to 0.4 and for :math:`a` 0.5
  - Name the new simulation and launch it. 
  - After all simulation are finished, you should be able to see the Parameter
    Space Exploration Visualizer.


.. figure:: screenshots/simulator_pse_configuration.jpg
   :width: 60%
   :align: center
  
Each point in this two dimensional graph represents two metrics: by default
Global Variance corresponds to the size of the point and Variance of the
Variance of nodes maps the color scale. 


  - From those results, critical combination of parameters can be 
    distinguished. 
 
  - If you go to `Project` area and enter the `Data Structure` page. The results 
    of all simulations will be held under one object called DataTypeGroup.

  - Export the results of all simulations. 
    
  - You can export the results by clicking on the aforementioned object and
    selecting the `Export` tab: -> TVB format.
    The data will be available for download in a .zip file. 

  - Within this folder you will find the TimeSeries for each possible parameter 
    combination. Data are stored as HDF5 files ("filename.h5") which can be used 
    to do further analysis using other software of your choice. 
