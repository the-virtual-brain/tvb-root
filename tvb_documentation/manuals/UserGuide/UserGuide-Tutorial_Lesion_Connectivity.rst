|example| Tutorial
------------------

**A simple lesion study using TVB**

**Objective**: 
learn how to generate simulated brain activity to study the impact of brain 
lesions on functional patterns of activity.

**Summary**: 
In |TVB| platform it is possible to edit a connectivity matrix to mimic brain 
lesions. In the present study, a lesion in a given node area is modeled by 
removing all the incoming and ongoing connections into and from this node.

  - Assuming that you have already created your project:
  
  - Enter the `Connectivity` area and click on `Long Range Connectivity`.
  - Select the default connectivity matrix and click the `Launch` button.
    
  - On the right side the node-by-node connectivity matrix is displayed. You 
    can select the quadrant to be displayed on the screen by accessing the 
    quadrant selector.  
    
  - Unselect the nodes you want to lesion by clicking directly on the region 
    labels. This will remove all the incoming and ongoing connections into 
    and from those nodes. Here we choose to lesion the following cortical 
    areas: lCCA, lCCP, and lPFCORB. 

  - Enter a name for this new selection, save it  and click on the star icon 
    on the top-right to save the new matrix.

  - Go to `Project` area and enter the `Data Structure` page. The new 
    connectivity matrix should be available in Intermediate Data. 

  - By selecting the new Connectivity, you will access its Datatype 
    Details allowing you to get information about the data, visualize them 
    and export them.

  - From the `Visualizers` tab you can launch a display to see the new matrix.
	
  - Simulate data using both connectivity matrices. You can now use the new 
    connectivity matrix to simulate brain activity, using the TVB `Simulator`. 
    Here we choose the BOLD model with underlying FitzHugh-Nagumo equations for 
    the local dynamics. The goal is to run two long simulations, one using an 
    intact connectivity and another using the connectivity with lesions. Except 
    for the connectivity matrix, all parameters are the same (including the 
    random number generator seed). 
    
    - In the `Simulator` area - central column - you can choose the long-range 
      connectivity. Select the new connectivity matrix containing the lesions. 
    - Set the local dynamic model to Fitz-Hugh Nagumo.
    - Select BOLD in Monitors. 
    - Name the new simulation and launch it. 
    - Repeat the previous steps but choosing the default matrix (intact) as the 
      Long-range connectivity. 
	
  - Go to the `Operations` board. You can follow the state of the simulations. 
    When simulations are finished your results will be represented by its 
    datatype icons.
    
  - Export the results of both simulations and the connectivity matrices. 
  
    - You can export the results by choosing `Export`: -> TVB format. The data 
      will be stored in an HDF5 file ("filename.h5") which can be used to do 
      further analysis using other software of your choice. 



Here we choose MatLab to process the data. To read the HDF5 file in MatLab do::

  hinfo = hdf5info('filename.h5');
  % hinfo is a structure containing the data in the field 
  % hinfo.GroupHierarchy.Datasets.  

To read the data use the `hdf5read` function. For our simulations 
`hinfo.GroupHierarchy.Datasets(1)` contains the BOLD activity and 
`hinfo.GroupHierarchy.Datasets(2)` contains the time::

  Time  = hdf5read(hinfo.GroupHierarchy.Datasets(2));
  bold  = hdf5read(hinfo.GroupHierarchy.Datasets(1));
  N     = size(bold,2);
  T     = size(bold,4);
  % Where N is the number of cortical areas (=74) and T the number of time 
  % points. 


You can compact the data to a 2-dimension matrix as::

  bold_new = zeros(T,N,2);
  bold_new(:,:,1) = reshape(bold,[T N]);

You are now ready to work with `bold_new` using the tools of MatLab.


**Results**

For each condition, intact and lesion, we calculated the correlation matrix 
(functional connectivity) and evaluated whether correlations change in pairs 
of nodes. To asses significant changes we calculated the correlation matrix 
in non-overlapping time windows of 200 time points. In this way, we obtained 
a distribution for each pair-wise correlation coefficient, allowing 
statistical treatment. Correlation coefficients were fisher z-transformed and 
compared (by means of t-tests) in intact vs. lesion conditions. We found that lesions 
induced both significant increases and decreases of correlations between 
intact nodes, even for pairs of nodes in different hemispheres.

    .. figure:: screenshots/tutorial_lesion_results.jpg
	:width: 40%
	:align: center

    A) Top: Intact connectivity matrix. Middle: Connectivity matrix with 
    lesions. Bottom: Difference between intact and injured connectivity 
    matrices. Connection strengths are indicated in color code. 
    B) Top: Intact functional connectivity. Middle: Functional connectivity with 
    lesions. Pearson pair-wise correlation coefficients are indicated in 
    color code. Bottom: Significantly different pair-wise correlations in 
    intact vs. lesion conditions (squares are proportional to correlation 
    difference). Black: lesion significantly decreased correlation 
    coefficient with respect to intact correlations. Gray: lesion 
    significantly increased correlation coefficient with respect to intact 
    correlations. 
    C) Example of seed-based based correlations. Dark and 
    light colors indicate significant and non-significant differences of 
    correlation coefficients, respectively. Seed: left prefrontal polar cortex.
