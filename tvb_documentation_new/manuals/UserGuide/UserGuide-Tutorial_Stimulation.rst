|example| Tutorial
------------------

**Simulate electrical neural stimulation**


**Objective**: 
Build and apply a stimulus on a region-based simulation.

**Summary**: 
In |TVB| platform it is possible to create neural stimulation waveforms, 
simulate brain activity while applying those stimuli to specific region nodes.
(e.g. injection of currents) 


#. Follow steps 1 and 2 from the previous example.
#. Build the stimulus (See How to create a stimulus)
#. Select a node from the 3D view on the left. Change its weight in the 
    'Current weight' cell.
#. Press the 'Update weight' button.
#. Do this for at least 5 nodes (assigning for instance 0.25, 0.15, 0.0625, 
    0.03125, 0.015625)
#. Use a Gaussian for the temporal evolution of the stimulus (mean=15000 
    and standard deviation=4)
#. Name the stimulus and press the 'Create the stimulus'
#. Go back to the Burst Area and choose this specific stimulus into the 
    simulator.
#. Follow steps 4 to 11 from the previous example.
