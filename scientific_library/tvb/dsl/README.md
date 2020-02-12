# TVB_DSL XML (LEMS) to CUDA code generation
This readme describes the usage of the automatic code generation of the XML to CUDA. As an example 'buildCUDAkernel.py' takes the Kuramoto model (model for the behaviour of large set of coupled ocillators) defined in 'RateBased_kura.xml' and translates this in a CUDA files which can be run on a GPU. It then compares the resulting output file 'ratebased.c' file to the golden model file 'kuramoto.c' and outputs any differences between the file.

# Author
Sandra Diaz & Michiel van der Vlag @ Forschungszentrum Juelich.

# License
Free for all.

# Files
* buildCUDAkernel.py 				: main python script
* RateBased_kura.xml 				: XML LEMS format containing Kuramoto model
* kuratmoto_network_template.c 		: Template with placeholders for XML model
* kuramoto.c 						: Golden model file for comparison

# Output
* ratebased.c 						: Output to CUDA

# Prerequisites
pyLEMS: https://github.com/LEMS/pylems. 
sudo pip install pylems


# XML LEMS Definitions 
http://lems.github.io/LEMS/elements.html
This section defines the fields on the XML file and how they have been interpretated for the Kuramoto model.

* [Parameter name="string" dimension="string"] <br/>
	Sets the name and dimensinality of the parameter that must be supplied when a component is defined. These are the inputs from outside to the model.

* [Requirement name="string" dimension="string" description="string"] <br/>
	Sets the name and dimensionality that should be accessible within the scope of a model component. Used for selecting the wrapping function for the limits of the model. The desciption holds the actual value.

* [Constant name="string" dimension="string" value="string"] <br/>
	This is like a parameter but the value is supplied within the model definition itself. The dimension is used to set the unit of the constant. 

* [Exposure name="string" dimension="string"] <br/>
	For variables that should be made available to other components. Used to output the results of the model. 

Dynamics <br/>
	Specifies the dynamical behaviour of the model.

   * [StateVariable name="string" dimension="string"] <br/>
   		Name of the state variable in state elements. Dimension is used for the value

   * [DerivedVariable name="string" exposure="string" value="string" reduce="add/mul" select="noiseOn/noiseOff"] <br/>
   		A quantity that depends algebraically on other quantities in the model. The 'value' field can be set to a mathematical expression. The reduce field is used to set the mathematical expression (Add: +=, Mul: *=). The select field can be used to add noise to the integration step.

   * [TimeDerivative variable="string" value="string"] <br/>
   		Expresses the time step uped for the model. Variable is used for the name.


# Running
 The 'buildCUDAkernel.py' holds static links to the files necessary and can be run by executing this file after cloning the repo (https://gitlab.version.fz-juelich.de/diaz1/tvb_dsl).

# TODO
* Automatically load set attributes 
* Deal with multiple component types
* Generate python code for models
* Move to reduced WangWong model (create XML and generation)
* Look into pyLEMS verification functionality
* USE nestLEMS verification functionality


