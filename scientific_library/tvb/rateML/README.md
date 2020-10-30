# CUDA model generation using LEMS format
This readme describes the usage of the code generation for models defined in LEMS based XML to Cuda (C) format.
The LEMS framework has been adopted and altered to match TVB model names. 
In LEMS2CUDA.py the function "cuda_templating('Modelname', 'path/to/your/XMLmodels')" will start the code generation.
It expects a [model+'_CUDA'].xml file to be present in ['path/to/your/XMLmodels']. 
The generated file will be placed in '~/scientific_libary/tvb/rateML/rateML_CUDA/CUDAmodels/' which is relative to your
project path.  The produced filename is a lower cased [model+'_CUDA'].c which contains a class named [model].

    .. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>
    .. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
    .. moduleauthor:: Sandra Diaz <s.diaz@fz-juelich.de>
    
# The CUDA memory model specification
![](GPUmemindex.png)

# Files in ~/rateML_CUDA/
* LEMS2CUDA.py   		    : python script for initiating model code generation
* tmpl8_CUDA.py 		    : Mako template converting XML to CUDA
* /XMLmodels                : folder containing LEMS based XML model example files
* /CUDAmodels               : resulting model folder
* /lems/                    : modified pyLEMS library tuned for CUDA model generation
* /lems/component.py        : maintains constants and exposures
* /lems/dynamics.py         : maintains all dynamic attributes
* /lems/LEMS.py             : LEMS XML file parser
* /lems/expr.py             : expression parser
* /run/                     : folder with example script to run generated model
* /run/__main__.py          : cuda setup file
* /run/cuda_run.py          : cuda run file using Pycuda
* /run/runEx                : SLURM start script
* /run/submit_sbatch.sh     : SLURM sbatch job submission script


# Requirements
Mako templating
Pycuda
tvb-library
tvb-data

# XML LEMS Definitions 
Based on http://lems.github.io/LEMS/elements.html, a number of attributes are tuned for CUDA models.
As an example an XML line and its translation to CUDA are given below.

* Constants\
If domain = 'none' no domain range will be added.\
Label is fixed to ':math:constant.name'

```xml
<Constant name="x0" domain="" default="-1.6" description="Epileptogenicity parameter."/>
```
translates to:
```c
const float x0 = -1.6;
```

* State variables
State variable ranges [lo, hi]" values are entered with keyword "boundaries" with a comma separator.\
The default option can be used to initialize if necessary (future)
For each state variable a set of boundaries can be added to encompass the dynamic range.\
A wrapping function according to the values entered will be generated and in numerical solving the wrapper 
will be applied. \

```xml
<StateVariable name="x1" default="0.0" boundaries="-2., 1."/>
```
translates to:
```c
    __device__ float wrap_it_x1(float x1)
{
    float x1dim[] = {-2.0, 1.0};
    if (x1 < x1dim[0]) x1 = x1dim[0];
    else if (x1 > x1dim[1]) x1 = x1dim[1];

    return x1;
}

    double x1 = 0.0;
    
    for(nsteps)
        for(inodes)
            x1 = state((t) % nh, i_node + 0 * n_node);
```

* Exposures\
Exposures are used for observables and translate to variables_of_interest.\
For the name enter variable to be observed (usually states).\
The field 'choices' are treated as lists with a (,) separator.\
The define is hardcoded for the the term tavg.\
Tavg is the datastruct containing the results of the parameter sweep simulation.

```xml
 <Exposure name="o_x1" choices="x1"/>
```
translates to:
```c
#define tavg(i_node) (tavg_pwi[((i_node) * size) + id])
tavg(i_node + 0 * n_node) = x1;
```

* Derived variables\
DerivedVariables can be used to 'easify' the time derivatives, enter the local coupling formulas or any formula.\
sytax: [name]=[expression].\
Define for example global and local coupling: c_0 = coupling.\
            
```xml
<DerivedVariable name="c_pop1" expression="coupling[0]"/>
```
translates to:
```c
c_pop1 *= global_coupling;
```

* Conditional Derived Variables\
ConditionalDerivedVariables are used to created if, else constructs.\
Use &lt(=); or &gt;(=) for less- or greater then (equal to).\
Syntax: if {condition} -> {cases[0]} else {cases[1]}. Cases are separated by (,).\
It will not produce an else if {cases[1]} is not present.
```xml
<ConditionalDerivedVariable name="ydot0" condition="x1 &lt; 0.0" cases="-a * x1**2 + b * x1, slope - x2 + 0.6 * (z - 4)**2 "/>
```
translates to:
```c
if (x1 < 0.0)
    ydot0 = -a * x1**2 + b * x1;
else
    ydot0 = slope - x2 + 0.6 * (z - 4)**2;
```

* Time Derivatives\
Used to define the models derivates functions solved numerically.\
Syntax: {name} = {expression}. Name field should not be equal to state variable name!
```xml
<TimeDerivative name="dV" expression="tt * (y1 - z + Iext + Kvf * c_pop1 + ydot0 * x1)"/>
<TimeDerivative name="dW" expression="..."/>
```
translates to:
```c
dV = tt * (y1 - z + Iext + Kvf * c_pop1 + ydot0 * x1);
dW = ...
```

* Coupling function\
For the coupling function a new component type can be defined.\
The dynamics can be defined with attributes similar to the solving of the numerical analysis.\

```xml
<ComponentType name="coupling_function_pop1">

<!--        Added function for pre and post synaptic activity. Fixed the power being **, however no parse checks for it-->

    <Constant name="c_a" domain="lo=0.0, hi=10., step=0.1" default="1" description="Rescales the connection strength."/>

        <!-- variable for pre synaptic function, only 1 param is allowed (and should be sufficient))  -->
        <!-- the dimension param indicates the nth statevariable to which the coupling function should apply  -->
        <!-- param name appears in the pre or post coupling function.  -->
    <Parameter name="x1_j" dimension='0'/>

    <Dynamics>
        <DerivedVariable name="pre" expression="sin(x1_j - x1)" description="pre synaptic function for coupling activity"/>
        <DerivedVariable name="post" expression="c_a" description="post synaptic = a * pre"/>
    </Dynamics>
        <!-- Handle local coupling result, full expression is c_0 *= 'value'. Name option is hardcoded -->
        <DerivedParameter name="c_pop1" expression="global_coupling * coupling" value="None"/>

    </ComponentType>
```
translates to:
```c
for (unsigned int j_node = 0; j_node < n_node; j_node++)
    {
        //***// Get the weight of the coupling between node i and node j
        float wij = weights[i_n + j_node]; // nb. not coalesced
        if (wij == 0.0)
            continue;

        //***// Get the delay between node i and node j
        unsigned int dij = lengths[i_n + j_node] * rec_speed_dt;

        //***// Get the state of node j which is delayed by dij
        float x1_j = state(((t - dij + nh) % nh), j_node + 0 * n_node);

        // Sum it all together using the coupling function. Kuramoto coupling: (postsyn * presyn) == ((a) * (sin(xj - xi))) 
        coupling += c_a * sin(x1_j - x1);

    } // j_node */

    // rec_n is used for the scaling over nodes
    c_pop1 *= global_coupling * coupling;
    c_pop2 *= g;
```


# Running an example from ~/run folder
Create a model, name it ['modelname'+'_CUDA'.xml] and execute the "cuda_templating('Modelname',
'path/to/your/XMLmodels')" function from LEMS2CUDA.py. The resulting model will be placed in the CUDA model 
folder (tvb/rateML/rateML_CUDA/CUDAmodels). This location is relative to your project path.
In the folder 'tvb/rateML/rateML_CUDA/run/' an example can be found on how to run the model generator and the CUDA model
on a GPU. From this folder, execute __main__.py on a CUDA enabled machine or execute 
'./run_example_on_cluster [Modelname]' on a SLURM cluster with GPU nodes to start model generation and 
a parameters sweep simulation with the produced model file on a GPU. The sbatch parameters in 'submit_sbatch.sh' 
should be altered to match the cluster of choice. The block dimensions default to 32x32 which has shown to have 
the best occupancy, the grid is adjusted accordingly. In the output.out a small report on the success of 
the simulation is printed. 

# TODO
Process in Readme: The parameters with name 'rec_speed_dt' is conidered to be integral part of tv coupling calculation
if not present, the delay part of coupling will be 0 for each node.
Powers in expressions should be entered between curly braces: {x^2}. The parser will pick this up and translate to
correct expression for the selected language: powf(x, 2) for CUDA and x**2 for Python models.
The 'nsig' variable is used for noise amplification. If noise component is present but this variable is not it will be
set to 1. 
Eplain that exposure attritbute in state variable will lead to :   
        
        state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"V": numpy.array([0.0000001, 1])"W": numpy.array([0.0000001, 1])},
        )
Description bug in LEMS. The description attibute is mapped to the symbol attribute in LEMS.py. For now no description 
in the model. 

Make XSD cases for coupling and noise.

Classname has to be capitalized, filename does not

c_pop1 are the standard variables for python long range coupling and local_coupling. Standard
4 c_pops are drawn from coupling array.

In exposures the name field hold the variabls of with which in the dimension field an expression can be 
formed for the monitors. 
