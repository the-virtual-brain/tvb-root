.. moduleauthor:: Michiel. A. van der Vlag <m.van.der.vlag@fz-juelich.de>

=============================
TVB DSL Usage
=============================


﻿TVB_DSL XML (LEMS) code generation
----------------------------------------
This documentation describes the usage of the code generation for models defined in LEMS based XML to TVB Python format.
The LEMS format has been adopted, modified and extended to match TVB nomenclature and consider the functionality required
by these models. 
In LEMS2python.py the function ``regTVB_templating('Model')`` will start the code generation.
It expects a ``[model].xml`` file to be present in ``tvb/dsl/NeuroML/XMLmodels``.
The generated file will be placed in ``tvb/simulator/models`` and ``__init__.py`` will be updated with model references in case
of a new model being added. In case of manual model updation the origin reference will not be updated.
The produced filename is a lower cased ``[model].py`` which contains a class named ``[model]``.

*Files:*
 * dsl/LEMS2pyton.py 				: python script for initiating model code generation
 * dsl/NeuroML/XMLmodels				: directory containing LEMS based XML model files
 * dsl/tmpl8_regTVB.py				: Mako template converting XML to python
 * dsl/NeuroML/lems                  		: modified pyLEMS library tuned for TVB
 * dsl/NeuroML/lems/component.py     		: maintains constants and exposures
 * dsl/NeuroML/lems/dynamics.py      		: maintains all dynamic attributes
 * dsl/NeuroML/lems/LEMS.py    		        : LEMS XML file parser
 * dsl/NeuroML/lems/expr.py          		: expression parser
 * tvb/simulator/models           		: TVB default model directory for generated results

Prerequisites
-------------
Mako templating

XML LEMS Definitions
---------------------- 
Based on http://lems.github.io/LEMS/elements.html but attributes are tuned for TVB models.
As an example an XML line and its translation to python are given. 

* Constants:

If domain = 'none' no domain range will be added. Label is fixed to ``:math:constant.name``

.. code-block:: xml

    <Constant name="x0" domain="lo=-3.0, hi=-1.0, step=0.1" default="-1.6" description="Epileptogenicity parameter."/>

translates to:

.. code-block:: python

     x0 = NArray(
        label=":math:`Ks`",
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        default=numpy.array([-1.6]),
        doc="Epileptogenicity parameter")


* State variables

State variable ranges [lo, hi]" values are entered with keyword "default" with a comma separator.\
For each state variable a set of bondaries can be added to encompass the boundaries of the dynamic range.\
Leave empty "" for no boundaries. Set None for one-sided boundaries, ie: "1.0, None".

.. code-block:: xml

    <StateVariable name="x1" default="-2., 1." boundaries="0.0, np.inf"/>
    <StateVariable name="y1" default="-20., 2." boundaries=""/>

translates to:

.. code-block:: python

    state_variable_range = Final(
        default={
            "x1": numpy.array([-2., 1.]),
            "y1": numpy.array([-20., 2.]),
            ...
        },
        label="State variable ranges [lo, hi]",
        doc="Typical bounds on state variables in the Epileptor model.")

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "x1": np.array([0.0, np.inf])
            ...
        },
    )
    state_variables = ('x1', 'y1', ...)

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)


* Exposures:

Exposures are used for observables and translate to variables_of_interest.
For the name enter variable to be observed (usually states).
For dimension enter the reduction functionality.
The fields 'choices' and 'default' are treated as lists with a (,) separator.

.. code-block:: xml

    <Exposure name="x1" default="x2 - x1, z" choices="x1, y1, z, x2, y2, g, x2 - x1"
    description="Quantities of the Epileptor available to monitor."/>

translates to:

.. code-block:: python

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1', ),
        default=('x2 - x1', 'z', ),
        doc="Quantities of the Epileptor available to monitor."
    )


* dfun and numba function signatures:

Automatically a dfun function with its numba variant is generated which will contain the dynamic XML elements.
All the constants will be arguments for the \_numba_dfun_{modelname} function call and parameters for the function
definition.
It will generate a @guvectorize signature based on the number of constants. They will all be of the float64 datatype.
The input derivatives and coupling arrays are reshaped. They will be float64[:] datatypes. As will be the return
derivative array.
The derivative array is translated into temp variables which match the expression in the time derivatives.

Example for the ``epileptor dfun`` function signatures:

.. code-block:: python

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_EpileptorT(vw_, c_, self.a, self.b, self.c, self.d, self.r, self.s, self.x0, 
            self.Iext, self.slope, self.Iext2, self.tau, self.aa, self.bb, self.Kvf, self.Kf, self.Ks, 
            self.tt, self.modification, local_coupling)

        return deriv.T[..., numpy.newaxis]

    @guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64,
        float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64,
        float64, float64[:])],
        '(n),(m)' + ',()'*19 + '->(n)', nopython=True)
    def _numba_dfun_EpileptorT(vw, coupling, a, b, c, d, r, s, x0, Iext, slope, Iext2, tau, aa, bb, Kvf, Kf,
        Ks, tt, modification, local_coupling, dx):
        "Gufunc for {modelname} model equations."

        x1 = vw[0]
        y1 = vw[1]
        z = vw[2]
        x2 = vw[3]
        y2 = vw[4]
        g = vw[5]


* Derived variables

DerivedVariables can be used to 'easify' the time derivatives, enter the local coupling formulas or any formula.\
sytax: [name]=[expression].
Define for example global and local coupling: c_0 = coupling[0, ] and lc_0 = local_coupling.
            
.. code-block:: xml

    <DerivedVariable name="c_pop1" expression="coupling[0]"/>

translates to:

.. code-block:: python

    c_pop1 = coupling[0]


* Conditional Derived Variables

ConditionalDerivedVariables are used to created if, else constructs.
Use &lt(=); or &gt;(=) for less- or greater then (equal to).
Syntax: if {condition} -> {cases[0]} else {cases[1]}. Cases are separated by (,).
It will not produce an else if {cases[1]} is not present.

.. code-block:: xml

    <ConditionalDerivedVariable name="ydot0" condition="x1 &lt; 0.0" cases="-a * x1**2 + b * x1, slope - x2 + 0.6 * (z - 4)**2 "/>

translates to:

.. code-block:: python

    if (x1 < 0.0):
        ydot0 = -a * x1**2 + b * x1
    else:
        ydot0 = slope - x2 + 0.6 * (z - 4)**2


* Time Derivatives

Used to define the models derivates functions solved numerically.
Syntax: dx[n] = {expression}. Name field is not used.

.. code-block:: xml

    <TimeDerivative name="dx0" expression="tt * (y1 - z + Iext + Kvf * c_pop1 + ydot0 * x1)"/>
    <TimeDerivative name="dx1" expression="..."/>

translates to:

.. code-block:: python

    dx[0] = tt * (y1 - z + Iext + Kvf * c_pop1 + ydot0 * x1)
    dx[1] = ...


Running
--------
1. Place model file in directory dsl/NeuroML/XMLmodels.
2. Execute regTVB_templating('modelname') function.
3. The resulting model will be automatically placed in model directory and recognized by TVB.

Future extensions
-----------------
Automatic numba, numbaCUDA and CUDA kernel generation will be directly supported by the DSL. 
For more information on the current developments please refer to: https://github.com/the-virtual-brain/tvb-hpc
