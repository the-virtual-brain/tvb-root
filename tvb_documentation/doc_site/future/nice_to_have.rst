.. _future_nice_to_have:


******************************************
Nice to have
******************************************

Tools/Libraries that might be useful for addressing current issues & desired functionality of TVB's scientific_library.
Blaze: http://blaze.pydata.org

Blaze is an expressive, compact set of foundational abstractions for composing computations over large amounts of semi-structured data, of arbitrary formats and distributed across arbitrary networks.
Numba: http://numba.pydata.org

Numba is an just-in-time specializing compiler which compiles annotated Python and NumPy code to LLVM (through decorators). Its goal is to seamlessly integrate with the Python scientific software stack and produce optimized native code, as well as integrate with native foreign languages.
Bokeh: https://github.com/ContinuumIO/Bokeh

Interactive visualization library for large datasets that natively uses the latest web technologies. Its goal is to provide elegant, concise construction of novel graphics in the style of Protovis/D3, while delivering high-performance interactivity over large data to thin clients.

Scientific visualisation, open source, main developers are those working on Numba and Blaze, outputs to html5 canvas, mention is made of positive properties of existing stuff, d3, node.js, etc, front end usage is simple mpl like plotting functions etc...

Benefits for TVB is regaining the simplicity we had with mplh5 integration of visualisations but, presumably, without the major performance issues as the Bokeh development is by a group focussed on efficiency in very large datasets and are building from the ground up, whereas mplh5 was/is a hack addon aimed at providing html5 access to an existing framework (mpl).
PANDAS: http://pandas.pydata.org/

Pandas is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.
SciKits-Learn: http://scikit-learn.org/stable/

Machine Learning in Python:

    Simple and efficient tools for data mining and data analysis
    Accessible to everybody, and reusable in various contexts
    Built on NumPy, SciPy, and matplotlib
    Open source - BSD license

Theano: http://deeplearning.net/software/theano/

Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently. Theano features:

    tight integration with NumPy – Use numpy.ndarray in Theano-compiled functions.
    transparent use of a GPU – Perform data-intensive calculations up to 140x faster than with CPU.(float32 only)
    efficient symbolic differentiation – Theano does your derivatives for function with one or many inputs.
    speed and stability optimizations – Get the right answer for log(1+x) even when x is really tiny.
    dynamic C code generation – Evaluate expressions faster.
    extensive unit-testing and self-verification – Detect and diagnose many types of mistake.

Pylearn2: http://deeplearning.net/software/pylearn2/

Machine learning library built on Theano.
Wakari: https://www.wakari.io/

A collaborative framework where custom Python environments can be configured and accessible through a web browser. It has ssh access, which would enable users to fork the scientific library code directly from the github repo. Had TVB project an account, scripts, ipython notebooks and even data could be shared. Programmers and scientists who wish to reproduce either simulations or analysis pipelines would just have to login to their wakari accounts and re-run the programs.

Wakari's motivation: "Data should be shareable, and analysis should be repeatable. Reproducibility should extend beyond just code to include the runtime environment, configuration, and input data."

They also provide the possibility to use Amazon S3 clusters if more computing power is required.

Benefits for TVB:

    hosting training courses
    storing complete (published / publishable) projects
    accelerate optimization of the scientific library (the latest Python libraries + computing resources are available)
    more exposition to the developers community

We could aim to make the scientific library available through conda.
Reasonwell: http://www.reasonwell.com/about

A tool for collaborative online discussion/debate. It provides multi-user argument mapping functionality.

Should the number of active developers ever grow beyond a few, it may be useful to have a structured forum like this where goals, priorities, and current state of development can be articulated and discussed, with the hope of producing a common understanding and thus direction. It may also be useful as a way of coordinating scientific discussions surrounding TVB, particularly between people of differing backgrounds (clinician, modeller).
More for the framework:
TogetherJS: https://togetherjs.com/

See https://hacks.mozilla.org/2013/10/introducing-togetherjs/

TogetherJS is a free, open source Javascript library by Mozilla that adds collaboration features and tools to your website. By adding TogetherJS to your site, your users can help each other out on a website in real time!
WebCL: http://www.khronos.org/webcl/

This is to OpenCL what webGL is to OpenGL. It is still only a draft standard, however, it's worth keeping an eye on as seems to provide a good solution for getting around the current bottleneck in webGL performance.

Demo implementation (nokia): http://webcl.nokiaresearch.com/
SciDB-Py: http://www.paradigm4.com/scidb-py/

SciDB-Py is an open-source high-performance library to use SciDB (written in C++). It runs on a cluster or on the cloud. This DB is optimized for managing massive and multi-dimensional data. It also provides support for sparse linear algebra operations.


