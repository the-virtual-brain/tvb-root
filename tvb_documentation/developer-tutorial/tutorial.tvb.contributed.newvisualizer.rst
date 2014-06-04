tutorial.tvb.contributed.newvisualizer
--------------------------------------

General guidelines for implementing a new Visualizer in |TVB|'s
web browser framework.

Introduction
------------

A typical visualization of data performed in MATLAB or Python consists in
choosing a subset of the existing data and using one of the available plotting
commands to generate either an immediately viewable figure or an image file to
opened by an external viewer.  When |TVB| is being used from the command line
in a fashion similar to MATLAB or Python, this workflow is recommended. 
However, in the context of the web browser interface to |TVB|, the framework
requires a bit more scaffolding to support visualizations.

Each visualizer in the TVB framework requires a number of steps
that allow it to be integrated with the other parts of the framework:

# Add an adapter class to implement the visualizer
# Add the adapter to the list of visualizer adapters
# Create an HTML template for the visualizer
# Add any supporting Javascript or CSS files
# Update the portlets XML configuration

Each of these steps will be explained in the following sections.

TODO: expand below pending decision on depth of background information necessary

Adding a visualizer adapter
---------------------------

# Add tvb.adapers.visualizers.new_visualizer.NewVisualizer class, inheriting from ABCDisplayer
# Add "new_visualizer" to the tvb.adapters.visualizers.__init__.__all__ list

where new_visualizer is the module name containing the new visualizer adapter class NewVisualizer.

Creating a Genshi visualizer template
-------------------------------------

# Add any supporting static Javascript or CSS to tvb/interfaces/web/static/js or /style
# Add a Genshi template in tvb/interfaces/web/templates/genshi/visualizers

Updating the portlets configuration
-----------------------------------

# Update tvb/portlets/python_portlets.xml

