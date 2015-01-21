TVB color palette texture
=========================
A 256x256 png file split in 32 horizontal stripes 8 pixel high.
Each stripe a color palette.

Palletes.xcf
============
Is the gimp source for the texture. Organized in layers.

First layer group
-----------------
Contains TVB schemes. These are taken from png exports of the legend.

Second layer group
------------------
Contains colorbrewer2.org schemes.
These have been manually created in gimp. First a checkerboard pattern was rendered.
That was bucket filled with colors copied from the site.

Third layer group
-----------------
Contains continuous versions of the color brewer schemes.
These have been manually created in gimp.
First clone the corresponding discrete layer.
Then rescale that layer *without filtering* to be as wide as the number of discrete colors
but keep the same 8px height. Then rescale it to the original size using the *cubic interpolation*.

Forth layer group
-----------------
Contains various perceptual color maps.
They have been rescaled from png files found on the authors web sites.

Fifth layer group
-----------------
Contains special degenerate color maps used by TVB.
One maps all activity to the same color thus disabling coloring for those regions/vertices.
The other contains colors for the measure points.

Others
------
A scheme edges layer is overlaid over most schemes. It implements coloring for out of range activity.
The background is a checkerboard so that accidental color scheme indexing will be visible.