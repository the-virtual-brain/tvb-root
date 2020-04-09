"""
Example of creating a radar chart (a.k.a. a spider or star chart) [1]_. The
source code in this module was adapted from the radar chart example in the
matplotlib gallery [2]_.

Although this example allows a frame of either 'circle' or 'polygon', polygon
frames don't have proper gridlines (the lines are circles instead of polygons).
It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in
matplotlib.axis to the desired number of vertices, but the orientation of the
polygon is not aligned with the radial axes.

.. [1] http://en.wikipedia.org/wiki/Radar_chart
.. [2] http://matplotlib.org/examples/api/radar_chart.html
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


class RadarAxes(PolarAxes):

    name = 'radar'

    def __init__(self, figure=None, rect=None, spoke_count=0,
                 radar_patch_type="polygon", radar_spine_type="circle",
                 *args, **kwargs):
        resolution = kwargs.pop("resolution", 1)
        self.spoke_count = spoke_count
        self.radar_patch_type = radar_patch_type
        # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
        self.radar_spine_type = radar_spine_type
        if figure == None:
            figure = plt.gcf()
        if rect == None:
            rect = figure.bbox_inches
                # calculate evenly-spaced axis angles
        self.radar_theta = (
            2 * np.pi *
            np.linspace(0, 1 - 1.0 / self.spoke_count, self.spoke_count))
        # rotate theta such that the first axis is at the top
        self.radar_theta += np.pi / 2
        super(RadarAxes, self).__init__(figure, rect, resolution=resolution,
                                        *args, **kwargs)

    def draw_patch(self):
        if self.radar_patch_type == "polygon":
            return self.draw_poly_patch()
        elif self.radar_patch_type == "circle":
            return draw_circle_patch()

    def draw_poly_patch(self):
        verts = unit_poly_verts(self.radar_theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def fill(self, *args, **kwargs):
        """Override fill so that line is closed by default"""
        closed = kwargs.pop('closed', True)
        return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

    def plot(self, *args, **kwargs):
        """Override plot so that line is closed by default"""
        lines = super(RadarAxes, self).plot(*args, **kwargs)
        for line in lines:
            self._close_line(line)

    def _close_line(self, line):
        x, y = line.get_data()
        # FIXME: markers at x[0], y[0] get doubled-up
        if x[0] != x[-1]:
            x = np.concatenate((x, [x[0]]))
            y = np.concatenate((y, [y[0]]))
            line.set_data(x, y)

    def set_varlabels(self, labels):
        self.set_thetagrids(self.radar_theta * 180 / np.pi, labels)

    def _gen_axes_patch(self):
        return self.draw_patch()

    def _gen_axes_spines(self):
        if self.radar_patch_type == 'circle':
            return PolarAxes._gen_axes_spines(self)
        # The following is a hack to get the spines (i.e. the axes frame)
        # to draw correctly for a polygon frame.
        spine_type = 'circle'
        verts = unit_poly_verts(self.radar_theta)
        # close off polygon by repeating first vertex
        verts.append(verts[0])
        path = Path(verts)
        spine = Spine(self, self.radar_spine_type, path)
        spine.set_transform(self.transAxes)
        return {'polar': spine}

    def _as_mpl_axes(self):
        return RadarAxes, {"spoke_count": self.spoke_count,
                           "radar_patch_type": self.radar_patch_type,
                           "radar_spine_type": self.radar_spine_type}


def draw_circle_patch(self):
    # unit circle centered on (0.5, 0.5)
    return plt.Circle((0.5, 0.5), 0.5)


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
    return verts
