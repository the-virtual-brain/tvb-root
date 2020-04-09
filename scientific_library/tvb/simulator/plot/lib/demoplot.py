import matplotlib as mpl
import matplotlib.pyplot as plt
from . import radar
from . import demodata


colors = mpl.rcParams['axes.color_cycle']


def hide_axes(axes):
    axes.set_frame_on(False)
    [n.set_visible(False) for n in axes.get_xticklabels() + axes.get_yticklabels()]
    [n.set_visible(False) for n in axes.get_xticklines() + axes.get_yticklines()]


def make_autos_radar_plot(
    figure, gs=None, pddata=None, title_axes=None, legend_axes=None,
    inner_axes=None, geometry=None, rotate=True):
    radar_colors = [1, 2, 0]
    min_data = pddata.groupby("make", sort=True).min()
    max_data = pddata.groupby("make", sort=True).max()
    mean_data = pddata.groupby("make", sort=True).mean()
    projection = radar.RadarAxes(spoke_count=len(mean_data.columns))
    if geometry:
        (row_num, col_num) = geometry
    else:
        (row_num, col_num) = gs.get_geometry()
    if not inner_axes:
        subplots = [x for x in gs]
        inner_axes = []
        for (i, m) in enumerate(subplots[col_num:]):
            # the left-most column is reserved for the legend
            if i % col_num != 0:
                inner_axes.append(plt.subplot(m, projection=projection))
    if not title_axes:
        title_axes = figure.add_subplot(gs[0, :])
    if legend_axes is None:
        legend_axes = figure.add_subplot(gs[0:, 0])
    if legend_axes != False:
        # setup legend axes
        max_patch = mpl.patches.Patch(color=colors[radar_colors[0]], alpha=0.7,
                                      label="Max")
        mean_patch = mpl.patches.Patch(color=colors[radar_colors[1]], alpha=0.7,
                                       label="Mean")
        min_patch = mpl.patches.Patch(color=colors[radar_colors[2]], alpha=0.7,
                                      label="Min")
        legend_axes.legend(handles=[max_patch, mean_patch, min_patch], loc=10)
        hide_axes(legend_axes)
    # setup title grid axes
    ti