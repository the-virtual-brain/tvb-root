import matplotlib
#matplotlib.use('nbagg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from lib import demoplot
figure = plt.figure(figsize=(15, 15))
gs_master = mpl.gridspec.GridSpec(4, 2, height_ratios=[1, 24, 128, 32], hspace=0, wspace=0)
# Layer 1 - Title
gs_1 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_master[0, :])
title_axes = figure.add_subplot(gs_1[0])
title_axes.set_title('Pre-Gsoc20: TVB: Reusable visualization tools for Jupyter',
                     fontsize=30, color='#cdced1')

demoplot.hide_axes(title_axes)

# Layer 2 - Price
gs_2 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_master[1, :])
price_axes = figure.add_subplot(gs_2[0])
# Layer 3 - Risks & Radar
gs_31 = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, height_ratios=[2, 1], subplot_spec=gs_master[2, :1])
risk_and_loss_axes = figure.add_subplot(gs_31[0, :])
risk_axes = figure.add_subplot(gs_31[1, :1])
loss_axes = figure.add_subplot(gs_31[1:, 1])
gs_32 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_master[2, 1])
radar_axes = figure.add_subplot(gs_32[0])
# Layer 4 - MPG
gs_4 = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_master[3, :])
mpg_axes = figure.add_subplot(gs_4[0])
# Tidy up
gs_master.tight_layout(figure)
plt.show()