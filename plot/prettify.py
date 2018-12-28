# Figure setup
from matplotlib import pyplot as plt
import matplotlib as mpl
# Fancy colors
from cycler import cycler
import palettable
import palettable.cmocean.sequential
import palettable.tableau
import palettable.colorbrewer.qualitative
import palettable.wesanderson

black = '#2d2d2d'
white = '#f5f5f5'
# Trying to make plots look OK
# Colors
mpl.rcParams['axes.prop_cycle'] = cycler('color',
                                         palettable.tableau.TableauMedium_10.mpl_colors)
# Fonts
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Helvetica']
mpl.rcParams['font.size'] = 16
# Setting normal framing colors to off-black
mpl.rcParams['text.color'] = black
mpl.rcParams['axes.labelcolor'] = black
mpl.rcParams['axes.edgecolor'] = black
mpl.rcParams['xtick.color'] = black
mpl.rcParams['ytick.color'] = black
# Background to off-white
mpl.rcParams['figure.facecolor'] = white
#mpl.rcParams['figure.facecolor'] = '#ffffff'
# Legend
mpl.rcParams['legend.framealpha'] = 0
mpl.rcParams['legend.borderpad'] = 0
mpl.rcParams['legend.markerscale'] = 1.2
mpl.rcParams['legend.handlelength'] = 0.8
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.handletextpad'] = 0.2
# Grid
#mpl.rcParams['grid.color'] = 'k'
#mpl.rcParams['grid.linestyle'] = ':'
#mpl.rcParams['grid.linewidth'] = 0.5